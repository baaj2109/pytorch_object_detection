import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .box_utils import match 


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log( torch.sum( torch.exp(x - x_max), 1, keepdim = True)) + x_max


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    def __init__(self,
                 n_classes,
                 overlap_thresh = 0.5,
                 prior_for_matching = True,
                 bkg_label = 0,
                 neg_mining = True,
                 neg_pos = 3,
                 neg_overlap = 0.5,
                 encode_target = False):
        super(MultiBoxLoss, self).__init__()
        self.n_classes = n_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]
        
    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net. (loc, conf)
            
            conf    (tensor): with shape (batch_size, num_priors, n_classes)
            loc     (tensor): with shape (batch_size, num_priors, 4)
            priors  (tensor): with shape (num_priors, 4)
            targets (tensor): Ground truth boxes and labels, with shape 
                              (batch_size, num_objects, 5), last index store
                              [xmin, ymin, xmax, ymax, label]
        """

        loc_data, conf_data = predictions
        priors = priors
        num_batch = loc_data.size(0)
        num_priors = (priors.size(0))
        n_classes = self.n_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num_batch, num_priors, 4)
        conf_t = torch.LongTensor(num_batch, num_priors)
        for idx in range(num_batch):
            groundtrue_boxes = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold,
                  groundtrue_boxes,
                  defaults,
                  self.variance,
                  labels,
                  loc_t,
                  conf_t,
                  idx)
   
        # wrap targets
        loc_t = Variable(loc_t, requires_grad = False)
        conf_t = Variable(conf_t, requires_grad = False)

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction = "sum")

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.n_classes)

        conf_tt = conf_t.view(-1, 1)
        conf_tt_index = (conf_tt != 0).nonzero()
        conf_tt[conf_tt_index] = 1
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_tt)
        # loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # loss_c = log_sum_exp(batch_conf) - batch_conf.gather(0, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view( pos.size()[0], pos.size()[1])
        loss_c[pos] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num_batch, -1)
        _,loss_idx = loss_c.sort(1, descending = True)
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim = True)
        num_neg = torch.clamp( self.negpos_ratio * num_pos, max = pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.n_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction = "sum")

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum().float()
        loss_l = loss_l / N
        loss_c /= N
        return loss_l,loss_c
    

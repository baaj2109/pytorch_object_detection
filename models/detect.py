

from torch.autograd import Function



class Detect(Function):
    def __init__(self, n_classes, bkg_label, cfg, object_score=0):
        self.n_classes = n_classes
        self.background_label = bkg_label
        self.object_score = object_score
        # self.thresh = thresh

        # Parameters used in nms.
        self.variance = cfg['variance']

    def forward(self, predictions, prior, arm_data=None):
        loc, conf = predictions
        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        print("loc_data size: ",num)
        if arm_data:
            arm_loc, arm_conf = arm_data
            arm_loc_data = arm_loc.data
            arm_conf_data = arm_conf.data
            arm_object_conf = arm_conf_data[:, 1:]
            no_object_index = arm_object_conf <= self.object_score
            conf_data[no_object_index.expand_as(conf_data)] = 0

        self.num_priors = prior_data.size(0)
        self.boxes = torch.zeros(num, self.num_priors, 4)
        self.scores = torch.zeros(num, self.num_priors, self.n_classes)

        if num == 1:
            # size batch x n_classes x num_priors
            conf_preds = conf_data.unsqueeze(0)

        else:
            conf_preds = conf_data.view(num, self.num_priors,
                                        self.n_classes)
            self.boxes.expand(num, self.num_priors, 4)
            self.scores.expand(num, self.num_priors, self.n_classes)
        # Decode predictions into bboxes.
        for i in range(num):
            if arm_data:
                default = decode(arm_loc_data[i], prior_data, self.variance)
                default = center_size(default)
            else:
                default = prior_data
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            self.boxes[i] = decoded_boxes
            self.scores[i] = conf_scores

        return self.boxes, self.scores

if __name__ == '__main__':
    MOBILEV3_300 = {
        "feature_maps": [19, 10, 5, 3, 2, 1],
        "min_dim": 300,
        "steps": [16, 32, 64, 100, 150, 300],
        "min_sizes": [60, 105, 150, 195, 240, 285],
        "max_sizes": [105, 150, 195, 240, 285, 330],
        "aspect_ratios": [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        "variance": [0.1, 0.2],
        "clip": True,
    }

    detector = Detect(199, 0, MOBILEV3_300)




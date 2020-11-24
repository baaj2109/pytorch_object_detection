import torch
import torch.nn as nn
from torch.nn import init
from .mobilenetv3 import MobileNetv3
from itertools import product as product
from math import sqrt



def conv_bn(inp, oup, stride, groups=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


def conv_1x1_bn(inp, oup, groups=1, activation=nn.ReLU6):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


class AddExtras(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AddExtras, self).__init__()

        self.extra_convs = []
    
        self.extra_convs.append(conv_1x1_bn(960, 256))
        self.extra_convs.append(conv_bn(256, 256, 2, groups = 256))
        self.extra_convs.append(conv_1x1_bn(256, 512, groups = 1))
    
        self.extra_convs.append(conv_1x1_bn(512, 128))
        self.extra_convs.append(conv_bn(128, 128, 2, groups = 128))
        self.extra_convs.append(conv_1x1_bn(128, 256))
    
        self.extra_convs.append(conv_1x1_bn(256, 128))
        self.extra_convs.append(conv_bn(128, 128, 2, groups = 128))
        self.extra_convs.append(conv_1x1_bn(128, 256))
    
        self.extra_convs.append(conv_1x1_bn(256, 64))
        self.extra_convs.append(conv_bn(64, 64, 2, groups = 64))
        self.extra_convs.append(conv_1x1_bn(64, 128))
        self.extra_convs = nn.Sequential(*self.extra_convs)
        
        self.init_conv2d()

    def init_conv2d(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std = 0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, conv7_feats):
        """
        Forward propagation.
        :param conv7_feats: lower-level conv7 feature map
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
 
        outs = []
        out = conv7_feats
        for i, conv in enumerate(self.extra_convs):
            
            out = conv(out)
            if i % 3 == 2:
                outs.append(out)
                
        conv8_2_feats = outs[0]
        conv9_2_feats = outs[1]
        conv10_2_feats = outs[2]
        conv11_2_feats = outs[3]
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class MultiBox(nn.Module):
    def __init__(self, n_classes):
        super(MultiBox, self).__init__()

        self.n_classes = n_classes
        n_boxes = {'conv4_3': 4, 'conv7': 6, 'conv8_2': 6,
                   'conv9_2': 6, 'conv10_2': 6, 'conv11_2': 6}
        
        input_channels=[672, 960, 512, 256, 256, 128]
        self.loc_conv4_3 = nn.Conv2d(input_channels[0], n_boxes['conv4_3'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv7 = nn.Conv2d(input_channels[1], n_boxes['conv7'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv8_2 = nn.Conv2d(input_channels[2], n_boxes['conv8_2'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv9_2 = nn.Conv2d(input_channels[3], n_boxes['conv9_2'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv10_2 = nn.Conv2d(input_channels[4], n_boxes['conv10_2'] * 4, kernel_size = 3, padding = 1)
        self.loc_conv11_2 = nn.Conv2d(input_channels[5], n_boxes['conv11_2'] * 4, kernel_size = 3, padding = 1)
        
        self.cl_conv4_3 = nn.Conv2d(input_channels[0], n_boxes['conv4_3'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv7 = nn.Conv2d(input_channels[1], n_boxes['conv7'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv8_2 = nn.Conv2d(input_channels[2], n_boxes['conv8_2'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv9_2 = nn.Conv2d(input_channels[3], n_boxes['conv9_2'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv10_2 = nn.Conv2d(input_channels[4], n_boxes['conv10_2'] * n_classes, kernel_size = 3, padding = 1)
        self.cl_conv11_2 = nn.Conv2d(input_channels[5], n_boxes['conv11_2'] * n_classes, kernel_size = 3, padding = 1)
        
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)
        
        ## bounding box MultiBox layer
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  

        l_conv7 = self.loc_conv7(conv7_feats)  
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  
        l_conv7 = l_conv7.view(batch_size, -1, 4)  

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)          

        ##  confidence MultiBox layer
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  
        c_conv4_3 = c_conv4_3.view(batch_size, -1,self.n_classes)  

        c_conv7 = self.cl_conv7(conv7_feats)  
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  
        c_conv7 = c_conv7.view(batch_size, -1,self.n_classes)  

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  
        
        loc_layers = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim = 1)  
        conf_layers = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim = 1)  
        return loc_layers, conf_layers



class SSDMobilenetv3(nn.Module):

    def __init__(self, base, n_classes):
        super(SSDMobilenetv3, self).__init__()

        self.base = base
        self.n_classes = n_classes

        self.aux_convs = AddExtras()
        self.pred_convs = MultiBox(n_classes)

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 672, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, x):      
        conv4_3_feats, conv7_feats = self.base(x)            
        norm = conv4_3_feats.pow(2).sum(dim = 1, keepdim = True).sqrt() + 1e-10  
        conv4_3_feats = conv4_3_feats / norm  
        conv4_3_feats = conv4_3_feats * self.rescale_factors  
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)  
        
        
        loc_layers, conf_layers = self.pred_convs(conv4_3_feats,
                                                  conv7_feats, 
                                                  conv8_2_feats, 
                                                  conv9_2_feats, 
                                                  conv10_2_feats,
                                                  conv11_2_feats)  
        return loc_layers, conf_layers


    def create_prior_boxes(self):
        feature_maps = {'conv4_3': 19, "conv7": 10, "conv8_2": 5,
                        'conv9_2': 3, 'conv10_2': 2, 'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1, "conv7": 0.2, "conv8_2": 0.375,
                        'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., .5], 
                         'conv7': [1., 2., 3., .5, .333], 
                         'conv8_2': [1., 2., 3., .5, .333],
                         'conv9_2': [1., 2., 3., .5, .333],
                         'conv10_2': [1., 2., 3., .5, .333], 
                         'conv11_2': [1., 2., 3., .5, .333]}

        fmaps = list(feature_maps.keys())
        prior_boxes = []
        for k, f in enumerate(fmaps):
            for i, j in product(range(feature_maps[f]), repeat = 2):
                cx = (j + 0.5) / feature_maps[f]
                cy = (i + 0.5) / feature_maps[f]
                for ratio in aspect_ratios[f]:
                    prior_boxes.append([cx, cy, obj_scales[f] * sqrt(ratio), obj_scales[f] * sqrt(ratio)])

                    if ratio == 1.:
                        try:
                            additional_scale = sqrt(obj_scales[f] * obj_scales[fmaps[k + 1]])
                        except IndexError:
                            additional_scale = 1.
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])
        prior_boxes = torch.FloatTensor(prior_boxes)
        prior_boxes.clamp_(0, 1)
        return prior_boxes






import torch
import torch.nn as nn
from .mobilenetv2 import mobilenetv2, InvertedResidual
from .ssd import SSD, GraphPath


def SeperableConv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                  out_channels = in_channels,
                  kernel_size = kernel_size,
                  groups = in_channels,
                  stride = stride,
                  padding = padding),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace = False),
        nn.Conv2d(in_channels = in_channels, 
                  out_channels = out_channels,
                  kernel_size = 1),
    )

class liteConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(liteConv, self).__init__()

        hidden_dim = out_channels // 2        
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace = False),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace = False),
            # pw-linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = False)
        )
    
    def forward(self, x):
        return self.conv(x)
        


def add_extras():
    extras_layers = nn.ModuleList([
        liteConv(1280, 512, stride = 2),
        liteConv(512,  256, stride = 2),
        liteConv(256,  256, stride = 2),
        liteConv(256,   64, stride = 2)
    ])
    return extras_layers


def multibox(n_classes, width_mult = 1.0,):
    '''each output featureMap produce 6 result, >> mbox = 6 in each layer
    '''
    loc_layers = nn.ModuleList([
        SeperableConv2d(in_channels = round(576 * width_mult),
                        out_channels = 6 * 4,
                        kernel_size = 3, 
                        padding = 1),
        SeperableConv2d(in_channels = 1280, out_channels = 6 * 4, kernel_size = 3, padding = 1),
        SeperableConv2d(in_channels = 512,  out_channels = 6 * 4, kernel_size = 3, padding = 1),
        SeperableConv2d(in_channels = 256,  out_channels = 6 * 4, kernel_size = 3, padding = 1),
        SeperableConv2d(in_channels = 256,  out_channels = 6 * 4, kernel_size = 3, padding = 1),
        nn.Conv2d(in_channels = 64, out_channels = 6 * 4, kernel_size = 1),
    ])
    
    
    conf_layers = nn.ModuleList([
        SeperableConv2d(in_channels = round(576 * width_mult),
                        out_channels = 6 * n_classes, 
                        kernel_size = 3, 
                        padding = 1),
        SeperableConv2d(in_channels = 1280, out_channels = 6 * n_classes, kernel_size = 3, padding = 1),
        SeperableConv2d(in_channels = 512,  out_channels = 6 * n_classes, kernel_size = 3, padding = 1),
        SeperableConv2d(in_channels = 256,  out_channels = 6 * n_classes, kernel_size = 3, padding = 1),
        SeperableConv2d(in_channels = 256,  out_channels = 6 * n_classes, kernel_size = 3, padding = 1),
        nn.Conv2d(in_channels = 64, out_channels = 6 * n_classes, kernel_size = 1),
    ])  
    return loc_layers, conf_layers




def create_mobilenetv2_ssd_lite(base, n_classes, width_mult = 1.0,  use_batch_norm = True, ):

    base = base.extract_feature

    extras_layer = add_extras()
    loc_layers, conf_layers = multibox(n_classes = n_classes, width_mult = width_mult)

    source_layer_indexes = [
        GraphPath(14, 'conv', 3),
        19,
    ] 

    return SSD(base,
               extras_layer,
               loc_layers,
               conf_layers,
               source_layer_indexes,
               n_classes)












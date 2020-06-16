import torch
import torch.nn as nn
from .mobilenetv2 import InvertedResidual
from .mobilenetv3_v2 import mobilenetv3
from .ssdv3 import SSDV3
from .ssd import GraphPath
from torchsummary import summary



def SeperableConv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                  out_channels = in_channels,
                  kernel_size = kernel_size,
                  groups = in_channels,
                  stride = stride,
                  padding = padding,
                  bias = False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace = False),
        nn.Conv2d(in_channels = in_channels, 
                  out_channels = out_channels,
                  kernel_size = 1,
                  bias = True),
    )


def add_extras():
    extras = nn.ModuleList([
        InvertedResidual(1280, 512, stride = 2, expand_ratio = 0.2),
        InvertedResidual(512,  256, stride = 2, expand_ratio = 0.25),
        InvertedResidual(256,  256, stride = 2, expand_ratio = 0.5),
        InvertedResidual(256,   64, stride = 2, expand_ratio = 0.25)
    ])
    return extras


def multibox(n_classes, model_mode, width_mult = 1.0):

    loc_layers = nn.ModuleList([
        SeperableConv2d(in_channels = round(288 * width_mult) if model_mode == "SMALL" 
                                                              else round(672 * width_mult),
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
        SeperableConv2d(in_channels = round(288 * width_mult) if model_mode == "SMALL" 
                                                              else round(672 * width_mult),
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


def create_mobilenetv3_ssd_lite(base, n_classes, model_mode = "SMALL", width_mult = 1.0,  use_batch_norm = True, ):

    base = base.features

    extras_layer = add_extras()
    loc_layers, conf_layers = multibox(n_classes = n_classes, model_mode = model_mode, width_mult = width_mult)

    source_layer_indexes = [GraphPath(11, 'conv', -1),
                            20] if model_mode == "SMALL" else [GraphPath(16, 'conv', -1), 22]

    return SSDV3(base,
                 extras_layer,
                 loc_layers,
                 conf_layers,
                 source_layer_indexes,
                 n_classes)



if __name__ == '__main__':
    
    n_classes = 199
    mode = "SMALL"
    model = mobilenetv3(model_mode = mode, n_classes = n_classes, width_mult = 1.0, dropout_ratio = 0.0)
    # model = mobilenetv3(model_mode = "LARGE", num_classes = n_classes, multiplier = 1.0, dropout_rate = 0.0)
    # summary(model, (3, 300, 300))
    ssd = create_mobilenetv3_ssd_lite(model, n_classes, model_mode = mode)
    ssd = ssd.eval()

    x = torch.rand(1,3,300,300)
    output = ssd(x)
    print(f"loc shape :{output[0].shape}")
    print(f"conf shape : {output[1].shape}")





















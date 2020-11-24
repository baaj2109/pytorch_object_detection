import torch
import torch.nn as nn
from torchsummary import summary



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def ConvbBNReLU6(in_channels, out_channels, stride, use_batch_norm = True):
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels= in_channels,
                      out_channels= out_channels,
                      kernel_size= 3,
                      stride= stride,
                      padding= 1,
                      bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace = False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels= in_channels, 
                      out_channels= out_channels,
                      kernel_size= 3,
                      stride= stride,
                      padding= 1,
                      bias= False),
            nn.ReLU6(inplace= False)
        )


def conv_1x1_bn(in_channels, out_channels, use_batch_norm = True):
    if use_batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace = False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      bias = False),
            nn.ReLU6(inplace = False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_batch_norm = True):
        super(InvertedResidual, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace = False),
                    # pw-linear
                    nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias = False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
                    nn.ReLU6(inplace = False),
                    # pw-linear
                    nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias = False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias = False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace = False),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace = False),
                    # pw-linear
                    nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias = False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias = False),
                    nn.ReLU6(inplace = False),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups = hidden_dim, bias = False),
                    nn.ReLU6(inplace = False),
                    # pw-linear
                    nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias = False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetv2(nn.Module):
    def __init__(self, 
                 n_classes = 80,
                 width_mult = 1.,
                 round_nearest = 8, 
                 dropout_ratio = 0.2,
                 use_batch_norm = True,):
        super(MobileNetv2, self).__init__()


        """MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
                                 Set to 1 to turn off rounding
            dropout_ratio (float): percetage of neural drop out at classifier
        """
        
        image_channels = 3
        
        last_channel = 1280
        input_channel = 32
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        
        inverted_residual_setting = [
            #expand_ratio, channel, number of residual bloc, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
        
        self.extract_feature = []
        self.extract_feature.append( ConvbBNReLU6(image_channels, input_channel, stride = 2))
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.extract_feature.append( InvertedResidual(in_channels = input_channel,
                                                              out_channels = output_channel,
                                                              stride = stride, 
                                                              expand_ratio = t,
                                                              use_batch_norm = use_batch_norm))
                input_channel = output_channel
        self.extract_feature.append(conv_1x1_bn(in_channels = input_channel, 
                                                out_channels = self.last_channel,
                                                use_batch_norm = use_batch_norm))
        # make it nn.Sequential
        self.extract_feature = nn.Sequential(*self.extract_feature)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Linear(self.last_channel, n_classes),
        )
    
    
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)    
                
    def forward(self, x):
        x = self.extract_feature(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x       
        


if __name__ == '__main__':

    model = MobileNetv2()
    summary(model, (3,512,512))







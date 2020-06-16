import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1

        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class h_sigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3., inplace = False) / 6.


class h_swish(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3., inplace = False) / 6.
        return out * x


def _make_divisible(v, divisor = 8, min_value = None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, se_wh, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace = False),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )
        self.kernel_size = [se_wh, se_wh]
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=self.kernel_size)
        

    def forward(self, x):
        batch, channels, height, width = x.size()

        out = F.avg_pool2d(x, kernel_size = self.kernel_size).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class MobileBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernal_size, 
                 stride, 
                 nonLinear, 
                 SE, 
                 exp_size, 
                 se_wh):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        self.exp_size = exp_size
        self.se_wh = se_wh
        # self.dropout_ratio = dropout_ratio
        padding = (kernal_size - 1) // 2

        self.use_connect = stride == 1 and in_channels == out_channels

        activation = nn.ReLU if self.nonLinear == "RE" else h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 
                      exp_size,
                      kernel_size = 1, 
                      stride = 1, 
                      padding = 0, 
                      bias = False),
            nn.BatchNorm2d(exp_size),
            activation()
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size,
                      exp_size, 
                      kernel_size = kernal_size, 
                      stride = stride, 
                      padding = padding, 
                      groups = exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size, se_wh)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size,
                      out_channels, 
                      kernel_size = 1, 
                      stride = 1, 
                      padding = 0),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)

        # Squeeze and Excite
        if self.SE:
            out = self.squeeze_block(out)

        # point-wise conv
        out = self.point_conv(out)

        # connection
        if self.use_connect:
            return x + out
        else:
            return out


class mobilenetv3(nn.Module):
    def __init__(self,
                 model_mode = "LARGE", 
                 n_classes = 30, 
                 width_mult = 1.0, 
                 dropout_ratio = 0.):
        super(mobilenetv3, self).__init__()
        self.n_classes = n_classes

        self.features = []

        if model_mode == "LARGE":
            layers = [
                [16, 16, 3, 1, "RE", False, 16, 0],
                [16, 24, 3, 2, "RE", False, 64, 0],
                [24, 24, 3, 1, "RE", False, 72, 0],
                [24, 40, 5, 2, "RE", True, 72, 38],
                [40, 40, 5, 1, "RE", True, 120, 38],

                [40, 40, 5, 1, "RE", True, 120, 38],
                [40, 80, 3, 2, "HS", False, 240, 0],
                [80, 80, 3, 1, "HS", False, 200, 0],
                [80, 80, 3, 1, "HS", False, 184, 0],
                [80, 80, 3, 1, "HS", False, 184, 0],

                [80, 112, 3, 1, "HS", True, 480, 19],
                [112, 112, 3, 1, "HS", True, 672, 19],
                [112, 160, 5, 1, "HS", True, 672, 19],
                [160, 160, 5, 2, "HS", True, 672, 10],
                [160, 160, 5, 1, "HS", True, 960, 10],
            ]
            init_conv_out = _make_divisible(16 * width_mult)
            # print(f'init_conv_out : {init_conv_out}')

            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels = 3, 
                          out_channels = init_conv_out, 
                          kernel_size = 3, 
                          stride = 2, 
                          padding = 1),
                nn.BatchNorm2d(init_conv_out),
                h_swish(),
            )

            self.features.append(nn.Conv2d(in_channels = 3, 
                                           out_channels = init_conv_out, 
                                           kernel_size = 3, 
                                           stride = 2, 
                                           padding = 1))
            self.features.append(nn.BatchNorm2d(init_conv_out))
            self.features.append(h_swish())

            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size, se_wh in layers:
                in_channels = _make_divisible(in_channels * width_mult)
                out_channels = _make_divisible(out_channels * width_mult)
                exp_size = _make_divisible(exp_size * width_mult)
                self.block.append(MobileBlock(in_channels, 
                                              out_channels, 
                                              kernal_size,
                                              stride, 
                                              nonlinear, 
                                              se, 
                                              exp_size, 
                                              se_wh))

                self.features.append(MobileBlock(in_channels, 
                                                 out_channels, 
                                                 kernal_size, 
                                                 stride, 
                                                 nonlinear, 
                                                 se, 
                                                 exp_size, 
                                                 se_wh))

            self.block = nn.Sequential(*self.block)

            out_conv1_in = _make_divisible(160 * width_mult)
            out_conv1_out = _make_divisible(960 * width_mult)
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(out_conv1_in, 
                          out_conv1_out, 
                          kernel_size = 1, 
                          stride = 1),
                nn.BatchNorm2d(out_conv1_out),
                h_swish(),
            )

            self.features.append(nn.Conv2d(out_conv1_in, 
                                           out_conv1_out, 
                                           kernel_size = 1, 
                                           stride = 1))
            self.features.append(nn.BatchNorm2d(out_conv1_out))
            self.features.append(h_swish())

            out_conv2_in = _make_divisible(960 * width_mult)
            out_conv2_out = _make_divisible(1280 * width_mult)
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(out_conv2_in, 
                          out_conv2_out, 
                          kernel_size = 1, 
                          stride = 1),
                h_swish(),
                nn.Dropout(dropout_ratio),
                nn.Conv2d(out_conv2_out, 
                          self.n_classes, 
                          kernel_size = 1, 
                          stride = 1),
            )

            self.features.append(nn.Conv2d(out_conv2_in, 
                                           out_conv2_out, 
                                           kernel_size = 1, 
                                           stride = 1))
            self.features.append(h_swish())

            self.features = nn.Sequential(*self.features)

        elif model_mode == "SMALL":
            layers = [
                [16, 16, 3, 2, "RE", True, 16, 75],
                [16, 24, 3, 2, "RE", False, 72, 0],
                [24, 24, 3, 1, "RE", False, 88, 0],
                [24, 40, 5, 2, "RE", True, 96, 19],
                [40, 40, 5, 1, "RE", True, 240, 19],
                [40, 40, 5, 1, "RE", True, 240, 19],
                [40, 48, 5, 1, "HS", True, 120, 19],
                [48, 48, 5, 1, "HS", True, 144, 19],
                [48, 96, 5, 2, "HS", True, 288, 10],
                [96, 96, 5, 1, "HS", True, 576, 10],
                [96, 96, 5, 1, "HS", True, 576, 10],
            ]

            self.features = []

            init_conv_out = _make_divisible(16 * width_mult)
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels = 3, 
                          out_channels = init_conv_out, 
                          kernel_size = 3, 
                          stride = 2, 
                          padding = 1),
                nn.BatchNorm2d(init_conv_out),
                h_swish(),
            )

            self.features.append(nn.Conv2d(in_channels = 3, 
                                           out_channels = init_conv_out, 
                                           kernel_size = 3, 
                                           stride = 2, 
                                           padding = 1))
            self.features.append(nn.BatchNorm2d(init_conv_out))
            self.features.append(h_swish())


            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size, se_wh in layers:                
                in_channels = _make_divisible(in_channels * width_mult)
                out_channels = _make_divisible(out_channels * width_mult)
                exp_size = _make_divisible(exp_size * width_mult)

                self.block.append(MobileBlock(in_channels, 
                                              out_channels, 
                                              kernal_size, 
                                              stride, 
                                              nonlinear, 
                                              se, 
                                              exp_size, 
                                              se_wh))

                self.features.append(MobileBlock(in_channels, 
                                                 out_channels, 
                                                 kernal_size, 
                                                 stride, 
                                                 nonlinear, 
                                                 se, 
                                                 exp_size, 
                                                 se_wh))

            self.block = nn.Sequential(*self.block)

            out_conv1_in = _make_divisible(96 * width_mult)
            out_conv1_out = _make_divisible(576 * width_mult)
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(out_conv1_in, 
                          out_conv1_out, 
                          kernel_size = 1, 
                          stride = 1),
                SqueezeBlock(out_conv1_out, se_wh = 10),
                nn.BatchNorm2d(out_conv1_out),
                h_swish(),
            )
            self.features.append(nn.Conv2d(out_conv1_in, 
                                           out_conv1_out, 
                                           kernel_size = 1, 
                                           stride = 1))
            self.features.append(SqueezeBlock(out_conv1_out, se_wh = 10))
            self.features.append(nn.BatchNorm2d(out_conv1_out))
            self.features.append(h_swish())

            out_conv2_in = _make_divisible(576 * width_mult)
            out_conv2_out = _make_divisible(1280 * width_mult)
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(out_conv2_in, 
                          out_conv2_out, 
                          kernel_size = 1, 
                          stride = 1),
                h_swish(),
                nn.Dropout(dropout_ratio),
                nn.Conv2d(out_conv2_out, 
                          self.n_classes, 
                          kernel_size = 1, 
                          stride = 1),
            )
            self.features.append(nn.Conv2d(out_conv2_in, 
                                           out_conv2_out, 
                                           kernel_size = 1, 
                                           stride = 1))
            self.features.append(h_swish())    

            self.features = nn.Sequential(*self.features)

        self.apply(_weights_init)


    def forward(self, x):
        out = self.init_conv(x)
        out = self.block(out)
        out = self.out_conv1(out)
        batch, channels, height, width = out.size()
        out = F.avg_pool2d(out, kernel_size=[10, 10])
        out = self.out_conv2(out).view(batch, -1)
        return out


if __name__ == '__main__':
    
    model = mobilenetv3(model_mode = "LARGE", n_classes = 199, width_mult = 1.0, dropout_ratio = 0.0)
    summary(model, (3, 300, 300))





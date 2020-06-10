import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchsummary import summary



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + float(3.0), inplace=True) / float(6.0)
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + float(3.0), inplace=True) / float(6.0)
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )
    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule
        self.output_status = False
        if kernel_size == 5 and in_size == 160 and expand_size == 672:
            self.output_status = True
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size = 1, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(out_size),
            )
    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        if self.output_status:
            expand = out
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
    
        if self.output_status:
            return (expand, out)
        return out


class mobilenetv3(nn.Module):
    def __init__(self, n_classes = 1000):

        super(mobilenetv3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 1, bias = False)
        
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )
        self.conv2 = nn.Conv2d(160, 960, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, 1000)
        self.init_weights()

    def init_weights(self, pretrained = None):#"./mbv3_large.old.pth.tar"
        if isinstance(pretrained, str):
            logger = logging.getLogger()

            checkpoint = torch.load(pretrained,map_location = 'cpu') ["state_dict"]      
            self.load_state_dict(checkpoint,strict = False)
            for param in self.parameters():
                param.requires_grad = True # to be or not to be
            print("\nLoaded base model.\n")

        elif pretrained is None:
            print("\nNo loaded base model.\n")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode = 'fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std = 0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                        
    def forward(self, x):
        outs = []
        out = self.hs1(self.bn1(self.conv1(x)))
        
        for i, block in enumerate(self.bneck):
            out = block(out)
            if isinstance(out, tuple):
                outs.append(out[0])
                
                conv4_3_feats = out[0]                
                out = out[1]
        out = self.hs2(self.bn2(self.conv2(out)))
        
        conv7_feats = out     
        return conv4_3_feats,conv7_feats



if __name__ == '__main__':
    
    n_classes = 200
    mobilenetv3_model = mobilenetv3(n_classes)
    summary(mobilenetv3_model,(3,300,300))











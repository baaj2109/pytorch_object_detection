import torch
import torch.nn as nn
import numpy as numpy
from ssd import GraphPath


def initWithXavier(m: nn.Module):
    if isinstance(m, nn.Module):
        nn.init.xavier_uniform_(m.weight)


class SSDV3(nn.Module):
    '''single shot multibox architecture
       backbone network is mobilenetv3
    '''
    def __init__(self,
                 base,
                 extras_layers,
                 loc_layers,
                 conf_layers,
                 source_layer_indexes,
                 n_classes,
                 dropout_ratio = 0.1,
                 is_train = True):
        super(SSDV3, self).__init__()       
        '''
        Args:
            base (nn.Module): backbone network - MobileNetV3 extract feature part
            extras_layers (nn.ModuleList): extra layers that feed to multibox loc and conf layers
            loc_layers (nn.ModuleList): bounding box output layer
            conf_layers (nn.ModuleList): class confidence output layer
            n_classes (int): number of class need to detect
            dropout_ratio (float): percentage of dropout ratio
        '''  
        self.n_classes = n_classes
        self.base = base
        self.source_layer_indexes = source_layer_indexes
        self.extras_layers = extras_layers
        self.dropout = nn.Dropout(p = dropout_ratio, inplace = False)
        self.loc_layers = loc_layers
        self.conf_layers = conf_layers
        self.is_train = is_train
        # self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes 
                                                    if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        # self.init()

    def init(self):
        self.base.apply(initWithXavier)
        self.source_layer_add_ons.apply(initWithXavier)
        self.loc_layers.apply(initWithXavier)
        self.conf_layers.apply(initWithXavier)
        self.extras_layers.apply(initWithXavier)

    def get_multibox_layer_output(self, i, x):

        conf = self.conf_layers[i](x)
        conf = conf.permute(0, 2, 3, 1).contiguous()
        conf = conf.view(conf.size(0), -1, self.n_classes)

        loc = self.loc_layers[i](x)
        loc = loc.permute(0, 2, 3, 1).contiguous()
        loc = loc.view(loc.size(0), -1, 4)
        return conf, loc


    def forward(self, x):
        loc = []
        conf = []
        source = []
        start_layer_index = 0
        header_index = 0

        x = x * 2 / 255.0
        x = x - 1

        # extract featuremap from layer which inside base network, according source layer index
        # save layer output to source list
        for end_layer_index in self.source_layer_indexes:
            
            if isinstance(end_layer_index, GraphPath):
                extras_from_base = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None

            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                extras_from_base = None

            else:
                added_layer = None
                extras_from_base = None
            
            for layer in self.base[start_layer_index: end_layer_index]:
                x = layer(x)
                
            if added_layer:
                y = added_layer(x)

            else:
                y = x
                
            if extras_from_base:
                print("11",self.base[end_layer_index])
                print("22",extras_from_base.name)
                sub = getattr(self.base[end_layer_index], extras_from_base.name)
                if extras_from_base.s1 < 0:
                    for layer in sub:
                        y = layer (y)
                else:
                    for layer in sub[:extras_from_base.s1]:
                        x = layer(x)
                    y = x 
                    for layer in sub[extras_from_base.s1:]:
                        x = layer(x)
                    end_layer_index += 1
                
            start_layer_index = end_layer_index
            
            conf_output, loc_output = self.get_multibox_layer_output(header_index, y)
            header_index += 1
            conf.append(conf_output)
            loc.append(loc_output)
        
        # connent rest base network
        for layer in self.base[end_layer_index:]:
            x = layer(x)

        # add extras layer 
        for layer in self.extras_layers:
            x = layer(x)
            conf_output, loc_output = self.get_multibox_layer_output(header_index, x)
            header_index += 1
            conf.append(conf_output)
            loc.append(loc_output)

        loc = torch.cat(loc, 1)
        conf = torch.cat(conf, 1)


        if self.is_train:
            output = (
                loc, conf
            )
        else:
            output = (
                loc,
                torch.max(self.sigmoid(conf.view(-1, self.n_classes)), dim = 1)[0],
                torch.max(self.sigmoid(conf.view(-1, self.n_classes)), dim = 1)[1],
            )

        return output



if __name__ == '__main__':
    pass



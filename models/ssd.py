
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple


GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  


class SSD(nn.Module):
    '''single shot multibox architecture
       backbone network is mobilenetv2
    '''
    def __init__(self,
                 base,
                 extras_layers,
                 loc_layers,
                 conf_layers,
                 source_layer_indexes,
                 n_classes,
                 dropout_ratio = 0.1):
        super(SSD, self).__init__()       
        '''
        Args:
            base (nn.Module): backbone network - MobileNetV2 extract feature part
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
        
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        loc = []
        conf = []
        source = []
        start_layer_index = 0
        header_index = 0

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
                sub = getattr(self.base[end_layer_index], extras_from_base.name)
                
                for layer in sub[:extras_from_base.s1]:
                    x = layer(x)
                    
                y = x
                
                for layer in sub[extras_from_base.s1:]:
                    x = layer(x)
                    
                end_layer_index += 1
                
            start_layer_index = end_layer_index
            source.append(y)
        
        # connent rest base network
        for layer in self.base[end_layer_index:]:
            x = layer(x)

        # add extras layer 
        for layer in self.extras_layers:
            x = layer(x)
            source.append(x)
        
        for (x, l, c) in zip(source, self.loc_layers, self.conf_layers):
            loc.append( l(x).permute(0, 2, 3, 1).contiguous())
            conf.append( c(x).permute(0, 2, 3, 1).contiguous())
        
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
       
        output = (
            loc.view(loc.size(0), -1, 4), 
            conf.view(conf.size(0), -1, self.n_classes)
        )
        return output



if __name__ == '__main__':
    pass
        

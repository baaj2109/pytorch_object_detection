
import os
from math import sqrt
import torch
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                
                # center of prior box (cx, cy)
                cx = (j + 0.5) / f_k 
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if self.max_sizes[k] != None:
                    s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    
    MOBILEV2_512 = {
        "feature_maps": [32, 16, 8, 4, 2, 1],
        "min_dim": 512,
        "steps": [16, 32, 64, 128, 256, 512],
        "min_sizes": [102.4,  174.08, 245.76, 317.44, 389.12, 460.8],
        "max_sizes": [174.08, 245.76, 317.44, 389.12, 460.8,  512  ],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        "variance": [0.1, 0.2],
        "clip": True,
    }    

    priorbox = PriorBox(MOBILEV2_512)
    output = priorbox.forward()
    print("output shape: ",output.shape)

    '''save priors to binary file 
    '''
    # np_priors = priors.detach().numpy()
    # with open("./priors.dat", "wb") as binfile:
    #     binfile.write(np_priors.tobytes())












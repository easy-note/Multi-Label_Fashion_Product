import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

class MultiHeadResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(MultiHeadResNet50, self).__init__()
        if pretrained == True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)

        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')

        # change the final layer
        self.layer_gender = nn.Linear(2048, 5)
        self.layer_articletype = nn.Linear(2048, 142)
        self.layer_season = nn.Linear(2048, 4)
        self.layer_usage = nn.Linear(2048, 9)
        
    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        layer_gender = self.layer_gender(x)
        layer_articletype = self.layer_articletype(x)
        layer_season = self.layer_season(x)
        layer_usage = self.layer_usage(x)
        return layer_gender, layer_articletype, layer_season, layer_usage
  
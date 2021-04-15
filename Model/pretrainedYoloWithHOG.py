import os
import sys
sys.path.insert(0, os.path.join(os.path.curdir, "HOGSVM"))

import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

from HOGextraction import getFlattenedHOGFeatures

# image size 3 x 448 x 448

vgg16 = models.vgg16(pretrained=True)
class pretrainedYoloWithHOG(nn.Module):
    def __init__(self, in_channels=3, model=vgg16, **kwargs):
        super(pretrainedYoloWithHOG , self).__init__()
        self.in_channels = in_channels
        # get all the weigths except for classifier layer
        self.pretrained = torch.nn.Sequential(*(list(model.children())[:-1]))
        for name, weights in self.pretrained.named_parameters():
            weights.requires_grad = False
        self.fcs = self._create_fcs(**kwargs)

    
    def forward(self, x):
        features = nn.Flatten()(self.pretrained(x))
        # print(features.shape)
        HOG = getFlattenedHOGFeatures(x).reshape(4, -1).to(features.device)
        # print(HOG.type())
        return self.fcs(torch.cat((features, HOG), dim=1))


    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7 + 54450, 496), # original paper 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)), # (S, S, 12)
        )
        
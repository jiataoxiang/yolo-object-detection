import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


vgg16 = models.vgg16(pretrained=True)
class pretrainedYolo(nn.Module):
    def __init__(self, in_channels=3, model=vgg16, **kwargs):
        super(pretrainedYolo, self).__init__()
        self.in_channels = in_channels
        self.pretrained = model
        self.fcs = self._create_fcs(**kwargs)
        # remove classifier and add our own classifier
        self.pretrained.classifier = self.fcs
        for name, weights in self.pretrained.named_parameters():
            if not name.startswith("classifier"):
                weights.requires_grad = False
        # for name, weights in model.named_parameters():
        #     print(weights.requires_grad)
        # summary(self.pretrained, (3, 448, 448))
    
    def forward(self, x):
        return self.pretrained(x)


    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 496), # original paper 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)), # (S, S, 12)
        )
        
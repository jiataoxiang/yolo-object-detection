import torch
from torchvision import models


if __name__ == "__main__":
    vgg16 = models.vgg16(pretrained=True)
    models = torch.nn.Sequential(*(list(vgg16.children())[:-1]))
    print(models)
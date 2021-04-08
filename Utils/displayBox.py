import torch
import pathlib
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import patches
import sys
import os
sys.path.insert(0, os.path.join(os.path.curdir, "DataLoader"))
print(sys.path[0])
from MaskDataset import MaskDataset



def displayBox(image, targetLabels):
    """
        image: image tensor (channel, height, width)
        targetLabels: list of [c, xcenter, ycenter, widthNorm, heightNorm]
    """
    # denormalize
    image_height, image_width = image.shape[1:]
    mul = [1,image_width,image_height,image_width,image_height] #[constant, image_width, image_height, image_width, image_height]
    finalStage = []
    for x in targetLabels:
        c, xx, yy, w, h = x[0] * mul[0], x[1] * mul[1], x[2] * mul[2], x[3] * mul[3], x[4] * mul[4]
        finalStage.append([c, xx, yy, w, h])
    
    fig = plt.figure()
    # add axis at position rect
    ax = fig.add_axes([0, 0,1,1]) # left, bottom, width, height, which are fraction of figure width height

    plt.imshow(image.permute(1, 2, 0))

    for x in finalStage:
        class_ = int(x[0])
        xcenter, ycenter, width, height = x[1:]
        xmin, ymin = xcenter - width//2, ycenter - height//2
        xmax, ymax = xcenter + width//2, ycenter + height//2
        if class_ == 0:
            edgecolor = "r"
            ax.annotate('MASK', xy=(xmax - 40, ymin + 20))
        elif class_ == 1:
            edgecolor = "b"
            ax.annotate('NO MASK', xy=(xmax - 40, ymin + 20))
        # adding bounding box
        rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
        ax.add_patch(rect)
    
    plt.show()

if __name__ == "__main__":
    trainPath = pathlib.Path.cwd() / "dataset/train"
    dataset = MaskDataset(trainPath / "images", trainPath / "labels")
    image, labels, label_matrix = dataset[0]
    displayBox(image, labels)
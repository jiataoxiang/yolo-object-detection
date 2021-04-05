import os
import torch
import pathlib
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import patches

class MaskDataset(Dataset):
    def __init__(self, input_dirs_path, target_dirs_path, transformer=ToTensor()):
        self.input_dirs_path = input_dirs_path
        self.target_dirs_path = target_dirs_path
        self.input_dirs = os.listdir(input_dirs_path)
        self.target_dirs = os.listdir(target_dirs_path)
        self.transformer = transformer

    def __len__(self):
        return len(self.input_dirs)

    def __getitem__(self, index):
        inputImage = self.transformer(Image.open(self.input_dirs_path / self.input_dirs[index]))
        targetFile = open(self.target_dirs_path / self.target_dirs[index])
        targets = []
        for line in targetFile:
            targets.append(line.strip().split())
        targetFile.close()
        # convert string to int
        targetLabels = []
        for i in range(len(targets)):
            targetLabels.append(list(map(float, targets[i])))
        return inputImage, targetLabels


if __name__ == "__main__":
    trainPath = pathlib.Path.cwd() / "dataset/train"
    dataset = MaskDataset(trainPath / "images", trainPath / "labels")
    image, labels = dataset[0]
    # denormalize
    mul = [1,2000,1363,2000,1363] #[constant, image_width, image_height, image_width, image_height]
    finalStage = []
    for x in labels:
        c, xx, yy, w, h = x[0] * mul[0], x[1] * mul[1], x[2] * mul[2], x[3] * mul[3], x[4] * mul[4]
        finalStage.append([c, xx, yy, w, h])
    
    fig = plt.figure()
    # add axis at position rect
    ax = fig.add_axes([0, 0,1,1]) # left, bottom, width, height, which are fraction of figure width height

    plt.imshow(image.permute(1, 2, 0))

    for x in finalStage:
        class_ = int(x[0])
        xcenter, ycenter, width, height = x[1:]
        xmin, ymin = xcenter - width//2, ycenter - width//2
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



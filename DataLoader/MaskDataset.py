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
    def __init__(self, input_dirs_path, target_dirs_path, S=7, B=2, C=2, transformer=None):
        self.input_dirs_path = input_dirs_path
        self.target_dirs_path = target_dirs_path
        self.input_dirs = os.listdir(input_dirs_path)
        # if len(self.input_dirs) > 300:
        #     self.input_dirs = self.input_dirs[:300]
        self.target_dirs = os.listdir(target_dirs_path)
        self.transformer = transformer
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.input_dirs)

    def __getitem__(self, index):
        imageName = self.input_dirs[index]
        prefix = imageName[:-4]
        inputImage = Image.open(self.input_dirs_path / imageName)
        targetFile = open(self.target_dirs_path / (prefix + ".txt"))
        targets = []
        for line in targetFile:
            targets.append(line.strip().split())
        targetFile.close()
        # convert string to int
        targetLabels = []
        for i in range(len(targets)):
            target = list(map(float, targets[i]))
            target[0] = int(target[0])
            targetLabels.append(target)
        targetLabels = torch.tensor(targetLabels)
        if self.transformer:
            inputImage, targetLabels = self.transformer(inputImage, targetLabels)

        # label matrix
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in targetLabels:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            # get label relative to each cell
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = width * self.S, height * self.S
            if label_matrix[i, j, self.C] == 0: # set to 1 if cell exist object
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, self.C + 1 :self.C + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1
        return inputImage, label_matrix
        # return inputImage, targetLabels, label_matrix

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

import torchvision.transforms as transforms
transformer = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

if __name__ == "__main__":
    trainPath = pathlib.Path.cwd() / "dataset/train"
    dataset = MaskDataset(trainPath / "images", trainPath / "labels", transformer=transformer)
    # image, labels, label_matrix = dataset[0]
    image, label_matrix = dataset[0]
    for inputImage, label_matrix in dataset:
        print(targetLabels.shape)
    # print(label_matrix)
    



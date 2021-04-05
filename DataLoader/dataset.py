import os
import torch
import pathlib
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

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
    print(dataset[0])

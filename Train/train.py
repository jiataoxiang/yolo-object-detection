import sys
import os
sys.path.insert(0, os.path.join(os.path.curdir, "Utils"))
sys.path.insert(0, os.path.join(os.path.curdir, "DataLoader"))
sys.path.insert(0, os.path.join(os.path.curdir, "Model"))

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from MaskDataset import MaskDataset
from Yolo import Yolov1
from util import (
    nonMaxSuppression,
    MeanAveragePrecision,
    IoU,
    cellBoxesToBoxes,
    getBoundingBoxes,
    plotImage,
    saveCheckpoint,
    loadCheckpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 4 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 40
NUM_WORKERS = 1
PIN_MEMORY = True
SAVE_MODEL_FILE = "Yolo.pth.tar"
IMG_TRAIN_DIR =  pathlib.Path.cwd() / "dataset/train/images"
LABEL_TRAIN_DIR =  pathlib.Path.cwd() / "dataset/train/labels"



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transformer = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

# train loop for one iteration
def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())
        del x, y, out, loss

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")
    return sum(mean_loss)/len(mean_loss)


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    train_dataset = MaskDataset(
        input_dirs_path=IMG_TRAIN_DIR,
        target_dirs_path=LABEL_TRAIN_DIR,
        transformer=transformer,
    )


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    mAPs = []
    losses = []
    for epoch in range(EPOCHS):
        
        pred_boxes, target_boxes = getBoundingBoxes(
            train_loader, model, IoUThreshold=0.5, Probabilitythreshold=0.4
        )

        mean_avg_prec = MeanAveragePrecision(
            pred_boxes, target_boxes, IoUThreshold=0.5
        )

        mAPs.append(mean_avg_prec)

        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.6:
           checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
           saveCheckpoint(checkpoint, filename=SAVE_MODEL_FILE)
           import time
           time.sleep(10)

        # train for a single step
        loss = train_fn(train_loader, model, optimizer, loss_fn)
        losses.append(loss)
    plt.plot(np.arange(epoch + 1), np.array(mAPs), "b")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.savefig("YoloMAP.jpg")

    plt.plot(np.arange(epoch + 1), np.array(losses), "b")
    plt.xlabel("Epoch")
    plt.ylabel("mean loss")
    plt.savefig("YoloLoss.jpg")


if __name__ == "__main__":
    main()
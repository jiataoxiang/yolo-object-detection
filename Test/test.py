import sys
import os
sys.path.insert(0, os.path.join(os.path.curdir, "Utils"))
sys.path.insert(0, os.path.join(os.path.curdir, "DataLoader"))
sys.path.insert(0, os.path.join(os.path.curdir, "Model"))

import torch
import torch.optim as optim
import torchvision.transforms as transforms
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
from torch.utils.data import DataLoader
from MaskDataset import MaskDataset
from Yolo import Yolov1
from pretrainedYolo import pretrainedYolo
from pretrainedYoloWithHOG import pretrainedYoloWithHOG
import pathlib

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 1 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL_FILE = "Yolo.pth.tar"
IMG_TEST_DIR = pathlib.Path.cwd() / "dataset/test/images"
LABEL_TEST_DIR =  pathlib.Path.cwd() / "dataset/test/labels"

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

def test():
    # Load model
    # get test data
    # predict and draw bounding box
    model = Yolov1(split_size=7, num_boxes=2, num_classes=2).to(DEVICE)
    # model = pretrainedYolo(split_size=7, num_boxes=2, num_classes=2).to(DEVICE)
    # model = pretrainedYoloWithHOG(split_size=7, num_boxes=2, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loadCheckpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    test_dataset = MaskDataset(
        input_dirs_path=IMG_TEST_DIR,
        target_dirs_path=LABEL_TEST_DIR,
        transformer=transformer,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
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
    
    test_data = test_dataset[0]
    # print(test_data)
    # display image
    # for batch_idx, (x, y) in enumerate(test_loader):
        # x, y = x.to(DEVICE), y.to(DEVICE)
    pred_boxes, target_boxes = getBoundingBoxes(
        test_loader, model, IoUThreshold=0.5, Probabilitythreshold=0.4
    )
    mean_avg_prec = MeanAveragePrecision(
        pred_boxes, target_boxes, IoUThreshold=0.5
    )
    print(f"Test mAP: {mean_avg_prec}")
        # bboxes = cellBoxesToBoxes(model(x))
        # bboxes = nonMaxSuppression(bboxes[0], 0.5, 0.4)
        # plotImage(x[0].permute(1,2,0).to("cpu"), bboxes)


if __name__ == "__main__":
    test()
import sys
import os
sys.path.insert(0, os.path.join(os.path.curdir, "Utils"))
sys.path.insert(0, os.path.join(os.path.curdir, "DataLoader"))
sys.path.insert(0, os.path.join(os.path.curdir, "Model"))

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from util import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from torch.utils.data import DataLoader
from MaskDataset import MaskDataset
from Yolo import Yolov1
import pathlib

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 1 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL_FILE = "overfit.pth.tar"
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
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

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
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        bboxes = cellboxes_to_boxes(model(x))
        bboxes = non_max_suppression(bboxes[0], iou_threshold=0.6, threshold=0.5, box_format="midpoint")
        plot_image(x[0].permute(1,2,0).to("cpu"), bboxes)


if __name__ == "__main__":
    test()
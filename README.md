## CSC413 Project

### Dependency requirement

* numpy
* matplotlib
* pathlib
* torch
* torchvision
* tqdm
* PIL
* cv2
* pathlib

### Dataset

We download data from roboflow: https://public.roboflow.com/object-detection/mask-wearing/4/download

### Data Augmentation:

Make sure you have folder structured as:

dataset/train/images, dataset/train/labels, dataset/test/images, dataset/test/labels

You should put train and validation images and labels into train folder first.

Run `python ./Util/image_aug.py` to do image augmentation

### Train Model

make sure you are in the /path/to/CSC413PROJECT directory

Run `python ./Train/train.py` to train with Yolo model

Run `python./Train/trainV2mod.py` to train with Yolo version 2 

Run `python ./Train/trainWithPretrained.py` to train with pretrained Yolo model

Run `python ./Train/trainWithHOG.py` to train with Yolo HOG model

You can specify the path to store the model by change `SAVE_MODEL_FILE` attribute  

### Test Model

make sure you are in the /path/to/CSC413PROJECT directory

Run `python ./Test/test.py` to run test file, and here are several attribute you need to know.

`LOAD_MODEL_FILE`: the path to the model you want to load

`model`: you have to use correct Model because different model have different architecture


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

The dataset we used can be retrieved from roboflow: 
https://public.roboflow.com/object-detection/mask-wearing/4/download

### Data Augmentation:

Make sure the following paths exists, where <root> is the root directory of this project:

<root>/dataset/train/images, <root>/dataset/train/labels, <root>/dataset/test/images, <root>/dataset/test/labels

Put all images in <root>/dataset/train/images, and put all labels in <root>/dataset/train/labels, then 
run `python ./Util/image_aug.py` to do image augmentation.

In <root>/Util/image_aug.py, the ratio trimed of the new images can be modified by changing the list "keep_rates", and the ratio bewteen the size of the train set and test set can be modified by changing "train_ratio".

### Train Model

make sure you to run under the directory /path/to/CSC413PROJECT 

Run `python ./Train/train.py` to train with Yolo model

Run `python./Train/trainV2mod.py` to train with our slightly modified Yolo V2 model 

Run `python ./Train/trainWithPretrained.py` to train with pretrained Yolo model

Run `python ./Train/trainWithHOG.py` to train with Yolo HOG model

You can specify the path to store the model by change `SAVE_MODEL_FILE` attribute  

### Test Model

make sure you to run under the directory /path/to/CSC413PROJECT  

Run `python ./Test/test.py` to run test file, and here are several attribute you need to know.

`LOAD_MODEL_FILE`: the path to the model you want to load

`model`: you have to use correct Model because different model have different architecture


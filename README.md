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

### Train Model

make sure you are in the /path/to/CSC413PROJECT directory

Run `python ./Train/train.py` to train with Yolo model

Run `python./Train/trainV2mod.py` to train with Yolo version 2 

Run `python ./Train/trainWithPretrained.py` to train with pretrained Yolo model

Run `python ./Train/trainWithHOG.py` to train with Yolo HOG model

You can specify the path to store the model by change `LOAD_MODEL_FILE` attribute  

### Test Model

make sure you are in the /path/to/CSC413PROJECT directory

Run `python ./Test/test.py` to run test file, and here are several attribute you need to know.

`LOAD_MODEL_FILE`


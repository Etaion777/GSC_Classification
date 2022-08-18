

# MObileNet V3-based Speech Commands Classifier using MFCC Features

Gong Cheng

York University


# Project Instruction
To run this project, the following steps need to be taken:
1. Download the datafolder from the provided link.
2. Extract all the dataset folders to ./Datasets
3. Modify the following local project paths to your own paths in:
	Utils.project_root

# Training

To training with all 35 classes, please run:
> $ python MobileNetV3_Train_and_Vali.py --class_num 35

To training with 8 essential classes, please run:
> $ python MobileNetV3_Train_and_Vali.py --class_num 8

To training with 4 classes for fast prototyping, please run:
> $ python MobileNetV3_Train_and_Vali.py --class_num 4


# Inference
> $ python MobileNetV3_Infer.py --class_num 35 (8, or 4)

A couple of helper scripts are provided in the file ./Utils/support.py that should be used independently.


# Reference

https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

https://arxiv.org/abs/1804.03209

https://pytorch.org/vision/main/models/mobilenetv3.html
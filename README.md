

#MObileNet V3-based Speech Commands Classifier using MFCC Features

Gong Cheng

York University

Aug 11th, 2022


#Project Instruction
To run this project, the following steps need to be taken:
1. Download the datafolder from the provided link.
2. Extract all the dataset folders to ./Datasets
3. Modify the following local project paths to your own paths:
	3.1 Eval.py Line 9
	3.2 MobileNetV3_Infer Line 42-44
	3.3 MobileNetV3_Train_and_Vali 42-43

#Training

To training with all 35 classes, please run:
> $ python MobileNetV3_Train_and_Vali.py --class_num 35

To training with 8 essential classes, please run:
> $ python MobileNetV3_Train_and_Vali.py --class_num 8

To training with 4 classes for fast prototyping, please run:
> $ python MobileNetV3_Train_and_Vali.py --class_num 4


#Inference
> $ python MobileNetV3_Infer.py --class_num 35 (8, or 4)

A couple of helper scripts are provided in the file ./Utils/support.py that should be used independently.


#Reference

https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

https://arxiv.org/abs/1804.03209

https://pytorch.org/vision/main/models/mobilenetv3.html
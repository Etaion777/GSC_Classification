
# This file contains all the different transforms for each networks.

import torchvision.transforms as transforms

# RESNET Transforms
def resnet_transforms():
    transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    return transform

# My simple transform
def simple_transforms():
    transform = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    return transform

# MobileNetV3 Small transforms
def mobilenetv3_small():
    transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    return transform
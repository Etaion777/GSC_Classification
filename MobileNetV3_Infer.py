# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import os
import argparse
import time
import datetime
import math
import sys

from tqdm import tqdm

# ----------------------------------------
import torchvision
import torchvision.transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.optim as optim
import torchaudio


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# ----------------------------------------
from Models.mobilenetv3_small import MOBILENET_V3_SMALL
import Utils.support as support
import Utils.project_root as proj_root
import Dataloaders.GCS_Dataset as GCS_Dataset
import Transforms.proj_transforms as proj_transforms

#==============================================
# Constants
#==============================================

# System params.
PROJECT_ROOT = proj_root.PROJECT_ROOT
DATASET_PATH = PROJECT_ROOT+'Datasets/GSC_Sub_Set_8/'

# 8-Class Example
MODEL_PATH = PROJECT_ROOT+'Results/Training_WITH_VALIDATION_2022-08-18_122703/mobilenetv3_small_2022-08-18_123516.pth'
SAVE_PATH = PROJECT_ROOT +'Results/Training_WITH_VALIDATION_2022-08-18_122703/'
SAVE_NAME = "Confusion_Matrix.png"

# # 4-Class Example
# # Command: python MobileNetV3_Infer.py --class_num 4
# MODEL_PATH = PROJECT_ROOT+'Results/Training_WITH_VALIDATION_2022-08-18_134700/mobilenetv3_small_2022-08-18_135047.pth'
# SAVE_PATH = PROJECT_ROOT +'Results/Training_WITH_VALIDATION_2022-08-18_134700/'
# SAVE_NAME = "Confusion_Matrix.png"


# Training params
BATCH_SIZE = 1
LOG_NAME = 'Testing'
MODEL = 'mobilenetv3_small'
CLASS_NUM = 8

#==============================================
# arg parser
#==============================================
parser = argparse.ArgumentParser(description='inference')
parser.add_argument('--dataset_path', default=DATASET_PATH)
parser.add_argument('--model_path', default=MODEL_PATH)
parser.add_argument('--save_path', default=SAVE_PATH)
parser.add_argument('--save_name', default=SAVE_NAME)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--model', default=MODEL)
parser.add_argument('--class_num', type=int, default=CLASS_NUM)
parser.add_argument('--result_folder', default=LOG_NAME)
args = parser.parse_args()



def main():

    #---------------------------------
    # Dataset Selection
    #---------------------------------
    if args.class_num == 35:
        args.dataset_path = PROJECT_ROOT + 'Datasets/GSC_Dataset/'
    elif args.class_num == 4:
        args.dataset_path = PROJECT_ROOT + 'Datasets/GSC_Sub_Set/'

    #---------------------------------
    # Model Selection
    #---------------------------------
    # Get cpu or gpu device for inference.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Load the training model.
    if args.model == 'mobilenetv3_small':

        training_model = MOBILENET_V3_SMALL(args.class_num)
        transform = proj_transforms.mobilenetv3_small()

    elif args.model == 'resnet18':
        transform = proj_transforms.resnet_transforms()
        training_model = torchvision.models.resnet18(pretrained=True).to(device)
        training_model.fc = nn.Linear(512,args.class_num).to(device)

    training_model.load_state_dict(torch.load(args.model_path))

    # Switch to the evaluation mode.
    training_model.eval()
    
    #---------------------------------
    # Data Loading
    #---------------------------------
    # Load the training & validation sets for training the network.
    test_wav_set, test_label_set = support.dataset_parsing(args.dataset_path+'Test',support.class_options(args.class_num))
    test_dataset =GCS_Dataset.SC_MFCC(test_wav_set, test_label_set,transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)
    

    #---------------------------------
    # Evaluation
    #---------------------------------
    test_correct = 0
    test_total = 0
    test_by_class = np.array([0]*(args.class_num))
    test_overall = np.array([0]*(args.class_num))
    test_result_mat = np.zeros((args.class_num,args.class_num))

    for (inputs, labels) in test_loader:
        # get the inputs from CPU and send them to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Accumulate the validation loss.
        outputs = training_model(inputs)

        # Compute the accuracy.
        _, test_predicted = torch.max(outputs, 1)

        # Update overall ACC
        test_total += labels.size(0)
        test_correct += (test_predicted==labels).sum().item()

        # Update by-class ACC
        test_overall[labels.item()]+=1
        if test_predicted == labels:
            test_by_class[labels.item()]+=1

        # Update confusion matrix.
        # print(test_predicted.item())
        test_result_mat[test_predicted.item()][labels.item()]+=1

    test_acc = 100*test_correct/len(test_loader)
    test_by_class_acc = 100*(test_by_class/test_overall)

    #---------------------------------
    # Display the inference results.
    #---------------------------------
    print('='*50)
    print(f'Test Vol: {len(test_loader)}')
    print(f'Test_Acc: {test_acc:.2f}')
    support.print_labels(args.class_num,test_by_class_acc)
    print(test_result_mat)
    print(support.class_options(args.class_num))
    print('='*50)

    # Show Confusion matrix.
    support.confusion_mat(args.class_num,test_result_mat,args.save_path+args.save_name)

if __name__ == "__main__":
    main()
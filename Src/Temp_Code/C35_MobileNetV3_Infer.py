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
import Dataloaders.GCS_Dataset as GCS_Dataset
import Transforms.proj_transforms as proj_transforms

#==============================================
# Constants
#==============================================

# System params.
dirname = os.path.dirname
PROJECT_ROOT = '/home/gong/Gong/Pytorch_Sample_Projects/Speech_Classification_V2/'
DATASET_PATH = '/home/gong/Gong/Pytorch_Sample_Projects/Speech_Classification_V2/Datasets/GSC_Sub_Set/'
MODEL_PATH = PROJECT_ROOT+'/'+'Results/Training_WITH_VALIDATION_2022-08-10_224353/mobilenetv3_small_2022-08-10_224711.pth'

# Training params
BATCH_SIZE = 1
LOG_NAME = 'Testing'
MODEL = 'mobilenetv3_small'
CLASS_NUM = 35

#==============================================
# arg parser
#==============================================
parser = argparse.ArgumentParser(description='inference')
parser.add_argument('--dataset_path', default=DATASET_PATH)
parser.add_argument('--model_path', default=MODEL_PATH)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--model', default=MODEL)
parser.add_argument('--class_num', type=int, default=CLASS_NUM)
parser.add_argument('--result_folder', default=LOG_NAME)
args = parser.parse_args()



def main():
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
    for (inputs, labels) in test_loader:
        # get the inputs from CPU and send them to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Accumulate the validation loss.
        outputs = training_model(inputs)

        # Compute the accuracy.
        _, test_predicted = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_correct += (test_predicted==labels).sum().item()

    test_acc = 100*test_correct/len(test_loader)

    print(f'EVali_Acc: {test_acc:.2f}\n')
    print(f'len(test_loader): {len(test_loader)}')

if __name__ == "__main__":
    main()
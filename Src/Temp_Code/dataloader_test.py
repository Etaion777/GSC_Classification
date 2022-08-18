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


# Training params
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
NUM_EPOCHS = 2
WEIGHT_DECAY = 1e-5
LOG_NAME = 'Training_WITH_VALIDATION'
MODEL = 'mobilenetv3_small'
VALI_STEP = 1 # Measured in epoch
BATCH_STEP = 20
CLASS_NUM = 8


#==============================================
# arg parser
#==============================================
parser = argparse.ArgumentParser(description='training')
parser.add_argument('--dataset_path', default=DATASET_PATH)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--model', default=MODEL)
parser.add_argument('--lr', type=float, default=LEARNING_RATE)
parser.add_argument('--momentum', type=float, default=MOMENTUM)
parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
parser.add_argument('--vali_step', type=int, default=VALI_STEP)
parser.add_argument('--class_num', type=int, default=CLASS_NUM)
parser.add_argument('--result_folder', default=LOG_NAME)
args = parser.parse_args()


#---------------------------------
# Dataset Selection
#---------------------------------
if args.class_num == 35:
    args.dataset_path = PROJECT_ROOT + 'Datasets/GSC_Dataset/'
elif args.class_num == 4:
    args.dataset_path = PROJECT_ROOT + 'Datasets/GSC_Sub_Set/'

#---------------------------------
# Data Loading
#---------------------------------
# Load the training & validation sets for training the network.
train_wav_set, train_label_set = support.dataset_parsing(args.dataset_path+'Training',support.class_options(args.class_num))    
vali_wav_set, vali_label_set = support.dataset_parsing(args.dataset_path+'Validation',support.class_options(args.class_num))
# test_wav_set, test_label_set = support.dataset_parsing(args.dataset_path+'Test',support.class_options(args.class_num))

# Setup the image transforms.
transform = proj_transforms.mobilenetv3_small()

# Setup the dataloaders.
train_dataset =GCS_Dataset.SC_MFCC(train_wav_set, train_label_set,transform)
vali_dataset =GCS_Dataset.SC_MFCC(vali_wav_set, vali_label_set,transform)


# print('='*60)
# print(len(train_dataset))
# print('='*60)

# raw_img = train_dataset[0][0]
# # print(f'raw_img_shape: {raw_img.shape}')
# raw_img.show()


# np_img = np.array(raw_img)
# print(np_img.max(),np_img.min())


# Create two dataloaders.
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=2)

vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=1,
                                      shuffle=True, num_workers=2)


# img, label = next(iter(train_loader))

# print(img.shape)
# print(label)
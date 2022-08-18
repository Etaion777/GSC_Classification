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
DATASET_PATH = '/home/gong/Gong/Pytorch_Sample_Projects/Speech_Classification_V2/Datasets/GSC_Dataset/'


# Training params
BATCH_SIZE = 64
LEARNING_RATE = 0.1
MOMENTUM = 0.9
NUM_EPOCHS = 10
WEIGHT_DECAY = 1e-5
LOG_NAME = 'Training_WITH_VALIDATION'
MODEL = 'mobilenetv3_small'
VALI_STEP = 1 # Measured in epoch
BATCH_STEP = 20
CLASS_NUM = 35

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


#==============================================
# Functions
#==============================================
def folder_stamp(log_name):
    filename = log_name + '_' +\
    datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return filename


def main():
    #---------------------------------
    # Initial Settings
    #---------------------------------
    # Build up the training folder for saving the model & log.
    folder_name = folder_stamp(args.result_folder)
    output_folder = os.path.join(PROJECT_ROOT,'Results/',folder_name)
    os.mkdir(output_folder)

    # Create a log file 
    txt_file = open(output_folder+'/'+folder_name+'_LOG.txt','w+')

    # Record the header of the log.
    txt_file.write('-------------Training Parameters--------------\n')
    txt_file.write('LEARNING_RATE: ' + str(args.lr)+'\n')
    txt_file.write('MOMENTUM: ' + str(args.momentum)+'\n')
    txt_file.write('WEIGHT_DECAY: ' + str(args.weight_decay)+'\n')
    txt_file.write('NUM_EPOCHS: ' + str(args.num_epochs)+'\n')
    txt_file.write('BATCH_SIZE: ' + str(args.batch_size)+'\n')
    txt_file.write('VALI_STEP: ' + str(args.vali_step)+'\n')
    

    #---------------------------------
    # Model Selection
    #---------------------------------
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    txt_file.write('DEVICE: ' + str(device)+'\n')

    # Load the training model.
    if args.model == 'mobilenetv3_small':
        txt_file.write('MODEL: ' + MODEL +'\n')

        training_model = MOBILENET_V3_SMALL(args.class_num)
        transform = proj_transforms.mobilenetv3_small()
        counter = 0
        for param in training_model.parameters():
            counter+=1

        for param in training_model.parameters():
            if counter >= 3:
                counter-=1
                param.requires_grad = False

    elif args.model == 'resnet18':
        txt_file.write('MODEL: ' + 'RESNET18' +'\n')
        transform = proj_transforms.resnet_transforms()
        training_model = torchvision.models.resnet18(pretrained=True).to(device)
        training_model.fc = nn.Linear(512,args.class_num).to(device)
    print(training_model)



    #---------------------------------
    # Data Loading
    #---------------------------------
    # Load the training & validation sets for training the network.
    train_wav_set, train_label_set = support.dataset_parsing(args.dataset_path+'Training',support.class_options(args.class_num))    
    vali_wav_set, vali_label_set = support.dataset_parsing(args.dataset_path+'Validation',support.class_options(args.class_num))
    # test_wav_set, test_label_set = support.dataset_parsing(args.dataset_path+'Test',support.class_options(args.class_num))

    # Setup the dataloaders.
    train_dataset =GCS_Dataset.SC_MFCC(train_wav_set, train_label_set,transform)
    vali_dataset =GCS_Dataset.SC_MFCC(vali_wav_set, vali_label_set,transform)



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=2)

    vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=1,
                                          shuffle=True, num_workers=2)

    txt_file.write('Training Vol.: ' + str(len(train_loader.dataset)) +'\n')
    txt_file.write('Validation Vol.: ' + str(len(vali_loader.dataset)) +'\n')
    txt_file.write('----------------------------------------------\n')

    #---------------------------------
    # Network Training
    #---------------------------------
    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(training_model.parameters(),
        lr = args.lr, momentum = args.momentum,
        weight_decay=args.weight_decay)

    
    for epoch in range(args.num_epochs):

        running_loss = 0.0
        vali_loss = 0.0

        vali_correct = 0
        vali_total = 0


        # Set a timer.
        t_start = time.time()

        # -------------------------------------
        # Set to the training mode.
        # -------------------------------------
        training_model.train()

        for ii, (inputs, labels) in enumerate(train_loader, 0):

            # get the inputs from CPU and send them to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels)
            # zero the gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            outputs = training_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()      
        
        running_loss /= len(train_loader)

        # Update the timer.
        delta = time.time() - t_start

        # -------------------------------------
        # Switch to validation
        # -------------------------------------
        if (epoch+1)%VALI_STEP == 0:

            # switch to evaluation mode.
            training_model.eval()

            for (inputs, labels) in vali_loader:
                # get the inputs from CPU and send them to the GPU
                inputs, labels = inputs.to(device), labels.to(device)

                # Accumulate the validation loss.
                outputs = training_model(inputs)
                
                # Compute the accuracy.
                _, vali_predicted = torch.max(outputs, 1)
                vali_total += labels.size(0)
                vali_correct += (vali_predicted==labels).sum().item()

                loss = criterion(outputs, labels)
                vali_loss += loss.item()

            # Compute validation loss.
            vali_loss /= len(vali_loader)

        vali_acc = 100*vali_correct/vali_total

        print(f'Epoch #{epoch+1}\t Time: {delta:.1f}s\t Train_Loss: {running_loss:.4f}\t Vali_Loss: {vali_loss:.4f}\t Vali_Acc: {vali_acc:.2f}\n')
        txt_file.write(f'Epoch #{epoch+1}\t Time: {delta:.1f}s\t Train_Loss: {running_loss:.4f}\t Vali_Loss: {vali_loss:.4f}\t Vali_Acc: {vali_acc:.2f}\n')

        # Update timer
        delta = time.time() - t_start

    #---------------------------------
    # Save Model
    #---------------------------------
    torch.save(training_model.state_dict(),(output_folder+'/'+\
        args.model+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')+'.pth'))


    # Finally, close txt file.
    txt_file.close()



if __name__ == "__main__":
    main()
# ----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.optim as optim
import torchaudio
import sys

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa

# ----------------------------------------
from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS
import os
import time
from PIL import Image


# -----------------------------------------
# MFCC feature extraction
# -----------------------------------------
n_fft = 2048
win_length = 64
hop_length = 64
n_mels = 256
n_mfcc = 256
sample_rate = 16000

mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
      'n_fft': n_fft,
      'n_mels': n_mels,
      'hop_length': hop_length,
      'mel_scale': 'htk',
    }
)

class SC_MFCC(Dataset):

    def __init__(self, wav_set, label_set, transform=None):
        self.wav_set = wav_set
        self.label_set = label_set
        self.transform = transform

    def __len__(self):
        return len(self.label_set)

    def __getitem__(self, ind):
        
        label = self.label_set[ind]

        # Reformat the waveforms to image.
        waveform, _ = torchaudio.load(self.wav_set[ind])
        waveform = mfcc_transform(waveform).numpy().squeeze()     
        waveform = (waveform-waveform.min())/(waveform-waveform.min()).max()
        waveform = np.uint8(np.dstack((waveform,waveform,waveform))*255)
        waveform = Image.fromarray(waveform)
        waveform = waveform.resize(size=(256,256))
        # waveform = np.array(waveform, dtype=np.float32)
        # waveform = np.dstack((waveform,waveform,waveform)).transpose(2,1,0)

        if self.transform != None:
            waveform = self.transform(waveform)


        # print(img.shape)

        return waveform, label

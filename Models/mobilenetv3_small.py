import torch
import torch.nn as nn
import torchvision



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def MOBILENET_V3_SMALL(class_num=35):

	# Load the pretrianed mobilenetv3
	training_model = torchvision.models.mobilenet_v3_small(pretrained=True,).to(device)


	# Modify the last layer of classifier to 35 classes.
	training_model.classifier[3] = nn.Linear(in_features=1024, out_features=class_num, bias=True).to(device)

	return training_model
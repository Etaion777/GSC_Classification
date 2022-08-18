
import os
import time
import shutil
import matplotlib.pyplot as plt
import Utils.support as support


LOG_PATH = '/home/gong/Gong/Pytorch_Sample_Projects/Speech_Classification_V2/Results/8-class/Training_WITH_VALIDATION_2022-08-11_011757/Training_WITH_VALIDATION_2022-08-11_011757_LOG.txt'


def main():
	# Get the parsed training log.
	training_log = support.parse_log(LOG_PATH)

	# Get the train_vali_loss curve.
	support.loss_train_vali_graph(training_log,"MobileNetV3-Small Training/Validation Loss", "C8_mobilenetv3_small_train_val_loss.png")


if __name__ == "__main__":
	main()
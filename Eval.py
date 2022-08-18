
import os
import time
import shutil
import matplotlib.pyplot as plt
import Utils.support as support
import Utils.project_root as proj_root


PROJECT_ROOT = proj_root.PROJECT_ROOT
LOG_PATH = PROJECT_ROOT+'Results/Training_WITH_VALIDATION_2022-08-18_122703/Training_WITH_VALIDATION_2022-08-18_122703_LOG.txt'


def main():
	# Get the parsed training log.
	training_log = support.parse_log(LOG_PATH)

	# Get the train_vali_loss curve.
	support.loss_train_vali_graph(training_log,"MobileNetV3-Small Training/Validation Loss", "C8_mobilenetv3_small_train_val_loss_v2.png")


if __name__ == "__main__":
	main()
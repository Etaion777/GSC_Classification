
import os
import time
import shutil
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# Default: Get the class names of all 35 classes.
LABEL_LIST = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


# Divide the training scheme to Class-4 and Class-35.
def class_options(class_opt = 35):
    if class_opt == 35:
        tmp_label_list = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    elif class_opt==4:
        tmp_label_list = ['one','two','three','four']
    elif class_opt==8:
        tmp_label_list = ["one", "two", "three", "four", "up", "down", "left", "right"]

    return tmp_label_list


# Parsing the dataset for dataload creation.
def dataset_parsing(dataset_path, label_list=LABEL_LIST):

    # Define two paths for saving waves & labels repsectively.
    wav_set = []
    label_set = []

    # Parsing the dataset.
    for root, dirs, files in os.walk(dataset_path):
     for file in files:
        with open(os.path.join(root, file), "r") as auto:
            wav_set.append(root+'/'+file)
            label_set.append(label_list.index(root.split('/')[-1]))


    return wav_set, label_set


# This is used to physically split the GSC dataset into (1) Training,
#  (2) Test, and (3) Validation
def dataset_split(dataset_path,target_path,data_list):

    # Open log file.
    input_file = open(data_list,'r')
    lines = input_file.readlines()

    for line in lines:

        # Get the label & wave names
        label, wav= line.split('/')

        # Create subfolders
        isExist = os.path.exists(target_path+label)
        if not isExist:
            os.makedirs(target_path+label, mode=0o777)

        origin_path = dataset_path+label+'/'+wav[:-1]
        new_path = target_path+label+'/'+wav[:-1]


        # Move the waves to the target folder
        os.rename(origin_path, new_path)

def parse_log(input_path):

    # Create arrays saving the results.
    dic = {}


    # Open log file.
    input_file = open(input_path,'r')
    lines = input_file.readlines()

    # Line counter.
    line_counter = 0

    for line in lines:
        line_counter+=1
        if line_counter >= 13:
            tmp_array = line.split()
            # print(tmp_array)
            # time.sleep(500)
            # ['Epoch', '#1', 'Time:', '16.9s', 'Train_Loss:', '1.7508', 'Vali_Loss:', '1.4981', 'Vali_Acc:', '46.46']
            
            # Epoch number
            if 'epoch' not in dic:
                dic['epoch'] = []
            else:
                dic['epoch'].append(int(tmp_array[1][1:]))

            # Time.
            if 'time' not in dic:
                dic['time'] = []
            else:
                dic['time'].append(float(tmp_array[3][:-1]))

            # 
            if 'train_loss' not in dic:
                dic['train_loss'] = []
            else:
                dic['train_loss'].append(float(tmp_array[5]))

            # 
            if 'vali_loss' not in dic:
                dic['vali_loss'] = []
            else:
                dic['vali_loss'].append(float(tmp_array[7]))

            # 
            if 'vali_acc' not in dic:
                dic['vali_acc'] = []
            else:
                dic['vali_acc'].append(float(tmp_array[9]))

    return dic

#  Plot training/vali loss graph
def loss_train_vali_graph(result_dic,title,save_name):

    epoch = result_dic['epoch']
    train_loss = result_dic['train_loss']
    vali_loss = result_dic['vali_loss']

    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title(title)
    plt.xticks(list(range(1,epoch[-1]+1)))
    plt.plot(epoch,train_loss,'o-',label="Training Loss")
    plt.plot(epoch,vali_loss,'o-', label="Validation Loss")
    plt.legend()
    plt.savefig(save_name)
    plt.show()
    plt.close()

#  Plot vali acc graph
def acc_vali_graph(result_dic,title,save_name):

    vali_acc = result_dic['vali_acc']
    epoch = result_dic['epoch']

    ax = plt.gca()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(Title)
    plt.xticks(list(range(1,epoch[-1]+1)))
    plt.yticks(list(range(0,101)))
    plt.plot(epoch,vali_acc,'o-',label="Validation Accuracy")
    plt.savefig(save_name)
    plt.show()
    plt.close()


def confusion_mat():
    array = [[385,   0,   0,   1,   1,   4,   5,  11],
     [  1, 414,   8,   2,   2,   4,   0,   1],
     [  0,   2, 396,   0,   0,   0,   4,  26],
     [  6,   8,   1, 395,  15,   3,   3,   5],
     [  3,   0,   0,   0, 405,   1,   3,   3],
     [  0,   0,   0,   0,   2, 388,   0,   0],
     [  2,   0,   0,   2,   0,   6, 391,   4],
     [  2,   0,   0,   0,   0,   0,   6, 346]]
    df_cm = pd.DataFrame(array, index = [i for i in ['one','two','three','four','up','down','left','right']],
                      columns = [i for i in ['one','two','three','four','up','down','left','right']])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,cmap='Blues')
    plt.show()


def main():
    print('xxxxxxxxxxxxxxxx')
    confusion_mat()
    print('main')

if __name__ == "__main__":
    main()
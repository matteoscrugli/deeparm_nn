#%%
import torch
import torchvision
import torch.quantization
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pk
import pandas as pd
# import wfdb
import math
import statistics
import os
import sys
import argparse
from pathlib import Path
import shutil
import copy
import time
import json
import re

import itertools
import threading

from torch.quantization import QuantStub, DeQuantStub
from pathlib import Path
from torch.utils import data

import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib



class color:
    NONE = ''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

parser = argparse.ArgumentParser()
parser.add_argument('-t','--train', dest='train', required=True, help="training session path")
parser.add_argument('-i','--input', dest='input', required=True, nargs='*', help="input (folders or files)")
parser.add_argument('--csort', dest='csort', action='store_true', help="sort for class")

parser.add_argument('-C','--calibrate', dest='calibrate', nargs='*', help='calibration file')
parser.add_argument('-G','--gforce', dest='gforce', default='', action='store_true', help='gforce remover')

if 'ipykernel_launcher.py' in sys.argv[0]:
    # args = parser.parse_args('-t output/train/papertest5020_5 -i dataset/TT'.split())
    args = parser.parse_args('-t output/train/papertest0 -i dataset/TT'.split())
else:
    args = parser.parse_args()


session_train = args.train
session_input = args.input
session_csort = args.csort

with open(f'{session_train}/training_summary.json', 'r') as json_file:
    json_train = json.load(json_file)









class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    topn = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        printProgressBar(0, len(data_loader), prefix = 'Calibrate:', suffix = '', length = 40)
        for image, target in data_loader:
            output = model(image.float())
            loss = criterion(output, target)
            cnt += 1
            acc1, accn = accuracy(output, target, topk=(1, fully_2_outdim))
            # print('.', end = '')
            top1.update(acc1[0], image.size(0))
            topn.update(accn[0], image.size(0))
            printProgressBar(cnt, len(data_loader), prefix = 'Calibrate:', suffix = '', length = 40)
            if cnt >= neval_batches:
                 return top1, topn

    return top1, top5

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

t_done = False
def animate(prefix = ''):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if t_done:
            break
        print('\r' + prefix + c, end = '\r')
        # sys.stdout.write('\r' + prefix + c)
        # sys.stdout.flush()
        time.sleep(0.1)
    print('\r' + prefix + 'Done!')
    # sys.stdout.write('\r' + prefix + 'Done!')

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

criterion = nn.CrossEntropyLoss()

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix









#%%

#██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗
#██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝
#██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║
#██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║
#██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║
#╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝

dataset_labels = json_train['dataset_labels']
# dataset_labels = ['B', 'FB', 'S', 'R', 'G'] # fixme
dataset_path = json_train['dataset_path']
dataset_split = float(json_train['dataset_split'])
separate_validation = json_train['separate_validation']
value_batch_size = json_train['batch_size']
normalization = json_train['normalization']
random_seed = json_train['random_seed']

files_split = True

frame_len_sec = json_train['frame_len_sec']
frame_shift_sec = json_train['frame_shift_sec']
downscaling = json_train['downscaling']
# median = json_train['median']

calibrate = args.calibrate
gforce = args.gforce

try:
    median = json_train['median'] # fixme
except:
    median = 1
augmentation_rsize = json_train['augmentation_rsize'] if json_train['augmentation_rsize'] else [0]
augmentation_shift = json_train['augmentation_shift'] if json_train['augmentation_shift'] else [0]
augmentation_rotat = json_train['augmentation_rotat'] if json_train['augmentation_rotat'] else [0]



if calibrate:
    calibration_items = []
    calibration_x = 0
    calibration_y = 0
    calibration_z = 0
    for i in calibrate:
        if os.path.isfile(i) and i not in calibration_items:
            calibration_items.append(i)
        for dirpath, dirnames, filenames in os.walk(i):
            for filename in [f for f in filenames if f.endswith(".json")]:
                if os.path.join(dirpath, filename) not in calibration_items: # dirpath.split('/')[-1] in dataset_labels
                    calibration_items.append(os.path.join(dirpath, filename))
    for i in calibration_items:
        with open(i, 'r') as json_file:
            history = json.load(json_file)
            calibration_x += statistics.median(history['data']['x'])
            calibration_y += statistics.median(history['data']['y'])
            calibration_z += statistics.median(history['data']['z'])
    if calibration_items:
        calibration_x = calibration_x / len(calibration_items)
        calibration_y = calibration_y / len(calibration_items)
        calibration_z = calibration_z / len(calibration_items)

    calibration_matrix = rotation_matrix_from_vectors([0, 0 , 1], [calibration_x, calibration_y, calibration_z])



data_items = []
data_class = []
data_files = []

for dataset in dataset_path:
    for dirpath, dirnames, filenames in os.walk(dataset):
        for filename in [f for f in filenames if f.endswith(".json")]:
            if dirpath.split('/')[-1] in dataset_labels and filename not in data_files:
                data_items.append(os.path.join(dirpath, filename))
                data_class.append(dirpath.split('/')[-1])
                data_files.append(filename)

# print(data_items)
# print(data_class)
# print(data_files)
# # exit()

X = []
Y = []
C = []
R = []

X_train = []
M_train = []
Y_train = []
C_train = []
R_train = []

X_valid = []
M_valid = []
Y_valid = []
C_valid = []
R_valid = []



if separate_validation:
    if random_seed == None:
        random_seed = 0
        balanced = False
        while not balanced:
            np.random.seed(random_seed)

            temp_list = list(zip(data_items, data_class, data_files))
            np.random.shuffle(temp_list)
            temp_data_items, temp_data_class, temp_data_files = zip(*temp_list)

            temp_data_files_train = temp_data_files[:round(np.size(temp_data_files, 0) * dataset_split)]
            temp_data_class_train = temp_data_class[:round(np.size(temp_data_class, 0) * dataset_split)]

            temp_list = []

            for l in dataset_labels:
                temp_list.append(temp_data_class_train.count(l))

            if (max(temp_list) - min(temp_list)) <= 1: # and random_seed not in [2, 5]: # and temp_list[-1] == min(temp_list)
                if 'SQ' in dataset_labels and 'G' in dataset_labels and False: # specific rule, removeme
                    if temp_list[0] == max(temp_list) and temp_list[-1] == max(temp_list):
                        balanced = True
                else:
                    balanced = True
            else:
                random_seed += 1
    else:
        np.random.seed(random_seed)

        temp_list = list(zip(data_items, data_class, data_files))
        np.random.shuffle(temp_list)
        temp_data_items, temp_data_class, temp_data_files = zip(*temp_list)

        temp_data_files_train = temp_data_files[:round(np.size(temp_data_files, 0) * dataset_split)]
        temp_data_class_train = temp_data_class[:round(np.size(temp_data_class, 0) * dataset_split)]

    data_files_train = temp_data_files_train
    data_class_train = temp_data_class_train

else:
    if random_seed == None:
        random_seed = 0
        np.random.seed(random_seed)
    X = []
    Y = []
    C = []
    R = []



printProgressBar(0, len(data_items), prefix = 'Dataset building:', suffix = '', length = 50)
for i, (item, file) in enumerate(zip(data_items, data_files)):
    with open(item, 'r') as json_file:
        history = json.load(json_file)

    for aug_rsize in augmentation_rsize:
        frame_len = int(frame_len_sec * (history['frequency'] * (1 + (aug_rsize / downscaling))))
        # print('fram_len')
        # print(frame_len_sec)
        # print(history['frequency'])
        # print(aug_rsize)
        # print(downscaling)
        # print((1 + (aug_rsize / downscaling)))
        # exit()
        frame_shift = int(frame_shift_sec * (history['frequency'] * (1 + (aug_rsize / downscaling))))
        for aug_shift in augmentation_shift:
            if 'events' in history and True:
                if history['class'] == 'SQ' and True:
                    temp_list = []
                    for j in history['events']:
                        for k in range(-50,51, 10):
                            if k:
                                temp_list.append(j + k)
                    history['events'] += temp_list
                if history['class'] == 'P' and True:
                    temp_list = []
                    for j in history['events']:
                        for k in range(-50,51, 10):
                            if k:
                                temp_list.append(j + k)
            if 'events' in history:
                frames = [h - int(frame_len / 2) for h in history['events']]
                temp_class = [f"{history['class']}"] * len(frames)

                if history['class'] == 'SQ':
                    temp_events = []
                else:
                    temp_events = [int((a + b) / 2) for a, b in zip(history['events'], history['events'][1:])]

                temp_newclass = 'G' # f"{history['class']}_R"

                if temp_newclass not in dataset_labels:
                    dataset_labels.append(temp_newclass)
                frames += [h - int(frame_len / 2) for h in temp_events]
                temp_class += [temp_newclass] * len(temp_events)

                if history['class'] == 'P' and True: # FIXME
                    frames += [h - int(frame_len / 2) for h in temp_list]
                    temp_class += [f"{history['class']}"] * len(temp_list)
            else:
                frames = list(range(0, history['samples'] - frame_len + 1, frame_shift))
            for j, frame in enumerate(frames):
                if frame + aug_shift >= 0 and frame + frame_shift + aug_shift <= history['samples']: # and sym[i] in sub_labels:
                    temp_X = [[history['data']['x'][frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize]], [history['data']['y'][frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize]], [history['data']['z'][frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize]]]
                    if median > 1:
                        temp_X_median = []
                        temp_Y_median = []
                        temp_Z_median = []
                        [
                            (
                                temp_X_median.append(statistics.median(temp_X[0][0][t : t + median])),
                                temp_Y_median.append(statistics.median(temp_X[1][0][t : t + median])),
                                temp_Z_median.append(statistics.median(temp_X[2][0][t : t + median]))
                            )
                            for t in range(len(temp_X[0][0]) - median + 1)
                        ]
                        temp_X = [[temp_X_median], [temp_Y_median], [temp_Z_median]]
                    if 'events' in history:
                        temp_Y = dataset_labels.index(temp_class[j])
                    else:
                        temp_Y = dataset_labels.index(history['class'])
                    temp_C = True if aug_shift == 0 else False
                    temp_R = data_files.index(file)

                    if len(temp_X[0][0]) != int((frame_len + downscaling - 1) / downscaling):
                        # print('')
                        # print(len(temp_X[0][0]))
                        # print(temp_X)
                        # print(temp_Y)
                        # print(temp_C)
                        # print(temp_R)
                        continue

                    if calibrate:
                        cal_X = []
                        cal_Y = []
                        cal_Z = []

                        for x_i, y_i, z_i in zip(temp_X[0][0], temp_X[1][0], temp_X[2][0]):
                            temp = np.matmul(np.array([x_i, y_i, z_i]), calibration_matrix)
                            cal_X.append(temp[0])
                            cal_Y.append(temp[1])
                            if gforce:
                                cal_Z.append(temp[2] - 981)
                            else:
                                cal_Z.append(temp[2])
                        temp_X = [[cal_X], [cal_Y], [cal_Z]]

                    if separate_validation:
                        if file in data_files_train:
                            X_train.append(temp_X)
                            Y_train.append(temp_Y)
                            C_train.append(temp_C)
                            R_train.append(temp_R)
                        else:
                            if not aug_rsize and not aug_shift: # and not aug_rotat:
                                X_valid.append(temp_X)
                                Y_valid.append(temp_Y)
                                C_valid.append(temp_C)
                                R_valid.append(temp_R)
                    else:
                        X.append(temp_X)
                        Y.append(temp_Y)
                        C.append(temp_C)
                        R.append(temp_R)
                    # else:
                    #     if random.random() < dataset_split:
                    #         X_train.append(temp_X)
                    #         Y_train.append(temp_Y)
                    #         C_train.append(temp_C)
                    #         R_train.append(temp_R)
                    #     else:
                    #         if not aug_rsize and not aug_shift: # and not aug_rotat:
                    #             X_valid.append(temp_X)
                    #             Y_valid.append(temp_Y)
                    #             C_valid.append(temp_C)
                    #             R_valid.append(temp_R)

    printProgressBar(i + 1, len(data_items), prefix = 'Dataset building:', suffix = '', length = 50)

if not separate_validation:
    temp_list = list(zip(X, Y, C, R))
    np.random.shuffle(temp_list)
    X, Y, C, R = zip(*temp_list)

    X_train = X[ : round(np.size(X, 0) * dataset_split)]
    Y_train = Y[ : round(np.size(Y, 0) * dataset_split)]
    C_train = C[ : round(np.size(C, 0) * dataset_split)]
    R_train = R[ : round(np.size(R, 0) * dataset_split)]

    X_valid = X[round(np.size(X, 0) * dataset_split) : ]
    Y_valid = Y[round(np.size(Y, 0) * dataset_split) : ]
    C_valid = C[round(np.size(C, 0) * dataset_split) : ]
    R_valid = R[round(np.size(R, 0) * dataset_split) : ]



t_done = False
t_dict = {'prefix' : f'{color.NONE}Data loader{color.END}: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()

item_perm_train = np.arange(np.size(X_train,0))
# item_perm_valid = np.arange(np.size(X_valid,0))

np.random.shuffle(item_perm_train)
# np.random.shuffle(item_perm_valid)

X_train = np.array(X_train)[item_perm_train]
Y_train = np.array(Y_train)[item_perm_train]
C_train = np.array(C_train)[item_perm_train]
R_train = np.array(R_train)[item_perm_train]

X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)
R_valid = np.array(R_valid)






if normalization:
    for i in range(np.size(X_train,0)):
        X_train[i]=X_train[i]/np.max(np.absolute(X_train[i]))
    for i in range(np.size(X_valid,0)):
        X_valid[i]=X_valid[i]/np.max(np.absolute(X_valid[i]))






X_train = torch.from_numpy(X_train)
X_valid = torch.from_numpy(X_valid)
Y_train = torch.from_numpy(Y_train)
Y_valid = torch.from_numpy(Y_valid)



t_dataset_train = torch.utils.data.TensorDataset(X_train,Y_train)
t_dataset_valid = torch.utils.data.TensorDataset(X_valid,Y_valid)

loader_train = torch.utils.data.DataLoader(t_dataset_train, batch_size=value_batch_size, shuffle=False)
loader_valid = torch.utils.data.DataLoader(t_dataset_valid, batch_size=value_batch_size, shuffle=False)

t_done = True
time.sleep(0.2)
print('\n\n')

# exit()









#%%

#███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
#████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
#██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝
#██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗
#██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
#╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

conv_indim = json_train['conv_indim']

pool_ks = json_train['pool_ks']

conv_1_if = json_train['conv_1_if']
conv_1_of = json_train['conv_1_of']
conv_1_ks = json_train['conv_1_ks']

conv_2_if = json_train['conv_2_if']
conv_2_of = json_train['conv_2_of']
conv_2_ks = json_train['conv_2_ks']

fully_1_indim = json_train['fully_1_indim']
fully_1_outdim = json_train['fully_1_outdim']

fully_2_indim = json_train['fully_2_indim']
fully_2_outdim = json_train['fully_2_outdim']



class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()

        # self.relu = False
        self.debug = False
        self.quantization = False
        self.quantization_inf = False
        self.temp = 0

        self.minoutput_0 = 0
        self.maxoutput_0 = 0

        self.conv1 = nn.Conv2d(conv_1_if, conv_1_of, (1, conv_1_ks), bias=False)
        self.conv2 = nn.Conv2d(conv_2_if, conv_2_of, (1, conv_2_ks), bias=False)

        self.pool = nn.MaxPool2d((1, pool_ks))

        self.fc1 = nn.Linear(fully_1_indim, fully_1_outdim, bias=False)
        self.fc2 = nn.Linear(fully_2_indim, fully_2_outdim, bias=False)

        self.sm = nn.Softmax(dim=-1)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):

        if(self.debug):
            torch.set_printoptions(threshold=500000, precision=10) #, linehalf_windowth=20
            f = open(session_path+"inference_data_example/input_"+str(self.temp)+".txt", "w")

            f.write("\n\ndequant\n")
            f.write(str(x))

        if(self.quantization):
            x = self.quant(x)

        if(self.debug):
            f.write("\n\nquant\n")
            f.write(str(x))

        x = self.conv1(x)

        if(self.quantization_inf):
            if(torch.min(x)<self.minoutput_0):
                self.minoutput_0 = torch.min(x)
            if(torch.max(x)>self.maxoutput_0):
                self.maxoutput_0 = torch.max(x)

        if(self.debug):
            f.write("\n\nconv1\n")
            f.write(str(x))

        x = F.relu(x)

        if(self.debug):
            f.write("\n\nrelu1\n")
            f.write(str(x))

        x = self.pool(x)

        if(self.debug):
            f.write("\n\npool1\n")
            f.write(str(x))



        x = self.conv2(x)

        if(self.debug):
            f.write("\n\nconv2\n")
            f.write(str(x))

        x = F.relu(x)

        if(self.debug):
            f.write("\n\nrelu2\n")
            f.write(str(x))

        x = self.pool(x)

        if(self.debug):
            f.write("\n\npool2\n")
            f.write(str(x))


        x = x.flatten(1)

        if(self.debug):
            f.write("\n\nflatten\n")
            f.write(str(x))


        x=self.fc1(x)

        if(self.debug):
            f.write("\n\nfc1\n")
            f.write(str(x))

        x = F.relu(x)

        if(self.debug):
            f.write("\n\nrelu3\n")
            f.write(str(x))


        x = self.fc2(x)

        if(self.debug):
            f.write("\n\nfc2\n")
            f.write(str(x))


        if(self.quantization):
            x = self.dequant(x)

        if(self.debug):
            f.write("\n\ndequant\n\n")
            f.write(str(x))
            f.close()


#        x = self.sm(x)

        return x

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)









#%%

# ██╗      ██████╗  █████╗ ██████╗     ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗
# ██║     ██╔═══██╗██╔══██╗██╔══██╗    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║
# ██║     ██║   ██║███████║██║  ██║    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║
# ██║     ██║   ██║██╔══██║██║  ██║    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║
# ███████╗╚██████╔╝██║  ██║██████╔╝    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
# ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝

model = Net()
model.load_state_dict(torch.load(session_train+'/model.pth'))
model.eval()

model_quantized = copy.deepcopy(model)
model_quantized.quantization = True

eval_batch_size = value_batch_size
num_calibration_batches = math.ceil(X_train.size(0)/eval_batch_size)
num_eval_batches = math.ceil(X_train.size(0)/eval_batch_size)



print('\nPost-training quantization')
model_quantized.eval()
model_quantized.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.backends.quantized.engine = 'qnnpack'
torch.quantization.prepare(model_quantized, inplace=True)

evaluate(model_quantized, criterion, loader_train, neval_batches=num_calibration_batches)

t_done = False
t_dict = {'prefix' : 'Covert: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()
torch.quantization.convert(model_quantized, inplace=True)
t_done = True
time.sleep(0.1)
print('\n\n')









#%%
#███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
#██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
#█████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
#██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
#╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

data_items = []
class_dettail = {
    'G' : {
        'color' : color.BLUE,
        'description' : 'Garbage'
    },
    'SQ' : {
        'color' : color.RED,
        'description' : 'Squat'
    },
    'SQ_R' : {
        'color' : color.END,
        'description' : 'Squat'
    },
    'P' : {
        'color' : color.GREEN,
        'description' : 'Pushup'
    },
    'P_R' : {
        'color' : color.END,
        'description' : 'Pushup'
    },
    'T' : {
        'color' : color.END,
        'description' : 'Test mode'
    },
}

# class_dettail = {
#     'G' : {
#         'color' : color.BLUE,
#         'description' : 'Garbage'
#     },
#     'GD' : {
#         'color' : color.RED,
#         'description' : 'Dritto'
#     },
#     'GR' : {
#         'color' : color.END,
#         'description' : 'Rovescio'
#     },
#     'T' : {
#         'color' : color.END,
#         'description' : 'Test mode'
#     }
# }

for i in session_input:
    if os.path.isfile(i) and i not in data_items:
        data_items.append(i)
    for dirpath, dirnames, filenames in os.walk(i):
        for filename in [f for f in filenames if f.endswith(".json")]:
            if os.path.join(dirpath, filename) not in data_items: # dirpath.split('/')[-1] in dataset_labels
                data_items.append(os.path.join(dirpath, filename))

if session_csort:
    data_items.sort(key=natural_sort_key)

else:
    data_files = [os.path.basename(i) for i in data_items]
    data_files.sort(key=natural_sort_key)
    data_items_sorted = []
    [[data_items_sorted.append(i) for i in data_items if f in i] for f in data_files] # [data_items_sorted.append(i) for i in data_items if f in i]
    data_items = data_items_sorted

print(color.BOLD + color.UNDERLINE + 'Class\r\tDescription' + color.END)
for l in dataset_labels:
    print(class_dettail[l]['color'] + l + '\r\t' + class_dettail[l]['description'] + color.END)
print(class_dettail['T']['color'] + 'T' + '\r\t' + class_dettail['T']['description'] + color.END)
print('')

events = []
linelengths = []
colorCodes = []
matplotlib.rcParams['font.size'] = 8.0
colorCodes_dic = {
    'G' : [0, 0, 0],
    'SQ' : [1, 0 , 0],
    'P' : [0, 0, 1]
}

events_dic = {
    'G' : {
        'x' : [],
        'y' : [],
    },
    'SQ' : {
        'x' : [],
        'y' : [],
    },
    'P' : {
        'x' : [],
        'y' : [],
    },
}

color_labels = ['#D0D0D0', 'orange', 'green']

# %%

print(color.BOLD + color.UNDERLINE + 'File name\r\t\t\t\t\t\tClass\r\t\t\t\t\t\t\tClassification' + color.END)
for i, item in enumerate(data_items):
    with open(item, 'r') as json_file:
        history = json.load(json_file)

    frame_len = int(frame_len_sec * history['frequency'])
    frame_shift_sec = 1 #/ 4
    frame_shift = int(frame_shift_sec * (history['frequency'] * (1 + (aug_rsize / downscaling))))

    X = []
    Y = []
    for frame in range(0, history['samples'] - frame_len + 1, frame_shift):
        X_temp = [[history['data'][0]['x'][frame : frame + frame_len : downscaling]], [history['data'][0]['y'][frame : frame + frame_len : downscaling]], [history['data'][0]['z'][frame : frame + frame_len : downscaling]]]
        if median > 1:
            temp_X_median = []
            temp_Y_median = []
            [
                (
                    temp_X_median.append(statistics.median(temp_X[0][0][t : t + median])),
                    temp_Y_median.append(statistics.median(temp_X[1][0][t : t + median])),
                    temp_Z_median.append(statistics.median(temp_X[2][0][t : t + median]))
                )
                for t in range(len(temp_X[0][0]) - median + 1)
            ]
            temp_X = [[temp_X_median], [temp_Y_median], [temp_Z_median]]
        # if frame + aug_shift >= 0 and frame + frame_shift + aug_shift <= history['samples']: # and sym[i] in sub_labels:
        X.append(X_temp)
        # Y.append(dataset_labels.index(history['class']))
        Y.append(0)

        # print(len(X_temp[0][0]))

    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))

    dataset = torch.utils.data.TensorDataset(X,Y)
    # loader = torch.utils.data.DataLoader(dataset, batch_size = value_batch_size, shuffle=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size = X.size(0), shuffle=False)

    # fig = plt.figure(figsize=(7,2)) #figsize=(12,8)) #8x11

    for data in loader:
        inputs, labels = data
        outputs = model(inputs.float()) # _quantized FIXME!
        outputs = [list(o).index(max(o)) for o in outputs]

        events.append([i for i, o in enumerate(outputs) if o])
        linelengths.append(0.8)
        colorCodes.append(colorCodes_dic[history["class"]])

        for j, o in enumerate(outputs):
            events_dic[dataset_labels[o]]['y'].append(i)
            events_dic[dataset_labels[o]]['x'].append(j)
        
        for c, d in zip(color_labels, dataset_labels):
            plt.scatter(events_dic[d]['x'], events_dic[d]['y'], marker="s", c=c, s = 20) #, s = 1

        if i == 2:
            events.append([])
            linelengths.append(0.8)
            colorCodes.append(colorCodes_dic[history["class"]])

        if history['class'] in dataset_labels:
            class_color = class_dettail[history['class']]['color']
        else:
            class_color = class_dettail['T']['color']

        print(f'{item}\r\t\t\t\t\t\t{color.BOLD}{class_color}{history["class"]}{color.END}\r\t\t\t\t\t\t\t', end = '')
        [print(f'{class_dettail[dataset_labels[o]]["color"]}▮{color.END}', end = '') for o in outputs]
        print('')

my_dict = {
    'model' : session_train.split("/")[-1],
    'shift' : frame_shift_sec,
    'dict' : events_dic
}
with open(f'{session_train.split("/")[-1]}_{frame_shift_sec}.json', 'w') as f:
    json.dump(my_dict, f, indent = 4)

#%%


# colorCodes = np.array([[1, 0, 0],
#                         [1, 0, 0],
#                         [1, 0, 0],
#                         [1, 1, 0],
#                         [0, 0, 1],
#                         [0, 0, 1],
#                         [0, 0, 1]])
                        
# plt.eventplot(events, linelengths=linelengths, color=colorCodes, linewidths=10)         #, color=colorCodes 
# plt.xlabel("Time interval [s]",fontsize=14)
# plt.ylabel("Exercise",fontsize=14)
# plt.yticks([])
# # plt.show()
# plt.savefig("mygraph.png")
# tikzplotlib.save("mygraph.tex")

plt.xlabel("Time [s]") # ,fontsize=14
plt.ylabel("Session id") # ,fontsize=14
# plt.yticks(['Squat', 'Squat', 'Squat', 'Push-up', 'Push-up', 'Push-up'])

labels = ["Other", "Squat", "Push-up"]
plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.25))
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)

# plt.savefig("output/plot/raster.png")
# plt.savefig("output/plot/raster.pdf", format="pdf")
# tikzplotlib.save("output/plot/raster.tex")

plt.show()

print('')

# %%

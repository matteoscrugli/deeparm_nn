#from turtle import down
import torch
import torchvision
import torch.quantization
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle as pk
# import pandas as pd
# import wfdb
# import pywt
#import h5py
import math
import statistics
import os, os.path
import sys
import argparse
from pathlib import Path
import shutil
import copy
import time
import json
import random

import itertools
import threading

from torch.quantization import QuantStub, DeQuantStub
from pathlib import Path
from torch.utils import data

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

parser = argparse.ArgumentParser()

parser.add_argument('-n','--name', dest='name', required=True, help="session name")
parser.add_argument('-e','--epoch', dest='epoch', required=True, type=int, help="number of epochs")
parser.add_argument('-d','--dataset', dest='dataset', required=True, nargs='*', help="dataset path")
parser.add_argument('-c','--classes', dest='classes', nargs='*', default=['G', 'SQ', 'P'], help="classes to train")
parser.add_argument('-s','--split', dest='split', nargs='*', default=['0.7', '0.15', '0.15'], help="choice of dataset splitting")
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")
parser.add_argument('-b','--batchsize', dest='batchsize', default=32, type=int, help="batchsize value")
# parser.add_argument('-a','--augmentation', dest='augmentation', nargs=2, type=int, default=[0,1], help='augmentation, number of lateral shifts and pitch (two arguments)')
parser.add_argument('-r','--randomseed', dest='randomseed', type=int, default=None, help='random seed for dataset randomization')
# parser.add_argument('-p','-.peak', dest='peak', help='peak detector path')

parser.add_argument('-v','--separate_validation', dest='separate_validation', action='store_true', help="separate files for validation")

parser.add_argument('-C','--calibrate', dest='calibrate', nargs='*', help='calibration file')
parser.add_argument('-G','--gforce', dest='gforce', default='', action='store_true', help='gforce remover')

parser.add_argument('--flen', dest='flen', default=3.8, type=float, help="frame lenght in seconds")
parser.add_argument('--fshift', dest='fshift', default=2.25, type=float, help="frame shift in seconds")
parser.add_argument('--dscaling', dest='dscaling', type=int, default=5, help='random seed for dataset randomization')
parser.add_argument('--median', dest='median', type=int, default=1, help='median value')

parser.add_argument('--augrsize', dest='augrsize', type=int, nargs=3, default=[0, 0, 1], help='zoom/dezoom augmentation')
parser.add_argument('--augshift', dest='augshift', type=int, nargs=3, default=[0, 0, 1], help='shift augmentation')
parser.add_argument('--augrotat', dest='augrotat', type=int, nargs=3, default=[0, 0, 1], help='rotation augmentation')

parser.add_argument('--norm', dest='normalization', action='store_true', help="during training, scales all inputs so that its absolute value is equal to 1")
# parser.add_argument('--indim', dest='indimension', default=2000, type=int, help="input dimension")
parser.add_argument('--ksize', dest='ksize', default=5, type=int, help="kernel size")
parser.add_argument('--conv1of', dest='conv1of', default=20, type=int, help="conv 1 output features value")
parser.add_argument('--conv2of', dest='conv2of', default=20, type=int, help="conv 2 output features value")
parser.add_argument('--foutdim', dest='foutdim', default=100, type=int, help="fully connected 1 output dimension")

args = parser.parse_args()



session_name = args.name
session_path = "output/train/"+session_name+"/"
if os.path.isdir(session_path):
    if args.overwrite:
        try:
            shutil.rmtree(session_path)
            Path(session_path).mkdir(parents=True, exist_ok=True)
            Path(session_path+'inference_data_example').mkdir(parents=True, exist_ok=True)
            Path(session_path+'parameters').mkdir(parents=True, exist_ok=True)
        except OSError:
            print("Error in session creation ("+session_path+").")
            exit()
    else:
        print(f'Session path ({session_path}) already exists')
        exit()
else:
    try:
        Path(session_path).mkdir(parents=True, exist_ok=True)
        Path(session_path+'inference_data_example').mkdir(parents=True, exist_ok=True)
        Path(session_path+'parameters').mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Error in session creation ("+session_path+").")
        exit()

print(f'{color.BOLD}Starting {color.NONE}training{color.END}{color.BOLD} session \'{session_name}\'\n\n\n{color.END}')









#██╗   ██╗ █████╗ ██████╗ ██╗ ██████╗ ██╗   ██╗███████╗
#██║   ██║██╔══██╗██╔══██╗██║██╔═══██╗██║   ██║██╔════╝
#██║   ██║███████║██████╔╝██║██║   ██║██║   ██║███████╗
#╚██╗ ██╔╝██╔══██║██╔══██╗██║██║   ██║██║   ██║╚════██║
# ╚████╔╝ ██║  ██║██║  ██║██║╚██████╔╝╚██████╔╝███████║
#  ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝  ╚═════╝ ╚══════╝

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_size_of_model(model):
    torch.save(model.state_dict(), session_path+"temp.p")
    print('Size model (MB):', os.path.getsize(session_path+"temp.p")/1e6)
    os.remove('temp.p')

def save_model(model, n):
    torch.save(model.state_dict(), session_path+"model"+n+".pth")

t_done = False
def animate(prefix = ''):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if t_done:
            break
        print('\r' + prefix + c, end = '\r')
        # sys.stdout.write('\r' + prefix + c)
        # sys.stdout.flush()
        time.sleep(0.2)
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
        print(f'\r{prefix} |{bar}| Done! {suffix}')

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









#██████╗  █████╗ ████████╗ █████╗ ███████╗███████╗████████╗
#██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝
#██║  ██║███████║   ██║   ███████║███████╗█████╗     ██║
#██║  ██║██╔══██║   ██║   ██╔══██║╚════██║██╔══╝     ██║
#██████╔╝██║  ██║   ██║   ██║  ██║███████║███████╗   ██║
#╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝

dataset_path = args.dataset
dataset_split = [float(s) for s in args.split]
value_batch_size = args.batchsize
normalization = args.normalization
random_seed = args.randomseed
dataset_labels = args.classes



# dataset_labels = ['B', 'FB', 'S', 'R', 'G']
# files_split = True

frame_len_sec = args.flen # 15
frame_shift_sec = args.fshift # 0.25
downscaling = args.dscaling # 7
median = args.median # 1
calibrate = args.calibrate
gforce = args.gforce
augmentation_rsize = range(args.augrsize[0], args.augrsize[1] + 1, args.augrsize[2]) # [-1, 0, 1]
augmentation_shift = range(args.augshift[0], args.augshift[1] + 1, args.augshift[2]) # [0] # args.augmentation # augmentation_shift = range(-args.augmentation[0] * args.augmentation[1], args.augmentation[0] * args.augmentation[1]+1, args.augmentation[1]) # args.augmentation
augmentation_rotat = range(args.augrotat[0], args.augrotat[1] + 1, args.augrotat[2]) # [-4, 0, 4]



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



# if args.separate_validation:
#     if random_seed == None:
#         random_seed = 0
#         balanced = False
#         while not balanced:
#             np.random.seed(random_seed)

#             temp_list = list(zip(data_items, data_class, data_files))
#             np.random.shuffle(temp_list)
#             temp_data_items, temp_data_class, temp_data_files = zip(*temp_list)

#             temp_data_files_train = temp_data_files[:round(np.size(temp_data_files, 0) * dataset_split)]
#             temp_data_class_train = temp_data_class[:round(np.size(temp_data_class, 0) * dataset_split)]

#             temp_list = []

#             for l in dataset_labels:
#                 temp_list.append(temp_data_class_train.count(l))

#             if (max(temp_list) - min(temp_list)) <= 1: # and random_seed not in [2, 5]: # and temp_list[-1] == min(temp_list)
#                 if 'SQ' in dataset_labels and 'G' in dataset_labels and False: # specific rule, removeme
#                     if temp_list[0] == max(temp_list) and temp_list[-1] == max(temp_list):
#                         balanced = True
#                 else:
#                     balanced = True
#             else:
#                 random_seed += 1
#     else:
#         np.random.seed(random_seed)

#         temp_list = list(zip(data_items, data_class, data_files))
#         np.random.shuffle(temp_list)
#         temp_data_items, temp_data_class, temp_data_files = zip(*temp_list)

#         temp_data_files_train = temp_data_files[:round(np.size(temp_data_files, 0) * dataset_split)]
#         temp_data_class_train = temp_data_class[:round(np.size(temp_data_class, 0) * dataset_split)]

#     data_files_train = temp_data_files_train
#     data_class_train = temp_data_class_train

# else:
if True:
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
                        for k in range(-20,21, 4):
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

                    # if args.separate_validation:
                    #     if file in data_files_train:
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
                    # else:
                    if True:
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

if not args.separate_validation:
    temp_list = list(zip(X, Y, C, R))
    np.random.shuffle(temp_list)
    X, Y, C, R = zip(*temp_list)

    X_train = X[ : round(np.size(X, 0) * dataset_split[0])]
    Y_train = Y[ : round(np.size(Y, 0) * dataset_split[0])]
    C_train = C[ : round(np.size(C, 0) * dataset_split[0])]
    R_train = R[ : round(np.size(R, 0) * dataset_split[0])]

    # X_valid = X[round(np.size(X, 0) * dataset_split[0]) : ]
    # Y_valid = Y[round(np.size(Y, 0) * dataset_split[0]) : ]
    # C_valid = C[round(np.size(C, 0) * dataset_split[0]) : ]
    # R_valid = R[round(np.size(R, 0) * dataset_split[0]) : ]

    X_valid = X[round(np.size(X, 0) * dataset_split[0]) : round(np.size(X, 0) * dataset_split[0] + np.size(X, 0) * dataset_split[1])]
    Y_valid = Y[round(np.size(Y, 0) * dataset_split[0]) : round(np.size(X, 0) * dataset_split[0] + np.size(X, 0) * dataset_split[1])]
    C_valid = C[round(np.size(C, 0) * dataset_split[0]) : round(np.size(X, 0) * dataset_split[0] + np.size(X, 0) * dataset_split[1])]
    R_valid = R[round(np.size(R, 0) * dataset_split[0]) : round(np.size(X, 0) * dataset_split[0] + np.size(X, 0) * dataset_split[1])]

    X_test = X[round(np.size(X, 0) * dataset_split[0] + np.size(X, 0) * dataset_split[1]) : ]
    Y_test = Y[round(np.size(Y, 0) * dataset_split[0] + np.size(X, 0) * dataset_split[1]) : ]
    C_test = C[round(np.size(C, 0) * dataset_split[0] + np.size(X, 0) * dataset_split[1]) : ]
    R_test = R[round(np.size(R, 0) * dataset_split[0] + np.size(X, 0) * dataset_split[1]) : ]



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

X_test = np.array(X_test)
Y_test = np.array(Y_test)
R_test = np.array(R_test)






if normalization:
    for i in range(np.size(X_train,0)):
        X_train[i]=X_train[i]/np.max(np.absolute(X_train[i]))
    for i in range(np.size(X_valid,0)):
        X_valid[i]=X_valid[i]/np.max(np.absolute(X_valid[i]))
    for i in range(np.size(X_test,0)):
        X_test[i]=X_test[i]/np.max(np.absolute(X_test[i]))






X_train = torch.from_numpy(X_train)
X_valid = torch.from_numpy(X_valid)
X_test = torch.from_numpy(X_test)
Y_train = torch.from_numpy(Y_train)
Y_valid = torch.from_numpy(Y_valid)
Y_test = torch.from_numpy(Y_test)

X_train = X_train.type(torch.LongTensor)
X_valid = X_valid.type(torch.LongTensor)
X_test = X_test.type(torch.LongTensor)
Y_train = Y_train.type(torch.LongTensor)
Y_valid = Y_valid.type(torch.LongTensor)
Y_test = Y_test.type(torch.LongTensor)



t_dataset_train = torch.utils.data.TensorDataset(X_train,Y_train)
t_dataset_valid = torch.utils.data.TensorDataset(X_valid,Y_valid)
t_dataset_test = torch.utils.data.TensorDataset(X_test,Y_test)

loader_train = torch.utils.data.DataLoader(t_dataset_train, batch_size=value_batch_size, shuffle=False)
loader_valid = torch.utils.data.DataLoader(t_dataset_valid, batch_size=value_batch_size, shuffle=False)
loader_test = torch.utils.data.DataLoader(t_dataset_test, batch_size=value_batch_size, shuffle=False)

t_done = True
time.sleep(0.2)
print('\n\n')

# exit()









#███╗   ██╗███████╗████████╗██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗
#████╗  ██║██╔════╝╚══██╔══╝██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝
#██╔██╗ ██║█████╗     ██║   ██║ █╗ ██║██║   ██║██████╔╝█████╔╝
#██║╚██╗██║██╔══╝     ██║   ██║███╗██║██║   ██║██╔══██╗██╔═██╗
#██║ ╚████║███████╗   ██║   ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗
#╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝

# conv_indim = args.indimension
conv_indim = len(X_train[0][0][0])
# print(conv_indim)
# exit()

pool_ks = 2

conv_1_if = 3 # 1
conv_1_of = args.conv1of
conv_1_ks = args.ksize

conv_2_if = conv_1_of
conv_2_of = args.conv2of
conv_2_ks = args.ksize

fully_1_indim = int(conv_2_of * int((((conv_indim - (conv_1_ks - 1)) / pool_ks) - (conv_2_ks - 1)) / pool_ks))
fully_1_outdim = args.foutdim

fully_2_indim = fully_1_outdim
fully_2_outdim = len(dataset_labels)

fully_ann1_indim = len(X_train[0][0][0])*3
fully_ann1_outdim = 150

fully_ann2_indim = fully_1_outdim
fully_ann2_outdim = len(dataset_labels)



class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()

        # self.relu6 = False
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

        # self.fc1ann = nn.Linear(fully_ann1_indim, fully_1_outdim, bias=False)
        # self.fc2ann = nn.Linear(fully_ann2_indim, fully_2_outdim, bias=False)

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



        # x = x.flatten(1)
        # x = self.fc1ann(x)
        # x = F.relu(x)
        # x = self.fc2ann(x)



        # x = self.sm(x)

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









#████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
#╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
#   ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#   ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
#   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝

num_trainepoch = args.epoch
num_trainepoch_effective = 0
dim_batches = 25



model = Net()

# optimizer = optim.Adam(model.parameters(), lr=0.005)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
# optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

criterion = nn.CrossEntropyLoss()



max_i = []

train_data = [[] for i in range(5)]
train_dic = {
    'train_loss' : 0,
    'valid_loss' : 1,
    'train_acc' : 2,
    'valid_acc' : 3,
    'learing_rate' : 4
}

epoch_loss = 0
epoch_loss_best = 0
epoch_acc = 0
epoch_acc_best = 0
epoch_best = 0
cnt_allbatches = 0
tmp_cnt = 0
tmp_cnt_t = 0
frac = 33



newline = 0

print('\n\n\n\n\n', end = '')
printProgressBar(cnt_allbatches, len(loader_train)/dim_batches * num_trainepoch, prefix = f'{color.NONE}Training:{color.END}', suffix = '', length = 55)
newline += 5


# print('\033[F\033[F\033[F\033[F\033[F', end = '')
try:
    for epoch in range(num_trainepoch):  # loop over the dataset multiple times

        cnt_batches = 0
        cnt = 0
        cnt_t = 0
        epoch_loss = 0

        running_loss = 0.0

        while newline:
            print('', end = '\033[F')
            newline -= 1

        # printProgressBar(0, len(loader_train), prefix = 'Epoch ' + str(epoch + 1) + '/' + str(num_trainepoch) + ':', suffix = '              ', length = 40)
        for i, data in enumerate(loader_train):

            inputs, labels = data

            # print(len(inputs))
            # exit()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            loss = criterion(outputs, labels)

            list.clear(max_i)
            for o in outputs:
                m=max(o)
                indx=list(o).index(m)
                max_i.append(indx)

            for o, m in zip(labels, max_i):
                if o == m:
                    cnt = cnt + 1
                cnt_t = cnt_t + 1

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i == len(loader_train) - 1:
                epoch_loss = epoch_loss/len(loader_train)

            running_loss += loss.item()
            if (i % dim_batches) == (dim_batches - 1):
                printProgressBar(i, len(loader_train) - 1, prefix = 'Session \''+session_name+'\', epoch ' + str(epoch + 1) + '/' + str(num_trainepoch) + ':', suffix = '                 ', length = 25)
                # if i == len(loader_train) - 1:
                #     print('', end='\033[F')
                    # training_loss.append(epoch_loss)
                print('\nLoss during training: %f\nAccuracy during training: %f\n\n' % (epoch_loss/i, (cnt/cnt_t)))
                printProgressBar(cnt_allbatches + i, len(loader_train) * num_trainepoch - 1, prefix = 'Training:', suffix = '', length = 55)
                # if cnt_allbatches + i == (len(loader_train) * num_trainepoch) - 1:
                #     print('', end='\033[F')
                print('\033[F\033[F\033[F\033[F\033[F\033[F')
                running_loss = 0.0
            elif i == (len(loader_train) - 1):
                # training_loss.append(epoch_loss)
                printProgressBar(i, len(loader_train) - 1, prefix = 'Session \''+session_name+'\', epoch ' + str(epoch + 1) + '/' + str(num_trainepoch) + ':', suffix = '                 ', length = 25)
                print('', end='\033[F')
                print('\nLoss during training: %f\nAccuracy during training: %f\n\n' % (epoch_loss, (cnt/cnt_t)))
                printProgressBar(cnt_allbatches + i, len(loader_train) * num_trainepoch - 1, prefix = 'Training:', suffix = '', length = 55)
                if cnt_allbatches + i == (len(loader_train) * num_trainepoch) - 1:
                    print('', end='\033[F')
                print('', end='\033[F\033[F')
                running_loss = 0.0

        cnt_allbatches = cnt_allbatches + len(loader_train)

        train_data[train_dic['learing_rate']].append(optimizer.param_groups[0]['lr'])

        train_data[train_dic['train_acc']].append(cnt/cnt_t)
        train_data[train_dic['train_loss']].append(epoch_loss)


        cnt = 0
        cnt_t = 0
        epoch_loss = 0

        # print(optimizer.param_groups[0]['lr'])

        t_done = False
        t_dict = {'prefix' : 'Accuracy on validation set: '}
        t = threading.Thread(target=animate, kwargs=t_dict)
        t.start()
        for i, data in enumerate(loader_valid):

            inputs, labels = data
            outputs = model(inputs.float())

            list.clear(max_i)
            for o in outputs:
                m=max(o)
                indx=list(o).index(m)
                max_i.append(indx)


            for o, m in zip(labels, max_i):
                if o == m:
                    cnt = cnt + 1
                cnt_t = cnt_t + 1

            epoch_loss += loss.item()
            if i == len(loader_valid) - 1:
                epoch_loss = epoch_loss/len(loader_valid)

        t_done = True
        time.sleep(0.2)

        train_data[train_dic['valid_acc']].append(cnt/cnt_t)
        train_data[train_dic['valid_loss']].append(epoch_loss)

        epoch_acc = cnt/cnt_t

        # if epoch_loss < epoch_loss_best or epoch_loss_best == 0:
        if epoch_acc > epoch_acc_best:
            model_best = copy.deepcopy(model)
            epoch_best = epoch
            epoch_loss_best = epoch_loss
            epoch_acc_best = epoch_acc

        # training_acc.append(epoch_acc)
        num_trainepoch_effective += 1
        # print('\n\n\n')
        print('\033[FAccuracy on validation set: %f\n' % epoch_acc)
        printProgressBar(cnt_allbatches, len(loader_train) * num_trainepoch, prefix = f'{color.NONE}Training{color.END}:', suffix = '', length = 55)
except KeyboardInterrupt:
    print('\n\n\n\n\n')
print('\n')

save_model(model_best,'')









# ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██╗███████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
#██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██║╚══███╔╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
#██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██║  ███╔╝ ███████║   ██║   ██║██║   ██║██╔██╗ ██║
#██║ █ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║ ███╔╝  ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ██║███████╗██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
#  ╚══█═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

# Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)



def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=7, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)



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
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
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

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end = '')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('\nLoss', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    return

def convert_state_dict(src_dict): #1
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    dst_dict = {}
    for k, v in src_dict.items():
        toks = k.split('.')
        if k.startswith('layer'):
            assert len(toks[0]) == 6
            res_id = int(toks[0][5]) + 1
            name = '.'.join(['res%d' % res_id] + toks[1:])
            dst_dict[name] = v
        elif k.startswith('fc'):
            continue
        else:
            name = '.'.join(['res1'] + toks)
            dst_dict[name] = v
    return dst_dict

def model_state_dict_parallel_convert(state_dict, mode): #2
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if mode == 'to_single':
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'to_parallel':
        for k, v in state_dict.items():
            name = 'module.' + k  # add 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'same':
        new_state_dict = state_dict
    else:
        raise Exception('mode = to_single / to_parallel')

    return new_state_dict

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor): #3
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict









# ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗    ██████╗  ██████╗ ███████╗████████╗████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
#██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
#██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       ██████╔╝██║   ██║███████╗   ██║█████╗██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#██║ █ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔═══╝ ██║   ██║╚════██║   ██║╚════╝██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ██║     ╚██████╔╝███████║   ██║      ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
# ╚═══█═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═╝      ╚═════╝ ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝

model_quantized = copy.deepcopy(model_best)
model_quantized.quantization = True

eval_batch_size = value_batch_size
num_calibration_batches = math.ceil(X_train.size(0)/eval_batch_size)
num_eval_batches = math.ceil(X_train.size(0)/eval_batch_size)



model_quantized.eval()



## Fuse Conv, bn and relu
#model_quantized.fuse_model()



# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
model_quantized.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.backends.quantized.engine = 'qnnpack'
# torch.backends.quantized.engine = 'fbgemm'
# print(model_quantized.qconfig)

torch.quantization.prepare(model_quantized, inplace=True)

print(f'\n{color.NONE}Post-training quantization{color.END}')

# Calibrate first
# print('Prepare: Inserting Observers')
#print('\n Inverted Residual Block:After observer insertion \n\n', model_quantized.features[1].conv)

# Calibrate with the training set
evaluate(model_quantized, criterion, loader_train, neval_batches=num_calibration_batches)
# print('Post Training Quantization: Calibration done')

# Convert to quantized model
t_done = False
t_dict = {'prefix' : 'Covert: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()
torch.quantization.convert(model_quantized, inplace=True)
t_done = True
time.sleep(0.2)
# print('Post Training Quantization: Convert done')

# top1, top5 = evaluate(model_quantized, criterion, loader_valid, neval_batches=num_eval_batches)
# print('\n\nEvaluation accuracy on %d samples, %.3f'%(num_eval_batches * eval_batch_size, top1.avg))

save_model(model_quantized, '_quantized')



plaincorder = True
f = open(session_path+"model_quantized.h", "w")

f.write(f"#define PYTORCH_MODEL\n")
f.write('\n')
f.write(f"#define CONV1_IN_DIM {conv_indim}\n")
f.write(f"#define CONV1_IF {conv_1_if}\n")
f.write(f"#define CONV1_OF {conv_1_of}\n")
f.write(f"#define CONV1_K_DIM {conv_1_ks}\n")
f.write('\n')
f.write(f"#define CONV2_IF {conv_2_if}\n")
f.write(f"#define CONV2_OF {conv_2_of}\n")
f.write(f"#define CONV2_K_DIM {conv_2_ks}\n")
f.write('\n')
f.write(f"#define POOL_K_DIM {pool_ks}\n")
f.write('\n')
f.write(f"#define FC1_IN_DIM {fully_1_indim}\n")
f.write(f"#define FC1_OUT_DIM {fully_1_outdim}\n")
f.write('\n')
f.write(f"#define FC2_IN_DIM {fully_2_indim}\n")
f.write(f"#define FC2_OUT_DIM {fully_2_outdim}\n")
f.write('\n')

for param_tensor in model_quantized.state_dict():
    try:
        temp_size = model_quantized.state_dict()[param_tensor].size()
    except:
        continue
    if temp_size not in [torch.Size([]), torch.Size([1])]:
        first = True
        if model_quantized.state_dict()[param_tensor].dtype in [torch.qint8, torch.quint8]:
            print('A', param_tensor)
            if ('conv' in param_tensor):
                if plaincorder:
                    temp_data = model_quantized.state_dict()[param_tensor].int_repr().numpy().flatten()
                    # print('AAAAAAAAAAAA')
                    # print(model_quantized.state_dict()[param_tensor].int_repr())
                    # print('AAAAAAAAAAAA')
                else:
                    temp_data = model_quantized.state_dict()[param_tensor].int_repr().numpy().flatten('F').reshape((temp_size[1]*temp_size[3], temp_size[0])).flatten('F')
            elif ('fc1' in param_tensor):
                temp_data = model_quantized.state_dict()[param_tensor].int_repr().numpy().flatten()
                temp_data = temp_data.reshape(fully_1_outdim,conv_2_of * int((((conv_indim - (conv_1_ks - 1)) / pool_ks) - (conv_2_ks - 1)) / pool_ks))
                for i in range(len(temp_data)):
                    temp_data[i] = temp_data[i].reshape(conv_2_of, int((((conv_indim - (conv_1_ks - 1)) / pool_ks) - (conv_2_ks - 1)) / pool_ks)).flatten('F') # 
                temp_data = temp_data.flatten()
            else:
                temp_data = model_quantized.state_dict()[param_tensor].int_repr().numpy().flatten()
            f.write(f"#define {str(param_tensor).replace('.', '_').replace('__', '_').upper()}_SCALE {model_quantized.state_dict()[param_tensor].q_scale()}\n")
            f.write(f"#define {str(param_tensor).replace('.', '_').replace('__', '_').upper()}_ZERO_POINT {model_quantized.state_dict()[param_tensor].q_zero_point()}\n")
        else:
            print('B', param_tensor)
            temp_data = model_quantized.state_dict()[param_tensor].numpy().flatten()
        f.write(f"#define {str(param_tensor).replace('.', '_').replace('__', '_').upper()}_DIM {len(temp_data)}\n")
        f.write(f"#define {str(param_tensor).replace('.', '_').replace('__', '_').upper()}" + ' {')
        for i in temp_data: #.flatten('F')
            if 'bias' in param_tensor:
                if first:
                    first = False
                    f.write(f'{int(i)}')
                else:
                    f.write(f', {int(i)}')
            else:
                if first:
                    first = False
                    f.write(f'{i}')
                else:
                    f.write(f', {i}')
        f.write('}\n')
    else:
        print('C', param_tensor)
        if model_quantized.state_dict()[param_tensor].dtype in [torch.qint8, torch.quint8]:
            temp_data = model_quantized.state_dict()[param_tensor].int_repr().numpy().flatten()[0]
            f.write(f"#define {str(param_tensor).replace('.', '_').replace('__', '_').upper()}_OUT_SCALE {model_quantized.state_dict()[param_tensor].q_scale()}\n")
            f.write(f"#define {str(param_tensor).replace('.', '_').replace('__', '_').upper()}_OUT_ZERO_POINT {model_quantized.state_dict()[param_tensor].q_zero_point()}\n")
        else:
            temp_data = model_quantized.state_dict()[param_tensor].numpy().flatten()[0]
        f.write(f"#define {str(param_tensor).replace('.', '_').replace('__', '_').upper()} {temp_data}\n")
    f.write('\n')

# print('')
# print(model_quantized.state_dict()['conv1.weight'].q_scale())
# print(model_quantized.state_dict()['conv1.scale'].numpy())
# print(model_quantized.state_dict()['quant.scale'].numpy())for param_tensor in model_quantized.state_dict():
# exit()

f = open(session_path+"model_quantized.txt", "w")
# print("Model's state_dict:")
for param_tensor in model_quantized.state_dict():
    try:
        f.write(f"{param_tensor}, {model_quantized.state_dict()[param_tensor].size()}\n")
        # print(param_tensor, ", ", model_quantized.state_dict()[param_tensor].size())
    except:
        f.write(f"{param_tensor}, Size error\n")
        # print(param_tensor, "Size error")

    f.write(str(model_quantized.state_dict()[param_tensor]))
    f.write('\n\n-----------\n\n')

    # print(model_quantized.state_dict()[param_tensor])
    # print('\n-----------\n')
f.close()

# # Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

# exit()









# ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗     █████╗ ██╗    ██╗ █████╗ ██████╗ ███████╗ ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
#██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔══██╗██║    ██║██╔══██╗██╔══██╗██╔════╝ ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
#██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       ███████║██║ █╗ ██║███████║██████╔╝█████╗█████╗██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
#██║ █ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔══██║██║███╗██║██╔══██║██╔══██╗██╔══╝╚════╝██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
#╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ██║  ██║╚███╔███╔╝██║  ██║██║  ██║███████╗    ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
# ╚═══█═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝

#model_temp = model

#num_trainepoch_quant = 8
#optimizer = torch.optim.SGD(model_temp.parameters(), lr = 0.005)
#eval_batch_size = value_batch_size
#num_train_batches = math.ceil(round(x.size(0)*value_dataset_split,0)/eval_batch_size)
#num_eval_batches = math.ceil(round(x.size(0)*(1-value_dataset_split),0)/eval_batch_size)

#model_temp.fuse_model()
#model_temp.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
#torch.quantization.prepare_qat(model_temp, inplace=True)




## Train and check accuracy after each epoch
#for nepoch in range(num_trainepoch_quant):
#    train_one_epoch(model_temp, criterion, optimizer, loader_train, torch.device('cpu'), num_train_batches)
#    if nepoch > 3:
#        # Freeze quantizer parameters
#        model_temp.apply(torch.quantization.disable_observer)
#    if nepoch > 2:
#        # Freeze batch norm mean and variance estimates
#        model_temp.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

#    # Check the accuracy after each epoch
#    model_quantized = torch.quantization.convert(model_temp.eval(), inplace=False)
#    model_quantized.eval()
#    top1, top5 = evaluate(model_quantized,criterion, loader_valid, neval_batches=num_eval_batches)
#    print('\nEpoch %d :Evaluation accuracy on %d images, %2.2f\n'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))









#███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
#██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
#█████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║██╔██╗ ██║
#██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
#███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
#╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝

model_best.quantization = False

cnt = 0
cnt_t = 0

cm_cnt = 0
cmatrix_temp = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)

# cmatrix = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)
# accmatrix = np.zeros(fully_2_outdim)

# cmatrix = np.zeros((len(data_names), fully_2_outdim, fully_2_outdim), dtype=int)
cmatrix = np.zeros((len(data_files), fully_2_outdim, fully_2_outdim), dtype=int)

# cmatrix = [[[[] for j in range(fully_2_outdim)] for k in range(fully_2_outdim)] for d in data_names]

print('\n\n')
printProgressBar(0, len(loader_test), prefix = f'{color.NONE}Floating model evaluation{color.END}:', suffix = '', length = 40)
for i, data in enumerate(loader_test):

    inputs, labels = data
    outputs = model_best(inputs.float())

    list.clear(max_i)
    for o in outputs:
        m=max(o)
        indx=list(o).index(m)
        max_i.append(indx)

    for idx, tgt in zip(max_i, labels):
        cmatrix[R_test[cm_cnt]][idx][tgt] += 1
        cm_cnt += 1


    # for o, m in zip(labels, max_i):
    #     if o == m:
    #         cnt = cnt + 1
    #     cnt_t = cnt_t + 1

    printProgressBar(i + 1, len(loader_test), prefix = f'{color.NONE}Floating model evaluation{color.END}:', suffix = '', length = 40)

# print('\nAccuracy on testing set with floating point model: %f' % (cnt/cnt_t))
#
# cnt = 0
# cnt_t = 0

for matrix in cmatrix:
    cmatrix_temp += matrix
    for i in range(fully_2_outdim):
        cnt += matrix[i][i]
        cnt_t += matrix.sum(axis=0)[i]

print('\nAccuracy on testing set with floating point model: %f' % (cnt/cnt_t))

print('\nConfusion matrix:')
print(cmatrix_temp)



cnt_q = 0
cnt_t_q = 0

cm_cnt = 0
cmatrix_temp = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)

# cmatrix_q = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)
# accmatrix_q = np.zeros(fully_2_outdim)

# cmatrix_q = np.zeros((len(data_names), fully_2_outdim, fully_2_outdim), dtype=int)
cmatrix_q = np.zeros((len(data_files), fully_2_outdim, fully_2_outdim), dtype=int)

# cmatrix_q = [[[[] for j in range(fully_2_outdim)] for k in range(fully_2_outdim)] for d in data_names]

print('\n\n')
printProgressBar(0, len(loader_test), prefix = f'{color.NONE}Fixed model evaluation{color.END}:', suffix = '', length = 40)
for i, data in enumerate(loader_test):

    inputs, labels = data

    outputs = model_quantized(inputs.float())

    list.clear(max_i)
    for o in outputs:
        m=max(o)
        indx=list(o).index(m)
        max_i.append(indx)

    for idx, tgt in zip(max_i, labels):
        cmatrix_q[R_test[cm_cnt]][idx][tgt] += 1
        cm_cnt += 1


    # for o, m in zip(labels, max_i):
    #     if o == m:
    #         cnt_q = cnt_q + 1
    #     cnt_t_q = cnt_t_q + 1

    printProgressBar(i + 1, len(loader_test), prefix = f'{color.NONE}Fixed model evaluation{color.END}:', suffix = '', length = 40)

for matrix in cmatrix_q:
    cmatrix_temp += matrix
    for i in range(fully_2_outdim):
        cnt_q += matrix[i][i]
        cnt_t_q += matrix.sum(axis=0)[i]

print('\nAccuracy on testing set with fixed point model: %f' % (cnt_q/cnt_t_q))

# for i in range(fully_2_outdim):
#     accmatrix_q[i] = (cmatrix_q[i][i]/cmatrix_q.sum(axis=0)[i])

print('\nConfusion matrix:')
print(cmatrix_temp)

# print('\nAccuracy per output:')
# print(accmatrix_q)









# ███████╗██╗  ██╗██████╗  ██████╗ ██████╗ ████████╗██╗███╗   ██╗ ██████╗     ██████╗  █████╗ ████████╗ █████╗
# ██╔════╝╚██╗██╔╝██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██║████╗  ██║██╔════╝     ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
# █████╗   ╚███╔╝ ██████╔╝██║   ██║██████╔╝   ██║   ██║██╔██╗ ██║██║  ███╗    ██║  ██║███████║   ██║   ███████║
# ██╔══╝   ██╔██╗ ██╔═══╝ ██║   ██║██╔══██╗   ██║   ██║██║╚██╗██║██║   ██║    ██║  ██║██╔══██║   ██║   ██╔══██║
# ███████╗██╔╝ ██╗██║     ╚██████╔╝██║  ██║   ██║   ██║██║ ╚████║╚██████╔╝    ██████╔╝██║  ██║   ██║   ██║  ██║
# ╚══════╝╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝

print('\n\n')
t_done = False
t_dict = {'prefix' : f'{color.NONE}Exporting data{color.END}: '}
t = threading.Thread(target=animate, kwargs=t_dict)
t.start()

model_quantized.debug = True

# FIXME

for i, data in enumerate(loader_test):

    inputs, labels = data

    if(model_quantized.debug):

        for j in range(0,value_batch_size):
            model_quantized.temp = j
            outputs = model_quantized(inputs[j].unsqueeze_(0).float())

        torch.set_printoptions(threshold=500000, precision=10) #,linehalf_windowth=20
        f = open(session_path+"inference_data_example/labels.txt", "w")
        f.write(str(labels))
        f.close()

    break

training_parameters = {
    'session_name': session_name,
    'dataset_labels' : dataset_labels,
    'dataset_path' : dataset_path,
    'dataset_split' : dataset_split,
    # 'separate_testing' : args.separate_testing,
    'frame_len_sec' : frame_len_sec,
    'frame_shift_sec' : frame_shift_sec,
    'downscaling' : downscaling,
    'median' : median,
    'augmentation_rsize' : False if list(augmentation_rsize) == [0] else list(augmentation_rsize),
    'augmentation_shift' : False if list(augmentation_shift) == [0] else list(augmentation_shift),
    'augmentation_rotat' : False if list(augmentation_rotat) == [0] else list(augmentation_rotat),
    'random_seed' : random_seed,
    'batch_size' : value_batch_size,
    'normalization' : normalization,
    'conv_indim' : conv_indim,
    'pool_ks' : pool_ks,
    'conv_1_if' : conv_1_if,
    'conv_1_of' : conv_1_of,
    'conv_1_ks' : conv_1_ks,
    'conv_2_if' : conv_2_if,
    'conv_2_of' : conv_2_of,
    'conv_2_ks' : conv_2_ks,
    'fully_1_indim' : fully_1_indim,
    'fully_1_outdim' : fully_1_outdim,
    'fully_2_indim' : fully_2_indim,
    'fully_2_outdim' : fully_2_outdim,
    'train_epoch' : num_trainepoch_effective,
    'best_epoch' : epoch_best + 1,
    'optimizer' : str(optimizer).replace("\n","").replace("    ",", "),
    # 'learning_rate' : train_data[train_dic['learing_rate']][-1],
    'criterion' : str(criterion),
    'qconfig' : str(torch.backends.quantized.engine),
    'training_acc' : train_data[train_dic['train_acc']][epoch_best],
    'validation_acc' : train_data[train_dic['valid_acc']][epoch_best],
    'training_loss' : train_data[train_dic['train_loss']][epoch_best],
    'validation_loss' : train_data[train_dic['valid_loss']][epoch_best],
    'floating_point_accuracy' : (cnt/cnt_t),
    'fixed_point_accuracy' : (cnt_q/cnt_t_q)
}

with open(session_path+'training_summary.json', 'w') as json_file:
    json.dump(training_parameters, json_file, indent=4)

# with open(session_path+'training_loss.pickle', 'wb') as output_file:
#     pk.dump(training_loss, output_file)
#
# with open(session_path+'training_acc.pickle', 'wb') as output_file:
#     pk.dump(training_acc, output_file)

with open(session_path+'training_data.pickle', 'wb') as output_file:
    pk.dump(train_data, output_file)

with open(session_path+'confusionmatrix_float.pickle', 'wb') as output_file:
    pk.dump(cmatrix, output_file)

with open(session_path+'confusionmatrix_fixed.pickle', 'wb') as output_file:
    pk.dump(cmatrix_q, output_file)

t_done = True
time.sleep(0.2)


print(f'\n{color.NONE}Summary{color.END}')
for par in training_parameters:
    print(repr(par),":",training_parameters[par])

print(f'{color.BOLD}\n\n\nEnding {color.NONE}training{color.END}{color.BOLD} session \'{session_name}\'{color.END}')

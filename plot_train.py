import os
import argparse
import numpy as np
import pickle as pk
import seaborn as sn
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
from scipy.interpolate import make_interp_spline, BSpline
import tikzplotlib

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', required=True, dest='train', type=dir_path, help="training session path")

args = parser.parse_args()

session_path = args.train

# with open(f'{session_path}/training_acc.pickle', 'rb') as input_file:
#     training_acc = pk.load(input_file)
# with open(f'{session_path}/training_loss.pickle', 'rb') as input_file:
#     training_loss = pk.load(input_file)
with open(f'{session_path}/training_data.pickle', 'rb') as input_file:
    train_data = pk.load(input_file)
with open(f'{session_path}/confusionmatrix_float.pickle', 'rb') as input_file:
    cmatrix_float = pk.load(input_file)
with open(f'{session_path}/confusionmatrix_fixed.pickle', 'rb') as input_file:
    cmatrix_fixed = pk.load(input_file)

json_file = open(f'{session_path}/training_summary.json', 'r')
json_data = json.load(json_file)
dataset_labels = json_data['dataset_labels']
fully_2_outdim = json_data['fully_2_outdim']

train_dic = {
    'train_loss' : 0,
    'valid_loss' : 1,
    'train_acc' : 2,
    'valid_acc' : 3,
    'learing_rate' : 4
}



cnt = 0
cnt_t = 0
matrix_float = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)

for matrix in cmatrix_float:
    matrix_float += matrix
    for i in range(fully_2_outdim):
        cnt += matrix[i][i]
        cnt_t += matrix.sum(axis=0)[i]



cnt_q = 0
cnt_t_q = 0
matrix_fixed = np.zeros((fully_2_outdim, fully_2_outdim), dtype=int)

for matrix in cmatrix_fixed:
    matrix_fixed += matrix
    for i in range(fully_2_outdim):
        cnt_q += matrix[i][i]
        cnt_t_q += matrix.sum(axis=0)[i]



y1_data = train_data[train_dic['train_acc']]
y3_data = train_data[train_dic['valid_acc']]
y2_data = train_data[train_dic['train_loss']]
y4_data = train_data[train_dic['valid_loss']]
y5_data = train_data[train_dic['learing_rate']]

x = np.array(range(1,len(y1_data)+1))


try:
    fig = plt.figure(figsize=(10, 3))


    xnew = np.linspace(x.min(), x.max(), 300)

    spl1 = make_interp_spline(x, y1_data, k=3)  # type: BSpline
    spl2 = make_interp_spline(x, y2_data, k=3)  # type: BSpline
    spl3 = make_interp_spline(x, y3_data, k=3)  # type: BSpline
    spl4 = make_interp_spline(x, y4_data, k=3)  # type: BSpline
    spl5 = make_interp_spline(x, y5_data, k=3)  # type: BSpline



    # fig.suptitle(f'Training path: {session_path}')



    graph1_smooth = fig.add_subplot(1, 2, 1)

    # graph1_smooth.set_title('Accuracy per epoch')
    # plt.ylabel("Threat score")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    graph1_smooth = spl1(xnew)
    graph1_smooth_2 = spl3(xnew)
    p1 = plt.plot(xnew,graph1_smooth)
    p2 = plt.plot(xnew,graph1_smooth_2)

    # plt.text(xnew[-1], np.amin((graph1_smooth,graph1_smooth_2)),f'Training\n', color='tab:blue', va='bottom', ha='right') #, weight="bold"
    # plt.text(xnew[-1], np.amin((graph1_smooth,graph1_smooth_2)),f'Validation', color='tab:orange', va='bottom', ha='right')

    plt.legend((p1[0], p2[0]), ('Training', 'Validation'), loc='lower right')



    graph2_smooth = fig.add_subplot(1, 2, 2)

    # graph2_smooth.set_title('Loss & learning rate per epoch')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    graph3_smooth = graph2_smooth.twinx()
    plt.ylabel("Learning rate", color='tab:green')

    graph2_smooth = spl2(xnew)
    graph2_smooth_2 = spl4(xnew)
    graph3_smooth = spl5(xnew)
    p1 = plt.plot(xnew,graph2_smooth)
    p2 = plt.plot(xnew,graph2_smooth_2)
    p3 = plt.plot(xnew,graph3_smooth)

    # plt.text(xnew[-1], np.amax((graph2_smooth,graph2_smooth_2)),f'Training', color='tab:blue', va='top', ha='right') #, weight="bold"
    # plt.text(xnew[-1], np.amax((graph2_smooth,graph2_smooth_2)),f'\nValidation', color='tab:orange', va='top', ha='right')

    plt.legend((p1[0], p2[0], p3[0]), ('Training', 'Validation', 'Learning rate'), loc='upper right')


    fig.tight_layout()
    # plt.show()
    plt.savefig("output/plot/train.pdf", format="pdf")
    # exit()
except:
    pass

try:
    fig = plt.figure(figsize=(9, 3))

    cm = fig.add_subplot(1, 2, 1)
    # cm.set_title(f'Floating point model confusion matrix\nAccuracy: {np.sum(np.diag(matrix_float))/np.sum(matrix_float):.5f}')

    # cmap = sn.cubehelix_palette(as_cmap=True, light=1)
    cmap = sn.cubehelix_palette(gamma= 8, start=1.4, rot=.55, dark=0.8, light=1, as_cmap=True)
    # cmap = sn.cubehelix_palette(gamma= 16, start=0.15, rot=.15, dark=0.9, light=1, as_cmap=True)

    df_cm = pd.DataFrame(matrix_float, index = [i for i in dataset_labels], columns = [i for i in dataset_labels])
    # sn.load('month', 'year', 'passengers')
    res = sn.heatmap(df_cm, annot=True, fmt='g', cmap = cmap) # vmax=2000.0
    for _, spine in res.spines.items():
        spine.set_visible(True)

    plt.ylabel("Predicted label")
    plt.xlabel("True label")

    cm = fig.add_subplot(1, 2, 2)
    # cm.set_title(f'Fixed point model confusion matrix\nAccuracy: {np.sum(np.diag(matrix_fixed))/np.sum(matrix_fixed):.5f}')

    df_cm = pd.DataFrame(matrix_fixed, index = [i for i in dataset_labels], columns = [i for i in dataset_labels])

    res = sn.heatmap(df_cm, annot=True, fmt='g', cmap=cmap) # vmax=2000.0
    for _, spine in res.spines.items():
        spine.set_visible(True)

    plt.ylabel("Predicted label")
    plt.xlabel("True label")

    fig.tight_layout()
    # plt.show()
    plt.savefig("output/plot/cmatrix.pdf", format="pdf")
    tikzplotlib.save("output/plot/cmatrix.tex")
except:
    pass

import json
import os
import argparse

import math
import statistics

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt



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



parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', dest='input', required=True, nargs='*', help="input (folders or files)")
parser.add_argument('-m','--median', dest='median', default=1, type=int, help='plot median data')
parser.add_argument('-C','--calibrate', dest='calibrate', nargs='*', help='calibration file')
parser.add_argument('-G','--gforce', dest='gforce', default='', action='store_true', help='gforce remover')
args = parser.parse_args()

session_input = args.input
session_median = args.median
session_calibrate = args.calibrate
session_gforce = args.gforce



if session_calibrate:
    calibration_items = []
    calibration_x = 0
    calibration_y = 0
    calibration_z = 0
    for i in session_calibrate:
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
for i in session_input:
    if os.path.isfile(i) and i not in data_items:
        data_items.append(i)
    for dirpath, dirnames, filenames in os.walk(i):
        for filename in [f for f in filenames if f.endswith(".json")]:
            if os.path.join(dirpath, filename) not in data_items: # dirpath.split('/')[-1] in dataset_labels
                data_items.append(os.path.join(dirpath, filename))



# for folder in os.listdir('dataset/test'):
#     if session_path == None or f'dataset/test/{folder}' in session_path:
#         if os.path.isdir(f'dataset/test/{folder}'):
#             for file in os.listdir(f'dataset/test/{folder}'):
#                 if session_file == None or f'dataset/test/{folder}/{file}' in session_file:
for i in data_items:
    with open(i, 'r') as json_file:
        history = json.load(json_file)
    # print(f"dataset/test/{folder}/{file}: [{len(history['data']['x'])}, {len(history['data']['y'])}, {len(history['data']['z'])}]")

    mpl.rcParams['legend.fontsize'] = 10

    if session_median > 1:
        x_temp = []
        y_temp = []
        z_temp = []

        for i in range(history['samples']):
            if i + session_median > history['samples']:
                break
            x_temp.append(statistics.median(history['data']['x'][i : i + session_median]))
            y_temp.append(statistics.median(history['data']['y'][i : i + session_median]))
            z_temp.append(statistics.median(history['data']['z'][i : i + session_median]))

        # ax.plot(x_m, y_m, z_m, label='parametric curve')

    else:
        x_temp = history['data']['x']
        y_temp = history['data']['y']
        z_temp = history['data']['z']

    if session_calibrate:
        x = []
        y = []
        z = []
        for x_i, y_i, z_i in zip(x_temp, y_temp, z_temp):
            temp = np.matmul(np.array([x_i, y_i, z_i]), calibration_matrix)
            x.append(temp[0])
            y.append(temp[1])
            if session_gforce:
                z.append(temp[2] - 981)
            else:
                z.append(temp[2])
    else:
        x = x_temp
        y = y_temp
        z = z_temp

        # temp = [np.matmul(np.array([x_i, y_i, z_i]), calibration_matrix) for x_i, y_i, z_i in zip(x_temp, y_temp, z_temp)]
        # x = [ for [x_i, y_i, z_i] in temp]

    module = [int(math.sqrt(x_i*x_i + y_i*y_i + z_i*z_i)) for x_i, y_i, z_i in zip(x, y, z)]


    fig = plt.figure()

    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title(f"{i}")
    ax1.plot(range(history['samples'] - session_median + 1), x)
    ax1.plot(range(history['samples'] - session_median + 1), y)
    ax1.plot(range(history['samples'] - session_median + 1), z)
    ax1.plot(range(history['samples'] - session_median + 1), module, linewidth=2)

    if 'events' in history:
        for e in history['events']:
            plt.text(e,max(module[e-20 : e+20]) + 20,f"{history['class']}", va='bottom', ha='center', weight="bold")

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    ax1.legend(['x', 'y', 'z', 'module'])

    fig.tight_layout()
    plt.show()

    exit()

    continue

    ax = fig.gca(projection='3d')
    ax.set(title = f'Class: {os.path.dirname(i).split("/")[-1]}\nFile: {os.path.basename(i)}')
    ax.set_zlabel('Time')

    if session_median > 1:
        z = range(history['samples'])
        x = [history['data']['x'][i] for i in z]
        y = [history['data']['y'][i] for i in z]

        x_m = []
        y_m = []
        z_m = []

        for i in z:
            if i + session_median > history['samples']:
                break
            x_temp.append(statistics.median(x[i : i + session_median]))
            y_temp.append(statistics.median(y[i : i + session_median]))
            z_temp.append(i)

        ax.plot(x_m, y_m, z_m, label='parametric curve')

    else:
        z = range(history['samples'])
        x = [history['data']['x'][i] for i in z]
        y = [history['data']['y'][i] for i in z]

        ax.plot(x, y, z, label='parametric curve')

    # fig.tight_layout()
    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.tight_layout()
    plt.show()

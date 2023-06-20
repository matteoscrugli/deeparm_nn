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
parser.add_argument('-g','--zgforce', dest='zgforce', default='', action='store_true', help='zgforce remover')
parser.add_argument('-G','--gforce', dest='gforce', default='', action='store_true', help='gforce remover')
parser.add_argument('-n','--normalize', dest='normalize', action='store_true', help='gforce normalizer')
parser.add_argument('-l','--labeling', dest='labeling', default='', action='store_true', help='enable labeling')
parser.add_argument('-L','--Labeling', dest='Labeling', default='', action='store_true', help='enable labeling and save results')
parser.add_argument('-T','--threshold', dest='threshold', default=1.0, help='choose threshold gain')
parser.add_argument('-I','--ignore', dest='ignore', default=175, help='choose threshold samples ignore')
args = parser.parse_args()

session_input = args.input
session_median = args.median
session_calibrate = args.calibrate
session_zgforce = args.zgforce
session_gforce = args.gforce
session_normalize = args.normalize
session_labeling = args.labeling
session_Labeling = args.Labeling
session_ignore = int(args.ignore)
session_threshold = float(args.threshold)

if session_zgforce and session_gforce:
    print('\nChoose only one argument between -g and -G')
    exit()



labeling_ignore = session_ignore
labeling_threshold = session_threshold

labeling_th = 981 + (350) * labeling_threshold
if session_zgforce:
    labeling_th = (700) * labeling_threshold
elif session_gforce:
    labeling_th = (350) * labeling_threshold

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
            calibration_x += statistics.median(history['data'][0]['x'])
            calibration_y += statistics.median(history['data'][0]['y'])
            calibration_z += statistics.median(history['data'][0]['z'])
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



for i in data_items:
    with open(i, 'r') as json_file:
        history = json.load(json_file)

    mpl.rcParams['legend.fontsize'] = 10

    if session_median > 1:
        x_temp = []
        y_temp = []
        z_temp = []

        for i in range(history['samples']):
            if i + session_median > history['samples']:
                break
            x_temp.append(statistics.median(history['data'][0]['x'][i : i + session_median]))
            y_temp.append(statistics.median(history['data'][0]['y'][i : i + session_median]))
            z_temp.append(statistics.median(history['data'][0]['z'][i : i + session_median]))

    else:
        x_temp = history['data'][0]['x']
        y_temp = history['data'][0]['y']
        z_temp = history['data'][0]['z']

    if session_calibrate:
        x = []
        y = []
        z = []
        for x_i, y_i, z_i in zip(x_temp, y_temp, z_temp):
            temp = np.matmul(np.array([x_i, y_i, z_i]), calibration_matrix)
            x.append(temp[0])
            y.append(temp[1])

            if session_zgforce:
                z.append(temp[2] - 981)
            else:
                z.append(temp[2])
    else:
        x = x_temp
        y = y_temp
        z = z_temp

    if session_gforce:
        module = [math.sqrt(x_i*x_i + y_i*y_i + z_i*z_i) - 981 for x_i, y_i, z_i in zip(x, y, z)]
    else:
        module = [math.sqrt(x_i*x_i + y_i*y_i + z_i*z_i) for x_i, y_i, z_i in zip(x, y, z)]

    if session_normalize:
        x = [j / 981 for j in x]
        y = [j / 981 for j in y]
        z = [j / 981 for j in z]

        module = [j / 981 for j in module]

        labeling_th /= 981

    if 'events' not in history and (session_labeling or session_Labeling):
        # if not os.path.isfile(i.replace('.json', '_labeled.json')):
        index = []
        index_ignore = 0
        # index_next_ignore = -10
        fall_ignore = True
        for j, m in enumerate(module):
            # if index_next_ignore + 10 > j:
            #     continue
            if j >= index_ignore:
                if m > labeling_th:
                    if not fall_ignore:
                        fall_ignore = True
                        index.append(module[j : j + labeling_ignore].index(max(module[j : j + labeling_ignore])) + j)
                        index_ignore = j + labeling_ignore
                else:
                    fall_ignore = False
                    # index_next_ignore = j
        history['events'] = index
        if session_Labeling:
            with open(i, 'w') as json_file: # .replace('.json', '_labeled.json')
                json.dump(history, json_file, ensure_ascii=False, indent = 4) # , indent = 4

    fig = plt.figure()

    temp = f"{i}"
    try:
        temp += f"\nRipetitions: {history['ripetitions']}"
    except:
        pass

    ax1 = fig.add_subplot(1,1,1)
    ax1.set_title(temp)
    ax1.plot(range(history['samples'] - session_median + 1), x)
    ax1.plot(range(history['samples'] - session_median + 1), y)
    ax1.plot(range(history['samples'] - session_median + 1), z)
    ax1.plot(range(history['samples'] - session_median + 1), module, linewidth=2)

    if 'events' in history:
        for e in history['events']:
            plt.text(e,max(module[e-20 : e+20]),f"{history['class']}", va='bottom', ha='center', weight="bold")

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")
    ax1.legend(['x', 'y', 'z', 'module'], loc='lower right')

    fig.tight_layout()
    plt.show()

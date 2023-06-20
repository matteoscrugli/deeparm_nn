#%%
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib

import matplotlib.font_manager as fm

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def copy_keys(dict_orig):
    dict_new = {}
    for k, v in dict_orig.items():
        if isinstance(dict_orig[k], dict):
            dict_new[k] = copy_keys(dict_orig[k])
        else:
            dict_new[k] = str(type(dict_orig[k]))
    return dict_new

path = 'output/json/'
json_dicts = []
color_labels = {
    'G' : '#D0D0D0',
    'P' : '#720069',
    'SQ' : 'green'
}
custom_labels = [
    'Non-augmented, $f_{CNN}$ 4 Hz',
    'Non-augmented, $f_{CNN}$ 1 Hz',
    'Augmented, $f_{CNN}$ 4 Hz',
    'Augmented, $f_{CNN}$ 1 Hz'
    ]
model_idx_mod = {
    0 : 2,
    1 : 3,
    2 : 0,
    3 : 1
}
model_mod = {
    0 : 1.95,
    1 : 2.5,
    2 : 0.5,
    3 : 1.05
}

for filename in os.listdir(path):
    if filename.endswith('.json'):
        with open(os.path.join(path, filename)) as f:
            json_dicts.append(json.load(f))

model_list = []
max_index = 0
for j in json_dicts:
    model_list.append(f"{j['model']}_{j['shift']}")
    for d in j['dict']:
        max_index = max(max(j['dict'][d]['y']), max_index)

data = [None] * (max_index + 1)

for j in json_dicts:
    model_idx = model_list.index(f"{j['model']}_{j['shift']}")
    for d in j['dict']:
        for x, y in zip(j['dict'][d]['x'], j['dict'][d]['y']):
            if data[y] is None:
                data[y] = {
                    'G' : {'x': [], 'y': []},
                    'P' : {'x': [], 'y': []},
                    'SQ' : {'x': [], 'y': []}
                }
            # if d not in data[y]:
            #     data[y][d] = {'x': [], 'y': []}
            data[y][d]['x'].append(x * j['shift'])
            data[y][d]['y'].append(model_mod[model_idx])

# %%
vertical_lines = [
    [1.53,  4.83,   8.97, 13.04, 16.67, 20.40, 24.56],
    [0.6, 5.16, 9.07, 13.57, 17.07, 22.56, 26.91],
    [0.42, 4.29, 6.97, 9.79, 13.24, 16.2, 19.51, 22.7, 25.6], 
    [2.16, 6.02, 9.52, 13.25, 17.23, 21.5, 25.3],
    [0.03, 4.71, 9.52, 13.61, 17.59, 21.92, 25.12],
    [0.86, 3.71, 6.32, 9.7, 12.96, 15.75, 19.01, 23.1]
]
miss_line = [
    [[1], [], [], [], [], [1], [1]],
    [[], [], [], [1], [], [1], [1]],
    [[1], [], [], [], [], [1], [1], [], [1]],
    [[1], [], [], [], [], [], []],
    [[], [], [], [], [], [1], []],
    [[1], [], [], [], [], [], [], []]
]

figure_w = 16
figure_h = 4
fig = plt.figure(figsize=(figure_w, figure_h))

for i in range(len(data)):
    for d in data[i]:
        plt.scatter(data[i][d]['x'], data[i][d]['y'], marker="s", s = 30, c = color_labels[d])
    
    miss_w = 0.75
    miss_h = 0.2
    ylim_min = 0.15
    ylim_max = 2.85

    xmin, xmax = plt.xlim()  # Ottieni i limiti dell'asse x
    for v_line, m_line in zip(vertical_lines[i], miss_line[i]):
        plt.axvline(x=v_line, color='k', linestyle='--', alpha=0.5)
        for m in m_line:
            relative_xmin = (v_line - miss_w - xmin) / (xmax - xmin)
            relative_xmax = (v_line + miss_w - xmin) / (xmax - xmin)
            plt.axhspan(model_mod[m] - miss_h, model_mod[m] + miss_h, xmin=relative_xmin, xmax=relative_xmax, facecolor='red', alpha=0.25) #, zorder = 0)

    plt.ylim(ylim_min, ylim_max)
    plt.yticks([model_mod[k] for k in model_mod], custom_labels)
    # plt.yticks(['Squat', 'Squat', 'Squat', 'Push-up', 'Push-up', 'Push-up'])

    if i == 0:
        labels = ["Other", "Push-up", "Squat", "Ground truth", "Missed classification"]
        plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(labels), fancybox=True, shadow=False)
    if i == len(data) - 1:
        plt.xlabel("Time [s]") # ,fontsize=14

    # plt.savefig("output/plot/raster.png")
    plt.savefig(f"output/plot/raster_{i}.pdf", format="pdf")
    tikzplotlib.save(f"output/plot/raster_{i}.tex")
    # plt.show()

    plt.clf()
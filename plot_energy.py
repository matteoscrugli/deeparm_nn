#%%
import sys
import argparse
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import tikzplotlib

parser = argparse.ArgumentParser()
# parser.add_argument('-o','--opmode', dest='opmode', required=True, type=int, help="select operating mode index")
parser.add_argument('-s','--sort', dest='sort', action='store_true', help="sort the results")
parser.add_argument('-p','--proportion', dest='proportion', action='store_true', help="keep proportions between tasks")

if 'ipykernel_launcher.py' in sys.argv[0]:
    args = parser.parse_args('-p -s'.split())
else:
    args = parser.parse_args()

printenergy = True
printpie = True

def getenergies(opmode, freq, sending=True):
    energy_dict = {
        'Idle' : {
            1 : 0.002568193895,
            2 : 0.002802488873, # 0.002662488873,
            4 : 0.003267113991,
            8 : 0.004466806826,
            16 : 0.007076667697
        },
        'Raw' : 0.00000305, # 0.00000501,
        'Movement' : 0.00000291, # 0.00000191,
        'CNN arm' : 0.00025926, # 0.00025926,
        'CNN balance board' : 0.00108618,
        'Threshold' : 0.00000273,
        'Send' : {
            # 1 : 0.00022575989128,
            2 : 0.00014073160438, #2 : 0.00017073160438,
            # 4 : 0.00012871202676,
            8 : 0.00006951126111,
            16 : 0.00007288449823,
            1 : 0.00021329600000,
            # 2 : 0.00018445600000,
            4 : 0.00010517600000, #4 : 0.00013517600000,
            # 8 : 0.00007021600000,
            # 16 : 0.00007469600000
        }
    }
    om_dict = {
        'Operating mode raw' : {
            'defaultfreq' : 4,
            'task' : {
                'Raw' : 100,
                'Movement' : 0,
                'CNN arm' : 0,
                'CNN balance board' : 0,
                'Threshold' : 0,
                'Send' : 50
            }
        },
        'Operating mode movement' : {
            'defaultfreq' : 2,
            'task' : {
                'Raw' : 50,
                'Movement' : 10,
                'CNN arm' : 0,
                'CNN balance board' : 0,
                'Threshold' : 10,
                'Send' : 10
            }
        },
        'Operating mode CNN arm' : {
            'defaultfreq' : 1,
            'task' : {
                'Raw' : 20,
                'Movement' : 10,
                'CNN arm' : 1,
                'CNN balance board' : 0,
                'Threshold' : 1,
                'Send' : 1
            }
        },
        'Operating mode CNN balance board' : {
            'defaultfreq' : 4,
            'task' : {
                'Raw' : 20,
                'Movement' : 10,
                'CNN arm' : 0,
                'CNN balance board' : 1,
                'Threshold' : 1,
                'Send' : 1
            }
        }
    }
    energies = []
    for i in energy_dict:
        if not sending and i == 'Send':
            continue
        if type(energy_dict[i]) is dict:
            if i in om_dict[opmode]['task']:
                # print('A', i)
                energies.append(energy_dict[i][freq] * om_dict[opmode]['task'][i])
            else:
                # print('B', i)
                energies.append(energy_dict[i][freq])
        else:
            if i in om_dict[opmode]['task']:
                # print('C', i)
                energies.append(energy_dict[i] * om_dict[opmode]['task'][i])
            else:
                # print('D', i)
                energies.append(energy_dict[i])
    return energies

energies = getenergies('Operating mode raw', 4)
# print(energies)
print(sum(energies))
energies = getenergies('Operating mode CNN arm', 1, sending=False)
# print(energies)
print(sum(energies))

# print(sum(getenergies('Operating mode raw', 4, sending=True)) -
#             sum(getenergies('Operating mode raw', 4, sending=False)))
# exit()



dic_om = {

    0 : {
        'title' : r"Operating mode $\it{Raw}$",
        'tasks' : ['Idle', 'Raw', 'Send']
    },

    1 : {
        'title' : r"Operating mode $\it{Movement}$",
        'tasks' : ['Idle', 'Raw', 'Movement', 'Threshold', 'Send']
    },

    2 : {
        'title' : r"Operating mode $\it{CNN\, arm}$",
        'tasks' : ['Idle', 'Raw', 'Balance', 'CNN', 'Threshold', 'Send']
    },

    3 : {
        'title' : r"Operating mode $\it{CNN\,balance\,board}$",
        'tasks' : ['Idle', 'Raw', 'Balance', 'CNN', 'Threshold', 'Send']
    }

}

e_g = [0.00000494, 0.00000551 * (841 / (1550 + 841)), 0.00000494] # * (841 / (1550 + 841))
e_b = 0.00000551 * (1550 / (1550 + 841))
e_c = 0.00038731
e_t = 0.00000273
e_s = 0.00014182

alpha = [1 / 8, 1, 1]

p_idle = [0.00447695681001606, 0.00263491152637147, 0.00263491152637147] # dic_om[om]['idlepower']

f_samp = [100, 100, 100 / 5]
f_bb = 100 / 8
f_cnn = 60 / 60
f_send = [f_samp[0], f_bb, f_cnn]

unit = 0.001 # milli

en_max_value = 0

if printenergy:
    employees=[r"Raw", r"   Movement", r"CNN", r"CNN"]
    earnings={
        # "Base":[
        #     (p_idle[0] + e_g[0] * f_samp[0]) / unit,
        #     (p_idle[1] + e_g[0] * f_samp[1] + e_b * f_samp[1] + f_send[1] * e_t) / unit,
        #     (p_idle[2] + e_g[0] * f_samp[2] + e_b * f_samp[2] + e_c * f_cnn + f_send[1] * e_t) / unit
        # ],
        # "With trasmission":[
        #     (f_send[0] * alpha[0] * e_s) / unit,
        #     (f_send[1] * alpha[1] * e_s) / unit,
        #     (f_send[2] * alpha[2] * e_s) / unit
        # ],
        # "Without frequency\noptimization":[
        #     0,
        #     (p_idle[0] - p_idle[1]) / unit,
        #     (p_idle[0] - p_idle[2]) / unit
        # ],
        "Base":[
            sum(getenergies('Operating mode raw', 4, sending=False)) / unit,
            sum(getenergies('Operating mode movement', 2, sending=False)) / unit,
            sum(getenergies('Operating mode CNN arm', 1, sending=False)) / unit,
            sum(getenergies('Operating mode CNN balance board', 4, sending=False)) / unit,
        ],
        "With trasmission":[
            sum(getenergies('Operating mode raw', 4, sending=True)) / unit -
            sum(getenergies('Operating mode raw', 4, sending=False)) / unit,
            sum(getenergies('Operating mode movement', 2, sending=True)) / unit -
            sum(getenergies('Operating mode movement', 2, sending=False)) / unit,
            sum(getenergies('Operating mode CNN arm', 1, sending=True)) / unit -
            sum(getenergies('Operating mode CNN arm', 1, sending=False)) / unit,
            sum(getenergies('Operating mode CNN balance board', 4, sending=True)) / unit -
            sum(getenergies('Operating mode CNN balance board', 4, sending=False)) / unit,
        ],
        "Without frequency\noptimization":[
            sum(getenergies('Operating mode raw', 4, sending=True)) / unit -
            sum(getenergies('Operating mode raw', 4, sending=True)) / unit,
            sum(getenergies('Operating mode movement', 4, sending=True)) / unit -
            sum(getenergies('Operating mode movement', 2, sending=True)) / unit,
            sum(getenergies('Operating mode CNN arm', 4, sending=True)) / unit -
            sum(getenergies('Operating mode CNN arm', 1, sending=True)) / unit,
            sum(getenergies('Operating mode CNN balance board', 4, sending=True)) / unit -
            sum(getenergies('Operating mode CNN balance board', 4, sending=True)) / unit,
        ],
    }

    df=pd.DataFrame(earnings,index=employees)

    df.plot(kind="barh", stacked=True, figsize=(10,3), width = 0.65, color=['#173f5f', '#ac585b', '#d0d0d0'])
    # for i, p in enumerate(df.index):
    plt.text(s="Arm version", x=0.1, y=2, color="w", va="center", ha="left") # , size=18
    plt.text(s="Balance board version", x=0.1, y=3, color="w", va="center", ha="left") # , size=18
    plt.xlim((0, 10))
    plt.xlabel("Power consumption in mW")
    plt.ylabel("Operating mode")
    plt.legend() # loc="lower left",bbox_to_anchor=(0.8,1.0)
    plt.tight_layout()
    plt.savefig("output/plot/energy.png")
    plt.savefig("output/plot/energy.pdf", format="pdf")
    tikzplotlib.save("output/plot/energy.tex")
    # plt.show()

printpie_item = ['Operating mode raw', 'Operating mode movement', 'Operating mode CNN arm', 'Operating mode CNN balance board']
printpie_freq = [4, 2, 1, 4]
if printpie:
    # fig = plt.figure(figsize=(13.6, 4))
    fig = plt.figure(figsize=(16, 4))
    for om in range(4):
        # fig.add_subplot(1, 3, om + 1)
        fig.add_subplot(1, 4, om + 1)
        # om = args.opmode



        explode_len = 0.15
        explode_fac = 0.25
        label_dist = 0.73
        en_min_value = True
        min_value = 0.004
        en_min_label_value = True
        min_label_value = 0.1 # 0.02

        data = []
        energies = getenergies(printpie_item[om], printpie_freq[om])
        # if 'Idle' in dic_om[om]['tasks']:
        #     data.append([p_idle[om] / unit, 'Idle', '#173f5f'])
        # if 'Raw' in dic_om[om]['tasks']:
        #     data.append([e_g[om] * f_samp[om] / unit, 'Raw', '#20639b'])
        # if 'Balance' in dic_om[om]['tasks']:
        #     data.append([e_b * f_samp[om] / unit, 'Balance', '#3caea3'])
        # if 'CNN' in dic_om[om]['tasks']:
        #     data.append([e_c * f_cnn / unit, 'CNN', '#f6a55c'])
        # if 'Threshold' in dic_om[om]['tasks']:
        #     data.append([f_send[om] * e_t / unit, 'Threshold', '#ed553b'])
        # if 'Send' in dic_om[om]['tasks']:
        #     data.append([f_send[om] * alpha[om] * e_s / unit, 'Send', '#ac585b'])

        # if 'Idle' in dic_om[om]['tasks']:
        #     data.append([energies[0] / unit, 'Idle', '#173f5f'])
        # if 'Raw' in dic_om[om]['tasks']:
        #     data.append([energies[1] / unit, 'Raw', '#20639b'])
        # if 'Balance' in dic_om[om]['tasks']:
        #     data.append([energies[2] / unit, 'Balance', '#3caea3'])
        # if 'CNN' in dic_om[om]['tasks']:
        #     data.append([energies[3] / unit, 'CNN', '#f6a55c'])
        # if 'Threshold' in dic_om[om]['tasks']:
        #     data.append([energies[4] / unit, 'Threshold', '#ed553b'])
        # if 'Send' in dic_om[om]['tasks']:
        #     data.append([energies[5] / unit, 'Send', '#ac585b'])
        
        data.append([energies[0] / unit, 'Idle', '#173f5f'])
        data.append([energies[1] / unit, 'Raw', '#20639b'])
        data.append([energies[2] / unit, 'Movement', '#3caea3'])
        data.append([energies[3] / unit, 'CNN arm', '#f6a55c'])
        data.append([energies[4] / unit, 'CNN balance board', '#f6a55c'])
        data.append([energies[5] / unit, 'Threshold', '#ed553b'])
        data.append([energies[6] / unit, 'Send', '#ac585b'])

        values = []
        names = []
        colors = []
        [
            (
                values.append(v),
                names.append(n),
                colors.append(c)
            )
            for [v, n, c] in data if v
        ]



        if args.sort:
            temp_index = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
            values = [values[i] for i in temp_index]
            names = [names[i] for i in temp_index]
            colors = [colors[i] for i in temp_index]

        if args.proportion:
            if not om:
                en_max_value = sum(values)
            else:
                values.append(en_max_value - sum(values))
                names.append('')
                colors.append('#ffffff')

        string_values = ['  ' + str("{:.2f}".format(v)) + ' mW' for v in values]

        if en_min_value:
            for i in range(len(values)):
                if values[i] < max(values) * min_value:
                    values[i] = max(values) * min_value
        if en_min_label_value:
            for i in range(len(values)):
                if values[i] < max(values) * min_label_value:
                    string_values[i] = ''



        counts = pd.Series(values, index=names)
        # explode = [((max(values) - v) / (max(values) - min(values))) * explode_len for v in values]
        explode = [((1 / pow(d, explode_fac)) * pow(min(values), explode_fac)) * explode_len for d in values]

        counts.plot(kind='pie', colors=colors, explode=explode, labeldistance=label_dist, labels=string_values, textprops = dict(color = 'w', rotation_mode = 'anchor', va='center', ha='center')) # , fontsize=17 , textprops={'color':"w"}

        if om == 3:
            # counts = pd.Series(values, index=names)
            # string_values = ['' + str("{:.2f}".format(v)) + ' mW' for v in values]
            # explode = [((1 / pow(d, explode_fac)) * pow(min(values), explode_fac)) * explode_len for d in values]
            # counts.plot(kind='pie', colors=colors, explode=explode, labeldistance=label_dist, labels=string_values, textprops = dict(color = 'w', rotation_mode = 'anchor', va='center', ha='center')) # , fontsize=17 , textprops={'color':"w"}

            # colors = [l[2] for l in data]
            # f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
            # handles = [f("s", colors[i]) for i in range(6)]
            # plt.legend(handles = handles, labels = [l[1] for l in data], loc="right", bbox_to_anchor=(1.5, 0.5)) #, loc="best"

            # labels = [l[1] for l in data]
            # colors = [c[2] for c in data]
            [mpatches.Patch(label=d[1], color=d[2]) for d in data]
            plt.legend(handles=[mpatches.Patch(label=d[1], color=d[2]) for d in data], loc='right', bbox_to_anchor=(1.7, 0.5))

        # plt.legend(labels=counts.index, loc="best")
        plt.title(dic_om[om]['title'])

        plt.axis('equal')
        plt.ylabel('')

        plt.tight_layout()
    
        # print('')
        # print(data)

    # plt.show()
    # plt.subplots_adjust(left=-0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("output/plot/pie.png")
    plt.savefig("output/plot/pie.pdf", format="pdf")
    tikzplotlib.save("output/plot/pie.tex")
# %%

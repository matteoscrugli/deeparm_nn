import argparse
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
# parser.add_argument('-o','--opmode', dest='opmode', required=True, type=int, help="select operating mode index")
parser.add_argument('-s','--sort', dest='sort', action='store_true', help="sort the results")
parser.add_argument('-p','--proportion', dest='proportion', action='store_true', help="keep proportions between tasks")
args = parser.parse_args()

printenergy = True
printpie = True

dic_om = {

    0 : {
        'title' : r"Operating mode $\it{Raw}$",
        'tasks' : ['Idle', 'Raw', 'Send']
    },

    1 : {
        'title' : r"Operating mode $\it{Balance}$",
        'tasks' : ['Idle', 'Raw', 'Balance', 'Threshold', 'Send']
    },

    2 : {
        'title' : r"Operating mode $\it{CNN}$",
        'tasks' : ['Idle', 'Raw', 'Balance', 'CNN', 'Threshold', 'Send']
    }

}



e_g = [0.00000296, 0.00000376 * (841 / (1550 + 841)), 0.00000296]
e_b = 0.00000376 * (1550 / (1550 + 841))
e_c = 0.000852385
e_t = 0.00000273
e_s = 0.00008396

alpha = [1 / 4, 1, 1]

p_idle = [0.004546, 0.002609, 0.003101] # dic_om[om]['idlepower']

f_samp = [100, 100 / 7, 100 / 7]
f_bb = 60 / 60
f_cnn = 60 / 60
f_send = [f_samp[0], f_bb, f_cnn]

unit = 0.001 # milli

en_max_value = 0

if printenergy:
    employees=[r"Operating mode $\it{Raw}$", r"Operating mode $\it{Balance}$", r"Operating mode $\it{CNN}$"]
    earnings={
        "Base":[
            (p_idle[0] + e_g[0] * f_samp[0]) / unit,
            (p_idle[1] + e_g[0] * f_samp[1] + e_b * f_samp[1] + f_send[1] * e_t) / unit,
            (p_idle[2] + e_g[0] * f_samp[2] + e_b * f_samp[2] + e_c * f_cnn + f_send[1] * e_t) / unit
        ],
        "With trasmission":[
            (f_send[0] * alpha[0] * e_s) / unit,
            (f_send[1] * alpha[1] * e_s) / unit,
            (f_send[2] * alpha[2] * e_s) / unit
        ],
        "Without frequency\noptimization":[
            0,
            (p_idle[0] - p_idle[1]) / unit,
            (p_idle[0] - p_idle[2]) / unit
        ],
    }

    df=pd.DataFrame(earnings,index=employees)

    df.plot(kind="barh", stacked=True, figsize=(10,3), width = 0.65, color=['#173f5f', '#ac585b', '#d0d0d0'])
    plt.xlim((0, 9))
    plt.xlabel("Power consumption in mW")
    plt.legend() # loc="lower left",bbox_to_anchor=(0.8,1.0)
    plt.tight_layout()
    plt.savefig("output/plot/energy.pdf", format="pdf")
    # plt.show()

if printpie:
    # fig = plt.figure(figsize=(13.6, 4))
    fig = plt.figure(figsize=(12, 4))
    for om in range(3):
        # fig.add_subplot(1, 3, om + 1)
        fig.add_subplot(1, 3, om + 1)
        # om = args.opmode



        explode_len = 0.15
        explode_fac = 0.25
        label_dist = 0.73
        en_min_value = True
        min_value = 0.004
        en_min_label_value = True
        min_label_value = 0.1 # 0.02

        data = []
        if 'Idle' in dic_om[om]['tasks']:
            data.append([p_idle[om] / unit, 'Idle', '#173f5f'])
        if 'Raw' in dic_om[om]['tasks']:
            data.append([e_g[om] * f_samp[om] / unit, 'Raw', '#20639b'])
        if 'Balance' in dic_om[om]['tasks']:
            data.append([e_b * f_samp[om] / unit, 'Balance', '#3caea3'])
        if 'CNN' in dic_om[om]['tasks']:
            data.append([e_c * f_cnn / unit, 'CNN', '#f6a55c'])
        if 'Threshold' in dic_om[om]['tasks']:
            data.append([f_send[om] * e_t / unit, 'Threshold', '#ed553b'])
        if 'Send' in dic_om[om]['tasks']:
            data.append([f_send[om] * alpha[om] * e_s / unit, 'Send', '#ac585b'])

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

        if om == 2:
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
            plt.legend(handles=[mpatches.Patch(label=d[1], color=d[2]) for d in data], loc="right", bbox_to_anchor=(1.5, 0.5))

        # plt.legend(labels=counts.index, loc="best")
        plt.title(dic_om[om]['title'])

        plt.axis('equal')
        plt.ylabel('')

        plt.tight_layout()

    # plt.show()
    # plt.subplots_adjust(left=-0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig("output/plot/pie.pdf", format="pdf")

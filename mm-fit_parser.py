import os
import argparse
import json
import csv
import sys
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path



parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True, dest='path', help="mm-fit files")
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the output folder if it already exists")
parser.add_argument('--fe', dest='f_exercises', nargs='*', default=[], help='exercises filter')
parser.add_argument('--fd', dest='f_devices', nargs='*', default=[], help='devices filter')
parser.add_argument('--fp', dest='f_position', nargs='*', default=[], help='position filter')
parser.add_argument('--ft', dest='f_type', nargs='*', default=[], help='type filter')
args = parser.parse_args()



def create_folder(input_path, ignore = False):
    if os.path.isdir(input_path):
        if not ignore:
            if args.overwrite:
                try:
                    shutil.rmtree(input_path)
                    Path(input_path).mkdir(parents=True, exist_ok=True)
                except OSError:
                    print("Error in folder creation ("+input_path+").")
                    exit()
            else:
                print(f'Path ({input_path}) already exists')
                exit()
    else:
        try:
            Path(input_path).mkdir(parents=True, exist_ok=True)
        except OSError:
            print("Error in folder creation ("+input_path+").")
            exit()



if args.path[-1] == '/':
    output_path = args.path[:-1] + '_parsed' + '/'
else:
    output_path = args.path + '_parsed' + '/'
create_folder(output_path)

mmfitdict = {}

for dirpath, dirnames, filenames in os.walk(args.path):
    for filename in sorted(filenames):
        temp = filename.split('.')[0].split('_')
        if temp[0] not in mmfitdict:
            mmfitdict[temp[0]] = {
                'labels' : {},
                'data' : {
                    'filepath' : [],
                    'device' : [],
                    'position' : [],
                    'type' : []
                }
            }
        if 'labels' in filename:
            mmfitdict[temp[0]]['labels'] = {
                'filepath' : os.path.join(dirpath, filename),
                'exercises' : {}
            }
        elif 'pose' in filename:
            mmfitdict[temp[0]]['data']['filepath'].append(os.path.join(dirpath, filename))
            mmfitdict[temp[0]]['data']['device'].append(temp[1])
            mmfitdict[temp[0]]['data']['position'].append('')
            mmfitdict[temp[0]]['data']['type'].append(temp[2])
        else:
            mmfitdict[temp[0]]['data']['filepath'].append(os.path.join(dirpath, filename))
            mmfitdict[temp[0]]['data']['device'].append(temp[1])
            mmfitdict[temp[0]]['data']['position'].append(temp[2])
            mmfitdict[temp[0]]['data']['type'].append(temp[3])



temp = []
for record in mmfitdict:
    if not mmfitdict[record]['labels']:
        temp.append(record)
for t in temp:
    mmfitdict.pop(t, None)



filter = {
    'exercises' : args.f_exercises, # ['squats', 'pushups'],
    'device' : args.f_devices, # ['sw'],
    'position' : args.f_position, # ['l'],
    'type' : args.f_type # ['acc']
}
classes = {
    'squats' : 'SQ',
    'pushups' : 'P',
    'dumbbell_shoulder_press' : 'DS',
    'lunges' : 'L',
    'dumbbell_rows' : 'DR',
    'situps' : 'SU',
    'tricep_extensions' : 'T',
    'bicep_curls' : 'B',
    'lateral_shoulder_raises' : 'LS',
    'jumping_jacks' : 'J'
}
frequencies = {
    'acc' : 100
}
paddig = 100



for record in mmfitdict:
    with open(mmfitdict[record]['labels']['filepath'], newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',') # , quotechar='|'
        for row in spamreader:

            try:
                mmfitdict[record]['labels']['exercises'][row[3]]['start'].append(int(row[0]))
                mmfitdict[record]['labels']['exercises'][row[3]]['end'].append(int(row[1]))
                mmfitdict[record]['labels']['exercises'][row[3]]['ripetitions'].append(int(row[2]))
            except:
                mmfitdict[record]['labels']['exercises'][row[3]] = {
                    'start' : [int(row[0])],
                    'end' : [int(row[1])],
                    'ripetitions' : [int(row[2])]
                }

    for filepath, device, position, type in zip(
        mmfitdict[record]['data']['filepath'],
        mmfitdict[record]['data']['device'],
        mmfitdict[record]['data']['position'],
        mmfitdict[record]['data']['type']
        ):
        if (
            (device in filter['device'] or filter['device'] == []) and
            (position in filter['position'] or filter['position'] == []) and
            (type in filter['type'] or filter['type'] == [])
            ):
            create_folder(output_path + device, ignore = True)
            data = np.load(filepath)
            if data.shape[-1] == 5:
                data = data[:, (0, 2, 3, 4)]
                shape = '3D'
            else:
                data = data[:, (0, 2)]
                shape = '1D'
            data_t = data[:, 0] - data[0, 0]
            for exercise in mmfitdict[record]['labels']['exercises']:
                if exercise in filter['exercises'] or filter['exercises'] == []:
                    create_folder(output_path + device + '/' + exercise, ignore = True)
                    for start, end, ripetitions in zip(
                        mmfitdict[record]['labels']['exercises'][exercise]['start'],
                        mmfitdict[record]['labels']['exercises'][exercise]['end'],
                        mmfitdict[record]['labels']['exercises'][exercise]['ripetitions']
                    ):
                        try:
                            start_t = int(np.where(data_t == start)[0][0])
                            end_t = int(np.where(data_t == end)[0][-1])
                        except Exception as e:
                            # print(e)
                            # print(np.where(data_t == start), np.where(data_t == start))
                            continue
                        if shape == '3D':
                            temp = {
                                'class': classes[exercise],
                                'samples': end_t - start_t,
                                'frequency': frequencies[type],
                                'ripetitions' : ripetitions,
                                'datetime': '',
                                'note': '',
                                'data': {
                                    'x': list(data[start_t : end_t, 1]),
                                    'y': list(data[start_t : end_t, 2]),
                                    'z': list(data[start_t : end_t, 3])
                                }
                            }
                        else:
                            temp = {
                                'class': classes[exercise],
                                'samples': end - start,
                                'frequency': frequencies[type],
                                'datetime': '',
                                'note': '',
                                'data': list(data[start_t : end_t, 1])
                            }
                        with open(output_path + device + '/' + exercise + f"/{filepath.split('.')[0].split('/')[-1]}_{start}_{end}.json", 'w') as json_file:
                            json.dump(temp, json_file, indent=4)

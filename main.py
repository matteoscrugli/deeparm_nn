import json
import os

dir = 'dataset/matteo/R'
file = 'dataset/matteo/R/20211028-190527.json'
size = 215

# for folder in os.listdir(dir):
    # if os.path.isdir(f'dataset/raw/{folder}'):
    # for file in os.listdir(dir):
# with open(f'{dir}/{file}', 'r') as json_file:
with open(file, 'r') as json_file:
    history = json.load(json_file)
with open('signalx.txt', 'w') as f:
    for i, x in enumerate(history['data']['x']):
        if i == 215:
            break
        f.write(f'{i} {x}\n')
with open('signaly.txt', 'w') as f:
    for i, y in enumerate(history['data']['x']):
        if i == 215:
            break
        f.write(f'{215 - i} {-y}\n')
# exit()
# print(f"dataset/raw/{folder}/{file}: [{len(history['data']['x'])}, {len(history['data']['y'])}, {len(history['data']['z'])}]")

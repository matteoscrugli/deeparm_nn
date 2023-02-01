import sys
import os
import time
import json
import argparse

import math
import numpy as np
import statistics
import random

import platform
import asyncio
import logging

from pathlib import Path

# import random
# from itertools import count
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation

# from numpy import cos, sin, pi, absolute, arange
# from scipy.signal import kaiserord, lfilter, firwin, freqz
# from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show

from bleak import BleakClient
from bleak import _logger as logger

# address = "C0:6E:33:30:41:4D" # DEV MATTEO
# address = "C0:6E:38:30:41:4D" # CRADLE MATTEO
# address = "C0:6E:28:30:3A:4D" # CRADLE BOJAN
# address = "C0:6E:38:30:41:4D" # CRADLE 1
# address = "C0:6E:29:31:3B:48" # CRADLE 2

devices_params = {
    'M0DEV' : {
        'address' : 'C0:6E:33:30:41:4D',
        'acc_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        },
        'gyro_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        }},
    'M0' : {
        'address' : 'C0:6E:38:30:41:4D',
        'acc_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        },
        'gyro_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        }},
    'M1' : {
        'address' : 'C0:6E:38:30:41:4D',
        'acc_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        },
        'gyro_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        }},
    'M2' : {
        'address' : 'C0:6E:29:31:3B:48',
        'acc_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        },
        'gyro_offset': {
            'x' : 420,
            'y' : 2800,
            'z' : 770,
        }},
    'M3' : {
        'address' : 'C0:6E:25:31:20:48',
        'acc_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        },
        'gyro_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        }},
    'B0' : {
        'address' : 'C0:6E:29:31:3B:48',
        'acc_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        },
        'gyro_offset': {
            'x' : 0,
            'y' : 0,
            'z' : 0,
        }}
}



parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', dest='name', required=True, help="session name")
parser.add_argument('-c','--class', dest='classes', required=True, help="class type")
parser.add_argument('-t','--test', dest='test', action='store_true', help='test mode')
parser.add_argument('-a','--address', dest='address', default=devices_params['M0DEV']['address'], help="SensorTile MAC address")
parser.add_argument('-N','--nsensors', dest='nsensors', default=2, type=int, help='number of sensors')
# parser.add_argument('--calibration', dest='calibration', action='store_true', help="caloibration samples")
parser.add_argument('-s','--samples', dest='samples', default=6000, type=int, help="samples limit")
parser.add_argument('-f','--frequency', dest='frequency', default=100, type=int, help="samples frequency (Hz)")
parser.add_argument('-ox','--offsetx', dest='offsetx', nargs='*', default=[0], type=int, help='X offset')
parser.add_argument('-oy','--offsety', dest='offsety', nargs='*', default=[0], type=int, help='Y offset')
parser.add_argument('-oz','--offsetz', dest='offsetz', nargs='*', default=[0], type=int, help='Z offset')
parser.add_argument('--note', dest='note', help="note")
parser.add_argument('-d','--delay', dest='delay', default=3, type=int, help="countdown")
# parser.add_argument('-p','--plot', dest='plot', action='store_true', help='plot data')
parser.add_argument('-m','--median', dest='median', default=0, type=int, help='plot n median values')
parser.add_argument('-g','--gain', dest='gain', default=1/2, type=float, help='plot gain')
parser.add_argument('-l','--live', dest='live', action='store_true', help='only plot data')
parser.add_argument('-i','--inference', dest='inference', help="applies inference to results")
args = parser.parse_args()

session_name = args.name
session_class = args.classes
session_test = args.test
session_nsensors = args.nsensors
session_samples = args.samples
session_frequency = args.frequency
session_note = args.note
session_delay = args.delay
# session_plot = args.plot
session_plot = False
session_median = args.median
session_gain = args.gain
session_live = args.live
save_offset = True
try:
    address = devices_params[args.address]['address']
except:
    address = args.address
# json_file = open(session_inference + '/training_summary.json', 'r')
# json_data = json.load(json_file)

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



if session_test:
    session_path = f'testset/{session_name}/{session_class}'
else:
    session_path = f'dataset/{session_name}/{session_class}'
if not os.path.isdir(session_path):
    try:
        os.makedirs(session_path, exist_ok=True)
    except OSError:
        print("Error in session creation ("+session_path+").")
        exit()

CONFIG_CHARACTERISTIC_UUID = "00000002-ffff-11e1-ac36-0002a5d5c51b"
SENSOR_CHARACTERISTIC_UUID = "00000000-0001-11e1-ac36-0002a5d5c51b" # OLD version "14000000-0001-11e1-ac36-0002a5d5c51b"

string_len = 16
progress_len = string_len * 2 - 1
gain = session_gain
z_ignore = False
newline = 0

list_cnt = 0
end = False
exit = False
session_countdown = True

x = []
y = []
z = []

for i in range(session_nsensors):
    x.append([0 for i in range(session_median)])
    y.append([0 for i in range(session_median)])
    z.append([0 for i in range(session_median)])


if session_nsensors == 2:
    scale = [1000 / (gain * string_len), 30000 / (gain * string_len)] # FIXME
else:
    scale = [1000 / (gain * string_len)]*session_nsensors

session_offsetx = args.offsetx
session_offsety = args.offsety
session_offsetz = args.offsetz

if isinstance(session_offsetx, list):
    if len(session_offsetx):
        if len(session_offsetx) != session_nsensors:
            session_offsetx = [session_offsetx[0]]*session_nsensors
    else:
        session_offsetx = [0]*session_nsensors
else:
    session_offsetx = [0]*session_nsensors

if isinstance(session_offsety, list):
    if len(session_offsety):
        if len(session_offsety) != session_nsensors:
            session_offsety = [session_offsety[0]]*session_nsensors
    else:
        session_offsety = [0]*session_nsensors
else:
    session_offsety = [0]*session_nsensors

if isinstance(session_offsetz, list):
    if len(session_offsetz):
        if len(session_offsetz) != session_nsensors:
            session_offsetz = [session_offsetz[0]]*session_nsensors
    else:
        session_offsetz = [0]*session_nsensors
else:
    session_offsetz = [0]*session_nsensors

try:
    temp_val = devices_params[args.address]
    session_offsetx[0] = devices_params[args.address]['acc_offset']['x']
    session_offsety[0] = devices_params[args.address]['acc_offset']['y']
    session_offsetz[0] = devices_params[args.address]['acc_offset']['z']
    session_offsetx[1] = devices_params[args.address]['gyro_offset']['x']
    session_offsety[1] = devices_params[args.address]['gyro_offset']['y']
    session_offsetz[1] = devices_params[args.address]['gyro_offset']['z']
except:
    pass

if save_offset:
    temp_session_offsetx = [0]*session_nsensors
    temp_session_offsety = [0]*session_nsensors
    temp_session_offsetz = [0]*session_nsensors
else:
    temp_session_offsetx = session_offsetx
    temp_session_offsety = session_offsety
    temp_session_offsetz = session_offsetz

history = {
    'class' : session_class,
    'samples' : session_samples,
    'frequency' : session_frequency,
    'offset' : {
        'x' : session_offsetx,
        'y' : session_offsety,
        'z' : session_offsetz
    },
    'datetime' : "",
    'note' : session_note,
    'data' : [] # {
    #     'x' : [],
    #     'y' : [],
    #     'z' : []
    # }
}

for i in range(session_nsensors):
    history['data'].append({
        'x' : [],
        'y' : [],
        'z' : []
    })

axes = True
ticks = False
grid = True

connected = False



# sample_rate = 333
# nyq_rate = sample_rate / 2.0
# width = 8/nyq_rate
# ripple_db = 15
# cutoff_hz = 1.0
#
# N, beta = kaiserord(ripple_db, width)
# # print(N)
# # exit()
# taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
#
# # filtered_x = lfilter(taps, 1.0, x)

def notification_handler(sender, data):
    try:
        global x
        global y
        global z
        global list_cnt
        global history
        global exit
        global end
        global session_countdown
        global session_nsensors
        global newline
        global string_len
        global progress_len
        global scales
        global gain
        global z_ignore

        # if False: # DEBUG
        #     byte_len = 2
        #     offset = 4
        #     X_t = 0
        #     X_t = offset + X_t*byte_len
        #     Y_t = 1
        #     Y_t = offset + Y_t*byte_len
        #     Z_t = 2
        #     Z_t = offset + Z_t*byte_len
        #
        #     print()
        #     print()
        #     print()
        #     print(data)
        #     print()
        #     print(int.from_bytes(data[X_t:X_t+byte_len], byteorder='little', signed=True))
        #     print(int.from_bytes(data[Y_t:Y_t+byte_len], byteorder='little', signed=True))
        #     print(int.from_bytes(data[Z_t:Z_t+byte_len], byteorder='little', signed=True))

        if session_countdown:
            pass

        elif session_live or list_cnt < session_samples:
            byte_len = 2
            index = 4

            # x.append(int.from_bytes(data[X_t:X_t+byte_len], byteorder='little', signed=True))
            # y.append(int.from_bytes(data[Y_t:Y_t+byte_len], byteorder='little', signed=True))
            # z.append(int.from_bytes(data[Z_t:Z_t+byte_len], byteorder='little', signed=True))
            # x.pop(0)
            # y.pop(0)
            # z.pop(0)

            for i in range(session_nsensors):
                x[i].append(int.from_bytes(data[index:index+byte_len], byteorder='little', signed=True))
                history['data'][i]['x'].append(int.from_bytes(data[index:index+byte_len], byteorder='little', signed=True))
                index += byte_len
                y[i].append(int.from_bytes(data[index:index+byte_len], byteorder='little', signed=True))
                history['data'][i]['y'].append(int.from_bytes(data[index:index+byte_len], byteorder='little', signed=True))
                index += byte_len
                z[i].append(int.from_bytes(data[index:index+byte_len], byteorder='little', signed=True))
                history['data'][i]['z'].append(int.from_bytes(data[index:index+byte_len], byteorder='little', signed=True))
                index += byte_len
                x[i].pop(0)
                y[i].pop(0)
                z[i].pop(0)

                if save_offset:
                    history['data'][i]['x'][-1] += session_offsetx[i]
                    history['data'][i]['y'][-1] += session_offsety[i]
                    history['data'][i]['z'][-1] += session_offsetz[i]


            # history['x'].append(statistics.median(x))
            # history['y'].append(statistics.median(y))
            # history['z'].append(statistics.median(z))

            # print(f"X: {int.from_bytes(data[X_t:X_t+len], byteorder='little', signed=True):4d}") # ({data[X]}, {data[X+len-1]}, {data[X:X+len]})")
            # print(f"Y: {int.from_bytes(data[Y_t:Y_t+len], byteorder='little', signed=True):4d}") # ({data[Y]}, {data[Y+len-1]}, {data[Y:Y+len]})")
            # print(f"Z: {int.from_bytes(data[Z_t:Z_t+len], byteorder='little', signed=True):4d}") # ({data[Z]}, {data[Z+len-1]}, {data[Z:Z+len]})")
            # print('')

            # print('')
            # print(f"X: {history['data']['x'][-1]}", end = '\r') # ({data[X]}, {data[X+len-1]}, {data[X:X+len]})")
            # print('\t\t', end = '')
            # print(f"Y: {history['data']['y'][-1]}", end = '\r') # ({data[Y]}, {data[Y+len-1]}, {data[Y:Y+len]})")
            # print('\t\t\t\t', end = '')
            # print(f"Z: {history['data']['z'][-1]}", end = '\r') # ({data[Y]}, {data[Y+len-1]}, {data[Y:Y+len]})")


            # print(f"X: |}", end = '\r') # ({data[X]}, {data[X+len-1]}, {data[X:X+len]})")

            while newline:
                print('', end = '\033[F')
                newline -= 1

            if not session_live:
                temp_val = round((list_cnt / session_samples) * progress_len)
                temp_string =  color.BLUE + '⧖ ' + '▮' * temp_val + ' ' * (progress_len - temp_val) + ' ⧖' + color.END
                print(f'{temp_string}\n')
                newline += 2

            # if True:
            for i in range(session_nsensors):
                if session_median > 0:
                    temp_val = min(string_len, max(-string_len, round((statistics.median(x[i]) + temp_session_offsetx[i]) / scale[i])))
                else:

                    temp_val = min(string_len, max(-string_len, round((history['data'][i]['x'][-1] + temp_session_offsetx[i]) / scale[i])))
                if temp_val > 0:
                    temp_string = color.GREEN + '|' + ' ' * string_len + 'X' + '▮' * temp_val + ' ' * (string_len - temp_val) + '|' + color.END
                else:
                    temp_string = color.GREEN + '|' + ' ' * (string_len + temp_val) + '▮' * -temp_val + 'X' + ' ' * string_len + '|' + color.END
                print(f'{temp_string}')
                newline += 1

                if session_median > 0:
                    temp_val = -min(string_len, max(-string_len, round((statistics.median(y[i]) + temp_session_offsety[i]) / scale[i])))
                else:
                    temp_val = -min(string_len, max(-string_len, round((history['data'][i]['y'][-1] + temp_session_offsety[i]) / scale[i])))
                if temp_val > 0:
                    temp_string = color.YELLOW + '|' + ' ' * string_len + 'Y' + '▮' * temp_val + ' ' * (string_len - temp_val) + '|' + color.END
                else:
                    temp_string = color.YELLOW + '|' + ' ' * (string_len + temp_val) + '▮' * -temp_val + 'Y' + ' ' * string_len + '|' + color.END
                print(f'{temp_string}')
                newline += 1

                if not z_ignore:
                    if session_median > 0:
                        temp_val = min(string_len, max(-string_len, round((statistics.median(z[i]) + temp_session_offsetz[i]) / scale[i])))
                    else:
                        temp_val = min(string_len, max(-string_len, round((history['data'][i]['z'][-1] + temp_session_offsetz[i]) / scale[i])))
                    if temp_val > 0:
                        temp_string = color.RED + '|' + ' ' * string_len + 'Z' + '▮' * temp_val + ' ' * (string_len - temp_val) + '|' + color.END
                    else:
                        temp_string = color.RED + '|' + ' ' * (string_len + temp_val) + '▮' * -temp_val + 'Z' + ' ' * string_len + '|' + color.END
                    print(f'{temp_string}')
                    newline += 1

                print(f'')
                newline += 1

            list_cnt += 1

        else:
            end = True

    except KeyboardInterrupt:
        exit = True
        session_countdown = True



async def run(address, debug=False):
    log = logging.getLogger(__name__)
    if debug:
        import sys

        log.setLevel(logging.DEBUG)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.DEBUG)
        log.addHandler(h)

    async with BleakClient(address) as client:
        log.info(f"Connected: {client.is_connected}")

        for service in client.services:
            log.info(f"[Service] {service}")
            for char in service.characteristics:
                if "read" in char.properties:
                    try:
                        value = bytes(await client.read_gatt_char(char.uuid))
                        log.info(
                            f"\t[Characteristic] {char} ({','.join(char.properties)}), Value: {value}"
                        )
                    except Exception as e:
                        log.error(
                            f"\t[Characteristic] {char} ({','.join(char.properties)}), Value: {e}"
                        )

                else:
                    value = None
                    log.info(
                        f"\t[Characteristic] {char} ({','.join(char.properties)}), Value: {value}"
                    )

                for descriptor in char.descriptors:
                    try:
                        value = bytes(
                            await client.read_gatt_descriptor(descriptor.handle)
                        )
                        log.info(f"\t\t[Descriptor] {descriptor}) | Value: {value}")
                    except Exception as e:
                        log.error(f"\t\t[Descriptor] {descriptor}) | Value: {e}")



async def notif(address, debug=False):
    if debug:
        import sys

        l = logging.getLogger("asyncio")
        l.setLevel(logging.DEBUG)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.DEBUG)
        l.addHandler(h)
        logger.addHandler(h)

    async with BleakClient(address) as client:
        logger.info(f"Connected: {client.is_connected}")

        await client.start_notify(SENSOR_CHARACTERISTIC_UUID, notification_handler)



async def conf(address, debug=False):
    log = logging.getLogger(__name__)
    global connected
    if debug:
        import sys

        log.setLevel(logging.DEBUG)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.DEBUG)
        log.addHandler(h)

    # while not connected:
    #     try:
    async with BleakClient(address) as client:
        global x
        global y
        global z
        global history
        global exit
        global end
        global list_cnt
        global session_delay
        global session_countdown

        # log.info(f"Connected: {client.is_connected}")

        print(f"{color.BOLD}Connected:{color.END} {client.is_connected}")
        print(f'{color.BOLD}Class:{color.END} {session_class}')
        print(f'{color.BOLD}Frequency:{color.END} {session_frequency} Hz')
        if not session_live:
            print(f'{color.BOLD}Duration:{color.END} {session_samples / session_frequency} sec')
            print(f'{color.BOLD}Samples:{color.END} {session_samples}')
        print('')

        connected = True

        # paired = await client.pair(protection_level=2)
        # log.info(f"Paired: {paired}")

        # await client.write_gatt_char(CONFIG_CHARACTERISTIC_UUID, b"\x0101")
        # await asyncio.sleep(1.0)

        await client.start_notify(SENSOR_CHARACTERISTIC_UUID, notification_handler)
        await client.write_gatt_char(CONFIG_CHARACTERISTIC_UUID, b"\x01\x01", response=False)

        if session_plot:
            dimension = 32
            diameter = 39.5
            height = 7.5

            pltangle = 2 * np.pi
            pltcamera = 8

            angle = np.linspace(0, pltangle, dimension)
            radius = np.linspace(0, diameter / 2, dimension)
            smallradius = np.linspace(0, diameter / 5, dimension)
            theta, phi = np.meshgrid(angle, angle)

            fig = plt.figure()
            ax = fig.gca(projection = '3d')
            ax.set_box_aspect((diameter, diameter, height))

            X = (smallradius * np.cos(phi))
            Y = (smallradius * np.sin(phi))
            Z = np.array([[(i / dimension) * (i / dimension) * height for i, x in enumerate(range(dimension))] for x in range(dimension)])
            ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1) #, color = 'w'

            X = (radius * np.cos(phi))
            Y = (radius * np.sin(phi))
            Z = np.array([[height for x in range(dimension)] for x in range(dimension)])
            ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.7)

            if not axes:
                plt.axis('off')

            if not ticks:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

            if grid:
                plt.grid(color='lightgray', linestyle='--')

            plt.tight_layout()

        while session_delay:
            print(f'Start in {session_delay} sec', end = '\r')
            await asyncio.sleep(1)
            session_delay -= 1

        session_countdown = False

        # while True:
        #     ax.view_init(math.sqrt(pow(statistics.median(x)/14, 2) + pow(statistics.median(y)/14, 2)), 30)
        #     plt.draw()
        #     plt.pause(.001)
        #     await asyncio.sleep(0.02)

        # await asyncio.sleep(duration)

        try:
            while True:
                if exit:
                    print('Session stopped')
                    break
                if end:
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    history['datetime'] = timestr
                    with open(f'{session_path}/{timestr}.json', 'w') as json_file:
                        json.dump(history, json_file, ensure_ascii=False, indent = 4) # , indent = 4
                    for i in range(session_nsensors):
                        print(f"{color.BOLD}Median X{i}:{color.END} {round(statistics.median(history['data'][i]['x']))}")
                        print(f"{color.BOLD}Median Y{i}:{color.END} {round(statistics.median(history['data'][i]['y']))}")
                        print(f"{color.BOLD}Median Z{i}:{color.END} {round(statistics.median(history['data'][i]['z']))}")
                    print(f'{color.BOLD}Timestamp:{color.END} {timestr}')
                    print(f'{color.BOLD}Session ended{color.END}')
                    break
                if session_plot:
                    ax.view_init(math.sqrt(pow(statistics.median(x)/14, 2) + pow(statistics.median(y)/14, 2)), 30)
                    plt.draw()
                    plt.pause(.001)
                    await asyncio.sleep(0.02)
                await asyncio.sleep(0.001)
        except KeyboardInterrupt:
            session_countdown = True
            print('Session stopped')
            pass

        # except:
        #     time.sleep(1)
        #     # await asyncio.sleep(1)

if __name__ == "__main__":
    print(f"{color.BOLD}Connected:{color.END}", end = '\r')
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    # loop.run_until_complete(run(address, True))
    # loop.run_until_complete(notif(address, True))
    # loop.run_until_complete(conf(address, True))

    try:
        loop.run_until_complete(conf(address, True))
    except KeyboardInterrupt:
        print('Session stopped')

print('')

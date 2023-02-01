# import sys
# import os
import time
# import json
import argparse

import math
import statistics
# import random

# import platform
import asyncio
import logging
import numpy as np

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

addresses = {
    'M0DEV' : 'C0:6E:33:30:41:4D',
    'M0' : 'C0:6E:38:30:41:4D',
    'M1' : 'C0:6E:38:30:41:4D', # NOT WORKING
    'M2' : 'C0:6E:29:31:3B:48',
    'M3' : 'C0:6E:25:31:20:48',
    'B0' : 'C0:6E:29:31:3B:48'
}



parser = argparse.ArgumentParser()
# parser.add_argument('-n','--name', dest='name', required=True, help="session name")
# parser.add_argument('-c','--class', dest='classes', required=True, help="class type")
parser.add_argument('-t','--test', dest='test', action='store_true', help='test mode')
parser.add_argument('-a','--address', dest='address', default=addresses['M0'], help="SensorTile MAC address")
# parser.add_argument('--calibration', dest='calibration', action='store_true', help="caloibration samples")
parser.add_argument('-s','--samples', dest='samples', default=6000, type=int, help="samples limit")
parser.add_argument('-f','--frequency', dest='frequency', default=100, type=int, help="samples frequency (Hz)")
parser.add_argument('-ox','--offsetx', dest='offsetx', default=0, type=int, help='X offset')
parser.add_argument('-oy','--offsety', dest='offsety', default=0, type=int, help='Y offset')
parser.add_argument('--note', dest='note', help="note")
# parser.add_argument('-d','--delay', dest='delay', default=3, type=int, help="countdown")
# parser.add_argument('-p','--plot', dest='plot', action='store_true', help='plot data')
parser.add_argument('-m','--median', dest='median', default=0, type=int, help='plot n median values')
parser.add_argument('-g','--gain', dest='gain', default=1/2, type=float, help='plot gain')
parser.add_argument('-l','--live', dest='live', action='store_true', help='only plot data')
parser.add_argument('-i','--inference', dest='inference', help="applies inference to results")
args = parser.parse_args()

# session_name = args.name
# session_class = args.classes
session_test = args.test
session_samples = args.samples
session_frequency = args.frequency
session_note = args.note
session_offsetx = args.offsetx
session_offsety = args.offsety
# session_delay = args.delay
# session_plot = args.plot
session_plot = False
session_median = args.median
session_gain = args.gain
session_live = args.live
try:
    address = addresses[args.address]
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



# if session_test:
#     session_path = f'testset/{session_name}/{session_class}'
# else:
#     session_path = f'dataset/{session_name}/{session_class}'
# if not os.path.isdir(session_path):
#     try:
#         os.makedirs(session_path, exist_ok=True)
#     except OSError:
#         print("Error in session creation ("+session_path+").")
#         exit()

CONFIG_CHARACTERISTIC_UUID = "00000002-ffff-11e1-ac36-0002a5d5c51b"
SENSOR_CHARACTERISTIC_UUID = "02000000-0001-11e1-ac36-0002a5d5c51b" # OLD version "14000000-0001-11e1-ac36-0002a5d5c51b"

string_len = 16
progress_len = string_len * 2 - 1
gain = session_gain
scale = 1000 / (gain * string_len)
z_ignore = False
newline = 0

list_cnt = 0
session_end = False
session_exit = False
# session_countdown = False

x = [0 for i in range(session_median)]
y = [0 for i in range(session_median)]
z = [0 for i in range(session_median)]

history = {
    # 'class' : session_class,
    # 'samples' : session_samples,
    # 'frequency' : session_frequency,
    'datetime' : "",
    'note' : session_note,
    'data' : [],
    'data_voting' : []
}

plot_item = {
    1:f'{color.END}▮{color.END}',
    2:f'{color.RED}▮{color.END}',
    3:f'{color.BLUE}▮{color.END}'
}


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
        global plot_item
        global session_exit
        global session_end
        # global session_countdown
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

        # if session_countdown:
        #     pass

        if True or session_live or list_cnt < session_samples:
            byte_len = 2
            offset = 4
            X_t = 0
            X_t = offset + X_t*byte_len
            Y_t = 1
            Y_t = offset + Y_t*byte_len
            Z_t = 2
            Z_t = offset + Z_t*byte_len

            history['data'].append(int.from_bytes(data[X_t:X_t+byte_len], byteorder='little', signed=True))
            # y.append(int.from_bytes(data[Y_t:Y_t+byte_len], byteorder='little', signed=True))
            # z.append(int.from_bytes(data[Z_t:Z_t+byte_len], byteorder='little', signed=True))
            # x.pop(0)
            # y.pop(0)
            # z.pop(0)

            # history['data']['x'].append(int.from_bytes(data[X_t:X_t+byte_len], byteorder='little', signed=True))
            # history['data']['y'].append(int.from_bytes(data[Y_t:Y_t+byte_len], byteorder='little', signed=True))
            # history['data']['z'].append(int.from_bytes(data[Z_t:Z_t+byte_len], byteorder='little', signed=True))
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

            # if True:
                # while newline:
                #     print('', end = '\033[F')
                #     newline -= 1

                # for d in history['data']:
                #     print(plot_item[d], end = '')
                # print('')
                # newline += 1

            # print('CIAOOOOOOOOOo',end='', flush=True)
            print(plot_item[history['data'][-1]], end = '', flush = True)
            
                # if not session_live:
                #     temp_val = round((list_cnt / session_samples) * progress_len)
                #     temp_string =  color.BLUE + '⧖ ' + '▮' * temp_val + ' ' * (progress_len - temp_val) + ' ⧖' + color.END
                #     print(f'{temp_string}\n')
                #     newline += 2

                # if session_median > 0:
                #     temp_val = min(string_len, max(-string_len, round((statistics.median(x) + session_offsetx) / scale)))
                # else:
                    # temp_val = min(string_len, max(-string_len, round((history['data']['x'][-1] + session_offsetx) / scale)))
                # if temp_val > 0:
                #     temp_string = color.GREEN + '|' + ' ' * string_len + 'X' + '▮' * temp_val + ' ' * (string_len - temp_val) + '|' + color.END
                # else:
                #     temp_string = color.GREEN + '|' + ' ' * (string_len + temp_val) + '▮' * -temp_val + 'X' + ' ' * string_len + '|' + color.END
                # print(f'{temp_string}')
                # newline += 1

                # if session_median > 0:
                #     temp_val = -min(string_len, max(-string_len, round((statistics.median(y) + session_offsety) / scale)))
                # else:
                #     temp_val = -min(string_len, max(-string_len, round((history['data']['y'][-1] + session_offsety) / scale)))
                # if temp_val > 0:
                #     temp_string = color.YELLOW + '|' + ' ' * string_len + 'Y' + '▮' * temp_val + ' ' * (string_len - temp_val) + '|' + color.END
                # else:
                #     temp_string = color.YELLOW + '|' + ' ' * (string_len + temp_val) + '▮' * -temp_val + 'Y' + ' ' * string_len + '|' + color.END
                # print(f'{temp_string}')
                # newline += 1

                # if not z_ignore:
                #     if session_median > 0:
                #         temp_val = min(string_len, max(-string_len, round(statistics.median(z) / scale)))
                #     else:
                #         temp_val = min(string_len, max(-string_len, round(history['data']['z'][-1] / scale)))
                #     if temp_val > 0:
                #         temp_string = color.RED + '|' + ' ' * string_len + 'Z' + '▮' * temp_val + ' ' * (string_len - temp_val) + '|' + color.END
                #     else:
                #         temp_string = color.RED + '|' + ' ' * (string_len + temp_val) + '▮' * -temp_val + 'Z' + ' ' * string_len + '|' + color.END
                #     print(f'{temp_string}')
                #     newline += 1

                # print(f'')
                # newline += 1

            list_cnt += 1

        else:
            session_end = True

    except KeyboardInterrupt:
        session_exit = True
        # session_countdown = True



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
        global session_exit
        global session_end
        global list_cnt
        # global session_delay
        # global session_countdown

        # log.info(f"Connected: {client.is_connected}")

        print(f"{color.BOLD}Connected:{color.END} {client.is_connected}")
        print(f"{color.BOLD}\nLegend\n{color.END}Garbage: ▮\r\t\t\t{color.RED}Squat: ▮\r\t\t\t\t\t\t{color.BLUE}Push-ups: ▮{color.END}")
        # print(f'{color.BOLD}Class:{color.END} {session_class}')
        # print(f'{color.BOLD}Frequency:{color.END} {session_frequency} Hz')
        # if not session_live:
        #     print(f'{color.BOLD}Duration:{color.END} {session_samples / session_frequency} sec')
        #     print(f'{color.BOLD}Samples:{color.END} {session_samples}')
        print('')

        connected = True

        # paired = await client.pair(protection_level=2)
        # log.info(f"Paired: {paired}")

        # await client.write_gatt_char(CONFIG_CHARACTERISTIC_UUID, b"\x0101")
        # await asyncio.sleep(1.0)

        await client.start_notify(SENSOR_CHARACTERISTIC_UUID, notification_handler)
        await client.write_gatt_char(CONFIG_CHARACTERISTIC_UUID, b"\x01\x03", response=False)

        # if session_plot:
        #     dimension = 32
        #     diameter = 39.5
        #     height = 7.5

        #     pltangle = 2 * np.pi
        #     pltcamera = 8

        #     angle = np.linspace(0, pltangle, dimension)
        #     radius = np.linspace(0, diameter / 2, dimension)
        #     smallradius = np.linspace(0, diameter / 5, dimension)
        #     theta, phi = np.meshgrid(angle, angle)

        #     fig = plt.figure()
        #     ax = fig.gca(projection = '3d')
        #     ax.set_box_aspect((diameter, diameter, height))

        #     X = (smallradius * np.cos(phi))
        #     Y = (smallradius * np.sin(phi))
        #     Z = np.array([[(i / dimension) * (i / dimension) * height for i, x in enumerate(range(dimension))] for x in range(dimension)])
        #     ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1) #, color = 'w'

        #     X = (radius * np.cos(phi))
        #     Y = (radius * np.sin(phi))
        #     Z = np.array([[height for x in range(dimension)] for x in range(dimension)])
        #     ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.7)

        #     if not axes:
        #         plt.axis('off')

        #     if not ticks:
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         ax.set_zticks([])

        #     if grid:
        #         plt.grid(color='lightgray', linestyle='--')

        #     plt.tight_layout()

        # while session_delay:
        #     print(f'Start in {session_delay} sec', end = '\r')
        #     await asyncio.sleep(1)
        #     session_delay -= 1

        # session_countdown = False

        # while True:
        #     ax.view_init(math.sqrt(pow(statistics.median(x)/14, 2) + pow(statistics.median(y)/14, 2)), 30)
        #     plt.draw()
        #     plt.pause(.001)
        #     await asyncio.sleep(0.02)

        # await asyncio.sleep(duration)

        try:
            while True:
                
                # main

                if session_exit:
                    print('Session stopped (exit)')
                    break
                if session_end:
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    # history['datetime'] = timestr
                    # with open(f'{session_path}/{timestr}.json', 'w') as json_file:
                    #     json.dump(history, json_file, ensure_ascii=False, indent = 4) # , indent = 4
                    # print(f"{color.BOLD}Median X:{color.END} {round(statistics.median(history['data']['x']))}")
                    # print(f"{color.BOLD}Median Y:{color.END} {round(statistics.median(history['data']['y']))}")
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
            # session_countdown = True
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

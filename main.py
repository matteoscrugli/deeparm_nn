import sys
import numpy as np


np.set_printoptions(threshold=sys.maxsize)

type = 'NORMAL'

a = np.asarray([[-30.5618133545,  39.0906906128, -23.4544143677]])

scale = 0.7107398509979248


if type == 'NORMAL':
  for item in a.flatten():
    if scale == 1:
      print(item, end = ', ')
    else:
      print(round(item/scale), end = ', ')



if type == 'INPUT':
  for item in a.flatten('F'):
    if scale == 1:
      print(item, end = ', ')
    else:
      print(round(item/scale), end = ', ')



if type == 'CONV1':
  for item in a.flatten('F'):
    if scale == 1:
      print(item, end = ', ')
    else:
      print(round(item/scale), end = ', ')



if type == 'FC1':
  a = a.reshape(100,320)
  
  for i in range(len(a)):
    a[i] = a[i].reshape(20,16).flatten('F')
  for item in a.flatten():
    if scale == 1:
      print(item, end = ', ')
    else:
      print(round(item/scale), end = ', ')

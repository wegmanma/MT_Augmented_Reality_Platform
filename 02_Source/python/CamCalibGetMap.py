import numpy as np
import csv
import png
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [27, 14]
mpl.rcParams['figure.dpi'] = 72
from matplotlib import pyplot as plt
import cv2 as cv
from functions import *
np.set_printoptions(threshold=np.inf)

# Resolution of ToF Camera: 352 x 286
XChan = np.zeros((286,352),dtype=np.int16)
YChan = np.zeros((286,352),dtype=np.int16)
for i in range(286):
    for j in range(352):
        XChan[i,j] = j
        YChan[i,j] = i

XChan = XChan.reshape((100672,1))
YChan = YChan.reshape((100672,1))

print(XChan.shape)
text = ''
for i in range(100672):
    text += str(XChan[i,0])+';'
remaining_string = text.rstrip(text[-1])
with open('calib_xchan.txt', 'w') as f:
    f.write(remaining_string)
text = ''
for i in range(100672):
    text += str(YChan[i,0])+';'
remaining_string = text.rstrip(text[-1])
with open('calib_ychan.txt', 'w') as f:
    f.write(remaining_string)

x_corr = load_n_undistort_int16("calib_xchan.txt")
y_corr = load_n_undistort_int16("calib_ychan.txt")

print(type(x_corr[0,0]))
print(x_corr.shape)

filename = "../data/x_corr_ToF.dat"
fileobj = open(filename, mode='wb')
x_corr.tofile(fileobj)
fileobj.close

filename = "../data/y_corr_ToF.dat"
fileobj = open(filename, mode='wb')
y_corr.tofile(fileobj)
fileobj.close

text = str(x_corr)
with open('corr_calib_xchan.txt', 'w') as f:
    f.write(text)
text = str(y_corr)
with open('corr_calib_ychan.txt', 'w') as f:
    f.write(text)
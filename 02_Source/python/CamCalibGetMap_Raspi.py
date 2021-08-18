import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import csv
import png
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [27, 14]
mpl.rcParams['figure.dpi'] = 72

undistort_mtx = np.array([[1.30269281e+03, 0.00000000e+00, 6.29842386e+02],
 [0.00000000e+00, 1.29407977e+03, 3.59106501e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


undistort_newcameramtx = np.array([[1.32807019e+03, 0.00000000e+00 ,6.26613794e+02],
 [0.00000000e+00, 1.30680530e+03, 3.59702855e+02],
 [0.00000000e+00, 0.00000000e+00 ,1.00000000e+00]])

undistort_dist = np.array(
    [[ 0.16106655 ,-0.11948599,  0.00374535, -0.00232904 ,-0.54201863]])

undistort_roi = (3, 4, 1273, 709)


def load_n_undistort_int16(dataraw):

    minimum = np.amin(dataraw)
    maximum = np.amax(dataraw)
    print("maximum: " +str(maximum))
    print("minimum: " +str(minimum))
    dataraw = dataraw.astype(np.uint16).reshape((720, 1280))
    dataRGB = np.zeros((720*1280*3), dtype=float).reshape((720, 1280, 3))
    dataRGB[:, :, 0] = (((dataraw[:, :].astype(float)) /
                        maximum).astype(float)).astype(float)
    dataRGB[:, :, 1] = (((dataraw[:, :].astype(float)) /
                        maximum).astype(float)).astype(float)
    dataRGB[:, :, 2] = (((dataraw[:, :].astype(float)) /
                        maximum).astype(float)).astype(float)
    dst = cv.undistort(dataRGB, undistort_mtx, undistort_dist,
                       None, undistort_newcameramtx)
    print("dataRGB Value: "+str(dataRGB[0,0,0]))
    x, y, w, h = undistort_roi
    src = dst[y:y+h, x:x+w]
    ret = np.zeros((709, 1273), dtype=np.uint16)
    print("Starting for loop")
    print("src Value: "+str(src[0,0]))
    for i in range(709):
        for j in range(1273):
            ret[i, j] = (src[i, j, 0]*(maximum)).astype(np.uint16)
    print("ret Value: "+str(ret[0,0]))
    return ret


np.set_printoptions(threshold=np.inf)

XChan = np.zeros((720, 1280), dtype=np.int16)
YChan = np.zeros((720, 1280), dtype=np.int16)
for i in range(720):
    for j in range(1280):
        XChan[i, j] = j
        YChan[i, j] = i

XChan = XChan.reshape((1280*720, 1))
YChan = YChan.reshape((1280*720, 1))

print(XChan.shape)

x_corr = load_n_undistort_int16(XChan)
y_corr = load_n_undistort_int16(YChan)

print(type(x_corr[0, 0]))
print(x_corr.shape)

filename = "../data/x_corr_Raspi.dat"
fileobj = open(filename, mode='wb')
x_corr.tofile(fileobj)
fileobj.close

filename = "../data/y_corr_Raspi.dat"
fileobj = open(filename, mode='wb')
y_corr.tofile(fileobj)
fileobj.close

text = str(x_corr)
with open('corr_calib_xchan_raspi.txt', 'w') as f:
    f.write(text)
text = str(y_corr)
with open('corr_calib_ychan_raspi.txt', 'w') as f:
    f.write(text)

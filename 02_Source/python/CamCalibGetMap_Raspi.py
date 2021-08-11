import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import csv
import png
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [27, 14]
mpl.rcParams['figure.dpi'] = 72

undistort_mtx = np.array([[2.70698401e+03,   0.00000000e+00,   9.71379742e+02],
                          [0.00000000e+00,  2.69656522e+03,   5.21649973e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


undistort_newcameramtx = np.array([[2.76536255e+03, 0.00000000e+00,  9.74480810e+02],
                                   [0.00000000e+00, 2.72011548e+03,  5.22169397e+02],
                                   [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

undistort_dist = np.array(
    [[2.45000445e-01, -6.42516377e-01,   7.64725607e-04,  3.61062798e-03,
        8.36154477e-01]])

undistort_roi = (4, 10, 1911, 1060)


def load_n_undistort_int16(dataraw):

    minimum = np.amin(dataraw)
    maximum = np.amax(dataraw)
    print("maximum: " +str(maximum))
    print("minimum: " +str(minimum))
    dataraw = dataraw.astype(np.uint16).reshape((1080, 1920))
    dataRGB = np.zeros((1920*1080*3), dtype=float).reshape((1080, 1920, 3))
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
    ret = np.zeros((1060, 1911), dtype=np.uint16)
    print("Starting for loop")
    print("src Value: "+str(src[0,0]))
    for i in range(1060):
        for j in range(1911):
            ret[i, j] = (src[i, j, 0]*(maximum)).astype(np.uint16)
    print("ret Value: "+str(ret[0,0]))
    return ret


np.set_printoptions(threshold=np.inf)

XChan = np.zeros((1080, 1920), dtype=np.int16)
YChan = np.zeros((1080, 1920), dtype=np.int16)
for i in range(1080):
    for j in range(1920):
        XChan[i, j] = j
        YChan[i, j] = i

XChan = XChan.reshape((1920*1080, 1))
YChan = YChan.reshape((1920*1080, 1))

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

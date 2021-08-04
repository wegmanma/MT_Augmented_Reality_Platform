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

ampl = load_n_undistort("../data/ToFData/ampl_flat_roof_diffusor_16.txt")
width, height, channels = ampl.shape
num_images = 19
radial_arr = np.zeros((width, height, num_images),dtype=float)
filenameprefix = "../data/ToFData/radial_flat_roof_diffusor_"
filenamesuffix = ".txt"
for i in range(num_images):
    print(i)
    filename = filenameprefix+str(i)+filenamesuffix
    radial_arr[:,:,i] = load_n_undistort_int16(filename)
    radial_arr[:,:,i] = cv.GaussianBlur(radial_arr[:,:,i],(0,0),4)
radial = np.mean(radial_arr, axis=2)

print(type(radial[0,0]))
print(radial.shape)

cos_a = np.zeros(radial.shape,dtype=np.float32)
for m in range(width):
    for n in range(height):
        cos_a[m,n] = radial[100,124]/radial[m,n]

filename = "../data/cos_alpha_ToF.dat"
fileobj = open(filename, mode='wb')
cos_a.tofile(fileobj)
fileobj.close

text = str(cos_a)
with open('cosAlpha_calib_xchan.txt', 'w') as f:
    f.write(text)

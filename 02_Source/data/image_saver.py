import numpy as np
import csv
import png
from matplotlib import pyplot as plt
import cv2 as cv


filename = './ampl_chessboard_'
suffix_open = '.txt'
suffix_save = '.jpg'

for i in range(16):
    filenameopen = filename+str(i)+suffix_open
    filenamesave = filename+str(i)+suffix_save
    with open(filenameopen, newline='') as csvfile:
        data = list(csv.reader(csvfile,  delimiter=';'))

    dataraw = np.array(data)
    dataraw = dataraw.astype(np.uint16).reshape((286,352))
    dataRGB = np.zeros((286*352*3),dtype=float).reshape((286,352,3))
    dataRGB[:,:,0] = ((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)
    dataRGB[:,:,1] = ((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)
    dataRGB[:,:,2] = ((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)
    print(np.amax(dataRGB))
    plt.imsave(filenamesave,dataRGB)
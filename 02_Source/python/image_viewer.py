import numpy as np
import csv
import png
from PIL import Image 

with open('../software/ampl.txt', newline='') as csvfile:
    data = list(csv.reader(csvfile,  delimiter=';'))

arr = np.array(data)
arr = arr.astype(np.uint16)
print(np.amax(arr))
print(np.amin(arr))
arr[0,:] = (((arr[0,:]-np.amin(arr).astype(float)))/(np.amax(arr)-np.amin(arr))*65535).astype(np.uint16)
arr = arr.reshape((286,352))
print(np.amax(arr))
print(np.amin(arr))
arr_larger = np.zeros((286*4,352*4),dtype=np.uint16) 
for i in range(286):
    for j in range(352):
        arr_larger[i*4][j*4] =arr[i][j]
        arr_larger[i*4+1][j*4] =arr[i][j]
        arr_larger[i*4+2][j*4] =arr[i][j]
        arr_larger[i*4+3][j*4] =arr[i][j]
        arr_larger[i*4][j*4+1] =arr[i][j]
        arr_larger[i*4+1][j*4+1] =arr[i][j]
        arr_larger[i*4+2][j*4+1] =arr[i][j]
        arr_larger[i*4+3][j*4+1] =arr[i][j]
        arr_larger[i*4][j*4+2] =arr[i][j]
        arr_larger[i*4+1][j*4+2] =arr[i][j]
        arr_larger[i*4+2][j*4+2] =arr[i][j]
        arr_larger[i*4+3][j*4+2] =arr[i][j]
        arr_larger[i*4][j*4+3] =arr[i][j]
        arr_larger[i*4+1][j*4+3] =arr[i][j]
        arr_larger[i*4+2][j*4+3] =arr[i][j]
        arr_larger[i*4+3][j*4+3] =arr[i][j]
image = Image.fromarray(arr_larger)
image.show() 
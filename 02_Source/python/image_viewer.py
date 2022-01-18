import numpy as np
import csv
import png
from PIL import Image

for i in range(50):
    filenametxt = 'D:\\UserFolders\\OneDrive\\Dokumente\\MT_wegr\\demoImages\DemoImages\\ampl_corr_256_205_demo_image_'+str(i)+'.txt'
    filenamepng = 'D:\\UserFolders\\OneDrive\\Dokumente\\MT_wegr\\demoImages\DemoImages\\ampl_corr_256_205_demo_image_'+str(i)+'.png'
    with open(filenametxt, newline='') as csvfile:
        data = list(csv.reader(csvfile,  delimiter=';'))

    width = 256
    height = 205
    arr = np.array(data)
    arr = arr.astype(np.uint16)
    print(np.amax(arr))
    print(np.amin(arr))
    arr[0, :] = (((arr[0, :]-np.amin(arr).astype(float))) /
                 (np.amax(arr)-np.amin(arr))*65535).astype(np.uint16)
    arr = arr.reshape((height, width))
    print(np.amax(arr))
    print(np.amin(arr))
    arr_larger = np.zeros((height*4, width*4), dtype=np.uint16)
    for i in range(height):
        for j in range(width):
            arr_larger[i*4][j*4] = arr[i][j]
            arr_larger[i*4+1][j*4] = arr[i][j]
            arr_larger[i*4+2][j*4] = arr[i][j]
            arr_larger[i*4+3][j*4] = arr[i][j]
            arr_larger[i*4][j*4+1] = arr[i][j]
            arr_larger[i*4+1][j*4+1] = arr[i][j]
            arr_larger[i*4+2][j*4+1] = arr[i][j]
            arr_larger[i*4+3][j*4+1] = arr[i][j]
            arr_larger[i*4][j*4+2] = arr[i][j]
            arr_larger[i*4+1][j*4+2] = arr[i][j]
            arr_larger[i*4+2][j*4+2] = arr[i][j]
            arr_larger[i*4+3][j*4+2] = arr[i][j]
            arr_larger[i*4][j*4+3] = arr[i][j]
            arr_larger[i*4+1][j*4+3] = arr[i][j]
            arr_larger[i*4+2][j*4+3] = arr[i][j]
            arr_larger[i*4+3][j*4+3] = arr[i][j]
    image = Image.fromarray(arr_larger)
    image.save(filenamepng)

import cv2 as cv
import numpy as np
import csv
import sys

undistort_mtx = np.array([[292.00435998,   0.,         184.46231254],
 [  0.,         290.13777429, 145.5500931 ],
 [  0.,           0.,           1.        ]])

undistort_newcameramtx = np.array([[219.46858215,   0.,         184.57709806],
 [  0.,         207.91963196, 150.10088495],
 [  0.,           0. ,          1. ,       ]])

undistort_dist = np.array([[-0.35620774,  0.35086004, -0.00051231,  0.0015257,  -0.38282677]])

undistort_roi = (46,46,265,205)

sys.setrecursionlimit(100000)

def load_n_undistort(filename):
    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile,  delimiter=';'))
        
    dataraw = np.array(data)
    dataraw = dataraw.astype(np.uint16).reshape((286,352))
    dataRGB = np.zeros((286*352*3),dtype=float).reshape((286,352,3))
    dataRGB[:,:,0] = (((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)).astype(float)
    dataRGB[:,:,1] = (((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)).astype(float)
    dataRGB[:,:,2] = (((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)).astype(float)
    dst = cv.undistort(dataRGB, undistort_mtx, undistort_dist, None, undistort_newcameramtx)
    x, y, w, h = undistort_roi
    src = dst[y:y+h, x:x+w]
    return src

def load_n_undistort_int16(filename):
    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile,  delimiter=';'))
        
    dataraw = np.array(data)
    dataraw = dataraw.astype(np.uint16).reshape((286,352))
    dataRGB = np.zeros((286*352*3),dtype=float).reshape((286,352,3))
    dataRGB[:,:,0] = (((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)).astype(float)
    dataRGB[:,:,1] = (((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)).astype(float)
    dataRGB[:,:,2] = (((dataraw[:,:]-np.amin(dataraw).astype(float))/(np.amax(dataraw)-np.amin(dataraw))).astype(float)).astype(float)
    dst = cv.undistort(dataRGB, undistort_mtx, undistort_dist, None, undistort_newcameramtx)
    x, y, w, h = undistort_roi
    src = dst[y:y+h, x:x+w]
    ret = np.zeros((205,265),dtype=np.uint16)
    for i in range(205):
        for j in range(265):
            ret[i,j] = (src[i,j,0]*(np.amax(dataraw)-np.amin(dataraw))+np.amin(dataraw)).astype(np.uint16)
    
    return ret

def sobel_filter_2D(src):
    sobel_x = np.array([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]])
    sobel_y = np.array([[-1., -2., -1.],
                        [0., 0., 0.],
                        [1., 2., 1.]])
    if len(src.shape)==3:
        width, height, channels = src.shape
    elif len(src.shape)==2:
        width, height = src.shape
        channels = 1
    else:
        return
    srcfloat = np.zeros(src.shape,dtype=float)
    if (np.amax(src)>1.0):
        for c in range(channels):
            for x in range(width):
                for y in range(height):
                    srcfloat[x,y,c] = float(src[x,y,c]/255) 
    else:
        srcfloat = np.copy(src) 
    dst_x = np.zeros(src.shape,dtype=float)
    dst_y = np.zeros(src.shape,dtype=float)
    dst = np.zeros(src.shape,dtype=float)
    dst_x = cv.filter2D(srcfloat,-1,sobel_x)
    dst_y = cv.filter2D(srcfloat,-1,sobel_y)
    ori = np.zeros(src.shape,dtype=float)
    for c in range(channels):
        for x in range(width):
            for y in range(height):
                dst[x,y,c] = dst_x[x,y,c]*dst_x[x,y,c]+dst_y[x,y,c]*dst_y[x,y,c]
                if dst_x[x,y,c]==0.0:
                    dst_x[x,y,c] = 0.000000001
                ori[x,y,c] = np.arctan(dst_y[x,y,c]/dst_x[x,y,c])


    return dst, ori

def laplace_filter_2D(src):
    laplace = np.array([[-1., -1., -1.],
                        [-1., 8., -1],
                        [-1., -1., -1]])
    if len(src.shape)==3:
        width, height, channels = src.shape
    elif len(src.shape)==2:
        width, height = src.shape
        channels = 1
    else:
        return
    dst = np.zeros(src.shape,dtype=float)
    dst = cv.filter2D(src,-1,laplace)
    return dst

def sub_images_2D(src_pos, src_neg):
    if src_pos.shape != src_neg.shape:
        return
    if len(src_pos.shape)==3:
        width, height, channels = src_pos.shape
    elif len(src_pos.shape)==2:
        width, height = src_pos.shape
        channels = 1
    else:
        return
    dst = np.zeros(src_pos.shape,dtype=float)
    for c in range(channels):
        for x in range(width):
            for y in range(height):
                dst[x,y,c] = src_pos[x,y,c]-src_neg[x,y,c]
    return dst

def sq_sum_images_2D(src_pos, src_neg):
    if src_pos.shape != src_neg.shape:
        return
    if len(src_pos.shape)==3:
        width, height, channels = src_pos.shape
    elif len(src_pos.shape)==2:
        width, height = src_pos.shape
        channels = 1
    else:
        return
    dst = np.zeros(src_pos.shape,dtype=float)
    for c in range(channels):
        for x in range(width):
            for y in range(height):
                dst[x,y,c] = np.sqrt(src_pos[x,y,c]*src_pos[x,y,c]+src_neg[x,y,c]*src_neg[x,y,c])
    return dst

def sum_images_2D(src_pos, src_neg):
    if src_pos.shape != src_neg.shape:
        return
    if len(src_pos.shape)==3:
        width, height, channels = src_pos.shape
        dst = np.zeros(src_pos.shape,dtype=float)
        for c in range(channels):
            for x in range(width):
                for y in range(height):
                    dst[x,y,c] = src_pos[x,y,c]+src_neg[x,y,c]
        return dst
    elif len(src_pos.shape)==2:
        width, height = src_pos.shape
        dst = np.zeros(src_pos.shape,dtype=float)
        for x in range(width):
            for y in range(height):
                dst[x,y] = src_pos[x,y]+src_neg[x,y]
        return dst
    else:
        return

    

def mul_images_2D(src1, src2):
    if src1.shape != src2.shape:
        return
    if len(src1.shape)==3:
        width, height, channels = src1.shape
    elif len(src1.shape)==2:
        width, height = src1.shape
        channels = 1
    else:
        return
    dst = np.zeros(src1.shape,dtype=float)
    for c in range(channels):
        for x in range(width):
            for y in range(height):
                dst[x,y,c] = src1[x,y,c]*src2[x,y,c]
    return dst

def boost_image_2D(src1):
    if len(src1.shape)==3:
        width, height, channels = src1.shape
        dst = np.zeros(src1.shape,dtype=float)
        maximum = np.amax(src1)
        minimum = np.amin(src1)
        for c in range(channels):
            for x in range(width):
                for y in range(height):
                    dst[x,y,c] = (src1[x,y,c]-minimum)/(maximum-minimum)
        return dst
    elif len(src1.shape)==2:
        width, height = src1.shape
        dst = np.zeros(src1.shape,dtype=float)
        maximum = np.amax(src1)
        minimum = np.amin(src1)        
        for x in range(width):
            for y in range(height):
                dst[x,y] = (src1[x,y]-minimum)/(maximum-minimum)
        return dst
    else:
        return
    


def x_in_range(x):
        return (x>=0) and (x<205)
def y_in_range(y):
        return (y>=0) and (y<265)

def fillArea(x,y,src1, brightness):
    if len(src1.shape)==3:
        width, height, channels = src1.shape
    elif len(src1.shape)==2:
        width, height = src1.shape
        channels = 1
    else:
        return
    dst_tmp = np.zeros((width, height), dtype=np.uint8)
    dst = np.zeros(src1.shape, dtype=float)
    mask = np.zeros((width+2, height+2), dtype=np.uint8)
    maximum = np.amax(src1)
    minimum = np.amin(src1)
    for c in range(channels):
        for m in range(width):
            for n in range(height):
                dst_tmp[m,n] = np.uint8(((src1[m,n,c]-minimum)/(maximum-minimum))*255)
    cv.floodFill(dst_tmp,mask,seedPoint=(x,y),newVal=255,upDiff=brightness,loDiff=brightness, flags=cv.FLOODFILL_FIXED_RANGE, )

    num = 0
    sum = 0
    mean = 0.0

    for m in range(width):
       for n in range(height):
            if (mask[m+1,n+1] > 0):
                sum += src1[m,n,0]
                num += 1
    mean = sum/num
    dst_tmp = np.zeros((width, height), dtype=np.uint8)
    maximum = np.amax(src1)
    minimum = np.amin(src1)
    for c in range(channels):
        for m in range(width):
            for n in range(height):
                dst_tmp[m,n] = np.uint8(((src1[m,n,c]-minimum)/(maximum-minimum))*255)
    dst_tmp[x,y] = mean
    cv.floodFill(dst_tmp,mask,seedPoint=(x,y),newVal=255,upDiff=brightness,loDiff=brightness, flags=cv.FLOODFILL_FIXED_RANGE, )
    maximum = np.amax(dst_tmp)
    minimum = np.amin(dst_tmp)
    for c in range(channels):
        for m in range(width):
            for n in range(height):
                dst[m,n,c] = float(((dst_tmp[m,n]-minimum)/(maximum-minimum))*(1.0-mask[m+1,n+1]))
    num = 0
    sum = 0
    mean = 0.0

    for m in range(width):
       for n in range(height):
            if (mask[m+1,n+1] > 0):
                sum += src1[m,n,0]
                num += 1
    mean = sum/num
    print("num="+str(num))
    return dst, mask

def findDominantLine(mask):
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    grad_x = cv.Sobel(mask, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(mask, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)  
    kernelmatrix = np.ones((7, 7), np.uint8)
    grad = cv.dilate(grad,kernelmatrix)
    grad = cv.erode(grad,kernelmatrix)
    grad = cv.dilate(grad,kernelmatrix)
    grad = cv.erode(grad,kernelmatrix)
    grad = boost_image_2D(grad)
    return grad  

def load_n_undistort_int(name, histoeq=True):
    im1_orig = load_n_undistort(name)
    if len(im1_orig.shape)==3:
        width, height, channels = im1_orig.shape
    elif len(im1_orig.shape)==2:
        width, height = im1_orig.shape
        channels = 1


    im1_int = np.zeros(im1_orig.shape,dtype=np.uint8)
    for c in range(channels):
        for m in range(width):
            for n in range(height):
                im1_int[m,n,c] = np.uint8(im1_orig[m,n,c]*255)
    
    im1_histoeq_int = np.zeros(im1_orig.shape,dtype=np.uint8)
    if histoeq:
        im1_histoeq_int[:,:,0] = cv.equalizeHist(im1_int[:,:,0])
        im1_histoeq_int[:,:,1] = cv.equalizeHist(im1_int[:,:,1])
        im1_histoeq_int[:,:,2] = cv.equalizeHist(im1_int[:,:,2])
    else:
        im1_histoeq_int = np.copy(im1_int)
    boost_image_2D(im1_histoeq_int)
    return im1_histoeq_int

def matchSiftFeatures(im1_histoeq_int, im2_histoeq_int, image_to_show, image_to_show_2):
    siftobject = cv.SIFT_create()
    im1_keypoint, im1_descriptor = siftobject.detectAndCompute(im1_histoeq_int[:,:,0], None)
    im2_keypoint, im2_descriptor = siftobject.detectAndCompute(im2_histoeq_int[:,:,0], None)
    #drawing the keypoints and orientation of the keypoints in the image and then displaying the image as the output on the screen
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(im1_descriptor,im2_descriptor,k=2)
    good = []
    sum = 0
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            sum += np.sqrt((im1_keypoint[m.queryIdx].pt[0]-im2_keypoint[m.trainIdx].pt[0])**2+(im1_keypoint[m.queryIdx].pt[1]-im2_keypoint[m.trainIdx].pt[1])**2)
            # print(" "+str(m.queryIdx)+": ("+ str(im1_keypoint[m.queryIdx].pt)+") --> "+str(m.trainIdx)+": ("+ str(im2_keypoint[m.trainIdx].pt)+")")
    mean = sum / len(matches)
    sum = 0
    for m in good:
        sum += np.sqrt((im1_keypoint[m[0].queryIdx].pt[0]-im2_keypoint[m[0].trainIdx].pt[0])**2+(im1_keypoint[m[0].queryIdx].pt[1]-im2_keypoint[m[0].trainIdx].pt[1])**2)
    std_dev = np.sqrt(sum/len(matches))
    matched_rgb = np.copy(image_to_show)
    matched_rgb[:,:,1] = im2_histoeq_int[:,:,2]
    matched_rgb[:,:,2] = image_to_show_2[:,:,2]
    output = []
    shortest = 1000
    longest = 0
    for m in good:
            length = np.sqrt((im1_keypoint[m[0].queryIdx].pt[0]-im2_keypoint[m[0].trainIdx].pt[0])**2+(im1_keypoint[m[0].queryIdx].pt[1]-im2_keypoint[m[0].trainIdx].pt[1])**2)
            if (length < mean+2*std_dev):
                if length < shortest:
                    shortest = length
                if length > longest:
                    longest = length
    for m in good:
        length = np.sqrt((im1_keypoint[m[0].queryIdx].pt[0]-im2_keypoint[m[0].trainIdx].pt[0])**2+(im1_keypoint[m[0].queryIdx].pt[1]-im2_keypoint[m[0].trainIdx].pt[1])**2)
        if (length < mean+2*std_dev):
            output.append(m)
            color_val = int((length-shortest)/(longest-shortest)*255)
            color = (color_val,255-color_val,0)
            cv.arrowedLine(matched_rgb,(int(im1_keypoint[m[0].queryIdx].pt[0]),int(im1_keypoint[m[0].queryIdx].pt[1])),(int(im2_keypoint[m[0].trainIdx].pt[0]),int(im2_keypoint[m[0].trainIdx].pt[1])), color ,1)
    return matched_rgb, output
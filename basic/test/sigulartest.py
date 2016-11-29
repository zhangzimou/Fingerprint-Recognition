#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 22:28:35 2016

@author: zhangzimou
"""
####complex filter method
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
import time
from basic import imshow
import basic
from preprocess import enhance
import preprocess as pre
from minutiaeExtract import minutiaeExtract


def getGaussianKernel(size,sigma):
    x=np.arange(-size[0]/2+1,size[0]/2+1)
    y=np.arange(-size[1]/2+1,size[1]/2+1)
    xx,yy=np.meshgrid(x,y)
    kernel=np.exp(-(xx**2+yy**2)/2)
    return kernel

FVC0='/home/zhangzimou/Desktop/code_lnk/database/FVC2000/'
FVC2='/home/zhangzimou/Desktop/code_lnk/database/FVC2002/'
FVC4='/home/zhangzimou/Desktop/code_lnk/database/FVC2004/'
path=FVC4+'DB1_B/'

img=cv2.imread(path+'101_2.tif',0)
img,imgfore=pre.segmentation(img)

sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
fx=convolve2d(img,sobel_x,mode='same')
fy=convolve2d(img,sobel_y,mode='same')
z=(fx+fy*1j)**2

N=33
kernel=getGaussianKernel((N,N),7)
x,y=np.arange(-N/2+1,N/2+1),np.arange(-N/2+1,N/2+1)
xx,yy=np.meshgrid(x,y)
fc=(xx+yy*1j)*kernel
fd=(xx-yy*1j)*kernel

fc_f=np.abs(convolve2d(z,fc,mode='same'))
fd_f=np.abs(convolve2d(z,fd,mode='same'))
plt.imshow(fd_f,cmap='gray')

#plt.figure()
#plt.subplot(1,2,1)
#imshow(img)
#plt.subplot(1,2,2)
#imshow(imgfore)
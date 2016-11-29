#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:48:18 2016

@author: zhangzimou
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
import time
import sys
sys.path.append("..")
from basic import imshow
import basic
from preprocess import enhance
import preprocess as pre
from minutiaeExtract import minutiaeExtract
from basic import block_view

FVC0='/home/zhangzimou/Desktop/code_lnk/database/FVC2000/'
FVC2='/home/zhangzimou/Desktop/code_lnk/database/FVC2002/'
FVC4='/home/zhangzimou/Desktop/code_lnk/database/FVC2004/'
path=FVC4+'DB3_B/'

blockSize=8

img=cv2.imread(path+'105_1.tif',0)
add0=(16-img.shape[0]%16)/2
add1=(16-img.shape[1]%16)/2
img=np.vstack((  255*np.ones((add0,img.shape[1])), img, 255*np.ones((add0,img.shape[1]))  ))
img=np.hstack((  255*np.ones((img.shape[0],add1)), img, 255*np.ones((img.shape[0],add1))  ))
img=np.uint8(img)
#
#theta=pre.calcDirection(img,blockSize)
#wl=pre.calcWl(img,blockSize)
#img=pre.GaborFilter(img,blockSize,wl,np.pi/2-theta)

img_b=block_view(img,(32,32))


sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
par_x=convolve2d(img,sobel_x,mode='same')
par_y=convolve2d(img,sobel_y,mode='same')

#img=basic.blockproc(img,cv2.equalizeHist,(blockSize,blockSize))

#Mx=basic.blockproc(par_x,np.mean,(8,8),True)
#My=basic.blockproc(par_y,np.mean,(8,8),True)
stdx=basic.blockproc(par_x,np.std,(blockSize,blockSize),True)
stdy=basic.blockproc(par_y,np.std,(blockSize,blockSize),True)
grddev=stdx+stdy
threshold=100
index=grddev[1:-1,1:-1].copy()
index[np.where(index<threshold)]=0
index[np.where(index>=threshold)]=1
a=np.zeros(grddev.shape)
a[1:-1,1:-1]=index
index=a
      
valid=np.zeros(img.shape)
valid_b=block_view(valid,(blockSize,blockSize))
valid_b[:]=index[:,:,np.newaxis,np.newaxis]

kernel = np.ones((8,8),np.uint8)
valid=cv2.dilate(valid,kernel,iterations = 3)
valid=cv2.erode(valid, kernel, iterations = 10)
valid=cv2.dilate(valid, kernel, iterations=3)

#img[np.where(valid==0)]=255


plt.figure()

#imshow(img)
plt.subplot(1,2,1)
imshow(img)
plt.subplot(1,2,2)
plt.imshow(valid, cmap='gray')

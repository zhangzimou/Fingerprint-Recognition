#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 12:22:57 2016

@author: zhangzimou
"""

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

FVC0='/home/zhangzimou/Desktop/code_lnk/database/FVC2000/'
FVC2='/home/zhangzimou/Desktop/code_lnk/database/FVC2002/'
FVC4='/home/zhangzimou/Desktop/code_lnk/database/FVC2004/'
path=FVC4+'DB1_B/'

start=time.clock()

img=cv2.imread(path+'110_2.tif',0)
image,imgfore=pre.segmentation(img)

blockSize=8
boxSize=4
theta=pre.calcDirectionBox(image,blockSize,boxSize)

P =[ theta[2:,1:-1], theta[2:,2:], theta[1:-1,2:], theta[:-2,2:], theta[:-2,1:-1],theta[:-2,:-2], theta[1:-1,:-2], theta[2:,:-2]]



N,M=image.shape
N1,M1=theta.shape
delta=np.zeros((N1-2,M1-2))
for i in range(8):
    if i==7:
        de=P[0]-P[7]
    else:
        de=P[i+1]-P[i]
    de[np.where(de<=-np.pi/2)]+=np.pi
    de[np.where(de>np.pi/2)]-=np.pi
    delta+=de
delta /= 2*np.pi
delta_new=np.zeros(theta.shape)
delta_new[1:-1,1:-1]=delta
delta=delta_new

delta_extend=np.zeros(image.shape)
delta_extend[blockSize/2::boxSize,blockSize/2::boxSize]=delta[:-1,:-1]
invalid=np.where(imgfore!=0)
delta_extend[invalid]=0
delta[:-1,:-1]=delta_extend[blockSize/2::boxSize,blockSize/2::boxSize]

core_index=np.where((delta<=0.55)*(delta>=0.45))
delta_index=np.where((delta<=-0.45)*(delta>=-0.55))

plt.figure()
imshow(image)
plt.plot(blockSize/2+core_index[1]*boxSize,blockSize/2+core_index[0]*boxSize,'b.')
plt.plot(blockSize/2+delta_index[1]*boxSize,blockSize/2+delta_index[0]*boxSize,'r.')
#X,Y=np.mgrid[0:N:blockSize,0:M:blockSize]
#plt.quiver(Y,X,np.cos(theta),np.sin(theta),color='g')

    
end=time.clock()
print end-start

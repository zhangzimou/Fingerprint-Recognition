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

img=cv2.imread(path+'105_2.tif',0)
image,imgfore=pre.segmentation(img)

image=enhance(img)

blockSize=8
theta=pre.calcDirection(image,blockSize,method='pixel-wise')

P =[ theta[2:,1:-1], theta[2:,2:], theta[1:-1,2:], theta[:-2,2:], theta[:-2,1:-1],theta[:-2,:-2], theta[1:-1,:-2], theta[2:,:-2]]



N,M=image.shape
delta=np.zeros((N-2,M-2))
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

imgfore=cv2.boxFilter(imgfore,-1,(29,29))
imgfore[np.where(imgfore>0)]=255

invalid=np.where(imgfore!=0)
delta[invalid]=0

core_index=np.where((delta<=0.55)*(delta>=0.45))
delta_index=np.where((delta<=-0.45)*(delta>=-0.55))

plt.figure()
imshow(image)
plt.plot(core_index[1],core_index[0],'b.')
plt.plot(delta_index[1],delta_index[0],'r.')
X,Y=np.mgrid[0:N,0:M]
shownum=8
plt.quiver(Y[::shownum,::shownum],X[::shownum,::shownum],np.cos(theta[::shownum,::shownum]),np.sin(theta[::shownum,::shownum]),color='r')

    
end=time.clock()
print end-start

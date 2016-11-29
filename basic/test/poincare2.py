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
import minutiaeExtract as extract

FVC0='/home/zhangzimou/Desktop/code_lnk/database/FVC2000/'
FVC2='/home/zhangzimou/Desktop/code_lnk/database/FVC2002/'
FVC4='/home/zhangzimou/Desktop/code_lnk/database/FVC2004/'
path=FVC4+'DB1_B/'

start=time.clock()

img=cv2.imread(path+'105_1.tif',0)
image,imgfore=pre.segmentation(img)

image=enhance(img)
core_index, delta_index=extract.singular(image,imgfore)

blockSize=8
#theta=pre.calcDirection(image,blockSize,method='block-wise')
#
#P =[ theta[2:,1:-1], theta[2:,2:], theta[1:-1,2:], theta[:-2,2:], theta[:-2,1:-1],theta[:-2,:-2], theta[1:-1,:-2], theta[2:,:-2]]
#
#
#
#N,M=image.shape
#N1,M1=theta.shape
#delta=np.zeros((N1-2,M1-2))
#for i in range(8):
#    if i==7:
#        de=P[0]-P[7]
#    else:
#        de=P[i+1]-P[i]
#    de[np.where(de<=-np.pi/2)]+=np.pi
#    de[np.where(de>np.pi/2)]-=np.pi
#    delta+=de
#delta /= 2*np.pi
#delta_new=np.zeros(theta.shape)
#delta_new[1:-1,1:-1]=delta
#delta=delta_new
#delta_extend=np.zeros(image.shape)
#delta_extend[::blockSize,::blockSize]=delta
#invalid=np.where(imgfore!=0)
#delta_extend[invalid]=0
#delta[:]=delta_extend[::blockSize,::blockSize]
#
#core_index=np.where((delta<=0.55)*(delta>=0.45))
#delta_index=np.where((delta<=-0.45)*(delta>=-0.55))

plt.figure()
imshow(image)
#plt.plot(core_index[1]*blockSize,core_index[0]*blockSize,'b.')
#plt.plot(delta_index[1]*blockSize,delta_index[0]*blockSize,'r.')
plt.plot(core_index[1],core_index[0],'b.')
plt.plot(delta_index[1],delta_index[0],'r.')
X,Y=np.mgrid[0:N:blockSize,0:M:blockSize]
#plt.quiver(Y,X,np.cos(theta),np.sin(theta),color='g')

    
end=time.clock()
print end-start

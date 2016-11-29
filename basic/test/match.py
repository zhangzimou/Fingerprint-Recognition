#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:58:39 2016

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
path=FVC2+'DB1_B/'

blockSize=8

img1=cv2.imread(path+'101_1.tif',0)
img2=cv2.imread(path+'101_2.tif',0)
img_seg1,imgfore1=pre.segmentation(img1)
img_seg2,imgfore2=pre.segmentation(img2)
imgE1=enhance(img1)
imgE2=enhance(img2)
imgB1=basic.binarize(imgE1)
imgB2=basic.binarize(imgE2)
imgT1=pre.thinning(imgB1)
imgT2=pre.thinning(imgB2)


plt.figure()
plt.subplot(1,2,1)
imshow(img_seg1)
plt.subplot(1,2,2)
imshow(img_seg2)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(imgE1,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(imgE2,cmap='gray')


ending1,bifur1,theta11,theta21=minutiaeExtract(imgT1,imgfore1)
ending2,bifur2,theta12,theta22=minutiaeExtract(imgT2,imgfore2)



plt.figure()
plt.subplot(1,2,1)
plt.imshow(imgT1,cmap='gray')
plt.plot(ending1.T[1],ending1.T[0],'b.',bifur1.T[1],bifur1.T[0],'r.')
plt.quiver(ending1.T[1],ending1.T[0],np.cos(theta11),np.sin(-theta11),color='b',width=0.003)
plt.quiver(bifur1.T[1],bifur1.T[0],np.cos(theta21),np.sin(-theta21),color='r',width=0.003)

plt.subplot(1,2,2)
plt.imshow(imgT2,cmap='gray')
plt.plot(ending2.T[1],ending2.T[0],'b.',bifur2.T[1],bifur2.T[0],'r.')
plt.quiver(ending2.T[1],ending2.T[0],np.cos(theta12),np.sin(-theta12),color='b',width=0.003)
plt.quiver(bifur2.T[1],bifur2.T[0],np.cos(theta22),np.sin(-theta22),color='r',width=0.003)





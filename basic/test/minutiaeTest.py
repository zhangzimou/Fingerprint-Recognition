#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:02:15 2016

@author: zhangzimou
"""
import pyximport
pyximport.install() 

import line_profiler
import os,os.path
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
from _match import minutiaeMatch

FVC0='/home/zhangzimou/Desktop/code_lnk/database/FVC2000/'
FVC2='/home/zhangzimou/Desktop/code_lnk/database/FVC2002/'
FVC4='/home/zhangzimou/Desktop/code_lnk/database/FVC2004/'
path=FVC2+'DB1_B/'



blockSize=8

img=cv2.imread(path+'108_3.tif',0)



imgE,imgfore=enhance(img)
imgB=basic.binarize(imgE)
imgT=pre.thinning(imgB)

#plt.figure()
#imshow(img)
plt.figure()
plt.imshow(imgE,cmap='gray')
plt.figure()
plt.imshow(imgT,cmap='gray')


ending,bifur,theta1,theta2=minutiaeExtract(imgT,imgfore)
plt.plot(ending.T[1],ending.T[0],'b.',bifur.T[1],bifur.T[0],'r.')
plt.quiver(ending.T[1],ending.T[0],np.cos(theta1),np.sin(-theta1),color='b',width=0.003)
plt.quiver(bifur.T[1],bifur.T[0],np.cos(theta2),np.sin(-theta2),color='r',width=0.003)






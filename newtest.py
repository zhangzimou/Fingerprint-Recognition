#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:01:52 2016

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


start=time.clock()
img_origin=cv2.imread('pic5.tif',0)
img_origin,imgfore=pre.segmentation(img_origin)
blockSize=16
theta=pre.calcDirection(img_origin,blockSize)
wl=pre.calcWl(img_origin,blockSize)
#img=ridgeComp2(img,theta,blockSize)
img_enhance=pre.GaborFilter(img_origin,blockSize,wl,np.pi/2-theta)
img_enhance[np.where(imgfore==255)]=255

img_thin=pre.thinning(basic.binarize(img_enhance))
ending,bifur,theta1,theta2=minutiaeExtract(img_thin,imgfore)
ending,bifur=ending.T,bifur.T

end=time.clock()
print end-start
plt.figure()
imshow(img_origin)
plt.figure()
imshow(img_enhance)
plt.figure()
plt.imshow(img_thin,cmap='gray')
plt.figure()
plt.imshow(img_thin,cmap='gray')
plt.plot(ending[1],ending[0],'b.',bifur[1],bifur[0],'r.')
plt.quiver(ending[1],ending[0],np.cos(theta1),np.sin(-theta1),color='b',width=0.003)
plt.quiver(bifur[1],bifur[0],np.cos(theta2),np.sin(-theta2),color='r',width=0.003)
plt.axis([0,320,0,480])
plt.gca().invert_yaxis()
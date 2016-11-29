#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:17:14 2016

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
path=FVC4+'DB1_B/'

start=time.clock()

blockSize=8
boxSize=8
img=cv2.imread(path+'106_1.tif',0)

img,imgfore=pre.segmentation(img)
theta=pre.calcDirectionBox(img,blockSize,boxSize)
wl=pre.calcWlBox(img,blockSize,boxSize)

# inverse
#image=255-img

image=pre.GaborFilterBox(255-img,blockSize,boxSize,wl,np.pi/2-theta)
image=255-image
image[np.where(imgfore==0)]=255


#theta=pre.calcDirection(img,blockSize)
#wl=pre.calcWl(img,blockSize)
#img_r=pre.ridgeComp2(img,theta,blockSize)
#img_r=img.copy()
# inverse
#img_r=255-img_r
#image=pre.GaborFilter(255-img_r,blockSize,wl,np.pi/2-theta)
#image=255-image
#image[np.where(imgfore==0)]=255       
    
plt.figure()
imshow(img)
#plt.figure()
#imshow(img_r)
plt.figure()
imshow(image)

end=time.clock()
print end-start
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
boxSize=4
img=cv2.imread(path+'101_2.tif',0)


img,imgfore=pre.segmentation(img)
N,M=img.shape
#img=basic.blockproc(np.uint8(img),cv2.equalizeHist,(16,16))
theta=pre.calcDirectionBox(img,blockSize,boxSize)

#imgR=pre.ridgeComp(img,theta,blockSize,boxSize,9)
imgR=img.copy()

wl=pre.calcWlBox(img,blockSize,boxSize)
#wl=6.5*np.ones((88,72))
# inverse
#image=255-img

image=pre.GaborFilterBox(imgR,blockSize,boxSize,wl,np.pi/2-theta)
#image=255-image

      
image2=pre.GaborFilterBox(image,blockSize,boxSize,wl,np.pi/2-theta) 
#image2[np.where(imgfore==0)]=255 

image3=pre.GaborFilterBox(image2,blockSize,boxSize,wl,np.pi/2-theta) 
#image3[np.where(imgfore==0)]=255
image4=pre.GaborFilterBox(image3,blockSize,boxSize,wl,np.pi/2-theta)
#image3=pre.GaborFilterBox(image3,blockSize,boxSize,wl,np.pi/2-theta)
#image3=pre.GaborFilterBox(image3,blockSize,boxSize,wl,np.pi/2-theta)
#image3=pre.GaborFilterBox(image3,blockSize,boxSize,wl,np.pi/2-theta)
#imgfore=cv2.erode(imgfore,np.ones((8,8)),iterations=3)
image4[np.where(imgfore==0)]=255
      
#image=basic.binarize(image)
#image=cv2.erode(image,np.ones((2,2)),2)
#image=basic.binarize(image)

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
X,Y=np.mgrid[blockSize/2:N:boxSize,blockSize/2:M:boxSize]
plt.quiver(Y[::3,::3],X[::3,::3],np.cos(theta[::3,::3]),np.sin(theta[::3,::3]),color='r')
plt.figure()
imshow(imgR)
plt.figure()
plt.imshow(basic.truncate(image,method='part'),cmap='gray')

plt.figure()
plt.imshow(basic.truncate(image2,method='part'),cmap='gray')

plt.figure()
plt.imshow(basic.truncate(image3,method='part'),cmap='gray')

plt.figure()
plt.imshow(basic.truncate(image4,method='part'),cmap='gray')

end=time.clock()
print end-start
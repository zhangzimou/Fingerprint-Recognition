#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 20:32:47 2016

@author: zhangzimou
"""

from initial import imshow
import numpy as np
from matplotlib import pyplot as plt
import cv2
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
import initial as init
import time
from scipy.signal import convolve2d

#x=np.arange(48);y=np.arange(48)
#theta=np.pi/2-1.33
#xx,yy=np.meshgrid(x,y)
#x1=xx*np.cos(theta)+yy*np.sin(theta);y1=-xx*np.sin(theta)+yy*np.cos(theta)
#img=(np.cos(2*np.pi*x1/10)+1)*120
#noise=np.zeros_like(img)
#cv2.randn(noise,(0),(70))
#imgN=img+noise
#imshow(imgN)
#cv2.imwrite('test.jpg',img)
#start=time.clock()
#
img=cv2.imread('pic3.tif',0)
#img1=100*(img-np.mean(img))
#img1[np.where(img1>255)]=255
#blockSize=51
#kernel=np.ones((blockSize,blockSize))/(blockSize**2)
#img2=cv2.filter2D(img1,-1,kernel)
#img3=img2.copy()
#img3[np.where(img2>150)]=255; img3[np.where(img2<=150)]=0
#plt.subplot(1,3,1)
#imshow(img)
#plt.subplot(1,3,2)
#imshow(img1)
#plt.subplot(1,3,3)
#imshow(img2)
#plt.figure()
#imshow(img3)


img=100*(img-np.mean(img))
blockSize=31
img[np.where(img>255)]=255
img=cv2.boxFilter(img,-1,(blockSize,blockSize))
img[np.where(img>150)]=255; img[np.where(img<=150)]=0   
img=cv2.boxFilter(img,-1,(31,31))
img[np.where(img>0)]=255
print np.min(img)
imshow(img)
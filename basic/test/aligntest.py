#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 22:28:35 2016

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

img_origin=cv2.imread(path+'107_6.tif',0)


img_align=pre.align(img_origin)
imshow(img_align)

#img,imgfore=pre.segmentation(img_origin)
#
#imgfore=pre.foreground(img_origin)
#
#sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
#sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
#par_x=convolve2d(imgfore,sobel_x,mode='same')
#par_y=convolve2d(imgfore,sobel_y,mode='same')
#N,M=np.shape(img)
#Vy=2*np.sum(par_x*par_y)
#Vx=np.sum(par_y**2-par_x**2)
#theta=0.5*np.arctan2(Vy,Vx)    
#Matrix=cv2.getRotationMatrix2D((M/2,N/2),(np.pi/2-theta)*180/np.pi,1)    
#img_ro=cv2.warpAffine(img_origin,Matrix,(M,N))
#
#imshow(img_ro)
#plt.figure()
#plt.subplot(1,2,1)
#imshow(img)
#plt.subplot(1,2,2)
#imshow(imgfore)
#plt.quiver(M/2,N/2,np.cos(theta),np.sin(theta),color='r')
#
#print theta*180/np.pi
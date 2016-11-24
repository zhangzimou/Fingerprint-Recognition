#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:36:42 2016

@author: zhangzimou
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
import time
from initial import imshow
from initial import normalize2
from initial import blockproc
from initial import ridgeComp
from initial import ridgeComp2
from initial import binarize
from initial import binarize2
from initial import segmentation
from initial import calcWlDire
from initial import GaborFilter
import initial as init
    
start=time.clock()
img=cv2.imread('pic2.tif',0)
img,a=segmentation(img)
#img=cv2.resize(img,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
w=16
wl=init.calcWl(img,w)



sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
par_x=convolve2d(img,sobel_x,mode='same')
par_y=convolve2d(img,sobel_y,mode='same')

N,M=np.shape(img)
Vx=np.zeros((N/w,M/w))
Vy=np.zeros((N/w,M/w))

for i in range(N/w):
    for j in range(M/w):
        a=i*w;b=a+w;c=j*w;d=c+w
        Vy[i,j]=2*np.sum(par_x[a:b,c:d]*par_y[a:b,c:d])
        Vx[i,j]=np.sum(par_y[a:b,c:d]**2-par_x[a:b,c:d]**2)
gaussianBlurSigma=2
gaussian_block=5
Vy=cv2.GaussianBlur(Vy,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
Vx=cv2.GaussianBlur(Vx,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
theta=0.5*np.arctan2(Vy,Vx)#+np.pi/2
#theta[np.where(theta>np.pi/2)]-=np.pi
#theta[np.where(theta<=-np.pi/2)]+=np.pi
#theta=0.5*cv2.GaussianBlur(2*theta,(gaussian_block,gaussian_block),1.5)
#theta_filter=convolve2d(theta,h,mode='same')
#theta_small=theta_filter[0:N:w,0:M:w]


#img1=ridgeComp2(img,theta,w)
##img2=ridgeComp(img1,theta,w)
##img2=blockproc(img1,binarize,(32,32))
#img2=binarize2(img1,theta,w)
imgGabor=GaborFilter(img,w,wl,np.pi/2-theta)
end=time.clock()
print end-start
X,Y=np.mgrid[0:N:w,0:M:w]
plt.figure()
plt.subplot(1,2,1)
plt.quiver(Y,X,np.cos(theta),np.sin(theta),color='r')
imshow(img)
plt.subplot(1,2,2)
imshow(imgGabor)
##currentAxis = plt.gca()
##currentAxis.add_patch(Rectangle((400,96),w,w,fill=None,color='b'))
#plt.figure()#
#img12=np.hstack((img1,img2))
#imshow(img12)
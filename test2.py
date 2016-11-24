#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:24:45 2016

@author: zhangzimou
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
import time
from initial import imshow
from initial import normalize
from initial import normalize2


def calcDirection(img):
    sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
    par_x=convolve2d(img,sobel_x,mode='same')
    par_y=convolve2d(img,sobel_y,mode='same')
    Vy=2*np.sum(par_x*par_y)
    Vx=np.sum(par_y**2-par_x**2)
    theta=0.5*np.arctan2(Vy,Vx)#+np.pi/2
    return theta



start=time.clock()
img=cv2.imread('pic1.png',0)

#img=np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
#img=img*255
#w=4
w=16
aa=200
bb=100
img=img[aa:aa+w,bb:bb+w]
img=normalize2(img)
#img=normalize2(img)
#img=cv2.resize(img,None, fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
#w=w*4
#img=normalize(img)
#img=(I-np.min(I))*255.0/(np.max(I)-np.min(I))
sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
par_x=convolve2d(img,sobel_x,mode='same')
par_y=convolve2d(img,sobel_y,mode='same')

N,M=np.shape(img)
Vx=np.zeros((N/w,M/w))
Vy=np.zeros((N/w,M/w))
#theta=np.zeros(np.shape(img))
for i in range(N/w):
    for j in range(M/w):
        a=i*w;b=a+w;c=j*w;d=c+w
        Vy[i,j]=2*np.sum(par_x[a:b,c:d]*par_y[a:b,c:d])
        Vx[i,j]=np.sum(par_y[a:b,c:d]**2-par_x[a:b,c:d]**2)
theta=0.5*np.arctan2(Vy,Vx)#+np.pi/2
end=time.clock()
print end-start
X,Y=np.mgrid[0:M:w,0:N:w]
plt.figure()
plt.quiver(X,Y,np.cos(theta),np.sin(theta),color='r')
imshow(img)
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((400,100),w,w,fill=None,color='b'))
plt.show()
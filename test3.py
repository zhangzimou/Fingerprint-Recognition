#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:18:09 2016

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
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
import initial as init


def calcDirection(img):
    sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
    par_x=convolve2d(img,sobel_x,mode='same')
    par_y=convolve2d(img,sobel_y,mode='same')
    Vy=2*np.sum(par_x*par_y)
    Vx=np.sum(par_y**2-par_x**2)
    theta=0.5*np.arctan2(Vy,Vx)#+np.pi/2
    return theta
    
img=cv2.imread('pic1.png',0)
w=60
aa=200
bb=100
img=img[aa:aa+w,bb:bb+w]
#img=cv2.equalizeHist(img)
mmin=np.min(img)
mmax=np.max(img)
q=0.2
#img[np.where(img<q*(mmax-mmin))]=0
#img[np.where(img>(1-q)*(mmax-mmin))]=255
img=normalize2(img,125,128)
theta=calcDirection(img)
imshow(img)
img[np.where(img>255)]=255;img[np.where(img<0)]=0
M,N=np.shape(img)
plt.quiver(M/2,N/2,np.cos(theta),np.sin(theta),color='r')

f=np.fft.fft2(img)
f=np.fft.fftshift(f)
fmag = np.abs(f)
#imshow(fmag)
aa=np.where(fmag==np.max(fmag))
fmag[aa]=0; argmax=np.where(fmag==np.max(fmag))
pho_c=((argmax[0][0]-aa[0][0])**2+(argmax[1][0]-aa[1][0])**2)**0.5
N,M=np.shape(img)
nn,mm=np.meshgrid(np.arange(-(N-1)/2.0,(N-1)/2.0+1),np.arange((M-1)/2.0+1,-(M-1)/2.0,-1))
bw=3
pho=(nn**2+mm**2)**0.5
HPho=np.exp(-(pho-pho_c)**2/(2*bw))/(2*np.pi)**0.5/bw
phi=np.arctan2(mm,nn)
phi_c=theta-np.pi/2
phiBW=0.6
phi[np.where(np.abs(phi-phi_c)>=phiBW)]=phi_c+phiBW
Hphi=np.cos(np.pi/2*(phi-phi_c)/phiBW)**2
H_filter=HPho*Hphi
img_filter_f=f*H_filter
img_filter=np.abs(np.fft.ifft2(np.fft.fftshift(img_filter_f)))

#plt.figure()
#plt.imshow(Hphi,cmap='gray')
plt.figure()
plt.imshow(img_filter,cmap='gray')

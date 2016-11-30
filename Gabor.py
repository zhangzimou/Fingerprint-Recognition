#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 19:37:36 2016

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
#img=(np.cos(2*np.pi*x1/10)+1)*40
#
#noise=np.zeros_like(img)
#cv2.randn(noise,(0),(70))
#imgN=img+noise
#imgN=cv2.imread('test.jpg',0)
#imshow(imgN)
#
#blockSize=16
##theta=init.calcDirection(img,blockSize)
#wl=init.calcWl(img,blockSize)
#kernel=cv2.getGaborKernel((48,48),20,theta,9,1,0)
#imgG=convolve2d(img,kernel,mode='same')
##imgG=init.GaborFilter(imgN,blockSize,wl,np.pi/2-theta)
#plt.figure()
#imshow(imgG)

#cv2.imwrite('test.jpg',imgN)



#start=time.clock()
#
img=cv2.imread('pic.png',0)
blockSize=16
img,imgfore=init.segmentation(img)
N,M=img.shape
X,Y=np.mgrid[0:N:blockSize,0:M:blockSize]
plt.figure()
#plt.subplot(1,3,1)

theta=init.calcDirection(img,blockSize)
plt.quiver(Y,X,np.cos(theta),np.sin(theta),color='r')
imshow(img)
#wl=init.calcWl(img,blockSize)
wl=16*np.ones((7,7))
#imgR=init.ridgeComp2(img,theta,blockSize)
imgR=img.copy()
#plt.subplot(1,3,2)
plt.figure()
imshow(imgR)
#imga=init.inverse(imgR)
imgG=init.GaborFilter(imgR,blockSize,wl,np.pi/2-theta)
#imgG[np.where(imgfore==255)]=255
#imgG=init.inverse(imga)
#plt.subplot(1,3,3)
plt.figure()
imshow(imgG)




#end=time.clock()
#print end-start
#w=16
#phi=init.calcDirection(img,w)
#result = gabor.gabor(img, w, phi)
#result.show()



#w=48
#aa=200
#bb=120
#img=cv2.imread('pic1.png',0)
#img=img[aa:aa+w,bb:bb+w]
#img=cv2.equalizeHist(img)
#
#img=cv2.imread('fuck.jpg',0)
##img=img.astype(np.float64)
#
##cv2.imwrite('test.jpg',img)
#w=16
#phi=init.calcDirection(img,w)
#wl,dire=init.calcWlDire(img,w)
#img1=init.ridgeComp2(img,phi,w)
#img2=init.ridgeComp2(img1,phi,w)
##cv2.imwrite('fuck.jpg',img2)
#plt.figure()
#plt.subplot(1,2,1)
#imshow(img)
#plt.subplot(1,2,2)
#imshow(img2)
###wl[np.where(wl==32)]=16
###dire[:,:]=theta
#imgG=init.GaborFilter(img2,w,wl,np.pi/2-phi)
#plt.figure()
#imshow(imgG)
##f=fftshift(fft2(img))
##fmag=np.abs(f)
#plt.figure()
#plt.subplot(1,2,1)
#xx,yy=np.meshgrid(np.arange(0,img.shape[1],16),np.arange(0,img.shape[0],16))
#plt.quiver(yy,xx,np.cos(phi),np.sin(phi),color='r')
#imshow(img)
#plt.subplot(1,2,2)
#imshow(imgG)
##plt.imshow(fmag,cmap='gray')
#
##imshow(img)
#
#
#sigma=4
##theta=dire
##lam=wl
##gamma=1
#img=np.ones((100,100))*255
#kernel=cv2.getGaborKernel((60,60),sigma,0,1,1)
#fimg=cv2.filter2D(img,-1,kernel)
#plt.figure()
##plt.subplot(1,2,1)
##imshow(img)
##plt.subplot(1,2,2)
#imshow(fimg)
#kernel_f=np.fft.fftshift(np.fft.fft2(kernel))
#kernelMag=20*np.log(np.abs(kernel_f))
#plt.imshow(kernelMag,cmap='gray')
#cython: profile=True
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:43:34 2016

@author: zhangzimou
"""

import cv2
import sys
sys.path.append("..")
from basic import block_view
cimport numpy as np
#cimport cython
import numpy as np
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from scipy.signal import convolve2d
from libc.math cimport pow,atan2

def GaborFilterBox(double[:,::1] img,int blockSize,int boxSize,
                   double[:,::1] wl,double[:,::1] dire,float sigma=20):
    """Gabor Filter
    img: input image
    blockSize: size of a block
    wl: wavelength
    dire: direction
    return: filtered image
    """
    cdef int N=img.shape[0]
    cdef int M=img.shape[1]
    cdef double[:,::1] imgout=img.copy()
    cdef double[:,::1] kernel
    kernel=np.zeros((img.shape[0]/boxSize*(blockSize+1),img.shape[1]/boxSize*(blockSize+1)))
    cdef int i,j,k1,k2
    cdef int a,b,c,d
    for i in range(N/boxSize):
        a=i*(blockSize+1)
        b=a+blockSize+1
        for j in range(M/boxSize):
            c=j*(blockSize+1)
            d=c+blockSize+1
            aaa=cv2.getGaborKernel((blockSize,blockSize),sigma,dire[i,j],wl[i,j],1,0)
            for k1 in range(a,b):
                for k2 in range(c,d):
                    kernel[k1,k2]=aaa[k1-a,k2-c]       
                 
    cdef int ii,jj
    cdef int a1,b1,c1,d1
    cdef double s=0
    for i in range(blockSize/2,N-blockSize/2):
        ii = (i-blockSize/2)/boxSize
        if ii>=N/boxSize:
            break
        a=i-blockSize/2
        b=a+blockSize+1
        a1=ii*(blockSize+1)
        b1=a1+blockSize+1
        for j in range(blockSize/2,M-blockSize/2):
            jj = (j-blockSize/2)/boxSize
            if jj>=M/boxSize:
                break
            c=j-blockSize/2
            d=c+blockSize+1
            c1=jj*(blockSize+1)
            d1=c1+blockSize+1
            s = 0
            for k1 in range(a,b):
                for k2 in range(c,d):
                    s += kernel[a1+k1-a,c1+k2-c]*img[k1,k2]
            imgout[i,j] = s
    
    return imgout
 
cdef inline double blkwl(double[:,::1] img):
    """Calculate wavelength  given an image block"""
    f=np.abs(fftshift(fft2(img)))
    origin=np.where(f==np.max(f))
    f[origin]=0
    mmax=np.where(f==np.max(f))
    cdef double wl
    wl=2*img.shape[0]/(((origin[0][0]-mmax[0][0])*2)**2+((origin[1][0]-mmax[1][0])*2)**2)**0.5
    return wl

def calcWlBox(double[:,::1] img,int blockSize,int boxSize):
    resize=5
    img=cv2.resize(np.asarray(img),None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
    blockSize=blockSize*resize
    boxSize=boxSize*resize
    N=img.shape[0]
    M=img.shape[1]
    wl=100*np.ones((img.shape[0]/boxSize,img.shape[1]/boxSize))    
    ii=-1
    for i in range(blockSize/2,N-blockSize/2,boxSize):
        ii += 1
        if ii>=N/boxSize:
            break
        a=i-blockSize/2
        b=a+blockSize
        jj=-1
        for j in range(blockSize/2,M-blockSize/2,boxSize):
            jj += 1
            if jj>=M/boxSize:
                break
            c=j-blockSize/2
            d=c+blockSize
            wl[ii,jj]=blkwl(img[a:b,c:d])
    gaussianBlurSigma=4;    gaussian_block=9
    wl=cv2.GaussianBlur(wl,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    return wl/resize
    
def calcDirectionBox(double[:,::1]img,int blockSize=8,int boxSize=4):
    """calculate ridge directions in an image, using gradient method
    return: ridge directions
    """
    cdef double[:,::1] sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]],np.float64)
    cdef double[:,::1] sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]],np.float64)
    cdef double[:,::1] par_x=convolve2d(img,sobel_x,mode='same')
    cdef double[:,::1] par_y=convolve2d(img,sobel_y,mode='same')
    cdef int N,M
    N=img.shape[0]
    M=img.shape[1]
    cdef double[:,::1] Vx,Vy
    Vx=np.zeros((N/boxSize,M/boxSize))
    Vy=np.zeros((N/boxSize,M/boxSize))
    cdef int i,j,ii,jj,k1,k2
    cdef double s1,s2
    ii=-1
    for i in range(blockSize/2,N-blockSize/2,boxSize):
        ii += 1
        if ii==N/boxSize:
            break
        a=i-blockSize/2
        b=a+blockSize
        jj=-1
        for j in range(blockSize/2,M-blockSize/2,boxSize):
            jj += 1
            if jj==M/boxSize:
                break
            c=j-blockSize/2
            d=c+blockSize
            s1=0
            s2=0
            for k1 in range(a,b):
                for k2 in range(c,d):
                    s1 += par_x[k1,k2] * par_y[k1,k2]
                    s2 += pow(par_y[k1,k2],2)-pow(par_x[k1,k2],2)
            Vy[ii,jj]=2*s1
            Vx[ii,jj]=s2
#            Vy[ii,jj]=2*np.sum(par_x[a:b,c:d]*par_y[a:b,c:d])
#            Vx[ii,jj]=np.sum(par_y[a:b,c:d]**2-par_x[a:b,c:d]**2)                
    gaussianBlurSigma=2;
    gaussian_block=9 
    Vy=cv2.GaussianBlur(np.asarray(Vy),(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    Vx=cv2.GaussianBlur(np.asarray(Vx),(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
#    cdef double[:,::1] theta
    theta=0.5*np.arctan2(Vy,Vx)            
    return theta    
    
    
    
    
    
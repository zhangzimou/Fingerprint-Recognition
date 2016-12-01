#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:13:26 2016

@author: zhangzimou

some basic functions
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from scipy.signal import convolve2d

def block_view(A, block):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)
def blockproc(M, fun, blk_size=(3,3),singleout=False):
    # This is some complex function of blk_size and M.shape
    #stride = blk_size
    output = np.zeros(M.shape if (singleout==False) else (M.shape[0]/blk_size[0],M.shape[1]/blk_size[1]))
    B = block_view(M, block=blk_size)
    if (singleout==False):
        O = block_view(output, block=blk_size)
    else:
        O = output
    for b,o in zip(B, O):
        #o[:,:] = fun(b);
        if (singleout==False):
            o[:,:]=np.asarray([ fun(a) for a in b ])
        else:
            o[:]=np.asarray([ fun(a) for a in b ])
    return output


    
def inverse(img):
    return 255-img

def binarize2(img,theta,blockSize,h=9):
    resize=5
    N,M=np.shape(img)
    imgout=np.zeros_like(img)
    imgresizeize=cv2.resizeize(img,None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
    blockMean=blockproc(img,np.mean,(blockSize,blockSize),True)
    hh=np.arange(-(h-1)/2,(h-1)/2+1)
    for i in xrange((h-1)/2,N-(h-1)/2):
        block_i=i/blockSize
        for j in xrange((h-1)/2,M-(h-1)/2):
            block_j=j/blockSize
            thetaHere=theta[block_i,block_j]
            ii=np.round((i-hh*np.sin(thetaHere))*resize).astype(np.int32)
            jj=np.round((j+hh*np.cos(thetaHere))*resize).astype(np.int32)
            imgout[i,j]=255 if (np.mean(imgresizeize[ii,jj])>blockMean[block_i,block_j]) else 0
    return imgout


    
def binarize(img,reverse=1):
    imgout=np.zeros_like(img)
    imgout[np.where(img>np.mean(img))]=0 if (reverse) else 1
    imgout[np.where(img<=np.mean(img))]=1 if (reverse) else 0
    return imgout

def truncate(img, method='default'):
    image=img.copy()
    if method == 'default':
        image[np.where(img>255)]=255
        image[np.where(img<0)]=0
    elif method == 'mean':
        image[np.where(img>=np.mean(img))]=255
        image[np.where(img<np.mean(img))]=0
    elif method == 'part':
        index=np.where(image!=255)
        image[index]=truncate(image[index],method='mean')
    elif method == 'original':
        return image
    
    return image
    
def imshow(img):
    plt.imshow(img,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    
def normalize(img):
    a=np.max(img)
    b=np.min(img)
    return (img-b)*255.0/(a-b)
    
def normalizeBox(img,blockSize,boxSize,mu=125,sigma=500):
    image=img.copy()
    N,M=img.shape
    for i in xrange(blockSize/2,N-blockSize/2-boxSize,boxSize):
        a=i-blockSize/2
        b=a+blockSize+boxSize
        for j in xrange(blockSize/2,M-blockSize/2-boxSize,boxSize):
            c=j-blockSize/2
            d=c+blockSize+boxSize
            image[i:i+boxSize,j:j+boxSize]=normalize2(img[a:b,c:d],mu,sigma)            [blockSize/2:blockSize/2+boxSize,blockSize/2:blockSize/2+boxSize]
    return image    
    
    
def normalize2(img,mu=125,sigma=1000):
    m=np.mean(img)
    std=np.std(img)
    if (std==0):
        return img
    return mu+sigma/std*(img-m)
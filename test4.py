#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:57:06 2016

@author: zhangzimou
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from scipy.signal import convolve2d

def foreground(img,blockSize=31):
    img=100*(img-np.mean(img))
    img[np.where(img>255)]=255
    img=cv2.boxFilter(img,-1,(blockSize,blockSize))
    img[np.where(img>150)]=255; img[np.where(img<=150)]=0   
    img=cv2.boxFilter(img,-1,(blockSize/2,blockSize/2))
    img[np.where(img>0)]=255
    return img

def testfun(img):
    img[:,:]=0
    return img
    
imgtest=img.copy()    
#a=foreground(img)
#a=testfun(imgtest)

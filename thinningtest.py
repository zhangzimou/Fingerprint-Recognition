#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:17:39 2016

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

imgB=init.binarize(imgG)
imgt=init.thinning(imgB)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(imgB,cmap='gray',vmin=0,vmax=1)
plt.subplot(1,2,2)
plt.imshow(imgt,cmap='gray',vmin=0,vmax=1)
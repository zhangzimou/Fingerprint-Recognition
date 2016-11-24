#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 22:11:48 2016

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
import initial as init

qq=init.fill(imgtest,(7,7),0)
#print init.countCrossNum(qq,0)
ww,theta=init.validateMinutiae(imgtest1,np.array([[7,7]]),1)

print ww,theta
#minutiae=np.array([[7,7]])
#blockSize=15
#aa=np.asarray(map(lambda x:init.fill(imgtest[x[0]-blockSize/2:x[0]+blockSize/2+1,
#                                       x[1]-blockSize/2:x[1]+blockSize/2+1],
#                                    (15/2,15/2),0), minutiae ))
#CN=np.asarray(map(lambda x:init.countCrossNum(x,0),aa))
#print CN

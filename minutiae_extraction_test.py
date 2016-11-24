#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 17:00:32 2016

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

image=imgt.copy()
P1=image[1:-1,1:-1]
valid=np.where(P1==1)
P1,P2,P3,P4,P5,P6,P7,P8,P9 = P1[valid],image[2:,1:-1][valid], image[2:,2:][valid], image[1:-1,2:][valid], image[:-2,2:][valid], image[:-2,1:-1][valid],image[:-2,:-2][valid], image[1:-1,:-2][valid], image[2:,:-2][valid]

CN=init.transitions_vec(P2,P3,P4,P5,P6,P7,P8,P9)
ending_index=np.where(CN==1)
bifur_index=np.where(CN==3)
ending=np.asarray((valid[0][ending_index]+1,valid[1][ending_index]+1))
bifur=np.asarray((valid[0][bifur_index]+1,valid[1][bifur_index]+1))

imgfored=cv2.boxFilter(imgfore,-1,(9,9))
imgfored[np.where(imgfored>0)]=255
edge1,edge2=np.where(imgfored[ending[0],ending[1]]==255),np.where(imgfored[bifur[0],bifur[1]]==255)
ending=np.delete(ending.T,edge1[0],0)
bifur=np.delete(bifur.T,edge2[0],0)

ending,theta1=init.validateMinutiae(image,ending,1)
bifur,theta2=init.validateMinutiae(image,bifur,0)
ending,bifur=ending.T,bifur.T
plt.imshow(image,cmap='gray')
plt.plot(ending[1],ending[0],'b.',bifur[1],bifur[0],'r.')
plt.quiver(ending[1],ending[0],np.cos(theta1),np.sin(-theta1),color='b',width=0.003)
plt.quiver(bifur[1],bifur[0],np.cos(theta2),np.sin(-theta2),color='r',width=0.003)
#plt.axis([0, 320, 0, 480])


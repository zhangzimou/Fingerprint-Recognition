#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:52:15 2016

@author: zhangzimou
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
import time
import sys
sys.path.append("..")
from basic import imshow
import basic
from preprocess import enhance
import preprocess as pre
from minutiaeExtract import minutiaeExtract
from basic import block_view

FVC0='/home/zhangzimou/Desktop/code_lnk/database/FVC2000/'
FVC2='/home/zhangzimou/Desktop/code_lnk/database/FVC2002/'
FVC4='/home/zhangzimou/Desktop/code_lnk/database/FVC2004/'
path=FVC4+'DB1_B/'

imgB=basic.binarize(image)
imgT=pre.thinning(imgB)
plt.figure()
plt.imshow(imgB,cmap='gray')
plt.figure()
plt.imshow(imgT,cmap='gray')
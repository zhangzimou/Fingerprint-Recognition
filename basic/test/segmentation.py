#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 23:44:21 2016

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
path=FVC4+'DB2_B/'

blockSize=8

img=cv2.imread(path+'106_1.tif',0)

img_seg,imgfore=pre.segmentation(img)


plt.figure()
imshow(img)
plt.figure()
plt.subplot(1,2,1)
imshow(img_seg)
plt.subplot(1,2,2)
plt.imshow(imgfore, cmap='gray')

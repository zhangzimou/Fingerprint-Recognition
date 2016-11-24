#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:15:01 2016

@author: zhangzimou
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
import time
from initial import imshow
from initial import normalize2
from initial import block_view
from initial import blockproc
from initial import ridgeComp
from initial import ridgeComp2
from initial import binarize
from initial import binarize2

start=time.clock()
img=cv2.imread('pic2.tif',0)
blockSize=16
factor=0.9
img_block=block_view(img,(blockSize,blockSize))
imgout=np.zeros_like(img)
out_block=block_view(imgout,(blockSize,blockSize))
threshold=np.min(img)+factor*(np.max(img)-np.min(img))
for b,o in zip(img_block,out_block):
    a=np.asarray(map(lambda x:np.mean(x),b))
    a[np.where(a>=threshold)]=255;a[np.where(a<threshold)]=0
    o[:,:]=np.asarray(map(lambda x,y:x*y,np.ones(o.shape),a))

img12=np.hstack((img,imgout))

end=time.clock()
print end-start
imshow(img12)
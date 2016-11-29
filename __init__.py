#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 2016

@author: Wenxin Fang
"""

import cv2
import numpy as np
import time
import basic
import preprocess as pre
import sys
import os,os.path

sys.path.append("..")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
from basic import imshow
from preprocess import enhance
from minutiaeExtract import minutiaeExtract
from match import minutiaeMatch

start=time.clock()

#open all images in source and store all filenames for search afterwards
path = 'D:/Grogia Tech/Digital Image processing/Project/Demo/new/basic/test/FVC2004/DB1_B'
filename = [1]*80 #ten different fingerprints. eight images for same fingerprint
count = 0
for file in os.listdir(path):
	filename[count] = file
	count += 1
#search all filenames for same images and acheive correct probability
matchedPoints_max = np.array([80,80])
for i in range(80):
	for j in range(80):
		#input
		img1 = cv2.imread(path+'/'+filename[i],0)
		img2 = cv2.imread(path+'/'+filename[j],0)
		#decide if they same fingerprints due to index of files
		if i/8 == j/8:
			print "They are from same fingerprint"
		else:
			print "They are from different fingerprint"
		#preprocess for images
		img_origin1,imgfore1=pre.segmentation(img1)
		img_origin2,imgfore2=pre.segmentation(img2)
		blockSize=16
		theta1=pre.calcDirection(img_origin1,blockSize)
		theta2=pre.calcDirection(img_origin2,blockSize)
		wl1=pre.calcWl(img_origin1,blockSize)
		wl2=pre.calcWl(img_origin2,blockSize)
		img_enhance1=pre.GaborFilter(img_origin1,blockSize,wl1,np.pi/2-theta1)
		img_enhance1[np.where(imgfore1==255)]=255
		img_thin1=pre.thinning(basic.binarize(img_enhance1))
		img_enhance2=pre.GaborFilter(img_origin2,blockSize,wl2,np.pi/2-theta2)
		img_enhance2[np.where(imgfore2==255)]=255
		img_thin2=pre.thinning(basic.binarize(img_enhance2))
        matchedPoints_max[i][j] = minutiaeMatch(img_thin1, img_thin2, imgfore1, imgfore2)
        print matchedPoints_max[i][j]



end=time.clock()
print end-start
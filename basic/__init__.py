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
matchedPoints_max = np.zeros([80,80])
for i in range(0,80):
	for j in range(i+8,80-i-1):
		#input
		img1 = cv2.imread(path+'/'+filename[i],0)
		img2 = cv2.imread(path+'/'+filename[j],0)
		#decide if they same fingerprints due to index of files
		if i/8 == j/8:
			print "They are from same fingerprint"
		else:
			print "They are from different fingerprint"
		#preprocess for images
		img_seg1,imgfore1=pre.segmentation(img1)
		img_seg2,imgfore2=pre.segmentation(img2)
		imgE1=enhance(img1)
		imgE2=enhance(img2)
		imgB1=basic.binarize(imgE1)
		imgB2=basic.binarize(imgE2)
		imgT1=pre.thinning(imgB1)
		imgT2=pre.thinning(imgB2)
		matchedPoints_max[i][j] = minutiaeMatch(imgT1, imgT2, imgfore1, imgfore2)
		print "Matched points: %d\n" % matchedPoints_max[i][j]



end = time.clock()
print end-start

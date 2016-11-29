#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 9:44 2016

@author: wenxinfang

functions for minutiae matching
"""

import cv2
import numpy as np
import math
import copy
import random
import scipy
from matplotlib import pyplot as plt
from minutiaeExtract import minutiaeExtract
import numpy.linalg


def minutiaeMatch(imgT, imgI, imgforeT, imgforeI):
	"""minutiae match:     
	imgTemplete: template image from dataset after thinning
	imgInput: image we need to recognize after thinning
	imgforeT: foreground image for template image
	imgforeI: foreground image for input image
	return: matching score between these two images
	"""
	endingT,bifurT,theta1T,theta2T = minutiaeExtract(imgT,imgforeT) 
	endingI,bifurI,theta1I,theta2I = minutiaeExtract(imgI,imgforeI) #ending,bigur:n*2, theta:n*1
	#ending
	lengthT1 = len(endingT) #n
	lengthI1 = len(endingI)
	closest1T_r1 = [[1,1]]*3
	closest1I_r1 = [[1,1]]*3
	dis1 = [0]*lengthI1
	corresPoint1 = np.zeros([lengthT1,5,2])
	#bifur
	lengthT2 = len(bifurT) #m
	lengthI2 = len(bifurI)
	closest1T_r2 = [[1,1]]*3
	closest1I_r2 = [[1,1]]*3
	dis2 = [0]*lengthI2
	corresPoint2 = np.zeros([lengthT2,5,2])
	#finding correspondences for ending
	#ending
	for i in range(0,lengthT1):
		for j in range(0,lengthI1):
			closest1T1 = findThreeClosest(endingT[i][0],endingT[i][1],endingT)
			closest1I1 = findThreeClosest(endingI[j][0],endingI[j][1],endingI)
			for n in range(0,3):
				closest1T_r1[n] = rotate(endingT[i][0],endingT[i][1],closest1T1[n][0],closest1T1[n][1],theta1T[i])
				closest1I_r1[n] = rotate(endingI[j][0],endingI[j][1],closest1I1[n][0],closest1I1[n][1],theta1I[j])
			dis1[j] = minDistance(closest1T_r1,closest1I_r1)
		dis_temp1 = copy.deepcopy(dis1)
		dis_temp1.sort()
		for n in range(0,5):
			corresPoint1[i][n] = endingI[dis1.index(dis_temp1[n])]
	#bifur
	for i in range(0,lengthT2):
		for j in range(0,lengthI2):
			closest1T2 = findThreeClosest(bifurT[i][0],bifurT[i][1],bifurT)
			closest1I2 = findThreeClosest(bifurI[j][0],bifurI[j][1],bifurI)
			for n in range(0,3):
				closest1T_r2[n] = rotate(bifurT[i][0],bifurT[i][1],closest1T2[n][0],closest1T2[n][1],theta2T[i])
				closest1I_r2[n] = rotate(bifurI[j][0],bifurI[j][1],closest1I2[n][0],closest1I2[n][1],theta2I[j])
			dis2[j] = minDistance(closest1T_r2,closest1I_r2)
		dis_temp2 = copy.deepcopy(dis2)
		dis_temp2.sort()
		for n in range(0,5):
			corresPoint2[i][n] = endingI[dis2.index(dis_temp2[n])]

	#RANSAC algorithm
	iterations = 2000
	iteration = 1
	requiredPoints1 = math.sqrt(lengthT1*lengthI1)/2 #number of matching points required to assert that the transformation fits well
	requiredPoints2 = math.sqrt(lengthT2*lengthI2)/2
	# print "Required matching points:"
	# print int(requiredPoints1) + int(requiredPoints2)
	matchedPoints_max = 0
	R01 = 1 #threshold for sd
	sigma01 = 1/180 #threshold for dd
	from_pt1 = [[0,0]]*3
	R02 = 1 #threshold for sd
	sigma02 = 1/180 #threshold for dd
	from_pt2 = [[0,0]]*3
	random.seed(10)
	for iteration in range(1,iterations+1):
		# print "iteration %d" % iteration
		#take three random points in the template and input set
		to_pt1 = [[1,1]]*3
		to_pt2 = [[1,1]]*3
		for n in range(3):
			indexT1 = int(random.random()*lengthT1)
			indexI1 = int(random.random()*5)
			from_pt1[n] = endingT[indexT1] #random thre points of template set
			to_pt1[n] = corresPoint1[indexT1][indexI1]
			if (to_pt1[0][0]==to_pt1[1][0] and to_pt1[0][1]==to_pt1[1][1]) or (to_pt1[1][0]==to_pt1[2][0] and to_pt1[1][1]==to_pt1[2][1]) or (to_pt1[0][0]==to_pt1[2][0] and to_pt1[0][1]==to_pt1[2][1]):
				continue
		for n in range(3):
			indexT2 = int(random.random()*lengthT2)
			indexI2 = int(random.random()*5)
			from_pt2[n] = bifurT[indexT2] #random thre points of template set
			to_pt2[n] = corresPoint2[indexT2][indexI2]
			if (to_pt2[0][0]==to_pt2[1][0] and to_pt2[0][1]==to_pt2[1][1]) or (to_pt2[1][0]==to_pt2[2][0] and to_pt2[1][1]==to_pt2[2][1]) or (to_pt2[0][0]==to_pt2[2][0] and to_pt2[0][1]==to_pt2[2][1]):
				continue
		#find affine tranformation
		flag1,matrix_a1 = affineParameters(from_pt1, to_pt1)
		flag2,matrix_a2 = affineParameters(from_pt2, to_pt2)
		if flag1 == 0 or flag2 == 0:
			continue
		#apply the transformation to the all template points
		newT1 = [[0,0]]*lengthT1
		for i in range(0,lengthT1):
			newT1[i] = affineCalculate(endingT[i],matrix_a1)    
		newT2 = [[0,0]]*lengthT2
		for i in range(0,lengthT2):
			newT2[i] = affineCalculate(bifurT[i],matrix_a2)     	
		#count number of match points
		d1 = 0 #count of matched points
		d2 = 0
		for i in range(0,lengthT1):
			for j in range(0,lengthI1):
				sd1 = pow(newT1[i][0]-endingI[j][0],2)+pow(newT1[i][1]-endingI[j][1],2)
				dd1 = min(abs(theta1T[i]-theta1I[j]),1-abs(theta1T[i]-theta1I[j]))
				if sd1 <= R01 and dd1 <= sigma01:
					d1 += 1
		for i in range(0,lengthT2):
			for j in range(0,lengthI2):
				sd2 = pow(newT2[i][0]-bifurT[j][0],2)+pow(newT2[i][1]-bifurI[j][1],2)
				dd2 = min(abs(theta2T[i]-theta2I[j]),1-abs(theta2T[i]-theta2I[j]))
				if sd2 <= R02 and dd2 <= sigma02:
					d2 += 1
		if d1+d2 > matchedPoints_max:
			matchedPoints_max = d1 + d2
		# print  d1+d2
		print matchedPoints_max
        return matchedPoints_max


def findThreeClosest(x,y,minutiaePoints):
	"""find three closest minutiae points for one certain point
	x,y: posiiton of certain point
	minutaiePoints: all minutaie points(two kind: ending, bifur) n*2
	return: closest1 3*2
	"""
	#calculate distance for each pair of points
	_length =len(minutiaePoints)
	xy = [[x,y]]*_length
	_xy = np.array(xy); _minutiaePoints = np.array(minutiaePoints)
	sub = _xy - _minutiaePoints
	d1 = [i*i for i in sub.T[0]]; d2 = [i*i for i in sub.T[1]];
	_d1 = np.array(d1); _d2 = np.array(d2)
	distance = _d1 + _d2 

	#sort and find three smallest distance
	_distance = copy.deepcopy(distance)
	_distance.sort()
	firstIndex = distance.tolist().index(_distance[1])
	secondIndex = distance.tolist().index(_distance[2])
	thirdIndex = distance.tolist().index(_distance[3])

	#new array to store indexs
	closest = np.ones([3,2])
	closest[0] = minutiaePoints[firstIndex]
	closest[1] = minutiaePoints[secondIndex]
	closest[2] = minutiaePoints[thirdIndex]

	return closest

def rotate(originx,originy,pointx,pointy,angle):
	"""
	Rotate a point cunterclockwise by a given angle around a given origin.
	"""
	qx=originx+math.cos(angle*180)*(pointx-originx)-math.sin(angle*180)*(pointy-originy)
	qy=originx+math.sin(angle*180)*(pointx-originx)+math.cos(angle*180)*(pointy-originy)
	return qx,qy

def minDistance(closest1T_r,closest1I_r):
	"""
	calculate minimum feature vector distance for each point in templete
	"""
	distance = [1]*9
	for i in range(3):
		for j in range(3):
			distance[i*3+j] = pow(closest1T_r[i][0]-closest1I_r[j][0],2)+pow(closest1T_r[i][1]-closest1I_r[j][1],2)
	d = copy.deepcopy(distance)
	d.sort() #small->big
	minDistance = d[0]
	x0 = distance.index(d[0])
	x0 = x0 / 3
	y0 = distance.index(d[0])
	y0 = y0 % 3
	for n in range(1,9):
		if n/3 != x0 and n%3 != y0:
			minDistance += distance[n]
			break
	return minDistance

def affineParameters(from_pt, to_pt):
	"""
	Find six parameters for affine transformation
	"""
	x1,y1 = from_pt[0]
	x2,y2 = from_pt[1]
	x3,y3 = from_pt[2]
	x1_,y1_ = to_pt[0]
	x2_,y2_ = to_pt[1]
	x3_,y3_ = to_pt[2]	
	matrix_x = [[x1,x2,x3],[y1,y2,y3],[1,1,1]]
	matrix_y = [[x1_,x2_,x3_],[y1_,y2_,y3_],[1,1,1]]
	matrix_x = scipy.mat(matrix_x)
	matrix_y = scipy.mat(matrix_y)
	matrix_a = np.zeros([3,3])
	if numpy.linalg.det(matrix_x) != 0:
		matrix_a = matrix_y * matrix_x.I
		flag = 1
	else:
		flag = 0
	return flag,matrix_a

def affineCalculate(matrix_x,matrix_a):
	"""
	Use six parameters to compute the output points after mapping
	"""
	matrix_a = np.array(matrix_a)
	a11 = matrix_a[0][0]
	a12 = matrix_a[0][1]
	a21 = matrix_a[1][0]
	a22 = matrix_a[1][1]
	b1 = matrix_a[0][2]
	b2 = matrix_a[1][2]
	a = [[a11,a12],[a21,a22]]
	x = [[matrix_x[0]],[matrix_x[1]]]
	b = [[b1],[b2]]
	a = scipy.mat(a)
	x = scipy.mat(x)
	b = scipy.mat(b)
	y = a*x+b
	x_ = y[0]
	y_ = y[1]
	return [x_,y_]

# cython
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
from libc.math cimport sqrt, sin, cos, pow
from minutiaeExtract import minutiaeExtract


def minutiaeMatch(imgT, imgI, imgforeT, imgforeI):
    """minutiae match:     
    imgTemplete: template image from dataset after thinning
    imgInput: image we need to recognize after thinning
    imgforeT: foreground image for template image
    imgforeI: foreground image for input image
    return: matching score between these two images
    """
    endingT, bifurT, theta1T, theta2T = minutiaeExtract(imgT, imgforeT)
    endingI, bifurI, theta1I, theta2I = minutiaeExtract(imgI, imgforeI)
    # ending,bigur:n*2, theta:n*1
    # endingT = np.array(endingT.tolist()+bifurT.tolist())
    # endingI = np.array(endingI.tolist()+bifurI.tolist())
    # theta1T = np.array(theta1T.tolist()+theta2T.tolist())
    # theta1I = np.array(theta1I.tolist()+theta2I.tolist())
    lengthT = len(endingT)  # n
    lengthI = len(endingI)
    length2T = len(bifurT)  # bifur
    length2I = len(bifurI)
    if lengthI > lengthT:
        endingI, endingT = endingT, endingI
        theta1I, theta1T = theta1T, theta1I
        lengthI, lengthT = lengthT, lengthI
    if length2I > length2T:
        bifurI, bifurT = bifurT, bifurI
        theta2I, theta2T = theta2T, theta2I
        length2I, length2T = length2T, length2I
    closest1T_r = [[1, 1]] * 3
    closest1I_r = [[1, 1]] * 3
    closest2T_r = [[1, 1]] * 3
    closest2I_r = [[1, 1]] * 3
    dis = [0] * lengthI
    corresPoint = np.zeros([lengthT, 5, 2])
    dis2 = [0] * length2I
    corresPoint2 = np.zeros([length2T, 5, 2])
    # finding correspondences
    cdef int i, j, n
    endingFlag = 0
    if lengthT >= 5 and lengthI >= 5:
        endingFlag = 1
        for i in range(0, lengthT):
            for j in range(0, lengthI):
                closest1T = findThreeClosest(
                    endingT[i][0], endingT[i][1], endingT)
                closest1I = findThreeClosest(
                    endingI[j][0], endingI[j][1], endingI)
                for n in range(0, 3):
                    closest1T_r[n] = rotate(
                        endingT[i][0], endingT[i][1], closest1T[n][0], closest1T[n][1], theta1T[i])
                    closest1I_r[n] = rotate(
                        endingI[j][0], endingI[j][1], closest1I[n][0], closest1I[n][1], theta1I[j])
                    dis[j] = minDistance(closest1T_r, closest1I_r)
                    dis_temp = copy.deepcopy(dis)
                    dis_temp.sort()
                    for n in range(0, 5):
                        corresPoint[i][n] = endingI[dis.index(dis_temp[n])]
    bifurFlag = 0
    if length2T >= 5 and length2I >= 5:
        bifurFlag = 1
        for i in range(0, length2T):
            for j in range(0, length2I):
                closest2T = findThreeClosest(
                    bifurT[i][0], bifurT[i][1], bifurT)
                closest2I = findThreeClosest(
                    bifurI[j][0], bifurI[j][1], bifurI)
                for n in range(0, 3):
                    closest2T_r[n] = rotate(
                        bifurT[i][0], bifurT[i][1], closest2T[n][0], closest2T[n][1], theta2T[i])
                    closest2I_r[n] = rotate(
                        bifurI[j][0], bifurI[j][1], closest2I[n][0], closest2I[n][1], theta2I[j])
                dis2[j] = minDistance(closest2T_r, closest2I_r)
            dis_temp2 = copy.deepcopy(dis2)
            dis_temp2.sort()
            for n in range(0, 5):
                corresPoint2[i][n] = bifurI[dis2.index(dis_temp2[n])]
    # RANSAC algorithm
    cdef int iterations
    iterations = 2000
    matchedPoints_max = 0
    matchedPoints_max2 = 0
    R0 = 20  # threshold for sd
    sigma0 = 10 * math.pi / 180  # threshold for dd
    from_pt = [[0, 0]] * 3
    from_pt2 = [[0, 0]] * 3
    for iteration in range(iterations):
        # take three random points in the template and input set
        # print iteration
        find = 0
        to_pt = [[1, 1]] * 3
        to_pt2 = [[1, 1]] * 3
        indexT = [random.randint(0, lengthT - 1) for _ in range(3)]
        indexI = [random.randint(0, 4) for _ in range(3)]
        index2T = [random.randint(0, length2T - 1) for _ in range(3)]
        for n in range(3):
            # random thre points of template set
            from_pt[n] = endingT[indexT[n]]
            to_pt[n] = corresPoint[indexT[n]][indexI[n]]
            # random thre points of template set
            from_pt2[n] = bifurT[index2T[n]]
            to_pt2[n] = corresPoint2[index2T[n]][indexI[n]]
        if endingFlag == 1:
            if (to_pt[0][0] == to_pt[1][0] and to_pt[0][1] == to_pt[1][1]) or (to_pt[1][0] == to_pt[2][0] and to_pt[1][1] == to_pt[2][1]) or (to_pt[0][0] == to_pt[2][0] and to_pt[0][1] == to_pt[2][1]):
                continue
        if bifurFlag == 1:
            if (to_pt2[0][0] == to_pt2[1][0] and to_pt2[0][1] == to_pt2[1][1]) or (to_pt2[1][0] == to_pt2[2][0] and to_pt2[1][1] == to_pt2[2][1]) or (to_pt2[0][0] == to_pt2[2][0] and to_pt2[0][1] == to_pt2[2][1]):
                continue
        # find affine tranformation
        flag, matrix_a = affineParameters(from_pt, to_pt)
        flag2, matrix_a2 = affineParameters(from_pt2, to_pt2)
        if flag == 0 or flag2 == 0:
            continue
        # apply the transformation to the all template points
        newT = [[0, 0]] * lengthT
        if endingFlag == 1:
            for i in range(0, lengthT):
                newT[i] = affineCalculate(endingT[i], matrix_a)
        new2T = [[0, 0]] * length2T
        if bifurFlag == 1:
            for i in range(0, length2T):
                new2T[i] = affineCalculate(bifurT[i], matrix_a2)
        # count number of match points
        d1 = 0  # count of matched points
        if endingFlag == 1:
            for i in range(0, lengthT):
                for j in range(0, lengthI):
                    sd = sqrt(
                        pow(newT[i][0] - endingI[j][0], 2) + pow(newT[i][1] - endingI[j][1], 2))
                    dd = min(
                        abs(theta1T[i] - theta1I[j]), 2 * math.pi - abs(theta1T[i] - theta1I[j]))
                    # print sd,dd
                    if sd <= R0 and dd <= sigma0:
                        d1 += 1
            if d1 > matchedPoints_max:
                matchedPoints_max = d1
        d2 = 0  # count of matched points
        if bifurFlag == 1:
            for i in range(0, length2T):
                for j in range(0, length2I):
                    sd = sqrt(
                        pow(new2T[i][0] - bifurI[j][0], 2) + pow(new2T[i][1] - bifurI[j][1], 2))
                    dd = min(
                        abs(theta2T[i] - theta2I[j]), 2 * math.pi - abs(theta2T[i] - theta2I[j]))
                    # print sd,dd
                    if sd <= R0 and dd <= sigma0:
                        d2 += 1
            if d2 > matchedPoints_max2:
                matchedPoints_max2 = d2
        # print  d1+d2
    print matchedPoints_max, matchedPoints_max2
    print lengthT, length2T
    if bifurFlag == 1 and endingFlag == 1:
        score = (float(matchedPoints_max) / lengthT +
                 float(matchedPoints_max2) / length2T) / 2
    else:
        score = float(matchedPoints_max) / lengthT
    return matchedPoints_max + matchedPoints_max2, score


cdef inline double[:, ::1] findThreeClosest(double x, double y, minutiaePoints):
    """find three closest minutiae points for one certain point
    x,y: posiiton of certain point
    minutaiePoints: all minutaie points(two kind: ending, bifur) n*2
    return: closest1 3*2
    """
    # calculate distance for each pair of points
    _length = len(minutiaePoints)
    xy = [[x, y]] * _length
    _xy = np.array(xy)
    _minutiaePoints = np.array(minutiaePoints)
    sub = _xy - _minutiaePoints
    cdef int i, j
    d1 = [i * i for i in sub.T[0]]
    d2 = [i * i for i in sub.T[1]]
    _d1 = np.array(d1)
    _d2 = np.array(d2)
    distance = [_d1 + _d2]

    # sort and find three smallest distance
    _distance = copy.deepcopy(distance)
    _distance = np.sort(_distance)
    # temp, firstIndex = np.where(distance == _distance[0][1])
    # temp, secondIndex = np.where(distance == _distance[0][2])
    # temp, thirdIndex = np.where(distance == _distance[0][3])

    # new array to store indexs
    cdef double[::1] IndexArray = np.zeros(3)

    for i in range(3):
        aaa = np.where(distance == _distance[0, i - 1])[1]
        if len(aaa) == 1:
            IndexArray[i] = aaa
        else:
            IndexArray[i] = aaa[0]

    cdef double[:, ::1] closest = np.ones([3, 2])
    for i in range(3):
        for j in range(2):
            closest[i, j] = minutiaePoints[IndexArray[i], j]
#    closest[0] = minutiaePoints[firstIndex[0]]
#    closest[1] = minutiaePoints[secondIndex[0]]
#    closest[2] = minutiaePoints[thirdIndex[0]]

    return closest


cdef inline double[::1] rotate(double originx, double originy, double pointx, double pointy, double angle):
    """
    Rotate a point cunterclockwise by a given angle around a given origin.
    """
    cdef double[::1] q = np.zeros(2)
    q[0] = originx + cos(angle) * (pointx - originx) - \
        sin(angle) * (pointy - originy)
    q[1] = originx + sin(angle) * (pointx - originx) + \
        cos(angle) * (pointy - originy)
    return q


cdef inline double minDistance(closest1T_r, closest1I_r):
    """
    calculate minimum feature vector distance for each point in templete
    """
    distance = [1] * 9
    cdef int i, j
    for i in range(3):
        for j in range(3):
            distance[i * 3 + j] = pow(closest1T_r[i][0] - closest1I_r[j][0], 2) + pow(
                closest1T_r[i][1] - closest1I_r[j][1], 2)
    d = copy.deepcopy(distance)
    d.sort()  # small->big
    cdef double  minDistance = d[0]
    x0 = distance.index(d[0])
    x0 = x0 / 3
    y0 = distance.index(d[0])
    y0 = y0 % 3
    cdef int n
    for n in range(1, 9):
        if n / 3 != x0 and n % 3 != y0:
            minDistance += distance[n]
            break
    return minDistance


def affineParameters(from_pt, to_pt):
    """
    Find six parameters for affine transformation
    """
    a = np.asarray(from_pt)
    b = np.asarray(to_pt)
    # x1, y1 = from_pt[0]
    # x2, y2 = from_pt[1]
    # x3, y3 = from_pt[2]
    # x1_, y1_ = to_pt[0]
    # x2_, y2_ = to_pt[1]
    # x3_, y3_ = to_pt[2]
    # matrix_x = [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]
    # matrix_y = [[x1_, x2_, x3_], [y1_, y2_, y3_], [1, 1, 1]]
    # matrix_x = scipy.mat(matrix_x)
    # matrix_y = scipy.mat(matrix_y)
    matrix_x = np.vstack((a.T, np.ones((1, 3))))
    matrix_y = np.vstack((b.T, np.ones((1, 3))))
    matrix_a = np.zeros((3, 3))
    if np.linalg.det(matrix_x) != 0:
        matrix_a = np.dot(matrix_y, np.linalg.inv(matrix_x))
        flag = 1
    else:
        flag = 0
    return flag, matrix_a


cdef inline affineCalculate(matrix_x, matrix_a):
    """
    Use six parameters to compute the output points after mapping
    """
    # matrix_a = np.array(matrix_a)
    # a11 = matrix_a[0][0]
    # a12 = matrix_a[0][1]
    # a21 = matrix_a[1][0]
    # a22 = matrix_a[1][1]
    # b1 = matrix_a[0][2]
    # b2 = matrix_a[1][2]
    # a = [[a11, a12], [a21, a22]]
    # x = [[matrix_x[0]], [matrix_x[1]]]
    # b = [[b1], [b2]]
    # a = scipy.mat(a)
    # x = scipy.mat(x)
    # b = scipy.mat(b)
    # y = a * x + b
    # x_ = y[0]
    # y_ = y[1]
    y = np.dot(matrix_a[:, 0:-1], matrix_x.reshape((2, 1))).T + matrix_a[:, -1]
    return y[0]

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 03:44:31 2016

@author: zhangzimou
"""
correct=0
wrong=0
a=filename[0][0:3]
for i in range(80):
    if filename[i][0:3]==a:
        if score[i]>=0.3:
            correct+=1
            print score[i]
        else:
            wrong+=1
    else:
        if score[i]<0.3:
            correct+=1
        else:
            wrong+=1
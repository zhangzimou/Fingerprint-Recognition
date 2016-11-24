#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:10:15 2016

@author: zhangzimou
"""
#
cimport cython


cdef double recip_square(double i):
    return 1./i**2

def approx_pi(int n=10000000):
    cdef double val = 0.
    cdef int k
    #val=0
    for k in range(1,n+1):
        val += recip_square(k)
    return (6 * val)**.5
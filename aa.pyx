#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 13:56:09 2016

@author: zhangzimou
"""


cimport cython

def recip_square(i):
    return 1./i**2

def approx_pi(n=10000000):
    #cdef double val = 0.
    #cdef int k
    val=0
    for k in xrange(1,n+1):
        val += recip_square(k)
    return (6 * val)**.5
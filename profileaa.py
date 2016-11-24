#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:57:49 2016

@author: zhangzimou
"""

import pstats, cProfile
#import pyximport
#pyximport.install()
import aa
import time
cProfile.runctx("aa.approx_pi()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
start=time.clock()
aa.approx_pi()
end=time.clock()
print end-start
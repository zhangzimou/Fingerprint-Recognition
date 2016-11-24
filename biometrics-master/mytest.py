

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('../')

from initial import imshow
import numpy as np
from matplotlib import pyplot as plt
import cv2
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
import initial as init
import gabor

img=cv2.imread('pic1.png',0)


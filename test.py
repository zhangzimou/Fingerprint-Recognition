import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d
import time
from initial import imshow
from initial import normalize2
start=time.clock()
img=cv2.imread('pic1.png',0)
w=16
#img=img[96:96+w,400:400+w]
sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
par_x=convolve2d(img,sobel_x,mode='same')
par_y=convolve2d(img,sobel_y,mode='same')

N,M=np.shape(img)
Vx=np.zeros((N,M))
Vy=np.zeros((N,M))

for i in range(w/2,N-w/2):
    for j in range(w/2,M-w/2):
        a=(i-w/2);b=a+w;c=(j-w/2);d=c+w
        Vy[i,j]=2*np.sum(par_x[a:b,c:d]*par_y[a:b,c:d])
        Vx[i,j]=-np.sum(par_y[a:b,c:d]**2-par_x[a:b,c:d]**2)
theta=0.5*np.arctan2(Vy,Vx)#+np.pi/2
h=np.ones((15,15))/(15.0*15.0)
theta_filter=convolve2d(theta,h,mode='same')
theta_small=theta_filter[0:N:w,0:M:w]
end=time.clock()
print end-start
X,Y=np.mgrid[0:N:w,0:M:w]
plt.figure()
plt.quiver(Y,X,np.cos(theta_small),np.sin(theta_small),color='r')
imshow(img)
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((400,96),w,w,fill=None,color='b'))
plt.show()
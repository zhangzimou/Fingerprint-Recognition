#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:57:34 2016

@author: zhangzimou
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from scipy.signal import convolve2d

def block_view(A, block= (3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)
def blockproc(M, fun, blk_size=(3,3),singleout=False):
    # This is some complex function of blk_size and M.shape
    #stride = blk_size
    output = np.zeros(M.shape if (singleout==False) else (M.shape[0]/blk_size[0],M.shape[1]/blk_size[1]))
    B = block_view(M, block=blk_size)
    if (singleout==False):
        O = block_view(output, block=blk_size)
    else:
        O = output
    for b,o in zip(B, O):
        #o[:,:] = fun(b);
        if (singleout==False):
            o[:,:]=np.asarray(map(lambda x:fun(x),b))
        else:
            o[:]=np.asarray(map(lambda x:fun(x),b))
    return output

def enhance(img,blockSize=16):
    img,imgfore=segmentation(img)
    theta=calcDirection(img,blockSize)
    wl=calcWl(img,blockSize)
    #img=ridgeComp2(img,theta,blockSize)
    img=GaborFilter(img,blockSize,wl,np.pi/2-theta)
    img[np.where(imgfore==255)]=255
    #img[np.where(imgfore==255)]=255
    return img

def inverse(img):
    return 255-img


  
    
def foreground_(img,blockSize,factor=0.9):
    img_block=block_view(img,(blockSize,blockSize))
    imgout=np.zeros_like(img)
    out_block=block_view(imgout,(blockSize,blockSize))
    thresizehold=np.min(img)+factor*(np.max(img)-np.min(img))
    for b,o in zip(img_block,out_block):
        a=np.asarray(map(lambda x:np.mean(x),b))
        a[np.where(a>=thresizehold)]=255;a[np.where(a<thresizehold)]=0
        o[:,:]=np.asarray(map(lambda x,y:x*y,np.ones(o.shape),a))
    return imgout
    
def foreground(img,blockSize=31):
    img=100*(img-np.mean(img))
    img[np.where(img>255)]=255
    img=cv2.boxFilter(img,-1,(blockSize,blockSize))
    img[np.where(img>150)]=255; img[np.where(img<=150)]=0   
    img=cv2.boxFilter(img,-1,(blockSize/2,blockSize/2))
    img[np.where(img>0)]=255
    return img

def segmentation(img,h=480,w=320):
    imgfore=foreground(img)
    index=np.where(imgfore==0)
    nmin=np.min(index[0]);nmax=np.max(index[0]);mmin=np.min(index[1]);mmax=np.max(index[1])
    #midpoint=(np.array([(nmin+nmax)/2]),np.array([(mmin+mmax)/2]))
    nmid=(nmin+nmax)/2;mmid=(mmin+mmax)/2
    a=nmid-h/2;b=nmid+h/2;c=mmid-w/2;d=mmid+w/2
    if (a<=0):
        a=0;b=h
    if (b>=img.shape[0]):
        b=img.shape[0];a=b-h
    if (c<=0):
        c=0;d=w
    if (d>=img.shape[1]):
        d=img.shape[1];c=d-w
    return img[a:b,c:d],imgfore[a:b,c:d]

def blkWlDire(img):
    """Calculate wavelength and direction given an image block"""
    f=np.abs(fftshift(fft2(img)))
    origin=np.where(f==np.max(f));f[origin]=0;mmax=np.where(f==np.max(f))
    dire=np.arctan2(origin[0]-mmax[0][0],origin[1]-mmax[1][0])
    wl=2*img.shape[0]/(((origin[0]-mmax[0][0])*2)**2+((origin[1]-mmax[1][0])*2)**2)**0.5
    return wl,dire

def calcWlDire(img,blockSize):
    wl=np.zeros((img.shape[0]/blockSize,img.shape[1]/blockSize))
    dire=np.zeros((img.shape[0]/blockSize,img.shape[1]/blockSize))
    B=block_view(img,(blockSize,blockSize))
    for w,d,b in zip(wl,dire,B):
        a=map(lambda x: blkWlDire(x),b)
        w[:]=map(lambda x: x[0],a)
        d[:]=map(lambda x: x[1],a)
    return wl,dire

def blkwl(img):
    """Calculate wavelength  given an image block"""
    f=np.abs(fftshift(fft2(img)))
    origin=np.where(f==np.max(f));f[origin]=0;mmax=np.where(f==np.max(f))
    wl=2*img.shape[0]/(((origin[0]-mmax[0][0])*2)**2+((origin[1]-mmax[1][0])*2)**2)**0.5
    return wl

def calcWl(img,blockSize):
    wl=np.zeros((img.shape[0]/blockSize,img.shape[1]/blockSize))
    B=block_view(img,(blockSize,blockSize))
    for w,b in zip(wl,B):
        w[:]=map(lambda b: blkwl(b),b)
    gaussianBlurSigma=4;    gaussian_block=7
    wl=cv2.GaussianBlur(wl,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    return wl
    
def GaborFilter_(img,blockSize,wl,dire,sigma=20):
    imgout=np.zeros_like(img)
    O=block_view(imgout,(blockSize,blockSize))
    B=block_view(img,(blockSize,blockSize))
    for w,d,o,b in zip(wl,dire,O,B):
        kernel=map(lambda w,d:cv2.getGaborKernel((blockSize,blockSize),sigma,d,w,1),w,d)
        o[:,:]=np.asarray(map(lambda x,kernel: cv2.filter2D(x,-1,kernel),b,kernel))
    return imgout

#def applyKernel(img,kernel,i,j):
    
    

def GaborFilter(img,blockSize,wl,dire,sigma=20):
    img=img.astype(np.float64)
    imgout=img.copy()
    kernel=np.zeros((img.shape[0]/blockSize*(blockSize+1),img.shape[1]/blockSize*(blockSize+1)))
    K=block_view(kernel,(blockSize+1,blockSize+1))
    for k,w,d in zip(K,wl,dire):
        k[:,:]=np.asarray(map(lambda w,d: cv2.getGaborKernel((blockSize+1,blockSize+1),sigma,d,w,1),w,d))
    for i in xrange(blockSize/2,img.shape[0]-blockSize/2):
        block_i=i/blockSize
        for j in xrange(blockSize/2,img.shape[1]-blockSize/2):
            block_j=j/blockSize
            imgout[i,j]=np.sum(K[block_i,block_j][::-1,::-1]
                        *img[i-blockSize/2:i+blockSize/2+1,j-blockSize/2:j+blockSize/2+1])
            
    imgout[np.where(imgout>255)]=255;imgout[np.where(imgout<0)]=0
    return imgout
            
def calcDirection(img,blockSize):
    sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
    par_x=convolve2d(img,sobel_x,mode='same')
    par_y=convolve2d(img,sobel_y,mode='same')
    N,M=np.shape(img)
    Vx=np.zeros((N/blockSize,M/blockSize))
    Vy=np.zeros((N/blockSize,M/blockSize))
    for i in xrange(N/blockSize):
        for j in xrange(M/blockSize):
            a=i*blockSize;b=a+blockSize;c=j*blockSize;d=c+blockSize
            Vy[i,j]=2*np.sum(par_x[a:b,c:d]*par_y[a:b,c:d])
            Vx[i,j]=np.sum(par_y[a:b,c:d]**2-par_x[a:b,c:d]**2)
    gaussianBlurSigma=2;    gaussian_block=5
    Vy=cv2.GaussianBlur(Vy,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    Vx=cv2.GaussianBlur(Vx,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    theta=0.5*np.arctan2(Vy,Vx)#+np.pi/2            
    return theta
    
def ridgeComp(img,theta, blockSize,w=3,h=9,alpha=100,beta=1):
    resize=5
    N,M=np.shape(img)
    imgout=np.zeros_like(img)
    imgresizeize=cv2.resizeize(img,None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
    mask=np.ones((w,h))*beta
    mask[(w-1)/2]=np.ones((1,h))*alpha
    ww=np.arange(-(w-1)/2,(w-1)/2+1)
    hh=np.arange(-(h-1)/2,(h-1)/2+1)
    hh,ww=np.meshgrid(hh,ww)
    for i in xrange((h-1)/2,N-(h-1)/2):
        block_i=i/blockSize
        for j in xrange((h-1)/2,M-(h-1)/2):
            block_j=j/blockSize
            thetaHere=theta[block_i,block_j]
            ii=np.round((i+ww*np.cos(thetaHere)-hh*np.sin(thetaHere))*resize).astype(np.int32)
            jj=np.round((j+ww*np.sin(thetaHere)+hh*np.cos(thetaHere))*resize).astype(np.int32)
            imgout[i,j]=np.sum(imgresizeize[ii,jj]*mask)/(((w-1)*beta+alpha)*h)

def ridgeComp2(img,theta,blockSize,h=15):
    resize=5
    N,M=np.shape(img)
    imgout=img.copy()
    imgresize=cv2.resize(img,None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
    hh=np.arange(-(h-1)/2,(h-1)/2+1)
    for i in xrange((h-1)/2,N-(h-1)/2):
        block_i=i/blockSize
        for j in xrange((h-1)/2,M-(h-1)/2):
            block_j=j/blockSize
            thetaHere=theta[block_i,block_j]
            ii=np.round((i-hh*np.sin(thetaHere))*resize).astype(np.int32)
            jj=np.round((j+hh*np.cos(thetaHere))*resize).astype(np.int32)
            imgout[i,j]=np.mean(imgresize[ii,jj])
    return imgout

def binarize2(img,theta,blockSize,h=9):
    resize=5
    N,M=np.shape(img)
    imgout=np.zeros_like(img)
    imgresizeize=cv2.resizeize(img,None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
    blockMean=blockproc(img,np.mean,(blockSize,blockSize),True)
    hh=np.arange(-(h-1)/2,(h-1)/2+1)
    for i in xrange((h-1)/2,N-(h-1)/2):
        block_i=i/blockSize
        for j in xrange((h-1)/2,M-(h-1)/2):
            block_j=j/blockSize
            thetaHere=theta[block_i,block_j]
            ii=np.round((i-hh*np.sin(thetaHere))*resize).astype(np.int32)
            jj=np.round((j+hh*np.cos(thetaHere))*resize).astype(np.int32)
            imgout[i,j]=255 if (np.mean(imgresizeize[ii,jj])>blockMean[block_i,block_j]) else 0
    return imgout


    
def binarize(img):
    imgout=np.zeros_like(img)
    imgout[np.where(img>np.mean(img))]=0
    imgout[np.where(img<=np.mean(img))]=1
    return imgout


    
def imshow(img):
    plt.imshow(img,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    
def normalize(img):
    a=np.max(img)
    b=np.min(img)
    return (img-b)*255.0/(a-b)
def normalize2(img,mu=125,sigma=128):
    m=np.mean(img)
    std=np.std(img)
    if (std==0):
        return img
    return mu+sigma/std*(img-m)

def fill(img,position,ifending=1,newvalue=2):
    image=img.copy()
    N=image.shape[0]
    a,b=position[0],position[1]
    image[a,b]=newvalue
    while(1):
#        neighbor[:] =np.array([ image[a-1,b],image[a-1,b+1],image[a,b+1],
#                                    image[a+1,b+1],image[a+1,b],image[a+1,b-1],
#                                    image[a,b-1],image[a-1,b-1] ])
        origin=np.array([a,b])
        a=(np.asarray(np.where(image[a-1:a+2,b-1:b+2]==1)).T+np.array([a-1,b-1])).T
        a,b=a[0],a[1]
        if (ifending):   
            if (len(a)!=1 and len(a)!=2):
                return image,np.array([None,None])
            image[(a,b)]=newvalue
            if (len(a)==1):
                a,b=a[0],b[0]
            else:
                a1,a2=np.array([a[0],b[0]]),np.array([a[1],b[1]])
                (a,b)=(a[0],b[0]) if np.sum(np.abs(a1-origin))>np.sum(np.abs(a2-origin)) else (a[1],b[1])
            if (a==0 or a==N-1 or b==0 or b==N-1):
                return image,np.array([a,b])
        else:
            if (len(a)!=3 and len(a)!=4):
                return image,np.array([None,None])
            elif (len(a)==3):
                a1,b1,a2,b2,a3,b3=a[0],b[0],a[1],b[1],a[2],b[2]
            else:
                d=np.zeros(4)
                a=np.vstack((a,b)).T
                for i in xrange(4):
                    d[i]=np.sum(map(lambda x: np.abs(a[i]-x),np.delete(a,i,0)))
                index=np.argmin(d)
                image[a[index][0],a[index][1]]=0
                a=np.delete(a,np.argmin(d),0)
                a1,b1,a2,b2,a3,b3=a[0,0],a[0,1],a[1,0],a[1,1],a[2,0],a[2,1]
            image[a1,b1],image[a2,b2],image[a3,b3]=2,3,4
            image1,pos1=fill(image,(a1,b1),1,2)
            image2,pos2=fill(image1,(a2,b2),1,3)
            image3,pos3=fill(image2,(a3,b3),1,4)
            return image3,np.vstack((pos1,pos2,pos3))
            
def countCrossNum(img,ifending=1,value=2):
    image=img.copy()
    image[np.where(image!=value)]=0
    N,M=img.shape
    series=np.hstack((image[0,0:-1], image[0:-1,-1], image[-1,-1:0:-1], image[-1:0:-1,0]))
    seriesShift=np.zeros_like(series)
    seriesShift[1:],seriesShift[0]=series[0:-1],series[-1]
    count1=(np.sum(np.abs(series-seriesShift))/value/2).astype(int)
    if (ifending):
        return count1
    else:
        return count1+countCrossNum(img,1,3)+countCrossNum(img,1,4)
        

            
def validateMinutiae(imgB,minutiae,ifending,blockSize=15):
    """validate minutiae, and calculate directions at the same
    return: validated minutiae, directions of minutiae
    """
    fill_output=np.asarray(map(lambda x:fill(imgB[x[0]-blockSize/2:x[0]+blockSize/2+1,
                                       x[1]-blockSize/2:x[1]+blockSize/2+1],
                                    (blockSize/2,blockSize/2),ifending), minutiae ))
    imgF,position=fill_output[:,0],fill_output[:,1]
    CN=np.asarray(map(lambda x:countCrossNum(x,ifending),imgF))
    if (ifending):
        valid=np.where(CN==1)
        position=position[valid]
        theta=np.asarray(map(lambda x:np.arctan2(blockSize/2-x[0],blockSize/2-x[1]),position))
        return minutiae[valid],theta
    else:
        valid=np.where(CN==3)
        position=position[valid]
        theta=np.asarray(map(lambda x:bifurDirec(x),position))        
        return minutiae[valid],theta
def bifurDirec(position,blockSize=15):
    theta=np.arctan2(blockSize/2-position[:,0],blockSize/2-position[:,1])
    d=np.zeros(3)
    for i in range(3):
        d[i]=np.sum(np.abs(theta[i]-np.delete(theta,i)))
    return theta[np.argmax(d)]
#def minutiaeDirec(imgB,minutiae,ifending,blockSize=15):
    
    
def thinning(img):
    """
    Zhang-Suen thinning algorithm
    """
    image=img.copy()
    while(1):
        P2,P3,P4,P5,P6,P7,P8,P9 = image[2:,1:-1], image[2:,2:], image[1:-1,2:], image[:-2,2:], image[:-2,1:-1],image[:-2,:-2], image[1:-1,:-2], image[2:,:-2]
        condition0 = image[1:-1,1:-1]
        condition4 = P4*P6*P8
        condition3 = P2*P4*P6
        condition2 = transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9) == 1
        condition1 = (2 <= P2+P3+P4+P5+P6+P7+P8+P9) * (P2+P3+P4+P5+P6+P7+P8+P9 <= 6)
        cond = (condition0 == 1) * (condition4 == 0) * (condition3 == 0) * (condition2 == 1) * (condition1 == 1)
        changing1 = np.where(cond == 1)
        if (len(changing1[0])==0):
            flag1=1
        else: 
            flag1=0
            image[changing1[0]+1,changing1[1]+1] = 0
        # step 2
        P2,P3,P4,P5,P6,P7,P8,P9 = image[2:,1:-1], image[2:,2:], image[1:-1,2:], image[:-2,2:], image[:-2,1:-1], image[:-2,:-2], image[1:-1,:-2], image[2:,:-2]
        condition0 = image[1:-1,1:-1]
        condition4 = P2*P6*P8
        condition3 = P2*P4*P8
        condition2 = transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9) == 1
        condition1 = (2 <= P2+P3+P4+P5+P6+P7+P8+P9) * (P2+P3+P4+P5+P6+P7+P8+P9 <= 6)
        cond = (condition0 == 1) * (condition4 == 0) * (condition3 == 0) * (condition2 == 1) * (condition1 == 1)
        changing2 = np.where(cond == 1)
        if (len(changing2[0])==0):
            flag2=1
        else:
            flag2=0
            image[changing2[0]+1,changing2[1]+1] = 0
        if (flag2 and flag1):
            break
    return image

def transitions_vec(P2, P3, P4, P5, P6, P7, P8, P9):
    return ((P3-P2) > 0).astype(int) + ((P4-P3) > 0).astype(int) + \
    ((P5-P4) > 0).astype(int) + ((P6-P5) > 0).astype(int) + \
    ((P7-P6) > 0).astype(int) + ((P8-P7) > 0).astype(int) + \
    ((P9-P8) > 0).astype(int) + ((P2-P9) > 0).astype(int)



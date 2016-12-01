#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:14:54 2016

@author: zhangzimou

functions for preprocessing:
    foreground detection
    segmentation
    direction calculation
    ridge wavelength calculation
    Gabor filter
    ridge compensation
    thinning
    count cross number(CN)
"""
import pyximport
pyximport.install()

import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from numpy.fft import fft2
from numpy.fft import ifft2
from numpy.fft import fftshift
from scipy.signal import convolve2d
from basic import block_view
from basic import blockproc
import basic

import _preprocess as _pre

def enhance(img,blockSize=8,boxSize=4):
    """image enhancement
    return: enhanced image
    """
#    img=cv2.equalizeHist(np.uint8(img))
    img,imgfore=segmentation(img)
#    img=blockproc(np.uint8(img),cv2.equalizeHist,(16,16))
    img=img.copy(order='C').astype(np.float64)
    theta=_pre.calcDirectionBox(img,blockSize,boxSize)
    wl=calcWlBox(img,blockSize,boxSize)
    
    img=_pre.GaborFilterBox(img,blockSize,boxSize,wl,np.pi/2-theta)
    img=_pre.GaborFilterBox(img,blockSize,boxSize,wl,np.pi/2-theta)
    img=_pre.GaborFilterBox(img,blockSize,boxSize,wl,np.pi/2-theta)
    img=_pre.GaborFilterBox(img,blockSize,boxSize,wl,np.pi/2-theta)
    img=_pre.GaborFilterBox(img,blockSize,boxSize,wl,np.pi/2-theta)
    
    img=np.asarray(img)
    imgfore=cv2.erode(imgfore,np.ones((8,8)),iterations=4)
    img[np.where(imgfore==0)]=255
    img=basic.truncate(img,method='default')
    
    return img,imgfore

 
def foreground(img,blockSize=31):
    """calculate foreground in an image
    return: foreground
    """
    img=100*(img-np.mean(img))
    img[np.where(img>255)]=255
    img=cv2.boxFilter(img,-1,(blockSize,blockSize))
    img[np.where(img>150)]=255; img[np.where(img<=150)]=0   
    img=cv2.boxFilter(img,-1,(blockSize/2,blockSize/2))
    img[np.where(img>0)]=255
    return img

def align(img, valid):
    N,M=np.shape(img)
    index=np.where(valid==1)
    nmin=np.min(index[0]);nmax=np.max(index[0]);mmin=np.min(index[1]);mmax=np.max(index[1])
    nmid,mmid=(nmin+nmax)/2,(mmin+mmax)/2
    # move the foreground to the center of the image
    if nmid-N/2>0:
        img=np.vstack(
                      (
                       img,
                       255*np.ones((nmid-N/2, img.shape[1]))
                       )
                      )
        img=np.delete(img,range(nmid-N/2),axis=0)
    elif N/2-nmid>0:
        img=np.vstack(
                      (
                       255*np.ones((N/2-nmid, img.shape[1])),
                       img
                       )
                      )
        img=np.delete(img,range(img.shape[0]+nmid-N/2,img.shape[0]),axis=0)
    if mmid-M/2>0:
        img=np.hstack(
                      (
                       img,
                       255*np.ones((img.shape[0],mmid-M/2 ))
                       )
                      )
        img=np.delete(img,range(mmid-N/2),axis=1)
    elif M/2-mmid>0:
        img=np.hstack(
                      (
                       255*np.ones((img.shape[0], M/2-mmid)),
                       img
                       )
                      )
        img=np.delete(img,range(img.shape[1]+mmid-M/2,img.shape[1]),axis=1)
    
    sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
    par_x=convolve2d(valid,sobel_x,mode='same')
    par_y=convolve2d(valid,sobel_y,mode='same')
    Vy=2*np.sum(par_x*par_y)
    Vx=np.sum(par_y**2-par_x**2)
    # the direction of the foreground
    theta=0.5*np.arctan2(Vy,Vx)    
    # if theta is between 45 degree and -45 degree, then we think there would 
    # be some problems calculating the direction.
    if np.abs(theta)<np.pi/4:
        return img
    elif theta < -np.pi/4:
        theta += np.pi
    Matrix=cv2.getRotationMatrix2D((M/2,N/2),(np.pi/2-theta)*180/np.pi,1)    
    img=cv2.warpAffine(img,Matrix,(M,N),borderValue=255)
    return img
 
def segmentation(img, blockSize=8, h=352, w=288):
    add0=(16-img.shape[0]%16)/2
    add1=(16-img.shape[1]%16)/2
    img=np.vstack((  255*np.ones((add0,img.shape[1])), img, 255*np.ones((add0,img.shape[1]))  ))
    img=np.hstack((  255*np.ones((img.shape[0],add1)), img, 255*np.ones((img.shape[0],add1))  ))
#    img=np.uint8(img)
    ## reference: IMPROVED FINGERPRINT IMAGE SEGMENTATION USING NEW MODIFIED GRADIENT
    #               BASED TECHNIQUE
    sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
    par_x=convolve2d(img,sobel_x,mode='same')
    par_y=convolve2d(img,sobel_y,mode='same')
    #img=basic.blockproc(img,cv2.equalizeHist,(blockSize,blockSize))
    stdx=blockproc(par_x,np.std,(16,16),True)
    stdy=blockproc(par_y,np.std,(16,16),True)
    grddev=stdx+stdy
    threshold=90
    index=grddev[1:-1,1:-1].copy()
    index[np.where(index<threshold)]=0
    index[np.where(index>=threshold)]=1
    a=np.zeros(grddev.shape)
    a[1:-1,1:-1]=index
    index=a
          
    valid=np.zeros(img.shape)
    valid_b=block_view(valid,(16,16))
    valid_b[:]=index[:,:,np.newaxis,np.newaxis]
    
    kernel = np.ones((8,8),np.uint8)
    # first dilate to delete the invalid value inside the fingerprint region
    valid=cv2.dilate(valid,kernel,iterations = 5)
    # then erode more to delete the valid value outside the fingerprint region
    valid=cv2.erode(valid, kernel, iterations = 12)
    # dilate again to increase the valid value area in compensate for the lose
    # due to erosion in the last step
    valid=cv2.dilate(valid, kernel, iterations=7)

    img[np.where(valid==0)]=255
    # align the image    
    #img=align(img, valid)         
    return cut(img, valid, h, w)
    
def cut(img, valid, h=352,w=288):
    """segment an image into given size
    return: segmented image, segmented foreground
    """
    index=np.where(valid==1)
    nmin=np.min(index[0]);nmax=np.max(index[0]);mmin=np.min(index[1]);mmax=np.max(index[1])
    nmid,mmid=(nmin+nmax)/2,(mmin+mmax)/2
    a=nmid-h/2;b=nmid+h/2;c=mmid-w/2;d=mmid+w/2
    if (a<=0):
        img=np.vstack( ( 
                        255*np.ones((-a, img.shape[1]))
                        ,img
                        )
                     )
        valid=np.vstack( ( 
                        np.zeros((-a, valid.shape[1]))
                        ,valid
                        )
                     )
        a=0;b=h
    if (b>=img.shape[0]):
        img=np.vstack(
                      (
                       img,
                       255*np.ones((b-img.shape[0], img.shape[1]))
                       )
                      )
        valid=np.vstack(
                      (
                       valid,
                       np.zeros((b-valid.shape[0], valid.shape[1]))
                       )
                      )
        #b=img.shape[0];a=b-h
    if (c<=0):
        img=np.hstack(
                      (
                       255*np.ones((img.shape[0],-c)),
                       img
                       )
                      )
        valid=np.hstack(
                      (
                       np.zeros((valid.shape[0],-c)),
                       valid
                       )
                      )
        c=0;d=w
    if (d>=img.shape[1]):
        img=np.hstack(
                      (
                       img,
                       255*np.ones((img.shape[0],d-img.shape[1]))
                       )
                      )
        valid=np.hstack(
                      (
                       valid,
                       np.zeros((valid.shape[0],d-valid.shape[1]))
                       )
                      )
        #d=img.shape[1];c=d-w
    return img[a:b,c:d],valid[a:b,c:d]

def calcDirection(img,blockSize,method='block-wise'):
    """calculate ridge directions in an image, using gradient method
    return: ridge directions
    """
    sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
    par_x=convolve2d(img,sobel_x,mode='same')
    par_y=convolve2d(img,sobel_y,mode='same')
    N,M=np.shape(img)
    if method=='block-wise':
        Vx=np.zeros((N/blockSize,M/blockSize))
        Vy=np.zeros((N/blockSize,M/blockSize))
        for i in xrange(N/blockSize):
            for j in xrange(M/blockSize):
                a=i*blockSize;b=a+blockSize;c=j*blockSize;d=c+blockSize
                Vy[i,j]=2*np.sum(par_x[a:b,c:d]*par_y[a:b,c:d])
                Vx[i,j]=np.sum(par_y[a:b,c:d]**2-par_x[a:b,c:d]**2)
        
    elif method=='pixel-wise':
        Vx,Vy=np.zeros((N,M)),np.zeros((N,M))
        for i in xrange(blockSize/2,N-blockSize/2):
            a=i-blockSize/2
            b=a+blockSize
            for j in xrange(blockSize/2,M-blockSize/2):
                c=j-blockSize/2
                d=c+blockSize
                Vy[i,j]=2*np.sum(par_x[a:b,c:d]*par_y[a:b,c:d])
                Vx[i,j]=np.sum(par_y[a:b,c:d]**2-par_x[a:b,c:d]**2)
                
    gaussianBlurSigma=2;
    gaussian_block=5 if method=='block-wise' else 21
    Vy=cv2.GaussianBlur(Vy,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    Vx=cv2.GaussianBlur(Vx,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    theta=0.5*np.arctan2(Vy,Vx)            
    return theta

def calcDirectionBox(img,blockSize=8,boxSize=4):
    """calculate ridge directions in an image, using gradient method
    return: ridge directions
    """
    sobel_x=np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_y=np.array([[1, 2, 1],[0, 0, 0],[-1,-2,-1]])
    par_x=convolve2d(img,sobel_x,mode='same')
    par_y=convolve2d(img,sobel_y,mode='same')
    N,M=np.shape(img)
    Vx=np.zeros((N/boxSize,M/boxSize))
    Vy=np.zeros((N/boxSize,M/boxSize))
    ii=-1
    for i in xrange(blockSize/2,N-blockSize/2,boxSize):
        ii += 1
        if ii==N/boxSize:
            break
        a=i-blockSize/2
        b=a+blockSize
        jj=-1
        for j in xrange(blockSize/2,M-blockSize/2,boxSize):
            jj += 1
            if jj==M/boxSize:
                break
            c=j-blockSize/2
            d=c+blockSize
            Vy[ii,jj]=2*np.sum(par_x[a:b,c:d]*par_y[a:b,c:d])
            Vx[ii,jj]=np.sum(par_y[a:b,c:d]**2-par_x[a:b,c:d]**2)                
    gaussianBlurSigma=2;
    gaussian_block=9 
    Vy=cv2.GaussianBlur(Vy,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    Vx=cv2.GaussianBlur(Vx,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    theta=0.5*np.arctan2(Vy,Vx)            
    return theta    
    
    
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
    origin=np.where(f==np.max(f))
    f[origin]=0
    mmax=np.where(f==np.max(f))
    wl=2*img.shape[0]/(((origin[0][0]-mmax[0][0])*2)**2+((origin[1][0]-mmax[1][0])*2)**2)**0.5
    return wl

def calcWl(img,blockSize):
    """calculation wavelength of every blocks in a given image
    """
    wl=np.zeros((img.shape[0]/blockSize,img.shape[1]/blockSize))
    B=block_view(img,(blockSize,blockSize))
    for w,b in zip(wl,B):
        w[:]=[blkwl(a) for a in b]
        #w[:]=map(lambda b: blkwl(b),b)
    # Gaussian smoothing
    gaussianBlurSigma=4;    gaussian_block=9
    wl=cv2.GaussianBlur(wl,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    return wl
  
def calcWlBox(img, blockSize, boxSize):
    resize=5
    img=cv2.resize(img,None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
    blockSize=blockSize*resize
    boxSize=boxSize*resize
    N,M=img.shape
    wl=100*np.ones((img.shape[0]/boxSize,img.shape[1]/boxSize))    
    ii=-1
    for i in xrange(blockSize/2,N-blockSize/2,boxSize):
        ii += 1
        if ii>=N/boxSize:
            break
        a=i-blockSize/2
        b=a+blockSize
        jj=-1
        for j in xrange(blockSize/2,M-blockSize/2,boxSize):
            jj += 1
            if jj>=M/boxSize:
                break
            c=j-blockSize/2
            d=c+blockSize
            wl[ii,jj]=blkwl(img[a:b,c:d])
    gaussianBlurSigma=4;    gaussian_block=9
    wl=cv2.GaussianBlur(wl,(gaussian_block,gaussian_block),gaussianBlurSigma,gaussianBlurSigma)
    return wl/resize
        
        
        
def GaborFilter_(img,blockSize,wl,dire,sigma=20):
    imgout=np.zeros_like(img)
    O=block_view(imgout,(blockSize,blockSize))
    B=block_view(img,(blockSize,blockSize))
    for w,d,o,b in zip(wl,dire,O,B):
        kernel=map(lambda w,d:cv2.getGaborKernel((blockSize,blockSize),sigma,d,w,1),w,d)
        o[:,:]=np.asarray(map(lambda x,kernel: cv2.filter2D(x,-1,kernel),b,kernel))
    return imgout
    
    

def GaborFilter(img,blockSize,wl,dire,sigma=20):
    """Gabor Filter
    img: input image
    blockSize: size of a block
    wl: wavelength
    dire: direction
    return: filtered image
    """
    img=img.astype(np.float64)
    imgout=img.copy()
    kernel=np.zeros((img.shape[0]/blockSize*(blockSize+1),img.shape[1]/blockSize*(blockSize+1)))
    K=block_view(kernel,(blockSize+1,blockSize+1))
    for k,w,d in zip(K,wl,dire):
        k[:,:]=np.asarray(map(lambda w,d: cv2.getGaborKernel((blockSize+1,blockSize+1),sigma,d,w,1,0),w,d))
    for i in xrange(blockSize/2,img.shape[0]-blockSize/2):
        block_i=i/blockSize
        for j in xrange(blockSize/2,img.shape[1]-blockSize/2):
            block_j=j/blockSize
            imgout[i,j]=np.sum(K[block_i,block_j]
                        *img[i-blockSize/2:i+blockSize/2+1,j-blockSize/2:j+blockSize/2+1])
            
    imgout[np.where(imgout>255)]=255;imgout[np.where(imgout<0)]=0
    return imgout
            
def GaborFilterBox(img,blockSize,boxSize,wl,dire,sigma=20):
    """Gabor Filter
    img: input image
    blockSize: size of a block
    wl: wavelength
    dire: direction
    return: filtered image
    """
    img=img.astype(np.float64)
    N,M=img.shape
    imgout=img.copy()
    kernel=np.zeros((img.shape[0]/boxSize*(blockSize+1),img.shape[1]/boxSize*(blockSize+1)))
    K=block_view(kernel,(blockSize+1,blockSize+1))
    for k,w,d in zip(K,wl,dire):
        k[:,:]=np.asarray(
            [cv2.getGaborKernel((blockSize,blockSize),sigma, d_, w_, 1, 0  ) 
                    for w_,d_ in zip(w,d)] )
    
    ii=-1
    for i in xrange(blockSize/2,N-blockSize/2):
        ii = (i-blockSize/2)/boxSize
        if ii>=N/boxSize:
            break
        a=i-blockSize/2
        b=a+blockSize+1
        img0=img[a:b]
        jj=-1
        for j in xrange(blockSize/2,M-blockSize/2):
            jj = (j-blockSize/2)/boxSize
            if jj>=M/boxSize:
                break
            c=j-blockSize/2
            d=c+blockSize+1
            imgout[i,j]=np.sum( K[ii,jj]*img0[:,c:d])
    
#    imgout[np.where(imgout>255)]=255;imgout[np.where(imgout<0)]=0

    return imgout
    

def ridgeComp2(img,theta,blockSize,h=9):
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

def ridgeComp(img,theta,blockSize,boxSize,h=11):
    resize=8
    N,M=np.shape(img)
    imgout=img.copy()
    imgResize=cv2.resize(img,None,fx=resize,fy=resize,interpolation = cv2.INTER_CUBIC)
    hh=np.arange(-(h-1)/2,(h-1)/2+1)
    for i in xrange(10,N-10):
        block_i = (i-blockSize/2)/boxSize
        if block_i>=theta.shape[0]:
            break
        for j in xrange(10,M-10):
            block_j = (j-blockSize/2)/boxSize
            if block_j>=theta.shape[1]:
                break
            theta0=theta[block_i,block_j]
            ii=np.round((i-hh*np.sin(theta0))*resize).astype(np.int32)
            jj=np.round((j+hh*np.cos(theta0))*resize).astype(np.int32)
            imgout[i,j]=np.mean(imgResize[ii,jj])
    return imgout    
    
def fill(img,position,ifending=1,newvalue=2):
    image=img.copy()
    N=image.shape[0]
    a,b=position[0],position[1]
    image[a,b]=newvalue
    while(1):
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
    
def thinning(img):
    """
    Zhang-Suen thinning algorithm
    return: thinned image
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
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 15:14:54 2016

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


def enhance(img,blockSize=16):
    img,imgfore=segmentation(img)
    theta=calcDirection(img,blockSize)
    wl=calcWl(img,blockSize)
    #img=ridgeComp2(img,theta,blockSize)
    img=GaborFilter(img,blockSize,wl,np.pi/2-theta)
    img[np.where(imgfore==255)]=255
    #img[np.where(imgfore==255)]=255
    return img




  
    
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


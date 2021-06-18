#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 00:13:32 2020

@author: caio
"""
import numpy as np
from skimage import img_as_ubyte
import cv2

def ruido(I,sigma,A=1,mu=0):
    I=I/I.max()
    I=img_as_ubyte(I)*(150/255)
    I+=50   
    m, n = np.shape(I)
    ruido=[]
    for i in range(m):
        s=A*np.random.normal(mu,sigma,n)
        ruido.append(s)
    ruido=np.array(ruido)
    J = I + ruido
    if np.abs(J.max()-J.min())>255:
        print('Erro')
    J=J.astype(int)
    J=img_as_ubyte(J/255)
    return(J)


def ruido_vdd(I,sigma,A=1,mu=0):
    I=I/I.max()
    I=img_as_ubyte(I)*(150/255)
    m, n = np.shape(I)
    ruido1=[]
    for i in range(m):
        s=A*np.random.normal(mu,sigma,n)
        ruido1.append(s)
    ruido1=np.array(ruido1)

    ruido2=[]
    for i in range(m):
        s=A*np.random.normal(mu,sigma,n)
        ruido2.append(s)
    ruido2=np.array(ruido2)

    J = np.sqrt(I**2 + ruido1**2 + ruido2**2)
    if np.abs(J.max()-J.min())>255:
        print('Erro')
    J=J.astype(int)
    J=img_as_ubyte(J/255)
    return(J)

def ruido_het(I,s0,sf,N,A=1,mu=0):
    I=img_as_ubyte(I)*(150/255)
    I+=50   
    m, n = np.shape(I)
    sigma = np.linspace(s0,sf,N)
    p=np.floor(m/N).astype(int)
    ruido=[]
    for i in range(N):
        for k in range(i*p,(i+1)*p):
            s = A*np.random.normal(mu,sigma[i],n)
            ruido.append(s)   
    for k in range(N*p,m):
        s = A*np.random.normal(mu,sigma[i],n)
        ruido.append(s)  
    ruido = np.array(ruido)
    J=I+ruido
    if np.abs(J.max()-J.min())>255:
        print('Erro')
    J=J.astype(int)
    J=img_as_ubyte(J/255)
    return(J)


def segmentação(J,tipo = bool):
    m, n = np.shape(J)
    J_gauss = cv2.GaussianBlur(J,(9,9),0)
    PA = cv2.Laplacian(J_gauss,cv2.CV_8UC1,ksize=5)
    otsu_threshold, cont = cv2.threshold(PA, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inundada = cont.copy() 
    mask = np.zeros((m+2, n+2), np.uint8)
    cv2.floodFill(inundada, mask, (0,0), 255) 
    inundada_inv = 255 - inundada
    dentro=(inundada_inv^cont)/255 
    return dentro.astype(tipo)


def SNR(J,tudo=False):
    dentro = segmentação(J)
    fora = ~dentro
    media = J[dentro].mean()
    std=np.std(J[fora], dtype=np.float64)
    SNR = media/std**2
    if tudo:
        return [SNR,media,std]
    else:
        return SNR




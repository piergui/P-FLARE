#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:28:38 2024

@author: ogurcan
"""

import cupy as xp
import numpy as np
from cupyx.scipy.fft import rfft2,irfft2,fft,ifft
#from scipy.fft import rfft2,irfft2

class slicelist:
    def __init__(self,Nx,Ny):
        shp=(Nx,Ny) #shape of the unpadded real space
        insl=[np.s_[0:1,1:int(Ny/2)],np.s_[1:int(Nx/2),:int(Ny/2)],np.s_[-int(Nx/2)+1:,1:int(Ny/2)]] #list of the slices to acces the 2D Fourier space
        shps=[[len(range(*(l[j].indices(shp[j])))) for j in range(len(l))] for l in insl] #shapes of the slices of the 2D Fourier space
        Ns=[np.prod(l) for l in shps] #list of size of each ravelled slice
        outsl=[np.s_[sum(Ns[:l]):sum(Ns[:l])+Ns[l]] for l in range(len(Ns))] #slices to access the 1D ravelled Fourier space
        self.insl,self.shape,self.shps,self.Ns,self.outsl=insl,shp,shps,Ns,outsl

class mlsarray(xp.ndarray):
    def __new__(cls,Nx,Ny):
        v=xp.zeros((Nx,int(Ny/2)+1),dtype=complex).view(cls)
        return v
    def __getitem__(self,key):
        if(isinstance(key,slicelist)):
            return [xp.ndarray.__getitem__(self,l).ravel() for l in key.insl]
        else:
            return xp.ndarray.__getitem__(self,key)
    def __setitem__(self,key,value):
        if(isinstance(key,slicelist)):
            for l,j,shp in zip(key.insl,key.outsl,key.shps):
                self[l]=value.ravel()[j].reshape(shp)
        else:
            xp.ndarray.__setitem__(self,key,value)
    def irfft2(self):
        self.view(dtype=float)[:,:-2]=irfft2(self,norm='forward',overwrite_x=True)
    def rfft2(self):
        self[:,:]=rfft2(self.view(dtype=float)[:,:-2],norm='forward',overwrite_x=True)
    def ifftx(self):
        self[:,:]=ifft(self,norm='forward',overwrite_x=True,axis=0)
    def fftx(self):
        self[:,:]=fft(self,norm='forward',overwrite_x=True,axis=0)
        
def init_kspace_grid(sl):
    #Construct the indices lists for kx and ky with a given slicelist (itself constructed with a specific padded resolution)
    Nx,Ny=sl.shape #shape of the real space
    kxl=np.r_[0:int(Nx/2),-int(Nx/2):0] #all indices for kx
    kyl=np.r_[0:int(Ny/2+1)] #all indices for ky
    kx,ky=np.meshgrid(kxl,kyl,indexing='ij')
    kx=xp.hstack([kx[l].ravel() for l in sl.insl])
    ky=xp.hstack([ky[l].ravel() for l in sl.insl])
    return kx,ky

# Npx,Npy=1024,1024
# Nx,Ny=int(np.floor(Npx/3))*2,int(np.floor(Npy/3))
# Nxh=int(Nx/2)
# Nyhp=int(Ny/2)+1
# Lx,Ly=2*np.pi,2*np.pi
# sl=slicelist(Nx,Ny)
# kx,ky=init_kspace_grid(sl)
# a=mlsarray(Nx,Ny)
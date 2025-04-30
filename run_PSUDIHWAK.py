#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:55:03 2025

@author: pguillon
"""

import cupy as xp
import numpy as np
from scipy.optimize import fmin
import sys
import os
from mlsarray import mlsarray,slicelist,init_kspace_grid,rfft2
from gensolver import gensolver, save_data
from time import time
import h5py as h5


### Physical parameters
def oneover(x):
    if(np.isscalar(x)):
        res=(1/x if x!=0 else 0)
    else:
        res=np.zeros_like(x)
        inds=np.nonzero(x)
        res[inds]=1/x[inds]
    return res

def Wks(kx, ky, C, kap, nu=0, D=0, dxom=0, dxn=0, u=0):
    ksqr = kx**2+ky**2
    iksqr = oneover(ksqr)
    Ak = 0.5*((D+nu)*kx**2+(D+nu)*ky**2+C+C*iksqr+1j*dxom*ky*iksqr)
    Bk = 0.5*((D-nu)*kx**2+(D-nu)*ky**2+C-C*iksqr-1j*dxom*ky*iksqr)
    Gk = C**2*oneover(ksqr)+Bk.real**2-Bk.imag**2
    Hk = np.sqrt(Gk**2+(C*(kap-dxn)*ky*oneover(ksqr)-2*Bk.real*Bk.imag)**2)
    Wkp = np.sign(ky)*np.sqrt((Hk-Gk)/2)+1j*np.sqrt((Hk+Gk)/2)
    Wkm = -np.sign(ky)*np.sqrt((Hk-Gk)/2)-1j*np.sqrt((Hk+Gk)/2)
    om_p = Wkp+u*ky-1j*Ak
    om_m = Wkm+u*ky-1j*Ak
    return om_p, om_m

C = 0.01
kap = 1.0

ky0 = fmin(lambda x: -np.imag(Wks(0, x, C, kap,nu=0, D=0)[0] if x>=0 else 0), 1.25)[0]
gam = Wks(0, ky0, C, kap, 0, 0)[0].imag

nu = 3e-2 * gam / ky0**2  # viscosity
D = 3e-2 * gam / ky0**2 # turbulent particle diffusion
# Du = 1e-4 # zonal velocity diffusion
# Dn = D * 1e-2 # zonal particle diffusion

### Grid resolution and box size
Npx, Npy = 1024, 1024 # padded resolution
Nx, Ny = 2 * int(np.floor(Npx/3)), 2 * int(np.floor(Npy/3)) # Fourier space resolution
Lx, Ly = 10 * 2*np.pi / ky0, 10 * 2 * np.pi / ky0 # box size
# Lx, Ly = 64*np.pi, 64*np.pi

### Integration time range and timesteps (solver, show, save, etc.)
dtstep, dtshow = np.round(1 / gam / 100, 3), np.round(1 / gam / 10, 3) # time step integration, time step to print where we are
t0, t1 = 0.0, np.round(200 * 100 * dtstep, 3) # time range to integrate /!\ use floats otherwise saving of time fails

### Buffer and penalisation parameters
im1, im2 = 45, -45 # indices where to flatten the zonal density profile
d_ixm =  40  # number of points over which the flattening gate transitions
ib1, ib2 = 90, -90 # boundary indices of the buffer zone
d_ixb = 60  # number of points over which the penalisation gate transitions
mupen = 1e2 # friction coefficient to prevent the fluctuations going in the buffer
sigS = 5 * Lx / Nx # width of the "artificial" sources for the boundary conditions 

### Saving name or loading existing simulation
wecontinue = False #flag to continue an already run simulation
flname = f'RUNS/outfdC{C}_{int(Lx/np.pi)}pi_{Npx}x{Npy}.h5' 

### Construct real space 
Xpad,  Ypad = xp.arange(0, Lx, Lx/Npx), xp.arange(0, Ly, Ly/Npy) #1D arrays for x and y axis in padded space
X,  Y = xp.arange(0, Lx, Lx/Nx), xp.arange(0, Ly, Ly/Ny) #1D arrays for x and y axis
x, y = xp.meshgrid(xp.array(X), xp.array(Y), indexing='ij') #2D arrays for xy grid

### Building the initial radial profiles
ur0 = xp.zeros_like(X) # flat initial poloidal velocity profile
nr0 = Lx * np.exp(- 4 * (X/Lx)**2) #gaussian initial density profile (centered on X=0)

### Source term
# alpha=0
# X0 = 1.5 * X[ib1]
# sig_n = Lx/50
# def S_n(X, X0, t, alpha, sig_n):
#     return alpha / (sig_n * np.sqrt(2*np.pi)) * xp.exp(-(X - X0) ** 2 / (2 * sig_n**2)) 

### Construct the slices and the Fourier space for 1D vectors
dkx, dky = 2*np.pi/Lx , 2*np.pi/Ly
sl = slicelist(Nx, Ny) 
lkx, lky = init_kspace_grid(sl)
kx, ky = lkx * dkx, lky * dky
ksqr = kx**2 + ky**2
slbar = np.s_[int(Ny/2) - 1:int(Ny/2) * int(Nx/2) - 1:int(Ny/2)] #Slice to access only zonal modes
slturb = np.setdiff1d(np.arange(len(kx)), np.arange(len(kx))[slbar]) #Complementary of the zonal slice to access all non-zonal turbulent modes

### Actual 2D Fourier space for saving
kxp, kyp = xp.r_[np.arange(0,int(Nx/2)+1)*dkx, xp.arange(-int(Nx/2)+1, 0)*dkx],  xp.arange(0, int(Ny/2)+1)*dky
kxp, kyp = xp.meshgrid(kxp, kyp, indexing='ij')
ksqrp=kxp**2+kyp**2

### Building the initial condition in Fourier space
### Here a Gaussian of width 10 * 2 * pi / Lx centered on (kx,ky) = (0,0), with random initial phases and initial amplitude of 1e-4
w = 10.0
phik0 = 1e-4 * xp.exp(-lkx**2 / 2 /w**2 - lky**2 / 2 / w**2) * xp.exp(1j * 2 * np.pi * xp.random.rand(lkx.size).reshape(lkx.shape))
nk0 = 1e-4 * xp.exp(-lkx**2 / 2 / w**2 - lky**2 / 2 / w**2) * xp.exp(1j * 2 * np.pi * xp.random.rand(lkx.size).reshape(lkx.shape))
del lkx,lky

### Building the inital 1D vector
### We solve the radial profiles and the Fourier transform of the turbulent (i.e.  non-zonal) modes
zr = xp.hstack((ur0,  nr0)) # concatenated initial radial profiles
zk = xp.hstack((phik0[slturb], nk0[slturb])) # concatenated initial turbulent Fourier modes
z = xp.hstack((zr, zk)) #initial condition

### Penalisation and smooth flattening functions
### Smooth function flat at 0
def p(y):
    p = xp.zeros_like(y)
    idp = y>0
    p[idp] = xp.exp(-1/y[idp])
    return p

 ### Smooth transition from 0 to 1
def jump(y):
    return p(y) / (p(y) + p(1-y))

### Smooth gate function
def smooth_gate(X, i1, i2, d_ix):
    f = xp.ones_like(X) # exactly one for Xb1 <= X <= Xb2
    il, ir = i1 - d_ix, i2 + d_ix
    idl = abs(X - (X[i1] + X[il]) /2) < (X[i1] - X[il]) /2  # points Xb1 - dX < X < Xb1
    idr = abs(X - (X[ir] + X[i2]) /2) < (X[ir] - X[i2]) /2 # points Xb2 < X < Xb2 + dX
    f[X <= X[il]] = 0 # exactly 0 for X <= Xb1 - dX
    f[idl] = jump((X[idl] - X[il]) / (X[i1] - X[il])) # transitions from 0 to 1 for Xb1 - dX < X < Xb1
    f[idr] = jump((X[ir] - X[idr]) / (X[ir] - X[i2])) # transitions from 1 to 0 for Xb2 < X < Xb2 + dX
    f[X >= X[ir]] = 0 # exactly 0 for X >= Xb2 + dX
    return f

### Building smooth gates for 1D and 2 penalisation
psi_1D = smooth_gate(X, ib1, ib2, d_ixb)

#"Interpolate" the penalisation fonction on the padded space for 2D multiplications
psik = xp.zeros(int(Npx/2)+1, dtype=complex)
psik[:int(Npx/3)] = xp.fft.rfft(psi_1D, norm='forward')[:int(Npx/3)]
psi_1D_pad = xp.fft.irfft(psik, norm='forward')
psi_2D = xp.zeros((Npx, Npy))
psi_2D += psi_1D_pad[:, None]

#Left and right penalisation functions 
H1 = (1 - psi_1D) * (X < X[ib1]) # 1 only in the left buffer
H2 = (1 - psi_1D) * (X > X[ib2]) # 1 only in the right buffer

del psik, psi_1D_pad

### Decompose and smooth the zonal profile in the buffer
def dec_prof(nr, X, ib1, ib2, im1, im2, d_ix):
    ### Compute kappa between Xb1 and Xb2
    kap = - (nr[ib2] - nr[ib1]) / (X[ib2] - X[ib1])
    
    ### Construct the linear profile
    nlin = -kap * (X - X[ib2]) + nr[ib2]

    ### Get the zonal profile from the radial profile
    nbar_raw = nr - nlin  
    
    ### Modify the zonal profile to make it flat in the buffer, starting the flattening at xm1 and xm2
    n_off = xp.mean(nbar_raw[np.r_[0, im1, im2, -1]]) # offset to shift the zonal profile with when multiplying by the smooth gate
    nbar_smth = (nbar_raw - n_off) * smooth_gate(X, im1, im2, d_ix) + n_off

    ### Remove the mean value from the zonal profile
    nm = xp.mean(nbar_smth)
    nbar_smth -= nm
    
    return nbar_smth, kap, nm

### Fourier transforms with padding, using the mlsarray libray
def rft2(u):
    '''
    Returns the concatenated slices of the 2D Fourier transform of u (all padded modes are removed).
    (Npx, Npy) --> (int(Ny/2) + int(Nx/2) * (int(Ny/2) +1) + int(Nx/2) * int(Ny/2), 1)
    '''
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return xp.hstack(uk[sl])

def irft2(uk):
    '''
    Returns the 2D inverse transform of uk (adds the padded modes with 0 amplitude)
    (int(Ny/2) + int(Nx/2) * (int(Ny/2) +1) + int(Nx/2) * int(Ny/2), 1) --> (Npx, Npy)
    '''
    u=mlsarray(Npx,Npy)
    u[sl]=uk
    u[-1:-int(Nx/2):-1,0]=u[1:int(Nx/2),0].conj()
    u.irfft2()
    return u.view(dtype=float)[:,:-2]

def rft(v):
    '''
    Returns the 1D Fourier transform of v.
    (Nx, 1) --> (int(Nx/2), 1)
    '''
    return  xp.fft.rfft(v, norm='forward')[1:int(Nx/2)]

def irft(vk):
    '''
    Returns the 1D inverse Fourier transform of vk.
    (int(Nx/2), 1) --> (Nx, 1) 
    '''
    v = xp.zeros(int(Nx/2)+1, dtype='complex128')
    v[1:int(Nx/2)] = vk[:]
    return xp.fft.irfft(v, norm='forward')

### Saving function
def save_callback(fl,t,y,l):
    z = y.view(dtype=complex)
    fl['last/t'][()]=t.get()
    fl['last/z'][:]=z.get()
    
    zr, zk = z[:X.size*2].real, z[X.size*2:]
    ur, nr = zr[:int(zr.size/2)], zr[int(zr.size/2):]
    
    ### Extract kappa and zonal profile from the radial density profile
    nbar, kap, nm = dec_prof(nr, X, ib1, ib2, im1, im2, d_ixm)
    
    ### Remove mean value from the radial profile of the velocity
    um = xp.mean(ur) #ur is not zero mean value due to penalisation
    ubar = ur - um
    
    ### Get the zonal modes by computing the Fourier transform of the zonal profiles
    phiq = -1j * rft(ubar) / kx[slbar]
    nq = rft(nbar)
    
    ### Gather zonal and non-zonal Fourier modes
    phik, nk = xp.zeros(kx.size, dtype=complex), xp.zeros(kx.size, dtype=complex)
    phik[slbar], nk[slbar] = phiq[:], nq[:]
    phik[slturb], nk[slturb] = zk[:int(zk.size/2)], zk[int(zk.size/2):]
    
    ### Create the 2D fourier arrays (Nyquist mode is set to 0)
    phikp, nkp = mlsarray(Nx,Ny), mlsarray(Nx,Ny)  
    phikp[sl],nkp[sl]=phik,nk
    phikp[-1:-int(Nx/2):-1,0]=phikp[1:int(Nx/2),0].conj()
    nkp[-1:-int(Nx/2):-1,0]=nkp[1:int(Nx/2),0].conj()
    uk=np.array([phikp.get(), nkp.get()])
    
    if l=='show':
        ### Showing the time, the total energy, the zonal fraction and kappa
        Ktot, Kbar = xp.sum(ksqr*abs(phik)**2), xp.sum(abs(kx[slbar] * phik[slbar])**2)
        print(f"t={t:.6}, {(time() - ct):.6} secs elapsed., Ktot={Ktot:.6}, Kbar/Ktot={np.round(Kbar / Ktot * 100,1)}%, kap={kap:.6}")
        del Ktot, Kbar
 
    elif l=="fields":
        ### Saving the full 2D Fourier fields and the radial profiles
        save_data(fl, 'fields', ext_flag=True, uk=uk, t=t.get(), ur=ur.get(), nr=nr.get())
    
    elif l=="profiles":
        ### Saving only the radial profiles
        save_data(fl, 'fields/profiles', ext_flag=True, t=t.get(), ur=ur.get(), nr=nr.get(), kap=kap.get())

    elif l=="reduced":
        #### Saving all zonal modes and the 10 first turbulent modes
        save_data(fl, 'fields/reduced', ext_flag=True, t=t.get(), ukred=uk[:, :, :10])
            
    elif l=="fluxes":
        ### Saving the particle and vorticiy fluxes, and the Reynolds stress, averaged along y
        Gam = -xp.mean(xp.fft.irfft2(1j * kyp * phikp, norm='forward') * xp.fft.irfft2(nkp, norm='forward'), axis=-1)
        Pi = xp.mean(xp.fft.irfft2(1j * kyp * phikp, norm='forward') * xp.fft.irfft2(ksqrp * phikp, norm='forward'),axis=-1)
        R = xp.mean(xp.fft.irfft2(1j * kyp * phikp, norm='forward') * xp.fft.irfft2(1j * kxp * phikp, norm='forward'),axis=-1)
        save_data(fl, "fields/fluxes", ext_flag=True, t=t.get(), Gam=Gam.get(), Pi=Pi.get(), R=R.get())
        del Gam, Pi, R
    
    elif l=="energies":
        ### Saving the mean kinetic and potential energies, and the "kinetic" enstrophy
        K, Kbar = xp.sum(ksqr*abs(phik)**2), xp.sum(abs(kx[slbar] * phik[slbar])**2)
        W, Wbar = xp.sum(ksqr**2 * abs(phik)**2), xp.sum(abs(kx[slbar]**2 * phik[slbar])**2)
        N, Nbar = xp.sum(abs(nk)**2), xp.sum(abs(nk[slbar])**2)
        vx, vy = xp.fft.irfft2(-1j * kyp * phikp, norm='forward'),  xp.fft.irfft2(1j * kxp * phikp, norm='forward')
        E_x = xp.mean(vx**2 + vy**2, axis=-1)
        save_data(fl, "fields/energies", ext_flag=True, t=t.get(), K=K.get(), W=W.get(), N=N.get(), Kbar=Kbar.get(), Wbar=Wbar.get(), Nbar=Nbar.get(), E_x=E_x.get())
        del K, W, N, Kbar, Wbar, Nbar, vx, vy, E_x
                
    del phik, nk, phiq, nq, ubar, nbar, phikp, nkp, uk
    
### Right-hand side of the flux-driven Hasegawa-Wakatani equations
def rhs(t,y):
    
    ### Get the fields and create the time derivative array
    z = y.view(dtype=complex)
    dzdt = xp.zeros_like(z)
    
    ### Separate radial profiles from (non-zonal) Fourier modes
    zr, zk = z[:X.size*2].real, z[X.size*2:]
    dzrdt, dzkdt = dzdt[:X.size*2].real, dzdt[X.size*2:]

    ### Separate velocity from density profiles
    ur, nr = zr[:int(zr.size/2)], zr[int(zr.size/2):]
    durdt, dnrdt = dzrdt[:int(zr.size/2)], dzrdt[int(zr.size/2):]
        
    ### Extract kappa and zonal profile from the radial density profile
    nbar, kap, nm = dec_prof(nr, X, ib1, ib2, im1, im2, d_ixm)
    
    ### Remove mean value from the radial profile of the velocity
    um = xp.mean(ur) #ur is not zero mean value due to penalisation
    ubar = ur - um

    ### Get the zonal modes by computing the Fourier transform of the zonal profiles
    phiq = -1j * rft(ubar) / kx[slbar]
    nq = rft(nbar)
    
    ### Gather zonal and non-zonal Fourier modes
    phik, nk = xp.zeros(kx.size, dtype=complex), xp.zeros(kx.size, dtype=complex)
    dphikdt, dnkdt = xp.zeros(kx.size, dtype=complex), xp.zeros(kx.size, dtype=complex)
    phik[slbar], nk[slbar] = phiq[:], nq[:]
    phik[slturb], nk[slturb] = zk[:int(zk.size/2)], zk[int(zk.size/2):]

    ### 2D inverse Fourier transforms for computing the non-linear terms
    # dxphi = irft2(1j * kx * phik)
    dyphi = irft2(1j * ky * phik)
    dxphi = irft2(1j * kx * phik) + um #add mean radial value of the poloidal velocity
    n = irft2(nk)
    
    ### Compute the non-linear terms
    dphikdt = -1 / ksqr * (kx * ky * rft2(dxphi**2 - dyphi**2) + (ky**2 - kx**2) * rft2(dxphi * dyphi))
    dnkdt = 1j * kx * rft2(dyphi * n) - 1j * ky * rft2(dxphi * n)
        
    ### Add the linear terms on non-zonal modes (except for dissipation)
    dphikdt += C * (phik - nk) * (-1 / ksqr) * (ky>0)
    dnkdt += (C * (phik - nk) - 1j * kap * ky * phik  ) * (ky>0) 
    
    ### Apply the penalisation only on the non-zonal modes, and apply dissipation on turbulence over the full domain
    dphikdt += -mupen * ( 1j * kx * rft2( (1 - psi_2D) * dxphi) + 1j * ky * rft2((1 - psi_2D) * dyphi))  * (-1 / ksqr) * (ky>0)  -  ksqr * nu * phik  * (ky>0)
    dnkdt += -mupen  * rft2((1 - psi_2D) * n)  * (ky>0) -  ksqr * D * nk  * (ky>0)
       
    ### Update the radial profiles from the time evolution of zonal modes
    durdt += irft(1j * kx[slbar] * dphikdt[slbar])
    dnrdt += irft(dnkdt[slbar])
    
    ### Apply the penalisation to force the zonal velocity to be zero in the buffer (no-slip boundary conditions)
    durdt += - mupen * (1 - psi_1D) * ur 
    
    ### Boundary conditions at x1 and x2, using artificial narrow gaussian sources
    dnrdt +=  -dnrdt[ib2] * xp.exp(-(X - X[ib2])**2 / sigS**2 /2) # impose dnrdt[i2] = 0
    dnrdt += dnrdt[ib2] * xp.exp(-(X - X[ib1])**2 / sigS**2 /2) # We want a free moving boundary, but we add the gaussian source to compensate for the artificial losses at the other boundary
    
    ### Apply physical source terms, here a constant source term
    # dnrdt += S_n(X, X[ib1], t, alpha = 0.02, sig_n = sigS * 5)
    
    ### Apply density diffusion
    # dnrdt += Dn * irft(-ksqr[slbar] * nq)
    
    ### Apply the penalisation to force the density profile to follow its initial shape in the buffer (no fluctuations in that region), but shifted according to the variations of nr[i1] and nr[i2]
    nbuff1, nbuff2 = nr0 - nr0[ib1] + nr[ib1], nr0 - nr0[ib2] + nr[ib2]
    dnrdt +=  - mupen * (H1 * (nr - nbuff1) + H2 * (nr - nbuff2)) 
    
    #Update the turbulent modes
    dzkdt[:] = xp.hstack((dphikdt[slturb], dnkdt[slturb]))

    del n, dxphi, dyphi, ubar, nbar, phik, nk, phiq, nq, dphikdt, dnkdt
    return dzdt.view(dtype=float)

### Some initial savings / get last values from already run simulation
if (wecontinue):
    fl=h5.File(flname,'r+',libver='latest')
    fl.swmr_mode = True
    t=fl['last/t'][()]
    z[:]=xp.array(fl['last/z'][()])
    
else:
    os.remove(flname) if os.path.exists(flname) else None
    fl=h5.File(flname,'w',libver='latest')
    fl.swmr_mode = True
    t=float(t0)
    save_data(fl,'last',ext_flag=False,z=z.get(),t=t)
    
save_data(fl,'params', ext_flag=False, C=C , nu=nu, D=D, Lx=Lx, Ly=Ly, Npx=Npx, Npy=Npy)
save_data(fl,'buffer', ext_flag=False, mupen=mupen, indices=(ib1, ib2, d_ixb, im1, im2, d_ixm), sigS=sigS)

###Setting saving
fsave=[
       (lambda t,y : save_callback(fl,t,y,"fields")),
       (lambda t,y : save_callback(fl,t,y,"profiles")),
       (lambda t,y : save_callback(fl,t,y,"reduced")),
       (lambda t,y : save_callback(fl,t,y,"energies")),
       (lambda t,y : save_callback(fl,t,y,"fluxes")),
       (lambda t,y : save_callback(fl,t,y,"show"))
       ]
dtsave=[
        10 * dtstep,
        dtstep,
        dtstep,
        dtstep,
        dtstep,
        dtshow] #time step to do all the actions above

### Building solver an run
ct=time()
r=gensolver('cupy_ivp.DOP853', rhs, t, z.view(dtype=float), t1, fsave=fsave, dtstep=dtstep, dtshow=dtshow, dtsave=dtsave, rtol=1e-10, atol=1e-12)
r.run()
fl.close()



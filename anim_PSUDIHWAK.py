#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:02:58 2025

@author: pguillon
"""
import sys
import os
import shutil
import numpy as np
import h5py as h5
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator

mpl.use("Agg")
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle

from mpi4py import MPI
comm=MPI.COMM_WORLD

infl="RUNS/SPREAD/outfdC0.05_32pi_1024x1024_kapl5.0.h5"
outfl="MOVIES/" + infl[5:-3] + ".mp4"

plt.rcParams.update({
    'text.usetex' : True,
    "font.family": "serif",
    'font.size': 22,
    'lines.linewidth': 3,
    'figure.dpi': 100,
    'axes.linewidth' : 2,
    'xtick.direction': 'in',
    'xtick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.direction': 'in',
    'ytick.major.size': 8,
    'ytick.minor.size': 4,
    'xtick.major.width': 2,
    'xtick.minor.width': 1,
    'ytick.major.width': 2,
    'ytick.minor.width': 1,})

fl=h5.File(infl,"r",libver='latest',swmr=True)
uk=fl['fields/uk']
ur=fl['fields/ur']
nr=fl['fields/nr']
Lx=fl['params/Lx'][()]
Ly=fl['params/Ly'][()]
Npx=fl['params/Npx'][()]
Npy=fl['params/Npy'][()]
C=fl['params/C'][()]
nu=fl['params/nu'][()]

vminom, vmaxom = -7, 7 
vminn, vmaxn = -25, 25

vmin_u, vmax_u = -1.1 * np.max(abs(ur[:])), 1.1 * np.max(abs(ur[:]))
vmin_n, vmax_n = np.min(nr[:,])*0.9, np.max(nr[:,])*1.1

ib1, ib2, d_ixb, im1, im2, d_ixm = fl['buffer/indices'][()]

Nx=int(Npx/3)*2
Ny=int(Npy/3)*2
Nxh=int(uk.shape[2]/2)

# Construct Fourier grid
dkx, dky = 2*np.pi/Lx, 2*np.pi/Ly
kx, ky = np.r_[np.arange(0,int(Nx/2)+1)*dkx, np.arange(-int(Nx/2)+1, 0)*dkx],  np.arange(0, int(Ny/2)+1)*dky
kx, ky = np.meshgrid(kx, ky, indexing='ij')
ksqr=kx**2+ky**2

# Construct real grid
dx=Lx/Nx;
dy=Ly/Ny;
X = np.arange(0,Nx)*dx
Y = np.arange(0,Ny)*dy
x,y=np.meshgrid(X,Y,indexing='ij')

def oneover(x):
    if(np.isscalar(x)):
        res=(1/x if x!=0 else 0)
    else:
        res=np.zeros_like(x)
        inds=np.nonzero(x)
        res[inds]=1/x[inds]
    return res

def plot_hatches(ax, x0, y0, w, h):
    ax.add_patch(Rectangle((x0, y0), w, h, ec='k', fc='None',  hatch='\\\\', lw = 2, zorder=4, alpha=0.5))
    
### Smooth function flat at 0
def p(y):
    p = np.zeros_like(y)
    idp = y>0
    p[idp] = np.exp(-1/y[idp])
    return p

 ### Smooth transition from 0 to 1
def jump(y):
    return p(y) / (p(y) + p(1-y))

### Smooth gate function
def smooth_gate(X, i1, i2, d_ix):
    # smooth gate function
    f = np.ones_like(X)
    il, ir = i1 -d_ix, i2 + d_ix
    idl = abs(X - (X[i1] + X[il]) /2) < (X[i1] - X[il]) /2  # points X[il] < X < X[ib1]
    idr = abs(X - (X[ir] + X[i2]) /2) < (X[ir] - X[i2]) /2 # points X[ib2] < X < X[ir]
    f[X <= X[il]] =0
    f[idl] = jump((X[idl] - X[il]) / (X[i1] - X[il]))
    f[idr] = jump((X[ir] - X[idr]) / (X[ir] - X[i2]))
    f[X >= X[ir]] =0
    return f
    
### Decompose and smooth the zonal profile in the buffer
def dec_prof(nr, X, ib1, ib2, im1, im2, d_ix):
    
    ### Get kappa and the linear profile
    kap = - (nr[ib2] - nr[ib1]) / (X[ib2] - X[ib1])
    nlin = -kap * (X - X[ib2]) + nr[ib2]

    ### Get the zonal profile from the radial profile
    nbar_raw = nr - nlin  
    
    ### Modify the zonal profile to make it periodic in the buffer
    n_off = np.mean(nbar_raw[np.r_[0, im1, im2, -1]]) #offset to shift the zonal profile with when multiplying by the smooth gate
    nbar_smth = (nbar_raw - n_off) * smooth_gate(X, im1, im2, d_ix) + n_off

    ### Remove the mean value from the zonal profile
    nm = np.mean(nbar_smth)
    nbar_smth -= nm
    return nbar_smth, kap, nm

fig=plt.figure(figsize=(15,8))
gs = gridspec.GridSpec(3, 2, height_ratios=[0.05, 20, 0.5], hspace=0.5, wspace=0.4)

# Create subplotsof
ax = [fig.add_subplot(gs[1, i]) for i in range(2)]
axi = [ax[i].twinx() for i in range(2)]

u0=np.fft.irfft2(-uk[0,0,:,:]*ksqr, norm='forward').real
u1=np.fft.irfft2(uk[0,1,:,:], norm='forward').real
nbar, kap, nm = dec_prof(nr[0,], X, ib1, ib2, im1, im2, d_ixm) 
# nlin =  -kap*(X-X[ib2]) + nr[0, ib2]

qd=[]
qd.append(ax[0].pcolormesh(x, y, u0,cmap='seismic',rasterized=True,vmin=vminom,vmax=vmaxom))
qd.append(ax[1].pcolormesh(x, y, u1,cmap='seismic',rasterized=True,vmin=vminn,vmax=vmaxn))
qd.append(axi[0].plot(x[:,0], ur[0,], color='w', lw=6)[0])
qd.append(axi[0].plot(x[:,0], ur[0,], color='black', lw=3)[0])
qd.append(axi[1].plot(x[:,0], nr[0,],  color='w', lw=6)[0])
qd.append(axi[1].plot(x[:,0], nr[0,], color='black', lw=3)[0])
# qd.append(axi[1].plot(x[:,0], nbar + nm + np.mean(nlin) , color='w', lw=6, alpha=0.5)[0])
# qd.append(axi[1].plot(x[:,0], nbar + nm + np.mean(nlin) , color='green', lw=3, alpha=0.5)[0])
# qd.append(axi[1].plot(x[:,0], nlin, color='black', lw=3, alpha=0.5)[0])

for i in range(len(ax)):
    ax[i].axvspan(0, x[ib1,0], alpha=0.5, color='grey')
    ax[i].axvspan(x[ib2,0], x[-1,0], alpha=0.5, color='grey')
    ax[i].set_xlabel(r'$x$', labelpad=-3)
    ax[i].set_ylabel(r'$y$', labelpad=3)
    ax[i].set_xticks(np.arange(0, x[-1,0], int(Lx/5)))
    ax[i].set_yticks(np.arange(0, y[0,-1], int(Ly/5)))
    ax[i].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[i].yaxis.set_minor_locator(AutoMinorLocator(4))
    ax[i].set_xlim([x[0,0],x[-1,0]])
    ax[i].set_ylim([y[0,0],y[0,-1]])
    axi[i].yaxis.set_minor_locator(AutoMinorLocator(4))
    axi[i].tick_params(axis='y', which='both', direction='out')

axi[0].set_ylim([vmin_u, vmax_u])
axi[1].set_ylim([vmin_n, vmax_n])
axi[0].set_ylabel(r'$\overline{v}_y(x,t)$', labelpad=3)
axi[1 ].set_ylabel(r'$n_r(x,t)$', labelpad=3)

plot_hatches(axi[0], 0, vmin_u, x[ib1, 0], vmax_u - vmin_u)
plot_hatches(axi[0], x[ib2, 0], vmin_u, Lx - x[ib2,0], vmax_u - vmin_u)
plot_hatches(axi[1], 0, vmin_n, x[ib1, 0], vmax_n - vmin_n)
plot_hatches(axi[1], x[ib2, 0], vmin_n, Lx, vmax_n - vmin_n)

cbar_ax1 = fig.add_subplot(gs[2, 0])
cbar_ax2 = fig.add_subplot(gs[2, 1])
cb1=fig.colorbar(qd[0], cax=cbar_ax1, orientation='horizontal')
cb2=fig.colorbar(qd[1], cax=cbar_ax2, orientation='horizontal')
cb1.set_label(r'$\Omega(x,y)$', labelpad=-10)
cb2.set_label(r'$n(x,y)$', labelpad=-10)

t=fl['fields/t'][()]
tx=fig.text(0.5, 0.9, f'$t={np.round(0,1)}, \\kappa={kap:.3f}, C/\\kappa={C/kap :.3f}$', ha='center')

nt0=10
Nt=t.shape[0]

if (comm.rank==0):
    lt=np.arange(Nt)
    lt_loc=np.array_split(lt,comm.size)
    if not os.path.exists('_tmpimg_folder'):
        os.makedirs('_tmpimg_folder')
else:
    lt_loc=None
lt_loc=comm.scatter(lt_loc,root=0)

for j in lt_loc:
    print(str(j)+", "+str(np.round((j-lt_loc[0])/(lt_loc[-1]-lt_loc[0]) *100,0))+"%")
    u0=np.fft.irfft2(-uk[j,0,:,:]*ksqr, norm='forward')
    u1=np.fft.irfft2(uk[j,1,:,:], norm='forward') 
    nbar, kap, nm = dec_prof(nr[j,], X, ib1, ib2, im1, im2, d_ixm) 
    # nlin =  -kap * (X - X[ib2]) + nr[j, ib2]
    qd[0].set_array(u0.ravel())
    qd[1].set_array(u1.ravel())
    qd[2].set_data(x[:,0], ur[j,])
    qd[3].set_data(x[:,0], ur[j,])
    qd[4].set_data(x[:,0], nr[j,])
    qd[5].set_data(x[:,0], nr[j,])
    # qd[6].set_data(x[:,0], nbar+nm+np.mean(nlin))
    # qd[7].set_data(x[:,0], nbar+nm+np.mean(nlin))
    # qd[8].set_data(x[:,0], nlin)
    tx.set_text(f'$t={np.round(t[j],1)}, \\kappa={kap:.3f}, C/\\kappa={C/kap :.3f}$')
    fig.savefig("_tmpimg_folder/tmpout%04i"%(j+nt0)+".png",dpi=200)#,bbox_inches='tight')
comm.Barrier()

if comm.rank==0:
    u0=np.fft.irfft2(-uk[0,0,:,:]*ksqr, norm='forward')
    u1=np.fft.irfft2(uk[0,1,:,:], norm='forward')
    nbar, kap, nm = dec_prof(nr[0,], X, ib1, ib2, im1, im2, d_ixm) 
    nlin =  -kap*(X - X[ib2]) + nr[0, ib2]
    qd[0].set_array(u0.ravel())
    qd[1].set_array(u1.ravel())
    qd[2].set_data(x[:,0], ur[0])
    qd[3].set_data(x[:,0], ur[0])
    qd[4].set_data(x[:,0], nr[0,])
    qd[5].set_data(x[:,0], nr[0,])
    # qd[6].set_data(x[:,0], nbar+nm+np.mean(nlin))
    # qd[7].set_data(x[:,0], nbar+nm+np.mean(nlin))
    # qd[8].set_data(x[:,0], nlin)
    tx.set_text('')

    str_intro =   "                 $\\textsc {P. L. Guillon}$ \n\n"
    str_intro += f"Box size : $[L_x,L_y]=[{Lx/np.pi}\\pi,{Ly/np.pi}\\pi]$ \n\n"
    str_intro += f"Padded Resolution : ${Npx} \\times {Npy}$ \n\n"
    str_intro += f"Viscosity : $\\nu = {nu:.2e}$ \n\n"
    str_intro += f"$C = {C}$"
    fig.text(0.35, 0.5, str_intro,fontsize=24,ha='left', 
             bbox=dict(facecolor='lightyellow', edgecolor='blueviolet', boxstyle='round,pad=1', lw=3))

    
    fig.savefig("_tmpimg_folder/tmpout%04i"%(0)+".png",dpi=200)#,bbox_inches='tight')
    for j in range(1,nt0):
        os.system("cp _tmpimg_folder/tmpout%04i"%(0)+".png _tmpimg_folder/tmpout%04i"%(j)+".png")
    
    os.system("ffmpeg -framerate 100 -y -i _tmpimg_folder/tmpout%04d.png -c:v libx264 -pix_fmt yuv420p -vf fps=100 "+outfl)
    shutil.rmtree("_tmpimg_folder")
    
    fl.close()

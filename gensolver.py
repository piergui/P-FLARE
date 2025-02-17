#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:24:19 2024

@author: ogurcan
"""
from time import time
import numpy as np

def save_data(fl,grpname,ext_flag,**kwargs):
    if not (grpname in fl):
        grp=fl.create_group(grpname)
    else:
        grp=fl[grpname]
    for l,m in kwargs.items():
        if not l in grp:
            if(not ext_flag):
                grp[l]=m
            else:
                if(np.isscalar(m)):
                    grp.create_dataset(l,(1,),maxshape=(None,),dtype=type(m))
                    if(not fl.swmr_mode):
                        fl.swmr_mode = True
                else:
                    grp.create_dataset(l,(1,)+m.shape,chunks=(1,)+m.shape,maxshape=(None,)+m.shape,dtype=m.dtype)
                    if(not fl.swmr_mode):
                        fl.swmr_mode = True
                lptr=grp[l]
                lptr[-1,]=m
                lptr.flush()
        elif(ext_flag):
            lptr=grp[l]
            lptr.resize((lptr.shape[0]+1,)+lptr.shape[1:])
            lptr[-1,]=m
            lptr.flush()
        fl.flush()

class gensolver:
    
    def __init__(self,solver,f,t0,y0,t1,fsave,dtstep=0.1,dtshow=None,dtsave=None,**kwargs):
        if(dtshow is None):
            dtshow=dtstep
        if(dtsave is None):
            dtsave=dtstep
        if isinstance(dtsave,float):
            dtsave=np.array([dtsave,])
        if isinstance(dtsave,list) or isinstance(dtsave,tuple):
            dtsave=np.array(dtsave)
        if solver=='scipy.DOP853':
            from scipy.integrate import DOP853
            print(kwargs)
            self.r=DOP853(f,t0,y0,t1,max_step=dtstep,**kwargs)
        if solver=='cupy_ivp.DOP853':
            from cupy_ivp import DOP853
            print(kwargs)
            self.r=DOP853(f,t0,y0,t1,max_step=dtstep,**kwargs)
        elif solver=='scipy.RK45':
            from scipy.integrate import RK45
            self.r=RK45(f,t0,y0,t1,max_step=dtstep,**kwargs)
        elif solver=='cupy_ivp.RK45':
            from cupy_ivp import RK45
            self.r=RK45(f,t0,y0,t1,max_step=dtstep,**kwargs)
        elif solver=='scipy.vode':
            from scipy.integrate import ode
            self.r=ode(f).set_integrator('vode',**kwargs)
            self.r.set_initial_value(y0,t0)
#            self.r.step = lambda : self.r.integrate(t=t1,step=True)
        if not hasattr(self.r, 'integrate'):
            def integrate(tnext):
                while(self.r.t<tnext):
                    self.r.step()
            self.r.integrate=integrate
        self.dtstep,self.dtshow,self.dtsave=dtstep,dtshow,dtsave
        self.t0,self.t1=t0,t1
        if(callable(fsave)):
            self.fsave=[fsave,]
        else:
            self.fsave=fsave
            
    def run(self):
        dtstep,dtshow,dtsave=self.dtstep,self.dtshow,self.dtsave
        t0,t1=self.t0,self.t1
        r=self.r
        trnd=int(-np.log10(min(dtstep,dtshow,min(dtsave))/100))
        # ct=time()
        t=t0
        if t0 < dtstep : #saving initial condition only if it's a new simulation staring at t0=0.0
            for l in range(len(dtsave)): #save t=t0
                self.fsave[l](r.t,r.y)
            
        tnext=round(t0+dtstep,trnd)
        tshownext=round(t0+dtshow,trnd)
        tsavenext=np.array([round(t0+l,trnd) for l in dtsave])
        while(t<t1):
            r.integrate(tnext)
            tnext=round(tnext+dtstep,trnd)
            t=r.t
            if(r.t>=tshownext):
                # print('t='+str(t)+', '+str(time()-ct)+" secs elapsed.")
                tshownext=round(tshownext+dtshow,trnd)
            for l in range(len(dtsave)):
                if(r.t>=tsavenext[l]):
                    tsavenext[l]=round(tsavenext[l]+dtsave[l],trnd)
                    self.fsave[l](r.t,r.y)
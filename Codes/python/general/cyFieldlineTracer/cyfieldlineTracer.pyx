from __future__ import division

"""
Base class and subclassed implementations of field line tracers used to trace the 3D
trajectory of magnetic field lines from a given magnetic equilibrium.

Now cythonized for 10x speedup

Nick Walkden, Dec 2015
"""

import numpy as np
cimport numpy as np
cimport cython
import sys
import matplotlib.pyplot as plt
from equilibrium import equilibrium 
from collections import namedtuple
from copy import deepcopy as copy
from scipy.interpolate import interp2d
from scipy.interpolate import UnivariateSpline as interp1d
from libc.math cimport cos,sin,fabs


class fieldline(object):
    """
    Simple class to hold information about a single field line.
    """

    
        
    def __init__(self,**kwargs):
        """
        R = Major Radius, Z = Vertical height, phi = toroidal angle
        X,Y,Z = Cartesian coordinates with origin at R=0
        S = Distance along field line
        B = magnetic field strength
        bR,bZ = magnetic field tangent vector components in R and Z
        bt,bp = magnetic field tangent vector components in toroidal and poloidal angle
        """
                
        #args = ['R','Z','phi','S','X','Y','B','bR','bt','bp','bR','bZ']        

        if 'R' in kwargs: self.R = np.array(kwargs['R'],dtype=np.float) 
        else: self.R = np.zeros(1,dtype=np.float)
        if 'Z' in kwargs: self.Z = np.array(kwargs['Z'],dtype=np.float)
        else: self.Z = np.zeros(1,dtype=np.float)
        if 'phi' in kwargs: self.phi = np.array(kwargs['phi'],dtype=np.float)
        else: self.phi = np.zeros(1,dtype=np.float)
        if 'S' in kwargs: self.S = np.array(kwargs['S'],dtype=np.float)
        else: self.S = np.zeros(1,dtype=np.float)
        if 'X' in kwargs: self.X = np.array(kwargs['X'],dtype=np.float)
        else: self.X = np.zeros(1,dtype=np.float)
        if 'Y' in kwargs: self.Y = np.array(kwargs['Y'],dtype=np.float)
        else: self.Y = np.zeros(1,dtype=np.float)
        if 'B' in kwargs: self.B = np.array(kwargs['B'],dtype=np.float)
        else: self.B = np.zeros(1,dtype=np.float)
        if 'bR' in kwargs: self.bR = np.array(kwargs['bR'],dtype=np.float)
        else: self.bR = np.zeros(1,dtype=np.float)
        if 'bt' in kwargs: self.bt = np.array(kwargs['bt'],dtype=np.float)
        else: self.bt = np.zeros(1,dtype=np.float)
        if 'bp' in kwargs: self.bp = np.array(kwargs['bp'],dtype=np.float)
        else: self.bp = np.zeros(1,dtype=np.float)
        if 'bR' in kwargs: self.bR = np.array(kwargs['bR'],dtype=np.float)
        else: self.bR = np.zeros(1,dtype=np.float)
        if 'bZ' in kwargs: self.bZ = np.array(kwargs['bZ'],dtype=np.float)
        else: self.bZ = np.zeros(1,dtype=np.float)
        
        
    def list(self):
        """
        return a list of variable names stored in the class 
        """
        return self.__dict__.keys()


    def _rotate_toroidal(self,double rotateAng=0.0):
        """
        Rotate the field line by a given angle in the toroidal direction
        """
        cdef int i
        for i in range(self.X.shape[0]):
            self.X[i] = self.R[i]*cos(self.phi[i] + rotateAng)
            self.Y[i] = self.R[i]*sin(self.phi[i] + rotateAng)
        self.phi = self.phi + rotateAng 
    
    def rotate_toroidal(self,double rotateAng=0.0):
        cdef int i
        for i in range(self.X.shape[0]):
            self.X[i] = self.R[i]*cos(self.phi[i] + rotateAng)
            self.Y[i] = self.R[i]*sin(self.phi[i] + rotateAng)
        self.phi = self.phi + rotateAng

    def at(self,key,value,returns=None):
    """
    Function to return values of parameters along the fieldline at a given
    point, eg S=10 or phi=3.14 or Z=-1.5
    
    example:   
        
        vals = fieldline.at('phi',3.14159)   #find all fieldline parameters at phi = pi
        
        R_at_pi = vals['R']
        Z_at_pi = vals['Z']
        
        #Find just the radial position at Z=-1.0
        
        vals = fieldline.at('Z',-1.0,returns=['R'])
        
        R = vals['R']
    """
    
    val_dict = {}
    if key not in self.list():
        print("WARNING: No "+key+" found in fieldline object, returning.\n")
        return val_dict
    
    inds_low = np.where(getattr(self,key) < value)[0]
    inds_high = np.where(getattr(self,key) > value)[0]
    
    #These will be used to identify discontinuities 
    inds_low_diff = np.diff(inds_low)
    inds_high_diff = np.diff(inds_high)
    
    if np.any(np.abs(inds_low_diff) > 1) or np.any(np.abs(inds_high_diff) > 1):
        #There are multiple points where the fieldline crosses value
        
    else:
        #There is only one point where the value is crossed
        
           
           
    
    
    def filter(self,keys,ranges,hard_check_range=True):
        """
        Function to return just the indices along the field-line that lie within the ranges requestion
        
        example:
            
            filtered_fieldline = my_fieldline.filter('R',[0.8,1.2])
            
            will return a new fieldline object that has been filtered so that it only
            contains the part of the fieldline between R = 0.8 and R = 1.2
            
            filtered_fieldline = my_fieldline.filter(['phi','R'],[[0.0,3.14],[0.8,1.2]])
            
            will filter between R = 0.8,1.2 and phi = 0,pi
            
            
        """
        bool_filter = np.ones(self.S.shape,dtype=bool)

        if type(keys)!=list:
            keys = [keys]
            ranges = [ranges]
        
        for key,limit in zip(keys,ranges):  
            if key in self.list():                
                bool_filter = bool_filter & (getattr(self,key) > limit[0]) & (getattr(self,key) < limit[1])
                
            else:
                print('Warning: Variable '+key+' not found, skipping filter of '+key+'\n')
        
        kwargs = {}
        
        for key in self.list():
            if getattr(self,key).shape == bool_filter.shape:
                kwargs[key] = getattr(self,key)[bool_filter]
               
                
        return fieldline(**kwargs)
    
points = namedtuple('points','x y')

cdef int ccw(float Ax,float Ay,float Bx,float By,float Cx,float Cy):
    """ check if points are counterclockwise """
    cdef int result
    if (Cy - Ay)*(Bx-Ax) > (By-Ay)*(Cx-Ax) : result = 1
    else: result = 0
    return result


cdef np.ndarray[dtype=np.float_t,ndim=1] wall_intersection(float Ax,float Ay,float Bx,float By,np.ndarray[dtype=np.float_t,ndim=1] wallR, np.ndarray[dtype=np.float_t,ndim=1] wallZ):
    """
    Check for an intersection between the line (x1,y1),(x2,y2) and the wall
    """
    cdef int i
    cdef float Cx,Cy,Dx,Dy,Mab,Mcd,Cab,Ccd
    cdef np.ndarray[dtype=np.float_t,ndim=1] intr = np.zeros(2,dtype=np.float)  
    for i in range(wallR.shape[0]-1):
        Cx = wallR[i]
        Cy = wallZ[i]
        Dx = wallR[i+1]
        Dy = wallZ[i+1]             
        if ccw(Ax,Ay,Cx,Cy,Dx,Dy) != ccw(Bx,By,Cx,Cy,Dx,Dy) and ccw(Ax,Ay,Bx,By,Cx,Cy) != ccw(Ax,Ay,Bx,By,Dx,Dy): 
            
            #Find gradient of lines
            if Ax == Bx:
                Mab = 1e10*(By-Ay)/fabs(By-Ay)
            else:
                Mab = (By - Ay)/(Bx - Ax)
            if Cx == Dx:
                Mcd = 1e10*(Dy-Cy)/fabs(Dy-Cy)
            else:   
                Mcd = (Dy - Cy)/(Dx - Cx)
                        
            #Find axis intercepts
            Cab = By - Mab*Bx
            Ccd = Cy - Mcd*Cx
            
            #Find line intersection point
            intr[0] = (Ccd - Cab)/(Mab - Mcd)
            intr[1] =  Cab + Mab*(Ccd - Cab)/(Mab - Mcd)
        
            
            return intr         

    return intr 

class fieldlineTracer(object):
    def __init__(self,**kwargs):
        """ initialize base class 
        
        keywords:
            shot = int      if given, read in data for shot on initialization
            gfile = file or str if given, read in data from gfile
            tind = int      if reading from idam use tind
            machine = str       state which machine to read data for
        """
        
        self.eq = equilibrium()     #Stored details of the efit equilibrium

        if 'psiN' in kwargs: self.psiN = kwargs['psiN']       
        else: self.psiN = 0.0
        if 'rev_Bt' not in kwargs: kwargs['rev_Bt'] = False
        if 'shot' in kwargs and 'gfile' not in kwargs or kwargs['gfile'] is None:
            shot = kwargs['shot']
            if 'machine' not in kwargs:
                print("\nWARNING: No machine given, assuming MAST")
                machine = 'MAST'
            else:
                machine = kwargs['machine']
            if 'time' not in kwargs:
                print("\nWarning: No time given, setting to 0.25")
                time = 0.25
            else:
                time = kwargs['time']
                
            self.get_equilibrium(shot=shot,machine=machine,time=time)
        
            
        elif 'gfile' in kwargs:
            #If a gfile is found use it by default
            gfile = kwargs['gfile']
            if gfile != None:
                self.get_equilibrium(gfile,rev_Bt=kwargs['rev_Bt'])
    
        else:
            print("WARNING: No gfile, or shot number given, returning.")
            
        self.kwargs = kwargs
        

    def get_equilibrium(self,gfile=None,shot=None,machine=None,time=None,rev_Bt=False):
        """ Read in data from efit

        keywords:
            machine = str   machine to load shot number for
        """
        
        if gfile==None and shot != None:
            #load from IDAM
            if machine == 'MAST':
                self.eq.load_MAST(shot,time)
            elif machine == 'JET':
                self.eq.load_JET(shot,time)
        elif gfile != None:
            #load from gfile
            self.eq.load_geqdsk(gfile,rev_Bt=rev_Bt)
    
        
    
    
    def set_psiN(self,psiN):
        self.psiN = psiN

    
    def trace(self,float Rstart,float Zstart,float phistart=0.0,int mxstep=1000,float ds=1e-3,int backward=0,
        int verbose=1,int full_output=0,float psiN=0.0,float tor_lim=628.0,int reverse=1):
        from scipy.interpolate import interp1d
        if self.psiN != 0.0:
            if verbose: print("psiN found in __dict__, reverting to psiN of "+str(self.psiN))
            psiN = self.psiN        

        if psiN:
                
            #Ensure that the starting points lie on the desired flux surface
            if verbose:
                print("\n Refining starting position to ensure flux surface alignment for psi_N = "+str(psiN)+"\n")
                print("Original:\tRstart = "+str(Rstart)+"\tZstart = "+str(Zstart)+"\n")
            
            #Keep Zstart and find refined Rstart
            Rax = np.linspace(np.max([np.min(self.eq.R),Rstart-0.5]),np.min([Rstart+0.5,np.max(self.eq.R)]),200)    
            psi_R = self.eq.psiN(Rax,Zstart)[0]
            #Now get R(psi)
            R_psi = interp1d(psi_R,Rax) 
            Rstart = R_psi(psiN)
            if verbose: print("Refined:\tRstart = "+str(Rstart)+"\tZstart = : "+str(Zstart)+"\n")           
            if backward:
                self._currentS=0.0
            
        cdef np.ndarray[dtype=np.float_t,ndim=1] s = np.zeros(mxstep,dtype=np.float)
        cdef np.ndarray[dtype=np.float_t,ndim=1] R = np.zeros(mxstep,dtype=np.float)
        cdef np.ndarray[dtype=np.float_t,ndim=1] Z = np.zeros(mxstep,dtype=np.float)
        cdef np.ndarray[dtype=np.float_t,ndim=1] phi = np.zeros(mxstep,dtype=np.float)
        cdef np.ndarray[dtype=np.float_t,ndim=1] X = np.zeros(mxstep,dtype=np.float)
        cdef np.ndarray[dtype=np.float_t,ndim=1] Y = np.zeros(mxstep,dtype=np.float)
        s[0] = 0.0
        R[0] = Rstart
        Z[0] = Zstart
        phi[0] = phistart
        X[0] = Rstart*cos(phistart)
        Y[0] = Rstart*sin(phistart)
        cdef int i = 0
        cdef float dR,dZ,dphi,phicol
        cdef int check_intr
        cdef object take_step = self.take_step  
        cdef np.ndarray[dtype=np.float_t,ndim=1] wallR,wallZ,step,intr
        if self.eq.wall != None:
            wallR = np.array(self.eq.wall['R'],dtype=np.float)
            wallZ = np.array(self.eq.wall['Z'],dtype=np.float)
            check_intr = 1
        else:
            check_intr = 0
        while i < mxstep-1:
            
            step = take_step(R=R[i],Z=Z[i],ds=ds)
            dR = step[0]
            dZ = step[1]
            dphi = step[2]

            if check_intr:                  
                    intr = wall_intersection(R[i],Z[i],R[i]+dR,Z[i]+dZ,wallR,wallZ)
                    if intr[0]:
        
                        R[i+1] = intr[0]
                        Z[i+1] = intr[1]
                        phicol = phi[i] + dphi*(intr[0]-R[i])/dR
                        phi[i+1] = phicol
                        X[i+1] = intr[0]*cos(phicol)
                        Y[i+1] = intr[0]*sin(phicol)
                        s[i+1] = s[i] + (ds/fabs(ds))*((X[i+1]-X[i])**2.0 + (Y[i+1] - Y[i])**2.0 + (Z[i+1] - Z[i])**2.0)**0.5
                        
                        if backward:
                            #Stop the trace if already running backwards
                            break
                            
                        back = self.trace(Rstart,Zstart,phistart,mxstep,-ds,backward=1,verbose=1,full_output=full_output,psiN=psiN)
                        R = np.delete(R,np.arange(mxstep-i-2)+i+2)
                        Z = np.delete(Z,np.arange(mxstep-i-2)+i+2)
                        phi = np.delete(phi,np.arange(mxstep-i-2)+i+2)
                        X = np.delete(X,np.arange(mxstep-i-2)+i+2)
                        Y = np.delete(Y,np.arange(mxstep-i-2)+i+2)
                        s = np.delete(s,np.arange(mxstep-i-2)+i+2)

                        R = np.concatenate((back.R[::-1],R[1:]))
                        Z = np.concatenate((back.Z[::-1],Z[1:]))
                        phi = np.concatenate((back.phi[::-1],phi[1:]))
                        X = np.concatenate((back.X[::-1],X[1:]))
                        Y = np.concatenate((back.Y[::-1],Y[1:]))
                        s = np.concatenate((back.S[::-1],s[1:]))
                        break
            
            R[i+1] = R[i] + dR
            Z[i+1] = Z[i] + dZ
            phi[i+1] = phi[i] + dphi            
            X[i+1] = R[i+1]*cos(phi[i+1])
            Y[i+1] = R[i+1]*sin(phi[i+1])       
            s[i+1] = s[i] + ds
            if fabs(phi[0]-phi[i+1]) > tor_lim:
                #Toroidal limit reached so stop tracing
                break

            i += 1
        

        
        if reverse and not intr[0] and not backward:
            back = self.trace(Rstart,Zstart,phistart,mxstep,-ds,backward=1,verbose=1,full_output=full_output,psiN=psiN,reverse=0,tor_lim=tor_lim)
            R = np.delete(R,np.arange(mxstep-i-2)+i+2)
            Z = np.delete(Z,np.arange(mxstep-i-2)+i+2)
            phi = np.delete(phi,np.arange(mxstep-i-2)+i+2)
            X = np.delete(X,np.arange(mxstep-i-2)+i+2)
            Y = np.delete(Y,np.arange(mxstep-i-2)+i+2)
            s = np.delete(s,np.arange(mxstep-i-2)+i+2)
                    
            R = np.concatenate((back.R[::-1],R[1:]))
            Z = np.concatenate((back.Z[::-1],Z[1:]))
            phi = np.concatenate((back.phi[::-1],phi[1:]))
            X = np.concatenate((back.X[::-1],X[1:]))
            Y = np.concatenate((back.Y[::-1],Y[1:]))
            s = np.concatenate((back.S[::-1],s[1:]))
        
        if backward and i != mxstep-1:
            R = np.delete(R,np.arange(mxstep-i-2)+i+2)
            Z = np.delete(Z,np.arange(mxstep-i-2)+i+2)
            phi = np.delete(phi,np.arange(mxstep-i-2)+i+2)
            X = np.delete(X,np.arange(mxstep-i-2)+i+2)
            Y = np.delete(Y,np.arange(mxstep-i-2)+i+2)
            s = np.delete(s,np.arange(mxstep-i-2)+i+2)
            
        return fieldline(R=R,Z=Z,phi=phi,X=X,Y=Y,S=s)
        
        

class RK4Tracer(fieldlineTracer):
    """
    RK4 field line tracer
    """     

    def __init__(self,**kwargs):
        fieldlineTracer.__init__(self,**kwargs)
        from scipy.interpolate import interp2d
        
        if 'interp' not in kwargs or kwargs['interp'] is None:
            interp = 'linear'
        else:
            interp = kwargs['interp']
        if self.eq.B is None:
            self.eq.calc_bfield()
            
        self._bR = interp2d(self.eq.R,self.eq.Z,self.eq.BR/self.eq.B,kind=interp)
        self._bZ = interp2d(self.eq.R,self.eq.Z,self.eq.BZ/self.eq.B,kind=interp)
        self._bt = interp2d(self.eq.R,self.eq.Z,self.eq.Bt/self.eq.B,kind=interp)

    def take_step(self,float R,float Z,float ds):
        """
        Take an RK4 step along the field line
        
        R,Z starting values
        ds  step size
        bR,bZ,bphi  Magnetic field functions that must be able to perform F(R,Z) = num 
        bpol not used here  
        """
        cdef object bR = self._bR
        cdef object bZ = self._bZ
        cdef object bt = self._bt
        cdef float dR1,dZ1,dphi1
        cdef float dR2,dZ2,dphi2
        cdef float dR3,dZ3,dphi3
        cdef float dR4,dZ4,dphi4
        cdef float dR,dZ,dphi
        dR1 = ds*bR(R,Z)
        dZ1 = ds*bZ(R,Z)
        dphi1 = ds*bt(R,Z)/R
        
        dR2 = ds*bR(R+0.5*dR1,Z+0.5*dZ1)
        dZ2 = ds*bZ(R+0.5*dR1,Z+0.5*dZ1)
        dphi2 = ds*bt(R+0.5*dR1,Z+0.5*dZ1)/R
        
        dR3 = ds*bR(R+0.5*dR2,Z+0.5*dZ2)
        dZ3 = ds*bZ(R+0.5*dR2,Z+0.5*dZ2)
        dphi3 = ds*bt(R+0.5*dR2,Z+0.5*dZ2)/R
        
        dR4 = ds*bR(R+dR3,Z+dZ3)
        dZ4 = ds*bZ(R+dR3,Z+dZ3)
        dphi4 = ds*bt(R+dR3,Z+dZ3)/R

        cdef np.ndarray[dtype=np.float_t,ndim=1] result = np.zeros(3,dtype=np.float)    
        dR = (1./6.)*(dR1 + 2.0*dR2 + 2.0*dR3 + dR4)
        dZ = (1./6.)*(dZ1 + 2.0*dZ2 + 2.0*dZ3 + dZ4)
        dphi = (1./6.)*(dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4)
        result[0] = dR
        result[1] = dZ
        result[2] = dphi
        
        return result
        


                
        
class EulerTracer(fieldlineTracer):
    """
    Forward Euler field line tracer (for benchmarking purposes ONLY)
    """     
    
    def __init__(self,**kwargs):
        fieldlineTracer.__init__(self,**kwargs)
            
        if self.eq.B is None:
            self.eq.calc_bfield()
            
        if 'interp' not in kwargs:
            interp = 'linear'
        else:
            interp = kwargs['interp']
            
        self._bR = interp2d(self.eq.R,self.eq.Z,self.eq.BR/self.eq.B,kind=interp)
        self._bZ = interp2d(self.eq.R,self.eq.Z,self.eq.BZ/self.eq.B,kind=interp)
        self._bt = interp2d(self.eq.R,self.eq.Z,self.eq.Bt/self.eq.B,kind=interp)
        
    def take_step(self,float R,float Z,float ds):
        """
        Take a forward Euler step along the field line
        
        R,Z starting values
        dl  step size
        bR,bZ,bphi  Magnetic field functions that must be able to perform F(R,Z) = num 
        bpol not used here  
        """
        cdef float dR,dZ,dphi
        cdef np.ndarray[dtype=np.float_t,ndim=1] result = np.zeros(3,dtype=float)
        dR = ds*self._bR(R,Z)
        dZ = ds*self._bZ(R,Z)
        dphi = ds*self._bt(R,Z)/R
        result[0] = dR
        result[1] = dZ
        result[2] = dphi
        
        return result
        

class fluxsurfaceTracer(fieldlineTracer):
    """
    Trace a field line constrained to an equilibrium flux surface
    
    Steps along a field line but ensures that the field line remains on the desired flux-surface
    
    This is curretly limited by the resolution of the equilibrium flux-surface location
    
    """     
    
    
    def __init__(self,**kwargs):
        fieldlineTracer.__init__(self,**kwargs) 
        if 'interp' not in kwargs:
            interp = 'linear'
        else:
            interp = kwargs['interp']
                
        if 'psiN' in kwargs and kwargs['psiN'] != None:
            self.psiN = kwargs['psiN']
        else:
            self.psiN = None    

        
        if self.eq.B is None:
            self.eq.calc_bfield()
            
        
            
        self._Bt = interp2d(self.eq.R,self.eq.Z,self.eq.Bt,kind=interp)
        self._Bp = interp2d(self.eq.R,self.eq.Z,self.eq.Bp,kind=interp)
        
        self._initialized=False
        
    def set_psiN(self,psiN):
        
        self.psiN = psiN
        self.FS = self.eq.get_fluxsurface(self.psiN,self._Rstart,self._Zstart)
        self._initialized = False
                
    def init(self,Rstart,Zstart):
        """
        initialization that cannot be contained in __init__
        """
        self._Rstart = Rstart
        self._Zstart = Zstart   

        if self.psiN is None:
            self.psiN = float(raw_input("Enter normalized psi of flux-surface"))
        self.set_psiN(self.psiN)
        
        #Get some interpolation functions for use later
        #when stepping along the field line
        phi = [0.0]
        L = [0.0]
        S = [0.0]
        
        for i in np.arange(len(self.FS.R)-1):
            
            #Shift in distance along flux surface in poloidal direction
            dl = ((self.FS.R[i+1] - self.FS.R[i])**2.0 + (self.FS.Z[i+1] - self.FS.Z[i])**2.0)**0.5
            
            #Corresponding shift along field line
            ds = self._Bt(self.FS.R[i+1],self.FS.Z[i+1])*dl/(self._Bp(self.FS.R[i+1],self.FS.Z[i+1]))
            
            #Shift in toroidal angle
            dphi = ds/self.FS.R[i+1]
            
            L.append(L[i] + dl)
            S.append(S[i] + ds)
            phi.append(phi[i] + dphi)
            
            
        #Find the starting point of S
        def sign_change(p1,p2):
            if p1*p2/np.abs(p1*p2) < 0:
                return True
            else:
                return False
                
        i = 0
        
        #Rtest = self.FS.R - Rstart
        Ztest = np.abs(np.asarray(self.FS.Z) - Zstart)
        i = Ztest.argmin()
        
            
        dl = ((Rstart - self.FS.R[i])**2.0 + (Zstart - self.FS.Z[i])**2.0)**0.5
        ds = self._Bt(Rstart,Zstart)*dl/(self._Bp(Rstart,Zstart))
        
        S -= S[i] + ds
            
        #Now generate R,Z,phi as a function of S
        
        
        self._Rs = interp1d(S,self.FS.R,s=0.0005)
        self._Zs = interp1d(S,self.FS.Z,s=0.0005)
        self._phis = interp1d(S,phi)
         
        self._initialized = True
        self._currentS = 0.0
        
    def take_step(self,R,Z,ds):
        
        if not self._initialized:
            self.init(R,Z)
            
        dR = self._Rs(self._currentS + ds) - R
        dZ = self._Zs(self._currentS + ds) - Z
        dphi = self._phis(self._currentS + ds) - self._phis(self._currentS)
        
        self._currentS += ds
        
        return float(dR),float(dZ),float(dphi)
            
            
            
def get_fieldline_tracer(mode,**kwargs):
    if mode == 'Euler':
        return EulerTracer(**kwargs)
    elif mode == 'RK4':
        return RK4Tracer(**kwargs)
    elif mode == 'fluxsurface':
        return fluxsurfaceTracer(**kwargs)

    else:
        raise TypeError("No tracer of type ",mode," implemented")
        return None


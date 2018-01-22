import os
import dd
import map_equ
from scipy.interpolate import UnivariateSpline
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib._cntr as cntr
from mpl_toolkits.mplot3d import Axes3D
from cyfieldlineTracer import get_fieldline_tracer
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline 
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from time_series_tools import identify_bursts2
from scipy import constants
from collections import namedtuple
fluxSurface = namedtuple('fluxSurface','R Z')
Point = namedtuple('Point', 'r z')
def interp2d(R,Z,field):
    return RectBivariateSpline(R,Z,np.transpose(field))


class equilibriumField(object):
    """
    Container for fields of the equilibrium. 

    An equilibrium field can be accessed either by indexing, ie
    
        data = myfield[i,j]

    or as a function, ie

        data = myfield(R,Z)

    NOTE:
        At present, operations on two equilibriumField objects, i.e

        equilibriumField + equilibriumField 

        return the value of their data only as a numpy array, and do not return
        an equilibriumField object
    """
    def __init__(self,data,function):

        self._data = data           
        self._func = function

    def __getitem__(self,inds):
        return self._data[inds]
    
    def __len__(self):
        return len(self._data)  
    
    def __add__(self,other):
        if type(other) is type(self):
            return self._data + other._data
        else:
            return self._data + other

    def __radd__(self,other):
        return self.__add__(other)

    def __sub__(self,other):
        if type(other) is type(self):
            return self._data - other._data
        else:
            return self._data - other

    def __truediv__(self,other):
        if type(other) is type(self):
            return self._data/other._data
        else:
            return self._data/other
    def __itruediv__(self,other):
        if type(other) is type(self):
            return other._data/self._data
        else:
            return other/self._data

    def __rsub__(self,other):
        return -1.0*(self.__sub__(other))

    def __mul__(self,other):
        if type(other) is type(self):
            return self._data*other._data
        else:
            return self._data*other
    
    def __rmul__(self,other):
        if type(other) is type(self):
            return other._data*self._data
        else:
            return other*self._data
    def __pow__(self,power):
        return self._data**power

    def __setitem__(self,inds,values):
        self._data[inds] = values

    def __call__(self,*args,**kwargs):
        if len(self._data.shape)>1:
            return np.transpose(self._func(*args,**kwargs))
        else:
            return self._func(*args,**kwargs)


class Target(object):

    def __init__(self, shot):
        """
        Class to deal with already processed data from divertor probes. It basically
        remap the position on the appropriate equilibrium,  compute the
        profiles and through a FLT code it also computes the LambdaDiv 
        
        Parameters
        ----------
        shot : :obj: 'int'
            Shot number

        Requirements
        ------------
        map_equ : class for equilibrium
        cyfieldlineTracer : cythonized version of Field Line Tracing code
        
        """

        self.shot = shot
        self._loadeq()
        # defining the positions of the probes
        self.OuterTarget = {
            'ua1': {'R': 1.582, 'z': -1.199, 's': 1.045},
            'ua2': {'R': 1.588, 'z': -1.175, 's': 1.070},
            'ua3': {'R': 1.595, 'z': -1.151, 's': 1.094},
            'ua4': {'R': 1.601, 'z': -1.127, 's': 1.126},
            'ua5': {'R': 1.608, 'z': -1.103, 's': 1.158},
            'ua6': {'R': 1.614, 'z': -1.078, 's': 1.189},
            'ua7': {'R': 1.620, 'z': -1.054, 's': 1.213},
            'ua8': {'R': 1.627, 'z': -1.030, 's': 1.246},
            'ua9': {'R': 1.640, 'z': -0.982, 's': 1.276}}

        self.OuterLimiter = {
            'uaa': {'R': 1.659, 'z': -0.936, 's': 1.326},
            'uab': {'R': 1.674, 'z': -0.909, 's': 1.356},
            'uac': {'R': 1.689, 'z': -0.884, 's': 1.387},
            'uad': {'R': 1.729, 'z': -0.843, 's': 1.447},
            'uae': {'R': 1.786, 'z': -0.788, 's': 1.526},
            'uaf': {'R': 1.857, 'z': -0.707, 's': 1.634},
            'uag': {'R': 1.899, 'z': -0.660, 's': 1.696},
            'uah': {'R': 1.963, 'z': -0.587, 's': 1.793}}

        self.InnerTarget={
            'ui9': {'R': 1.288, 'z': -0.959, 's': 0.339},
            'ui8': {'R': 1.281, 'z': -0.993, 's': 0.373},
            'ui7': {'R': 1.275, 'z': -1.011, 's': 0.391},
            'ui6': {'R': 1.269, 'z': -1.029, 's': 0.411},
            'ui5': {'R': 1.262, 'z': -1.047, 's': 0.429},
            'ui4': {'R': 1.256, 'z': -1.065, 's': 0.448},
            'ui3': {'R': 1.250, 'z': -1.083, 's': 0.468},
            'ui2': {'R': 1.244, 'z': -1.101, 's': 0.486},
            'ui1': {'R': 1.238, 'z': -1.119, 's': 0.505}}

        self.InnerLimiter = {
            'uig': {'R': 1.151, 'z': -0.671, 's': 0.012},
            'uif': {'R': 1.176, 'z': -0.710, 's': 0.059},
            'uie': {'R': 1.198, 'z': -0.744, 's': 0.099},
            'uid': {'R': 1.233, 'z': -0.798, 's': 0.163},
            'uic': {'R': 1.267, 'z': -0.858, 's': 0.230},
            'uib': {'R': 1.281, 'z': -0.902, 's': 0.278},
            'uia': {'R': 1.286, 'z': -0.929, 's': 0.309}}

        self.Dome = {
            'um1': {'R': 1.297, 'z': -1.084, 's': 0.617},
            'um2': {'R': 1.344, 'z': -1.062, 's': 0.673}, 
            'um3': {'R': 1.372, 'z': -1.062, 's': 0.701},
            'um4': {'R': 1.405, 'z': -1.062, 's': 0.732},
            'um5': {'R': 1.429, 'z': -1.062, 's': 0.757},
            'um6': {'R': 1.457, 'z': -1.062, 's': 0.786},
            'um7': {'R': 1.508, 'z': -1.119, 's': 0.862},
            'um8': {'R': 1.537, 'z': -1.153, 's': 0.906}}
        
        # now define the shotfile where processed langmuir data
        # are stored
        try:
            self.Lsd = dd.shotfile('LSD', self.shot)
            self.time = self.Lsd('time-lsd')
            try:
                self._InnerTargetProfile()
            except:
                print('Inner target profile not computed')

            try:
                self._OuterTargetProfile()
            except:
                print('Outer target profile not computed')

            try:
                self._InnerLimiterProfile()
            except:
                print('Inner limiter profile not computed')

            try:
                self._OuterLimiterProfile()
            except:
                print('Outer limiter profile not computed')
        except:
            print('LSD shot file not available for shot %5i' % shot)
            pass

    def plotLocation(self, t0=3):
        """
        Simple method plot the probe positions together with an
        equilibrium at a given time point
        """

        self._setTimeEq(time=t0)
        # plot the 
        fig, ax = mpl.pylab.subplots(figsize=(6, 8), nrows=1, ncols=1)
        for key in self.rg.iterkeys():
            ax.plot(self.rg[key], self.zg[key], 'k')
        # now contour of the equilibrium
        # core
        ax.contour(self.R, self.Z, self.psiN,
                   np.linspace(0.01, 0.95, num=9), colors='k', linestyle='-')
        ax.contour(self.R, self.Z, self.psiN, [1],
                   colors='r', linestyle='-', linewidths=2)
        # SOL
        ax.contour(self.R, self.Z, self.psiN,
                   np.linspace(1.01, 1.16, num=5), colors='k', linestyles='--')
        ax.set_xlabel(r'R [m]')
        ax.set_ylabel(r'Z [m]')
        ax.set_title(r'# %5i' % self.shot + ' @ t = %3.2f' % self._time)
        # plot also the position of the probes with different
        # colors according to 
        _all = (self.InnerLimiter, self.InnerTarget, self.Dome,
                self.OuterTarget, self.OuterLimiter)
        _col = ( '#D72085', '#7AB800', '#E29317', '#0098D8', '#E00034')
        for _p, _c in zip(_all, _col):
            for key in _p.iterkeys():
                ax.plot(_p[key]['R'], _p[key]['z'], 'o', color=_c)

        ax.set_aspect('equal')

    def _loadeq(self):
        # loading the equilibrium
        self.Eqm = map_equ.equ_map()
        status = self.Eqm.Open(self.shot, diag='EQH')
        self.Eqm._read_scalars()
        self.Eqm._read_profiles()
        self.Eqm._read_pfm()
        # load the wall for aug 
        self.rg, self.zg = map_equ.get_gc()
        self._psi = self.Eqm.pfm.transpose()
        self._time_array = self.Eqm.t_eq
        nr = self._psi.shape[0]
        nz = self._psi.shape[1]
        self._r = self.Eqm.Rmesh       
        self._z   = self.Eqm.Zmesh      
        self._psi_axis = self.Eqm.psi0
        self._psi_bnd  = self.Eqm.psix
        # get the fpol in similar way
        # as done in eqtools
        self._jpol = self.Eqm.jpol
        # these are the lower xpoints
        self._rxpl = self.Eqm.ssq['Rxpu']       
        self._zxpl = self.Eqm.ssq['Zxpu']       
        # read also the upper xpoint
        self._rxpu = self.Eqm.ssq['Rxpo']
        self._zxpu = self.Eqm.ssq['Zxpo']
        # R magnetic axis
        self._axisr = self.Eqm.ssq['Rmag']          
        # Z magnetic axis
        self._axisz = self.Eqm.ssq['Zmag'] 
        # eqm does not load the RBphi on axis
        Mai = dd.shotfile('MAI', self.shot)
        self.Rcent = 1.65
        # we want to interpolate on the same time basis
        Spl = UnivariateSpline(Mai('BTF').time, Mai('BTF').data, s=0)
        self._bphi = Spl(self._time_array)*self.Rcent
        Mai.close()
        Mag = dd.shotfile('MAG', self.shot)
        Spl = UnivariateSpline(Mag('Ipa').time, Mag('Ipa').data, s=0)
        self._cplasma = Spl(self._time_array)
        # we want to load also the plasma curent
        # now define the psiN
        self._psiN = (self._psi-self._psi_axis[:, np.newaxis, np.newaxis])/ \
            (self._psi_bnd[:, np.newaxis, np.newaxis]-self._psi_axis[:, np.newaxis, np.newaxis])


    def _setTimeEq(self, time=3):
        """
        Create an equilibrium object at a given time which can
        be later used for plot or for saving gfiles
        """
        self.R = self._r
        self.Z = self._z
        tind = np.abs(self._time_array - time).argmin()     
        psi_func = interp2d(self.R,self.Z,self._psi[tind])
        self.psi = equilibriumField(self._psi[tind],psi_func) 
        self.nr = len(self.R)
        self.nz = len(self.Z)       
        self.psi_axis = self._psi_axis[tind]
        self.psi_bnd = self._psi_bnd[tind]
        self.Btcent = self._bphi[tind]
        self.sigBp = np.sign(self._cplasma[tind])
        fpol = self._jpol[:, tind]*2e-7
        fpol = fpol[:np.argmin(np.abs(fpol))]
        psigrid = np.linspace(self.psi_axis, self.psi_bnd, len(fpol))
        self.fpol = equilibriumField(fpol, UnivariateSpline(psigrid, fpol, s=0))
        self.nxpt = 2
        tind_xpt = tind#np.abs(self._xpoint1r.time - time).argmin()
        self.xpoints = {'xpl':Point(self._rxpl[tind], self._zxpl[tind]),
                        'xpu':Point(self._rxpu[tind], self._zxpu[tind])}
        self.spoints = None
        #self.xpoint.append(Point(self._xpoint2r.data[tind_xpt],self._xpoint2z.data[tind_xpt]))
        self.axis = Point(self._axisr[tind],self._axisz[tind])      
        self.fpol = None        
    
        self._loaded = True
        self._time = self._time_array[tind]
    
        psiN_func = interp2d(self.R,self.Z,(self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis))
        self.psiN = equilibriumField(
            (self.psi[:] - self.psi_axis)/(self.psi_bnd-self.psi_axis),psiN_func)
    
        VesselFile = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'augVesseldata.txt')
        x, y = np.loadtxt(VesselFile, unpack=True)
        self.wall = { 'R' : x, 'Z' : y }
        self.calc_bfield()

    def __calc_psi_deriv(self,method='CD'):
        """
        Calculate the derivatives of the poloidal flux on the grid in R and Z
        """
        if self.psi is None or self.R is None or self.Z is None:
            print("ERROR: Not enough information to calculate grad(psi). Returning.")
            return
        
        
        if method is 'CD':  #2nd order central differencing on an interpolated grid
            R = np.linspace(np.min(self.R),np.max(self.R),200)
            Z = np.linspace(np.min(self.Z),np.max(self.Z),200)
            Rgrid,Zgrid = R,Z #np.meshgrid(R,Z)
            psi  = self.psi(Rgrid,Zgrid)
            deriv = np.gradient(psi)    #gradient wrt index
            #Note np.gradient gives y derivative first, then x derivative
            ddR = deriv[1]
            #ddR = self.psi(Rgrid,Zgrid,dx=1)
            ddZ = deriv[0]
            #ddZ = self.psi(Rgrid,Zgrid,dy=1)
            dRdi = 1.0/np.gradient(R)
            dZdi = 1.0/np.gradient(Z)
            dpsidR = ddR*dRdi[np.newaxis,:] #Ensure broadcasting is handled correctly
            dpsidZ = ddZ*dZdi[:,np.newaxis]
            dpsidR_func = interp2d(R,Z,dpsidR)
            dpsidZ_func = interp2d(R,Z,dpsidZ)

            RR,ZZ = self.R,self.Z
            self.dpsidR = equilibriumField(np.transpose(dpsidR_func(RR,ZZ)),dpsidR_func)
            self.dpsidZ = equilibriumField(np.transpose(dpsidZ_func(RR,ZZ)),dpsidZ_func)
        else:
            print("ERROR: Derivative method not implemented yet, reverting to CD")
            self.calc_psi_deriv(method='CD')
            
            
    def calc_bfield(self):
                
        """Calculate magnetic field components"""

        self.__calc_psi_deriv()     
            
        BR = -1.0*self.dpsidZ/self.R[np.newaxis,:]
        BZ = self.dpsidR/self.R[np.newaxis,:]
        Bp = self.sigBp*(BR**2.0 + BZ**2.0)**0.5    
            
        self.__get_fpolRZ()
        Bt = self.fpolRZ/self.R[np.newaxis,:]
        B = (BR**2.0 + BZ**2.0 + Bt**2.0)**0.5

        BR_func = interp2d(self.R,self.Z,BR)
        BZ_func = interp2d(self.R,self.Z,BZ)
        Bp_func = interp2d(self.R,self.Z,Bp)
        Bt_func = interp2d(self.R,self.Z,Bt)
        B_func = interp2d(self.R,self.Z,B)

        self.BR = equilibriumField(BR,BR_func)
        self.BZ = equilibriumField(BZ,BZ_func)
        self.Bp = equilibriumField(Bp,Bp_func)
        self.Bt = equilibriumField(Bt,Bt_func)
        self.B = equilibriumField(B,B_func)
         

    def __get_fpolRZ(self,plasma_response=False):
        """ 
        Generate fpol on the RZ grid given fpol(psi) and psi(RZ)
        fpol(psi) is given on an evenly spaced grid from psi_axis to psi_bnd. This means that
        some regions of psi will exceed the range that fpol is calculated over. 
        When this occurs (usually in the SOL) we assume that the plasma contribution
        is negligable and instead just take the vacuum assumption for the B-field and 
        reverse engineer
        
        """
        from scipy.interpolate import interp1d
                
        fpolRZ = np.zeros((self.nz,self.nr))
        
        if plasma_response and self.fpol is not None:
            psigrid = np.linspace(self.psi_axis,self.psi_bnd,len(self.fpol))
            for i in np.arange(self.nr):
                for j in np.arange(self.nz):
                    if self.psi[i,j] < psigrid[-1] and self.psi[i,j] > psigrid[0]:
                        fpolRZ[i,j] = self.fpol(self.psi[i,j])
                    else:
                        fpolRZ[i,j] = self.Btcent*self.Rcent
        
        else:
            fpolRZ[:,:] = self.Btcent*self.Rcent
                
        fpolRZ_func = interp2d(self.R,self.Z,fpolRZ)
        self.fpolRZ = equilibriumField(fpolRZ,fpolRZ_func)


    def _InnerTargetProfile(self):
        """
        Build the appropriate dictionary containing the information with the
        name of the probes the R and Z position and dividing into inner-target
        outer-targer, dome
        """

        self.InnerTargetNe = np.asarray([self.Lsd('ne-' + key).data
                                        for key in self.InnerTarget.keys()])
        self.InnerTargetTe = np.asarray([self.Lsd('te-' + key).data
                                        for key in self.InnerTarget.keys()])
        self.InnerTargetRhop = np.asarray([
                self.Eqm.rz2rho(self.InnerTarget[key]['R'],
                                self.InnerTarget[key]['z'],
                                self.time, extrapolate=True)
                for key in self.InnerTarget.keys()]).squeeze()

    def _OuterTargetProfile(self):
        """
        Build the appropriate dictionary containing the information with the
        name of the probes the R and Z position and dividing into inner-target
        outer-targer, dome
        """

        self.OuterTargetNe = np.asarray([self.Lsd('ne-' + key).data
                                        for key in self.OuterTarget.keys()])
        self.OuterTargetTe = np.asarray([self.Lsd('te-' + key).data
                                        for key in self.OuterTarget.keys()])
        self.OuterTargetRhop = np.asarray([
                self.Eqm.rz2rho(self.OuterTarget[key]['R'],
                                self.OuterTarget[key]['z'],
                                self.time, extrapolate=True)
                for key in self.OuterTarget.keys()]).squeeze()

    def _InnerLimiterProfile(self):
        """
        Build the appropriate dictionary containing the information with the
        name of the probes the R and Z position and dividing into inner-target
        outer-targer, dome
        """

        self.InnerLimiterNe = np.asarray([self.Lsd('ne-' + key).data
                                        for key in self.InnerLimiter.keys()])
        self.InnerLimiterTe = np.asarray([self.Lsd('te-' + key).data
                                        for key in self.InnerLimiter.keys()])
        self.InnerLimiterRhop = np.asarray([
                self.Eqm.rz2rho(self.InnerLimiter[key]['R'],
                                self.InnerLimiter[key]['z'],
                                self.time, extrapolate=True)
                for key in self.InnerLimiter.keys()]).squeeze()

    def _OuterLimiterProfile(self):
        """
        Build the appropriate dictionary containing the information with the
        name of the probes the R and Z position and dividing into inner-target
        outer-targer, dome
        """

        self.OuterLimiterNe = np.asarray([self.Lsd('ne-' + key).data
                                        for key in self.OuterLimiter.keys()])
        self.OuterLimiterTe = np.asarray([self.Lsd('te-' + key).data
                                        for key in self.OuterLimiter.keys()])
        self.OuterLimiterRhop = np.asarray([
                self.Eqm.rz2rho(self.OuterLimiter[key]['R'],
                                self.OuterLimiter[key]['z'],
                                self.time, extrapolate=True)
                for key in self.OuterLimiter.keys()]).squeeze()


    def PlotEnProfile(self, Type='OuterTarget', trange=[3, 3.1],
                      interelm=False, elm=False,
                      Plot=True, **kwargs):
        """
        Create the appropriate plot of the Density profile in the
        desired time range for the desired set of discharges
        
        Parameters
        ----------
        Type : 'str'. Possible values are 'OuterTarget',  'OuterLimiter',
            'InnerTarget', 'InnerLimiter'
        
        trange : 2D array indicating the time range for the profiles

        interelm : Boolean,  default is no. If True it create an interELM mask
               based on a smoothed 
        
        """

        if Type == 'OuterTarget':
            sig = self.OuterTargetNe
            rho = self.OuterTargetRhop
        elif Type == 'InnerTarget':
            sig = self.InnerTargetNe
            rho = self.InnerTargetRhop
        elif Type == 'InnerLimiter':
            sig = self.InnerLimiterNe
            rho = self.InnerLimiterRhop
        elif Type == 'OuterLimiter':
            sig = self.OuterLimiterNe
            rho = self.OuterLimiterRhop
        else:
            'assuming outer Target'
            sig = self.OuterTargetNe
            rho = self.OuterTargetRhop

        _idx = np.where((self.time >= trange[0]) & 
                        (self.time <= trange[1]))[0]
        t = self.time[_idx]
        sig = sig[:, _idx]
        rho = rho[:, _idx]

        if interelm:
            print('Computing profiles in inter-ELM phase')
            self._maskElm(trange=trange, **kwargs)
            t = t[self._interElm]
            sig = sig[:, self._interElm]
            rho = rho[:, self._interElm]
        if elm:
            self._maskElm(trange=trange, **kwargs)
            t = t[self._Elm]
            sig = sig[:, self._Elm]
            rho = rho[:, self._Elm]
        # since there is no way to compute the mean and standard
        # deviation eliminating at the same time nan and inf we need to
        # cycle
        xOut = np.array([])
        yOut = np.array([])
        eOut = np.array([])
        for r in range(rho.shape[0]):
            if np.count_nonzero(np.isfinite(sig[r, :])) != 0:
                xOut = np.append(xOut, np.mean(rho[r, np.isfinite(sig[r, :])]))
                yOut = np.append(yOut, np.mean(sig[r, np.isfinite(sig[r, :])]))
                eOut = np.append(eOut, np.std(sig[r, np.isfinite(sig[r, :])],
                                                 dtype='float64'))

        if Plot:
            fig, ax = mpl.pylab.subplots(figsize=(7, 5), nrows=1, ncols=1)
            ax.plot(rho, sig/1e19, 'ok', alpha=0.5, mec='white')
            ax.plot(xOut, yOut/1e19,
                    'sc', ms=10, mec='white')
            ax.errorbar(xOut, yOut/1e19,
                        yerr=eOut/1e19, fmt='none', ecolor='c')
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel(r'n$_e [10^{19}$m$^{-3}]$')
            ax.set_title(r'Shot %5i' % self.shot +
                         r' $\Delta$t = %3.2f' % trange[0] +
                         '- %3.2f' % trange[1])
        

        return xOut[np.argsort(xOut)], yOut[np.argsort(xOut)], eOut[np.argsort(xOut)]

    def PlotTeProfile(self, Type='OuterTarget', trange=[3, 3.1],
                      interelm=False, elm=False,
                      Plot=True, **kwargs):
        """
        Create the appropriate plot of the Temperature profile in the
        desired time range for the desired set of discharges
        
        Parameters
        ----------
        Type : 'str'. Possible values are 'OuterTarget',  'OuterLimiter',
            'InnerTarget', 'InnerLimiter'
        
        trange : 2D array indicating the time range for the profiles

        interelm : Boolean,  default is no. If True it create an interELM mask
               based on a smoothed 
        
        """

        if Type == 'OuterTarget':
            sig = self.OuterTargetTe
            rho = self.OuterTargetRhop
        elif Type == 'InnerTarget':
            sig = self.InnerTargetTe
            rho = self.InnerTargetRhop
        elif Type == 'InnerLimiter':
            sig = self.InnerLimiterTe
            rho = self.InnerLimiterRhop
        elif Type == 'OuterLimiter':
            sig = self.OuterLimiterTe
            rho = self.OuterLimiterRhop
        else:
            'assuming outer Target'
            sig = self.OuterTargetTe
            rho = self.OuterTargetRhop

        _idx = np.where((self.time >= trange[0]) & 
                        (self.time <= trange[1]))[0]
        t = self.time[_idx]
        sig = sig[:, _idx]
        rho = rho[:, _idx]

        if interelm:
            self._maskElm(trange=trange, **kwargs)
            t = t[self._interElm]
            sig = sig[:, self._interElm]
            rho = rho[:, self._interElm]
        if elm:
            self._maskElm(trange=trange, **kwargs)
            t = t[self._Elm]
            sig = sig[:, self._Elm]
            rho = rho[:, self._Elm]

        xOut = np.array([])
        yOut = np.array([])
        eOut = np.array([])
        for r in range(rho.shape[0]):
            if np.count_nonzero(np.isfinite(sig[r, :])) != 0:
                xOut = np.append(xOut, np.mean(rho[r, np.isfinite(sig[r, :])]))
                yOut = np.append(yOut, np.mean(sig[r, np.isfinite(sig[r, :])]))
                eOut = np.append(eOut, np.std(sig[r, np.isfinite(sig[r, :])],
                                                 dtype='float64'))

        if Plot:
            fig, ax = mpl.pylab.subplots(figsize=(7, 5), nrows=1, ncols=1)
            ax.plot(rho, sig, 'ok', alpha=0.5, mec='white')
            ax.plot(xOut, yOut,
                    'sc', ms=10, mec='white')
            ax.errorbar(xOut, yOut,
                        yerr=eOut, fmt='none', ecolor='c')
            ax.set_xlabel(r'$\rho$')
            ax.set_ylabel(r'T$_e$ [eV]')
            ax.set_title(r'Shot %5i' % self.shot +
                         r' $\Delta$t = %3.2f' % trange[0] +
                         '- %3.2f' % trange[1])

        return xOut[np.argsort(xOut)], yOut[np.argsort(xOut)], eOut[np.argsort(xOut)]


    def _maskElm(self, usedda=False, threshold=3000, trange=[2, 3],
                 check=False):
        """
        Provide an appropriate mask where we identify
        both the ELM and inter-ELM regime

        Parameters
        ----------
        usedda : :obj: `bool`
            Boolean, if True use the default ELM
            diagnostic ELM in the shotfile

        threshold : :obj: `float`
            If we choose to detect as threshold in the
            SOL current then this is the threshold chosen
        Returns
        -------
        None

        Attributes
        ----------
        Define the class hidden attributes
        self._elm
        self._interelm
        which are the indexes of the ELM and inter
        ELM intervals
        """

        if usedda:
            print("Using ELM dda")
            ELM = dd.shotfile("ELM", self.shot, experiment='AUGD')
            elmd = ELM("t_endELM", tBegin=trange[0], tEnd=trange[1])
            # limit to the ELM included in the trange
            _idx = np.where((elmd.time>= trange[0]) & (elmd.time <= trange[1]))[0]
            self.tBegElm = eldm.time[_idx]
            self.tEndElm = elmd.data[_idx]
            ELM.close()
        else:
            print("Using IpolSolI")
            Mac = dd.shotfile("MAC", self.shot, experiment='AUGD')
            Ipol = Mac('Ipolsoli')
            _idx = np.where(((Ipol.time >= trange[0]) & (Ipol.time <= trange[1])))[0]
            # now create an appropriate savgolfile
            IpolS = savgol_filter(Ipol.data[_idx], 501, 3)
            IpolT = Ipol.time[_idx]
            # on these we choose a threshold
            # which can be set as also set as keyword
            window, _a, _b, _c = identify_bursts2(IpolS, threshold)
            # now determine the tmin-tmax of all the identified ELMS
            _idx, _idy = zip(*window)
            self.tBegElm = IpolT[np.asarray(_idx)]
            self.tEndElm = IpolT[np.asarray(_idy)]

        # and now set the mask
        _dummyTime = self.time[np.where((self.time >= trange[0]) &
                                        (self.time <= trange[1]))[0]]

        self._interElm = []
        self._Elm=[]
        for i in range(self.tBegElm.size):
            _a = np.where((_dummyTime >= self.tBegElm[i]) &
                          (_dummyTime <= self.tEndElm[i]))[0]
            self._Elm.append(_a[:])
            try:
                _a = np.where((_dummyTime >= self.tEndElm[i]) &
                              (_dummyTime <= self.tBegElm[i+1]))[0]
                self._interElm.append(_a[:])
            except:
                pass
                
        self._interElm = np.concatenate(np.asarray(self._interElm))
        self._Elm = np.concatenate(np.asarray(self._Elm))

        if check:
            fig, ax = mpl.pylab.subplots(nrows=1, ncols=1, figsize=(6, 4))
            fig.subplots_adjust(bottom=0.15, left=0.15)
            ax.plot(IpolT, IpolS, color='#1f77b4')
            ax.set_xlabel(r't[s]')
            ax.set_ylabel(r'Ipol SOL I')
            ax.axhline(threshold, ls='--', color='#d62728')
            for _ti, _te in zip(self.tBegElm, self.tEndElm):
                ax.axvline(_ti, ls='--', color='#ff7f0e')
                ax.axvline(_te, ls='--', color='#ff7f0e')

    def computeLambda(self, Type='OuterTarget', trange=[3, 3.1],
                      Plot=False, **kwargs):
        """
        Compute the Normalized Divertor Collisionality computing the parallel
        connection length through the field-line tracing code and then averaging
        the profiles of density and temperature. The Lambda can be computed for the
        inner/outer divertor in the SOL (not in the private flux region)

        Parameters
        ----------
        Type :obj: `string`
            Possible values are 'OuterTarget',
            'InnerTarget'

        trange :obj: 'ndarray'
            2D array indicating the time minium and maximum

        interelm : Boolean,  default is no. If True it create an interELM mask
           based on a smoothed
        """

        rho, Lpar = self._computeLpar(trange=trange)

        # now we compute the profiles of density and temperature at the divertor
        rhoEn, en, errEn = self.PlotEnProfile(trange=trange, Plot=False, **kwargs)
        rhoTe, Te, errTe = self.PlotTeProfile(trange=trange, Plot=False, **kwargs)
        # we now consider a spline interpolation taking into account the error
        sEn = interp1d(rhoEn, en, fill_value='extrapolate')
        sTe = interp1d(rhoTe, Te, fill_value='extrapolate')
    
        # compute the Ion Sound Speed
        Cs = np.sqrt(2*constants.e*sTe(rho)/(constants.proton_mass))
        nuEi = 5e-11*sEn(rho)/(sTe(rho)**1.5)
        Lambda = (nuEi * Lpar * constants.electron_mass /
                  (constants.proton_mass*Cs))

        # we now create the appropriate plot
        # which include both the field lines in 2D and 3D
        # the profiles of density and temperature and
        # the computed Lambda
        if Plot:
            fig = mpl.pyplot.figure(figsize=(12, 10))
            fig.subplots_adjust(wspace=0.3)
            # 2D equilibria and field lines
            ax1=fig.add_subplot(221, aspect='equal')
            for key in self.rg.iterkeys():
                ax1.plot(self.rg[key], self.zg[key], 'k-', lw=2)
            ax1.contour(self.R, self.Z, self.psiN, [1], linestyle='--', color='k')
            for line in fieldLines:
                ax1.plot(line.R, line.Z, '--', lw=0.7)
            for line in fieldLinesZ:
                ax1.plot(line.R, line.Z, '-', lw=1.2)

            ax1.set_xlabel(r'R')
            ax1.set_ylabel(r'Z')
            ax1.set_title(r'# %5i' % self.shot + ' t = %3.2f' % self._time)
            # 3D plot
            ax2 = fig.add_subplot(222, projection='3d', aspect='equal')
            for line in fieldLines:
                ax2.plot(line.X,line.Y,line.Z,'-', lw=0.7)

            ax2.set_zlim3d(-2.0,2.0)
            ax2.set_xlim3d(-2.0,2.0)
            ax2.set_ylim3d(-2.0,2.0)            
            # profile of density and temperature with relative
            # interpolation
            ax3 = fig.add_subplot(223)
            ax3.plot(rhoEn, en/1e19, 'ko', markersize=10)
            ax3.errorbar(rhoEn, en/1e19, yerr=errEn/1e19, fmt='none', ecolor='k')
            ax3.plot(rho, sEn(rho)/1e19, '--', color='grey', lw=2)
            ax3.set_xlabel(r'$\rho$')
            ax3.set_ylabel(r'n$_e [10^{19}$m$^{-3}]$')
            ax3.set_ylim([0, np.max((en+errEn)/1e19)])
            ax4=ax3.twinx()
            ax4.plot(rhoTe, Te, 'ro', markersize=10)
            ax4.errorbar(rhoTe, Te, yerr=errTe, fmt='none', ecolor='r')
            ax4.plot(rho, sTe(rho), '--', color='orange')
            ax4.spines['right'].set_color('red')
            ax4.tick_params(axis='y', colors='red')
            ax4.set_ylabel(r'T$_e$ [eV]', color='red')
            ax4.set_ylim([0, np.max((Te+errTe))])
            # and now the parallel connection length and the Lambda
            ax5 = fig.add_subplot(224)
            ax5.plot(rho[Lambda>0], Lambda[Lambda>0], 'ko--')
            ax5.set_ylabel(r'$\Lambda_{div}$')
            ax5.set_xlabel(r'$\rho$')
            ax5.set_yscale('log')
            ax6 = ax5.twinx()
            ax6.plot(rho[Lambda>0], Lpar[Lambda>0], 'rp--')
            ax6.spines['right'].set_color('red')
            ax6.tick_params(axis='y', colors='red')
            ax6.set_ylabel(r'L$_{\parallel}$ [m]', color='red')

        rho = rho[Lambda >0]
        Lambda=Lambda[Lambda>0]

        return rho[~np.isnan(Lambda)], Lambda[~np.isnan(Lambda)]


    def dump_geqdsk(self,filename="equilibrium.g"):
        from geqdsk import Geqdsk
        if self._loaded:
            print("Writing gfile: "+filename)
            outgf = Geqdsk()
            outgf.set('nw',self.nr)
            outgf.set('nh',self.nz)
            outgf.set('rleft',np.min(self.R))
            outgf.set('rdim',np.max(self.R) - np.min(self.R))
            outgf.set('zdim',np.max(self.Z) - np.min(self.Z))
            outgf.set('zmid',0.5*(np.max(self.Z) + np.min(self.Z)))
            outgf.set('psirz',self.psi[:])
            outgf.set('simag',self.psi_axis)
            outgf.set('sibry',self.psi_bnd)
            outgf.set('current',self.sigBp)
            outgf.set('rmaxis',self.axis.r)
            outgf.set('zmaxis',self.axis.z)
            outgf.set('xdum',0.00000)
            outgf.set('pres',np.zeros(self.nr))
            outgf.set('pprime',np.zeros(self.nr))
            outgf.set('ffprime',np.zeros(self.nr))
            outgf.set('qpsi',np.zeros(self.nr))
            if self.fpol is not None:
                outgf.set('fpol',self.fpol[:])
            else:
                outgf.set('fpol',np.zeros(self.nr) + self.Rcent*self.Btcent)                
            outgf.set('bcentr',self.Btcent)
            outgf.set('rcentr',self.Rcent)
            outgf.set('rlim',self.wall['R'])
            outgf.set('zlim',self.wall['Z'])
            boundary = self.get_fluxsurface(1.0)
            outgf.set('rbbbs',boundary.R)
            outgf.set('zbbbs',boundary.Z)
            outgf.set('nbbbs',len(list(boundary.R)))    
            outgf.dump(filename)
        else:
            print("WARNING: No equilibrium loaded, cannot write gfile")

    def _computeLpar(self, trange=[3, 3.1], Plot=False, Type='OuterTarget'):

        """
        Method for the computation of the parallel connection length
        using the cyFieldLine Tracer Code. Presently it is
        implemented only for the lenght in the Divertor region

        Parameters
        ----------
        Type :obj: `string`
            The only Possible value so far is 'OuterTarget'

        trange :obj: 'ndarray'
            2D array indicating the time minium and maximum

        """


        t0 = (trange[0]+trange[1])/2.
        # reload the equilibrium at the appropriate time instant
        self._setTimeEq(time=t0)
        # save into a dummy gfile to be loaded by the Tracer
        self.dump_geqdsk(filename='tmp.g')
        # We need to reload the equilibrium since there is no way
        # for the moment to pass an istance of equilibrium to the
        # tracer CLASS
        self.Tracer = get_fieldline_tracer('RK4', gfile='tmp.g',
                                           interp='quintic', rev_bt=True)
        if Type is 'OuterTarget':
            # height of magnetic axis
            zMAxis = self.axis.__dict__['z']
            # height of Xpoint
            zXPoint = self.xpoints['xpl'].__dict__['z']
            rXPoint = self.xpoints['xpl'].__dict__['r']
            # now determine at the height of the zAxis the R of the LCFS
            Boundary = self.get_fluxsurface(1.0)
            RLcfs = Boundary.R
            ZLcfs = Boundary.Z
            # only the part greater then xaxis
            ZLcfs = ZLcfs[RLcfs > self.axis.__dict__['r']]
            RLcfs = RLcfs[RLcfs > self.axis.__dict__['r']]
            Rout = RLcfs[np.argmin(np.abs(ZLcfs[~np.isnan(ZLcfs)]-zMAxis))]
            rmin = np.linspace(Rout+0.001, 2.19, num=30)            
            # this is the R-Rsep
            rMid = rmin-Rout
            # get the corrisponding Rho
            rho = self.Eqm.rz2rho(rmin, np.repeat(zMAxis, rmin.size), self._time,
                                  extrapolate=True).squeeze()

            # compute the field lines
            fieldLines = [self.Tracer.trace(r, zMAxis, mxstep=100000, ds=1e-2, tor_lim=20.0*np.pi) for r in rmin]
            # compute the parallel connection length from the divertor plate to the X-point
            fieldLinesZ = [line.filter(['R', 'Z'],
                                       [[rXPoint, 2], [-10, zXPoint]]) for line in fieldLines]
            Lpar =np.array([])
            for line in fieldLinesZ:
                try:
                    _dummy = np.abs(line.S[0] - line.S[-1])
                except:
                    _dummy = np.nan
                Lpar = np.append(Lpar, _dummy)

            # we then remove the temporary created G-file
            os.remove('tmp.g')
            return rho, Lpar
        else:
            print('Only outer divertor implemented so far')
            pass

    def get_fluxsurface(self,psiN,Rref=1.5,Zref=0.0):
        """
        Get R,Z coordinates of a flux surface at psiN
        """
        R, Z = scipy.meshgrid(self.R, self.Z)
        c = cntr.Cntr(R, Z, self.psiN[:])
        nlist = c.trace(psiN)
        segs=nlist[: len(nlist)//2]
        if len(segs) > 1:
            if len(segs[1]) > 20:
                R = segs[1].transpose()[0]
                Z = segs[1].transpose()[1]
            else:
                R = segs[0].transpose()[0]
                Z = segs[0].transpose()[1]
        else:
            R = segs[0].transpose()[0]
            Z = segs[0].transpose()[1]

        return fluxSurface(R = R, Z = Z)
#         try:
#             import matplotlib.pyplot as plt
#         except:
#             print("ERROR: matplotlib required for flux surface construction, returning")
#             return
            
#         if type(psiN) is list:
#             surfaces = []
#             for psiNval in psiN:
#                 psi_cont = plt.contour(self.R,self.Z,(self.psi - self.psi_axis)/(self.psi_bnd-self.psi_axis),levels=[0,psiN],alpha=0.0)
#                 paths = psi_cont.collections[1].get_paths()
#                 #Figure out which path to use
#                 i = 0
#                 old_dist = 100.0
#                 for path in paths:
#                     dist = np.min(((path.vertices[:,0] - Rref)**2.0 + (path.vertices[:,1] - Zref)**2.0)**0.5)
#                     if  dist < old_dist:
#                         true_path = path
#                     old_dist = dist 
                


#                 R,Z =  true_path.vertices[:,0],true_path.vertices[:,1]
#                 surfaces.append(fluxSurface(R = R, Z = Z))
#             return surfaces
#         else:
#             psi_cont = plt.contour(self.R,self.Z,(self.psi - self.psi_axis)/(self.psi_bnd-self.psi_axis),levels=[0,psiN],alpha=0.0)
#             paths = psi_cont.collections[1].get_paths()
#             old_dist = 100.0
#             for path in paths:
#                 dist = np.min(((path.vertices[:,0] - Rref)**2.0 + (path.vertices[:,1] - Zref)**2.0)**0.5)
#                 if  dist < old_dist:
#                     true_path = path
#                 old_dist = dist
#             R,Z =  true_path.vertices[:,0],true_path.vertices[:,1]
#             plt.clf()  
#             return fluxSurface(R = R, Z = Z)

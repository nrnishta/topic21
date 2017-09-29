"""
Global class for the analysis of the shot for Topic 21
experiment based on the structure, lambda and profile analysis
"""
from __future__ import print_function
import MDSplus as mds
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import io
from scipy import stats
from scipy.signal import detrend as Detrend
from scipy import constants
import eqtools
from langmuir import LP
import tcv.diag.frp as frp
import pycwt
import xarray as xray
import bottleneck
import quickspikes as qs


class Turbo(object):
    """
    Python class for the evaluation of
    different quantities and analysis for
    the TCV15-2.2-3 Filamentary studies.
    It does have a lot of dependences
    Inputs:
    ------
       shot = shot number
       gas = 'D2' (Default), 'H', 'He' according to the gas
              used
       Lp = 'Div' (Default), 'All' it states if the Lparallel
             used in the computation is the Lp from LFS target to
             X-point (the )

    Dependencies:
    ------------
    On lac all the dependencies can be solved by adding the
    following to your .bashrc
    export PATH="/home/vianello/NoTivoli/anaconda2/bin":$PATH
    and adding the following whenever you start python
    >>> import sys
    >>> sys.path.append('/home/vianello/pythonlib/submodules/tcvpy')

       eqtools
       MDSplus
       FastRP from tcvpy diagnostic
       pandas
       bottleneck
       xray
       langmuir
       scipy
       quickspikes
    """

    def __init__(self, shot, gas='D2', Lp='Div'):
        self.shot = shot
        self.Lp=Lp
        # globally define the gas
        self.gas = gas
        if gas == 'D2':
            self.Z = 1
            self.mu = 2
        elif gas == 'H':
            self.Z = 1
            self.mu = 1
        elif gas == 'He':
            self.Z = 4
            self.mu = 4
        else:
            print('Gas not found, assuming Deuterium')
            self.gas = 'D2'
            self.Z = 1
            self.mu = 2
        # equilibrium quantities
        self._eq = eqtools.TCVLIUQETree(self.shot)
        # this is the iP in the time bases of LIUQE
        self._tLiuqe = self._eq.getTimeBase()
        self.Ip = self._eq.getIpMeas()
        self.BtSign = np.sign(self._eq.getBtVac().mean())
        # this is for different quantities
        self._tree = mds.Tree('tcv_shot', shot)
        # average density node
        self._enavgNode = self._tree.getNode(r'\results::fir:n_average')
        # now define the tree where the probe data are saved
        self._filament = mds.Tree('tcv_topic21', shot)
#        self._doubleDir = '/home/tsui/idl/library/data/double/dpm' + \
#                          str(int(self.shot)) + '_'
#        self._connection = '/home/vianello/work/tcv15.2.2.3/data/connection/'
        # this can be done one single time and then accordingly
        # to analysis region we choose the appropriate timing
        try:
            self.Target = LP(self.shot)
        except:
            print('Langmuir probe data not found')
            self.Target = None


    def blob(self, plunge=1, rrsep=None, rho=None,
             gas='D2', Z=1, iwin=125, thr=3, normalize=True,
             detrend=False, outlier=False, outThr=150):
        """

        Given the plunge stated and the region defined in
        R-Rsep or rho it computes various quantities for the
        blob, including the auto-correlation time,
        the size as FWHM of the iSat, the radial velocity
        as computed from Epoloidal, and also the perpendicular
        velocity as computed according to propagation of
        floating potential corresponding structure

        Parameters:
        -----------
        plunge
           Plunge number, default is 1
        rrsep
           2D array indicating the minimum and maximum
           for the computation of the CAS in R-R_sep
        rho
           2D array indicating the minimum and maximum
           for the computation of the CAS in rho poloidal
        gas
           string. Gas used, default is Deuterium
        Z
           int. Effective charge
        iwin
           int. Half of the window for Conditional Average Sampling
           in number of points
        normalize
           Boolean. If True (default) normalize to local std
           each of the window before average
        detrend
           Boolean. If False (default) does not perform a linear
           detrending on each window before averaging
        outlier
           Boolean. If False (default) does not eliminate the
           outliers to signals (specified by outThr)
        outThr
           Threshold for outliers
          
        Returns
        -------
        blob
           Xarray dataset containing all the information
           needed for the computation of filament properties including
           Lambda and Theta

        """
        # in case the quantities are not loaded let load it
        try:
            if self.plunge != plunge:
                self._loadProbe(plunge=plunge, outlier=outlier, outThr=outThr)
                self._loadProfile()
            else:
                print('Already loaded')
        except:
            self._loadProbe(plunge=plunge, outlier=outlier, outThr=outThr)
            self._loadProfile()
            
        # now check if we have define the distances in
        # rho or rrsep
        if (rho is None) and (rrsep is not None):
            _drmn = rrsep[0]
            _drmx = rrsep[1]
        elif (rrsep is None) and (rho is not None):
            _drmn = (self._eq.psinorm2rmid(np.power(rho[0], 2),
                                           self.iS.time.mean().item()) -
                     self._eq.getRmidOutSpline()(self.iS.time.mean().item()))
            _drmx = (self._eq.psinorm2rmid(np.power(rho[1], 2),
                                           self.iS.time.mean().item()) -
                     self._eq.getRmidOutSpline()(self.iS.time.mean().item()))
        else:
            print('You must specify region in the profile')

        tmin = self.RRsep[
            ((self.RRsep.values >= _drmn) &
             (self.RRsep.values <= _drmx))].time.min().item()
        tmax = self.RRsep[ 
            ((self.RRsep.values >= _drmn) &
             (self.RRsep.values <= _drmx))].time.max().item()
        # we can also compute the absolute radial position
        # which can be used to save the Btot
        Ravg = self.R.where(((self.R.time >= tmin) &
                             (self.R.time <= tmax))).mean().item()
        Btot = self._eq.rz2B(Ravg, 0, (tmax+tmin)/2)
        # limit to this time interval
        sIs = self.iS[((self.iS.time >= tmin) &
                       (self.iS.time <= tmax))].values
        sVf = self.vF[:, ((self.vF.time >= tmin) &
                          (self.vF.time <= tmax))].values
        
        sEp = self.Epol[((self.Epol.time >= tmin) &
                         (self.Epol.time <= tmax))].values
        sEr = self.Erad[((self.Erad.time >= tmin) &
                         (self.Erad.time <= tmax))].values
        _n = np.min([sIs.size, sVf.shape[1], sEp.size, sEr.size])
        sigIn = np.vstack((sIs[:_n], sEp[:_n], sEr[:_n], sVf[:,:_n]))
        self.dt = (self.iS.time.max().item()-
                   self.iS.time.min().item()) / (self.iS.size-1)
        # this is the determination of the corresponding 
        cs, tau, err, amp = self.casMultiple(
            sigIn, thr=thr, normalize=normalize, detrend=detrend,
            iwin=iwin)
        # the output will be an xray DataArray containing
        # the results of CAS plus additional informations
        names = np.append(['Is', 'Epol', 'Erad'], self.vF.Probe.values)
        data = xray.DataArray(cs, coords={'t':tau, 'sig':names})
        # add the errorss
        data.attrs['err'] = err
        # position, time
        data.attrs['R'] = Ravg
        data.attrs['tmin'] = tmin
        data.attrs['tmax'] = tmax
        data.attrs['RrsepMin'] = _drmn
        data.attrs['RrsepMax'] = _drmx
        # start adding the interesting quantities
        # the FWHM
        delta, errDelta = self._computeDeltaT(tau, cs[0, :],
                                              err[0, :])
        data.attrs['FWHM'] = delta
        data.attrs['FWHMerr'] = errDelta
        # the vr from the poloidal electric field
        out = self._computeExB(data)
        data.attrs['vrExB'] = out['Epol']/Btot
        data.attrs['vrExBerr'] = out['EpolErr']/Btot
        # the fluctuating poloidal velocity
        data.attrs['vpExB'] = out['Er']/Btot
        data.attrs['vpExBerr'] = out['ErErr']/Btot
        # now compute the v_perp according using the propagation
        # along theta and r from floating potential pins. Consider
        # that VFM_1 and VFR_1 are distant in the Z direction of 
        # less than 1mm
        out = self._computeVperp(data)
        data.attrs['vperp'] = out['vperp']
        data.attrs['vAutoP'] = out['vpol']
        data.attrs['vAutoR'] = out['vrad']
        data.attrs['vAutoPErr'] = out['vpolErr']
        # autocorrelation time
        c = np.correlate(sIs, sIs, mode='full')
        c /= c.max()
        lag = np.arange(c.size)-c.size/2
        data.attrs['T_ac'] = 2*np.abs(
            lag[np.argmin(np.abs(c-1/2.))])*self.dt
        # compute the Ion sound gyroradius in this zone
        # we need the standard deviation of the te
        teStd = self.profileTe[((self.profileTe.rho >= _drmn) &
                                (self.profileTe.rho <= _drmx))].std().item()
        CsDict = self._computeRhos(Ravg, np.asarray([_drmn, _drmx]),
                                   (tmax+tmin)/2)
        data.attrs['rhos'] = CsDict['rhos']
        data.attrs['drhos'] = CsDict['drhos']
        data.attrs['Cs'] = CsDict['Cs']
        # we also compute the corresponding Lambda and Delta
        # so that we save all the information on one single
        # xray dataArray
        # Lambda Div  Computation
        Lambda, Err = self._computeLambda(_drmn, _drmx, tmin, tmax, Lp='Div')
        data.attrs['LambdaDiv'] = Lambda
        data.attrs['LambdaDivErr'] = Err
        # All Lp
        Lambda, Err = self._computeLambda(_drmn, _drmx, tmin, tmax, Lp='Tot')
        data.attrs['Lambda'] = Lambda
        data.attrs['LambdaErr'] = Err
        # Theta computation again require to
        # Divertor
        Theta, Err = self._computeTheta(data, Lp='Div')
        data.attrs['ThetaDiv'] = Theta
        data.attrs['ThetaDivErr'] = Err
        # Total
        Theta, Err = self._computeTheta(data, Lp='Tot')
        data.attrs['Theta'] = Theta
        data.attrs['ThetaErr'] = Err

        # add also the results of the conditional average
        # which is useful
        data.attrs['CAS'] = cs
        data.attrs['CAStau'] = tau
        data.attrs['CASerr'] = err
        data.attrs['CASamp'] = amp

        return data

    def _getNames(self, plunge=1):
        """
        Get the probes name for floating potential
        and ion saturation current. They are defined as
        attributes to the class
        :param plunge:
        :return: None
        """
        # determine the available probe names
        _string = 'getnci(getnci(\\TOP.DIAGZ.MEASUREMENTS.UCSDFP,"MEMBER_NIDS"),"NODE_NAME")'
        _nameProbe = np.core.defchararray.strip(mds.Data.compile(_string).evaluate().data())
        # determine the names corresponding to the strokes
        _nameProbe = _nameProbe[
            np.asarray([int(n[-1:]) for n in _nameProbe]) == plunge]
        self.vfNames = _nameProbe[np.asarray([n[:2] for n in _nameProbe]) == 'VF']
        isNames = _nameProbe[
            np.asarray([n[:2] for n in _nameProbe]) == 'IS']
        # we need to eliminate the IS of the single probe
        self.isNames = isNames[
            (np.asarray(map(lambda n:n[3],isNames)) == '_')]

    def _loadProbe(self, plunge=1, outlier=False, outThr=150):
        """

        Load the signal from the Probe and ensure
        we are using exclusively the insertion part of the
        plunge. If it has not been already loaded the profile
        it loads also the profile

        """
        self.plunge = plunge
        # now load the data and save them in the appropriate xarray
        self._getNames(plunge=plunge)
        # now we load the floating potential and save them in an xarray
        vF = []
        for name in self.vfNames:
            vF.append(self._tree.getNode(r'\FP'+name).data())
        # convert in a numpy array
        vF = np.asarray(vF)
        # get the time basis
        time = self._tree.getNode(r'\FP'+self.vfNames[0]).getDimensionAt().data()
        # we need to build an appropriate time basis since it has not a
        # constant time step
        time = np.linspace(time.min(),time.max(),time.size,dtype='float64')
        vF = xray.DataArray(vF,coords=[self.vfNames,time],dims=['Probe','time'])
        # repeat for the ion saturation current
        if self.isNames.size == 1:
            iS = self._tree.getNode(r'\FP'+self.isNames[0]).data()
            iS = xray.DataArray(iS,coords=[time],dims=['time'])
        else:
            iS = []
            for name in self.isNames:
                iS.append(self._tree.getNode(r'\FP'+name).data())
            # convert in a numpy array
            iS = np.asarray(iS)
            # save an xarray dataset
            iS = xray.DataArray(iS,coords=[self.isNames,time],dims=['Probe','time'])
        # this is upstream remapped I load the node not the data
        RRsep = self._filament.getNode(r'\FP_%1i' % plunge + 'PL_RRSEPT')
        # this is in Rhopoloidal
        Rhop = self._filament.getNode(r'\FP_%1i' % plunge + 'PL_RHOT')
        # limit to first insertion of the probe
        Rtime = Rhop.getDimensionAt().data()[:RRsep.data().argmin()]
        Rhop = Rhop.data()[:RRsep.data().argmin()]
        RRsep = RRsep.data()[:RRsep.data().argmin()]
        # limit to the timing where we insert the
        ii = np.where((iS.time >= Rtime.min()) &
              (iS.time <= Rtime.max()))[0]
        if 1 == self.isNames.size:
            self.iS = iS[ii]
        else:
            self.iS = iS[:,ii]
        self.vF = vF[:, ii]
        # now check if the number of size between position and signal
        # are the same otherwise univariate interpolate the position
        if Rtime.size != self.iS.time.size:
            S = UnivariateSpline(Rtime,Rhop,s=0)
            self.Rhop = S(self.iS.time.values)
            S = UnivariateSpline(Rtime,RRsep,s=0)
            self.RRsep = S(self.iS.time.values)
            self.Rtime = self.iS.time.values
        else:
            self.Rhop = Rhop
            self.RRsep = RRsep
            self.Rtime = Rtime
        # in case we set outlier we eliminate outlier with NaN
        if outlier:
            det = qs.detector(outThr, 40)
            for _probe in range(self.vF.shape[0]):
                times = det.send(np.abs(
                    self.vF[_probe, :]).values.astype('float64'))
                if len(times) != 0:
                    for t in times:
                        if t+40 < self.vF.shape[1]:
                            self.vF[_probe, t-40:t+40] = np.nan
                        else:
                            self.vF[_probe, t-40:-1 ] = np.nan
                        # perform a median filter to get rid of possible
#        self.R = R.rolling(time=20).mean()[:R.argmin().item()]
        self.Epol = (vF.sel(Probe='VFT_' + str(int(self.plunge))) -
                     vF.sel(Probe='VFM_' + str(int(self.plunge))))/4e-3
        self.Erad = (vF.sel(Probe='VFM_' + str(int(self.plunge))) -
                     vF.sel(Probe='VFR1_' + str(int(self.plunge))))/1.57e-3

    def _loadProfile(self, npoint=20, plunge=1):

        """
        Load the profile, compute the profile ensuring we
        are dealing just with the insertion, compute the
        lambda decay length
        """

        if self.Rhop is None:
            self._loadProbe(plunge=plunge)

        # load the time basis of the profile
        _data = self._filament.getNode(
            r'\FP_%1i' % plunge + 'PL_EN').data()
        _rho = self._filament.getNode(
            r'\FP_%1i' % plunge + 'PL_RRSEP').data()
        _err = self._filament.getNode(
            r'\FP_%1i' % plunge + 'PL_ENERR').data()
        _time = self._filament.getNode(
            r'\FP_%1i' % plunge + 'PL_EN').getDimensionAt().data()
        _ii = np.where(_time <= self.Rtime.max())[0]

        # in case the number of point is already low we use
        # directly the part saved in the doubleP
        if _ii.size < npoint:
            self.profileEn = xray.DataArray(
                _data[_ii]/1e19,
                coords={'rho': _rho[_ii]})
            self.profileEn.attrs['err'] = _err[_ii]/1e19
        else:
        # density profile and corresponding spline
            self.profileEn = frp.FastRP._getprofileR(
                _rho[_ii],
                _data[_ii]/1e19,
                npoint=npoint)
        self.splineEn = UnivariateSpline(self.profileEn.rho.values,
                                         self.profileEn.values,
                                         w=1./self.profileEn.err,
                                         ext=0)
        # compute also the profile of Te
        _data = self._filament.getNode(
            r'\FP_%1i' % plunge + 'PL_TE').data()
        try:
            _err = self._filament.getNode(
                r'\FP_%1i' % plunge + 'PL_TEERR').data()
        except:
            _err = None
        if _ii.size < npoint:
            self.profileTe = xray.DataArray(
                _data[_ii],
                coords={'rho': _rho[_ii]})
            if _err:
                self.profileEn.attrs['err'] = _err[_ii]
        else:
            self.profileTe = frp.FastRP._getprofileR(
                _rho[_ii],
                _data[_ii],
                npoint=npoint)
        self.splineTe = UnivariateSpline(self.profileTe.rho.values,
                                         self.profileTe.values,
                                         w=1./self.profileTe.err,
                                         ext=0)
        # compute the Efolding length
        self.rhoArray = np.linspace(_rho[_ii].min(),
                                    _rho[_ii].max(),
                                    100)
        self.Efolding = np.abs(self.splineEn(self.rhoArray) /
                               self.splineEn.derivative()(self.rhoArray))


    def _computeDeltaT(self, x, y, e):
        """
        useful to compute the timing and error on the Isat
        conditional average sample
        """
        _dummy = (y-y.min())
        spline = UnivariateSpline(x, _dummy-_dummy.max()/2, s=0)
        if spline.roots().size > 2:
            a = np.sort(spline.roots())
            r1 = a[a<0][-1]
            r2 = a[a>0][0]
        else:
            r1, r2 = spline.roots()
        delta = (r2-r1)
        # now compute an estimate of the error
        _dummy = (y+e)-(y+e).min()
        spline = UnivariateSpline(x, _dummy-_dummy.max()/2, s=0)
        if spline.roots().size > 2:
            a = np.sort(spline.roots())
            r1 = a[a<0][-1]
            r2 = a[a>0][0]
        else:
            r1, r2 = spline.roots()
        deltaUp = (r2-r1)
        _dummy = (y-e)-(y-e).min()
        spline = UnivariateSpline(x, _dummy-_dummy.max()/2, s=0)
        if spline.roots().size > 2:
            a = np.sort(spline.roots())
            r1 = a[a<0][-1]
            r2 = a[a>0][0]
        else:
            r1, r2 = spline.roots()
        deltaDown = (r2-r1) 
        err = np.asarray([np.abs(delta),
                          np.abs(deltaUp),
                          np.abs(deltaDown)]).std()
        return delta, err

    def _computeRhos(self, ravg, rrsep, t, gas='D2', Z=1):
        """
        Given the absolute value in R and the timing
        it compute the ion sound gyroradius

        """
        # compute the total B field at the R radial location
        Gamma = 5./3.
        Btot = self._eq.rz2B(ravg, 0, t)
        Omega_i = 9.58e3/2.*Btot*self.Z*1e4/self.mu
        Te = self.profileTe[((self.profileTe.rho >= rrsep[0]) &
                             (self.profileTe.rho <= rrsep[1]))].mean().item()
        dte = self.profileTe[((self.profileTe.rho >= rrsep[0]) &
                             (self.profileTe.rho <= rrsep[1]))].std().item()
        Cs = 9.79*1e3*np.sqrt(Te*Gamma*self.Z/self.mu)
        dCs = 9.79*1e3/(2.*np.sqrt(self.splineTe(np.mean(rrsep)))) * \
            np.sqrt(Gamma*self.Z/self.mu)*dte
        rhoS = Cs/Omega_i
        drhoS = dCs/Omega_i
        out = {'Cs': Cs,
               'Omega_i': Omega_i,
               'rhos': rhoS, 'drhos': drhoS}
        return out

    def _computeVperp(self, data):
        """
        Takes the xarray DataArray as results
        from the computation of CAS and compute the
        vperp assumed from propagation of floating
        potential structure
        """

        # for noisy signal we perform a moving average
        # with 3 points for better timing
        a = bottleneck.move_mean(
            data.sel(sig='VFM_' + str(int(self.plunge))), window=3)
        b = bottleneck.move_mean(
            data.sel(sig='VFT_' + str(int(self.plunge))), window=3)
        # there is the possibility of some nan_policy
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        # compute the cross correlation
        xcorA = np.correlate(a, b, mode='same')
        # normalize appropriately
        xcorA /= np.sqrt(np.dot(a, a)*np.dot(b, b))
        lagA = np.arange(xcorA.size)-xcorA.size/2
        vpA = (0.2433-0.0855)*constants.inch/(lagA[np.argmax(xcorA)]*self.dt)
        # repeat for another couple
        a = bottleneck.move_mean(
            data.sel(sig='VFB_' + str(int(self.plunge))), window=3)
        b = bottleneck.move_mean(
            data.sel(sig='VFT_' + str(int(self.plunge))), window=3)
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        xcorB = np.correlate(a, b, mode='same')
        xcorB /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagB = np.arange(xcorB.size)-xcorB.size/2
        vpB = (0.2433+0.1512)*constants.inch/(lagB[np.argmax(xcorB)]*self.dt)
        # repeat for the last couple
        a = bottleneck.move_mean(
            data.sel(sig='VFB_' + str(int(self.plunge))), window=3)
        b = bottleneck.move_mean(
            data.sel(sig='VFM_' + str(int(self.plunge))), window=3)
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        # compute the cross correlation
        xcorC = np.correlate(a, b, mode='same')
        xcorC /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagC = np.arange(xcorC.size)-xcorC.size/2
        vpC = (0.0855+0.1512)*constants.inch/(lagC[np.argmax(xcorC)]*self.dt)
        deltaPoloidal = np.nanmean([vpA, vpB, vpC])
        deltaPoloidalStd = np.nanstd([vpA, vpB, vpC])

        a = bottleneck.move_mean(
            data.sel(sig='VFM_' + str(int(self.plunge))), window=3)
        b = bottleneck.move_mean(
            data.sel(sig='VFR1_' + str(int(self.plunge))), window=3)
        deltaTimeR = np.abs(
            np.nanargmin(a) - np.nanargmin(b))
        deltaRadial = 4e-3/(deltaTimeR*self.dt)
        vperp = np.sqrt(deltaPoloidal ** 2 + deltaRadial ** 2)
        out = {'vperp': vperp, 'vpol': deltaPoloidal,
               'vrad': deltaRadial, 'vpolErr': deltaPoloidalStd}
        return out

    def _computeLambda(self, _drmn, _drmx, tmin, tmax, Lp='Div'):
        """
        Computation of Divertor Lambda given the upstream
        R-Rsep range and the minimum and maximum times (
        Actually the time chosen is [tmin-0.06, tmax+0.06]).
        The output are the Lambda and corresponding error
        """

        try:
            self.cL = io.loadmat(
                '../data/connection/' +
                self.database['LpString'][
                    ((self.database['shot'] == self.shot) &
                     (self.database['plunge'] == self.plunge))].values[0])
            if Lp == 'Div':
                _Lambda = self.Target.Lambda(self.cL['x'].ravel(),
                                             self.cL['div'].ravel(),
                                             gas=self.gas,
                                             trange=[tmin-0.06, tmax+0.06])
            else:
                _Lambda = self.Target.Lambda(self.cL['x'].ravel(),
                                             self.cL['lfs'].ravel(),
                                             gas=self.gas,
                                             trange=[tmin-0.06, tmax+0.06])
            Lambda = _Lambda[((self.cL['x'].ravel() >= _drmn) &
                             (self.cL['x'].ravel() <= _drmx))].mean()
            LambdaErr = _Lambda[((self.cL['x'].ravel() >= _drmn) &
                                 (self.cL['x'].ravel() <= _drmx))].std()
        except:
            print('Lambda not computed')
            Lambda = np.nan
            LambdaErr = np.nan

        return Lambda, LambdaErr

    def _computeTheta(self, data, Lp='Div'):
        try:
            self.cL = io.loadmat(
                '../data/connection/' +
                self.database['LpString'][
                    ((self.database['shot'] == self.shot) &
                     (self.database['plunge'] == self.plunge))].values[0])
            if Lp == 'Div':
                Lpar = self.cL['div'].ravel()[
                    ((self.cL['x'].ravel() >= data.RrsepMin) &
                     (self.cL['x'].ravel() <= data.RrsepMax))].mean()
                dLpar = self.cL['div'].ravel()[
                    ((self.cL['x'].ravel() >= data.RrsepMin) &
                     (self.cL['x'].ravel() <= data.RrsepMax))].std()
            else:
                Lpar = self.cL['lfs'].ravel()[
                    ((self.cL['x'].ravel() >= data.RrsepMin) &
                     (self.cL['x'].ravel() <= data.RrsepMax))].mean()
                dLpar = self.cL['lfs'].ravel()[
                    ((self.cL['x'].ravel() >= data.RrsepMin) &
                     (self.cL['x'].ravel() <= data.RrsepMax))].std()
            Theta = ((data.FWHM * np.sqrt(
                data.vrExB**2 + data.vAutoP**2))**(5/2.) * np.sqrt(0.88)) / \
                (Lpar * data.rhos**2)
            dTheta = Theta * np.sqrt(
                (data.FWHMerr/data.FWHM)**2 +
                (data.vrExBerr/data.vrExB)**2 +
                (dLpar/Lpar)**2 + (data.drhos/data.rhos)**2)
        except:
            Theta = np.nan
            dTheta = np.nan
        return Theta, dTheta

    def casMultiple(self, inputS, **kwargs):
        """
        Conditional average sampling on multiple signals over
        the computed  structure location
        There are two keywords which can be used
        input:
        ------
            inputS = Array of signals to be analyzed (not the one already used)
                     in the form (#sign, #sample). The reference signal is
                     assumed to be the first one 
            iwin   = dimension of the window for the CAS in
                     number of points. Default is 125
            thr = Threshold.  If set it computes the CAS using the
                  given threshold. Default is 3 (assuming that the signal
                  is normalized)
            detrend = Boolean, default is True and in this case subtract a
                      linear trend in the chosen window
            normalize = Boolean. If set the reference signal is normalized
                      as (x-<x>)/rms(x) where the mean and the rms are
                      moving average and rms on a dtS
            dtS = time interval used for the computation of moving
                      average and rms. Default is 0.2 ms
            idx = Number of signal to be used as reference. Default is 0
        output:
        ------
            cs  = conditional sampled structure on all the signals
            tau = corresponding time basis
        """

        Shape = inputS.shape
        nSig = Shape[0]
        detrend = kwargs.get('detrend', True)
        iwin = kwargs.get('iwin',125)
        iwin *= 2
        thr = kwargs.get('thr', 3)
        normalize = kwargs.get('normalize', True)
        idx = kwargs.get('idx', 0)
        dtS = kwargs.get('dtS', 0.0002)
        _nPointS = int(np.round(dtS/self.dt))
        if normalize:
            signal = (inputS[idx, :] -
                      self._smooth(inputS[idx, :].ravel(),
                                   window_len= _nPointS)) / \
                      bottleneck.move_std(inputS[idx, :].ravel(),
                                          window=_nPointS)
        else:
            signal = inputS[idx, :]
        print('CAS using threshold method')
        maxima, allmax = self._threshold(signal, thr)
        maxima[0: iwin - 1] = 0
        maxima[-iwin:] = 0
        print('Number of structures mediated %3i' % maxima.sum())
        csTot = np.ones((nSig, 2 * iwin + 1, int(maxima.sum())))
        d_ev = np.asarray(np.where(maxima >= 1))
        ampTot = np.zeros((nSig, int(maxima.sum())))
        for i in range(d_ev.size):
            for n in range(nSig):
                if detrend:
                    # distinguish the cases where we have NaN as
                    # we can't use Detrend
                    if np.isnan(
                            np.sum(
                                inputS[n, d_ev[0][i] - iwin:
                                       d_ev[0][i] + iwin + 1])):
                        # check the first and last point of NaN and perform a
                        # a linear polynomial fit
                        _tmp = inputS[
                            n, d_ev[0][i] - iwin: d_ev[0][i] + iwin + 1]
                        _xtmp = np.arange(_tmp.size)
                        _idx0 = _xtmp[~np.isnan(_tmp)][0]
                        _idx1 = _xtmp[~np.isnan(_tmp)][-1]
                        z = np.polyfit([_xtmp[_idx0], _xtmp[_idx1]],
                                       [_tmp[_idx0], _tmp[_idx1]], 1)
                        pF = np.poly1d(z)
                        dummy = _tmp-pF(_xtmp)

                    else:
                        dummy = Detrend(np.ma.masked_invalid(
                            inputS[n, d_ev[0][i] - iwin: d_ev[0][i] + iwin + 1]))
                else:
                    dummy = inputS[
                        n, d_ev[0][i] - iwin: d_ev[0][i] + iwin + 1]
                csTot[n, :, i] = dummy
                ampTot[n, i] = dummy[iwin / 2: 3 * iwin / 2].max() - \
                               dummy[iwin / 2: 3 * iwin / 2].min()

        # now compute the cas
        cs = np.nanmean(csTot, axis=2)
        tau = np.linspace(-iwin, iwin, 2 * iwin + 1) * self.dt
        err = stats.sem(csTot, axis=2, nan_policy='omit')
        return cs, tau, err, ampTot

    def _threshold(self, signal, threshold):
        """
        Given the signal initialized by the class it computes
        the location of the point above the threshold and
        creates a nd.array equal to 1 at the maximum above the
        given threshold.  The threshold is given
        Output:
            maxima = A binary array equal to 1 at the identification
                     of the structure (local maxima)
            allmax = A binary array equal to 1 in all the region
                     where the signal is above the threshold

        Example:
        >>> turbo = intermittency.Intermittency(signal, dt, fr)
        >>> maxima = turbo.threshold()
        >>> maxima = turbo.threshold(thr = xxx)
        """

        # this will be the output
        maxima = np.zeros(signal.size)
        allmax = np.zeros(signal.size)
        allmax[(signal > threshold)] = 1
        imin = 0
        for i in range(maxima.size - 1):
            i += 1
            if signal[i] >= threshold > signal[i - 1]:
                imin = i
            if signal[i] < threshold <= signal[i - 1]:
                imax = i - 1
                if imax == imin:
                    d = 0
                else:
                    d = signal[imin: imax].argmax()
                maxima[imin + d] = 1
        return maxima, allmax

    def _smooth(self, x, window_len=10, window='hanning'):
        """
        Smooth the data using a window with requested size.

        This method is based on the convolution of a scaled
        window with the signal.
        The signal is prepared by introducing
        reflected copies of the signal
        (with the window size) in both ends so that transient
        parts are minimized
        in the begining and end part of the output signal.

        input:
        ------
            x: the input signal
            window_len: the dimension of the smoothing window
            window: the type of window from 'flat', 'hanning',
                    'hamming', 'bartlett', 'blackman'
                     flat window will produce a moving average smoothing.
        output:
        -------
            the smoothed signal
        """

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, '''Input vector needs to be bigger than
            window size.'''

        if window_len < 3:
            return x

        if window not in ['flat', 'hanning', 'hamming', 'bartlett',
                          'blackman']:
            raise ValueError, '''Window is on of 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman'''

        s = np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]

        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = getattr(np, window)(window_len)
        y = np.convolve(w/w.sum(), s, mode='same')
        return y[window_len-1:-window_len+1]

    def _computeExB(self, data):
        """
        Giving the output of the conditional average
        it compute the amplitude of radial and poloidal
        electric field fluctuations taking into account the
        amplitude of the Isat conditional average structure

        """
        
        signal = data.sel(sig='Is') - data.sel(sig='Is').min()
        spline = UnivariateSpline(
            data.t, signal.values-signal.max().item()/6., s=0)
        # find the roots and the closest roots to 0
        roots = spline.roots()
        tmin = roots[roots<0][-1]
        try:
            tmax = roots[roots>0][0]
        except:
            tmax = 2.5e-5
        # now the fluctuations of the Epol
        ii = ((data.t >= tmin) & (data.t <= tmax))
        # recompute Epol from CAS floating potential
        _Epol =  (data.sel(sig='VFT_' + str(int(self.plunge))) -
                  data.sel(sig='VFM_' + str(int(self.plunge)))) / 4e-3
        Epol = np.abs(_Epol[ii].max().item() -
                      _Epol[ii].min().item())
        Erad = np.abs(data.sel(sig='Erad')[ii].max().item() -
                      data.sel(sig='Erad')[ii].min().item())       
        EpolErr = np.abs(
            np.max(data.sel(sig='Epol').values[ii]+data.err[1, ii]).item() -
            np.min(data.sel(sig='Epol').values[ii]-data.err[1, ii]).item())
        EradErr = np.abs(
            np.max(data.sel(sig='Erad').values[ii]+data.err[2, ii]).item() -
            np.min(data.sel(sig='Erad').values[ii]-data.err[2, ii]).item())
        
        out = {'Er': Erad, 'ErErr': EradErr,
               'Epol': Epol, 'EpolErr': EpolErr}
        return out

"""
Global class for the analysis of the shot for Topic 21
experiment based on the structure, lambda and profile analysis
"""
from __future__ import print_function
import MDSplus as mds
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import constants
import eqtools
import langmuir
import timeseries
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
    >>> sys.path.append('/home/vianello/work/topic21/Codes/python/general')
    >>> sys.path.append('/home/vianello/work/topic21/Codes/python/tcv')

       eqtools
       MDSplus
       bottleneck
       xarray
       langmuir
       scipy
       quickspikes
    """

    def __init__(self, shot, gas='D2', Lp='Div'):
        """

        Parameters
        ----------
        shot :
            Shot number
        gas :
            String indicating the gas. 'D2','H','He'
        Lp :
            String indicating if the parallel connection
            length should be considered from target to
            midplane 'Mid' or from target to X-point
            'Div'

        """
        self.shot = shot
        self.Lp = Lp
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
        self.BtSign = np.sign(self._eq.getBtVac().mean())
        # now define the tree where the probe data are saved
        self._filament = mds.Tree('tcv_topic21', shot)
        # this is for different quantities
        self._tree = mds.Tree('tcv_shot', shot)
        # this can be done one single time and then accordingly
        # to analysis region we choose the appropriate timing
        try:
            self.Target = langmuir.LP(self.shot)
        except:
            print('Langmuir probe data not found')
            self.Target = None

    def blob(self, plunge=1, rrsep=None, rho=None,
             **kwargs):
        """
        Given the plunge stated and the region defined in
        R-Rsep or rho it computes various quantities for the
        blob, including the auto-correlation time,
        the size as FWHM of the iSat, the radial velocity
        as computed from Epoloidal, and also the perpendicular
        velocity as computed according to propagation of
        floating potential corresponding structure

        Parameters
        ----------
        plunge : int
            Plunge number
        rrsep: ndarray
            ndarray or list of the form [min,max] indicating
            the region in terms of distance from the separatrix
        rho : ndarray
            ndarray or list of the form [min,max] indicating
            the region in terms of rho
        **kwargs :
            These are the keyword that can be used in calling timeseries
            and the method therein

        Results
        -------
        Xarray dataset containing all the information
        needed for the computation of filament properties including
        Lambda and Theta
        """
        # in case the quantities are not loaded let load it
        if 'Type' in kwargs:
            Type = kwargs['Type']
        else:
            Type ='THRESHOLD'

        if 'outlier' not in kwargs:
            outlier = False
        try:
            if self.plunge != plunge:
                self._loadProbe(plunge=plunge,outlier=outlier)
                self._loadProfile()
            else:
                print('Already loaded')
        except:
            self._loadProbe(plunge=plunge, outlier=outlier)
            self._loadProfile()

        # now check if we have define the distances in
        # rho or rrsep
        if (rho is None) and (rrsep is not None):
            _idx = np.where(
                ((self.RRsep >= rrsep[0]) &
                 (self.RRsep <= rrsep[1])
                 ))[0]
            tmin = self.Rtime[_idx].min()
            tmax = self.Rtime[_idx].max()
        elif (rrsep is None) and (rho is not None):
            _idx = np.where(
                ((self.Rhop >= rrsep[0]) &
                 (self.Rhop <= rrsep[1])
                 ))[0]
            tmin = self.Rtime[_idx].min()
            tmax = self.Rtime[_idx].max()
        else:
            print('You must specify region in the profile')

        # we can also compute the absolute radial position
        # which can be used to save the Btot
        Ravg = self.R[_idx].mean()
        Btot = self._eq.rz2B(Ravg, 0, (tmax + tmin) / 2)
        # limit to this time interval
        sIs = self.iS[((self.iS.time >= tmin) &
                       (self.iS.time <= tmax))].values
        sVf = self.vF[:, ((self.vF.time >= tmin) &
                          (self.vF.time <= tmax))].values

        sEp = self.Epol[((self.Epol.time >= tmin) &
                         (self.Epol.time <= tmax))].values
        sEr = self.Erad[((self.Erad.time >= tmin) &
                         (self.Erad.time <= tmax))].values
        sigIn = np.vstack((sEp, sEr, sVf))
        self.dt = (self.iS.time.max().item() -
                   self.iS.time.min().item()) / (self.iS.size - 1)
        if 'dtS' in kwargs:
            dtS = dtS
        else:
            dtS = 0.0002
        self.Structure = timeseries.Timeseries(
            sIs,
            self.iS.time.values[
                ((self.iS.time >= tmin) &
                 (self.iS.time <= tmax))], dtS=dtS)
        # this is the determination of the correspondingreload
        cs, tau, err, amp = self.Structure.casMultiple(
            sigIn, **kwargs)
        # the output will be an xray DataArray containing
        # the results of CAS plus additional informations
        names = np.append(['Is', 'Epol', 'Erad'], self.vF.Probe.values)
        data = xray.DataArray(cs, coords=[names, tau], dims=['sig','t'])
        # add the errorss
        data.attrs['err'] = err
        # position, time
        data.attrs['R'] = Ravg
        data.attrs['tmin'] = tmin
        data.attrs['tmax'] = tmax
        data.attrs['RrsepMin'] = self.RRsep[_idx].min()
        data.attrs['RrsepMax'] = self.RRsep[_idx].max()
        # start adding the interesting quantities
        # the FWHM
        delta, errDelta = self._computeDeltaT(tau, cs[0, :],
                                              err[0, :])
        data.attrs['FWHM'] = delta
        data.attrs['FWHMerr'] = errDelta
        # the vr from the poloidal electric field
        out = self._computeExB(data)
        data.attrs['vrExB'] = out['Epol'] / Btot
        data.attrs['vrExBerr'] = out['EpolErr'] / Btot
        # the fluctuating poloidal velocity
        data.attrs['vpExB'] = out['Er'] / Btot
        data.attrs['vpExBerr'] = out['ErErr'] / Btot
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
        data.attrs['T_ac'] = self.Structure.act
        # compute the Ion sound gyroradius in this zone
        # we need the standard deviation of the te
        teStd = self.profileTe[
            ((self.profileTe.rho >= self.RRsep[_idx].min()) &
             (self.profileTe.rho <= self.RRsep[_idx].max()))].std().item()
        CsDict = self._computeRhos(Ravg, np.asarray(
            [self.RRsep[_idx].min(), self.RRsep[_idx].max()]),
                                       (tmax + tmin) / 2)
        data.attrs['rhos'] = CsDict['rhos']
        data.attrs['drhos'] = CsDict['drhos']
        data.attrs['Cs'] = CsDict['Cs']
        # we also compute the corresponding Lambda and Delta
        # so that we save all the information on one single
        # xray dataArray
        # Lambda Div  Computation
        Lambda, Err = self._computeLambda(
            rrsep=[self.RRsep[_idx].min(),
                   self.RRsep[_idx].max()],
            trange=[tmin-0.04, tmax+0.04], Lp='Div')
        data.attrs['LambdaDiv'] = Lambda
        data.attrs['LambdaDivErr'] = Err
        # All Lp
        Lambda, Err = self._computeLambda(
            rrsep=[self.RRsep[_idx].min(),
                   self.RRsep[_idx].max()],
            trange=[tmin-0.04, tmax+0.04], Lp='Tot')
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

        Parameters
        ----------
        plunge :
            Plunge number

        Returns
        -------
        None
            It only define the attributes with the appropriate
            names
        """
        # for some reason we need befor makin and mds.Data.compile I need to
        # restore some data from Tree
        _dummy = self._tree.getNode(r'\results::fir:n_average').data()
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
            (np.asarray(map(lambda n: n[3], isNames)) == '_')]

    def _loadProbe(self, plunge=1, outlier=False, outThr=150):
        """
        Load the signal from the Probe and ensure
        we are using exclusively the insertion part of the
        plunge. If it has not been already loaded the profile
        it loads also the profile

        Parameters
        ----------
        plunge :
            Integer indicating the plunge number
        outlier :
            Boolean for indicating if outlier should be
            eliminated
        outThr :
            Threshold for eliminating the outlier

        Returns
        -------
        Save as attributes for the class the Ion saturation current
        and the floating potential (as xarray.DataSet) plus the
        values of rho and drSep as function of time
        """

        self.plunge = plunge
        # now load the data and save them in the appropriate xarray
        self._getNames(plunge=plunge)
        # now we load the floating potential and save them in an xarray
        vF = []
        for name in self.vfNames:
            vF.append(self._tree.getNode(r'\FP' + name).data())
        # convert in a numpy array
        vF = np.asarray(vF)
        # get the time basis
        time = self._tree.getNode(r'\FP' + self.vfNames[0]).getDimensionAt().data()
        # we need to build an appropriate time basis since it has not a
        # constant time step
        time = np.linspace(time.min(), time.max(), time.size, dtype='float64')
        vF = xray.DataArray(vF, coords=[self.vfNames, time], dims=['Probe', 'time'])
        # repeat for the ion saturation current
        if self.isNames.size == 1:
            iS = self._tree.getNode(r'\FP' + self.isNames[0]).data()
            iS = xray.DataArray(iS, coords=[time], dims=['time'])
        else:
            iS = []
            for name in self.isNames:
                iS.append(self._tree.getNode(r'\FP' + name).data())
            # convert in a numpy array
            iS = np.asarray(iS)
            # save an xarray dataset
            iS = xray.DataArray(iS, coords=[self.isNames, time], dims=['Probe', 'time'])
        # this is upstream remapped I load the node not the data
        RRsep = self._filament.getNode(r'\FP_%1i' % plunge + 'PL_RRSEPT')
        # this is in Rhopoloidal
        Rhop = self._filament.getNode(r'\FP_%1i' % plunge + 'PL_RHOT')
        R = self._tree.getNode(r'\FPCALPOS_%1i' % self.plunge)
        # limit to first insertion of the probe
        Rtime = Rhop.getDimensionAt().data()[:np.nanargmin(RRsep.data())]
        Rhop = Rhop.data()[:np.nanargmin(RRsep.data())]
        RRsep = RRsep.data()[:np.nanargmin(RRsep.data())]
        # limit to the timing where we insert the
        ii = np.where((iS.time >= Rtime.min()) &
                      (iS.time <= Rtime.max()))[0]
        if 1 == self.isNames.size:
            self.iS = iS[ii]
        else:
            self.iS = iS[:, ii]
        self.vF = vF[:, ii]
        # now check if the number of size between position and signal
        # are the same otherwise univariate interpolate the position
        if Rtime.size != self.iS.time.size:
            S = interp1d(Rtime[~np.isnan(Rhop)], Rhop[~np.isnan(Rhop)],
                         kind='linear',fill_value='extrapolate')
            self.Rhop = S(self.iS.time.values)
            S = interp1d(Rtime[~np.isnan(RRsep)], RRsep[~np.isnan(RRsep)],
                         kind='linear',fill_value='extrapolate')
            self.RRsep = S(self.iS.time.values)
            self.Rtime = self.iS.time.values
        else:
            self.Rhop = Rhop
            self.RRsep = RRsep
            self.Rtime = Rtime
        S = interp1d(R.getDimensionAt().data(), R.data() / 1e2, kind='linear',
                     fill_value='extrapolate')
        self.R = S(self.iS.time.values)
        # in case we set outlier we eliminate outlier with NaN
        if outlier:
            det = qs.detector(outThr, 40)
            for _probe in range(self.vF.shape[0]):
                times = det.send(np.abs(
                    self.vF[_probe, :]).values.astype('float64'))
                if len(times) != 0:
                    for t in times:
                        if t + 40 < self.vF.shape[1]:
                            self.vF[_probe, t - 40:t + 40] = np.nan
                        else:
                            self.vF[_probe, t - 40:-1] = np.nan
                            # perform a median filter to get rid of possible
                        #        self.R = R.rolling(time=20).mean()[:R.argmin().item()]
        self.Epol = (vF.sel(Probe='VFT_' + str(int(self.plunge))) -
                     vF.sel(Probe='VFM_' + str(int(self.plunge)))) / 4e-3
        self.Erad = (vF.sel(Probe='VFM_' + str(int(self.plunge))) -
                     vF.sel(Probe='VFR1_' + str(int(self.plunge)))) / 1.57e-3

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
                _data[_ii] / 1e19,
                coords={'drsep': _rho[_ii]})
            self.profileEn.attrs['err'] = _err[_ii] / 1e19
        else:
            # density profile and corresponding spline
            self.profileEn = self._profileaverage(
                _rho[_ii],
                _data[_ii] / 1e19,
                npoint=npoint)
        self.splineEn = UnivariateSpline(self.profileEn.rho.values,
                                         self.profileEn.values,
                                         w=1. / self.profileEn.err,
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
                coords={'drsep': _rho[_ii]})
            if _err:
                self.profileEn.attrs['err'] = _err[_ii]
        else:
            self.profileTe = self._profileaverage(
                _rho[_ii],
                _data[_ii],
                npoint=npoint)
        self.splineTe = UnivariateSpline(self.profileTe.rho.values,
                                         self.profileTe.values,
                                         w=1. / self.profileTe.err,
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
        _dummy = (y - y.min())
        spline = UnivariateSpline(x, _dummy - _dummy.max() / 2, s=0)
        if spline.roots().size > 2:
            a = np.sort(spline.roots())
            r1 = a[a < 0][-1]
            r2 = a[a > 0][0]
        else:
            r1, r2 = spline.roots()
        delta = (r2 - r1)
        # now compute an estimate of the error
        _dummy = (y + e) - (y + e).min()
        spline = UnivariateSpline(x, _dummy - _dummy.max() / 2, s=0)
        if spline.roots().size > 2:
            a = np.sort(spline.roots())
            r1 = a[a < 0][-1]
            r2 = a[a > 0][0]
        else:
            r1, r2 = spline.roots()
        deltaUp = (r2 - r1)
        _dummy = (y - e) - (y - e).min()
        spline = UnivariateSpline(x, _dummy - _dummy.max() / 2, s=0)
        if spline.roots().size > 2:
            a = np.sort(spline.roots())
            r1 = a[a < 0][-1]
            r2 = a[a > 0][0]
        else:
            r1, r2 = spline.roots()
        deltaDown = (r2 - r1)
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
        Gamma = 5. / 3.
        Btot = self._eq.rz2B(ravg, 0, t)
        Omega_i = 9.58e3 / 2. * Btot * self.Z * 1e4 / self.mu
        Te = self.profileTe[((self.profileTe.rho >= rrsep[0]) &
                             (self.profileTe.rho <= rrsep[1]))].mean().item()
        dte = self.profileTe[((self.profileTe.rho >= rrsep[0]) &
                              (self.profileTe.rho <= rrsep[1]))].std().item()
        Cs = 9.79 * 1e3 * np.sqrt(Te * Gamma * self.Z / self.mu)
        dCs = 9.79 * 1e3 / (2. * np.sqrt(self.splineTe(np.mean(rrsep)))) * \
              np.sqrt(Gamma * self.Z / self.mu) * dte
        rhoS = Cs / Omega_i
        drhoS = dCs / Omega_i
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
        xcorA /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagA = np.arange(xcorA.size) - xcorA.size / 2
        vpA = (0.2433 - 0.0855) * constants.inch / (lagA[np.argmax(xcorA)] * self.dt)
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
        lagB = np.arange(xcorB.size) - xcorB.size / 2
        vpB = (0.2433 + 0.1512) * constants.inch / (lagB[np.argmax(xcorB)] * self.dt)
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
        lagC = np.arange(xcorC.size) - xcorC.size / 2
        vpC = (0.0855 + 0.1512) * constants.inch / (lagC[np.argmax(xcorC)] * self.dt)
        deltaPoloidal = np.nanmean([vpA, vpB, vpC])
        deltaPoloidalStd = np.nanstd([vpA, vpB, vpC])

        a = bottleneck.move_mean(
            data.sel(sig='VFM_' + str(int(self.plunge))), window=3)
        b = bottleneck.move_mean(
            data.sel(sig='VFR1_' + str(int(self.plunge))), window=3)
        deltaTimeR = np.abs(
            np.nanargmin(a) - np.nanargmin(b))
        deltaRadial = 4e-3 / (deltaTimeR * self.dt)
        vperp = np.sqrt(deltaPoloidal ** 2 + deltaRadial ** 2)
        out = {'vperp': vperp, 'vpol': deltaPoloidal,
               'vrad': deltaRadial, 'vpolErr': deltaPoloidalStd}
        return out

    def _computeLambda(self, rrsep=[0.001,0.003],
                       trange=[0.8,0.9], Lp='Div'):
        """
        Restore from MDSplus Tree the value of divertor collisionality
        If not available it computes it using Field Line Tracing code through
        the class Target

        :param rrsep:
            The range in distance from the separatrix upstream remapped
        :param trange:
            Range of time where the Lambda should be average
        :param Lp:
            String indicating if we should consider the parallel
            connection length from Target to X-point heigth ('Div')
            or from Target to midplane ('Midplane'
        :return:
            Value of divertor normalized collisionality
        """
        try:
            if Lp == 'Div':
                _LambdaN = self._filament.getNode(
                    r'\LDIVX'
                )
            else:
                _LambdaN = self._filament.getNode(
                    r'\LDIVU'
                )
            LpT = _LambdaN.getDimensionAt(0).data()
            _idx = np.where(((LpT >= trange[0]) &
                            (LpT <= trange[1])))[0]
            xCl = _LambdaN.getDimensionAt(1).data()
            tmp = np.nanmean(_LambdaN.data()[_idx,:],axis=0)
            tmpstd = np.nanstd(_LambdaN.data()[_idx,:],axis=0)
            _rdx = np.where(((xCl >= rrsep[0]) & (xCl <= rrsep[1])))[0]
            Lambda = np.mean(tmp[_rdx], weights=1./tmpstd[_rdx])
            LambdaErr = np.std(tmp[_rdx],weights=1./tmpstd[_rdx])
        except:
            _Lambda, xCl = self.Target.Lambda(gas=self.gas, trange=trange)
            _rdx = np.where(((xCl >= rrsep[0]) & (xCl <= rrsep[1])))[0]
            Lambda = np.nanmean(_Lambda[_rdx])
            LambdaErr = np.nanstd(_Lambda[_rdx])

        return Lambda, LambdaErr

    def _computeTheta(self, data, Lp='Div'):
        try:
            if Lp == 'Div':
                LpN = self._filament.getNode(r'\LPDIVX')
            else:
                LpN = self._filament.getNode(r'\LPDIVU')
            # the input Data has all the variables we need to compute
            # the averages
            _idx = np.where(((LpN.getDimensionAt(0).data() >= data.tmin-0.06) &
                             (LpN.getDimensionAt(0).data() <= data.tmax + 0.06)))[0]
            # average over time
            tmp = np.nanmean(LpN.data()[_idx, :],axis=0)
            tmpstd = np.nanstd(LpN.data()[_idx, :], axis=0)
            # average over distance from the separatrix
            _rdx = np.where(((LpN.getDimensionAt(1).data() >= data.RrsepMin) &
                             (LpN.getDimensionAt(1).data() <= data.RrsepMax)))[0]
            Lpar = np.mean(tmp[_rdx],weights=1./tmpstd)
            dLpar = np.std(tmp[_rdx],weights=1./tmpstd)
            Theta = ((data.FWHM * np.sqrt(
                data.vrExB ** 2 + data.vAutoP ** 2)) ** (5 / 2.) * np.sqrt(0.88)) / \
                    (Lpar * data.rhos ** 2)
            dTheta = Theta * np.sqrt(
                (data.FWHMerr / data.FWHM) ** 2 +
                (data.vrExBerr / data.vrExB) ** 2 +
                (dLpar / Lpar) ** 2 + (data.drhos / data.rhos) ** 2)
        except:
            Theta = np.nan
            dTheta = np.nan
        return Theta, dTheta

    def _computeExB(self, data):
        """
        Giving the output of the conditional average
        it compute the amplitude of radial and poloidal
        electric field fluctuations taking into account the
        amplitude of the Isat conditional average structure

        """

        signal = data.sel(sig='Is') - data.sel(sig='Is').min()
        spline = UnivariateSpline(
            data.t, signal.values - signal.max().item() / 6., s=0)
        # find the roots and the closest roots to 0
        roots = spline.roots()
        tmin = roots[roots < 0][-1]
        try:
            tmax = roots[roots > 0][0]
        except:
            tmax = 2.5e-5
        # now the fluctuations of the Epol
        ii = ((data.t >= tmin) & (data.t <= tmax))
        # recompute Epol from CAS floating potential
        _Epol = (data.sel(sig='VFT_' + str(int(self.plunge))) -
                 data.sel(sig='VFM_' + str(int(self.plunge)))) / 4e-3
        Epol = np.abs(_Epol[ii].max().item() -
                      _Epol[ii].min().item())
        Erad = np.abs(data.sel(sig='Erad')[ii].max().item() -
                      data.sel(sig='Erad')[ii].min().item())
        EpolErr = np.abs(
            np.max(data.sel(sig='Epol').values[ii] + data.err[1, ii]).item() -
            np.min(data.sel(sig='Epol').values[ii] - data.err[1, ii]).item())
        EradErr = np.abs(
            np.max(data.sel(sig='Erad').values[ii] + data.err[2, ii]).item() -
            np.min(data.sel(sig='Erad').values[ii] - data.err[2, ii]).item())

        out = {'Er': Erad, 'ErErr': EradErr,
               'Epol': Epol, 'EpolErr': EpolErr}
        return out

    def _profileaverage(self, x, y, npoint=20):
        """
        Given x an y compute the profile assuming
        x is the coordinate and y the variable. It does it
        by sorting along x, splitting into npoint
        and computing the mean and standard deviation
        Parameters
        ----------
        x : coordinate
        y : variables
        npoint : number of point for avergae

        Returns
        -------
        Xarray DataFrame containing variables rho, values and
        corresponding error
        """
        y = y[np.argsort(x)]
        x = x[np.argsort(x)]
        yS = np.array_split(y, npoint)
        xS = np.array_split(x, npoint)
        yO = np.asarray([np.nanmean(k) for k in yS])
        xO = np.asarray([np.nanmean(k) for k in xS])
        eO = np.asarray([np.nanstd(k) for k in yS])
        data = xray.DataArray(yO, coords=[xO], dims=['rho'])
        data.attrs['err'] = eO
        return data

    

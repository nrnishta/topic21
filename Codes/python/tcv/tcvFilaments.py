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
from scipy import signal
import eqtools
import langmuir
import timeseries
import xarray as xray
import bottleneck
import quickspikes as qs
from lmfit.models import GaussianModel
import tcvProfiles


# noinspection PyAttributeOutsideInit
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
       profiletools
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
        except RuntimeError:
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
            Type = 'THRESHOLD'

        if 'outlier' not in kwargs:
            outlier = False
        try:
            if self.plunge != plunge:
                self._loadProbe(plunge=plunge, outlier=outlier)
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
        # this is the high-pass filter signal floating potential
        # used for the computation of cross-correlation velocity
        # using the same method used by C.Tsui and J.Boedo
        # only for this we don't take the value but the xarray
        sVfF = self.vFF[:, ((self.vF.time >= tmin) &
                            (self.vF.time <= tmax))]
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
        # noinspection PyAttributeOutsideInit
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
        data = xray.DataArray(cs, coords=[names, tau], dims=['sig', 't'])
        # add the errorss
        data.attrs['err'] = err
        # position, time
        data.attrs['R'] = Ravg
        data.attrs['Rho'] = self._eq.rz2psinorm(
            Ravg, 0, (tmax + tmin) / 2, sqrt=True)
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
        # now we also add the third type of evaluation of
        # the vperp as done by C. Tsui and J. Boedo
        vpol, dvpol, _, _ = self._computeVpolCC(sVfF)
        data.attrs['vpol'] = vpol
        data.attrs['dvpol'] = dvpol
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
            trange=[tmin - 0.04, tmax + 0.04], Lp='Div')
        data.attrs['LambdaDiv'] = Lambda
        data.attrs['LambdaDivErr'] = Err
        # All Lp
        Lambda, Err = self._computeLambda(
            rrsep=[self.RRsep[_idx].min(),
                   self.RRsep[_idx].max()],
            trange=[tmin - 0.04, tmax + 0.04], Lp='Tot')
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

        # add the Efolding 
        _idx = np.where(((self.rhoArray >= self.RRsep[_idx].min()) &
                         (self.rhoArray <= self.RRsep[_idx].max())))[0]
        data.attrs['Efold'] = self.Efolding[_idx].mean()
        data.attrs['EfoldErr'] = self.Efolding[_idx].std()
        data.attrs['EfoldGpr'] = self.EfoldingGpr[_idx].mean()
        data.attrs['EfoldGprErr'] = self.EfoldingGpr[_idx].mean()
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
        _nameProbe = np.core.defchararray.strip(
            mds.Data.compile(_string).evaluate().data())
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
        it loads also the profile. We also create the appropriate
        attribute with floating potential with 10 kHz high pass
        filtering due to a pick-up from FPS coils

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
        # get the time basis
        time = self._tree.getNode(
            r'\FP' + self.vfNames[0]).getDimensionAt().data()
        dt = (time.max() - time.min()) / (time.size - 1)
        Fs = np.round(1. / dt)
        # now we load the floating potential and save them in an xarray
        vF = []
        vFF = []
        for name in self.vfNames:
            vF.append(self._tree.getNode(r'\FP' + name).data())
            vFF.append(self.bw_filter(
                self._tree.getNode(r'\FP' + name).data(),
                10e3, Fs, 'highpass', order=5))
            # convert in a numpy array
        vF = np.asarray(vF)
        vFF = np.asarray(vFF)
        # we need to build an appropriate time basis since it has not a
        # constant time step
        time = np.linspace(time.min(), time.max(), time.size, dtype='float64')
        vF = xray.DataArray(vF, coords=[self.vfNames, time],
                            dims=['Probe', 'time'])
        vFF = xray.DataArray(vFF, coords=[self.vfNames, time],
                             dims=['Probe', 'time'])
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
            iS = xray.DataArray(iS, coords=[self.isNames, time],
                                dims=['Probe', 'time'])
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
        self.vFF = vFF[:, ii]
        # now check if the number of size between position and signal
        # are the same otherwise univariate interpolate the position
        if Rtime.size != self.iS.time.size:
            S = interp1d(Rtime[~np.isnan(Rhop)], Rhop[~np.isnan(Rhop)],
                         kind='linear', fill_value='extrapolate')
            self.Rhop = S(self.iS.time.values)
            S = interp1d(Rtime[~np.isnan(RRsep)], RRsep[~np.isnan(RRsep)],
                         kind='linear', fill_value='extrapolate')
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
                coords=[_rho[ii]], dims=['drsep'])
            self.profileEn.attrs['err'] = _err[_ii] / 1e19
        else:
            # density profile and corresponding spline
            self.profileEn = self._profileaverage(
                _rho[_ii],
                _data[_ii] / 1e19,
                npoint=npoint)
        _idx0 = np.where(self.profileEn.err == 0)[0]
        if _idx0.size != 0:
            self.profileEn.err[_idx0] = \
                np.mean(self.profileEn.err[np.where(self.profileEn.err != 0)[0]])
        self.splineEn = UnivariateSpline(self.profileEn.rho.values,
                                         self.profileEn.values,
                                         w=1. / self.profileEn.err,
                                         ext=0)
        # we had the computation of the profiles using the profiletools
        # which allows to combine also with Thomson data
        Profile = tcvProfiles.tcvProfiles(self.shot)
        EnProf = Profile.profileNe(t_min=_time.min() - 0.05,
                                   t_max=_time.max() + 0.05,
                                   abscissa='Rmid')
        # Compute the GPR estimate of the Fit
        _rhoN = np.linspace(EnProf.X.ravel().min(),
                            EnProf.X.ravel().max(), 151)
        _yN, _yE, _gp = Profile.gpr_robustfit(_rhoN, gaussian_length_scale=0.3,
                                              nu_length_scale=0.03)
        # now convert to R-Rmid
        _rhoN -= self._eq.getRmidOutSpline()(_time.mean())
        # drop the value in the profile below 2 cm inside the LCFS
        _ = EnProf.remove_points((EnProf.X[:, 0] - self._eq.getRmidOutSpline()(_time.mean())) < -0.02)
        # remember that EnProf save the density in [10^20]
        self.profileEnGpr = xray.DataArray(
            EnProf.y * 10,
            coords=[EnProf.X.ravel() - self._eq.getRmidOutSpline()(_time.mean())],
            dims='drsep')
        self.profileEnGpr.attrs['err'] = EnProf.err_y * 10
        self.profileEnGpr.attrs['err X'] = EnProf.err_X
        # the spline is built on GPR fit
        _idx = np.where(_rhoN >= -0.02)[0]
        self.splineGpr = UnivariateSpline(
            _rhoN[_idx],
            _yN[_idx] * 10,
            w=1. / (_yE[_idx] * 10), ext=0)

        # compute also the profile of Te
        _data = self._filament.getNode(
            r'\FP_%1i' % plunge + 'PL_TE').data()
        try:
            _err = self._filament.getNode(
                r'\FP_%1i' % plunge + 'PL_TEERR').data()
        except ValueError:
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
        self.EfoldingGpr = np.abs(self.splineGpr(self.rhoArray) /
                                  self.splineGpr.derivative()(self.rhoArray))

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
        potential structure. It uses both the simple
        computation and also the estimate using the 
        evaluation of the binormal velocity
        using the formula in D. Carralero NF vol54, p 123005 (2014).
        This can't be simply applied unfortunately since the radial
        distances of the pins is too small and we get a zero time delay
        cross-correlation. So basically we use the method of Daniel where
        
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
        lagA = np.arange(xcorA.size, dtype='float') - xcorA.size / 2
        lagA *= self.dt
        # replicate what done for Cedric, i.e. Gaussian Fit and centroid
        mod = GaussianModel()
        pars = mod.guess(xcorA, x=lagA)
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorA, pars, x=lagA)
        dtA = out.params['center'].value
        vpA = (0.2433 - 0.0855) * constants.inch / (dtA)
        vpAS = vpA * (out.params['center'].stderr / dtA)
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
        lagB = np.arange(xcorB.size, dtype='float') - xcorB.size / 2
        lagB *= self.dt
        mod = GaussianModel()
        pars = mod.guess(xcorB, x=lagB)
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorB, pars, x=lagB)
        dtB = out.params['center'].value
        vpB = (0.2433 + 0.1512) * constants.inch / (dtB)
        vpBS = vpB * (out.params['center'].stderr / dtB)

        # --------------------------
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
        lagC = np.arange(xcorC.size, dtype='float') - xcorC.size / 2
        lagC *= self.dt
        mod = GaussianModel()
        pars = mod.guess(xcorC, x=lagC)
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorC, pars, x=lagC)
        dtC = out.params['center'].value
        vpC = (0.0855 + 0.1512) * constants.inch / (dtC)
        vpCS = vpB * (out.params['center'].stderr / dtC)

        _all = np.asarray([vpA, vpB, vpC])
        _allDt = np.asarray([dtA, dtB, dtC])
        deltaPoloidal = np.mean(_all[np.isfinite(_all)])
        deltaPoloidalStd = np.nanstd(_all[np.isfinite(_all)])
        deltaTPoloidal = np.mean(_allDt[_allDt != 0])
        deltaTPoloidalStd = np.mean(_allDt[_allDt != 0])
        # -----------------------------
        # now we compute the same stuff
        # for the radially separated
        # pins
        # M-R1
        a = bottleneck.move_mean(
            data.sel(sig='VFM_' + str(int(self.plunge))), window=3)
        b = bottleneck.move_mean(
            data.sel(sig='VFR1_' + str(int(self.plunge))), window=3)
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        xcorD = np.correlate(a, b, mode='same')
        xcorD /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagD = np.arange(xcorD.size, dtype='float') - xcorD.size / 2
        lagD *= self.dt
        mod = GaussianModel()
        pars = mod.guess(xcorD, x=lagD)
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorD, pars, x=lagD)
        # centroid with the error
        dtD = out.params['center'].value
        dtDS = out.params['center'].stderr
        # compute velocity with error
        vrD = (1.57e-3) / dtD
        vrDS = vrD * (dtDS / dtD)
        # ------------
        # M-R2
        a = bottleneck.move_mean(
            data.sel(sig='VFM_' + str(int(self.plunge))), window=3)
        b = bottleneck.move_mean(
            data.sel(sig='VFR2_' + str(int(self.plunge))), window=3)
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        xcorE = np.correlate(a, b, mode='same')
        xcorE /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagE = np.arange(xcorE.size, dtype='float') - xcorE.size / 2
        lagE *= self.dt
        mod = GaussianModel()
        pars = mod.guess(xcorE, x=lagE)
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorE, pars, x=lagE)
        # centroid with the error
        dtE = out.params['center'].value
        dtES = out.params['center'].stderr
        # compute velocity with error
        vrE = (1.57e-3) / dtE
        vrES = vrE * (dtES / dtE)
        #
        _allDt = np.asarray([dtD, dtE])
        deltaRadial = np.nanmean([vrD, vrE])
        deltaRadialStd = np.nanstd([vrD, vrE])
        deltaTimeR = np.mean(_allDt[_allDt != 0])
        vperp = np.sqrt(deltaPoloidal ** 2 + deltaRadial ** 2)

        # use the formula introduced in Carralero NF paper
        Ltheta = (0.2433 + 0.1512) * constants.inch
        Lr = 0.00157
        _dummy = np.sqrt(np.power(dtB / Ltheta, 2) +
                         np.power((2 * deltaTimeR - dtB) / (2 * Lr), 2))
        vperp2 = 1. / _dummy
        vrad2 = np.cos(np.arcsin(vperp2 * dtB / Ltheta)) * vperp2
        vpol2 = np.power(vperp2, 2) * (dtB / Ltheta)

        out = {'vperp': vperp, 'vpol': deltaPoloidal,
               'vrad': deltaRadial, 'vpolErr': deltaPoloidalStd,
               'vperp2': vperp2, 'vrad2': vrad2, 'vpol2': vpol2}
        return out

    def _computeVpolCC(self, data):
        """
        2D cross-correlation method to compute the poloidal flow
        using the high-pass filtered floating potential signal.
        Differently from before it uses a larger time window with
        respect to CAS output and once the CC is computed is fit
        with a gaussian using the centroid as better estimate of 
        velocity.
        """

        # cross-correlation filtered vfm and vft
        a = data.sel(Probe='VFM_' + str(int(self.plunge)))
        b = data.sel(Probe='VFT_' + str(int(self.plunge)))
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        # compute the cross correlation
        xcorA = np.correlate(a, b, mode='same')
        # normalize appropriately
        xcorA /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagA = np.arange(xcorA.size, dtype='float') - xcorA.size / 2
        lagA *= self.dt
        mod = GaussianModel()
        pars = mod.guess(xcorA, x=lagA)
        # limit the sigma of the gaussian to a
        # suitable initial value of 1e-5
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorA, pars, x=lagA)
        # centroid with the error
        dtA = out.params['center'].value
        dtAS = out.params['center'].stderr
        # compute velocity with error
        vpA = (0.2433 - 0.0855) * constants.inch / dtA
        vpAS = vpA * (dtAS / dtA)


        # ---------------------
        # repeat with the couple bottom top
        a = data.sel(Probe='VFB_' + str(int(self.plunge)))
        b = data.sel(Probe='VFT_' + str(int(self.plunge)))
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        # compute the cross correlation
        xcorB = np.correlate(a, b, mode='same')
        # normalize appropriately
        xcorB /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagB = np.arange(xcorB.size, dtype='float') - xcorB.size / 2
        lagB *= self.dt
        mod = GaussianModel()
        pars = mod.guess(xcorB, x=lagB)
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorB, pars, x=lagB)
        # centroid with the error
        dtB = out.params['center'].value
        dtBS = out.params['center'].stderr
        # compute velocity with error
        vpB = (0.2433 + 0.1512) * constants.inch / dtB
        vpBS = vpB * (dtBS / dtB)
        # for some reason the use of the couple Top bottom yealds
        # unreliable results. We introduce a confidence as normalized
        # difference and eventually if too large we use only
        # the couple Top middle
        confidence = (vpA - vpB) / (vpA)
        if confidence > 0.2:
            vP = vpA
            vPErr = vpAS
        else:
            # now determine the weighted average and
            # corresponding error
            vP = np.average(np.asarray([vpA, vpB]),
                            weights=np.asarray([1. / vpAS, 1. / vpBS]))
            vPErr = np.std(np.asarray([vpA, vpB]))
        # -------------------
        # now do the same
        # for the radial ones
        # couple M-R1
        a = data.sel(Probe='VFM_' + str(int(self.plunge)))
        b = data.sel(Probe='VFR1_' + str(int(self.plunge)))
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        # compute the cross correlation
        xcorC = np.correlate(a, b, mode='same')
        # normalize appropriately
        xcorC /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagC = np.arange(xcorC.size, dtype='float') - xcorC.size / 2
        lagC *= self.dt
        mod = GaussianModel()
        pars = mod.guess(xcorC, x=lagC)
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorC, pars, x=lagC)
        # centroid with the error
        dtC = out.params['center'].value
        dtCS = out.params['center'].stderr
        # compute velocity with error
        vrC = (1.57e-3) / dtC
        vrCS = vrC * (dtCS / dtC)
        # -----------
        # couple M-R2
        a = data.sel(Probe='VFM_' + str(int(self.plunge)))
        b = data.sel(Probe='VFR2_' + str(int(self.plunge)))
        _dummy = np.vstack((a, b)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        a = _dummy[:, 0]
        b = _dummy[:, 1]
        # compute the cross correlation
        xcorD = np.correlate(a, b, mode='same')
        # normalize appropriately
        xcorD /= np.sqrt(np.dot(a, a) * np.dot(b, b))
        lagD = np.arange(xcorD.size, dtype='float') - xcorD.size / 2
        lagD *= self.dt
        mod = GaussianModel()
        pars = mod.guess(xcorD, x=lagD)
        pars['sigma'].set(value=1e-5, vary=True)
        out = mod.fit(xcorD, pars, x=lagD)
        # centroid with the error
        dtD = out.params['center'].value
        dtDS = out.params['center'].stderr
        # compute velocity with error
        vrD = (1.57e-3) / dtD
        vrDS = vrD * (dtDS / dtD)
        # now determine the weighted average and corresponding error
        vR = np.average(np.asarray([vrC, vrD]),
                        weights=np.asarray([1. / vrCS, 1. / vrDS]))
        vRErr = np.std(np.asarray([vrC, vrD]))

        # and now the 2D cross-correlation
        # determine the angle
        theta = np.arctan(vR / vP)
        dtheta = theta * np.sqrt(
            np.power(vRErr / vR, 2) + np.power(vPErr / vP, 2))
        vZ = vP * np.power(np.sin(theta), 2)
        # propagate the error
        _dumm = np.sqrt(np.power(vPErr / vP, 2) +
                        np.power(dtheta / theta, 2))
        dvZ = vZ * _dumm
        return vZ, dvZ, vP, vR

    def _computeLambda(self, rrsep=None,
                       trange=None, Lp='Div'):
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
        if rrsep is None:
            rrsep = [0.001, 0.003]
        if trange is None:
            trange = [0.8, 0.9]

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
            tmp = np.nanmean(_LambdaN.data()[_idx, :], axis=0)
            tmpstd = np.nanstd(_LambdaN.data()[_idx, :], axis=0)
            _rdx = np.where(((xCl >= rrsep[0]) & (xCl <= rrsep[1])))[0]
            Lambda = np.mean(tmp[_rdx], weights=1. / tmpstd[_rdx])
            LambdaErr = np.std(tmp[_rdx], weights=1. / tmpstd[_rdx])
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
            _idx = np.where(
                ((LpN.getDimensionAt(0).data() >= data.tmin - 0.06) &
                 (LpN.getDimensionAt(0).data() <= data.tmax + 0.06)))[0]
            # average over time
            tmp = np.nanmean(LpN.data()[_idx, :], axis=0)
            # average over distance from the separatrix
            _rdx = np.where(
                ((LpN.getDimensionAt(1).data() >= data.RrsepMin) &
                 (LpN.getDimensionAt(1).data() <= data.RrsepMax)))[0]
            Lpar = np.mean(tmp[_rdx])
            dLpar = np.std(tmp[_rdx])
            _dBlob = data.FWHM * np.sqrt(
                np.power(data.vrExB, 2) +
                np.power(data.vpol3, 2))
            _numerator = _dBlob * np.power(0.88, 1. / 5)
            _denominator = (np.power(Lpar, 2. / 5) *
                            np.power(data.rhos, 4. / 5.))
            Theta = np.power(_numerator / _denominator, 5. / 2.)
            dTheta = Theta * np.sqrt(
                (data.FWHMerr / data.FWHM) ** 2 +
                (data.vrExBerr / data.vrExB) ** 2 +
                (data.dvpol3 / data.vpol3) ** 2 +
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
        and including only the fluctuation between 2.5 sigma of
        the Isat amplitude

        """

        signal = data.sel(sig='Is') - data.sel(sig='Is').min()
        spline = UnivariateSpline(
            data.t, signal.values - signal.max().item() / 2., s=0)
        # find the roots and the closest roots to 0
        roots = spline.roots()
        tmin = roots[roots < 0][-1] * 2
        try:
            tmax = roots[roots > 0][0] * 2
        except:
            tmax = 2.5e-5
        # now the fluctuations of the Epol
        ii = np.where((data.t.values >= tmin) & (data.t.values <= tmax))[0]
        # recompute Epol from CAS floating potential with an appropriate
        # smoothing otherwise we have too noisy signal
        _Epol = (data.sel(sig='VFT_' + str(int(self.plunge))) -
                 data.sel(sig='VFM_' + str(int(self.plunge)))) / 4e-3
        _Epol = bottleneck.move_mean(_Epol, window=10)
        Epol = np.abs(_Epol[ii].max() -
                      _Epol[ii].min())
        Erad = np.abs(data.sel(sig='Erad')[ii].max().item() -
                      data.sel(sig='Erad')[ii].min().item())
        EpolErr = np.mean(data.err[1, ii])
        EradErr = np.mean(data.err[2, ii])
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

    def bw_filter(self, data, freq, fs, ty, order=5):
        ny = 0.5 * fs
        if np.size(freq) == 1:
            fr = freq / ny
            b, a = signal.butter(order, fr, btype=ty)
        else:
            frL = freq[0] / ny
            frH = freq[1] / ny
            b, a = signal.butter(order, [frL, frH], btype=ty)
        y = signal.filtfilt(b, a, data)
        return y

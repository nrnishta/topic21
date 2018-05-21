import dd
from scipy.interpolate import UnivariateSpline
import numpy as np
from scipy.signal import savgol_filter
from time_series_tools import identify_bursts2
import matplotlib as mpl
import eqtools
import logging


class Libes(object):
    """
    Class to deal with Li-beam data. It compute the Li-Beam on
    an uniform rho-Grid and time,  and save as attributes
    the value normalize to the value at the separatrix
    
    """

    def __init__(self, shot, trange=None):
        self.shot = shot
        try:
            LiBD = dd.shotfile('LIN', shot, experiment='AUGD')
        except:
            LiBD = dd.shotfile('LIN', shot, experiment='LIBE')
        ne = LiBD('ne').data
        time = LiBD('ne').time
        rhoP = LiBD('ne').area        
        LiBD.close
        # limit to the part in trange if given
        if trange:
            _idx = np.where(((time >= trange[0]) & (time <= trange[1])))[0]
            ne = ne[_idx, :]
            time = time[_idx]
            rhoP = rhoP[_idx, :]
        # load the equilibrium
        self.eq= eqtools.AUGDDData(self.shot)
        self.eq.remapLCFS()
        # now build your own basis on the same rhoPoloidal
        # outsize the separatrix
        self.neraw = ne[:, ::-1]
        self.rhoraw = rhoP.data[:, ::-1]
        self.rho = np.linspace(0.9, 1.1, 50)
        self.ne = np.zeros((time.size, self.rho.size))
        self.neNorm = np.zeros((time.size, self.rho.size))
        self.time = time
        # now compute the UnivariateSpline
        for t in range(time.size):
            S = UnivariateSpline(self.rhoraw[t, :],
                                 self.neraw[t, :], s=0)
            self.ne[t, :] = S(self.rho)
            self.neNorm[t, :] = S(self.rho)/S(1)


    def averageProfile(self, trange=[2, 3], interelm=False,
                       elm=False, **kwargs):

        _idx = np.where((self.time >= trange[0]) &
                        (self.time <= trange[1]))[0]

        ne = self.ne[_idx, :]
        neN = self.neNorm[_idx, :]
        if interelm:
            logging.warning('Computing inter-ELM profiles')
            self._maskElm(trange=trange, **kwargs)
            ne = ne[self._interElm, :]
            neN = neN[self._interElm, :]
        if elm:
            self._maskElm(trange=trange, **kwargs)
            logging.warning('Computing ELM profiles')
            ne = ne[self._Elm, :]
            neN = neN[self._Elm, :]

        profiles = np.nanmean(ne, axis=0)
        error = np.nanstd(ne, axis=0)
        profilesN = np.nanmean(neN, axis=0)
        errorN = np.nanstd(neN, axis=0)
        # now build the appropriate Efolding length on the
        # recomputed profiles using an UnivariateSpline
        # interpolation weighted on the error
        Rmid = self.eq.rho2rho('sqrtpsinorm', 'Rmid', self.rho, self.time[_idx].mean())
        # 
        S = UnivariateSpline(Rmid, profiles, w=1./error, s=0)
        Efold = np.abs(S(Rmid)/S.derivative()(Rmid))
        return profiles, error, Efold, profilesN, errorN

    def amplitudeShoulder(self, dt=0.1, reference=[1.5, 1.6],
                          start=1.6, **kwargs):
        """
        Compute the amplitude and location of the shoulder defined as
        the difference between normalized profiles in the SOL w.r.t.
        a reference profile. It also compute the location in term of
        normalized poloidal flux and Rmid of the maximum of the shoulder

        Parameters
        ----------
        dt : floating
            resolution in [s]. If given compute the evolution with the
            given time resolution

        reference : 2D floating 
            It contains the time interval computed for the reference
            profile

        start : floating
            Starting point for the evaluation of the amplitude 

        kwargs : 
            These are the keyowords which can be passed to average profiles
            in order to eventually compute the interELM behavior

        Returns
        -------
        None

        Attributes
        ----------
        Define the following attributes to the class
        Amplitude :
            Amplitude as normalized difference with respect to the reference profile
            computed during the time interval defined in reference
        Location :
            The Location as a function of rho of the maximum of the normalized difference
        Efold :
            Efolding as a function of rho and time
        rhoAmplitude:
            The rho values of the normalized difference (defined for rho >= 1)
        
        """
        # first compute the normalized reference profile
        _, _, _, pN, eN = self.averageProfile(trange=reference, **kwargs)
        # limit to the region for rho > 1
        _idx = np.where((self.time > start))[0]
        _npoint = int(np.floor((self.time[_idx].max()-self.time[_idx].min())/dt))
        # we use array split
        _dummy = self.neNorm[np.where(self.time > start)[0], :]
        Split = np.asarray(np.array_split(_dummy, _npoint, axis=0))
        Amplitude = np.asarray([np.nanmean(p, axis=0)-pN for p in Split])
        self.timeAmplitude = np.nanmean(
            np.asarray(np.array_split(self.time[_idx], _npoint)))
        self.Amplitude = Amplitude[:, np.where(self.rho>1)]
        self.Location = np.max(self.Amplitude, axis=1)
        self.rhoAmplitude = self.rho[self.rho>=1]

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
            logging.warning("Using ELM dda")
            ELM = dd.shotfile("ELM", self.shot, experiment='AUGD')
            elmd = ELM("t_endELM", tBegin=trange[0], tEnd=trange[1])
            # limit to the ELM included in the trange
            _idx = np.where((elmd.time>= trange[0]) & (elmd.time <= trange[1]))[0]
            self.tBegElm = eldm.time[_idx]
            self.tEndElm = elmd.data[_idx]
            ELM.close()
        else:
            logging.warning("Using IpolSolI")
            Mac = dd.shotfile("MAC", self.shot, experiment='AUGD')
            Ipol = Mac('Ipolsoli')
            _idxT = np.where(((Ipol.time >= trange[0]) & (Ipol.time <= trange[1])))[0]
            # now create an appropriate savgolfile
            IpolS = savgol_filter(Ipol.data[_idxT], 301, 3)
            IpolT = Ipol.time[_idxT]
            IpolO = Ipol.data[_idxT]
            # we generate an UnivariateSpline object
            _dummyTime = self.time[np.where(
                (self.time >= trange[0]) &
                (self.time <= trange[1]))[0]]
            IpolSp = UnivariateSpline(IpolT, IpolS, s=0)(_dummyTime)
            # on these we choose a threshold
            # which can be set as also set as keyword
            self._Elm = np.where(IpolSp > threshold)
            # generate a fake interval
            ElmMask = np.zeros(IpolSp.size,dtype='bool')
            ElmMask[self._Elm] = True
            self._interElm = np.where(ElmMask == False)[0]
            if check:
                fig, ax = mpl.pylab.subplots(nrows=1, ncols=1, figsize=(6, 4))
                fig.subplots_adjust(bottom=0.15, left=0.15)
                ax.plot(IpolT, IpolO, color='gray',alpha=0.5)
                ax.plot(IpolT, IpolS, 'k',lw=1.2, alpha=0.5)
                ax.plot(_dummyTime[self._interElm],IpolSp[self._interElm],'g',lw=1.5)
                ax.set_xlabel(r't[s]')
                ax.set_ylabel(r'Ipol SOL I')
                ax.axhline(threshold, ls='--', color='#d62728')

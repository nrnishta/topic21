import dd
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import savgol_filter
from time_series_tools import identify_bursts2
import matplotlib as mpl
import eqtools


class Libes(object):
    """
    Class 
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
        if interelm:
            print('Computing inter-ELM profiles')
            self._maskElm(trange=trange, **kwargs)
            ne = ne[self._interElm, :]
        if elm:
            self._maskElm(trange=trange, **kwargs)
            print('Computing ELM profiles')
            ne = ne[self._Elm, :]

        profiles = np.nanmean(ne, axis=0)
        error = np.nanstd(ne, axis=0)
        # now build the appropriate Efolding length on the
        # recomputed profiles using an UnivariateSpline
        # interpolation weighted on the error
        Rmid = self.eq.rho2rho('sqrtpsinorm', 'Rmid', self.rho, self.time[_idx].mean())
        # 
        S = UnivariateSpline(Rmid, profiles, w=1./error, s=0)
        Efold = np.abs(S(Rmid)/S.derivative()(Rmid))
        return profiles, error, Efold


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
            _idxT = np.where(((Ipol.time >= trange[0]) & (Ipol.time <= trange[1])))[0]
            # now create an appropriate savgolfile
            IpolS = savgol_filter(Ipol.data[_idxT], 501, 3)
            IpolT = Ipol.time[_idxT]
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
            ax.plot(Ipol.time[_idxT], Ipol.data[_idxT], color='gray', alpha=0.3)
            ax.set_xlabel(r't[s]')
            ax.set_ylabel(r'Ipol SOL I')
            ax.axhline(threshold, ls='--', color='#d62728')
            for _ti, _te in zip(self.tBegElm, self.tEndElm):
                ax.axvline(_ti, ls='--', color='#ff7f0e')
                ax.axvline(_te, ls='--', color='#ff7f0e')

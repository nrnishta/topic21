import dd
import map_equ
import dd
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib as mpl


class Target(object):

    def __init__(self, shot):

        self.shot = shot
        self._loadeq()
        # defining the positions of the probes
        self.OuterTarget = {
            'ua1': {'R': 1.602, 'z': -1.212, 's': 1.045},
            'ua2': {'R': 1.606, 'z': -1.187, 's': 1.070},
            'ua3': {'R': 1.610, 'z': -1.164, 's': 1.094},
            'ua4': {'R': 1.615, 'z': -1.132, 's': 1.126},
            'ua5': {'R': 1.620, 'z': -1.100, 's': 1.158},
            'ua6': {'R': 1.625, 'z': -1.070, 's': 1.189},
            'ua7': {'R': 1.628, 'z': -1.046, 's': 1.213},
            'ua8': {'R': 1.634, 'z': -1.013, 's': 1.246},
            'ua9': {'R': 1.640, 'z': -0.984, 's': 1.276}}

        self.OuterLimiter = {
            'uaa': {'R': 1.658, 'z': -0.938, 's': 1.326},
            'uab': {'R': 1.672, 'z': -0.911, 's': 1.356},
            'uac': {'R': 1.688, 'z': -0.885, 's': 1.387},
            'uad': {'R': 1.729, 'z': -0.843, 's': 1.447},
            'uae': {'R': 1.785, 'z': -0.789, 's': 1.526},
            'uaf': {'R': 1.856, 'z': -0.708, 's': 1.634},
            'uag': {'R': 1.898, 'z': -0.661, 's': 1.696},
            'uah': {'R': 1.962, 'z': -0.588, 's': 1.793}}

        self.InnerTarget={
            'ui9': {'R': 1.288, 'z': -0.962, 's': 0.339},
            'ui8': {'R': 1.280, 'z': -0.995, 's': 0.373},
            'ui7': {'R': 1.274, 'z': -1.013, 's': 0.391},
            'ui6': {'R': 1.268, 'z': -1.031, 's': 0.411},
            'ui5': {'R': 1.262, 'z': -1.049, 's': 0.429},
            'ui4': {'R': 1.255, 'z': -1.067, 's': 0.448},
            'ui3': {'R': 1.249, 'z': -1.085, 's': 0.468},
            'ui2': {'R': 1.243, 'z': -1.103, 's': 0.486},
            'ui1': {'R': 1.237, 'z': -1.121, 's': 0.505}}

        self.InnerLimiter = {
            'uig': {'R': 1.152, 'z': -0.672, 's': 0.012},
            'uif': {'R': 1.177, 'z': -0.711, 's': 0.059},
            'uif': {'R': 1.177, 'z': -0.711, 's': 0.059},
            'uie': {'R': 1.198, 'z': -0.745, 's': 0.099},
            'uid': {'R': 1.233, 'z': -0.798, 's': 0.163},
            'uic': {'R': 1.266, 'z': -0.857, 's': 0.230},
            'uib': {'R': 1.281, 'z': -0.903, 's': 0.278},
            'uia': {'R': 1.286, 'z': -0.932, 's': 0.309}}
        self.Dome = {
            'um1': {'R': 1.295, 'z': -1.086, 's': 0.617},
            'um2': {'R': 1.343, 'z': -1.062, 's': 0.673}, 
            'um3': {'R': 1.371, 'z': -1.062, 's': 0.701},
            'um4': {'R': 1.402, 'z': -1.062, 's': 0.732},
            'um5': {'R': 1.428, 'z': -1.062, 's': 0.757},
            'um6': {'R': 1.456, 'z': -1.062, 's': 0.786},
            'um7': {'R': 1.507, 'z': -1.117, 's': 0.862},
            'um8': {'R': 1.535, 'z': -1.151, 's': 0.906}}
        
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
        _idx = np.argmin(np.abs(self.tEq-t0))
        # plot the 
        fig, ax = mpl.pylab.subplots(figsize=(6, 8), nrows=1, ncols=1)
        for key in self.rg.iterkeys():
            ax.plot(self.rg[key], self.zg[key], 'k')
        # now contour of the equilibrium
        # core
        ax.contour(self.R, self.Z, self.psiN[:, :, _idx].transpose(),
                   np.linspace(0.01, 0.95, num=9), colors='k', linestyle='-')
        ax.contour(self.R, self.Z, self.psiN[:, :, _idx].transpose(), [1],
                   colors='r', linestyle='-', linewidths=2)
        # SOL
        ax.contour(self.R, self.Z, self.psiN[:, :, _idx].transpose(),
                   np.linspace(1.01, 1.16, num=5), colors='k', linestyles='--')
        ax.set_xlabel(r'R [m]')
        ax.set_ylabel(r'Z [m]')
        ax.set_title(r'# %5i' % self.shot + ' @ t = %3.2f' % self.tEq[_idx])
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
        # define quantities
        self.psi = self.Eqm.pfm
        self.tEq = self.Eqm.t_eq
        nr = self.psi.shape[0]
        nz = self.psi.shape[1]
        self.R = self.Eqm.Rmesh       
        self.Z   = self.Eqm.Zmesh      
        self.psi_axis = self.Eqm.psi0
        self.psi_bnd  = self.Eqm.psix
        # get the fpol in similar way
        # as done in eqtools
        self.jpol = self.Eqm.jpol
        # these are the lower xpoints
        self.rxpl = self.Eqm.ssq['Rxpu']       
        self.zxpl = self.Eqm.ssq['Zxpu']       
        # read also the upper xpoint
        self.rxpu = self.Eqm.ssq['Rxpo']
        self.zxpu = self.Eqm.ssq['Zxpo']
        # R magnetic axis
        self.axisr = self.Eqm.ssq['Rmag']          
        # Z magnetic axis
        self.axisz = self.Eqm.ssq['Zmag'] 
        # eqm does not load the RBphi on axis
        Mai = dd.shotfile('MAI', self.shot)
        self.Rcent = 1.65
        # we want to interpolate on the same time basis
        Spl = UnivariateSpline(Mai('BTF').time, Mai('BTF').data, s=0)
        self.bphi = Spl(self.tEq)*self.Rcent
        Mai.close()
        Mag = dd.shotfile('MAG', self.shot)
        Spl = UnivariateSpline(Mag('Ipa').time, Mag('Ipa').data, s=0)
        self.cplasma = Spl(self.tEq)
        # now define the psiN
        self.psiN = (self.psi-self.psi_axis[np.newaxis, np.newaxis, :])/ \
            (self.psi_bnd[np.newaxis, np.newaxis, :]-self.psi_axis[np.newaxis, np.newaxis, :])

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
                      interelm=False, elm=False, **kwargs):
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
                        (self.time <= trange[1]))
        t = self.time[_idx]
        sig = sig[:, _idx]
        rho = rho[:, _idx]

        if interelm:
            self._maskElm(usedda=True, trange=trange, check=True)
            t = t[self._interElm]
            sig = sig[:, self._interElm]
            rho = rho[:, self._interElm]
        if elm:
            self._maskElm(usedda=True, trange=trange, check=True)
            t = t[self._Elm]
            sig = sig[:, self._Elm]
            rho = rho[:, self._Elm]

        fig, ax = mpl.pylab.subplots(figsize=(7, 5), nrows=1, ncols=1)
        ax.plot(rho, sig, 'ok', alpha=0.5, mec='white')
        ax.plot(np.nanmean(rho, axis=1), np.nanmean(sig, axis=1),
                'sc', ms=10, mec='white')
        ax.errorbar(np.nanmean(rho, axis=1), np.nanmean(sig, axis=1),
                    yerr=np.nanstd(sig, axis=1), fmt='none', ecolor='c')
        

        return np.nanmean(rho, axis=1), np.nanmean(sig, axis=1), np.nanstd(sig, axis=1)

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
            elmd = ELM("t_endELM", tBegin=ti, tEnd=tf)
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

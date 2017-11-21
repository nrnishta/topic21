import timeseries
import dd
import eqtools
import numpy as np
import matplotlib as mpl
import copy
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from time_series_tools import identify_bursts2
import langmuir
import xarray as xray
import libes


class Filaments(object):
    """
    Class for analysis of Filaments from the HFF probe on the
    midplane manipulator

    Parameters
    ----------
    Shot : :obj: `int`
        Shot number
    Xprobe : :obj: `float`
        Probe starting point decided on a shot to shot basis during
        the radial scan
    Probe: :obj: `string`
        String indicating the probe head used. Possible values are
        - `HFF`: for High Heat Flux probe for fluctuation measurements
        - 'IPP': Innsbruck-Padua probe head

    Requirements
    ------------
    timeseries : Class in https://github.com/nicolavianello/topic21
    dd : Class on AUG toks cluster for AUG signal reading
    eqtools : https://github.com/PSFCPlasmaTools/eqtools
    libes : Class for dealing with Li-Be data available https://github.com/nicolavianello/topic21
    """

    def __init__(self, shot, Probe='HFF', Xprobe=None):
        self.shot = shot
        # load the equilibria
        try:
            self.Eq = eqtools.AUGDDData(self.shot)
        except BaseException:
            print('Equilibrium not loaded')
        self.Xprobe = Xprobe
        # open the shot file
        self.Probe = Probe
        if self.Probe == 'HFF':
            self._HHFGeometry(angle=80)
            self._loadHHF()
        else:
            print('Other probe head not implemented yet')
        # load the data from Langmuir probes
        self.Target = langmuir.Target(self.shot)
        # load the profile of the Li-Beam so that
        # we can evaluate the Efolding length
        try:
            self.LiB = libes.Libes(self.shot)
            self._tagLiB = True
        except:
            self._tabLiB = False


    def _HHFGeometry(self, angle=80.):
        """
        Define the dictionary containing the geometrical information concerning
        each of the probe tip of the HHF probe
        """
        self.Zmem = 312
        self.Xlim = 1738
        RZgrid = {'m01': {'x': 10, 'z': 16.5, 'r': 4},
                  'm02': {'x': 3.5, 'z': 16.5, 'r': 4},
                  'm03': {'x': -3.5, 'z': 16.5, 'r': 4},
                  'm04': {'x': -10, 'z': 16.5, 'r': 4},
                  'm05': {'x': 10, 'z': 5.5, 'r': 0},
                  'm06': {'x': 3.5, 'z': 5.5, 'r': 0},
                  'm07': {'x': -3.5, 'z': 5.5, 'r': 0},
                  'm08': {'x': -10, 'z': 5.5, 'r': 0},
                  'm09': {'x': 10, 'z': -5.5, 'r': 4},
                  'm10': {'x': 3.5, 'z': -5.5, 'r': 2},
                  'm11': {'x': -3.5, 'z': -5.5, 'r': 2},
                  'm12': {'x': -10, 'z': -5.5, 'r': 2},
                  'm13': {'x': 10, 'z': -16.5, 'r': 8},
                  'm14': {'x': 3.5, 'z': -16.5, 'r': 8},
                  'm15': {'x': -3.5, 'z': -16.5, 'r': 8},
                  'm16': {'x': -10, 'z': -16.5, 'r': 8}}
        self.RZgrid = {}
        for probe in RZgrid.iterkeys():
            x, y = self._rotate(
                (RZgrid[probe]['x'], RZgrid[probe]['z']), np.radians(angle))
            self.RZgrid[probe] = {
                'r': RZgrid[probe]['r'],
                'z': self.Zmem + y,
                'x': x}

    def _loadHHF(self):
        """
        Load the data for the HHF probe by reading the Shotfile
        and distinguishing all the saved them as dictionary for
        different type of signal including their geometrical
        information

        """

        # open the MHC experiment
        Mhc = dd.shotfile('MHC', self.shot)
        Mhg = dd.shotfile('MHG', self.shot)
        # read the names of all the signal in Mhc
        namesC = Mhc.getObjectNames()
        namesG = Mhg.getObjectNames()
        # save a list with the names of the acquired ion
        # saturation current
        self.vfArr = {}
        self.isArr = {}
        for n in namesC.itervalues():
            if n[:6] == 'Isat_m':
                self.isArr[n] = dict([('data', -Mhc(n).data),
                                      ('t', Mhc(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z']),
                                      ('x', self.RZgrid[n[-3:]]['x']),
                                      ('name', n)])
            elif n[:5] == 'Ufl_m':
                self.vfArr[n] = dict([('data', Mhc(n).data),
                                      ('t', Mhc(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z']),
                                      ('x', self.RZgrid[n[-3:]]['x']),
                                      ('name', n)])

        for n in namesG.itervalues():
            if n[:6] == 'Isat_m':
                self.isArr[n] = dict([('data', -Mhg(n).data),
                                      ('t', Mhg(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z']),
                                      ('x', self.RZgrid[n[-3:]]['x']),
                                      ('name', n)])
            elif n[:5] == 'Ufl_m':
                self.vfArr[n] = dict([('data', Mhg(n).data),
                                      ('t', Mhg(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z']),
                                      ('x', self.RZgrid[n[-3:]]['x']),
                                      ('name', n)])

        # generale also the list of names for is and vf
        self.isName = []
        for p in self.isArr.itervalues():
            self.isName.append(p['name'])
        self.vfName = []
        for p in self.vfArr.itervalues():
            self.vfName.append(p['name'])

        Mhc.close()
        Mhg.close()
        # generate a class aware time basis
        self._timebasis = self.vfArr[self.vfName[0]]['t']
        # generate a class aware dt
        self.dt = (self._timebasis.max()-self._timebasis.min())/(
            self._timebasis.size-1)

    def plotProbeSetup(self, save=False):
        """
        Method to plot probe head with color code according to
        the type of measurement existing

        Parameters
        ----------
        save : Boolean
            If True save a pdf file with the probe configuration in the
            working directory. Default is False

        """

        ProbeHead = mpl.pyplot.Circle((0, 0), 32.5, ec='k', fill=False, lw=3)

        fig, ax = mpl.pylab.subplots(figsize=(6, 6), nrows=1, ncols=1)
        ax.add_artist(ProbeHead)
        for probe in self.RZgrid.iterkeys():
            if 'Isat_' + probe in self.isArr.keys():
                col = 'red'
            elif 'Ufl_' + probe in self.vfArr.keys():
                col = 'blue'
            else:
                col = 'black'

            tip = mpl.pyplot.Circle(
                (self.RZgrid[probe]['x'], self.RZgrid[probe]['z'] - self.Zmem), 2, fc=col)
            ax.add_artist(tip)
            ax.text(
                self.RZgrid[probe]['x'] -
                10,
                self.RZgrid[probe]['z'] -
                2 -
                self.Zmem,
                probe,
                fontsize=8)
        ax.set_xlim([-70, 70])
        ax.set_ylim([-70, 70])
        ax.text(-40, 60, r'I$_s$', fontsize=18, color='red')
        ax.text(-40, 50, r'V$_f$', fontsize=18, color='blue')

    def loadPosition(self, trange=None):
        """
        Load the position and compute for each of the pin the
        corresponding rho values taking into account the
        (R, Z) position

        Parameters
        ----------
        None

        Returns
        -------
        None

        Attributes
        ----------
        Add to the self.RZgrid dictionary the time basis of the rhopoloidal
        and the corresponding values of rhop

        """

        Lsm = dd.shotfile('LSM', self.shot)
        sPos = np.abs(Lsm('S-posi').data - Lsm('S-posi').data.min())
        tPos = Lsm('S-posi').time
        # convert into absolute value according to transformation
        R = (2188 - (self.Xprobe - self.Xlim) - sPos + 100) / 1e3
        # smooth it
        R = self.smooth(R, window_len=100)
        # check if trange exist or not
        if not trange:
            # convert in Rhopoloidal
        
            self.rhoProbe = np.zeros(R.size)
            for r, t, i in zip(R, tPos, range(R.size)):
                self.rhoProbe[i]=self.Eq.rz2psinorm(r, self.Zmem*1e-3, t, sqrt=True)
            self.tPos = tPos
        else:
            _idx = np.where(((tPos >= trange[0]) & (tPos <= trange[1])))[0]
            _R = R[_idx]
            _t = tPos[_idx]
            self.rhoProbe = np.zeros(_R.size)
            for r, t, i in zip(_R, _t, range(_R.size)):
                self.rhoProbe[i]=self.Eq.rz2psinorm(r, self.Zmem*1e-3, t, sqrt=True)
            self.tPos = _t

    def blobAnalysis(self, Probe='Isat_m01', trange=[2, 3],
                     interELM=False, block=[0.015, 0.12],
                     usedda=False, threshold=3000,
                     otherProbe=['Isat_m10', 'Isat_m14', 'Isat_m07'],
                     **kwargs):
        """
        Given the probe call the appropriate timeseries class
        for the analysis of the blobs.

        Parameters
        ----------
        Probe : :obj: `string`
            Probe name used for the analysis and eventually the
            trigger. If the name is not included among the possible
            available names print the list of available signal
            and ask for inserting the appropriate name

        trange : :obj: list
            List of the type [tmin, tmax]

        interElm : :obj: Boolean
             If true create an appropriate mask for the
             signal keeping only the interELM values (Not yet
             implemented)
        """

        # firs of all limit the isAt and vfFloat to the desired time interval
        isSignal, vfSignal = self._defineTime(trange=trange)

        self.blockmin = block[0]
        self.blockmax = block[1]
        if Probe not in self.isName + self.vfName:
            print('Available Ion saturation current signals are')
            for p in self.isName:
                print p
            print('Available Floating potential signal are')
            for p in self.vfName:
                print p

            Probe = str(raw_input('Provide the probe '))

        if Probe[:4] == 'Isat':
            # for the ion saturation current
            # we need to mask for the arcless system
            # and propagate the mask for

            _idx = np.where((isSignal[Probe]['data'] > self.blockmin) &
                         (isSignal[Probe]['data'] < self.blockmax))[0]
            dt = (isSignal[Probe]['t'].max() - isSignal[Probe]['t'].min()) / (
                isSignal[Probe]['t'].size - 1)
            if interELM:
                self._maskElm(threshold=threshold, usedda=usedda,
                              trange=trange)            
                # now we need to combine the inter ELM mask and the
                # mask for arcless
                self._idx = _idx[np.in1d(_idx, self._interElm, assume_unique=True)]
            else:
                self._idx = _idx
            # we need to generate a dummy time basis
            tDummy = np.arange(np.count_nonzero(self._idx)) * dt + trange[0]

            self.blob = timeseries.Timeseries(
                isSignal[Probe]['data'][self._idx], tDummy)
            self.refSignal = Probe
        else:
            self.blob = timeseries.Timeseries(
                vfSignal[Probe]['data'], isSignal[Probe]['t'])
            self.refSignal = Probe
            self._idx = None
        # to have also the values of the signals in the considered
        # interval for al the collected signal
        self.iSChunck = isSignal
        self.vFChunck = vfSignal

        # this is the cas multiple
        sigIn = []
        for p in otherProbe:
            if p[:4] == 'Isat':
                sigIn.append(self.iSChunck[p]['data'][self._idx])
            else:
                sigIn.append(self.vFChunck[p]['data'][self._idx])
        self._sigIn = np.asarray(sigIn)
        cs, tau, err, amp = self.blob.casMultiple(self._sigIn, **kwargs)
        # now build the xarray used as output
        names = np.append(Probe, otherProbe)
        data = xray.DataArray(cs, coords=[names, tau], dims=['sig', 't'])
        data.attrs['err'] = err
        data.attrs['Amp'] = err
        # start adding the interesting quantities
        delta, errDelta = self._computeDeltaT(tau, cs[0, :], err[0, :])
        data.attrs['TauB'] = delta
        data.attrs['TauBErr'] = errDelta
        # compute the lags and save as attributes. 
        data.attrs['Lags'] = self._computeLag(cs)
        # now we need to compute the lambda through the langmuir class
        # since we are using small intervals in the analysis
        # we 
        if interELM:
            rhoLambda, Lambda = self.Target.computeLambda(
                Type='OuterTarget',
                trange=[trange[0]-0.01, trange[1]+0.01],
                interelm=True, threshold=threshold)
        else:
            rhoLambda, Lambda = self.Target.computeLambda(
                Type='OuterTarget',
                trange=[trange[0]-0.01, trange[1]+0.01])
        # compute the lambda corresponding to the chosen probe
        self.loadPosition(trange=trange)
        # determine the rho corresponding to the Probe
        data.attrs['Rho'] = np.nanmean(self.rhoProbe)
        S = UnivariateSpline(rhoLambda, Lambda, s=0)
        data.attrs['Lambda'] = S(1.03)
        data.attrs['rhoLambda'] = rhoLambda
        data.attrs['LambdaProfile'] = Lambda
        # save also 
        if interELM:
            data.attrs['ELMThreshold'] = threshold

        if self._tagLiB:
            if interElm:
                p, ep, efold = self.LiB.averageProfile(
                    trange=trange, interelm=True, threshold=threshold)
            else:
                p, ep, efold = self.LiB.averageProfile(
                    trange=trange)
            data.attrs['LiB'] = p
            data.attrs['LiBerr'] = ep
            data.attrs['rhoLiB'] = self.LiB.rho
            data.attrs['Efold'] = efold

        return data

    def _defineTime(self, trange=[2, 3]):
        """
        Internal use to limit to a given time interval

        Parameters
        ----------
        trange : :obj: `ndarray`
            2D element of the form [tmin, tmax]

        Returns
        -------
        isOut : Dictionary
            Contains all the ion saturation
            currents in the limited time interval

        vfOut : Dictionary
            Contains all the floating potential
            in the limited time interval
        """

        isOut = copy.deepcopy(self.isArr)
        vfOut = copy.deepcopy(self.vfArr)
        for p in isOut.itervalues():
            _idx = ((p['t'] >= trange[0]) & (p['t'] <= trange[1]))
            p['data'] = p['data'][_idx]
            p['t'] = p['t'][_idx]
        for p in vfOut.itervalues():
            _idx = ((p['t'] >= trange[0]) & (p['t'] <= trange[1]))
            p['data'] = p['data'][_idx]
            p['t'] = p['t'][_idx]
        return isOut, vfOut

    def _rotate(self, point, angle):
        """
        Provide rotation of a point in a plane with respect to origin
        """
        px, py = point
        qx = np.cos(angle) * px - py * np.sin(angle)
        qy = np.sin(angle) * px + np.cos(angle) * py
        return qx, qy

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
        _dummyTime = self._timebasis[np.where((self._timebasis >= trange[0]) &
                                              (self._timebasis <= trange[1]))[0]]

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


    def smooth(self, x, window_len=10, window='hanning'):
        """smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.

        input:
            x: the input signal 
            window_len: the dimension of the smoothing window
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.

        output:
            the smoothed signal

        example:

        import numpy as np    
        t = np.linspace(-2,2,0.1)
        x = np.sin(t)+np.random.randn(len(t))*0.1
        y = smooth(x)

        see also: 

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

        TODO: the window parameter could be the window itself if an array instead of a string   
        """

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
        #print(len(s))

        if window == 'flat': #moving average
            w = np.ones(window_len,'d')
        else:
            w = getattr(np, window)(window_len)
        y = np.convolve(w/w.sum(), s, mode='same')
        return y[window_len-1:-window_len+1]

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

    def _computeLag(self, data):
        """
        Given the output of conditional average compute the
        time lag corresponding to the maximum correlation
        of each of the structure with respect to the first

        Parameters
        ----------
        data : ndarray
            ndarray of shape (#sig, #tau) with the results
            of conditional average on multiple signal

        Results
        -------
        ndarray of lenght (#sig-1) with the values of
        Tau Lag

        """
        lag = np.arange(data.shape[1]) - data.shape[1]/2
        tau = np.zeros(data.shape[0]-1)
        for i in range(data.shape[0]-1):
            xcor = np.correlate(data[i+1, :], data[0, :], mode='same')
            xcor /= np.sqrt(np.dot(data[i+1, :], data[i+1, :]) *
                            np.dot(data[0, :], data[0, :]))
            tau[i-1] = lag[xcor.argmax()]*self.dt
        return tau

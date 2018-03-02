from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import str
from builtins import zip
from builtins import input
from builtins import range
from builtins import object
from past.utils import old_div
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
import logging
from six.moves import input
from lmfit.models import GaussianModel
import time
from collections import OrderedDict


# noinspection PyDefaultArgument,PyAttributeOutsideInit
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
        - '14Pin' : 14 Pin probe head

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
            start = time.time()
            self.Eq = eqtools.AUGDDData(self.shot)
            print('Equilibrium loaded in %5.4f' % (time.time() - start) + ' s')
        except BaseException:
            print('Equilibrium not loaded')
        self.Xprobe = Xprobe
        # open the shot file
        self.Probe = Probe
        if self.Probe == 'HFF':
            self._HHFGeometry(angle=80)
            start = time.time()
            self._loadHHF()
            print('Probe signal loaded in %5.4f' % (time.time() - start) + ' s')
        else:
            logging.warning('Other probe head not implemented yet')
        # load the data from Langmuir probes
        self.Target = langmuir.Target(self.shot)
        # load the profile of the Li-Beam so that
        # we can evaluate the Efolding length
        try:
            start = time.time()
            self.LiB = libes.Libes(self.shot)
            self._tagLiB = True
            print('LiB loaded in %5.4f' % (time.time() - start) + ' s')
        except:
            self._tagLiB = False
            logging.warning('Li-Beam not found for shot %5i' % shot)

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
        for probe in list(RZgrid.keys()):
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
        # now save as xarray with positions
        # as attributes
        vfArr = []
        vfName = []
        isArr = []
        isName = []
        vpArr = []
        vpName = []
        for n in list(namesC.values()):
            if n[:6] == 'Isat_m':
                isArr.append(-Mhc(n).data)
                isName.append(n)
            elif n[:5] == 'Ufl_m':
                vfArr.append(Mhc(n).data)
                vfName.append(n)
            elif n[:6] == 'Usat_m':
                vpArr.append(Mhc(n).data)
                vpName.append(n)

        for n in list(namesG.values()):
            if n[:6] == 'Isat_m':
                isArr.append(-Mhg(n).data)
                isName.append(n)
            elif n[:5] == 'Ufl_m':
                vfArr.append(Mhg(n).data)
                vfName.append(n)
            elif n[:6] == 'Usat_m':
                vpArr.append(Mhg(n).data)
                vpName.append(n)

        # generale also the list of names for is and vf
        self.isName = isName
        self.vfName = vfName
        self.vpName = vpName
        # read the appropriate time basis. They have the same time basis
        t = Mhc(isName[0]).time
        Mhc.close()
        Mhg.close()
        self.isArr = xray.DataArray(np.asarray(isArr),
                                    coords=[isName, t],
                                    dims=['Probe', 't'])
        # add as attributes the coordinates of the single tips
        posDictionary = {}
        for p in self.isArr.Probe.values:
            posDictionary[p] = OrderedDict([('r', self.RZgrid[p[-3:]]['r']),
                                            ('z', self.RZgrid[p[-3:]]['z']),
                                            ('x', self.RZgrid[p[-3:]]['x'])])
        self.isArr.attrs['Grid'] = posDictionary
        # repeat for the floating potential
        self.vfArr = xray.DataArray(np.asarray(vfArr),
                                    coords=[vfName, t],
                                    dims=['Probe', 't'])
        # add as attributes the coordinates of the single tips
        posDictionary = {}
        for p in self.vfArr.Probe.values:
            posDictionary[p] = OrderedDict([('r', self.RZgrid[p[-3:]]['r']),
                                            ('z', self.RZgrid[p[-3:]]['z']),
                                            ('x', self.RZgrid[p[-3:]]['x'])])
        self.vfArr.attrs['Grid'] = posDictionary

        # finally for the applied potential
        self.vpArr = xray.DataArray(np.asarray(vpArr),
                                    coords=[vpName, t],
                                    dims=['Probe', 't'])
        # add as attributes the coordinates of the single tips
        posDictionary = {}
        for p in self.vpArr.Probe.values:
            posDictionary[p] = OrderedDict([('r', self.RZgrid[p[-3:]]['r']),
                                            ('z', self.RZgrid[p[-3:]]['z']),
                                            ('x', self.RZgrid[p[-3:]]['x'])])

        self.vpArr.attrs['Grid'] = posDictionary
        # generate a class aware time basis
        self._timebasis = self.vfArr.t.values
        # generate a class aware dt
        self.dt = old_div((self._timebasis.max() - self._timebasis.min()), (
                self._timebasis.size - 1))

    def plotProbeSetup(self):
        """
        Method to plot probe head with color code according to
        the type of measurement existing

        """

        ProbeHead = mpl.pyplot.Circle((0, 0), 32.5, ec='k', fill=False, lw=3)

        fig, ax = mpl.pylab.subplots(figsize=(6, 6), nrows=1, ncols=1)
        ax.add_artist(ProbeHead)
        for probe in list(self.RZgrid.keys()):
            if 'Isat_' + probe in self.isName:
                col = 'red'
            elif 'Ufl_' + probe in self.vfName:
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
        trange
            2d array to eventually load in a given time window

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
        R = old_div((2188 - (self.Xprobe - self.Xlim) - sPos + 100), 1e3)
        # smooth it
        R = self.smooth(R, window_len=100)
        # check if trange exist or not
        if not trange:
            # convert in Rhopoloidal

            self.rhoProbe = np.zeros(R.size)
            for r, t, i in zip(R, tPos, list(range(R.size))):
                self.rhoProbe[i] = self.Eq.rz2psinorm(r, self.Zmem * 1e-3, t, sqrt=True)
            self.tPos = tPos
        else:
            _idx = np.where(((tPos >= trange[0]) & (tPos <= trange[1])))[0]
            _R = R[_idx]
            _t = tPos[_idx]
            self.rhoProbe = np.zeros(_R.size)
            for r, t, i in zip(_R, _t, list(range(_R.size))):
                self.rhoProbe[i] = self.Eq.rz2psinorm(r, self.Zmem * 1e-3, t, sqrt=True)
            self.tPos = _t

    def blobAnalysis(self, Probe='Isat_m01', trange=[2, 3],
                     interELM=False, block=190,
                     usedda=False, threshold=3000,
                     otherProbe=None,
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

        interELM : :obj: Boolean
             If true create an appropriate mask for the
             signal keeping only the interELM values (Not yet
             implemented)
        block
            This is the value used to determine the threshold in the applied
            voltage of the ion saturation current to mask the unuseful time window
            where active arc-preventing system was in operation
        usedda
            Boolean, default is False. If true uses the saved values of ELM in
            shotfile
        otherProbe: List with the names of the other probe used for the computation
            of cross-correlation. If not given it uses all the available floating
            potential and ion saturation current
        threshold
            Indicate the threshold on Ipolsola used to disentangle the ELMs
        """

        # firs of all limit the isAt and vfFloat to the desired time interval
        isSignal = self.isArr.where((self.isArr.t >= trange[0]) &
                                    (self.isArr.t <= trange[1]), drop=True)
        vfSignal = self.vfArr.where((self.vfArr.t >= trange[0]) &
                                    (self.vfArr.t <= trange[1]), drop=True)
        vpSignal = self.vpArr.where((self.vpArr.t >= trange[0]) &
                                    (self.vpArr.t <= trange[1]), drop=True)

        self.block = block
        if Probe not in self.isName + self.vfName:
            print('Available Ion saturation current signals are')
            for p in self.isName:
                print(p)
            print('Available Floating potential signal are')
            for p in self.vfName:
                print(p)
            try:
                Probe = str(input('Provide the probe '))
            except:
                Probe = str(eval(input('Provide the probe')))
        if Probe[:4] == 'Isat':
            # for the ion saturation current
            # we need to mask for the arcless system
            # and propagate the mask for

            _idx = np.where((np.abs(vpSignal.sel(Probe='Usat' + Probe[4:]).values) >= self.block))[0]
            if interELM:
                self._maskElm(threshold=threshold, usedda=usedda,
                              trange=trange)
                # now we need to combine the inter ELM mask and the
                # mask for arcless
                self._idx = _idx[np.in1d(_idx, self._interElm, assume_unique=True)]
            else:
                self._idx = _idx
            # we need to generate a dummy time basis
            tDummy = np.arange(self._idx.size) * self.dt + trange[0]

            self.blob = timeseries.Timeseries(
                isSignal.sel(Probe=Probe).values[self._idx], tDummy)
            self.refSignal = Probe
        else:
            self.blob = timeseries.Timeseries(
                vfSignal.self(Probe=Probe).values, vfSignal.self(Probe=Probe).t.values)
            self.refSignal = Probe
            self._idx = None
        # to have also the values of the signals in the considered
        # interval for al the collected signal
        self.iSChunck = isSignal
        self.vFChunck = vfSignal
        if otherProbe is None:
            # determine the name of isSignal which are different
            # with respect to the probe
            if Probe[:4] == 'Isat':
                _Name = [n for n in self.isName if n != Probe]
                self._sigIn = xray.concat([isSignal.sel(Probe=_Name)[:, self._idx],
                                           vfSignal[:, self._idx]],
                                          dim='Probe')
            else:
                _Name = [n for n in self.vfName if n != Probe]
                self._sigIn = xray.concat([isSignal[:, self._idx],
                                           vfSignal.sel(Probe=_Name)[:, self._idx]],
                                          dim='Probe')
        else:
            if otherProbe[0] in self.isName:
                a = isSignal.sel(Probe=otherProbe[0])
            else:
                a = vfSignal.sel(Probe=otherProbe[0])
            for p in otherProbe[1:]:
                if p in self.isName:
                    a = xray.concat([a, isSignal.sel(Probe=p)], dim='Probe')
                else:
                    a = xray.concat([a, vfSignal.sel(Probe=p)], dim='Probe')
            self._sigIn = a
        cs, tau, err, amp = self.blob.casMultiple(self._sigIn.values, **kwargs)
        # now build the xarray used as output
        data = xray.DataArray(cs,
                              coords=[
                                  np.insert(self._sigIn.Probe.values, 1, Probe),
                                  tau],
                              dims=['sig', 't'])
        data.attrs['err'] = err
        data.attrs['Amp'] = err
        # start adding the interesting quantities
        delta, errDelta = self._computeDeltaT(tau, cs[0, :], err[0, :])
        data.attrs['TauB'] = delta
        data.attrs['TauBErr'] = errDelta
        # compute the lags and save as attributes. 
        LagsD = self._computeLag(data)
        # since we want to have the possibility to save in netcdf
        # we can't use OrderedDictionary and we need to save them explicitely
        LagsN = np.array([])
        LagsV = np.array([])
        LagsE = np.array([])
        for n in LagsD.keys():
            LagsN = np.append(LagsN,n)
            LagsV = np.append(LagsV,LagsD[n]['tau'])
            LagsE = np.append(LagsE,LagsD[n]['err'])
        data.attrs['LagTimeNames'] = LagsN
        data.attrs['LagTimeValues'] = LagsV
        data.attrs['LagTimeErr'] = LagsE
        # given the geometry of the probe head we can now evaluate the
        # appropriate velocity using binormal velocity estimate done
        # accordingly to Carralero NF 2014. This should be a probe head aware
        # method since the other
        if 'HFF' == self.Probe:
            VperpD = self._computeVperp(
                LagsD,
                probeTheta='Isat_m07',
                probeR='Isat_m10')
            # now save the appropriate values in the DataArray. We can't save
            # as OrderedDictionary since otherwise we can't save as NETcdf
            data.attrs['Vperp'] = VperpD['vperp']['value']
            data.attrs['VperpErr'] = VperpD['vperp']['err']
            data.attrs['Vr'] = VperpD['vr']['value']
            data.attrs['VrErr'] = VperpD['vr']['err']
            data.attrs['Vz'] = VperpD['vz']['value']
            data.attrs['VzErr'] = VperpD['vz']['err']
        # now we need to compute the lambda through the langmuir class
        # since we are using small intervals in the analysis
        # we 
        if interELM:
            rhoLambda, Lambda = self.Target.computeLambda(
                Type='OuterTarget',
                trange=[trange[0] - 0.01, trange[1] + 0.01],
                interelm=True, threshold=threshold)
        else:
            rhoLambda, Lambda = self.Target.computeLambda(
                Type='OuterTarget',
                trange=[trange[0] - 0.01, trange[1] + 0.01])
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
            if interELM:
                p, ep, efold, pN, eN = self.LiB.averageProfile(
                    trange=trange, interelm=True, threshold=threshold)
            else:
                p, ep, efold, pN, eN = self.LiB.averageProfile(
                    trange=trange)
            data.attrs['LiB'] = p
            data.attrs['LiBerr'] = ep
            data.attrs['LiBN'] = pN
            data.attrs['LiBNerr'] = eN
            data.attrs['rhoLiB'] = self.LiB.rho
            data.attrs['Efold'] = efold

        return data

    def _rotate(self, point, angle):
        """
        Provide rotation of a point in a plane with respect to origin
        """
        px, py = point
        qx = np.cos(angle) * px - py * np.sin(angle)
        qy = np.sin(angle) * px + np.cos(angle) * py
        return qx, qy

    # noinspection PyDefaultArgument
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
            elmd = ELM("t_endELM", tBegin=ti, tEnd=tf)
            # limit to the ELM included in the trange
            _idx = np.where((elmd.time >= trange[0]) & (elmd.time <= trange[1]))[0]
            self.tBegElm = eldm.time[_idx]
            self.tEndElm = elmd.data[_idx]
            ELM.close()
        else:
            logging.warning("Using IpolSolI")
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
            _idx, _idy = list(zip(*window))
            self.tBegElm = IpolT[np.asarray(_idx)]
            self.tEndElm = IpolT[np.asarray(_idy)]
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

        # and now set the mask
        _dummyTime = self._timebasis[np.where((self._timebasis >= trange[0]) &
                                              (self._timebasis <= trange[1]))[0]]

        self._interElm = []
        self._Elm = []
        for i in range(self.tBegElm.size):
            _a = np.where((_dummyTime >= self.tBegElm[i]) &
                          (_dummyTime <= self.tEndElm[i]))[0]
            self._Elm.append(_a[:])
            try:
                _a = np.where((_dummyTime >= self.tEndElm[i]) &
                              (_dummyTime <= self.tBegElm[i + 1]))[0]
                self._interElm.append(_a[:])
            except:
                pass

        self._interElm = np.concatenate(np.asarray(self._interElm))
        self._Elm = np.concatenate(np.asarray(self._Elm))

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
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = getattr(np, window)(window_len)
        y = np.convolve(old_div(w, w.sum()), s, mode='same')
        return y[window_len - 1:-window_len + 1]

    def _computeDeltaT(self, x, y, e):
        """
        useful to compute the timing and error on the Isat
        conditional average sample
        """
        _dummy = (y - y.min())
        spline = UnivariateSpline(x, _dummy - old_div(_dummy.max(), 2), s=0)
        if spline.roots().size > 2:
            a = np.sort(spline.roots())
            r1 = a[a < 0][-1]
            r2 = a[a > 0][0]
        else:
            r1, r2 = spline.roots()
        delta = (r2 - r1)
        # now compute an estimate of the error
        _dummy = (y + e) - (y + e).min()
        spline = UnivariateSpline(x, _dummy - old_div(_dummy.max(), 2), s=0)
        if spline.roots().size > 2:
            a = np.sort(spline.roots())
            r1 = a[a < 0][-1]
            r2 = a[a > 0][0]
        else:
            r1, r2 = spline.roots()
        deltaUp = (r2 - r1)
        _dummy = (y - e) - (y - e).min()
        spline = UnivariateSpline(x, _dummy - old_div(_dummy.max(), 2), s=0)
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
        of each of the structure with respect to the first. In order
        increase resolution we make a gaussian fit of the cross-correlation
        function in analogy to what done for TCV using the
        lmfit class and determine the center of the gaussian

        Parameters
        ----------
        data
            xarray DataArray containing the saved Conditional Average structure
        Returns
        -------
        Dictionary with keys indicating the names of the signals for which cross-correlation
        are saved and reporting correlation and error
        """

        lag = np.arange(data.shape[1], dtype='float') - old_div(data.shape[1], 2)
        lag *= self.dt
        outDictionary = {}
        _Name = [n for n in data.sig.values if n != self.refSignal]
        for n in _Name:
            a = data.sel(sig=n).values
            b = data.sel(sig=self.refSignal).values
            xcor = np.correlate(a,
                                b,
                                mode='same')
            xcor /= np.sqrt(np.dot(a, a) *
                            np.dot(b, b))
            mod = GaussianModel()
            pars = mod.guess(xcor, x=lag)
            pars['sigma'].set(value=1e-5, vary=True)
            out = mod.fit(xcor, pars, x=lag)
            outDictionary[n + '-' + self.refSignal] = {'tau': out.params['center'].value,
                                                       'err': out.params['center'].stderr}

        return outDictionary

    def _computeVperp(self, Lags, probeTheta='Isat_m07', probeR='Isat_m02'):
        """

        Parameters
        ----------
        Lags
            This is the OrderedDictionary obtained from _computeLags
        probeTheta
            Name of the probe used for the poloidal cross-correlation
        probeR
            Name of the probe used for the radial cross-correlation
        Returns
        -------
        Dictionary with values and error for vr, vz, vperp

        """
        Lz = 1e-3 * np.abs(self.RZgrid[probeTheta[-3:]]['z'] -
                           self.RZgrid[self.refSignal[-3:]]['z'])
        Lr = 1e-3 * np.abs(self.RZgrid[probeR[-3:]]['r'] -
                           self.RZgrid[self.refSignal[-3:]]['r'])
        Lzr = 1e-3 * np.abs(self.RZgrid[probeR[-3:]]['z'] -
                            self.RZgrid[self.refSignal[-3:]]['z'])
        # now identify the correct time delay
        tZ = Lags[probeTheta + '-' + self.refSignal]['tau']
        tR = Lags[probeR + '-' + self.refSignal]['tau']
        # and the corresponding errors
        dtZ = Lags[probeTheta + '-' + self.refSignal]['err']
        dtR = Lags[probeR + '-' + self.refSignal]['err']

        alpha = np.arctan(-Lr * tZ / (Lzr * tZ - Lz * tR))
        vperp = Lz / tZ * np.sin(alpha)
        vr = vperp * np.cos(alpha)
        vz = vperp * np.sin(alpha)
        # error propagation
        # Error on alpha
        _xdummy = (-Lr * tZ / (Lzr * tZ - tR * Lz))
        dxdummy_dtz = (
                old_div(-Lr, (Lzr * tZ - tR * Lz)) + Lr * Lzr * tZ / np.power(Lzr * tZ - tR * Lz, 2)
        )
        dxdummy_dtr = (
                Lr * Lz * tZ / np.power(Lzr * tZ - tR * Lz, 2)
        )
        dxdummy = np.sqrt(np.power(dxdummy_dtz, 2) * np.power(dtZ, 2) +
                          np.power(dxdummy_dtr, 2) * np.power(dtR, 2))
        dAlpha = old_div(dxdummy, (1 + np.power(_xdummy, 2)))
        # Error on vPerp
        dvperp = np.sqrt(np.power(old_div(vperp, tZ), 2) * np.power(dtZ, 2) +
                         np.power(Lz * np.cos(alpha) / tZ, 2) * np.power(dAlpha, 2)
                         )
        # error in the evaluation of vr
        dvr = np.sqrt(np.power(np.cos(alpha), 2) * np.power(dvperp, 2) +
                      np.power(vperp * np.sin(alpha), 2) * np.power(dAlpha, 2))
        # error in the evaluation of vz
        dvz = np.sqrt(np.power(np.sin(alpha), 2) * np.power(dvperp, 2) +
                      np.power(vperp * np.cos(alpha), 2) * np.power(dAlpha, 2))

        out = OrderedDict([('vperp', {'value': vperp, 'err': dvperp}),
                           ('vz', {'value': vz, 'err': dvz}),
                           ('vr', {'value': vr, 'err': dvr}),
                           ('alpha', {'value': alpha, 'err': dAlpha})])
        return out

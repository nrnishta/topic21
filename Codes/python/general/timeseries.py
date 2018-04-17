from __future__ import print_function

__author__ = 'N. Vianello and N. Walkden'
__version__ = '0.2'
__data__ = '19.04.2017'

import numpy as np
import scipy
import pycwt as wav
import astropy.stats as Astats
from scipy.interpolate import UnivariateSpline
from scipy import signal
import copy
import bottleneck


class Timeseries(object):
    def __init__(self, signal, time, dtS=0.0002):
        """

        Parameters
        ----------
        signal : ndarray
            signal to be analyzed
        time : ndarray
            time basis
        dtS : floating
            At the init we also compute a normalize
            signal where normalization is of the form
            (x-<x>)/std(x) where the mean and average
            is a rolling mean and standard deviation on
            a window of the time dtS/dt
        Dependences
        -----------
            numpy
            scipy
            pycwt https://github.com/regeirk/pycwt.git
            astropy for better histogram function
            bottleneck (https://pypi.python.org/pypi/Bottleneck)
            for moving average
        """

        self.sig = copy.deepcopy(signal)
        self.time = copy.deepcopy(time)
        self.dt = (self.time.max() - self.time.min()) / (self.time.size - 1)
        self.nsamp = self.time.size
        self.signorm = (self.sig - self.sig.mean()) / self.sig.std()
        # since the moments of the signal are
        # foundamental quantities we compute them
        # at the initial
        self.moments()
        _nPoint = int(dtS / self.dt)
        self.rmsnorm = (
                           self.sig -
                           bottleneck.move_mean(self.sig, window=_nPoint)) / \
                       bottleneck.move_std(self.sig, window=_nPoint)

    def moments(self):
        """
        Compute moments of the signal
        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if successful, False otherwise.

        Attribute
        ---------
        Define the following attributes
        mean : mean
        skewness : skewness
        kurtosis : Kurtosis
        flatness : Flatnes
        variance : Variance with ddof = 0
        act : autocorrelation time
        """
        from scipy.stats import describe
        try:
            nobs, minmax, mean, var, skew, kurt = describe(
                self.sig, nan_policy='omit')
            self.mean = mean
            self.skewness = skew
            self.kurtosis = kurt
            self.flatness = self.kurtosis + 3
            self.variance = var
            self.acf()
            return True
        except BaseException:
            return False

    def identify_bursts(self, thresh, rmsNorm=False):
        """
        Identify the windows in the time series where the signal
        is above a given threshold

        Parameters
        ----------
        thresh : float
            Threshold value
        rmsNorm : Boolean default False
            if set to True it normalize the signal in as
            If set the reference signal is normalized
            as (x-<x>)/rms(x) where the mean and the rms are
            moving average and rms on a dtS

        Returns
        -------
        nbursts: int
            Number of bursts identified
        ratio: float
            Ratio between number of samples and number of bursts
        avwin: float
            mean window size of the burst
        windows: tuple
            a list of tuples that contains indices that
            bracket each of the burst detected
        """
        if not rmsNorm:
            crossings = np.where(np.diff(np.signbit(self.sig - thresh)))[0]
        else:
            crossings = np.where(np.diff(np.signbit(self.rmsnorm - thresh)))[0]
        windows = list(zip(crossings[::2], crossings[1::2] + 1))
        nbursts = len(windows)
        ratio = float(self.nsamp) / nbursts
        avwin = np.mean([y - x for x, y in windows])
        return nbursts, ratio, avwin, windows

    def limStructure(self, frequency=100e3,
                     wavelet='Mexican',
                     peaks=False, valleys=False):
        """
        Determination of the time location of the intermittent
        structure accordingly to the method defined in
        *M. Onorato et al Phys. Rev. E 61, 1447 (2000)*

        Parameters
        ----------
        frequency : :obj: `float`
            Fourier frequency considered for the analysis
        wavelet : :obj: `string`
            Mother wavelet for the continuous wavelet analysis
            possibilityes are *Mexican [default]*,  *DOG1* or
            *Morlet*
        peaks : :obj: `Boolean`
            if set it computes the structure only for the peaks
            Default is False
        valleys : :obj: `Boolean`
            if set it computes the structure only for the valleys
            Default is False

        Returns
        -------
        maxima : :obj: `ndarray`
           A binary array equal to 1 at the identification of
           the structure (local maxima)
        allmax : :obj: `ndarray`
            A binary array equal to 1 in all the region where the
            signal is above the threshold

        Attributes
        ----------
        scale : :obj: `float`
            Corresponding scale for the chosen wavelet

        """
        if wavelet == 'Mexican':
            self.mother = wav.MexicanHat()
        elif wavelet == 'DOG1':
            self.mother = wav.DOG(m=1)
        elif wavelet == 'Morlet':
            self.mother = wav.Morlet()
        else:
            print('Not a valid wavelet using Mexican')
            self.mother = wav.Mexican_hat()

        self.freq = frequency
        self.scale = 1. / self.mother.flambda() / self.freq

        # compute the continuous wavelet transform
        wt, sc, freqs, coi, fft, fftfreqs = wav.cwt(
            self.sig, self.dt, 0.25, self.scale, 0, self.mother)
        wt = np.real(np.squeeze(wt))
        wtOr = wt.copy()
        # normalization
        wt = (wt - wt.mean()) / wt.std()
        self.lim = np.squeeze(
            np.abs(wt ** 2) / np.mean(wt ** 2))
        flatness = self.flatness
        newflat = flatness
        threshold = 20.
        while newflat >= 3.05 and threshold > 0:
            threshold -= 0.2
            d_ev = (self.lim > threshold)
            count = np.count_nonzero(d_ev)
            if 0 < count < self.lim.size:
                newflat = np.mean(wt[~d_ev] ** 4) / \
                          np.mean(wt[~d_ev] ** 2) ** 2

        # now we have identified the threshold
        # we need to find the maximum above the treshold
        maxima = np.zeros(self.sig.size)
        allmax = np.zeros(self.sig.size)
        allmax[(self.lim > threshold)] = 1
        imin = 0
        for i in range(maxima.size - 1):
            i += 1
            if self.lim[i] >= threshold > self.lim[i - 1]:
                imin = i
            if self.lim[i] < threshold <= self.lim[i - 1]:
                imax = i - 1
                if imax == imin:
                    d = 0
                else:
                    d = self.lim[imin: imax].argmax()
                maxima[imin + d] = 1

        if peaks:
            ddPeak = ((maxima == 1) & (wtOr > 0))
            maxima[~ddPeak] = 0
        if valleys:
            ddPeak = ((maxima == 1) & (wtOr < 0))
            maxima[~ddPeak] = 0
        return maxima, allmax

    def casMultiple(
            self,
            inputS,
            **kwargs):
        """
        Conditional average sampling on multiple signals

        Parameters
        ----------
        inputS : :obj: `ndarray`
            Array of signals to be analyzed (not the one already used)
            in the form (#sign, #sample)
        All the other parameters coincide with the ones defined in the
        *cas* method
        Type :obj: `str`
            String indicating the method used to identify the structure.
            Can be the 'LIM' method or the standard 'THRESHOLD' method
        **kwargs
           These are the same as defined in the cas method

        Results
        -------
        cs : :obj: `ndarray`
            Conditional Sampled Structures (nw, #signal).
            **In case normalize is True it is
            not in physical unit**
        tau : :obj: `ndarray`
            Appropriate time basis for the CAS
        err : :obj: `ndarray`
            Standard error
        ampTot :obj: `ndarray`
            Array of the form (#signal, #events) where the amplitude
            with respect to the mean of each of the signal in the time
            window around the event are given
        """
        if 'detrend' in kwargs:
            detrend = kwargs['detrend']
        else:
            detrend = False
        if 'normalize' in kwargs:
            normalize = kwargs['normalize']
        else:
            normalize = False
        if 'rmsNorm' in kwargs:
            rmsNorm = kwargs['rmsNorm']
        else:
            rmsNorm = False
        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            if rmsNorm:
                threshold = 3
 #               print('Threshold is 3 in rmsNormalized')
            else:
                threshold = 3 * np.sqrt(self.variance) + self.mean
 #               print('Threshold is 3 sigma in signal not normalized')
        if 'nw' in kwargs:
            nw = kwargs['nw']
        else:
            nw = 501
        if 'Type' in kwargs:
            Type = kwargs['Type']
        else:
            Type='THRESHOLD'

        nSig = inputS.shape[0]
        if Type == 'LIM':
            if 'frequency' in kwargs:
                frequency = kwargs['frequency']
            else:
                frequency = 100e3
            if 'wavelet' in kwargs:
                wavelet = kwargs['wavelet']
            else:
                wavelet = 'Mexican'
            if 'peaks' in kwargs:
                peaks = kwargs['peaks']
            else:
                peaks = False
            if 'valleys' in kwargs:
                valleys = kwargs['valleys']
            else:
                valleys = False
            csO, tau, errO = self.cas(
                Type='LIM',
                frequency=frequency,
                wavelet= wavelet,
                peaks=peaks,valleys=valleys,
                detrend=detrend)
        else:
            csO, tau, errO = self.cas(
                Type='THRESHOLD',
                normalize=normalize, detrend=detrend,
                rmsNorm=rmsNorm, threshold=threshold,nw=nw)
        maxima = np.zeros(self.nsamp, dtype='intp')
        maxima[self._locationindex] = 1
        # we need to ensure that we have 0 up to iwin
        maxima[-self.iwin-1:] = 0
        maxima[:self.iwin]=0
        csTot = np.zeros((nSig + 1, self.nw,
                          maxima.sum()))
        print('Number of structure mediated %4i' % maxima.sum())
        d_ev = np.asarray(np.where(maxima >= 1)[0])
        ampTot = np.zeros((nSig + 1, int(maxima.sum())))
        for i in range(d_ev.size):
            for n in range(nSig):
                dummy = inputS[n,
                        d_ev[i] - self.iwin:d_ev[i] + self.iwin + 1]
                if detrend:
                    dummy = scipy.signal.detrend(dummy, type='linear')
                else:
                    dummy -= dummy.mean()
                ampTot[n + 1, i] = dummy[
                                   int(self.iwin / 2.): int(3. * self.iwin / 2.)].max() - \
                                   dummy[int(self.iwin / 2):
                                   int(3 * self.iwin / 2)].min()
                if normalize:
                    dummy /= dummy.std()
                csTot[n + 1, :, i] = dummy
            # add also the amplitude of the reference signal
            dummy = copy.deepcopy(self.sig)[
                    d_ev[i] - self.iwin:
                    d_ev[i] + self.iwin + 1]

            if detrend:
                dummy = scipy.signal.detrend(dummy, type='linear')
            else:
                dummy -= dummy.mean()
            ampTot[0, i] = dummy[
                            int(self.iwin / 2): 3 * int(self.iwin / 2)].max() - \
                            dummy[int(self.iwin / 2):
                            int(3 * self.iwin / 2)].min()
        # now compute the cas
        cs = np.mean(csTot, axis=2)
        cs[0, :] = csO
        err = scipy.stats.sem(csTot, axis=2)
        err[0, :] = errO
        return cs, tau, err, ampTot

    def waitingTimes(self, Type='LIM', **kwargs):
        """
        Waiting time distribution for events identified with different
        methods

        Parameters
        ----------
        Type :obj: `str`
            String indicating the method used to identify the structure.
            Can be the 'LIM' method or the standard 'THRESHOLD' method
        **kwargs
           These are the same as defined in the *limStructure* method or
           *identify_bursts* method according to the method used

        Results
        -------
        wt : :obj: `ndarray`
            Waiting times
        qt : :obj: `ndarray`
            Quiescent time. The quiescent time is computed
            differently according to
            the definition of Sanchez.

        """

        if Type == 'LIM':
            csO, tau, errO = self.cas(
                Type='LIM', **kwargs)
        else:
            csO, tau, errO = self.cas(
                Type='THRESHOLD', **kwargs)
        # in this case
        # compute maxima derivative
        maxima = np.zeros(self.nsamp)
        maxima[self._locationindex] = 1
        dEv = np.diff(maxima)
        # starting and ending time
        ti = np.asarray(np.where(dEv > 0))[0][:]
        te = np.asarray(np.where(dEv < 0))[0][:]
        # check if they have the sime number
        if ti.size != te.size:
            mn = np.minimum(ti.size, te.size)
            ti = ti[0: mn]
            te = te[0: mn]

        if ((ti.size > 1) & (te.size > 1)):
            if (ti[0] < te[0]):
                waiting_times = ti[1:] - te[0:- 1]
            else:
                waiting_times = ti[:] - te[:]

        # repeat for the computation of the quiescent_times
        dEv = np.diff(self.__allmaxima)
        # starting and ending time
        ti = np.asarray(np.where(dEv > 0))[0][:]
        te = np.asarray(np.where(dEv < 0))[0][:]
        # check if they have the sime number
        if ti.size != te.size:
            mn = np.minimum(ti.size, te.size)
            ti = ti[0: mn]
            te = te[0: mn]

        if ((ti.size > 1) & (te.size > 1)):
            if (ti[0] < te[0]):
                quiescent_times = ti[1:] - te[0:- 1]
            else:
                quiescent_times = ti[:] - te[:]

        return waiting_times * self.dt, quiescent_times * self.dt

    def cas(
            self,
            Type='LIM',
            nw=None,
            **kwargs):
        """
        Conditional Average Sampling

        Parameters
        ----------
        Type :obj: `str`
            String indicating the method used to identify the structure.
            Can be the 'LIM' method or the standard 'THRESHOLD' method
        nw : :obj: `int`
            dimension of the window for the CAS. Can be optional if type
            is LIM. In this case it is assumed 6 times the corresponding
            wavelet structure
        **kwargs
           These are the same as defined in the *limStructure* method or
           *identify_bursts* method according to the method used

        Results
        -------
        cs : :obj: `ndarray`
            Conditional Sampled Structure. **In case normalize is True it is
            not in physical unit**
        tau : :obj: `ndarray`
            Appropriate time basis for the CAS
        err : :obj: `ndarray`
            Standard error

        Attributes
        ----------
        location : :obj: 'ndarray'
            Time location where the structure are determined

        """
        cut = False
        if 'detrend' in kwargs:
            detrend = kwargs['detrend']
        else:
            detrend = False
        if 'normalize' in kwargs:
            normalize = kwargs['normalize']
        else:
            normalize = False
        if 'rmsNorm' in kwargs:
            rmsNorm = kwargs['rmsNorm']
        else:
            rmsNorm = False

        if Type == 'LIM':
            peaks = kwargs.get('peaks', False)
            valleys = kwargs.get('valleys', False)
            frequency = kwargs.get('frequency',100e3)
            maxima, allmax = self.limStructure(
                frequency=frequency,
                peaks=peaks,
                valleys=valleys,
                wavelet=kwargs.get('wavelet','Mexican'))
            self.location = self.time[maxima == 1]
            self._locationindex = np.where(maxima == 1)[0]
            self._allmaxima = allmax
            if nw is None:
                nw = np.int(np.round(2. / self.freq / self.dt))
            if nw % 2 == 0:
                iwin = nw / 2
                nw += 1
            else:
                iwin = (nw - 1) / 2
            iwin = np.int(iwin)
            maxima[0: iwin - 1] = 0
            maxima[-iwin:] = 0
            print('Number of structures mediated %4i' % maxima.sum())
            csTot = np.ones((int(nw), np.sum(maxima, dtype='int')))
            d_ev = np.asarray(np.where(maxima >= 1))
            for i in range(d_ev.size):
                _dummy = self.sig[
                         d_ev[0][i] -
                         iwin: d_ev[0][i] +
                               iwin +
                               1]
                if detrend:
                    _dummy = scipy.signal.detrend(
                        _dummy, type='linear')
                _dummy -= _dummy.mean()
                if normalize is True:
                    _dummy /= _dummy.std()

                csTot[:, i] = _dummy

        else:
            if 'threshold' in kwargs:
                thresh = kwargs['threshold']
            else:
                if rmsNorm:
                    thresh = 3
                    print('Threshold is 3 in rmsNormalized')
                else:
                    thresh = 3 * np.sqrt(self.variance) + self.mean
                    print('Threshold is 3 sigma in signal not normalized')
            Nbursts, ratio, av_width, windows = self.identify_bursts(
                thresh, rmsNorm=rmsNorm)

            if nw is None:
                print('Window length not set assumed 501 points')
                nw = 501
            if nw % 2 == 0:
                nw += 1
            csTot = np.ones((nw, Nbursts))
            inds = []
            self.__allmaxima = np.zeros(self.nsamp)
            for window, i in zip(windows, range(Nbursts)):
                self.__allmaxima[window[0]:window[1]] = 1
                ind_max = np.where(
                    self.sig[window[0]:window[1]] ==
                    np.max(self.sig[window[0]:window[1]]))[0][0]
                if ((window[0] + ind_max - (nw - 1) / 2) >= 0) and \
                                (window[0] + ind_max + (nw - 1) / 2 + 1) <= self.nsamp:
                    _dummy = copy.deepcopy(self.sig[
                             window[0] + ind_max - (nw - 1) / 2 :
                             window[0] + ind_max + (nw - 1) / 2 + 1])
                    if detrend:
                        _dummy = scipy.signal.detrend(
                            _dummy, type='linear')
                    _dummy -= _dummy.mean()
                    if normalize:
                        _dummy /= _dummy.std()
                    csTot[:, i] = _dummy
                    inds.append(window[0] + ind_max)
                else:
                    cut = True
            self.location = self.time[inds]
            self._locationindex = inds
        # now compute the cas
        if cut:
            csTot = csTot[:, :-1]
        self.nw = nw
        self.iwin = (nw - 1) / 2
        cs = np.mean(csTot, axis=1)
        tau = np.linspace(- self.iwin, self.iwin, self.nw) * self.dt
        err = scipy.stats.sem(csTot, axis=1)

        return cs, tau, err

    def pdf(self, bins=10, range=None, weights=None, normed=False, **kwargs):
        """
        Computation of the Probability Density function of the signal

        Wrapper around histogram function from astropy.stats package.

        Parameters
        ----------
        normed :obj: `bool`
            If set it compute the PDF of the normalize signal. Default is False

        bins : :obj: `int` or `list` or `str` (optional)
            If bins is a string, then it must be one of:

            - 'blocks' : use bayesian blocks for dynamic bin widths

            - 'knuth' : use Knuth's rule to determine bins

            - 'scott' : use Scott's rule to determine bins

            - 'freedman' : use the Freedman-Diaconis rule to determine bins

        range : tuple or None (optional)
            the minimum and maximum range for the histogram.  If not specified,
            it will be (x.min(), x.max())

        weights : array_like, optional
            Not Implemented

        other keyword arguments are described in numpy.histogram().

        Returns
        -------
        hist : array
            The values of the histogram. See ``normed`` and ``weights`` for a
            description of the possible semantics.
        bin_edges : array of dtype float
            Return the bin edges ``(length(hist)+1)``.

        See Also
        --------
        numpy.histogram
        astropy.stats.histogram
        """
        if normed:
            hist, bins_e = Astats.histogram(
                self.signorm, bins=bins, range=range,
                weights=weights, **kwargs)
        else:
            hist, bins_e = Astats.histogram(
                self.sig, bins=bins, range=range,
                weights=weights, **kwargs)
        return hist, bins_e

    def coarse_grain(self, factor, dx=None, integrate=False):
        """
        # TODO: Insert documentation
        """

        if not integrate:
            new_sig = self.sig[::factor]
        else:
            if factor % 2 == 0:
                nx = factor + 1
            else:
                nx = factor
            new_sig = np.zeros(self.sig[::factor].shape)
            if dx is None:
                dx = np.ones(self.sig.shape) / (factor + 1 * (nx % 2))
            new_sig += (self.sig * dx)[::factor]
            for i in np.int32(np.arange((nx - 1) / 2) + 1):
                new_sig += np.roll(self.sig * dx, i)[::factor]
                new_sig += np.roll(self.sig * dx, -i)[::factor]

        return new_sig

    def signed_diff(self, window):
        """
        # TODO: Insert documentation
        """

        snf = np.zeros(self.signorm.shape[0])
        for i in np.arange(window):
            snf -= np.roll(self.signorm, i) + np.roll(self.signorm, -i)
        snf /= 2 * window
        snf += self.signorm
        return snf

    def significance(self, window):
        """
        # TODO: Insert documentation
        """

        prevnegs = np.zeros(self.nsamp)
        prevposs = np.zeros(self.nsamp)
        snf = np.zeros(self.nsamp)
        for i in np.arange(1, window):
            negs = self.signorm - np.roll(self.signorm, i)
            poss = self.signorm - np.roll(self.signorm, -i)
            prevnegs[np.where(negs > prevnegs)
            ] = negs[np.where(negs > prevnegs)]
            prevposs[np.where(poss > prevposs)
            ] = poss[np.where(poss > prevposs)]
        snf = 0.5 * (prevnegs + prevposs)
        return snf

    def acf(self):
        """
        Compute the autocorrelation function according to
        http://stackoverflow.com/q/14297012/190597
        http://en.wikipedia.org/wiki/Autocorrelation#Estimation

        Parameters
        ----------
        None

        Results
        -------
        Autocorrelation function

        Attributes
        ----------
        Define the autocorrelation time as attribute to the class (self.act)
        computed as the time where the correlation is 1/e the maximum
        value

        """
        n = self.nsamp
        variance = self.variance
        xx = self.sig - self.mean
        r = np.correlate(xx, xx, mode='full')[-n:]
        result = r / (variance * (np.arange(n, 0, -1)))
        # define the lag
        lag = np.arange(result.size, dtype='float') * self.dt
        # interpolate
        S = UnivariateSpline(lag, result - 1. / np.exp(1), s=0)
        self.act = S.roots()[0]
        return result

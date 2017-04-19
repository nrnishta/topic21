__author__ = 'N. Vianello and N. Walkden'
__version__ = '0.2'
__data__ = '19.04.2017'

import numpy as np
import scipy 
import pycwt as wav
import lmfit


class Timeseries:
    """
    Python class for turbulent signal analysis

    The class implement standard and non standard
    methods for turbulent signal analysis

    Parameters
    ----------
    signal : :obj:`ndarray`
      signal to be analyzed
    time  : :obj:`ndarray`
      time basis

    Attributes
    ----------
    dt : Floating. Sampling rate of the signal
    nsamp : :obj:`int`. Size of the signal

    Dependences
    -----------
    numpy
    scipy
    pycwt https://github.com/regeirk/pycwt.git
    lmfit http://lmfit.github.io/lmfit-py/index.html
    future Used by lmfit
    """

    def __init__(self, signal, time):

        self.sig = signal
        self.time = time
        self.dt = (self.time.max()-self.time.min())/(self.time.size-1)
        self.nsamp = self.time.size
        # since the moments of the signal are
        # foundamental quantities we compute them
        # at the initial
        self.moments()

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
            return True
        except:
            return False

    def identify_bursts(self, thresh, analysis=True):
        """
        Identify the windows in the time series where the signal
        is above a given threshold

        Parameters
        ----------
        thresh: :obj: `float`
            Threshold value

        Returns
        -------
        nbursts: :obj: `int`
            Number of bursts identified
        ratio: :obj: `float`
            Ratio between number of samples and number of bursts
        avwin: :obj: `float`
            mean window size of the burst
        windows: :obj: `tuple`
            a list of tuples that contains indices that
            bracket each of the burst detected
        """
    
        crossings = np.where(np.diff(np.signbit(self.sig-thresh)))[0]
        windows = list(zip(crossings[::2], crossings[1::2]+1))
        nbursts = len(windows)
        ratio = len(list(x))/nbursts
        avwin = np.mean([y - x for x,y in windows])
        return nbursts, ratio, avwin, windows

    def limStructure(self, frequency, mother='Mexican',
                     peaks=False, valley=False):
        """
        Determination of the time location of the intermittent
        structure accordingly to the method defined in 
        *M. Onorato et al Phys. Rev. E 61, 1447 (2000)*

        Parameters
        ----------
        frequency : :obj: `float`
            Fourier frequency considered for the analysis
        mother : :obj: `string`
            Mother wavelet for the continuous wavelet analysis
            possibilityes are *Mexican [default]*,  *DOG1* or
            *Morlet*
        peak : :obj: `Boolean`
            if set it computes the structure only for the peaks
            Default is False
        valley : :obj: `Boolean`
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
            self.mother = wav.Mexican_hat()
        elif wavelet == 'DOG1':
            self.mother = wav.DOG(m=1)
        elif wavelet == 'Morlet':
            self.mother = wav.Morlet()
        else:
            print 'Not a valid wavelet using Mexican'
            self.mother = wav.Mexican_hat()
        
        self.freq = frequency
        self.scale = 1. / self.mother.flambda() / self.fr

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
        newflat = flatness.copy()
        threshold = 20.
        while newflat >= 3.05 and threshold > 0:
            threshold -= 0.2
            d_ev = (lim > threshold)
            count = np.count_nonzero(d_ev)
            if count > 0 and count < lim.size:
                newflat = np.mean(wt[~d_ev] ** 4) / \
                    np.mean(wt[~d_ev] ** 2) ** 2

        # now we have identified the threshold
        # we need to find the maximum above the treshold
        maxima = np.zeros(self.sig.size)
        allmax = np.zeros(self.sig.size)
        allmax[(lim > threshold)] = 1
        imin = 0
        for i in range(maxima.size - 1):
            i += 1
            if lim[i] >= threshold and lim[i - 1] < threshold:
                imin = i
            if lim[i] < threshold and lim[i - 1] >= threshold:
                imax = i - 1
                if imax == imin:
                    d = 0
                else:
                    d = lim[imin: imax].argmax()
                maxima[imin + d] = 1

        if self.peak:
            ddPeak = ((maxima == 1) & (wtOr > 0))
            maxima[~ddPeak] = 0
        if self.valley:
            ddPeak = ((maxima == 1) & (wtOr < 0))
            maxima[~ddPeak] = 0
        return maxima, allmax

    def cas(self, Type='LIM', nw=None, detrend=True, **kwargs):
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
        detrend: :obj: `Boolean`
            Boolean to implement linear detrend in each of the time window
            before computing the average. Default is True
        normalize : :obj: `Boolean`
            Boolean to implement normalization (signal-<signal>)/sigma in
            each of the time window before normalization. Otherwise
            in each time window the signal is only mean subtracted.
            Default is False
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
        
        if Type == 'LIM':
            peak = kwargs.get('peak', False)
            valley = kwargs.get('valley', False)
            maxima, allmax = self.limStructure(
                kwargs.get('frequency', 100e3),
                mother=kwargs.get('mother', 'Mexican'),
                peak=peak, valley=valley)
            self.location = self.time[maxima == 1]
            self.__locationindex = (maxima == 1)
            self.__allmaxima = allmax
            if nw is None:
                nw = np.round(1./self.frequency/self.dt)
                if nw % 2 == 0:
                    iwin = nw/2
                    nw += 1
                else:
                    iwin = (nw-1)/2

            maxima[0: iwin - 1] = 0
            maxima[- iwin:] = 0
            print 'Number of structures mediated ' + str(maxima.sum())
            csTot = np.ones((nw, maxima.sum()))
            d_ev = np.asarray(np.where(maxima >= 1))
            for i in range(d_ev.size):
                if detrend is True:
                    _dummy = scipy.signal.detrend(
                        self.sig[
                            d_ev[0][i] -
                            iwin: d_ev[0][i] +
                            iwin +
                            1], type='linear')
                else:
                    _dummy = self.sig[
                            d_ev[0][i] -
                            iwin: d_ev[0][i] +
                            iwin +
                            1]
                _dummy -= _dummy.mean()
                if normalize is True:
                    _dummy /= _dummy.std()
                    
                csTot[:,i] = _dummy

        else: 
            thresh = kwargs.get('thresh', sqrt(self.variance)*2.5)
            if nw is None:
                print('Window length not set assumed 501 points')
                nw = 501
            if nw%2 == 0:
                nw+=1
            windows,Nbursts,ratio,av_width = identify_bursts2(
                self.sig,thresh)
            csTot = np.ones((nw, len(windows))
            inds = []
            self.__allmaxima = np.zeros(self.nsamp)
            for window, i in zip(windows, range(len(windows))):
                self.__allmaxima[window[0]:window[1]] = 1
                ind_max = np.where(
                    self.sig[window[0]:window[1]] ==
                    np.max(self.sig[window[0]:window[1]]))[0][0]

                _dummy = self.sig[window[0] + ind_max - (nw-1)/2 :
                                       window[0] + ind_max + (nw-1)/2 + 1]
                if detrend is True:
                            _dummy = scipy.signal.detrend(
                                _dummy, type='linear')
                _dummy -= _dummy.mean()
                if normalize is True:
                            _dummy /= _dummy.std()
                csTot[:, i] = _dummy
                inds.append(window[0]+ind_max)
            self.location = self.time[inds]
            self.__locationindex = inds
        # now compute the cas
        cs = np.mean(csTot, axis=1)
        tau = np.linspace(- iwin, iwin, 2 * iwin + 1) * self.dt
        err = scipy.stats.sem(csTot, axis=1)
        self.nw = nw
        self.iwin = (nw-1)/2
                            
        return cs, tau, err

    def casMultiple(self,inputS, type='LIM', nw=None, detrend=True, **kwargs):

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
        nw : :obj: `int`
            dimension of the window for the CAS. Can be optional if type
            is LIM. In this case it is assumed 6 times the corresponding
            wavelet structure
        detrend: :obj: `Boolean`
            Boolean to implement linear detrend in each of the time window
            before computing the average. Default is True
        normalize : :obj: `Boolean`
            Boolean to implement normalization (signal-<signal>)/sigma in
            each of the time window before normalization. Otherwise
            in each time window the signal is only mean subtracted.
            Default is False
        **kwargs
           These are the same as defined in the *limStructure* method or
           *identify_bursts* method according to the method used
        
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
        Shape = inputS.shape
        nSig = Shape[0]
        if Type == 'LIM':
            peak = kwargs.get('peak', False)
            valley = kwargs.get('valley', False)
            csO, tau, errO = self.cas(
                Type='LIM',
                kwargs.get('frequency', 100e3),
                mother=kwargs.get('mother', 'Mexican'),
                peak=peak, valley=valley,
                detrend=kwargs.get('detrend', True),
                normalize=kwargs.get('normalize', True))
        else:
            csO, tau, errO = self.cas(
                Type='THRESHOLD',
                nw=kwargs.get('nw', 501),
                thrs=kwargs.get('thresh', sqrt(self.variance)*2.5)
                normalize=kwargs.get('normalize', False),
                detrend=kwargs.get('detrend'), False)                  

        maxima = np.zeros(self.nsamp)
        maxima[self.__locationindex] = 1                
        csTot = np.ones((nSig + 1, self.nw, maxima.sum()))
        d_ev = np.asarray(np.where(maxima >= 1))
        ampTot = np.zeros((nSig + 1, maxima.sum()))
        for i in range(d_ev.size):
                for n in range(nSig):
                    dummy = scipy.signal.detrend(
                        inputS[n, d_ev[0][i] - iwin: d_ev[0][i] + iwin + 1])
                    if detrend is True:
                        dummy = scipy.signal.detrend(dummy, type='linear')
                    dummy -= dummy.mean()
                    ampTot[n + 1, i] = dummy[
                        self.iwin / 2: 3 * self.iwin / 2].max() -
                            dummy[self.iwin / 2:
                                  3 * self.iwin / 2].min()
                    if normalize:
                            dummy -= dummy.std()
                    csTot[n + 1,: , i] = dummy
                    # add also the amplitude of the reference signal
                    dummy = scipy.signal.detrend(
                        self.sig[d_ev[0][i] - iwin: d_ev[0][i] + iwin + 1])
                    if detrend is True:
                        dummy = scipy.signal.detrend(dummy, type='linear')
                    ampTot[0, i] = dummy[
                        self.iwin / 2: 3 * self.iwin / 2].max() -
                            dummy[self.iwin / 2:
                                  3 * self.iwin / 2].min()
        # now compute the cas
        cs = np.mean(csTot, axis = 2)
        cs[0, :] = csO
        err = scipy.stats.sem(csTot, axis = 2)
        err[0, :] = errO
        return cs, tau, err, ampTot


    def waitingTimes(self, Type='LIM', detrend=True, **kwargs):
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
            peak = kwargs.get('peak', False)
            valley = kwargs.get('valley', False)
            csO, tau, errO = self.cas(
                Type='LIM',
                kwargs.get('frequency', 100e3),
                mother=kwargs.get('mother', 'Mexican'),
                peak=peak, valley=valley,
                detrend=kwargs.get('detrend', True),
                normalize=kwargs.get('normalize', True))
        else:
            csO, tau, errO = self.cas(
                Type='THRESHOLD',
                nw=kwargs.get('nw', 501),
                thrs=kwargs.get('thresh', sqrt(self.variance)*2.5)
                normalize=kwargs.get('normalize', False),
                detrend=kwargs.get('detrend'), False)                  

        # in this case
        # calcolo la derivata del maxima
        maxima = np.zeros(self.nsamp)
        maxima[self.__locationindex] = 1                
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

    def castaing(self, x, s0, l, sk, am):
        """
        Define the castaing - like function to fit the Probability
        density function of normalized fluctuations.  The function is intoduced in
        B. Castaing et ak, Physica D: Nonlinear Phenomena 46, 177 (1990) and further developed in
        order to take into account possible asymmetric function in
        L. Sorriso-Valvo, R. Marino, L. Lijoi, S. Perri, and V. Carbone, Astrophys J 807, 86 (2015).

        P(\delta v) =  am / \sqrt(2\pi)\int G_{\lambda}(\sigma)\exp( - \delta v^2 / 2\sigma^2) * (1 +
        a_s\frac{dv / sigma}{np.sqrt{1 + dv^2 / sigma^2}})d\sigma / sigma
        G_{\lambda}(\sigma) =  \frac{1}{\sqrt{2}\lambda}\exp( - \ln^2(\sigma / sigma_0) / 2\lambda)

        Parameters:
           x  =  these are the increments which represents the bins of the PDF of the increments
           s0 =  2.  Is \sigma_0
           l  =  \lambda
           sk = skewness
           am = Amplitude for proper normalization
        out:  the computed function

        """
        # we need this for a proper definition of the integral
        def integrand(sigma, s, lamb, dv, skw):
            return 1. / 2 * np.pi / lamb * np.exp(
                -np.log(sigma / s) ** 2 / 2 / lamb ** 2) * 1. / sigma * np.exp(
                    - dv ** 2 / 2 / sigma ** 2 * (1 + skw * ((dv / sigma) /
                                                             np.sqrt(1. + dv ** 2 / sigma ** 2))))

        # for proper integration in \sigma define
        from scipy.integrate import quad
        cst = np.asarray([ am / np.sqrt(2 * np.pi) *
                           quad(integrand, 0, np.inf,
                                args=(s0, l, x[i], sk))[0] for i in range(x.size)])
        return cst

    def castaingFit(self, **kwargs):
        """
        Perform a fit of the Probability Distribution function based on the Castaing model.
        It has four keyword which can
        can be used and are defined in the function model castaing:
        Input:
        s0 = Sigma_0 in the model, default value is 0.1
        l  = Lambda parameter of the model, default values is 0.1
        am = Amplitude for the correct scaling, default value is 10
        sk = Skewness in the model, default values is the Skewness of the increments of the
        signal computed at given frequency
        Output:
        fit = This method is based on the lmfit package (http://lmfit.github.io/lmfit-py/index.html).
        The result is what is a
        ModelFit class (see http://lmfit.github.io/lmfit-py/model.html#the-modelfit-class).
        To get the parameters of the fit:
        turbo = intermittency.Intermittency(s, dt, 100e3)
        fit = castaingFit()
        s0 = fit.params['s0'].value
        l  = fit.params['l'].value
        am = fit.params['am'].value
        sk = fit.params['sk'].value
        To plot the resulf of the fit
        pdf, x, err = turbo.pdf()
        fit = turbo.castaingFit()
        semilogy(x, pdf, 'o--')
        plot(x, fit.best_fit, ' - ', linewidth = 2)
        """
        s0 = kwargs.get('s0', 0.1)
        l = kwargs.get('l', 1)
        am = kwargs.get('am', 10)
        sk = kwargs.get('sk', scipy.stats.skew(self.cwt()))
        self.xrange = kwargs.get('xrange', [- 4.5, 4.5])
        self.nbins = kwargs.get('nbins', 41)
        # build the appropriateModel
        csMod = lmfit.models.Model(
            self.castaing,
            independent_vars='x',
            param_names=(
                's0',
                'l',
                'sk',
                'am'))
        # initialize the parameters
        pars = csMod.make_params(s0=s0, l=l, sk=sk, am=am)
        pdf, x, err = self.pdf(xrange=self.xrange, nbins=self.nbins)
        fit = csMod.fit(pdf, pars, x=x, weights=1. / err ** 2)
        return fit


    def strFun(self, nMax=7):
        """
        Compute the structure function the given frequency up to a maximum nMax order.  As a default
        the nMax is equal to 7. This would be useful for the ESS analysis introduced in
        R. Benzi et al.  , Phys. Rev. E 48, R29 (1993).
        """
        return np.asarray([np.mean(self.cwt() ** (k + 1))
                           for k in range(nMax)])

    def twoGamma(self, x, c1, n1, b1, c2, n2, b2):
        """
        Define the sum of two gamma function which will be further used for the fit according to the formula
        suggested in F. Sattin et al. PPCF 2006.  It will be depend basically on one single parameter a which
        will contain the variables (C, n, beta, C, n, beta) of the two gamma function (see reference)
        x = these are the bins of the PDF
        a = varabiles of the form (C, N, beta, C, n, beta)
        """

        twoG = self.oneGamma(x, c1, n1, b1) + self.oneGamma(x, c2, n2, b2)
        return twoG

    def oneGamma(self, x, c1, n1, b1):

        return c1 * (n1 * b1) ** (n1) / scipy.special.gamma(n1) * \
            x ** (n1 - 1) * np.exp(- b1 * n1 * x)

    def twoGammaFit(self, normed=False, density=False, **kwargs):
        # first of all compute the pdf with the keyword used
        if normed is True:
            dummy = (self.sig - self.sig.mean()) / self.sig.std()
            rMax = 6
        else:
            dummy = self.sig
            rMax = dummy.mean() + 6 * dummy.std()
#        alpha = kwargs.get('alpha', 1.5)
        nbin = kwargs.get('nbins', 41)
        # compute the PDF limiting to the equivalent of 6 std
        if not density:
            pdfT, binT = np.histogram(
                dummy, bins=nbin, range=[
                    dummy.min(), rMax])
        else:
            pdfT, binT = np.histogram(
                dummy, bins=nbin, density=True, range=[
                    dummy.min(), rMax])

        xpdfT = (binT[:- 1] + binT[1:]) / 2
        # build also the weigths
        err = np.sqrt(pdfT / (np.count_nonzero((dummy <= rMax)))
                      * (xpdfT[1] - xpdfT[0]))

#        from scipy import stats
        area = pdfT.sum() * (xpdfT[1] - xpdfT[0])
        m1 = dummy.mean()
        m2 = scipy.var(dummy)
#        m3 = stats.skew(dummy)
#        m4 = stats.kurtosis(dummy, fisher=True)
        b1 = 1. / m1
        n1 = 1. / (b1 ** 2 * m2)
        c1 = area * b1 * (n1 ** n1) / scipy.special.gamma(n1)
        c2 = 1
        n2 = 0.5
        b2 = 2
        c1 = kwargs.get('c1', c1)
        n1 = kwargs.get('n1', n1)
        b1 = kwargs.get('b1', b1)
        c2 = kwargs.get('c2', c2)
        n2 = kwargs.get('n2', n2)
        b2 = kwargs.get('b2', b2)
        # first of all we build the model for the one gamma function and fi
        oneGMod = lmfit.models.Model(
            self.oneGamma,
            independent_vars='x',
            param_names=(
                'c1',
                'n1',
                'b1'))
        parsOne = oneGMod.make_params(c1=c1, n1=n1, b1=b1)
        fitO = oneGMod.fit(pdfT, parsOne, x=xpdfT, weights=1 / err ** 2)

        # now we can build the model
        twoGMod = lmfit.models.Model(
            self.twoGamma, independent_vars='x',
            param_names=('c1', 'n1', 'b1', 'c2', 'n2', 'b2'))
        pars = twoGMod.make_params(c1=fitO.params['c1'].value,
                                   n1=fitO.params['n1'].value,
                                   b1=fitO.params['b1'].value,
                                   c2=c2, n2=n2, b2=b2)
        fit = twoGMod.fit(pdfT, pars, x=xpdfT, weights=1 / err ** 2)
        return fit, xpdfT, pdfT, fitO

    def levyFunction(self, x, a, mu, norm):
        """
        For the definition of the appropriate levy Function used for
        the fit consider the article F.Lepreti, ApJ
        """
        def integrandI(z, xd, amp, m):
            return 1. / np.pi * np.cos(z * xd) * np.exp(- amp * np.abs(z) ** m)
        from scipy.integrate import quad
        levF = norm * np.asarray([quad(integrandI, 0,
                                       np.inf, args=(x[i], a, mu))[0] for i in range(x.size)])
        return levF

    def levyFit(self, **kwargs):

        """
        Fit the Waiting times (or eventually the quiescent time) distribution with a
        Levy type function as defined in Lepreti et al,  ApJ 55: L133

        Parameters
        ----------
        None. Method working on the defined function

        Keywords
        ----------
        amp = amplitude for the fit. Default is 100
        mu  = See the defintion of the Function. Default is 1
        norm = See the definition of the function. Default is 1e6
        peak = Boolean (default is false). In this way you use the peak in the
               determination of the structure where the waiting times are computed
        valley = Boolean (default is false).In this way you use the peak in the
               determination of the structure where the waiting times are computed
        b99 = Boolean default is True. Compute the threshold according to Boffetta paper
        thr = Floating. If set it gives directly the threshold. If set automatically exclude the B99 method
        resolution = resolution in coars-graining for the loop in determining the threshold
        factor = Default is 2. It is used for the method
        iterM = Maximum possible iteration
        quiet = Boolean. Default is false. If it is True compute the fit on quiescent time
        Returns
        -------
        fit (as output from lmfit),  pdf, bins, err

        Examples
        --------
        """



        amp = kwargs.get('amp', 100.)
        mu = kwargs.get('mu', 1.)
        norm = kwargs.get('norm', 1e6)
        # insert a boolean label so that we can use the levyFit on
        # quiescent_times
        quiet = kwargs.get('qt', False)
        # now we need to perform the Waiting time distribution
        self.peak = kwargs.get('peak', False)
        self.valley = kwargs.get('valley', False)
        b99 = kwargs.get('b99', False)
        thr = kwargs.get('thr', None)
        # these is the resolution in the coars - graining
        resolution = kwargs.get('resolution',
                                (self.sig.max() - self.sig.min()) / 100.)
        # this is a factor used in the b99 method.  Default = 2
        factor = kwargs.get('factor', 2)
        iterM = kwargs.get('iterM', 40)
        wt, at = self.waitingTimes(peak=self.peak, valley=self.valley, b99=b99,
                                   thr=thr, resolution=resolution,
                                   factor=factor, iterM=iterM)

        # this is where you use the boolean
        if quiet:
            print('Levy fit on quiescent times')
            us = at.copy()
        else:
            print('Levy fit on waiting times')
            us = wt.copy()
        # now build the pdf based on a logarithmic scale
        xmin = kwargs.get('xmin', us.min())
        xmax = kwargs.get('xmax', us.max())
        nb = kwargs.get('nbins', 30)
        bb = np.logspace(np.log10(xmin), np.log10(xmax), nb)
        pdf, bins = np.histogram(us, bins=bb, density=True)
        err = np.sqrt(pdf / (us.size * (bins[1] - bins[0])))
        # and now perform the real fit
        levyMod = lmfit.models.Model(self.levyFunction,
                                     independent_vars='x',
                                     param_names=('a', 'mu', 'norm'))
        pars = levyMod.make_params(a=amp, mu=mu, norm=norm)
        pars['mu'].set(min=0, max=2)
        fit = levyMod.fit(pdf / 1e6,
                          pars,
                          x=bins[1:] * 1e6,
                          weights=1 / (err / 1e6)**2)
        return fit, pdf / 1e6, bins * 1e6, err / 1e6

    def powerCutFit(self, **kwargs):
        # now we need to perform the Waiting time distribution
        self.peak = kwargs.get('peak', False)
        self.valley = kwargs.get('valley', False)
        b99 = kwargs.get('b99', False)
        thr = kwargs.get('thr', None)
        resolution = kwargs.get('resolution',
                                (self.sig.max() -
                                 self.sig.min()) / 100.)  # these is the
        # resolution in the coars - graining
        # this is a factor used in the b99 method.
        factor = kwargs.get('factor', 2)
        # Default = 2
        iterM = kwargs.get('iterM', 40)
        quiet = kwargs.get('quiet', False)


        wt, at = self.waitingTimes(peak=self.peak, valley=self.valley, b99=b99,
                                   thr=thr, resolution=resolution,
                                   factor=factor, iterM=iterM)
        if quiet:
            print('Levy fit on quiescent times')
            us = at.copy()
        else:
            print('Levy fit on waiting times')
            us = wt.copy()
        # now build the pdf based on a logarithmic scale
        xmin = kwargs.get('xmin', us.min())
        xmax = kwargs.get('xmax', us.max())
        nb = kwargs.get('nbins', 30)
        bb = np.logspace(np.log10(xmin), np.log10(xmax), nb)
        pdf, bins = np.histogram(us, bins=bb, density=True)
        err = np.sqrt(
            pdf / (np.count_nonzero(((us >= xmin) & (us <= max))) * (bins[1] - bins[0])))
        # now we define the appropriate fitting function

        def powerCut(x, amp, a, tc):
            a *= -1
            return amp * x**a * np.exp(-x / tc)
        # and now we define the appropriate model
        powCutMod = lmfit.models.Model(powerCut, indepentent_vars='x',
                                       param_names=('amp', 'a', 'tc'))
        # now the default parameter choice
        amp = kwargs.get('amp', 1e-2)
        a = kwargs.get('a', 1.5)
        tc = kwargs.get('tc', 200)
        pars = powCutMod.make_params(amp=amp, a=a, tc=tc)
        fit = powCutMod.fit(pdf / 1e6,
                            pars,
                            x=bins[1:] * 1e6,
                            weights=1. / (err / 1e6)**2)
        return fit, pdf / 1e6, bins * 1e6, err / 1e6


    #     fr : Corresponding Fourier frequency for the Wavelet analysis
    # wavelet = String considering the type of wavelet to be used.  Default is 'Mexican',  other choices
    # could be 'DOG1', 'Morlet'
    # Methods: 
    #    # Statistics
    #    cwt: Compute the CWT

    #    lim: Compute the Local Intermittency Measurement
    #         [M. Onorato, R. Camussi, and G. Iuso, Phys. Rev. E 61, 1447 (2000)]

    #    flatness: Compute the flatness at a given scale
    #    # ---------- 
    #    # structures
    #    strTime:  Compute the time occurrence of the Intermittent structure defined in
    #              

    #    threshold:Define the time occurrence of structure above a threshold,
    #              where threshold can be defined by the
    #              user or is the one defined
    #              in G. Boffetta et al, Phys. Rev. Lett. 83, 4662 (1999)

    #    cas: Compute the Conditional average sampling of the signal.
    #         Condition can be the
    #         occurrence of intermittent structure according
    #         to the LIM method, or the threshold method.  It also gives all the amplitude of each structures
    #    casMultiple: Compute the conditional average sampling on an array of signal using
    #         as condition the occurrence of intermittent structure (LIM) or the threshold method
    #         on the classes function.  It also gives all the amplitude of each structures



    #    # ------------ 
    #    # Distribution

    #    waitingTimes: Waiting and activity times computed using
    #         one of the above method (LIM or threshold)

    #    pdf: Probability density function of the normalized fluctuation

    #    castaingFit: Fit of the PDF according to the modified Castaing function

    #    strFun: Compute the structure function up to a given order
    #            (Default is nMax = 7)

    #    pdfAll: Compute the Probability Distribution Function of the entire signal.
    #            Eventually can
    #            be normalized (mean subtracted and std deviation divided)

    #    twoGammaFit: Perform the fit with two gamma function as described in
    #            F. Sattin et al, Plasma Phys. Contr. Fus. 48, 1033 (2006).

    #    levyFit: Perform a fit on the Waiting time distribution
    #             according to the Levy function introduced in Lepreti
    #             et al,  ApJ 55: L133
    #    powerCutFit: Perform a fit on the Waiting time distribution according to
    #             a power law sacling with an exponential cutoff (T_c) as done
    #             in R. D'Amicis, Ann. Geophysics vol 24,  p 2735, 2006

    # def pdf(self, **kwargs):
    #     """
    #     Computation of the Probability Density function of the increments (normalizing them).
    #     The PDF is calculated as Density (i.e.) normalized and provide appropriate estimate of
    #     the errors according to
    #     a Poisson distribution.  If xrange keyword is not set it computes the increments in
    #     a xrange [ - 4.5, 4.5] \sigma
    #     The number of bins can be given with keyword nbins.  Otherwise it is 41.
    #     pdf, xpdf, err = intermittency.pdf(intermittency.cwt(s, dt, scale)[, xrange = xrange, nbins = nbins])

    #     """
    #     self.xrange = kwargs.get('xrange', [- 4.5, 4.5])
    #     self.nbins = kwargs.get('nbins', 41)
    #     s = self.cwt()
    #     # normalization of the wavelet increments
    #     sNorm = (s - s.mean()) / (s.std())
    #     # pdf normalized within the range
    #     pdf, binE = np.histogram(
    #         sNorm, bins=self.nbins, density=True, range=self.xrange)
    #     # center of the bins
    #     xpdf = (binE[:-1] + binE[1:]) / 2.
    #     # error assuming a Poissonian deviation and propagated in a normalized
    #     # pdf
    #     err = np.sqrt(pdf / (np.count_nonzero(((sNorm >=
    #                                             self.xrange[0]) &
    #                                            (sNorm <= self.xrange[1]))) * (binE[1] - binE[0])))
    #     return pdf, xpdf, err

    # def threshold(self, **kwargs):
    #     """
    #     Given the signal initialized by the class it computes the location of the point above the threshold and
    #     creates a nd.array equal to 1 at the maximum above the given threshold.  The threshold can be given
    #     (let's say three times rms) or it can be automatically determined using the method described in
    #     G. Boffetta, V. Carbone, P. Giuliani, P. Veltri, and A. Vulpiani, Phys. Rev. Lett. 83, 4662 (1999).

    #     Keywords:
    #         b99 = Boolean default is True. Compute the threshold according to Boffetta paper
    #         thr = Floating. If set it gives directly the threshold. If set automatically exclude the B99 method
    #         resolution = resolution in coars-graining for the loop in determining the threshold
    #         factor = Default is 2. It is used for the method
    #         iterM = Maximum possible iteration
    #     Output:
    #         maxima = A binary array equal to 1 at the identification of the structure (local maxima)
    #         allmax = A binary array equal to 1 in all the region where the signal is above the threshold

    #     Example:
    #     >>> turbo = intermittency.Intermittency(signal, dt, fr)
    #     >>> maxima = turbo.threshold() # compute the maxima using the method of Boffetta
    #     >>> maxima = turbo.threshold(thr = xxx) # compute the threshold using xxx and automatically
    #     exclude the b99 keyword

    #     """

    #     self.b99 = kwargs.get('b99', True)
    #     self.thr = kwargs.get('thr', None)
    #     self.resolution = kwargs.get(
    #         'resolution',
    #         (self.sig.max() -
    #          self.sig.min()) /
    #         100.)  # these is the resolution in the coars - graining
    #     # this is a factor used in the b99 method.  Default = 2
    #     self.factor = kwargs.get('factor', 2)
    #     self.iterM = kwargs.get('iterM', 40)
    #     if self.thr is not None:
    #         self.b99 = False
    #     # this will be the output
    #     maxima = np.zeros(self.sig.size)
    #     allmax = np.zeros(self.sig.size)
    #     # method with Boffetta 99 method
    #     if self.b99:
    #         delta = self.sig.max() - self.sig.min()
    #         n_bin = np.long(delta / self.resolution)
    #         h, bE = np.histogram(self.sig, bins=n_bin)
    #         xh = (bE[:-1] + bE[1:]) / 2.
    #         while True:
    #             mean = np.sum(xh * h) / h.sum()
    #             sd = np.sqrt(np.sum(xh * xh * h) / h.sum() - mean ** 2)
    #             self.thr = mean + self.factor * sd
    #             iThr = np.long((self.thr - self.sig.min()) / self.resolution)
    #             r = np.sum(h[iThr + 1:])
    #             h[iThr + 1:] = 0
    #             if (r == 0 or iter == self.iterM):
    #                 break

    #     allmax[(self.sig>self.thr)] = 1
    #     # now we have or build the threshold using B99 or we have already the
    #     # threshold given
    #     imin = 0
    #     for i in range(maxima.size - 1):
    #         i += 1
    #         if self.sig[i] >= self.thr and self.sig[i - 1] < self.thr:
    #             imin = i
    #         if self.sig[i] < self.thr and self.sig[i - 1] >= self.thr:
    #             imax = i - 1
    #             if imax == imin:
    #                 d = 0
    #             else:
    #                 d = self.sig[imin: imax].argmax()
    #             maxima[imin + d] = 1
    #     return maxima, allmax
                            

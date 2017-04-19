__author__ = 'N. Vianello and N. Walkden'
__version__ = '0.2'
__data__ = '19.04.2017'

import numpy as np
import scipy 
import wav
import lmfit


class Timeseries:
    """
    Python class for turbulent signal analysis
    Inputs:
    sig = signal to be analyzed
    dt  = time resolution
    fr  = Corresponding Fourier frequency for the analysis
    wavelet = String considering the type of wavelet to be used.  Default is 'Mexican',  other choices
    could be 'DOG1', 'Morlet'
    Methods: 
       # Statistics
       cwt: Compute the CWT

       lim: Compute the Local Intermittency Measurement
            [M. Onorato, R. Camussi, and G. Iuso, Phys. Rev. E 61, 1447 (2000)]

       flatness: Compute the flatness at a given scale
       # ---------- 
       # structures
       strTime:  Compute the time occurrence of the Intermittent structure defined in
                 M. Onorato, R. Camussi, and G. Iuso, Phys. Rev. E 61, 1447 (2000)

       threshold:Define the time occurrence of structure above a threshold,
                 where threshold can be defined by the
                 user or is the one defined
                 in G. Boffetta et al, Phys. Rev. Lett. 83, 4662 (1999)

       cas: Compute the Conditional average sampling of the signal.
            Condition can be the
            occurrence of intermittent structure according
            to the LIM method, or the threshold method.  It also gives all the amplitude of each structures
       casMultiple: Compute the conditional average sampling on an array of signal using
            as condition the occurrence of intermittent structure (LIM) or the threshold method
            on the classes function.  It also gives all the amplitude of each structures



       # ------------ 
       # Distribution

       waitingTimes: Waiting and activity times computed using
            one of the above method (LIM or threshold)

       pdf: Probability density function of the normalized fluctuation

       castaingFit: Fit of the PDF according to the modified Castaing function

       strFun: Compute the structure function up to a given order
               (Default is nMax = 7)

       pdfAll: Compute the Probability Distribution Function of the entire signal.
               Eventually can
               be normalized (mean subtracted and std deviation divided)

       twoGammaFit: Perform the fit with two gamma function as described in
               F. Sattin et al, Plasma Phys. Contr. Fus. 48, 1033 (2006).

       levyFit: Perform a fit on the Waiting time distribution
                according to the Levy function introduced in Lepreti
                et al,  ApJ 55: L133
       powerCutFit: Perform a fit on the Waiting time distribution according to
                a power law sacling with an exponential cutoff (T_c) as done
                in R. D'Amicis, Ann. Geophysics vol 24,  p 2735, 2006


    Attributes:


    Usage:


    Dependences:
       numpy:
       scipy:
       pycwt: https://github.com/regeirk/pycwt.git
       lmfit: http://lmfit.github.io/lmfit-py/index.html
       future: Used by lmfit
    """

    def __init__(self, sig, dt, fr, wavelet='Mexican'):

        # inizializza l'opportuna wavelet definita in wavelet
        if wavelet == 'Mexican':
            self.mother = wav.Mexican_hat()
        elif wavelet == 'DOG1':
            self.mother = wav.DOG(m=1)
        elif wavelet == 'Morlet':
            self.mother = wav.Morlet()
        else:
            print 'Not a valid wavelet using Mexican'
            self.mother = wav.Mexican_hat()
        # inizializza l'opportuna scala
        self.fr = fr
        self.scale = 1. / self.mother.flambda() / self.fr
        self.sig = sig
        self.dt = dt

    def cwt(self):
        wt, sc, freqs, coi, fft, fftfreqs = wav.cwt(
            self.sig, self.dt, 0.25, self.scale, 0, self.mother)
        # keep only the real part
        return np.real(np.squeeze(wt))

    def lim(self):
        """
        Computation of Local Intermittency Measurements

        """
        wt = self.cwt()
        # normalization
        wt = (wt - wt.mean()) / wt.std()
        lim = np.abs(wt ** 2) / np.mean(wt ** 2)
        return np.squeeze(lim)

    def flatness(self):
        """
        Computation of the flatness

        """
        wt = self.cwt()
        # normalization
        wt = (wt - wt.mean()) / wt.std()
        flatness = np.mean(wt ** 4) / np.mean(wt ** 2) ** 2
        return flatness

    def strTime(self, **kwargs):
        """
        Determination of the time location of the structure identified accordingly to the LIM method
        described in M. Onorato, R. Camussi, and G. Iuso, Phys. Rev. E 61, 1447 (2000).
        Keywords:
            peak   = Boolean, if set True it computes the CAS only when the structure is a peak.
                     Default is False
            valley = Boolean, if set True it computes the CAS onlye when the structure is a valley.
                     Default is False
        Output:
            maxima = A binary array equal to 1 at the identification of the structure (local maxima)
            allmax = A binary array equal to 1 in all the region where the signal is above the threshold
        
        """
        self.peak = kwargs.get('peak', False)
        self.valley = kwargs.get('valley', False)

        wt = self.cwt()
        wtOr = wt.copy()
        wt = (wt - wt.mean()) / (wt.std())
        lim = np.abs(wt ** 2) / np.mean(wt ** 2)  # compute the LIM
        flatness = self.flatness()                # compute the flatness
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

    def cas(self, **kwargs):
        """
        Conditional average sampling over the computed intermittent structure location
        There are two keywords which can be used
        input:
        iwin   = dimension of the window for the CAS.  If it not given the amplitude
                 of the window is chosen to be
                 equal to 6 times the scale. It is in number of points.
                 If not given it is two times 1 / fr / dt
        peak   = Boolean, if set True it computes the CAS only when the structure is a peak.
                 Default is False.  Valid only when
        the computation is done with the LIM
        valley = Boolean, if set True it computes the CAS onlye when the structure is a valley.
                 Default is False. Valid only when compuation is done using the LIM
        b99 = Boolean. If set it computes the CAS using a threshold method based on Boffetta PRL (1999)
        thr = Threshold.  If set it computes the CAS using the given threshold
        resolution = see the definition of resolution in the threshold method
        factor = see the definition of factor in the threshold method
        iterM = see the definition of iterM in the threshold method

        output:
        cs  = conditional sampled structure
        tau = corresponding time basis

        """
        iwin = kwargs.get('iwin', np.round(1. / self.fr / self.dt))
        iwin *= 2
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
        if b99 or thr is not None:
            print('CAS using threshold method')
            maxima, allmax = self.threshold(
                b99=b99,
                thr=thr,
                resolution=resolution,
                factor=factor,
                iterM=iterM)
        else:
            print('CAS using the LIM')
            # determine location of the structures
            maxima, allmax = self.strTime(peak=self.peak, valley=self.valley)
        # zeros in a iwin region
        maxima[0: iwin - 1] = 0
        maxima[- iwin:] = 0
        print 'Number of structures mediated ' + str(maxima.sum())
        csTot = np.ones((2 * iwin + 1, maxima.sum()))
        d_ev = np.asarray(np.where(maxima >= 1))
        for i in range(d_ev.size):
            csTot[
                :,
                i] = sp.signal.detrend(
                self.sig[
                    d_ev[0][i] -
                    iwin: d_ev[0][i] +
                    iwin +
                    1])

        # now compute the cas
        cs = np.mean(csTot, axis=1)
        tau = np.linspace(- iwin, iwin, 2 * iwin + 1) * self.dt
        err = sp.stats.sem(csTot, axis=1)
        return cs, tau, err

    def casMultiple(self,inputS, **kwargs):
        """
        Conditional average sampling on multiple signals over the computed  structure location
        There are two keywords which can be used
        input: 
        inputS = Array of signals to be analyzed (not the one already used) in the form (#sign, #sample)
        iwin   = dimension of the window for the CAS.  If it not given the amplitude of the window is chosen to be
                 equal to 6 times the scale. It is in number of points.  If not given it is two times 1 / fr / dt
        peak   = Boolean, if set True it computes the CAS only when the structure is a peak.
                 Default is False.  Valid only when
                 the computation is done with the LIM
        valley = Boolean, if set True it computes the CAS onlye when the structure is a valley.
                 Default is False. Valid
                 only when compuation is done using the LIM
        b99 = Boolean. If set it computes the CAS using a threshold method based on Boffetta PRL (1999)
        thr = Threshold.  If set it computes the CAS using the given threshold
        resolution = see the definition of resolution in the threshold method
        factor = see the definition of factor in the threshold method
        iterM = see the definition of iterM in the threshold method
        
        output:
        cs  = conditional sampled structure on all the signals
        tau = corresponding time basis 
        
        """
        Shape = inputS.shape
        nSig = Shape[0]

        iwin = kwargs.get('iwin', np.round(1./ self.fr / self.dt))
        iwin *= 2
        self.peak = kwargs.get('peak', False)
        self.valley = kwargs.get('valley', False)
        b99 = kwargs.get('b99', False)
        thr = kwargs.get('thr', None)
        resolution = kwargs.get('resolution',
                                     (self.sig.max() - self.sig.min()) / 100.) # these is the resolution in the coars - graining
        factor = kwargs.get('factor', 2) # this is a factor used in the b99 method.  Default = 2
        iterM = kwargs.get('iterM', 40)
        if b99 == True or thr != None:
            print('CAS using threshold method')
            maxima, allmax = self.threshold(b99 = b99, thr = thr,
                                    resolution = resolution, factor = factor, iterM = iterM)
        else: 
            print('CAS using the LIM')
            # determine location of the structures
            maxima, allmax = self.strTime(peak = self.peak, valley = self.valley)
        # zeros in a iwin region
        maxima[0: iwin - 1] = 0
        maxima[ - iwin: ]   = 0
        print 'Number of structures mediated ' + str(maxima.sum())
        csTot = np.ones((nSig + 1, 2 * iwin + 1, maxima.sum()))
        d_ev = np.asarray(np.where(maxima >= 1))
        ampTot = np.zeros((nSig + 1, maxima.sum()))
        
        for i in range(d_ev.size):
                dummy = sp.signal.detrend(self.sig[d_ev[0][i] - iwin: d_ev[0][i] + iwin + 1])
                csTot[0, : , i] = dummy
                # compute also the amplitude as max - min in the interval iwin / 2
                ampTot[0, i] = dummy[iwin / 2: 3 * iwin / 2].max() - dummy[iwin / 2: 3 * iwin / 2].min()
                for n in range(nSig):
                    dummy = sp.signal.detrend(inputS[n, d_ev[0][i] - iwin: d_ev[0][i] + iwin + 1])
                    csTot[n + 1,: , i] = dummy
                    ampTot[n + 1, i] = dummy[iwin / 2: 3 * iwin / 2].max() - dummy[iwin / 2: 3 * iwin / 2].min()

        # now compute the cas
        cs = np.mean(csTot, axis = 2)
        tau = np.linspace( - iwin, iwin, 2 * iwin + 1) * self.dt
        err = sp.stats.sem(csTot, axis = 2)
        return cs, tau, err, ampTot

    def pdf(self, **kwargs):
        """
        Computation of the Probability Density function of the increments (normalizing them).
        The PDF is calculated as Density (i.e.) normalized and provide appropriate estimate of
        the errors according to
        a Poisson distribution.  If xrange keyword is not set it computes the increments in
        a xrange [ - 4.5, 4.5] \sigma
        The number of bins can be given with keyword nbins.  Otherwise it is 41.
        pdf, xpdf, err = intermittency.pdf(intermittency.cwt(s, dt, scale)[, xrange = xrange, nbins = nbins])

        """
        self.xrange = kwargs.get('xrange', [- 4.5, 4.5])
        self.nbins = kwargs.get('nbins', 41)
        s = self.cwt()
        # normalization of the wavelet increments
        sNorm = (s - s.mean()) / (s.std())
        # pdf normalized within the range
        pdf, binE = np.histogram(
            sNorm, bins=self.nbins, density=True, range=self.xrange)
        # center of the bins
        xpdf = (binE[:-1] + binE[1:]) / 2.
        # error assuming a Poissonian deviation and propagated in a normalized
        # pdf
        err = np.sqrt(pdf / (np.count_nonzero(((sNorm >=
                                                self.xrange[0]) &
                                               (sNorm <= self.xrange[1]))) * (binE[1] - binE[0])))
        return pdf, xpdf, err

    def waitingTimes(self, **kwargs):
        """
        Waiting time distribution:
        Keywords:
            peak   = Boolean, default is False.  If set it computes the intermittent structure WT
            using only the peak
            valley = Boolean, default is False.  If set it computes the intermittent structure WT
            using only the valley
            b99 = Boolean, default is False.  If set it computes the WT distribution using the B99 method
            thr = Threshold for the computation of the WT distribution. This is the case where we
            use the threshold method for the
            identification of blobs
            resolution = see the definition of resolution in the threshold method
            factor = see the definition of factor in the threshold method
            iterM = see the definition of iterM in the threshold method
        Output:
            waiting times and quiescent time. The quiescent time is computed differently according to
            the definition of Sanchez. 

        """

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

        if b99 or thr is not None:
            # determine location of the structures
            print('Waiting times distribution using threshold method')
            maxima, allmax = self.threshold(
                b99=b99,
                thr=thr,
                resolution=resolution,
                factor=factor,
                iterM=iterM)
            # we need also to create a maxima which is 1 above all the
            # threshold to be used for the quiescent time
        else:
            print('Waiting times distribution on Intermittent structure')
            maxima, allmax = self.strTime(peak=self.peak, valley=self.valley)

        # in this case
        # calcolo la derivata del maxima
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
        dEv = np.diff(allmax)
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
        sk = kwargs.get('sk', sp.stats.skew(self.cwt()))
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

    def threshold(self, **kwargs):
        """
        Given the signal initialized by the class it computes the location of the point above the threshold and
        creates a nd.array equal to 1 at the maximum above the given threshold.  The threshold can be given
        (let's say three times rms) or it can be automatically determined using the method described in
        G. Boffetta, V. Carbone, P. Giuliani, P. Veltri, and A. Vulpiani, Phys. Rev. Lett. 83, 4662 (1999).

        Keywords:
            b99 = Boolean default is True. Compute the threshold according to Boffetta paper
            thr = Floating. If set it gives directly the threshold. If set automatically exclude the B99 method
            resolution = resolution in coars-graining for the loop in determining the threshold
            factor = Default is 2. It is used for the method
            iterM = Maximum possible iteration
        Output:
            maxima = A binary array equal to 1 at the identification of the structure (local maxima)
            allmax = A binary array equal to 1 in all the region where the signal is above the threshold

        Example:
        >>> turbo = intermittency.Intermittency(signal, dt, fr)
        >>> maxima = turbo.threshold() # compute the maxima using the method of Boffetta
        >>> maxima = turbo.threshold(thr = xxx) # compute the threshold using xxx and automatically
        exclude the b99 keyword

        """

        self.b99 = kwargs.get('b99', True)
        self.thr = kwargs.get('thr', None)
        self.resolution = kwargs.get(
            'resolution',
            (self.sig.max() -
             self.sig.min()) /
            100.)  # these is the resolution in the coars - graining
        # this is a factor used in the b99 method.  Default = 2
        self.factor = kwargs.get('factor', 2)
        self.iterM = kwargs.get('iterM', 40)
        if self.thr is not None:
            self.b99 = False
        # this will be the output
        maxima = np.zeros(self.sig.size)
        allmax = np.zeros(self.sig.size)
        # method with Boffetta 99 method
        if self.b99:
            delta = self.sig.max() - self.sig.min()
            n_bin = np.long(delta / self.resolution)
            h, bE = np.histogram(self.sig, bins=n_bin)
            xh = (bE[:-1] + bE[1:]) / 2.
            while True:
                mean = np.sum(xh * h) / h.sum()
                sd = np.sqrt(np.sum(xh * xh * h) / h.sum() - mean ** 2)
                self.thr = mean + self.factor * sd
                iThr = np.long((self.thr - self.sig.min()) / self.resolution)
                r = np.sum(h[iThr + 1:])
                h[iThr + 1:] = 0
                if (r == 0 or iter == self.iterM):
                    break

        allmax[(self.sig>self.thr)] = 1
        # now we have or build the threshold using B99 or we have already the
        # threshold given
        imin = 0
        for i in range(maxima.size - 1):
            i += 1
            if self.sig[i] >= self.thr and self.sig[i - 1] < self.thr:
                imin = i
            if self.sig[i] < self.thr and self.sig[i - 1] >= self.thr:
                imax = i - 1
                if imax == imin:
                    d = 0
                else:
                    d = self.sig[imin: imax].argmax()
                maxima[imin + d] = 1
        return maxima, allmax

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

        return c1 * (n1 * b1) ** (n1) / sp.special.gamma(n1) * \
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
        m2 = sp.var(dummy)
#        m3 = stats.skew(dummy)
#        m4 = stats.kurtosis(dummy, fisher=True)
        b1 = 1. / m1
        n1 = 1. / (b1 ** 2 * m2)
        c1 = area * b1 * (n1 ** n1) / sp.special.gamma(n1)
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

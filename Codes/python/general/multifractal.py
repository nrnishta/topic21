import lmfit
import numpy as np
import pycwt as wav
import scipy
import astropy.stats as Astats


class Multifractal(object):
    """
    Class for multifractal and intermittency analysis

    Attributes
    ----------
    fr : Float
        Fourier frequency used for the analysis
    scale : Float
        Time scale corresponding to the given Fourier Frequency
        and computed accordingly to the chosen mother wavelet
    dt : Float
        Time sampling
    Fs : float
        Frequency sampling
    mother : str
        String indicating the type of wavelet used for the analysis

    Dependences
    -----------
    lmfit http://lmfit.github.io/lmfit-py/index.html
    pycwt
    astropy

    To Do
    -----
    - Add the stretched function fit
    - Add the computation of fractal dimension
    """

    def __init__(self, signal, time, frequency=100e3, wavelet='Mexican'):
        """

        Parameters
        ----------
        signal : ndarray
            Signal to be analyzed
        time : ndarray
            Time basis
        frequency : Float
            Fourier frequency for the analysis
        wavelet : :obj: `str`
            String indicating the type of wavelet used
            for the analysis. Default is 'Mexican'

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
        # inizializza l'opportuna scala
        self.fr = frequency.copy()
        self.scale = 1. / self.mother.flambda() / self.fr
        self.sig = signal.copy()
        self.nsamp = signal.size
        self.time = time.copy()
        self.dt = (time.max() - time.min()) / (self.nsamp - 1)
        self.Fs = 1. / self.dt
        self.cwt()

    def cwt(self):
        """
        Compute the continuous wavelet transform

        Returns
        -------
        bool
            True if successful, False otherwise

        Atributes
        ---------
        wt : ndarray
            Real part of the wavelet coefficients
        wtN : ndarray
            Real part of the wavelet coefficients normalized
            (wt-<wt>)/\sigma
        """
        try:
            wt, sc, freqs, coi, fft, fftfreqs = wav.cwt(
                self.sig, self.dt, 0.25, self.scale, 0, self.mother)
            self.wt = np.real(np.squeeze(wt))
            self.wtN = (self.wt - self.wt.mean()) / self.wt.std()
            return True
        except BaseException:
            return False

    def lim(self):
        """
        Compute the Local Intermittency Measurements
        define in M. Onorato, R. Camussi, and G. Iuso,
        Phys. Rev. E 61, 1447 (2000)

        Returns
        -------
        lim : ndarray
            Local intermittency measurements
        """
        wt = self.cwt()
        # normalization
        wt = (wt - wt.mean()) / wt.std()
        lim = np.abs(wt ** 2) / np.mean(wt ** 2)
        return np.squeeze(lim)

    def flatness(self):
        """
        Computate the flatness starting
        from the wavelet coefficients

        Returns
        -------
        Flatness : ndarray
            Flatness computed from wavelet coefficients

        """

        flatness = np.mean(self.wtN ** 4) / np.mean(self.wtN ** 2) ** 2
        return flatness

    def pdf(self, bins=10, xrange=None, weights=None, normed=False, **kwargs):
        """
        Computation of the Probability Density function of the normalized
        increments

        Wrapper around histogram function from astropy.stats package.

        Parameters
        ----------
        bins : :obj: `int` or `list` or `str` (optional)
            If bins is a string, then it must be one of:

            - 'blocks' : use bayesian blocks for dynamic bin widths

            - 'knuth' : use Knuth's rule to determine bins

            - 'scott' : use Scott's rule to determine bins

            - 'freedman' : use the Freedman-Diaconis rule to determine bins

        xrange : tuple or None (optional)
            the minimum and maximum range for the histogram.  If not specified,
            it will be (x.min(), x.max())

        weights : array_like, optional
            Not Implemented

        **kwargs
           keyword accepted by self.pdf() and astropy.stats.histogram


        Returns
        -------
        hist : Float
            The values of the histogram. See ``normed`` and ``weights`` for a
            description of the possible semantics.
        xbin : Float of dtype float
            Return the bins center.
        err : Float
           Error assuming a Poissonian deviation and propagated in a normalized
           pdf

        See Also
        --------
        numpy.histogram
        astropy.stats.histogram
        """

        hist, bins_e = Astats.histogram(
            self.wtN, bins=bins, range=xrange,
            weights=weights, density=True, **kwargs)
        xpdf = (bins_e[1:] + bins_e[:-1]) / 2.
        if xrange is None:
            xrange = [self.wtN.min(), self.wtN.max()]
        err = np.sqrt(hist / (
            np.count_nonzero(
                ((self.wtN >=
                      xrange[0]) &
                     (self.wtN <= xrange[1]))) * (bins_e[1] - bins_e[0])))
        return hist, xpdf, err

    def castaingFit(self, s0=0.1, l=1, am=10, xrange=[-4.5, 4.5],
                    bins=41, **kwargs):
        """
        Perform a fit of the Probability Distribution function
        based on the Castaing model defined in
        B. Castaing et ak, Physica D: Nonlinear Phenomena 46, 177 (1990)
        and refined in
        L. Sorriso-Valvo, et al Astrophys J 807, 86 (2015).

        Parameters
        ----------
        s0 : Float
            Sigma_0 in the model, default value is 0.1
        l : Float
            Lambda parameter of the model, default values is 0.1
        am : Float
            Amplitude for the correct scaling, default value is 10

        Returns
        -------
        Fit : :obj: `Model fit`
            Model fit output

        Example
        -------
        >>> turbo = multifractal.Multifractal(s, time, frequency=100e3)
        >>> pdf, x, err = turbo.pdf()
        >>> fit = turbo.castaingFit()
        >>> semilogy(x, pdf, 'o--')
        >>> plot(x, fit.best_fit, ' - ', linewidth = 2)
        """

        sk = scipy.stats.skew(self.wt)
        # build the appropriateModel
        csMod = lmfit.models.Model(
            self._castaing,
            independent_vars='x',
            param_names=(
                's0',
                'l',
                'sk',
                'am'))
        # initialize the parameters
        pars = csMod.make_params(s0=s0, l=l, sk=sk, am=am)
        # compute the PDF
        pdf, x, err = self.pdf(xrange=xrange, nbins=bins)
        fit = csMod.fit(pdf, pars, x=x, weights=1. / err ** 2)
        return fit

    def strFun(self, nMax=7):
        """
        Compute the structure function the given
        frequency up to a maximum nMax order.  As a default
        the nMax is equal to 7.
        This would be useful for the Extended Self Similarity
        analysis introduced in
        R. Benzi et al.  , Phys. Rev. E 48, R29 (1993).

        Parameters
        ----------
        nMax : int
            Maximum number of structure function. Optional
            default is 7

        Returns
        -------
        structure function

        """
        return np.asarray([np.mean(self.cwt() ** (k + 1))
                           for k in range(nMax)])

    def _castaing(self, x, s0, l, sk, am):
        """
        Define the castaing - like function to fit the Probability
        density function of normalized fluctuations.
        The function is intoduced in
        B. Castaing et ak, Physica D: Nonlinear Phenomena 46, 177 (1990)
        and further developed in
        order to take into account possible asymmetric function in
        L. Sorriso-Valvo, et al Astrophys J 807, 86 (2015).
        ..math:
        P(\delta v) =  am / \sqrt(2\pi)\int G_{\lambda}(\sigma)
            \exp( - \delta v^2 / 2\sigma^2) * (1 +
            a_s\frac{dv / sigma}{np.sqrt{1 + dv^2 / sigma^2}})d\sigma / sigma

        G_{\lambda}(\sigma) =  \frac{1}{\sqrt{2}\lambda}
            \exp( - \ln^2(\sigma / sigma_0) / 2\lambda).

        Parameters
        ----------
        x : ndararray
            These are the increments which represents
            the bins of the PDF of the increments
        s0 : Float
            2.  Is \sigma_0
        l : Float
            \lambda
        sk : Float
            skewness
        am : Float
           Amplitude for proper normalization

        Returns
        -------
        out:  the computed function

        """
        # we need this for a proper definition of the integral
        def integrand(sigma, s, lamb, dv, skw):
            return 1. / 2 * np.pi / lamb * np.exp(
                -np.log(sigma / s) ** 2 / 2 / lamb ** 2) * 1. / sigma * np.exp(
                - dv ** 2 / 2 / sigma ** 2 *
                (1 + skw * ((dv / sigma) /
                            np.sqrt(1. + dv ** 2 / sigma ** 2))))

        # for proper integration in \sigma define
        from scipy.integrate import quad
        cst = np.asarray([am / np.sqrt(2 * np.pi) *
                          quad(integrand, 0, np.inf,
                               args=(s0, l, x[i], sk))[0]
                          for i in range(x.size)])
        return cst

import numpy as np
import lmfit
import astropy.stats as Astats
import pint


class wtFit(object):
    """
    Class to provide different possible fit of
    waiting times and quiescent times
    fit

    Requirements
    ------------
    - Astropy for better histogram function
    - lmfit for fitting procedure
    - pint: to handle proper conversion between time

    """

    def __init__(self, wt, qt=None, units=None):
        """
        Parameters
        ----------
        wt : :obj: `ndarray`
            Computed waiting times whose distribution needs to
            be analyzed

        qt : :obj: `ndarray` Optional
            If provided execute the analysis and fit also on the
            quiescent time as defined in Sanchez R, Newman D E
            and Carreras B A 2002 Phys. Rev. Lett. 88 068302

        units : :obj: `string`
            measurements units of the waiting times input. Can be
            `seconds`, `milliseconds`, `microseconds`

        Attributes
        ----------
        units : :obj: 'units'
            Unit values for the waiting and eventually quiescent time
            see. Pint documentation

        Beware by default conversion of waiting times to microseconds is
        performed
        """

        if units is None:
            units = 'seconds'
        ureg = pint.UnitRegistry()
        # naturally convert to microseconds
        self.Wt = wt.copy()*ureg(units).to('microseconds')
        if qt is not None:
            self.At = qt.copy()*ureg(units).to('microseconds')
            self.at = self.At.magnitude
        self.wt = self.Wt.magnitude
        self.units = self.Wt.units

    def pdf(self, x, bins=10, range=None, weights=None,
            log=False, **kwargs):
        """
        Computation of the Probability Density function of the signal

        Wrapper around histogram function from astropy.stats package.

        Parameters
        ----------

        x : :obj: `ndarray`
            Variable for the computation of the probability density
            function

        bins : :obj: `int` or `list` or `str` (optional)
            If bins is a string, then it must be one of:

            - 'blocks' : use bayesian blocks for dynamic bin widths

            - 'knuth' : use Knuth's rule to determine bins

            - 'scott' : use Scott's rule to determine bins

            - 'freedman' : use the Freedman-Diaconis rule to determine bins

        range : tuple or None (optional)
            the minimum and maximum range for the histogram.  If not specified,
            it will be (x.min(), x.max())

        log : :obj: `bool`
            Default is False. If True and used with integer bins
            compute logarithmically spaced bins

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
        if log:
            if range is None:
                range = [x.min(), x.max()]
            if not isinstance(bins, int):
                print('Warning log bins only with nbins as int')
                print('Assuming 20 bins')
                nb = 20
            else:
                nb = bins
            bins = np.logspace(np.log10(range[0]), np.log10(range[1]), nb)
        hist, bins_e = Astats.histogram(
            x, bins=bins, range=range,
            weights=weights, **kwargs)
        return hist, bins_e

    def levyFit(self, amp=100., mu=1., norm=1e6, quiet=False,
                **kwargs):
        """
        Fit the Waiting times (or eventually the quiescent time)
        distribution with a
        Levy type function as defined in Lepreti et al,  ApJ 55: L133 eq. 6
        .. math::
        P(\Delta t ) =
            \frac{1}{\pi}\int_0^{\infty}\cos(z\Delta t)\exp(-a|z|^{\mu})dz.


        Parameters
        ----------
        amp : :obj: `Float`
            amplitude for the fit. Default is 100

        mu : :obj: `Float`
            See the defintion of the Function. Default is 1

        norm : :obj: Float`
            See the definition of the function. Default is 1e6

        quiet : :obj: `Boolean`
            Default is false. If it is True compute the fit on quiescent time

        other keyword arguments are described in self.pdf() or in the
        lmfit.Model.model.fit()

        Returns
        -------
        fit (as output from lmfit),  pdf, bins, err

        """
        if quiet:
            print('Levy fit on quiescent times')
            us = self.at.copy()
        else:
            print('Levy fit on waiting times')
            us = self.wt.copy()

        pdf, bins = self.pdf(us, **kwargs)
        err = np.sqrt(pdf / (us.size * (bins[1] - bins[0])))
        # and now perform the real fit
        levyMod = lmfit.models.Model(self._levyFunction,
                                     independent_vars='x',
                                     param_names=('a', 'mu', 'norm'),
                                     missing='drop')
        pars = levyMod.make_params(a=amp, mu=mu, norm=norm)
        pars['mu'].set(min=0, max=2)
        fit = levyMod.fit(pdf,
                          pars,
                          x=(bins[1:]+bins[:-1])/2.,
                          weights=1 / (err)**2, **kwargs)
        return fit, pdf, bins, err

    def powerCutFit(self, amp=1e-2, a=1.5, tc=200, quiet=False,
                    **kwargs):
        """
        Fit the Waiting times (or eventually the quiescent time)
        distribution with a
        Truncated power law as done in R. D'Amicis et al,
        Ann. Geophysics **24** 2735 (2006)
        .. math::
        P(\Delta t ) =
            A\Delta t^{-\alpha} \exp(-\Delta t/T_c).


        Parameters
        ----------
        amp : :obj: `Float`
            amplitude for the fit. Default is 1e-2

        a : :obj: `Float`
            Scaling of the power law. Default is 1.5

        tc : :obj: `Float`
            Exponential cut

        quiet : :obj: `Boolean`
            Default is false. If it is True compute the fit on quiescent time

        other keyword arguments are described in self.pdf()

        Returns
        -------
        other keyword arguments are described in self.pdf() or in the
        lmfit.Model.model.fit()

        Examples
        --------
        """

        if quiet:
            print('Levy fit on quiescent times')
            us = self.at.copy()
        else:
            print('Levy fit on waiting times')
            us = self.wt.copy()

        pdf, bins = self.pdf(us, **kwargs)
        err = np.sqrt(pdf / (us.size * (bins[1] - bins[0])))
        # and now we define the appropriate model
        powCutMod = lmfit.models.Model(self._powerCut, independent_vars='x',
                                       param_names=('amp', 'a', 'tc'),
                                       missing='drop')
        # now the default parameter choice
        pars = powCutMod.make_params(amp=amp, a=a, tc=tc)
        fit = powCutMod.fit(pdf,
                            pars,
                            x=(bins[1:]+bins[:-1])/2.,
                            weights=1./(err)**2, **kwargs)
        return fit, pdf, bins, err

    def _levyFunction(self, x, a, mu, norm):
        """
        For the definition of the appropriate levy Function used for
        the fit consider the article F.Lepreti, ApJ
        """
        def integrandI(z, xd, amp, m):
            return 1. / np.pi * np.cos(z * xd) * np.exp(- amp * np.abs(z) ** m)
        from scipy.integrate import quad
        levF = norm * np.asarray([quad(integrandI, 0,
                                       np.inf, args=(x[i], a, mu))[0]
                                  for i in range(x.size)])
        return levF

    def _powerCut(self, x, amp, a, tc):
        a *= -1
        return amp * x**a * np.exp(-x / tc)

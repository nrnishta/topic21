import numpy as np
import scipy 
import pycwt as wav
import lmfit

class wtFit(object):
    """
    Class to provide different possible fit of
    waiting times and quiescent times

    def __init__(self, )"""


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
    

import lmfit

class fitPdf(object):

    def __init__(self)
    
    def _twoGamma(self, x, c1, n1, b1, c2, n2, b2):
        """
        
        Define the sum of two gamma function 
        according to Eq (1) of 
        F Sattin et al 2006 Plasma Phys. Control. Fusion 48 1033

        Parameters
        ----------
        x : :obj: `ndarray`
          These are the center of the bins of the PDF
        C1 : :obj: `float`
          See the definition of _oneGamma
        N1 : :obj: `float`
          See the definition of _oneGamma
        beta1 : :obj: `float`
          See the definition of _oneGamma
        C2 : :obj: `float`
          See the definition of _oneGamma
        N2 : :obj: `float`
          See the definition of _oneGamma
        beta2 : :obj: `float`
          See the definition of _oneGamma

        Return
        ------
        The function obtained from the sum of two Gammas defind in _oneGamma

        Example
        -------
        F = self._twoGamma(x, C1, N1, beta1, C2, N2, beta2)
        """

        twoG = self._oneGamma(x, c1, n1, b1) + self._oneGamma(x, c2, n2, b2)
        return twoG

    def _oneGamma(self, x, c1, n1, b1):
        """
        Define one gamma function according to the definition in
        Eq (1) of F Sattin et al 2006
        Plasma Phys. Control. Fusion 48 1033
        F = C*(N*beta)^n / Gamma(N)* x^(N-1)*exp(-beta*N*x)

        Parameters
        ----------
        x : :obj: `ndarray`
          These are the center of the bins of the PDF
        C : :obj: `float`
          See above equation
        N : :obj: `float`
          See above equation
        beta : :obj: `float`
          See above equation

        Returns
        -------
        The function F
        
        Example
        -------
        F = self._oneGamma(x, C, N, beta)
        """
        return c1 * (n1 * b1) ** (n1) / scipy.special.gamma(n1) * \
            x ** (n1 - 1) * np.exp(- b1 * n1 * x)

    def twoGammaFit(self, normed=False, **kwargs):
        """
        Perform a Fit of the Probability Density Function of the
        signal according to Eq (1) of paper 
        F Sattin et al 2006
        Plasma Phys. Control. Fusion 48 1033        

        F = C1*(N*beta1)^N1 / Gamma(N1)* x^(N1-1)*exp(-beta1*N1*x) + 
            C2*(N2*beta2)^N2 / Gamma(N2)* x^(N2-1)*exp(-beta2*N2*x)


        Parameters
        ----------
        normed : :obj: `boolean'
            If set compute the fit on the PDF of normalized signal
            (signal-<signal>)/sigma
        C1 : :obj: `float`, optional
          Initial value for the parameter C1 [See above equation]
        N1 : :obj: `float`, optional
          Initial value for the parameter N1 [See above equation]
        beta1 : :obj: `float`, optional
          Initial value for the parameter beta1 [See above equation]
        C2 : :obj: `float`, optional
          Initial value for the parameter C1 [See above equation]
        N2 : :obj: `float`, optional
          Initial value for the parameter N1 [See above equation]
        beta2 : :obj: `float`, optional
          Initial value for the parameter beta1 [See above equation]
        **kwargs
          These are the keywords accepted by pdf method
        Returns
        -------
        fit : :obj: 
            This is the output from lmfit referring to the 2Gamma fit
        xpdf : :obj: `ndarray`
            These are the center of the bins
        pdf : :obj: `ndarray`
            PDF of the signal which has been fitted
        fitO : :obj:
            This is the output from lmfit referring to a 1 Gamma fit

        Example
        -------
        fit, x, pdf, fit1 = self.twoGammaFit(normed=True, density=True,
           bins='freedman', C1=C1, N1=N1, beta1=beta1)
        """
                            
        # first of all compute the pdf with the keyword used
        if normed:
            dummy = self.signorm
            rMax = 6
        else:
            dummy = self.sig
            rMax = dummy.mean() + 6 * dummy.std()
        # compute the PDF limiting to the equivalent of 6 std
        pdfT, binT = self.pdf(range=[dummy.min(), rMax], **kwargs)

        xpdfT = (binT[:- 1] + binT[1:]) / 2
        # eliminate the 0 in pdfT
        xpdfT = xpdfT[pdfT != 0]
        pdfT = pdfT[pdfT != 0]
        # build also the weigths
        err = np.sqrt(pdfT / (np.count_nonzero((dummy <= rMax)))
                      * (xpdfT[1] - xpdfT[0]))
        print('Number of NaN on pdf', np.count_nonzero(np.isnan(pdfT)))
        print('Number of NaN on x', np.count_nonzero(np.isnan(xpdfT)))
        print('Number of NaN on err', np.count_nonzero(np.isnan(err)))
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
        c1 = kwargs.get('C1', c1)
        n1 = kwargs.get('N1', n1)
        b1 = kwargs.get('beta1', b1)
        c2 = kwargs.get('C2', c2)
        n2 = kwargs.get('N2', n2)
        b2 = kwargs.get('beta2', b2)
        print('c1', c1)
        print('n1', n1)
        print('b1', b1)
        print('c2', c2)
        print('n2', n2)
        print('b2', b2)
        
        # first of all we build the model for the one gamma function and fi
        oneGMod = lmfit.models.Model(
            self._oneGamma,
            independent_vars='x',
            param_names=(
                'c1',
                'n1',
                'b1'))
        parsOne = oneGMod.make_params(c1=c1, n1=n1, b1=b1)
        fitO = oneGMod.fit(pdfT, parsOne, x=xpdfT, weights=1 / err ** 2)

        # now we can build the model
        twoGMod = lmfit.models.Model(
            self._twoGamma, independent_vars='x',
            param_names=('c1', 'n1', 'b1', 'c2', 'n2', 'b2'))
        pars = twoGMod.make_params(c1=fitO.params['c1'].value,
                                   n1=fitO.params['n1'].value,
                                   b1=fitO.params['b1'].value,
                                   c2=c2, n2=n2, b2=b2)
        fit = twoGMod.fit(pdfT, pars, x=xpdfT, weights=1 / err ** 2)
        return fit, xpdfT, pdfT, fitO


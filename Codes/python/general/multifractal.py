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
    


"""
Class to implement evaluation of pressure drop along fluxtube from
upstream to target
"""

from __future__ import print_function
import numpy as np
from scipy import constants
import eqtools
import langmuir
import tcvProfiles
from scipy.interpolate import interp1d
import xarray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,
                                              ConstantKernel,
                                              WhiteKernel,
                                              Matern,ExpSineSquared,
                                              RationalQuadratic)


class fluxpressure(object):
    def __init__(self, shot):

        self.shot = shot
        # init of langmuir probe
        self.Target = langmuir.LP(self.shot)
        # init of equilibrium
        self.Eq = eqtools.TCVLIUQETree(self.shot)
        # init of profile
        self.Profile = tcvProfiles.tcvProfiles(self.shot)

#    def fraction(self, trange=None, **kwargs):

    def _computeUpstream(self, trange=None, **kwargs):
        """
        Collect the profile combining existing diagnostic, both density
        and temperature. Profile obtained using a gaussian-process fit for rho>0.9
        for both the quantities on an uniform rho basis from 0.9<rho<1.12 and
        computing the corresponding temperature.

        Parameters
        ----------
        trange: time range for the evaluation of the profiles
        kwargs: any keywords accepted by gpr_robustfit of tcvProfiles

        Returns
        -------
        PUpRaw: define also as attribute of the class. xarray DataArray containing the
                raw data of pressure profile together with the error as attribute
        PUpFit: defined also as attribute of the class. xarray DataArray containing
                the fitted Pressure profile built on top of the gpr fit of the
                density and temperature
        """
        if not trange:
            trange = [0.5, 0.7]

        self.TeUpstream = self.Profile.profileTe(t_min=trange[0], t_max=trange[1], abscissa='sqrtpsinorm')
        self.NeUpstream = self.Profile.profileNe(t_min=trange[0], t_max=trange[1], abscissa='sqrtpsinorm')
        # we are only interested in the SOL ma we retain part of the profile in the
        # confined region since in this way the gpr fit is more robust
        _ = self.TeUpstream.remove_points((self.TeUpstream.X[:,0] < 0.95))
        _ = self.NeUpstream.remove_points((self.NeUpstream.X[:,0] < 0.95))
        # GPR fit for both the quantities in a uniform rhogrid
        self.rhoUpstream = np.linspace(0.95,1.15,50)
        self.NeUFit, self.NeUFitErr, self.NeUGpr = self.Profile.gpr_robustfit(
            self.rhoUpstream, density=True, temperature=False,**kwargs)
        self.TeUFit, self.TeUFitErr, self.TeUGpr = self.Profile.gpr_robustfit(
            self.rhoUpstream, density=False, temperature=True,**kwargs)
        self.PUpFit = xarray.DataArray(constants.e * self.TeUFit * self.NeUFit * 1e20,
                                       coords=[self.rhoUpstream],dims=['rho'])
        self.PUpFit.attrs['Err'] = 1e20* constants.e * np.sqrt(
            np.power(self.TeUFit * self.NeUFitErr,2) +
            np.power(self.NeUFit * self.TeUFitErr,2))
        # for consistency we need also the raw data
        # check which has fewer points so that we don't need extreme extrapolation
        if self.TeUpstream.X.ravel().size <= self.NeUpstream.X.ravel().size:
            _x = self.NeUpstream.X.ravel()
            _y = self.NeUpstream.y
            S = interp1d(_x[np.argsort(_x)],_y[np.argsort(_x)],kind='linear')
            self.PUpRaw = xarray.DataArray(
                1e20* constants.e * self.TeUpstream.y * S(self.TeUpstream.X.ravel()),
                coords=[self.TeUpstream.X.ravel()],dims=['rho'])
            self.PUpRaw.attrs['Err'] = \
                1e20 * constants.e * self.TeUpstream.err_y * S(self.TeUpstream.X.ravel())
        else:
            _x = self.TeUpstream.X.ravel()
            _y = self.TeUpstream.y
            S = interp1d(_x[np.argsort(_x)],_y[np.argsort(_x)],kind='linear')
            self.PUpRaw = xarray.DataArray(
                1e20 * constants.e * self.NeUpstream.y * S(self.NeUpstream.X.ravel()),
                coords=[self.NeUpstream.X.ravel()], dims=['rho'])
            self.PUpRaw.attrs['Err'] = \
                1e20 * constants.e * self.NeUpstream.err_y * S(self.NeUpstream.X.ravel())

        return self.PUpRaw, self.PUpFit

    def _computeTarget(self, trange=None, **kwargs):
        """

        Parameters
        ----------
        trange: time range where computation of profiles at the lower target will be
            evaluated
        kwargs: keywords which are passed to method _gpfit

        Returns
        -------

        """
        if not trange:
            trange = [0.5, 0.7]

        out = self.Target.UpStreamProfile(trange=trange)
        _npoints = []
        for key in ('en', 'r','rho','te'):
            _npoints.append(np.size(out[key]))
        for key in ('en', 'r','rho','te'):
            out[key] = out[key][:np.min(_npoints)]

        # for some reason there are cases where the number of points
        # of density and temperature are different
        # check if this is the case
        # now we need to double check for anomalous high
        # values of temperature or density
        _ida = np.where((out['te'] < 100))[0]
        _idb = np.where((out['en']/1e19 < 2))[0]
        _id = np.unique(np.concatenate((_ida, _idb),axis=0))
        for key in ('en', 'r','rho','te'):
            print('formatting ' + key)
            out[key] = out[key][_id]

        neFit, neStd = self._gpfit(out['rho'],out['en']/1e19, **kwargs)
        teFit, teStd = self._gpfit(out['rho'],out['te'], **kwargs)

        # define the pressure both raw and Fit
        self.PDoFit = xarray.DataArray(
            constants.e * teFit * neFit * 1e19,
            coords=[self.rhoUpstream],
            dims=['rho']
        )
        _dummy = 1e19 * constants.e * np.sqrt(
            np.power(teFit * neStd,2) +
            np.power(neFit*teStd,2)
        )
        self.PDoFit.attrs['Err'] = _dummy

        self.PDoRaw = xarray.DataArray(
            constants.e * out['te'] * out['en'],
            coords=[out['rho']],dims=['rho']
        )

        self.TeDownstream = xarray.DataArray(out['te'],coords=[out['rho']],dims=['rho'])
        self.TeDownstream.attrs['Fit'] = teFit
        self.TeDownstream.attrs['FitErr'] = teStd
        self.NeDownstream = xarray.DataArray(out['en'],coords=[out['rho']],dims=['rho'])
        self.NeDownstream.attrs['Fit'] = neFit
        self.NeDownstream.attrs['FitErr'] = neStd
        return self.PDoRaw, self.PDoFit

    def _gpfit(self, x, y, **kwargs):
        """
        Perform a gaussian process regression fit using a combination
        of ConstantKernel,Rational Quadratic and WhiteKernel.
        Parameters
        ----------
        x: indipendent parameter of the fit
        y: dependent parameter of the fit
        kwargs: any argument for ConstantKernel,
            RationalQuadratic and WhiteKernel
        Returns
        -------
        yfit : fitted value on the rhoUpstream basis
        std : error on the fit evaluation
        """
        if constant_value in kwargs:
            constant_value = kwargs['constant_value']
        else:
            constant_value = 0.01

        if length_scale in kwargs:
            length_scale = kwargs['length_scale']
        else:
            length_scale = 0.25

        if noise_level in kwargs:
            noise_level = kwargs['noise_level']
        else:
            noise_level = 0.5
        # avoid NaNs
        _dummy = np.vstack((x, y)).transpose()
        _dummy = _dummy[~np.isnan(_dummy).any(1)]
        x = _dummy[:, 0]
        y = _dummy[:, 1]
        # we have multiple values for rho,
        # we decide to average over these values
        # before fitting
        yy = np.asarray([])
        for _x in np.unique(x):
            _idx = np.where(x == _x)[0]
            if _idx.size > 1:
                yy = np.append(yy,np.nanmean(y[_idx]))
            else:
                yy = np.append(yy,y[_idx])

        X = np.atleast_2d(np.unique(x))
        kernel = ConstantKernel(constant_value=constant_value,
                                constant_value_bounds=(0.001,1)) + \
                 RationalQuadratic(length_scale=length_scale) + \
                 WhiteKernel(noise_level=noise_level)
        gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=55)
        gp.fit(X.T,yy)
        xN = np.atleast_2d(self.rhoUpstream)
        yFit, sigma= gp.predict(xN.T,return_std=True)

        return yFit, sigma
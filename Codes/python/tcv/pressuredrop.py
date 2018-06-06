
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

class fluxpressure(object):
    def __init__(self, shot):

        self.shot = shot
        # init of langmuir probe
        self.Target = langmuir.LP(self.shot)
        # init of equilibrium
        self.Eq = eqtools.TCVLIUQETree(self.shot)
        # init of profile
        self.Profile = tcvProfiles.tcvProfiles(self.shot)

    def _computeUpstream(self, trange=None, **kwargs):
        """
        Collect the profile combining existing diagnostic, both density
        and temperature. Profile obtained using a gaussian-process fit for rho>0.9
        for both the quantities on an uniform rho basis from 0.9<rho<1.12 and
        computing the corresponding temperature.

        Parameters
        ----------
        trange: time range for the evaluation of the profiles


        Returns
        -------

        """
        if not trange:
            trange = [0.5, 0.7]

        self.TeUpstream = self.Profile.profileTe(t_min=trange[0], t_max=trange[1], abscissa='sqrtpsinorm')
        self.NeUpstream = self.Profile.profileNe(t_min=trange[0], t_max=trange[1], abscissa='sqrtpsinorm')
        # we are only interested in the SOL ma we retain part of the profile in the
        # confined region since in this way the gpr fit is more robust
        _ = self.TeUpstream.remove_points((self.TeUpstream.X[:,0] < 0.9))
        _ = self.NeUpstream.remove_points((self.NeUpstream.X[:,0] < 0.9))
        # GPR fit for both the quantities in a uniform rhogrid
        self.rhoUpstream = np.linspace(0.9,1.15,50)
        self.NeUFit, self.NeUFitErr, self.NeUGpr = self.Profile.gpr_robustfit(
            self.rhoUpstream, density=True, temperature=False,**kwargs)
        self.TeUFit, self.TeUFitErr, self.TeUGpr = self.Profile.gpr_robustfit(
            self.rhoUpstream, density=False, temperature=True,**kwargs)
        self.PUpFit = constants.e * self.TeUFit * self.NeUFit * 1e20
        self.PUpFitE = 1e20* constants.e * np.sqrt(
            np.power(self.TeUFit * self.NeUFitErr,2) +
            np.power(self.NeUFit * self.TeUFitErr,2))
        # for consistency we need also the raw data
        # check which has fewer points so that we don't need extreme extrapolation
        if self.TeUpstream.X.ravel().size <= self.NeUpstream.X.ravel().size:
            _x = self.NeUpstream.X.ravel()
            _y = self.NeUpstream.y
            S = interp1d(_x[np.argsort(_x)],_y[np.argsort(_x)],kind='linear')
            self.PUpRaw = 1e20* constants.e * self.TeUpstream.y * S(self.TeUpstream.X.ravel())
            self.PUpRawE = 1e20 * constants.e * self.TeUpstream.err_y * S(self.TeUpstream.X.ravel())
        else:
            _x = self.TeUpstream.X.ravel()
            _y = self.TeUpstream.y
            S = interp1d(_x[np.argsort(_x)],_y[np.argsort(_x)],kind='linear')
            self.PUpRaw = 1e20 * constants.e * self.NeUpstream.y * S(self.NeUpstream.X.ravel())
            self.PUpRawE = 1e20 * constants.e * self.NeUpstream.err_y * S(self.NeUpstream.X.ravel())


"""
Class to implement evaluation of pressure drop along fluxtube from
upstream to target
"""

from __future__ import print_function
import MDSplus as mds
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import constants
from scipy import signal
import eqtools
import langmuir
import xarray as xray
import tcvProfiles

class fluxpressure(object):
    def __init__(self, shot):

        self.shot = shot
        # init of langmuir probe
        self.Target = langmuir.LP(self.shot)
        # init of equilibrium
        self.Eq = eqtools.TCVLIUQETree(self.shot)
        # init of profile
        self.Profile = tcvProfiles.tcvProfiles(self.shot)

    def _computeUpstream(self, trange=None):
        """
        Collect the profile combining existing diagnostic, both density
        and temperature. Profile the gaussian-process fit for rho>0.9
        for both the quantities on an uniform rho basis from 0.9<rho<1.12

        Parameters
        ----------
        trange: time range for the evaluation of the profiles


        Returns
        -------
        Output is the BivariatePlasmaProfile obtained as an output of tcvProfile
        Furthermore it returns also an xarray dataframe with the fitted profiles and
        corresponding error
        """
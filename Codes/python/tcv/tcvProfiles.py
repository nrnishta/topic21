from __future__ import print_function
import sys
if sys.platform == 'darwin':
    sys.path.append('/Users/vianello/Documents/Fisica/Computing/'
                    'cd cd 60884pythonlib/submodules/profiletools')
    sys.path.append('/Users/vianello/Documents/Fisica/Computing/OMFIT-source/omfit/classes')
else:
    sys.path.append('/home/vianello/pythonlib/submodules/profiletools')
    sys.path.append('/home/vianello/OMFIT-source/omfit/classes')
import profiletools
from omfit_gptools import *
from omfit_gpr1d import *
import warnings


class tcvProfiles(object):

    def __init__(self, shot):
        """

        Parameters
        ----------
        shot
            Shot number
        """
        self.shot = shot
    def profileNe(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs
            All the keywords inherited from profiletools.TCV.ne

        Returns
        -------
            Return a profile istances from profiletools class
        """

        self.ne = profiletools.TCV.ne(self.shot, **kwargs)
        self.ne.time_average()
        return self.ne

    def profileTe(self, **kwargs):
        """

        Parameters
        ----------
        kwargs
            All the keywords inherited from profiletools.TCV.te
        Returns
        -------
        Return a profile istances from profiletools class
        """
        self.te = profiletools.TCV.te(self.shot, **kwargs)
        self.te.time_average()
        return self.te

    def gpr_robustfit(self, xnew, gaussian_length_scale=3.0,
                      nu_length_scale=0.1, density=True, temperature=False):
        """
        Perform a gaussian process regression fit using a combination
        of two kernels (GSE_GL and MATERN_HI)
        Parameters
        ----------
        xnew : new abscissa for the evaluation of the regression
        gaussian_length_scale :
            gaussian scale for a GSE_GL kernel
        nu_length_scale
            scale length for a Matern Kernel
        density : boolean
            If set [default] performs the fit on the density profile
        temperature : boolean
            If set performs the fit on the temperature profile
        Returns
        -------
        ynew :
            Interpolated profiles
        err :
            standard deviation
        gpr :
            Instance to gaussian process regression from OMFIT_gptools

        """
        if density:
            x = self.ne.X[:,0]
            y = self.ne.y
            e = self.ne.err_y
        elif temperature:
            x = self.te.X[:,0]
            y = self.te.y
            e = self.te.err_y
        else:
            warnings.warn('You must specify density or temperature. Assume temeperature')
            x = self.ne.X[:,0]
            y = self.ne.y
            e = self.ne.err_y

        kg = GSE_GL_Kernel(var=1.0, lb=gaussian_length_scale, gh=0.5, lm=0.0, lsig=0.5)
        km = Matern_HI_Kernel(amp=0.1, ls=nu_length_scale, nu=2.5)
        krnl = Sum_Kernel(klist=[kg, km])
        gpr = GaussianProcessRegression1D(kernel=krnl, xdata=x, ydata=y, yerr=e)
        ynew, yerr = gpr(xnew)
        return ynew, yerr, gpr

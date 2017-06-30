import numpy
from scipy import io
from scipy import interpolate
from scipy import constants
import numpy as np
import MDSplus as mds


class LP:
    """
    Python class to read and compute appropriate
    profiles of LPs data once they have been saved
    in mat files
    Inputs:
    -------
      shot = shot number
      folder = Optional absolute path to locate the
               file save of langmuir

    Methods:
    --------
       UpStreamProfile : Compute the density and temperature
                profile average on a
                given time interval remapped as
                upstream distance from the separatrix

    Attributes:
    -----------
    Attributes
    """

    def __init__(self, shot, Type='floor'):
        self.shot = shot
        self.type = Type
        # open the appropriate Tree and get the available probes and
        # locations
        self._Tree = mds.Tree('tcv_shot', self.shot)
        # these are the probes number used in the present shot
        self.Probes = self._Tree.getNode(
            r'\results::langmuir:probes').data()
        # load the position of the langmuir probes
        self.pos = io.loadmat(r'/home/vianello/work/tcv15.2.2.3/' +
                              'data/lp_codes/LPposit.mat')['lppos']
        # we now have properly defined all the quantities needed
        self._defineProbe(type=self.type)
        # appropriate remapping upstream
        self.RemapUpstream()

    def _defineProbe(self, type='floor'):
        """
        Choose the appropriate type of probes which will
        be remapped through the type keyword. Possibilities
        are 'floor', 'HFSwall','LFSwall' with obvious
        meanings
        """
        # now we read the position for each of the probe in
        # term of (R, Z)
        R, Z = zip(self._Tree.getNode(
            r'\results::langmuir:pos').data())
        R = np.asarray(R).ravel()
        Z = np.asarray(Z).ravel()
        self.t = self._Tree.getNode(
            r'\results::langmuir:time').data()
        self.t2 = self._Tree.getNode(
            r'\results::langmuir:time2').data()
        if type == 'floor':
            # limit to the floor probe
            self._idx = np.where(Z == -0.75)[0]
        elif type == 'HFSwall':
            self._idx = np.where(R == 0.624)[0]
        elif type == 'LFSwall':
            self._idx = np.where(R == 1.136)[0]
        else:
            print('Not implemented')

        self.en = self._Tree.getNode(
           r'\results::langmuir:four_par_fit:dens').data()[:, self._idx]
        self.te = self._Tree.getNode(
            r'\results::langmuir:four_par_fit:Te').data()[:, self._idx]
        self.angle = self._Tree.getNode(
            r'\results::langmuir:area').data()[:, self._idx]
        self.jSat2 = self._Tree.getNode(
            r'\results::langmuir:jsat2').data()[:, self._idx]
        self.pPer = self._Tree.getNode(
            r'\results::langmuir:four_par_fit:P_perp').data()[:, self._idx]
        self.R = R[self._idx]
        self.Z = Z[self._idx]

    def RemapUpstream(self):
        # first of all trasform remap the R, Z to
        # Rmid

        self.RUpStream = self._Tree.getNode(
            r'\results::langmuir:dsep_mid').data()[:, self._idx]
        self.RUpStream2 = self._Tree.getNode(
            r'\results::langmuir:dsep_mid2').data()[:, self._idx]
        self.Rho = self._Tree.getNode(
            r'\results::langmuir:rho_psi').data()[:, self._idx]
        self.Rho2 = self._Tree.getNode(
            r'\results::langmuir:rho_psi2').data()[:, self._idx]

    def UpStreamProfile(self, trange=[0.6, 0.8]):
        """
        In this way we define the profile in a given
        time range upstream remapped
        """
        # now we only need to average appropriately
        # taking into account that at each time
        # step we have different remapping
        rOut = numpy.asarray([])
        neOut = numpy.asarray([])
        teOut = numpy.asarray([])
        for r in range(self.R.size):
            _idx = ((self.t >= trange[0]) &
                    (self.t <= trange[1])).nonzero()[0]
            for i in _idx:
                neOut = numpy.append(neOut, self.en[_idx, r])
                teOut = numpy.append(teOut, self.te[_idx, r])
                rOut = numpy.append(rOut,
                                    self.RUpStream[_idx, r])

        # we include also an univariate spline interpolation
        # and we decide to output a dictionary containing the
        # values
        _dummy = numpy.vstack((rOut, neOut)).transpose()
        _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
        x = _dummy[:, 0]
        y = _dummy[:, 1]
        neInt = interpolate.interp1d(x, y, fill_value='extrapolate')

        _dummy = numpy.vstack((rOut, teOut)).transpose()
        _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
        x = _dummy[:, 0]
        y = _dummy[:, 1]
        teInt = interpolate.interp1d(x, y, fill_value='extrapolate')

        out = dict([('r', rOut),
                    ('en', neOut),
                    ('te', teOut),
                    ('neInt', neInt),
                    ('teInt', teInt)])
        return out

    def Lambda(self, xCl, yCl, gas='D2', trange=[0.6, 0.8]):
        """
        Compute the Lambda divertor profile given the
        array of connection length, the gas and the trange
        to perform the computation of the profile

        """

        if gas == 'D2':
            Z = 1
        elif gas == 'H':
            Z = 1
        elif gas == 'He':
            Z = 4
        else:
            print('Gas not found')
        out = self.UpStreamProfile(trange=trange)
        neInt = out['neInt'](xCl)
        teInt = out['teInt'](xCl)
        nuEi = 5e-11*neInt/(teInt**1.5)
        Cs = numpy.sqrt(2 * constants.e * teInt /
                        (Z*constants.proton_mass))
        Lambda = (nuEi * yCl * constants.electron_mass /
                  (Z*constants.proton_mass*Cs))
        return Lambda

    def TotalSpIonFlux(self):
        """
        Method for the computation of the total ion flux
        to the Strike point chosen (if floor is chosen it uses the
        second strike point)
        """
        # now we compute the flux
        intFlux = numpy.zeros(self.t2.size)
        for i in range(self.t2.size):
            _x = self.R
            _y = self.jSat2[i, :]/constants.elementary_charge*1e4
            # eliminate the NaN
            _dummy = numpy.vstack((_x, _y)).transpose()
            _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
            _x = _dummy[:, 0]
            _y = _dummy[:, 1][numpy.argsort(_x)]
            _x = numpy.sort(_x)
            intFlux[i] = numpy.trapz(_y, x=2*numpy.pi*_x)
        return intFlux


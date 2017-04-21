import timeseries
import dd
import eqtools
import numpy as np

class Filaments(object):
    """
    Class for analysis of Filaments from the HFF probe on the
    midplane manipulator

    Parameters
    ----------
    Shot : :obj: `int`
        Shot number
    Probe: :obj: `string`
        String indicating the probe head used. Possible values are 
        - `HFF`: for High Heat Flux probe for fluctuation measurements
        - 'IPP': Innsbruck-Padua probe head
    """

    def __init__(self, shot, Probe='HFF'):
        self.shot = shot
        # load the equilibria
        self.Eq = eqtools.AUGDDData(self.shot)
        # open the shot file
        self.Probe = Probe
        if self.Probe == 'HFF':
            self._HHFGeometry(angle=80)
            self._loadHHF()
        else:
            print('Other probe head not implemented yet')

    def _HHFGeometry(self, angle=80.):
        """
        Define the dictionary containing the geometrical information concerning
        each of the probe tip of the HHF probe
        """
        self.Zmem = 312
        self.Xprobe = None
        RZgrid = {'m01':{'r':10, 'z':16.5},
                  'm02':{'r':3.5, 'z':16.5},
                  'm03':{'r':-3.5, 'z':16.5}, 
                  'm04':{'r':-10, 'z':16.5},
                  'm05':{'r':10, 'z':5.5},
                  'm06':{'r':3.5, 'z':5.5},
                  'm07':{'r':-3.5, 'z':5.5},
                  'm08':{'r':-10, 'z':5.5},
                  'm09':{'r':10, 'z':-5.5},
                  'm10':{'r':3.5, 'z':-5.5},
                  'm11':{'r':-3.5, 'z':-5.5},
                  'm12':{'r':-10, 'z':-5.5}, 
                  'm13':{'r':10, 'z':-16.5},
                  'm14':{'r':3.5, 'z':-16.5},
                  'm15':{'r':-3.5, 'z':-16.5},
                  'm16':{'r':-10, 'z':-16.5}}
        self.RZgrid = {}
        for probe in RZgrid.iterkeys():
            x, y = self._rotate((RZgrid[probe]['r'], RZgrid[probe]['z']), np.radians(angle))
            self.RZgrid[probe] = {'r':x, 'z':self.Zmem + y}


    def _loadHHF(self):
        """
        Load the data for the HHF probe by reading the Shotfile
        and distinguishing all the saved them as dictionary for
        different type of signal including their geometrical
        information

        """

        # open the MHC experiment
        Mhc = dd.shotfile('MHC', self.shot)
        Mhg = dd.shotfile('MHG', self.shot)
        # read the names of all the signal in Mhc
        namesC = Mhc.getObjectNames()
        namesG = Mhg.getObjectNames()
        # save a list with the names of the acquired ion
        # saturation current
        self.vfArr = {}
        self.isArr = {}
        for n in namesC.itervalues():
            if n[:6] == 'Isat_m':
                self.isArr[n] = dict([('data', Mhc(n).data),
                                      ('t', Mhc(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z'])])
            elif n[:5] == 'Ufl_m':
                self.vfArr[n] = dict([('data', Mhc(n).data),
                                      ('t', Mhc(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z'])])

        for n in namesG.itervalues():
            if n[:6] == 'Isat_m':
                self.isArr[n] = dict([('data', Mhg(n).data),
                                      ('t', Mhg(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z'])])
            elif n[:5] == 'Ufl_m':
                self.vfArr[n] = dict([('data', Mhg(n).data),
                                      ('t', Mhg(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z'])])

        Mhc.close()
        Mhg.close()

    def loadPosition(self):
        Lsm = dd.shotfile('LSM', self.shot)
        sPos = Lsm('S-posi').data
        tPos = Lsm('S-posi').time
        # convert into absolute value according to transformation
        R = (2188 + (1719 - self.Xprobe) + sPos + 100)/1e3
        # convert in Rhopoloidal
        self.rho = self.Eq.rz2psinorm(R, np.repeat(self.Zmem*1e-3), tPos, sqrt=True)
        # then we need to compute for each of the different probe tips
        # this quantity and save them into a dictionary

    def _rotate(self, point, angle):
        """
        Provide rotation of a point in a plane with respect to origin
        """
        px, py = point
        qx = np.cos(angle) * px - py *np.sin(angle)
        qy = np.sin(angle) * px + np.cos(angle) * py
        return qx, qy

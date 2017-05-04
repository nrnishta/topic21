import dd
import numpy as np
import matplotlib as mpl
import copy


class Neutrals(object):
    """
    Class to handle all the information concerning
    neutral measurements from the gauges and
    
    Parameters
    ----------
    shot : :obj: int
        Shot number

    """

    def __init__(self, shot):

        self.shot = shot
        # load the geometry
        self._geometry()
        # open the shotfile
        self.Ioc = dd.shotfile('IOC', self.shot)
        # read and create the attribute dictionary
        self._read()
        # close all the stuff
        self.Ioc.close()
        # open and read available gas information
        self.Uvs = dd.shotfile('UVS', self.shot)
        self._readGas()
        self.Uvs.close()

    def compression(self, Midplane=''):
        pass

    def _geometry(self):
        """
        Paramaters
        ----------

        Returnes
        --------

        Attributes
        ----------
        Provide as attribute the dictionary with
        the R, Z coordinates and orientation of the
        gauges
        """
        self.geometry={
            'F01':{'R':1.423, 'Z':-1.106, 'Angle':90, 'Sector':3},
            'F02':{'R':1.837, 'Z':-0.873, 'Angle':270, 'Sector':3},
            'F03':{'R':1.390, 'Z':-1.178, 'Angle':270, 'Sector':7},
            'F04':{'R':1.423, 'Z':-1.106, 'Angle':90, 'Sector':7},
            'F05':{'R':1.837, 'Z':-0.873, 'Angle':180, 'Sector':11},
            'F06':{'R':1.837, 'Z':-0.873, 'Angle':270, 'Sector':11},
            'F07':{'R':1.423, 'Z':-1.106, 'Angle':90, 'Sector':15},
            'F08':{'R':1.619, 'Z':-1.110, 'Angle':171, 'Sector':15},
            'F09':{'R':1.390, 'Z':-1.178, 'Angle':270, 'Sector':15},
            'F10':{'R':1.220, 'Z':-1.100, 'Angle':45, 'Sector':15},
            'F11':{'R':1.040, 'Z':-0.580, 'Angle':270, 'Sector':15},
            'F12':{'R':2.135, 'Z':-1.010, 'Angle':90, 'Sector':15},
            'F13':{'R':2.300, 'Z': 0.700, 'Angle':315, 'Sector':7},
            'F14':{'R':2.510, 'Z':-0.450, 'Angle':15, 'Sector':5},
            'F15':{'R':2.510, 'Z':-0.450, 'Angle':15, 'Sector':13},
            'F16':{'R':2.300, 'Z':-0.600, 'Angle':270, 'Sector':7},
            'F17':{'R':2.397, 'Z': 0.000, 'Angle':180, 'Sector':12},
            'F18':{'R':1.040, 'Z':-0.580, 'Angle':270, 'Sector':7}} 

    def _read(self):
        """
        Method to read all the signals available from
        manometers and define dictionary containing
        all the information
        """
        self.signal = {}
        for sig in self.geometry.iterkeys():
            try:
                dummy = self.Ioc(sig)
                self.signal[sig] = {'t':dummy.time,
                                    'data':dummy.data,
                                    'R':self.geometry[sig]['R'],
                                    'Z':self.geometry[sig]['Z'],
                                    'Angle':self.geometry[sig]['Angle'],
                                    'Sector':self.geometry[sig]['Sector']}
            except:
                self.signal[sig] = {'t':None,
                                    'data':None,
                                    'R':self.geometry[sig]['R'],
                                    'Z':self.geometry[sig]['Z'],
                                    'Angle':self.geometry[sig]['Angle'],
                                    'Sector':self.geometry[sig]['Sector']}

    def _readGas(self):

        self.gas={'D2':{'t':self.Uvs('D_tot').time, 'data':self.Uvs('D_tot').data},
                  'N2':{'t':self.Uvs('N_tot').time, 'data':self.Uvs('N_tot').data}}

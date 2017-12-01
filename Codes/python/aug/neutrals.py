import dd
import numpy
import logging
import os
import equilibrium
import map_equ
import matplotlib as mpl


class Neutrals(object):
    """
    Class to handle all the information concerning
    neutral measurements from the gauges and details
    from fueling. In case neutrals have been computed
    it also load the appropriate neutral estimate from
    saved file

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
        self._valves()
        self._readGas()
        self.Uvs.close()
        # now the directory where eventually the neutrals resides
        try:
            _path = os.path.abspath(os.path.join(os.path.join(
                        __file__, '../../../..'),
                                                 'Experiments/AUG/analysis/data/neutrals/%5i' % self.shot))
            data = numpy.load(_path+'/n0_avg.npy')
            self.n0 = data[:, 0]
            self.n0Err = data[:, 1]
            self.n0Time = numpy.loadtxt(_path+'/time_brillanza.txt')
        except:
            logging.warning('File not found')
            pass

        # now also the equilibrium since it will be useful for
        # the plot of gauges and valves location
        self.Eq = equilibrium.equilibrium(device='AUG', time=2, shot=self.shot)
        self.rg, self.zg = map_equ.get_gc()

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
        self.geometry = {
            'F01': {'R': 1.423, 'Z': -1.106, 'Angle': 90, 'Sector': 3},
            'F02': {'R': 1.837, 'Z': -0.873, 'Angle': 270, 'Sector': 3},
            'F03': {'R': 1.390, 'Z': -1.178, 'Angle': 270, 'Sector': 7},
            'F04': {'R': 1.423, 'Z': -1.106, 'Angle': 90, 'Sector': 7},
            'F05': {'R': 1.837, 'Z': -0.873, 'Angle': 180, 'Sector': 11},
            'F06': {'R': 1.837, 'Z': -0.873, 'Angle': 270, 'Sector': 11},
            'F07': {'R': 1.423, 'Z': -1.106, 'Angle': 90, 'Sector': 15},
            'F08': {'R': 1.619, 'Z': -1.110, 'Angle': 171, 'Sector': 15},
            'F09': {'R': 1.390, 'Z': -1.178, 'Angle': 270, 'Sector': 15},
            'F10': {'R': 1.220, 'Z': -1.100, 'Angle': 45, 'Sector': 15},
            'F11': {'R': 1.040, 'Z': -0.580, 'Angle': 270, 'Sector': 15},
            'F12': {'R': 2.135, 'Z': -1.010, 'Angle': 90, 'Sector': 15},
            'F13': {'R': 2.300, 'Z': 0.700, 'Angle': 315, 'Sector': 7},
            'F14': {'R': 2.510, 'Z': -0.450, 'Angle': 15, 'Sector': 5},
            'F15': {'R': 2.510, 'Z': -0.450, 'Angle': 15, 'Sector': 13},
            'F16': {'R': 2.300, 'Z': -0.600, 'Angle': 270, 'Sector': 7},
            'F17': {'R': 2.397, 'Z': 0.000, 'Angle': 180, 'Sector': 12},
            'F18': {'R': 1.040, 'Z': -0.580, 'Angle': 270, 'Sector': 7}}

    def _valves(self):
        """
        Simple hidden method to get all the names of the available
        valves in UVS signal
        """
        Names = self.Uvs.getObjectNames()
        self._allValves = []
        for n in Names.viewvalues():
            if n[:2] == 'CF':
                self._allValves.append(n)

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
                self.signal[sig] = {'t': dummy.time,
                                    'data': dummy.data,
                                    'R': self.geometry[sig]['R'],
                                    'Z': self.geometry[sig]['Z'],
                                    'Angle': self.geometry[sig]['Angle'],
                                    'Sector': self.geometry[sig]['Sector']}
            except BaseException:
                self.signal[sig] = {'t': None,
                                    'data': None,
                                    'R': self.geometry[sig]['R'],
                                    'Z': self.geometry[sig]['Z'],
                                    'Angle': self.geometry[sig]['Angle'],
                                    'Sector': self.geometry[sig]['Sector']}

    def _readGas(self):
        """
        Read the gas injected by all the used valves as well as the total
        amount of gas for both the D2 and the N2. For the valves assumes that
        the valve is closed if the mean of the signal is below 1e19
        """

        self.gas = {}
        self.valves= []
        for v in self._allValves:
            if self.Uvs(v).data.mean() > 1e19:
                self.gas[v] = {'t': self.Uvs(v).time, 'data': self.Uvs(v).data}
                self.valves.append(v)

        self.gas['D2'] = {'t': self.Uvs('D_tot').time, 'data': self.Uvs('D_tot').data}
        self.gas['N2'] = {'t': self.Uvs('N_tot').time, 'data': self.Uvs('N_tot').data}

        # also categorize the valves according to location
        # now determine which are the valves used for Gas Injection
        # and plot arrows with different colors and labels accordingly
        self.Div = []
        self.Top = []
        self.Equatorial = []
        self.TopEquatorial = []
        for key in self.valves:
            if key[2] == 'D':
                self.Div.append(key)
            elif key[2] == 'A':
                self.Equatorial.append(key)
            elif key[2] == 'C':
                self.TopEquatorial.append(key)
            elif key[2] == 'F':
                self.Top.append(key)
            else:
                log.warnings('I do not understand location of valves ' + key)



    def plotSetup(self, time=3):
        """
        Plot location of Gauges and of used valves
        together with the equilibrium 

        """
        self.Eq.set_time(time)
        fig, ax = mpl.pylab.subplots(figsize=(5, 8), nrows=1, ncols=1)
        fig.subplots_adjust(left=0.15)
        for key in self.rg.iterkeys():
            ax.plot(self.rg[key], self.zg[key], 'k', alpha=0.5)
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN,
                   numpy.linspace(0, 0.95, 10), colors='gray', linestyles='-')
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN,
                   [1], colors='red', linestyles='-', linewidths=2)
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN,
                   numpy.linspace(1.01, 1.1, 5), colors='gray', linestyles='--',
                   linewidths=2)
        ax.set_xlabel('R (m)')
        ax.set_ylabel('Z (m)')
        ax.set_xlim(0.7, 3.2)
        ax.set_ylim(-1.7, 1.7)
        # now plot the location of the gauges and the corresponding names
        for key in self.geometry.keys():
            ax.plot(self.geometry[key]['R'], self.geometry[key]['Z'], 'pr', markersize=10)
            ax.annotate(key, xy=(self.geometry[key]['R'], self.geometry[key]['Z']),
                        xycoords='data',
                        xytext=(self.geometry[key]['R']+0.01, self.geometry[key]['Z']+0.01), 
                        textcoords='data', fontsize=10)
        ax.set_aspect('equal')

        if len(self.Div) != 0:
            for d, i in zip(self.Div, range(len(self.Div))):
                ax.text(2.6, -0.7-i*0.1, d, color='#2D5F73')
            ax.arrow(1.4, -1.4, 0, 0.3, lw=4, color='#2D5F73')

        if len(self.Top) != 0:
            for d, i in zip(self.Top, range(len(self.Top))):
                ax.text(0.8, 1.48-i*0.1, d, color='#981C2D')
            ax.arrow(1.25, 1.4, 0, -0.3, lw=4, color='#981C2D')

        if len(self.Equatorial) != 0:
            for d, i in zip(self.Equatorial, range(len(self.Equatorial))):
                ax.text(2.6, 0.1-i*0.1, d, color='#BF3D6A')
            ax.arrow(2.8, 0, -0.3, 0, lw=4, color='#BF3D6A')

        if len(self.TopEquatorial) != 0:
            for d, i in zip(self.TopEquatorial, range(len(self.TopEquatorial))):
                ax.text(2.55, 1.1-i*0.1, d, color='#F4AB29')
            ax.arrow(2.5, 0.9, -0.3, 0, lw=4, color='#F4AB29')

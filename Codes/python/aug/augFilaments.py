import timeseries
import dd
import eqtools
import numpy as np
import matplotlib as mpl
import copy


class Filaments(object):
    """
    Class for analysis of Filaments from the HFF probe on the
    midplane manipulator

    Parameters
    ----------
    Shot : :obj: `int`
        Shot number
    Xprobe : :obj: `float`
        Probe starting point decided on a shot to shot basis during
        the radial scan
    Probe: :obj: `string`
        String indicating the probe head used. Possible values are 
        - `HFF`: for High Heat Flux probe for fluctuation measurements
        - 'IPP': Innsbruck-Padua probe head

    Requirements
    ------------
    timeseries : Class in https://github.com/nicolavianello/topic21
    dd : Class on AUG toks cluster for AUG signal reading
    eqtools : https://github.com/PSFCPlasmaTools/eqtools

    """

    def __init__(self, shot, Probe='HFF', Xprobe=None):
        self.shot = shot
        # load the equilibria
        self.Eq = eqtools.AUGDDData(self.shot)
        self.Xprobe = Xprobe
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
        self.Xlim = 1738
        RZgrid = {'m01':{'x':10, 'z':16.5, 'r':4},
                  'm02':{'x':3.5, 'z':16.5, 'r':4},
                  'm03':{'x':-3.5, 'z':16.5, 'r':4}, 
                  'm04':{'x':-10, 'z':16.5, 'r':4},
                  'm05':{'x':10, 'z':5.5, 'r':0},
                  'm06':{'x':3.5, 'z':5.5, 'r':0},
                  'm07':{'x':-3.5, 'z':5.5, 'r':0},
                  'm08':{'x':-10, 'z':5.5, 'r':0},
                  'm09':{'x':10, 'z':-5.5, 'r':4},
                  'm10':{'x':3.5, 'z':-5.5, 'r':2},
                  'm11':{'x':-3.5, 'z':-5.5, 'r':2},
                  'm12':{'x':-10, 'z':-5.5, 'r':2}, 
                  'm13':{'x':10, 'z':-16.5, 'r':8},
                  'm14':{'x':3.5, 'z':-16.5, 'r':8},
                  'm15':{'x':-3.5, 'z':-16.5, 'r':8},
                  'm16':{'x':-10, 'z':-16.5, 'r':8}}
        self.RZgrid = {}
        for probe in RZgrid.iterkeys():
            x, y = self._rotate((RZgrid[probe]['x'], RZgrid[probe]['z']), np.radians(angle))
            self.RZgrid[probe] = {'r':RZgrid[probe]['r'], 'z':self.Zmem + y, 'x':x}


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
                self.isArr[n] = dict([('data', -Mhc(n).data),
                                      ('t', Mhc(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z']),
                                      ('x', self.RZgrid[n[-3:]]['x']), 
                                      ('name', n)])
            elif n[:5] == 'Ufl_m':
                self.vfArr[n] = dict([('data', Mhc(n).data),
                                      ('t', Mhc(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z']),
                                      ('x', self.RZgrid[n[-3:]]['x']), 
                                      ('name', n)])

        for n in namesG.itervalues():
            if n[:6] == 'Isat_m':
                self.isArr[n] = dict([('data', -Mhg(n).data),
                                      ('t', Mhg(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z']),
                                      ('x', self.RZgrid[n[-3:]]['x']), 
                                      ('name', n)])
            elif n[:5] == 'Ufl_m':
                self.vfArr[n] = dict([('data', Mhg(n).data),
                                      ('t', Mhg(n).time),
                                      ('r', self.RZgrid[n[-3:]]['r']),
                                      ('z', self.RZgrid[n[-3:]]['z']),
                                      ('x', self.RZgrid[n[-3:]]['x']), 
                                      ('name', n)])

        # generale also the list of names for is and vf
        self.isName = []
        for p in self.isArr.itervalues():
            self.isName.append(p['name'])
        self.vfName = []
        for p in self.vfArr.itervalues():
            self.vfName.append(p['name'])

        Mhc.close()
        Mhg.close()

    def plotProbeSetup(self, save=False):
        """
        Method to plot probe head with color code according to
        the type of measurement existing

        Parameters
        ----------
        save : Boolean
            If True save a pdf file with the probe configuration in the 
            working directory. Default is False

        """

        ProbeHead = mpl.pyplot.Circle((0, 0), 32.5, ec='k', fill=False, lw=3)

        fig, ax = mpl.pylab.subplots(figsize=(6, 6), nrows=1, ncols=1)
        ax.add_artist(ProbeHead)
        for probe in self.RZgrid.iterkeys():
            if 'Isat_'+probe in self.isArr.keys():
                col='red'
            elif 'Ufl_'+probe in self.vfArr.keys():
                col='blue'
            else:
                col='black'

            tip = mpl.pyplot.Circle((self.RZgrid[probe]['x'], self.RZgrid[probe]['z']-self.Zmem), 2, fc=col)
            ax.add_artist(tip)
            ax.text(self.RZgrid[probe]['x']-10, self.RZgrid[probe]['z']-2-self.Zmem, probe,
                    fontsize=8)
        ax.set_xlim([-70, 70])
        ax.set_ylim([-70, 70])
        ax.text(-40, 60, r'I$_s$', fontsize=18, color='red')
        ax.text(-40, 50, r'V$_f$', fontsize=18, color='blue')

    def loadPosition(self):
        """
        Load the position and compute for each of the pin the
        corresponding rho values taking into account the
        (R, Z) position
        
        Parameters
        ----------
        None

        Returns
        -------
        None

        Attributes
        ----------
        Add to the self.RZgrid dictionary the time basis of the rhopoloidal
        and the corresponding values of rhop

        """

        Lsm = dd.shotfile('LSM', self.shot)
        sPos = np.abs(Lsm('S-posi').data - Lsm('S-posi').data.min())
        tPos = Lsm('S-posi').time
        # convert into absolute value according to transformation
        R = (2188 - (self.Xprobe - self.Xlim) - sPos + 100)/1e3
        # convert in Rhopoloidal
        self.rhoProbe = self.Eq.rz2psinorm(R, np.repeat(self.Zmem*1e-3), tPos, sqrt=True)
        # then we need to compute for each of the different probe tips
        # this quantity and save them into a dictionary

    def blobAnalysis(self, Probe='Isat_m01', trange=[2, 3],
                     interELM=False, block=[0.015, 0.12]):
        """
        Given the probe call the appropriate timeseries class
        for the analysis of the blobs.

        Parameters
        ----------
        Probe : :obj: `string`
            Probe name used for the analysis and eventually the
            trigger. If the name is not included among the possible
            available names print the list of available signal
            and ask for inserting the appropriate name

        trange : :obj: list
            List of the type [tmin, tmax]

        interElm : :obj: Boolean
             If true create an appropriate mask for the
             signal keeping only the interELM values (Not yet
             implemented)
        """

        # firs of all limit the isAt and vfFloat to the desired time interval
        isSignal, vfSignal = self._defineTime(trange=trange)
        self.blockmin=block[0]
        self.blockmax=block[1]
        if Probe is not self.isName + self.vfName:
            print('Available Ion saturation current signals are')
            for p in self.isName:
                print p
            print('Available Floating potential signal are')
            for p in self.vfName:
                print p

            Probe = str(raw_input('Provide the probe '))

        if Probe[:4] == 'Isat':
            # for the ion saturation current
            # we need to mask for the
            # 
            _idx = ((isSignal[Probe]['data'] > self.blockmin) &
                    (isSignal[Probe]['data'] < self.blockmax))
            dt  = (isSignal[Probe]['t'].max()-isSignal[Probe]['t'].min())/(isSignal[Probe]['t'].size-1)
            # we need to generate a dummy time basis
            tDummy = np.arange(np.count_nonzero(_idx))*dt + trange[0]
            
            self.blob = timeseries.Timeseries(isSignal[Probe]['data'][_idx], tDummy)
            self.refSignal = Probe
        else:
            self.blob = timeseries.Timeseries(vfSignal[Probe]['data'], isSignal[Probe]['t'])
            self.refSignal = Probe

    def _defineTime(self, trange=[2, 3]):
        """
        Internal use to limit to a given time interval
        
        Parameters
        ----------
        trange : :obj: `ndarray`
            2D element of the form [tmin, tmax]

        Returns
        -------
        isOut : Dictionary
            Contains all the ion saturation
            currents in the limited time interval

        vfOut : Dictionary
            Contains all the floating potential 
            in the limited time interval
        """

        isOut = copy.deepcopy(self.isArr)
        vfOut = copy.deepcopy(self.vfArr)
        for p in isOut.itervalues():
            _idx = ((p['t'] >= trange[0]) & (p['t'] <= trange[1]))
            p['data'] = p['data'][_idx]
            p['t'] = p['t'][_idx]
        for p in vfOut.itervalues():
            _idx = ((p['t'] >= trange[0]) & (p['t'] <= trange[1]))
            p['data'] = p['data'][_idx]
            p['t'] = p['t'][_idx]
        return isOut, vfOut

    def _rotate(self, point, angle):
        """
        Provide rotation of a point in a plane with respect to origin
        """
        px, py = point
        qx = np.cos(angle) * px - py *np.sin(angle)
        qy = np.sin(angle) * px + np.cos(angle) * py
        return qx, qy

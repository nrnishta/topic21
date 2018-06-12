from __future__ import print_function
import MDSplus as mds
import numpy as np
import xarray as xray


class Gas(object):

    def __init__(self, shot, gases=('D2', 'He', 'N2'),
                 valves=np.linspace(1, 3, dtype='int')):
        """
        Class to read the appropriate Gas injected
        taking into account proper calibration for
        the different Gases

        Parameters
        ----------
        Shot :
            Shot number
        valves :
            The number of valves. If not given then it will
            read all the valves
        """

        self.shot = shot
        self.gas = np.atleast_1d(gases)
        self.valves = np.asarray(np.atleast_1d(valves))
        # define the calibration factor dictionary for the
        # different
        self.Calibration = {
            1: {'D2': {'A': 50.5627, 'B': 0.4237, 'C': 2.0000},
                'He': {'A': 72.9584, 'B': 0.2983, 'C': 2.0000}},
            2: {'D2': {'A': 30.9550, 'B': 0.2623, 'C': 3.0000},
                'He': {'A': 158.0017, 'B': 0.1066, 'C': 2.600},
                'Ne': {'A': 13.9691, 'B': 0.2695, 'C': 2.5000},
                'N2': {'A': 7.92650, 'B': 0.4535, 'C': 2.5500}},
            3: {'D2': {'A': 27.0899, 'B': 0.3016, 'C': 2.8047},
                'He': {'A': 72.0665, 'B': 0.1497, 'C': 2.4616},
                'Ne': {'A': 13.1245, 'B': 0.2847, 'C': 2.3536},
                'N2': {'A': 8.23560, 'B': 0.4576, 'C': 2.4000}}}
        # open the Tree
        self._Tree = mds.Tree('tcv_shot', self.shot)
        self._readall()
        self._Tree.quit

    def _readall(self):
        """
        For each of the requested valves and gas read the
        appropriate calibrated values
        """
        # read all the raw data
        self._readraw()
        flow = []
        for _gas, _v, _idx in zip(
                self.gas, self.valves, range(len(self.gas))):
            flow.append(self._calibrateflow(_gas, _v, self.raw[_idx, :]))
        # now create a DataArray
        flow = np.asarray(flow)
        self.flow = xray.DataArray(flow.transpose(),
                                   coords=[self.time, self.valves],
                                   dims=['time', 'Valves'])
        self.flow.attrs['Gas'] = self.gas

    def _readraw(self):
        """
        This is the routine to really read the data
        """
        valvNumber = self.valves*3+1
        raw = []
        for v in valvNumber:
            bN = self._Tree.getNode(
                r'\diagz::trcf_gaz:channel_{:03}'.format(v))
            raw.append(bN.data())

        self.raw = np.asarray(raw)
        self.time = bN.getDimensionAt().data()

    def _calibrateflow(self, gas, valves, flow):
        """
        Given the gas as a string and the flow it
        properly calibrated the signal
        """
        if (gas == 'Ne') & (valves == 1):
            print('Valve 1 not calibrated for Ne')
        elif (gas == 'N2') & (valves == 1):
            print('Valve 1 not calibrated for N2')
        else:
            # from Volts to mbar*l/s
            flow[flow < 0] = 0
            out = self.Calibration[valves][gas]['A']*(np.power(
                (np.power(self.Calibration[valves][gas]['B'] * flow,
                          self.Calibration[valves][gas]['C'])+1),
                1./self.Calibration[valves][gas]['C'])-1)
            out *= 0.1/(1.38e-23*293.)
            return out

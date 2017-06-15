from __future__ import print_function
import MDSplus as mds
from scipy.interpolate import UnivariateSpline
import numpy as np
from smooth import smooth


class Baratrons(object):
    """
    Class to load the midplane and target pressure
    from MDSplus tree
    """
    def __init__(self, shot, window=100):
        self.window = window
        self.shot = shot
        self._tree = mds.Tree('tcv_shot', self.shot)
        self._loadmidplane(window=self.window)
        self._loadtarget(window=self.window)
        if (self._Midplane is not None) and (self._Target is not None):
            self._computecompression()
        else:
            print('Either Midplane or Target baratrons not available')

    def _loadmidplane(self, window=100):
        """
        Load data from midplane baratron gauges
        """
        if self.shot > 55358:
            try:
                Midplane = self._tree.getNode(
                    r'\base::trch_baratron:channel_001')
                self._Mtime = Midplane.getDimensionAt().data()
                self._Midplane = -Midplane.data()*0.267
                self._Midplane -= self._Midplane[
                    np.where(((self._Mtime >= -6) &
                              (self._Mtime <= -4)))[0]].mean()
                self._Midplane = smooth(self._Midplane,
                                        window_len=window)
            except:
                self._Midplane = None
                self._Mtime = None
                print('Midplane baratron not available')
        else:
            try:
                Midplane = self._tree.getNode(
                    r'\base::trch_ece_pols:channel_002')
                self._Mtime = Midplane.getDimensionAt().data()
                self._Midplane = -Midplane.data()*0.267
                self._Midplane -= self._Midplane[
                    np.where(((self._Mtime >= -6) &
                              (self._Mtime <= -4)))[0]].mean()
                self._Midplane = smooth(self._Midplane,
                                        window_len=window)
            except:
                self._Midplane = None
                self._Mtime = None
                print('Midplane baratron not available')

    def _loadtarget(self, window=100):
        """
        Load the data from the target baratron gauges
        """
        self._Target = self._tree.getNode(
            r'\diagz::trcf_gaz:channel_012').data()*0.267
        self._Ttime = self._tree.getNode(
            r'\diagz::trcf_gaz:channel_012').getDimensionAt().data()
        self._Target -= self._Target[
            np.where((self._Ttime >= -6) & (self._Ttime <= -4.))].mean()
        self._Target = smooth(self._Target,
                              window_len=window)

    def _computecompression(self):
        """
        Compute compression and recalculate both midplane and
        target on the same time basis
        """
        tmin = np.maximum(self._Ttime.min(), self._Mtime.min())
        tmax = np.minimum(self._Ttime.max(), self._Mtime.max())
        # now time the same for the two baratrons
        idxTarget = np.where((self._Ttime >= tmin) &
                             (self._Ttime <= tmax))
        idxMidplane = np.where((self._Mtime >= tmin) &
                               (self._Mtime <= tmax))
        # determine which is the variable with the larger
        # number of points and then interpolate
        if (self._Ttime[idxTarget].size <= self._Mtime[idxMidplane].size):
            S = UnivariateSpline(self._Mtime[idxMidplane],
                                 self._Midplane[idxMidplane], s=0)
            self.Midplane = S(self._Ttime[idxTarget])
            self.Target = self._Target[idxTarget]
            self.t = self._Ttime[idxTarget]
            self.Compression = self.Target/self.Midplane
        else:
            S = UnivariateSpline(self._Ttime[idxTarget],
                                 self._Target[idxTarget], s=0)
            self.Target = S(self._Mtime[idxMidplane])
            self.Target = self._Midplane[idxMidplane]
            self.t = self._Midplane[idxMidplane]
            self.Compression = self.Target/self.Midplane

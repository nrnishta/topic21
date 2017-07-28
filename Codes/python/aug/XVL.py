import numpy as np
import matplotlib.pylab as plt
import sys, os
sys.path.append('/afs/ipp/u/mcavedon/repository/python')
from Debug.Ipsh import *
from dd import ddPlus
import idlwrap
import logging

__idl_path__ = '/afs/ipp/u/sprd/idlpro/lev1/'
__idl_routine__ = 'collect_sig_for_heads_from_all_xvl'
"""
pro collect_sig_for_heads_from_all_xvl, $
   shot, signame, heads, $
   data, time, $
   data_diag, time_diag, nt_diag, $
   tran = tran, $
   midplane=midplane, $
   dt_smooth=dt_smooth, $
   los_info = los_info, $
   sig_info=sig_info,  $
   error = error
"""


class XVL:
    def __init__(self, signal, head, shot, dt_smooth=None):
        """
        Port of the IDL routine to read out from the XVL
        shotifles of the divertor spectroscopy

        Input:
            signal: signal name to read (e.g. 'Ne')
            head: head to select (e.g. 'ROV')
            shot: shot number
        """
        # Initialize IDL session
        self.idl = idlwrap.idl()
        # Define class variables and also in idl
        self.shot = shot
        self.idl.shot = shot
        self.head = head
        self.idl.head = [head,]
        self.signal = signal
        self.idl.signal = signal
        if dt_smooth != None:
            self.idl.dt_smooth = dt_smooth
        # Read data
        self._read()

    def _read(self):
        # Compile programm
        self.idl('.compile '+__idl_path__+__idl_routine__+'.pro')
        # Pre requisites
        self.idl("defsysv,'!LIBDDWW','/afs/ipp/aug/ads/lib64/@sys/libddww8.so',0")
        # Build string to run
        toRun = __idl_routine__+',shot,signal,head,data,time,data_diag,time_diag,'+\
                'nt_diag,tran=tran,midplane=midplane,dt_smooth=dt_smooth,'+\
                'los_info=los_info,sig_info=sig_info,error=error'
        self.idl(toRun)
        # Build a signal group with the data
        self.data = ddPlus.signalGroup(data=self.idl.data,time=self.idl.time,name=self.signal)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    shot = 33985
    xvl = XVL('N_1_3995','ROV',shot)
    import scipy.ndimage
    data = scipy.ndimage.zoom(xvl.idl.data, 3)
    plt.imshow(np.nan_to_num(np.log(data)),vmin=35,aspect='auto',\
               extent=[xvl.idl.time.min(),xvl.idl.time.max(),xvl.idl.los_info['x'].min(),xvl.idl.los_info['x'].max()])
    plt.show()
    ipsh()

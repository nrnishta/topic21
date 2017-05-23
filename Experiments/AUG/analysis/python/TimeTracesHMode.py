from __future__ import print_function
import sys
sys.path.append('/afs/ipp-garching.mpg.de/home/n/nvianell/analisi/topic21/Codes/python/general/pyEquilibrium')
from equilibrium import *
import dd
import map_equ
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc("font", size=18)
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Tahoma']})
fig, ax = mpl.pylab.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)
fig.subplots_adjust(bottom=0.15, left=0.15, wspace=0.05)
# time traces of the power
tECRH = np.asarray([1.75, 1.75, 2.25, 2.25, 6.3, 6.3])
pEcrh = np.asarray([0, 0.65, 0.65, 1.3, 1.3, 0])
ax[0].plot(tECRH, pEcrh, 'r', lw=3, label='ECRH')
tNBI = np.asarray([2.25, 2.25, 6.3, 6.3])
NBI = np.asarray([0, 4, 4, 0])
ax[0].plot(tNBI, NBI, 'b', lw=3, label='NBI')
ax[0].legend(loc='best', numpoints=1, frameon=False)
ax[0].set_ylabel(r'[MW]')
ax[0].axes.get_xaxis().set_visible(False)
ax[0].set_xlim([0, 7])
ax[0].set_ylim([0, 5])
ax[0].axvline(1.75, ls='-.', lw=3, color='green')
ax[0].axvline(2.25, ls='-.', lw=3, color='green')
ax[0].axvline(2.75, ls='-.', lw=3, color='green')
# time traces of D2/N
tD2 = np.asarray([1.75, 1.75, 2.75, 6.3, 6.3])
D2 = np.asarray([0, 3.61, 3.61, 16, 0])
N2 = np.asarray([0, 4.3, 4.3, 0])
tN2 = np.asarray([3, 3, 6.3, 6.3])
ax[1].plot(tD2, D2, 'k-', lw=3, label=r'D$_2$')
ax[1].plot(tN2, N2, 'r-', lw=3, label=r'N$_2$')
ax[1].legend(loc='best', numpoints=1, frameon=False)
ax[1].set_xlabel(r't[s]')
ax[1].set_ylabel(r'[10$^{21}$]')
ax[1].set_ylim([0, 20])
ax[1].axvline(1.75, ls='-.', lw=3, color='green')
ax[1].axvline(2.25, ls='-.', lw=3, color='green')
ax[1].axvline(2.75, ls='-.', lw=3, color='green')
ax[1].text(1.73, 15, '1.75', color='green', fontsize=14)
ax[1].text(2.23, 15, '2.25', color='green', fontsize=14)
ax[1].text(2.73, 15, '2.25', color='green', fontsize=14)
mpl.pylab.savefig('../pdfbox/HModeTiming.pdf', bbox_to_inches='tight')

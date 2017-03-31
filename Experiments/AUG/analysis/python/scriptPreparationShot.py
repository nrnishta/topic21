# script in order to compare heating fueling and equilibria
# for shot at different current/bt/q95 for the proper scan
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


shotList = (30269, 29311, 29309, 29315)
colorL = ('#82A17E', '#1E4682', '#DD6D3D', '#69675E')
fig, ax  = mpl.pyplot.subplots(figsize=(18, 14), nrows = 4, ncols = 2)
fig.subplots_adjust(wspace=0.3, right=0.6, top=0.95, bottom=0.12, left=0.1)
_xlim = [0, 7]
axE = fig.add_axes([0.6, 0.3, 0.4, 0.4])
for shot, col, i in zip(shotList, colorL, range(len(shotList))):
    diag = dd.shotfile('MAG', shot)
    iP = diag('Ipa')
    diag.close()
    # load the toroidal field
    diag = dd.shotfile('MAI', shot)
    bT = diag('BTF')
    diag.close()

    # load the q95
    diag = dd.shotfile('FPG', shot)
    q95 = diag('q95')
    diag.close()
    # load the edge density
    diag = dd.shotfile('DCN', shot)
    enE = diag('H-5')
    diag.close()
    # greenwald fraction
    diag = dd.shotfile('TOT', shot)
    nG = diag('n/nGW')
    diag.close()    
    # load the Prad
    diag = dd.shotfile('BPD', shot)
    pRad = diag('Pradtot')
    diag.close()
    # load the total fueling
    diag = dd.shotfile('UVS', shot)
    iD2 = diag('D_tot')
    diag.close()
    # load the equilibrium
    Eq = equilibrium(device='AUG', shot=shot, time=2.9)

    # load the current


    ax[0, 0].plot(iP.time, iP.data/1e6, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[0, 0].axes.get_xaxis().set_visible(False)
    ax[0, 0].set_ylabel(r'I$_p$ [MA]')
    ax[0, 0].set_xlim(_xlim)


    ax[1, 0].plot(bT.time, bT.data, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[1, 0].axes.get_xaxis().set_visible(False)
    ax[1, 0].set_ylabel(r'B$_t$ [T]')
    ax[1, 0].set_xlim(_xlim)

    ax[2, 0].plot(q95.time, np.abs(q95.data), col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[2, 0].axes.get_xaxis().set_visible(False)
    ax[2, 0].set_ylabel(r'q$_{95}$')
    ax[2, 0].set_xlim(_xlim)
    ax[2, 0].set_ylim([2.5, 10])
    try:
    # load the ECRH
        diag = dd.shotfile('ECS', shot)
        pEch = diag('PECRH')
        diag.close()
        ax[3, 0].plot(pEch.time, pEch.data/1e3, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
        ax[3, 0].set_xlabel(r't [s]')
        ax[3, 0].set_ylabel(r'P$_{ECH}$ [kW]')
        ax[3, 0].set_xlim(_xlim)
    except:
        print('No ECH for shot %5i' % shot)

    ax[0, 1].plot(enE.time, enE.data/1e19, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[0, 1].axes.get_xaxis().set_visible(False)
    ax[0, 1].set_ylabel(r'n$_e [10^{19}$m$^{-2}]$ Edge')
    ax[0, 1].set_xlim(_xlim)

    ax[1, 1].plot(nG.time, nG.data, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[1, 1].axes.get_xaxis().set_visible(False)
    ax[1, 1].set_ylabel(r'n/n$_G$')
    ax[1, 1].set_xlim(_xlim)
    ax[1, 1].set_ylim([0, 1])

    ax[2, 1].plot(pRad.time, pRad.data/1e3, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[2, 1].axes.get_xaxis().set_visible(False)
    ax[2, 1].set_ylabel(r'P$_{rad}$ tot [kW]')
    ax[2, 1].set_xlim(_xlim)
    ax[2, 1].set_ylim([0, 1.2e3])
    
    ax[3, 1].plot(iD2.time, iD2.data/1e21, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[3, 1].set_xlabel(r't [s]')
    ax[3, 1].set_ylabel(r'D$_{2} [10^{21}]$')
    ax[3, 1].set_xlim(_xlim)

    # ora plottiamo l'equilibrio in un asse a parte
    if i ==0:
        rg, zg = map_equ.get_gc()
        for key in rg.iterkeys():
            axE.plot(rg[key], zg[key], 'k')
    axE.contour(Eq.R, Eq.Z, Eq.psiN(Eq.R, Eq.Z), np.linspace(0, 1, 5), colors=col)
    axE.contour(Eq.R, Eq.Z, Eq.psiN(Eq.R, Eq.Z), np.linspace(1, 1.09, 3),
                colors=col, linestyles='--')
    axE.set_xlabel('R')
    axE.set_ylabel('Z')
    axE.set_aspect('equal')

ax[0, 0].legend(loc='best', numpoints=1, fontsize=10, frameon=False)
mpl.pylab.savefig('../pdfbox/ComparisonCurrentQScan.pdf', bbox_to_inches='tight')

# check current scan at constant q95
shotL = (29311, 30269, 29315)
colorL = ('#1E4682', '#DD6D3D', '#69675E')
fig, ax  = mpl.pyplot.subplots(figsize=(13, 8), nrows = 2, ncols = 1)
fig.subplots_adjust(wspace=0.3, left=0.6, top=0.95, bottom=0.12, right=0.98)
_xlim = [0, 7]
axE = fig.add_axes([0.05, 0.3, 0.5, 0.5])
for shot, col, i in zip(shotL, colorL, range(len(shotL))):
    diag = dd.shotfile('MAG', shot)
    iP = diag('Ipa')
    diag.close()
    # load the toroidal field
    diag = dd.shotfile('MAI', shot)
    bT = diag('BTF')
    diag.close()
    # load the equilibrium
    Eq = equilibrium(device='AUG', shot=shot, time=2.9)
    ax[0].plot(iP.time, iP.data/1e6, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].set_ylabel(r'I$_p$ [MA]')
    ax[0].set_xlim(_xlim)


    ax[1].plot(bT.time, bT.data, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[1].set_xlabel(r't[s]')
    ax[1].set_ylabel(r'B$_t$ [T]')
    ax[1].set_xlim(_xlim)
    # ora plottiamo l'equilibrio in un asse a parte
    if i ==0:
        rg, zg = map_equ.get_gc()
        for key in rg.iterkeys():
            axE.plot(rg[key], zg[key], 'k')
    axE.contour(Eq.R, Eq.Z, Eq.psiN(Eq.R, Eq.Z), np.linspace(0, 1, 5), colors=col)
    axE.contour(Eq.R, Eq.Z, Eq.psiN(Eq.R, Eq.Z), np.linspace(1, 1.09, 3),
                colors=col, linestyles='--')
    axE.set_xlabel('R')
    axE.set_ylabel('Z')
    axE.set_aspect('equal')
ax[0].legend(loc='best', numpoints=1, fontsize=10, frameon=False)
mpl.pylab.savefig('../pdfbox/CurrentScanConstantQ95.pdf', bbox_to_inches='tight')

# constant scan at constant Bt
shotL = (29302, 30269, 28738)
colorL = ('#1E4682', '#DD6D3D', '#69675E')
fig, ax  = mpl.pyplot.subplots(figsize=(13, 8), nrows = 2, ncols = 1)
fig.subplots_adjust(wspace=0.3, left=0.6, top=0.95, bottom=0.12, right=0.98)
_xlim = [0, 7]
axE = fig.add_axes([0.05, 0.3, 0.5, 0.5])
for shot, col, i in zip(shotL, colorL, range(len(shotL))):
    diag = dd.shotfile('MAG', shot)
    iP = diag('Ipa')
    diag.close()
    diag = dd.shotfile('FPG', shot)
    q95 = diag('q95')
    diag.close()
    # load the equilibrium
    Eq = equilibrium(device='AUG', shot=shot, time=2.9)
    ax[0].plot(iP.time, iP.data/1e6, col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].set_ylabel(r'I$_p$ [MA]')
    ax[0].set_xlim(_xlim)


    ax[1].plot(q95.time, np.abs(q95.data), col, ls='-', lw=1.7, label=r'Shot # %5i' % shot)
    ax[1].set_xlabel(r't[s]')
    ax[1].set_ylabel(r'q$_{95}$')
    ax[1].set_xlim(_xlim)
    ax[1].set_ylim([2, 6])
    # ora plottiamo l'equilibrio in un asse a parte
    if i ==0:
        rg, zg = map_equ.get_gc()
        for key in rg.iterkeys():
            axE.plot(rg[key], zg[key], 'k')
    axE.contour(Eq.R, Eq.Z, Eq.psiN(Eq.R, Eq.Z), np.linspace(0, 1, 5), colors=col)
    axE.contour(Eq.R, Eq.Z, Eq.psiN(Eq.R, Eq.Z), np.linspace(1, 1.09, 3),
                colors=col, linestyles='--')
    axE.set_xlabel('R')
    axE.set_ylabel('Z')
    axE.set_aspect('equal')
ax[0].legend(loc='best', numpoints=1, fontsize=10, frameon=False)
mpl.pylab.savefig('../pdfbox/CurrentScanConstantBT.pdf', bbox_to_inches='tight')

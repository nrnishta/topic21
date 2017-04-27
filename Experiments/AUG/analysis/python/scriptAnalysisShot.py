# script in order to compare heating fueling and equilibria
# for shot at different current/bt/q95 for the proper scan
from __future__ import print_function
import numpy as np
import sys
import dd
import itertools
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc("font", size=18)
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Tahoma']})
shotList = (34103, 34102, 34104)
currentL = (0.6, 0.8, 0.99)
colorL = ('#82A17E', '#1E4682', '#DD6D3D')
colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')
tList = (1.18, 2.6, 3.43)
fig = mpl.pylab.figure(figsize=(16, 12))
fig.subplots_adjust(hspace=0.25)
ax1 = mpl.pylab.subplot2grid((2, 3), (0, 0), colspan=3)
ax2 = mpl.pylab.subplot2grid((2, 3), (1, 0))
ax3 = mpl.pylab.subplot2grid((2, 3), (1, 1))
ax4 = mpl.pylab.subplot2grid((2, 3), (1, 2))
axL = (ax2, ax3, ax4)

for shot, ip, cc, i, _ax in itertools.izip(
    shotList, currentL, colorLS, range(len(shotList)), axL):
    # this is the line average density
    diag = dd.shotfile('TOT', shot)
    neAvg = diag('H-1/(2a)')
    diag.close()
    ax1.plot(neAvg.time, neAvg.data/1e20, '-', color=cc, lw=2,
             label = r'Shot # %5i' % shot +' I$_p$ = %2.1f' % ip +' MA')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 4.5])
    for t, col in zip(tList, colorL):
        ax1.axvspan(t-0.02, t+0.02, ec='none', fc=col, alpha=0.5)
    # load the Li-Beam profiles
    try:
        LiBD = dd.shotfile('LIN', shot, experiment='AUGD')
    except:
        LiBD = dd.shotfile('LIN', shot, experiment='LIBE')
    neLB = LiBD('ne').data
    neLBtime = LiBD('ne').time
    rhoP = LiBD('ne').area
    LiBD.close()
    rhoFake = np.linspace(0.9, 1.11, 200)
    for t, col in zip(tList, colorL):
        _idx = np.where((neLBtime >= t-0.02) & (neLBtime <= t+0.02))[0]
        y = np.zeros((_idx.size, 200))
        for n, _iDummy in zip(_idx, range(_idx.size)):
            S = UnivariateSpline(rhoP[n, ::-1],
                                 neLB[n, ::-1], s=0)
            y[_iDummy, :] = S(rhoFake)/S(1)
        _ax.plot(rhoFake, np.mean(y, axis=0), '-', color=col,
                      lw=3)
        _ax.fill_between(rhoFake, np.mean(y, axis=0)-
                              np.std(y, axis=0), np.mean(y, axis=0)+
                              np.std(y, axis=0),
                              facecolor=col, edgecolor='none',
                              alpha=0.5)
    _ax.set_yscale('log')
    _ax.set_xlabel(r'$\rho_p$')
    _ax.set_title(r'Shot # % 5i' %shot)
ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
ax1.set_ylabel(r'$\langle n_e \rangle [10^{20}$m$^{-3}]$')
ax1.set_title(r'I$_p$ scan at constant q$_{95}$')
ax2.set_ylabel(r'n$_e$/n$_e(\rho_p = 1)$')
mpl.pylab.savefig('../pdfbox/IpConstantq95_density.pdf',
                  bbox_to_inches='tight')    

shotList = (34105, 34102, 34106)
currentL = (0.6, 0.8, 0.99)
colorL = ('#82A17E', '#1E4682', '#DD6D3D')
colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')
tList = (1., 2.5, 3)
fig = mpl.pylab.figure(figsize=(16, 12))
fig.subplots_adjust(hspace=0.25)
ax1 = mpl.pylab.subplot2grid((2, 3), (0, 0), colspan=3)
ax2 = mpl.pylab.subplot2grid((2, 3), (1, 0))
ax3 = mpl.pylab.subplot2grid((2, 3), (1, 1))
ax4 = mpl.pylab.subplot2grid((2, 3), (1, 2))
axL = (ax2, ax3, ax4)

for shot, ip, cc, i, _ax in itertools.izip(
    shotList, currentL, colorLS, range(len(shotList)), axL):
    # this is the line average density
    diag = dd.shotfile('TOT', shot)
    neAvg = diag('H-1/(2a)')
    diag.close()
    ax1.plot(neAvg.time, neAvg.data/1e20, '-', color=cc, lw=2,
             label = r'Shot # %5i' % shot +' I$_p$ = %2.1f' % ip +' MA')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 4.5])
    for t, col in zip(tList, colorL):
        ax1.axvspan(t-0.02, t+0.02, ec='none', fc=col, alpha=0.5)
    # load the Li-Beam profiles
    try:
        LiBD = dd.shotfile('LIN', shot, experiment='AUGD')
    except:
        LiBD = dd.shotfile('LIN', shot, experiment='LIBE')
    neLB = LiBD('ne').data
    neLBtime = LiBD('ne').time
    rhoP = LiBD('ne').area
    LiBD.close()
    rhoFake = np.linspace(0.9, 1.11, 200)
    for t, col in zip(tList, colorL):
        _idx = np.where((neLBtime >= t-0.02) & (neLBtime <= t+0.02))[0]
        y = np.zeros((_idx.size, 200))
        for n, _iDummy in zip(_idx, range(_idx.size)):
            S = UnivariateSpline(rhoP[n, ::-1],
                                 neLB[n, ::-1], s=0)
            y[_iDummy, :] = S(rhoFake)/S(1)
        _ax.plot(rhoFake, np.mean(y, axis=0), '-', color=col,
                      lw=3)
        _ax.fill_between(rhoFake, np.mean(y, axis=0)-
                              np.std(y, axis=0), np.mean(y, axis=0)+
                              np.std(y, axis=0),
                              facecolor=col, edgecolor='none',
                              alpha=0.5)
    _ax.set_yscale('log')
    _ax.set_xlabel(r'$\rho_p$')
    _ax.set_title(r'Shot # % 5i' %shot)
ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
ax1.set_ylabel(r'$\langle n_e \rangle [10^{20}$m$^{-3}]$')
ax1.set_title(r'I$_p$ scan at constant B$_p$')
ax2.set_ylabel(r'n$_e$/n$_e \rho_p = 1)$')
mpl.pylab.savefig('../pdfbox/IpConstantBt_density.pdf',
                  bbox_to_inches='tight')    


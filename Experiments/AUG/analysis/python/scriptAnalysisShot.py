# script in order to compare heating fueling and equilibria
# for shot at different current/bt/q95 for the proper scan
#from __future__ import print_function
import bottleneck
import numpy as np
import sys
import dd
import itertools
import matplotlib as mpl
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
from scipy.signal import decimate
import pandas as pd
import map_equ
import equilibrium
import libes
import xarray as xray
import h5py
sys.path.append('/afs/ipp/home/n/nvianell/pythonlib/submodules/pycwt/')
sys.path.append(
    '/afs/ipp/home/n/nvianell/analisi/topic21/Codes/python/general/')
sys.path.append('/afs/ipp/home/n/nvianell/analisi/topic21/Codes/python/aug/')
sys.path.append('/afs/ipp/home/n/nvianell/pythonlib/signalprocessing/')
sys.path.append(
    '/afs/ipp/home/n/nvianell/analisi/topic21/Codes/python/general/cyFieldlineTracer')
import augFilaments
import neutrals
import peakdetect
import langmuir
from bw_filter import bw_filter
from matplotlib.colors import LogNorm
from cyfieldlineTracer import get_fieldline_tracer
import eqtools
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc("font", size=22)
mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Tahoma']})


def print_menu():
    print 30 * "-", "MENU", 30 * "-"
    print "1. Li-Beam density profile Ip scan constant q95"
    print "2. Li-Beam density profile Ip scan constant Bt"
    print "3. General figure Ip scan constant q95"
    print "4. General figure Ip scan constant Bt"
    print "5. H-Mode general figure during the scan"
    print "6. H-Mode comparison with respect to reference"
    print "7. Edge ne image vs Time/rho for constant q95"
    print "8. Edge ne image vs Time/rho for constant Bt"
    print "9. Auto-correlation time vs Ip"
    print "10. Degraded H-Mode Wmhd vs Edge density"
    print "11. Radiation profiles "
    print "12. ELM frequency vs H-5 for H-mode shots"
    print "13. Check Neutral response to Puffing in H-Mode"
    print "14. Compare Neutral compression with reference 2016"
    print "15. Compare divertor/midplane puffing w/o cryopumps"
    print "16. Compare compression with puffing from divertor/midplane w/o cryompumps"
    print "17. Compare shots same puffing with and wo cryompumps"
    print "18. Compare shots with/wo cryopumps try to match edge density"
    print "19. Compare Li-Beam contour profiles with/wo crypumps when trying to math edge density"
    print "20. Compare Li-Be contour Lower/Upper divertor puffing"
    print "21. Compare Li-Be contour same fueling with/without cryo"
    print "22. Compare Shot constant q95 same edge density"
    print "23. Compare Li-Beam contour profiles with/wo crypumps, Density and Fueling"
    print "24. Compare Li-Beam and Lambda_Div Ip scan constant q95"
    print "25. Compare Li-Beam and Lambda_Div Ip scan constant Bt"
    print "26. Compare Lparallel constant q95 constant Bt"
    print "27. Equilibria and Lparallel at constant q95"
    print "28. Equilibria and Lparallel at constant q95"
    print "29. Fluctuations and PDF during Ip scan constant q95"
    print "30. Fluctuations and PDF during Ip scan constant Bt"
    print "31. Fluctuations and PDF Cryo On/OFF"
    print "32. Fluctuations and PDF Match Cryo ON/OFF"
    print "33. Equilibria with arrow and location of Gas Puffing"
    print "34. ELMs and puffing same fueling with/without cryopumps"
    print "35. ELMS and puffing different fueling with/without cryompumps"
    print "36. Radiation front and target density movement vs Density Ip scan constant Q95"
    print "37. Radiation front and target density movement vs Density Ip scan constant BT"
    print "38. Radiation front and target density movement vs Density DifferentPuffing"
    print "39. Radiation front and target density movement vs Density Cryo ON/OFF"
    print "40. Evolution of single inter-ELM profiles cryo on/off"
    print "41. Blob size vs Lambda Current Scan constant Q95"
    print "42. Blob size vs Lambda Current Scan constant Bt"
    print "43. Blob size vs Lambda All L-Mode"
    print "44. Evolution of single inter-ELM profiles different-puffing"
    print "45. Evolution of single inter-ELM profiles different fueling H-Mode"
    print "46. Compare CAS 34278-34281"
    print "47. Compare 34276-34278 same fueling"
    print "48. Compare 34276-34278 plunges and CAS"
    print "49. Compare neutral evolution same current different field"
    print "50. Neutral density vs density and normalized greenwald fraction for Ip scan"
    print "51. Shoulder amplitude vs neutral density"
    print "52. Shoulder amplitude vs LambdaDiv"
    print "53. Shoulder amplitude vs Greenwald fraction"
    print "54. Shoulder amplitude vs Edge density fraction"
    print "55. Neutrals vs Edge density"
    print "56. Compare shots 34276-34278-34281 general"
    print "57. Compare shots 34276-34278-34281 profiles"
    print "58. Elm behavior 34276-34278-34281 "
    print "59. Radiation and peak density vs Greenwald fraction (data saved)"
    print "60. Compare profiles including L-Mode"
    print "99: End"
    print 67 * "-"


loop = True

while loop:
    print_menu()
    selection = input("Enter your choice [1-99] ")
    if selection == 1:

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
                     label=r'Shot # %5i' % shot + ' I$_p$ = %2.1f' % ip + ' MA')
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
                _ax.fill_between(rhoFake, np.mean(y, axis=0) -
                                 np.std(y, axis=0), np.mean(y, axis=0) +
                                 np.std(y, axis=0),
                                 facecolor=col, edgecolor='none',
                                 alpha=0.5)
            _ax.set_yscale('log')
            _ax.set_xlabel(r'$\rho_p$')
            _ax.set_title(r'Shot # % 5i' % shot)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'$\langle n_e \rangle [10^{20}$m$^{-3}]$')
        ax1.set_title(r'I$_p$ scan at constant q$_{95}$')
        ax2.set_ylabel(r'n$_e$/n$_e(\rho_p = 1)$')
        mpl.pylab.savefig('../pdfbox/IpConstantq95_density.pdf',
                          bbox_to_inches='tight')

    elif selection == 2:
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
                     label=r'Shot # %5i' % shot + ' I$_p$ = %2.1f' % ip + ' MA')
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
                _ax.fill_between(rhoFake, np.mean(y, axis=0) -
                                 np.std(y, axis=0), np.mean(y, axis=0) +
                                 np.std(y, axis=0),
                                 facecolor=col, edgecolor='none',
                                 alpha=0.5)
            _ax.set_yscale('log')
            _ax.set_xlabel(r'$\rho_p$')
            _ax.set_title(r'Shot # % 5i' % shot)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'$\langle n_e \rangle [10^{20}$m$^{-3}]$')
        ax1.set_title(r'I$_p$ scan at constant B$_p$')
        ax2.set_ylabel(r'n$_e$/n$_e \rho_p = 1)$')
        mpl.pylab.savefig('../pdfbox/IpConstantBt_density.pdf',
                          bbox_to_inches='tight')
    elif selection == 3:
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')

        fig, ax = mpl.pylab.subplots(figsize=(17, 15),
                                     nrows=4, ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1)
        for shot, _col, _idx in zip(shotList,
                                    colorLS, range(len(shotList))):
            diag = dd.shotfile('MAG', shot)
            ax[0, 0].plot(diag('Ipa').time, diag('Ipa').data/1e6, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_ylim([0, 1.1])
            diag.close()

            diag = dd.shotfile('DCN', shot)
            ax[1, 0].plot(diag('H-1').time, diag('H-1').data /
                          1e19, color=_col, lw=3)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel(r'$\overline{n}_e$ H-1 [10$^{19}$]')
            ax[1, 0].set_ylim([0, 10])

            ax[2, 0].plot(diag('H-1').time, diag('H-5').data /
                          1e19, color=_col, lw=3)

            ax[2, 0].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            diag.close
            ax[2, 0].set_xlim([0, 4.5])
            ax[2, 0].set_ylim([0, 6])
            ax[2, 0].axes.get_xaxis().set_visible(False)

            diag = dd.shotfile('TOT', shot)
            ax[3, 0].plot(diag('P_TOT').time, diag(
                'P_TOT').data/1e6, color=_col, lw=3)
            ax[3, 0].set_ylabel('P$_{tot}$ [MW]')
            ax[3, 0].set_xlim([0, 4.5])
            ax[3, 0].set_ylim([0, .8])
            diag.close()

            diag = dd.shotfile('BPD', shot)
            ax[0, 1].plot(diag('Prad').time, diag('Prad').data/1e6, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            ax[0, 1].set_ylabel(r'P$_{rad}$ [MW]')
            ax[0, 1].set_ylim([0, 2])
            diag.close()

            Gas = neutrals.Neutrals(shot)

            ax[1, 1].plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                          color=_col, lw=3)
            ax[1, 1].axes.get_xaxis().set_visible(False)
            ax[1, 1].set_ylabel(r'D$_2$  [10$^{21}$]')

            ax[2, 1].plot(Gas.signal['F01']['t'],
                          Gas.signal['F01']['data']/1e22, color=_col, lw=3)
            ax[2, 1].set_ylabel(r'F01 [10$^{21}$m$^{-2}$s$^{-1}$]')
            ax[2, 1].axes.get_xaxis().set_visible(False)
            diag.close
            ax[2, 1].set_xlim([0, 4.5])
            diag = dd.shotfile('MAC', shot)
            ax[3, 1].plot(diag('Tdiv').time, diag(
                'Tdiv').data, color=_col, lw=3)
            ax[3, 1].set_ylabel(r'T$_{div}$')
            ax[3, 1].set_xlim([0, 4.5])
            ax[3, 1].set_ylim([-10, 30])
            diag.close()
        ax[3, 0].set_xlabel(r't [s]')
        ax[3, 1].set_xlabel(r't [s]')
        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        mpl.pylab.savefig('../pdfbox/GeneralIpScanConstantq95.pdf',
                          bbox_to_inches='tight')

    elif selection == 4:
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')

        fig, ax = mpl.pylab.subplots(figsize=(17, 15),
                                     nrows=4, ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1)
        for shot, _col, _idx in zip(shotList,
                                    colorLS, range(len(shotList))):
            diag = dd.shotfile('MAG', shot)
            ax[0, 0].plot(diag('Ipa').time, diag('Ipa').data/1e6, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_ylim([0, 1.1])
            diag.close()

            diag = dd.shotfile('DCN', shot)
            ax[1, 0].plot(diag('H-1').time, diag('H-1').data /
                          1e19, color=_col, lw=3)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel(r'$\overline{n}_e$ H-1 [10$^{19}$]')
            ax[1, 0].set_ylim([0, 10])

            ax[2, 0].plot(diag('H-5').time, diag('H-5').data /
                          1e19, color=_col, lw=3)
            ax[2, 0].axes.get_xaxis().set_visible(False)
            ax[2, 0].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            diag.close
            ax[2, 0].set_xlim([0, 4.5])
            ax[2, 0].set_ylim([0, 6])

            diag = dd.shotfile('TOT', shot)
            ax[3, 0].plot(diag('P_TOT').time, diag(
                'P_TOT').data/1e6, color=_col, lw=3)
            ax[3, 0].set_ylabel('P$_{tot}$ [MW]')
            ax[3, 0].set_xlim([0, 4.5])
            ax[3, 0].set_ylim([0, 3])
            diag.close()

            diag = dd.shotfile('BPD', shot)
            ax[0, 1].plot(diag('Prad').time, diag('Prad').data/1e6, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            ax[0, 1].set_ylabel(r'P$_{rad}$ [MW]')
            ax[0, 1].set_ylim([0, 2])
            diag.close()

            Gas = neutrals.Neutrals(shot)

            ax[1, 1].plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                          color=_col, lw=3)
            ax[1, 1].axes.get_xaxis().set_visible(False)
            ax[1, 1].set_ylabel(r'D$_2$  [10$^{21}$]')

            ax[2, 1].plot(Gas.signal['F01']['t'],
                          Gas.signal['F01']['data']/1e21, color=_col, lw=3)

            ax[2, 1].set_ylabel(r'F01 [10$^{21}$m$^{-2}$s$^{-1}$]')
            diag.close
            ax[2, 1].set_xlim([0, 4.5])
            ax[2, 1].axes.get_xaxis().set_visible(False)
            diag = dd.shotfile('MAC', shot)
            ax[3, 1].plot(diag('Tdiv').time, diag(
                'Tdiv').data, color=_col, lw=3)
            ax[3, 1].set_ylabel(r'T$_{div}$')
            ax[3, 1].set_xlim([0, 4.5])
            ax[3, 1].set_ylim([-10, 30])

        ax[3, 0].set_xlabel(r't [s]')
        ax[3, 1].set_xlabel(r't [s]')
        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        mpl.pylab.savefig('../pdfbox/GeneralIpScanConstantBt.pdf',
                          bbox_to_inches='tight')

    elif selection == 5:
        shotList = (34107, 34108, 34115)
        for shot in shotList:
            # open the plot
            fig, ax = mpl.pylab.subplots(figsize=(17, 15),
                                         nrows=3, ncols=2, sharex=True)
            fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1)
            # current
            diag = dd.shotfile('MAG', shot)
            ax[0, 0].plot(diag('Ipa').time, diag('Ipa').data/1e6, color='k', lw=3,
                          label=r'# %5i' % shot)
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_ylim([0, 1.1])
            ax[0, 0].set_title(r'# %5i' % shot)
            diag.close()
            # power
            diag = dd.shotfile('TOT', shot)
            ax[1, 0].plot(diag('PNBI_TOT').time, diag('PNBI_TOT').data/1e6, color='red',
                          lw=3, label=r'NBI')
            ax[1, 0].plot(diag('PECR_TOT').time, diag('PECR_TOT').data/1e6, color='blue',
                          lw=3, label=r'ECRH')
            diag.close()
            diag = dd.shotfile('BPD', shot)
            ax[1, 0].plot(diag('Prad').time, diag('Prad').data/1e6, color='violet', lw=3,
                          label=r'Prad')
            ax[1, 0].legend(loc='best', numpoints=1, frameon=False)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel('[MW]')
            ax[1, 0].set_ylim([0, 8])

            diag = dd.shotfile('DCN', shot)
            ax[2, 0].plot(diag('H-1').time, diag('H-1').data /
                          1e19, color='blue', lw=3)
            ax[2, 0].set_ylabel(r'$\overline{n}_e$ [10$^{19}$]')
            ax[2, 0].set_ylim([0, 10])
            ax[2, 0].plot(diag('H-5').time, diag('H-5').data /
                          1e19, color='orange', lw=3)
            ax[2, 0].set_xlabel(r't [s]')
            ax[2, 0].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            diag.close
            ax[2, 0].set_xlim([0, 7.5])

            diag = dd.shotfile('TOT', shot)
            ax[0, 1].plot(diag('Wmhd').time, diag('Wmhd').data/1e5, color='k', lw=3,
                          label=r'# %5i' % shot)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            ax[0, 1].set_ylabel(r'W$_{mhd}$ [10$^5$ J]')
            diag.close()

            Gas = neutrals.Neutrals(shot)

            ax[1, 1].plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                          color='k', label=r'D$_2$', lw=3)
            ax[1, 1].plot(Gas.gas['N2']['t'], Gas.gas['N2']['data']/1e21,
                          color='magenta', lw=3, label=r'N')
            ax[1, 1].legend(loc='best', numpoints=1, frameon=False)
            ax[1, 1].axes.get_xaxis().set_visible(False)
            ax[1, 1].set_ylabel(r'[10$^{21}$]')

            ax[2, 1].plot(Gas.signal['F01']['t'],
                          Gas.signal['F01']['data']/1e22, color='k', lw=3,
                          label=r'F01 [10$^{22}$m$^{-2}$s$^{-1}$]')
            diag = dd.shotfile('MAC', shot)
            ax[2, 1].plot(diag('Tdiv').time, diag('Tdiv').data, color='red', lw=3,
                          label=r'T$_{div}$')
            ax[2, 1].legend(loc='best', numpoints=1, frameon=False)
            ax[2, 1].set_xlabel(r't [s]')
            ax[2, 1].set_ylim([0, 50])
            diag.close()
            mpl.pylab.savefig('../pdfbox/Shot' + str(int(shot)) +
                              '_GeneralHmod.pdf', bbox_to_inches='tight')

    elif selection == 6:
        # we compare the same general plot for reference and
        shotList = (33478, 34115)
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(17, 17),
                                     nrows=4, ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1)
        for shot, _col in zip(shotList, colorList):

            # current
            diag = dd.shotfile('MAG', shot)
            ax[0, 0].plot(diag('Ipa').time, diag('Ipa').data/1e6, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_ylim([0, 1.1])
            diag.close()
            # power
            diag = dd.shotfile('TOT', shot)
            ax[1, 0].plot(diag('PNBI_TOT').time, diag('PNBI_TOT').data/1e6, color=_col,
                          lw=3)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel('NBI [MW]')
            ax[1, 0].set_ylim([0, 5])
            ax[2, 0].plot(diag('PECR_TOT').time, diag('PECR_TOT').data/1e6, color=_col,
                          lw=3)
            diag.close()
            ax[2, 0].axes.get_xaxis().set_visible(False)
            ax[2, 0].set_ylabel('ECRH [MW]')
            ax[2, 0].set_ylim([0, 5])
            diag = dd.shotfile('BPD', shot)
            ax[3, 0].plot(diag('Prad').time, diag('Prad').data/1e6, color=_col, lw=3,
                          label=r'Prad')
            ax[3, 0].set_ylabel(r'Prad [MW]')
            ax[3, 0].set_xlim([0, 7])
            ax[3, 0].set_ylim([0, 5])
            ax[3, 0].set_xlabel(r't [s]')

            diag = dd.shotfile('DCN', shot)
            ax[0, 1].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            ax[0, 1].set_ylim([0, 10])
            ax[0, 1].plot(diag('H-5').time, diag('H-5').data /
                          1e19, color=_col, lw=3)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            diag.close

            diag = dd.shotfile('TOT', shot)
            ax[1, 1].plot(diag('Wmhd').time, diag('Wmhd').data/1e5, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[1, 1].axes.get_xaxis().set_visible(False)
            ax[1, 1].set_ylabel(r'W$_{mhd}$ [10$^5$ J]')
            diag.close()

            Gas = neutrals.Neutrals(shot)

            ax[2, 1].plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                          color=_col, label=r'D$_2$', lw=3)
            ax[2, 1].plot(Gas.gas['N2']['t'], Gas.gas['N2']['data']/1e21,
                          '--', color=_col, lw=3, label=r'N')
            ax[2, 1].legend(loc='best', numpoints=1, frameon=False)
            ax[2, 1].axes.get_xaxis().set_visible(False)
            ax[2, 1].set_ylabel(r'[10$^{21}$]')

            ax[3, 1].plot(Gas.signal['F01']['t'],
                          Gas.signal['F01']['data']/1e22, color=_col, lw=3,
                          label=r'F01 [10$^{22}$m$^{-2}$s$^{-1}$]')
            diag = dd.shotfile('MAC', shot)
            ax[3, 1].plot(diag('Tdiv').time, diag('Tdiv').data, '--', color=_col, lw=3,
                          label=r'T$_{div}$')

            ax[3, 1].set_xlabel(r't [s]')
            diag.close()

        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].set_ylim([0, 50])
        ax[2, 1].legend(loc='best', numpoints=1, frameon=False)
        mpl.pylab.savefig('../pdfbox/CompareShot33478_34115.pdf',
                          bbox_to_inches='tight')
    elif selection == 7:
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')
        tList = (1.18, 2.6, 3.43)
        # create the appropriate plot
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
            diag = dd.shotfile('DCN', shot)
            neAvg = diag('H-5')
            diag.close()
            ax1.plot(neAvg.time, neAvg.data/1e19, '-', color=cc, lw=2,
                     label=r'Shot # %5i' % shot + ' I$_p$ = %2.1f' % ip + ' MA')
            ax1.set_ylim([0, 6])
            ax1.set_xlim([0, 4.5])
            # load the Li-Beam profiles
            try:
                LiBD = dd.shotfile('LIN', shot, experiment='AUGD')
            except:
                LiBD = dd.shotfile('LIN', shot, experiment='LIBE')
            neLB = LiBD('ne').data
            neLBtime = LiBD('ne').time
            rhoP = LiBD('ne').area
            LiBD.close()
            rhoFake = np.linspace(0.98, 1.06, 50)
            profFake = np.zeros((neLBtime.size, 50))
            for n in range(neLBtime.size):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1]/1e19, s=0)
                profFake[n, :] = S(rhoFake)
            im = _ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto', cmap=mpl.cm.viridis,
                            extent=(neLBtime.min(),
                                    neLBtime.max(), 0.98, 1.06),
                            norm=LogNorm(vmin=0.5, vmax=1.5))
            _ax.set_xlim([1, 3.9])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' % shot)
            _ax.set_ylim([0.99, 1.05])
        ax2.set_ylabel(r'$\rho_p$')
        ax3.axes.get_yaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'$\overline{n_e}$ H-5 $[10^{19}$m$^{-3}]$')
        ax1.set_title(r'I$_p$ scan at constant q$_{95}$')
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax, format='%1.1f')
        cbar.set_ticks([0.5, 1, 1.5])
        cbar.set_label(r'n$_{e}$ [10$^{19}$m$^{-3}$]')

        mpl.pylab.savefig('../pdfbox/EvolutionEdgeProfileConstantq95.pdf',
                          bbox_to_inches='tight')

    elif selection == 8:
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')
        tList = (1.18, 2.6, 3.43)
        # create the appropriate plot
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
            diag = dd.shotfile('DCN', shot)
            neAvg = diag('H-5')
            diag.close()
            ax1.plot(neAvg.time, neAvg.data/1e19, '-', color=cc, lw=2,
                     label=r'Shot # %5i' % shot + ' I$_p$ = %2.1f' % ip + ' MA')
            ax1.set_ylim([0, 6])
            ax1.set_xlim([0, 4.5])
            # load the Li-Beam profiles
            try:
                LiBD = dd.shotfile('LIN', shot, experiment='AUGD')
            except:
                LiBD = dd.shotfile('LIN', shot, experiment='LIBE')
            neLB = LiBD('ne').data
            neLBtime = LiBD('ne').time
            rhoP = LiBD('ne').area
            LiBD.close()
            rhoFake = np.linspace(0.98, 1.06, 50)
            profFake = np.zeros((neLBtime.size, 50))
            for n in range(neLBtime.size):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1]/1e19, s=0)
                profFake[n, :] = S(rhoFake)
            im = _ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto', cmap=mpl.cm.viridis,
                            extent=(neLBtime.min(),
                                    neLBtime.max(), 0.98, 1.06),
                            norm=LogNorm(vmin=0.5, vmax=1.5))
            _ax.set_xlim([1, 3.9])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' % shot)
            _ax.set_ylim([0.99, 1.05])
        ax2.set_ylabel(r'$\rho_p$')
        ax3.axes.get_yaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'$\overline{n_e}$ H-5 $[10^{19}$m$^{-3}]$')
        ax1.set_title(r'I$_p$ scan at constant B$_t$')
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax, format='%1.1f')
        cbar.set_ticks([0.5, 1, 1.5, 2])
        cbar.set_label(r'n$_{e}$ [10$^{19}$m$^{-3}$]')

        mpl.pylab.savefig('../pdfbox/EvolutionEdgeProfileConstantBt.pdf',
                          bbox_to_inches='tight')

    elif selection == 9:
        # compute the PDF and the Autocorrelation time
        # for the last plunge in the q95 and constant Bt scan
        shotQ95L = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#C90015', '#7B0Ce7', '#F0CA37')
        fig, ax = mpl.pylab.subplots(figsize=(14, 14), nrows=2, ncols=2)
        for shot, ip, _col in zip(shotQ95L, currentL, colorL):

            Probe = augFilaments.Filaments(shot)
            if shot == 34103:
                trange = [3.48, 3.53]
            else:
                trange = [3.7, 3.8]
            dcn = dd.shotfile('DCN', shot)
            h5 = dcn('H-5')
            en = h5.data[np.where(
                ((h5.time >= trange[0]) &
                 (h5.time <= trange[1])))[0]].mean()/1e19
            dcn.close()
            Probe.blobAnalysis(
                Probe='Isat_m06', trange=trange, block=[0.015, 0.5])
            h, b = Probe.blob.pdf(bins='freedman', density=True, normed=True)
            ax[0, 0].plot((b[1:]+b[:-1])/2, h, color=_col, lw=3, label=r'I$_p$=%1.2f' % ip +
                          r' $\overline{n}_e$ = %2.1f' % en)

            ax[1, 0].plot(ip, Probe.blob.act*1e6, 's', color=_col, ms=15)

        ax[0, 0].set_yscale('log')
        ax[0, 0].set_xlabel(r'$\tilde{I}_s/\sigma$')
        ax[0, 0].set_title(r'Constant q$_{95}$')
        ax[1, 0].set_xlabel(r'I$_p$ [MA]')
        ax[1, 0].set_ylabel(r'$\tau_{ac} [\mu$s]')
        ax[1, 0].set_ylim([5, 120])
        ax[1, 0].set_xlim([0.5, 1.1])
        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        # repeat the same for constant Bt
        shotBtL = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#C90015', '#7B0Ce7', '#F0CA37')
        for shot, ip, _col in zip(shotBtL, currentL, colorL):
            Probe = augFilaments.Filaments(shot)
            if shot == 34105:
                trange = [3.1, 3.2]
            else:
                trange = [3.7, 3.8]
            dcn = dd.shotfile('DCN', shot)
            h5 = dcn('H-5')
            en = h5.data[np.where(
                ((h5.time >= trange[0]) &
                 (h5.time <= trange[1])))[0]].mean()/1e19
            dcn.close()
            Probe.blobAnalysis(
                Probe='Isat_m06', trange=trange, block=[0.015, 0.5])
            h, b = Probe.blob.pdf(bins='freedman', density=True, normed=True)
            ax[0, 1].plot((b[1:]+b[:-1])/2, h, color=_col, lw=3, label=r'I$_p$=%1.2f' % ip +
                          r' $\overline{n}_e$ = %2.1f' % en)

            ax[1, 1].plot(ip, Probe.blob.act*1e6, 's', color=_col, ms=15)

        ax[0, 1].set_yscale('log')
        ax[0, 1].set_xlabel(r'$\tilde{I}_s/\sigma$')
        ax[0, 1].set_title(r'Constant B$_t$')
        ax[1, 1].set_xlabel(r'I$_p$ [MA]')
        ax[1, 1].set_ylabel(r'$\tau_{ac} [\mu$s]')
        ax[0, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[1, 1].set_ylim([5, 120])
        ax[1, 1].set_xlim([0.5, 1.1])
        mpl.pylab.savefig('../pdfbox/ScalingAutoCorrelation.pdf',
                          bbox_to_inches='tight')

    elif selection == 10:
        shotList = (34108, 34115)
        tmaxL = (6.33, 6.26)
        fig, ax = mpl.pylab.subplots(figsize=(8, 10),
                                     nrows=2, ncols=1, sharex=True)
        for shot, i, tm in zip(shotList,
                               range(len(shotList)),
                               tmaxL):
            FPG = dd.shotfile('FPG', shot)
            Wmhd = FPG('Wmhd')
            FPG.close()
            dcn = dd.shotfile('DCN', shot)
            h5 = dcn('H-5')
            dcn.close
            # interpolate h5 on the same time basis of Wmh
            S = UnivariateSpline(h5.time, h5.data/1e19, s=0)
            _id = np.where(((Wmhd.time >= 3) & (Wmhd.time <= tm)))[0]
            ax[i].plot(S(Wmhd.time[_id]), Wmhd.data[_id]/1e5)
            ax[i].text(0.7, 0.9, 'Shot # %5i' % shot,
                       transform=ax[i].transAxes)
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel(r'W$_{mhd}[10^5$ J]')
        ax[1].set_ylabel(r'W$_{mhd}[10^5$ J]')
        ax[1].set_xlabel(r'$\overline{n_e}$ H-5 [10$^{19}$m$^{-3}$]')
        ax[1].set_xlim([2, 7])
        mpl.pylab.savefig('../pdfbox/DegradedHMode.pdf',
                          bbox_to_inches='tight')

        # Create a plot also for each of the figure
        for shot, i, tm in zip(shotList,
                               range(len(shotList)),
                               tmaxL):
            fig, ax = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
            fig.subplots_adjust(bottom=0.15)
            FPG = dd.shotfile('FPG', shot)
            Wmhd = FPG('Wmhd')
            FPG.close()
            dcn = dd.shotfile('DCN', shot)
            h5 = dcn('H-5')
            dcn.close
            # interpolate h5 on the same time basis of Wmh
            S = UnivariateSpline(h5.time, h5.data/1e19, s=0)
            _id = np.where(((Wmhd.time >= 3) & (Wmhd.time <= tm)))[0]
            ax.plot(S(Wmhd.time[_id]), Wmhd.data[_id]/1e5)
            ax.text(0.6, 0.9, 'Shot # %5i' % shot,
                    transform=ax.transAxes)
            ax.set_ylabel(r'W$_{mhd}[10^5$ J]')
            ax.set_ylabel(r'W$_{mhd}[10^5$ J]')
            ax.set_xlabel(r'$\overline{n_e}$ H-5 [10$^{19}$m$^{-3}$]')
            ax.set_xlim([2, 7])
            mpl.pylab.savefig('../pdfbox/DegradedHModeShot' +
                              str(int(shot))+'.pdf', bbox_to_inches='tight')

    elif selection == 11:
        shotL = (34108, 34115)
        diodsL = ['S2L3A00', 'S2L3A01', 'S2L3A02', 'S2L3A03',
                  'S2L3A12', 'S2L3A13', 'S2L3A14', 'S2L3A15',
                  'S2L3A08', 'S2L3A09', 'S2L3A10', 'S2L3A11',
                  'S2L3A04', 'S2L3A05', 'S2L3A06', 'S2L3A07']
        fig, ax = mpl.pylab.subplots(figsize=(8, 12),
                                     nrows=2, ncols=1)
        for shot, _is in zip(shotL, range(len(shotL))):
            Xvs = dd.shotfile('XVS', shot)
            t = Xvs(diodsL[0]).time
            nsamp = t.size
            diods = np.zeros((len(diodsL), nsamp))
            for d, i in zip(diodsL, range(len(diodsL))):
                _dummy = Xvs(d).data-Xvs(d).data[1000:50000].mean()
                diods[i, :] = pd.rolling_mean(_dummy, 5000)
                del _dummy
            # now create the appropriate image
            # limit to a smaller time intervals since
            # we only need the end
            _idx = np.where((t >= 5.5) & (t <= 6.3))[0]
            ax[_is].imshow(diods[:, _idx],
                           origin='lower', aspect='auto',
                           extent=(5.5, 6.3, 1, 17), cmap=mpl.cm.jet,
                           interpolation='bilinear')

            ax[_is].text(0.2, 0.9, r'Shot # %5i' % shot,
                         color='white',
                         transform=ax[_is].transAxes)
            del diods
        ax[0].axes.get_xaxis().set_visible(False)
        ax[1].set_xlabel(r't[s]')
        mpl.pylab.savefig('../pdfbox/RadiationDegradedHmode.pdf',
                          rasterize=True, bbox_to_inches='tight')

    elif selection == 12:
        shotL = (34107, 34108, 34115)
        colorL = ('#C90015', '#7B0Ce7', '#F0CA37')
        fig, ax = mpl.pylab.subplots(figsize=(10, 6), nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.15)
        for shot, _col in zip(shotL, colorL):
            # load and interpolate the H-5
            dcn = dd.shotfile('DCN', shot)
            H5 = dcn('H-5')
            dcn.close()
            # load ipolsola
            Mac = dd.shotfile('MAC', shot)
            iPol = Mac('Ipolsoli')
            # limit to the time from 3-6.3 s
            _idx = np.where((iPol.time >= 2.5) & (iPol.time <= 5.5))[0]
            S = savgol_filter(iPol.data[_idx], 501, 3)
            tS = iPol.time[_idx]
            # find the maxima
            peaksM, peaksMn = peakdetect.peakdetect(S, x_axis=tS)
            # ELMfreq
            tE, y = zip(*peaksM)
            y = np.asarray(y)
            tE = np.asarray(tE)
            tE = tE[(y > 2200)]
            ELMf = 1./np.diff(tE)
            ELMfTime = (tE[1:]+tE[:-1])/2.
            ax.plot(ELMfTime, ELMf, 'o', ms=6, color=_col,
                    alpha=0.5, mec=_col, label=r'Shot # %5i' % shot)
        ax.legend(loc='best', numpoints=1, frameon=True)
        ax.set_xlabel(r't [s]')
        ax.set_ylabel(r'ELM frequency [Hz]')
        mpl.pylab.savefig('../pdfbox/ELMfrequency.pdf',
                          bbox_to_inches='tight')

    elif selection == 13:
        shotL = (34107, 34108, 34115)
        for shot in shotL:
            Gas = neutrals.Neutrals(shot)
            fig, ax = mpl.pylab.subplots(
                figsize=(6, 12), nrows=4, ncols=1, sharex=True)
            fig.subplots_adjust(left=0.2)
            ax[0].plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e22, 'k-', lw=2.5,
                       label=r'D$_2$')
            ax[0].plot(Gas.gas['N2']['t'], Gas.gas['N2']['data']/1e22, '-',
                       color='magenta', lw=2.5, label=r'N$_2$')
            ax[0].legend(loc='best', numpoints=1, frameon=False)
            ax[0].set_xlim([0, 7])
            ax[0].axes.get_xaxis().set_visible(False)
            ax[0].set_ylabel('[10$^{21}$]')

            ax[1].plot(Gas.signal['F01']['t'],
                       Gas.signal['F01']['data']/1e21, lw=2.5)
            ax[1].set_xlabel(r't [s]')
            ax[1].set_ylabel(r'[10$^{21}$m$^{-2}$s$^{-1}$]')
            ax[1].text(0.1, 0.8, 'F01 Z=%3.2f' % Gas.signal['F01']['Z'],
                       transform=ax[1].transAxes)
            ax[1].axes.get_xaxis().set_visible(False)

            ax[2].plot(Gas.signal['F14']['t'],
                       Gas.signal['F14']['data']/1e21, lw=2.5)
            ax[2].set_xlabel(r't [s]')
            ax[2].set_ylabel(r'[10$^{21}$m$^{-2}$s$^{-1}$]')
            ax[2].text(0.1, 0.8, 'Z=%3.2f' % Gas.signal['F14']['Z'],
                       transform=ax[2].transAxes)
            ax[2].set_ylim([0, 20])
            ax[2].axes.get_xaxis().set_visible(False)

            ax[3].plot(Gas.signal['F14']['t'], Gas.signal['F01']
                       ['data']/Gas.signal['F14']['data'])
            ax[3].set_xlim([0, 7])
            ax[3].text(0.1, 0.8, 'F01/F14 Compression',
                       transform=ax[3].transAxes)
            ax[3].set_ylim([0, 500])
            ax[3].set_xlabel(r't[s]')
            mpl.pylab.savefig('../pdfbox/NeutralCompressionShot' +
                              str(int(shot))+'.pdf', bbox_to_inches='tight')

    elif selection == 14:
        shotL = (33478, 34115)
        colorL = ('black', 'red')
        fig, ax = mpl.pylab.subplots(
            figsize=(6, 12), nrows=4, ncols=1, sharex=True)
        fig.subplots_adjust(left=0.2)
        for shot, _col in zip(shotL, colorL):
            Gas = neutrals.Neutrals(shot)
            ax[0].plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e22, '-', lw=2.5,
                       label=r'#%5i' % shot, color=_col)
            ax[0].set_xlim([0, 7])
            ax[0].axes.get_xaxis().set_visible(False)
            ax[0].set_ylabel('[10$^{21}$]')

            ax[1].plot(Gas.signal['F01']['t'],
                       Gas.signal['F01']['data']/1e21, lw=2.5, color=_col)
            ax[1].set_xlabel(r't [s]')
            ax[1].set_ylabel(r'[10$^{21}$m$^{-2}$s$^{-1}$]')
            ax[1].axes.get_xaxis().set_visible(False)

            ax[2].plot(Gas.signal['F14']['t'],
                       Gas.signal['F14']['data']/1e21, lw=2.5, color=_col)
            ax[2].set_xlabel(r't [s]')
            ax[2].set_ylabel(r'[10$^{21}$m$^{-2}$s$^{-1}$]')
            ax[2].set_ylim([0, 20])
            ax[2].axes.get_xaxis().set_visible(False)

            ax[3].plot(Gas.signal['F14']['t'],
                       Gas.signal['F01']['data']/Gas.signal['F14']['data'],
                       color=_col)
            ax[3].set_xlim([0, 7])
            ax[3].set_ylim([0, 1200])
            ax[3].set_xlabel(r't[s]')

        ax[0].legend(loc='best', numpoints=1, frameon=False)
        ax[1].text(0.1, 0.8, 'F01 Z=%3.2f' % Gas.signal['F01']['Z'],
                   transform=ax[1].transAxes)
        ax[2].text(0.1, 0.8, 'F14 Z=%3.2f' % Gas.signal['F14']['Z'],
                   transform=ax[2].transAxes)
        ax[3].text(0.1, 0.8, 'F01/F14 Compression', transform=ax[3].transAxes)

        mpl.pylab.savefig('../pdfbox/CompareCompression33478_34115.pdf',
                          bbox_to_inches='tight')

    elif selection == 15:
        shotList = (34276, 34277)
        pufL = ('Low Div', 'Up Mid')
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(17, 17),
                                     nrows=4, ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1)
        for shot, _col, _str in zip(shotList, colorList, pufL):

            # current
            diag = dd.shotfile('MAG', shot)
            ax[0, 0].plot(diag('Ipa').time, diag('Ipa').data/1e6,
                          color=_col, lw=3,
                          label=r'# %5i' % shot + ' Puff from '+_str)
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_ylim([0, 1.1])
            diag.close()
            # power
            diag = dd.shotfile('TOT', shot)
            ax[1, 0].plot(diag('PNBI_TOT').time, diag('PNBI_TOT').data/1e6, color=_col,
                          lw=3)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel('NBI [MW]')
            ax[1, 0].set_ylim([0, 5])
            ax[2, 0].plot(diag('PECR_TOT').time, diag('PECR_TOT').data/1e6, color=_col,
                          lw=3)
            diag.close()
            ax[2, 0].axes.get_xaxis().set_visible(False)
            ax[2, 0].set_ylabel('ECRH [MW]')
            ax[2, 0].set_ylim([0, 5])
            diag = dd.shotfile('BPD', shot)
            ax[3, 0].plot(diag('Prad').time, diag('Prad').data/1e6, color=_col, lw=3,
                          label=r'Prad')
            ax[3, 0].set_ylabel(r'Prad [MW]')
            ax[3, 0].set_xlim([0, 7])
            ax[3, 0].set_ylim([0, 5])
            ax[3, 0].set_xlabel(r't [s]')

            diag = dd.shotfile('DCN', shot)
            ax[0, 1].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            ax[0, 1].set_ylim([0, 10])
            ax[0, 1].plot(diag('H-5').time, diag('H-5').data /
                          1e19, color=_col, lw=3)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            diag.close

            diag = dd.shotfile('TOT', shot)
            ax[1, 1].plot(diag('Wmhd').time, diag('Wmhd').data/1e5, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[1, 1].axes.get_xaxis().set_visible(False)
            ax[1, 1].set_ylabel(r'W$_{mhd}$ [10$^5$ J]')
            diag.close()

            try:
                Gas = neutrals.Neutrals(shot)
            except:
                pass
            Uvs = dd.shotfile('UVS', shot)

            ax[2, 1].plot(Uvs('D_tot').time,
                          Uvs('D_tot').data/1e21,
                          color=_col, label=r'D$_2$', lw=3)
            ax[2, 1].plot(Uvs('N_tot').time,
                          Uvs('N_tot').data/1e21,
                          '--', color=_col, lw=3, label=r'N')
            ax[2, 1].legend(loc='best', numpoints=1, frameon=False)
            ax[2, 1].axes.get_xaxis().set_visible(False)
            ax[2, 1].set_ylabel(r'[10$^{21}$]')
            Uvs.close()
            try:
                ax[3, 1].plot(Gas.signal['F01']['t'],
                              Gas.signal['F01']['data']/1e22, color=_col, lw=3,
                              label=r'F01 [10$^{22}$m$^{-2}$s$^{-1}$]')
            except:
                pass
            diag = dd.shotfile('MAC', shot)
            ax[3, 1].plot(diag('Tdiv').time, diag('Tdiv').data, '--', color=_col, lw=3,
                          label=r'T$_{div}$')

            ax[3, 1].set_xlabel(r't [s]')
            diag.close()

        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].set_ylim([-10, 50])
        ax[2, 1].legend(loc='best', numpoints=1, frameon=False)
        mpl.pylab.savefig('../pdfbox/CompareShot'+str(int(shotList[0]))+'_' +
                          str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')
    elif selection == 16:
        shotList = (34276, 34277)
        pufL = ('Low Div', 'Up Div')
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(
            figsize=(8, 10), nrows=2, ncols=1, sharex=True)
        fig.subplots_adjust(bottom=0.15)
        for shot, col, p in zip(shotList, colorList, pufL):
            Gas = neutrals.Neutrals(shot)
            ax[0].plot(Gas.signal['F01']['t'],
                       Gas.signal['F01']['data']/Gas.signal['F14']['data'],
                       color=col, lw=2, label=r'Shot # %5i' % shot + 'Puff from '+p)
            Msp = dd.shotfile('MSP', shot)
            Press = 100.*Msp('B25_08Fu').data / \
                (0.0337*np.log10(Msp('B25_08Fu').data*100.)+0.7304)
            ax[1].plot(Msp('B25_08Fu').time, Press, color=col, lw=2)
            Msp.close()
        ax[1].set_xlabel('t [s]')
        ax[0].set_ylabel(r'Compression F01/F14')
        ax[0].legend(loc='best', fontsize=14, frameon=False, numpoints=1)
        ax[0].set_xlim([0, 7])
        ax[0].axes.get_xaxis().set_visible(False)
        ax[1].set_ylabel(r'P$_{div}$ [Pa]')
        ax[1].text(0.1, 0.9, r'100*B25_08Fu/(0.0337*log(B25_08Fu*100) + 0.7304)',
                   fontsize=12, transform=ax[1].transAxes)
        mpl.pylab.savefig('../pdfbox/CompareCompression'+str(int(shotList[0]))+'_' +
                          str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')

    elif selection == 17:
        shotList = (34276, 34278)
        pufL = ('Off', 'On')
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(17, 17),
                                     nrows=5, ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1)
        for shot, _col, _str in zip(shotList, colorList, pufL):

            # current
            diag = dd.shotfile('MAG', shot)
            ax[0, 0].plot(diag('Ipa').time, diag('Ipa').data/1e6, color=_col, lw=3,
                          label=r'# %5i' % shot + ' Cryo '+_str)
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_ylim([0, 1.1])
            diag.close()
            # power
            diag = dd.shotfile('TOT', shot)
            ax[1, 0].plot(diag('PNBI_TOT').time, diag('PNBI_TOT').data/1e6, color=_col,
                          lw=3)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel('NBI [MW]')
            ax[1, 0].set_ylim([0, 5])
            ax[2, 0].plot(diag('PECR_TOT').time, diag('PECR_TOT').data/1e6, color=_col,
                          lw=3)
            diag.close()
            ax[2, 0].axes.get_xaxis().set_visible(False)
            ax[2, 0].set_ylabel('ECRH [MW]')
            ax[2, 0].set_ylim([0, 5])

            # WMHD
            diag = dd.shotfile('TOT', shot)
            ax[3, 0].plot(diag('Wmhd').time, diag('Wmhd').data/1e5, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[3, 0].axes.get_xaxis().set_visible(False)
            ax[3, 0].set_ylabel(r'W$_{mhd}$ [10$^5$ J]')
            diag.close()
            # Radiation
            diag = dd.shotfile('BPD', shot)
            ax[4, 0].plot(diag('Prad').time, diag('Prad').data/1e6, color=_col, lw=3,
                          label=r'Prad')
            ax[4, 0].set_ylabel(r'Prad [MW]')
            ax[4, 0].set_xlim([0, 7])
            ax[4, 0].set_ylim([0, 5])
            ax[4, 0].set_xlabel(r't [s]')

            # second column
            diag = dd.shotfile('DCN', shot)
            ax[0, 1].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            ax[0, 1].set_ylim([0, 10])
            ax[0, 1].plot(diag('H-5').time, diag('H-5').data /
                          1e19, color=_col, lw=3)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            diag.close

            try:
                Gas = neutrals.Neutrals(shot)
            except:
                pass
            Uvs = dd.shotfile('UVS', shot)

            ax[1, 1].plot(Uvs('D_tot').time,
                          Uvs('D_tot').data/1e21,
                          color=_col, label=r'D$_2$', lw=3)
            ax[1, 1].set_ylabel(r'D$_2$ [10$^{21}$]')
            ax[1, 1].axes.get_xaxis().set_visible(False)
            ax[2, 1].plot(Uvs('N_tot').time,
                          Uvs('N_tot').data/1e21,
                          color=_col, lw=3, label=r'N')
            ax[2, 1].axes.get_xaxis().set_visible(False)
            ax[2, 1].set_ylabel(r'N$_2$ [10$^{21}$]')
            Uvs.close()
            Msp = dd.shotfile('MSP', shot)
            Press = 100.*Msp('B25_08Fu').data/(
                0.0337*np.log10(Msp('B25_08Fu').data*100.)+0.7304)
            ax[3, 1].plot(Msp('B25_08Fu').time, Press, ls='--',
                          color=_col, lw=2, label='B25_08Fu')
            Msp.close()
            try:
                ax[3, 1].plot(Gas.signal['F01']['t'],
                              Gas.signal['F01']['data']/1e22, ls='-', color=_col, lw=3,
                              label=r'F01 [10$^{22}$m$^{-2}$s$^{-1}$]')
            except:
                pass
            diag = dd.shotfile('MAC', shot)
            ax[4, 1].plot(diag('Tdiv').time, diag('Tdiv').data, '-', color=_col, lw=3,
                          label=r'T$_{div}$')

            ax[4, 1].set_xlabel(r't [s]')
            diag.close()

        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[4, 1].set_ylim([-10, 50])
        ax[4, 1].set_ylabel(r'T$_{div}$')
        mpl.pylab.savefig('../pdfbox/CompareShot'+str(int(shotList[0]))+'_' +
                          str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')

    elif selection == 18:
        shotList = (34276, 34281)
        pufL = ('Off', 'On')
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(17, 17),
                                     nrows=5, ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1)
        for shot, _col, _str in zip(shotList, colorList, pufL):

            # current
            diag = dd.shotfile('MAG', shot)
            ax[0, 0].plot(diag('Ipa').time, diag('Ipa').data/1e6, color=_col, lw=3,
                          label=r'# %5i' % shot + ' Cryo '+_str)
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_ylim([0, 1.1])
            diag.close()
            # power
            diag = dd.shotfile('TOT', shot)
            ax[1, 0].plot(diag('PNBI_TOT').time, diag('PNBI_TOT').data/1e6, color=_col,
                          lw=3)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel('NBI [MW]')
            ax[1, 0].set_ylim([0, 5])
            ax[2, 0].plot(diag('PECR_TOT').time, diag('PECR_TOT').data/1e6, color=_col,
                          lw=3)
            diag.close()
            ax[2, 0].axes.get_xaxis().set_visible(False)
            ax[2, 0].set_ylabel('ECRH [MW]')
            ax[2, 0].set_ylim([0, 5])

            # WMHD
            diag = dd.shotfile('TOT', shot)
            ax[3, 0].plot(diag('Wmhd').time, diag('Wmhd').data/1e5, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[3, 0].axes.get_xaxis().set_visible(False)
            ax[3, 0].set_ylabel(r'W$_{mhd}$ [10$^5$ J]')
            diag.close()
            # Radiation
            diag = dd.shotfile('BPD', shot)
            ax[4, 0].plot(diag('Prad').time, diag('Prad').data/1e6, color=_col, lw=3,
                          label=r'Prad')
            ax[4, 0].set_ylabel(r'Prad [MW]')
            ax[4, 0].set_xlim([0, 7])
            ax[4, 0].set_ylim([0, 5])
            ax[4, 0].set_xlabel(r't [s]')

            # second column
            diag = dd.shotfile('DCN', shot)
            ax[0, 1].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            ax[0, 1].set_ylim([0, 10])
            ax[0, 1].plot(diag('H-5').time, diag('H-5').data /
                          1e19, color=_col, lw=3)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            diag.close

            try:
                Gas = neutrals.Neutrals(shot)
            except:
                pass
            Uvs = dd.shotfile('UVS', shot)

            ax[1, 1].plot(Uvs('D_tot').time,
                          Uvs('D_tot').data/1e21,
                          color=_col, label=r'D$_2$', lw=3)
            ax[1, 1].set_ylabel(r'D$_2$ [10$^{21}$]')
            ax[1, 1].axes.get_xaxis().set_visible(False)
            ax[2, 1].plot(Uvs('N_tot').time,
                          Uvs('N_tot').data/1e21,
                          color=_col, lw=3, label=r'N')
            ax[2, 1].axes.get_xaxis().set_visible(False)
            ax[2, 1].set_ylabel(r'N$_2$ [10$^{21}$]')
            Uvs.close()
            Msp = dd.shotfile('MSP', shot)
            Press = 100.*Msp('B25_08Fu').data / \
                (0.0337*np.log10(Msp('B25_08Fu').data*100.)+0.7304)
            ax[3, 1].plot(Msp('B25_08Fu').time, Press, ls='--',
                          color=_col, lw=2, label='B25_08Fu')
            Msp.close()
            try:
                ax[3, 1].plot(Gas.signal['F01']['t'],
                              Gas.signal['F01']['data']/1e22, ls='-', color=_col, lw=3,
                              label=r'F01 [10$^{22}$m$^{-2}$s$^{-1}$]')
            except:
                pass
            diag = dd.shotfile('MAC', shot)
            ax[4, 1].plot(diag('Tdiv').time, diag('Tdiv').data, '-', color=_col, lw=3,
                          label=r'T$_{div}$')

            ax[4, 1].set_xlabel(r't [s]')
            diag.close()

        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[4, 1].set_ylim([-10, 50])
        ax[4, 1].set_ylabel(r'T$_{div}$')
        mpl.pylab.savefig('../pdfbox/CompareShot'+str(int(shotList[0]))+'_' +
                          str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')

    elif selection == 19:
        shotList = (34276, 34280)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7')
        tList = (1.8, 2.4, 4.0)
        fig = mpl.pylab.figure(figsize=(12, 12))
        fig.subplots_adjust(hspace=0.25, right=0.86)
        ax1 = mpl.pylab.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = mpl.pylab.subplot2grid((2, 2), (1, 0))
        ax3 = mpl.pylab.subplot2grid((2, 2), (1, 1))
        axL = (ax2, ax3)
        Cryo = ('No', 'On')
        for shot, cc, _ax, _cr in itertools.izip(
                shotList, colorLS, axL, Cryo):
            # this is the line average density
            diag = dd.shotfile('TOT', shot)
            neAvg = diag('H-1/(2a)')
            diag.close()
            ax1.plot(neAvg.time, neAvg.data/1e20, '-', color=cc, lw=2,
                     label=r'Shot # %5i' % shot + ' Cryo ' + _cr)
            ax1.set_ylim([0, 1])
            ax1.set_xlim([0, 7])
            # load the Li-Beam profiles
            LiBD = dd.shotfile('LIN', shot, experiment='AUGD')
            neLB = LiBD('ne').data
            neLBtime = LiBD('ne').time
            rhoP = LiBD('ne').area
            LiBD.close()
            rhoFake = np.linspace(0.98, 1.06, 50)
            profFake = np.zeros((neLBtime.size, 50))
            for n in range(neLBtime.size):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1]/1e19, s=0)
                profFake[n, :] = S(rhoFake)
            im = _ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto', cmap=mpl.cm.viridis,
                            extent=(neLBtime.min(),
                                    neLBtime.max(), 0.98, 1.06),
                            norm=LogNorm(vmin=0.5, vmax=2.5))

            _ax.set_title(r'Shot # % 5i' % shot)

            _ax.set_xlim([1, 6.5])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' % shot)
            _ax.set_ylim([0.99, 1.05])
        ax2.set_ylabel(r'$\rho_p$')
        ax3.axes.get_yaxis().set_visible(False)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'$\overline{n_e}$ H-5 $[10^{19}$m$^{-3}]$')
        ax1.set_title(r'I$_p$ scan at constant q$_{95}$')
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax, format='%1.1f')
        cbar.set_ticks([0.5, 1, 1.5, 2.5])
        cbar.set_label(r'n$_{e}$ [10$^{19}$m$^{-3}$]')

        mpl.pylab.savefig('../pdfbox/EvolutionEdgeProfiles_' + str(int(shotList[0])) +
                          '_'+str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')

    elif selection == 20:
        shotList = (34276, 34277)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7')
        tList = (1.8, 2.4, 4.0)
        fig = mpl.pylab.figure(figsize=(12, 12))
        fig.subplots_adjust(hspace=0.25, right=0.86)
        ax1 = mpl.pylab.subplot2grid((3, 2), (0, 0), colspan=2)
        ax2 = mpl.pylab.subplot2grid((3, 2), (1, 0))
        ax3 = mpl.pylab.subplot2grid((3, 2), (1, 1))
        axL = (ax2, ax3)
        ax4 = mpl.pylab.subplot2grid((3, 2), (2, 0))
        ax5 = mpl.pylab.subplot2grid((3, 2), (2, 1))
        axD = (ax4, ax5)
        Cryo = ('Lower', 'Upper')
        for shot, cc, _ax, _ax2, _cr in itertools.izip(
                shotList, colorLS, axL, axD, Cryo):
            # this is the line average density
            diag = dd.shotfile('DCN', shot)
            neAvg = diag('H-5')
            diag.close()
            ax1.plot(neAvg.time, neAvg.data/1e20, '-', color=cc, lw=2,
                     label=r'Shot # %5i' % shot + ' Puff from ' + _cr + ' Divertor')
            ax1.set_ylim([0, 1])
            ax1.set_xlim([0, 7])
            LiB = libes.Libes(shot)

            im = _ax.imshow(np.log(LiB.ne.transpose()/1e19),
                            origin='lower', aspect='auto', cmap=mpl.cm.hot,
                            extent=(LiB.time.min(), LiB.time.max(),
                                    LiB.rho.min(), LiB.rho.max()),
                            norm=LogNorm(vmin=0.5, vmax=2.5))
            _ax.set_title(r'Shot # % 5i' % shot)

            _ax.set_xlim([1, 6.5])
            _ax.set_title(r'Shot # % 5i' % shot)
            _ax.set_ylim([0.99, 1.05])
            # add the computation from the RIC profile
            if shot != 34277:
                shotfile = dd.shotfile('RIC', shot)
                En = shotfile('Ne_Ant4')
                RicFake = np.zeros((En.time.size, 50))
                rhoFake = np.linspace(LiB.rho.min(), LiB.rho.max(), 50)
                for n in range(En.time.size):
                    S = UnivariateSpline(En.area[n, :], En.data[n, :], s=0)
                    RicFake[n, :] = S(rhoFake)

                im = _ax2.imshow(np.log(RicFake.transpose()),
                                 origin='lower', aspect='auto', cmap=mpl.cm.viridis,
                                 extent=(En.time.min(),
                                         En.time.max(), 0.98, 1.06),
                                 norm=LogNorm(vmin=0.5, vmax=2.5))
                _ax2.set_xlim([1, 6.5])
                _ax2.set_xlabel(r't [s]')
                _ax2.set_ylim([0.99, 1.05])

        ax2.set_ylabel(r'$\rho_p$')
        ax3.axes.get_yaxis().set_visible(False)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'$\overline{n_e}$ H-5 $[10^{19}$m$^{-2}]$')
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.4])
        cbar = fig.colorbar(im, cax=cbar_ax, format='%1.1f')
        cbar.set_ticks([0.5, 1, 1.5, 2.5])
        cbar.set_label(r'n$_{e}$ [10$^{19}$m$^{-3}$]')
        ax4.set_ylabel(r'$\rho_p$')
        ax5.axes.get_yaxis().set_visible(False)

        mpl.pylab.savefig('../pdfbox/EvolutionEdgeProfiles_' + str(int(shotList[0])) +
                          '_'+str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')

    elif selection == 21:
        shotList = (34276, 34278)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7')
        tList = (1.8, 2.4, 4.0)
        fig = mpl.pylab.figure(figsize=(12, 12))
        fig.subplots_adjust(hspace=0.25, right=0.86)
        ax1 = mpl.pylab.subplot2grid((3, 2), (0, 0), colspan=2)
        ax2 = mpl.pylab.subplot2grid((3, 2), (1, 0))
        ax3 = mpl.pylab.subplot2grid((3, 2), (1, 1))
        axL = (ax2, ax3)
        ax4 = mpl.pylab.subplot2grid((3, 2), (2, 0))
        ax5 = mpl.pylab.subplot2grid((3, 2), (2, 1))
        axD = (ax4, ax5)
        Cryo = ('Off', 'On')
        for shot, cc, _ax, _ax2,  _cr in itertools.izip(
                shotList, colorLS, axL, axD, Cryo):
            # this is the line average density
            diag = dd.shotfile('TOT', shot)
            neAvg = diag('H-1/(2a)')
            diag.close()
            ax1.plot(neAvg.time, neAvg.data/1e20, '-', color=cc, lw=2,
                     label=r'Shot # %5i' % shot + ' Cryo ' + _cr)
            ax1.set_ylim([0, 1])
            ax1.set_xlim([0, 7])
            # load the Li-Beam profiles
            LiBD = dd.shotfile('LIN', shot, experiment='AUGD')
            neLB = LiBD('ne').data
            neLBtime = LiBD('ne').time
            rhoP = LiBD('ne').area
            LiBD.close()
            rhoFake = np.linspace(0.98, 1.06, 50)
            profFake = np.zeros((neLBtime.size, 50))
            for n in range(neLBtime.size):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1]/1e19, s=0)
                profFake[n, :] = S(rhoFake)
            im = _ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto',
                            cmap=mpl.cm.viridis,
                            extent=(neLBtime.min(),
                                    neLBtime.max(), 0.98, 1.06),
                            norm=LogNorm(vmin=0.5, vmax=2.5))

            _ax.set_title(r'Shot # % 5i' % shot)

            _ax.set_xlim([1, 6.5])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' % shot)
            _ax.set_ylim([0.99, 1.05])
            shotfile = dd.shotfile('RIC', shot)
            En = shotfile('Ne_Ant4')
            RicFake = np.zeros((En.time.size, 50))
            for n in range(En.time.size):
                S = UnivariateSpline(En.area[n, :], En.data[n, :], s=0)
                RicFake[n, :] = S(rhoFake)

            im = _ax2.imshow(np.log(profFake.transpose()),
                             origin='lower', aspect='auto', cmap=mpl.cm.viridis,
                             extent=(En.time.min(), En.time.max(), 0.98, 1.06),
                             norm=LogNorm(vmin=0.5, vmax=2.5))
            _ax2.set_xlim([1, 6.5])
            _ax2.set_xlabel(r't [s]')
            _ax2.set_ylim([0.99, 1.05])

        ax2.set_ylabel(r'$\rho_p$')
        ax3.axes.get_yaxis().set_visible(False)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'$\overline{n_e}$ H-5 $[10^{19}$m$^{-3}]$')
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.4])
        cbar = fig.colorbar(im, cax=cbar_ax, format='%1.1f')
        cbar.set_ticks([0.5, 1, 1.5, 2.5])
        cbar.set_label(r'n$_{e}$ [10$^{19}$m$^{-3}$]')
        ax4.set_ylabel(r'$\rho_p$')
        ax5.axes.get_yaxis().set_visible(False)

        mpl.pylab.savefig('../pdfbox/EvolutionEdgeProfiles_' + str(int(shotList[0])) +
                          '_'+str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')

    elif selection == 22:
        shotL = (34103, 34102, 34104)
        iPL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        fig = mpl.pylab.figure(figsize=(14, 18))
        fig.subplots_adjust(hspace=0.3, top=0.96, right=0.98)
        ax1 = mpl.pylab.subplot2grid((4, 3), (0, 0), colspan=3)
        ax12 = mpl.pylab.subplot2grid((4, 3), (1, 0), colspan=3)
        ax13 = mpl.pylab.subplot2grid((4, 3), (2, 0), colspan=3)
        ax2 = mpl.pylab.subplot2grid((4, 3), (3, 0))
        ax3 = mpl.pylab.subplot2grid((4, 3), (3, 1))
        ax4 = mpl.pylab.subplot2grid((4, 3), (3, 2))
        axL = (ax2, ax3, ax4)
        # these are approximately the time at the same edge density
        tList = (3.43, 3.187, 2.75)
        for shot, tMax, iP, col, _ax in itertools.izip(
                shotL, tList, iPL, colorL, axL):
            diag = dd.shotfile('MAG', shot)
            ax1.plot(diag('Ipa').time, diag('Ipa').data/1e6, color=col, lw=3,
                     label=r'# %5i' % shot)
            ax1.axes.get_xaxis().set_visible(False)
            ax1.set_xlim([0, 4.2])
            ax1.set_ylim([0, 1.1])
            diag.close()

            diag = dd.shotfile('TOT', shot)
            neAvg = diag('H-1/(2a)')
            ax12.plot(neAvg.time, neAvg.data/1e20, '-', color=col, lw=3)
            ax12.set_ylim([0, 1])
            ax12.set_xlim([0, 4.2])

            ax12.axes.get_xaxis().set_visible(False)

            Gas = neutrals.Neutrals(shot)
            ax13.plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                      color=col, lw=3)
            ax13.set_ylabel(r'D$_2$  [10$^{21}$]')
            ax13.set_xlim([0, 4.2])
            ax13.set_xlabel(r't [s]')

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
            # plot the first of the normalized profile
            # with label according to the average density
            _idx = np.where((neLBtime >= 1.78) & (neLBtime <= 1.82))[0]
            y = np.zeros((_idx.size, 200))
            for n, _iDummy in zip(_idx, range(_idx.size)):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1], s=0)
                y[_iDummy, :] = S(rhoFake)/S(1)
            _idx = np.where((neAvg.time >= 1.78) & (neAvg.time <= 1.82))[0]
            _ne = neAvg.data[_idx].mean()/1e20
            _ax.plot(rhoFake, np.mean(y, axis=0), '-', color='k',
                     lw=3, label=r'n$_e$ = %3.2f' % _ne)
            _ax.fill_between(rhoFake, np.mean(y, axis=0) -
                             np.std(y, axis=0), np.mean(y, axis=0) +
                             np.std(y, axis=0),
                             facecolor='grey', edgecolor='none',
                             alpha=0.5)

            # repeat and now it will be tmax
            _idx = np.where((neLBtime >= tMax-0.02) &
                            (neLBtime <= tMax+0.02))[0]
            y = np.zeros((_idx.size, 200))
            for n, _iDummy in zip(_idx, range(_idx.size)):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1], s=0)
                y[_iDummy, :] = S(rhoFake)/S(1)
            _idx = np.where((neAvg.time >= tMax-0.02) &
                            (neAvg.time <= tMax+0.02))[0]
            _ne = neAvg.data[_idx].mean()/1e20
            _ax.plot(rhoFake, np.mean(y, axis=0), '-', color='orange',
                     lw=3, label=r'n$_e$ = %3.2f' % _ne)
            _ax.fill_between(rhoFake, np.mean(y, axis=0) -
                             np.std(y, axis=0), np.mean(y, axis=0) +
                             np.std(y, axis=0),
                             facecolor='orange', edgecolor='none',
                             alpha=0.5)
            _ax.set_yscale('log')
            _ax.set_xlabel(r'$\rho_p$')
            _ax.set_title(r'Shot # % 5i' % shot)
            _ax.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'I$_p$ [MA]')
        ax12.set_ylabel(r'$\overline{n}_e$ H-5 [10$^{20}$]')
        ax13.set_ylabel(r'D$_2 [10^{21}$s$^{-1}]$')
        ax1.set_title(r'I$_p$ scan at constant q$_{95}$')
        ax2.set_ylabel(r'n$_e$/n$_e(\rho_p = 1)$')
        mpl.pylab.savefig('../pdfbox/IpConstantq95_samedensity.pdf',
                          bbox_to_inches='tight')
        mpl.pylab.savefig('../pngbox/IpConstantq95_samedensity.png',
                          bbox_to_inches='tight', dpi=300)

    elif selection == 23:
        shotList = (34276, 34281)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7')
        tList = (1.8, 2.4, 4.0)
        fig = mpl.pylab.figure(figsize=(12, 16))
        fig.subplots_adjust(hspace=0.25, right=0.86, top=0.98)
        ax1 = mpl.pylab.subplot2grid((4, 2), (0, 0), colspan=2)
        ax12 = mpl.pylab.subplot2grid((4, 2), (1, 0), colspan=2)
        ax2 = mpl.pylab.subplot2grid((4, 2), (2, 0))
        ax3 = mpl.pylab.subplot2grid((4, 2), (2, 1))
        axL = (ax2, ax3)
        ax4 = mpl.pylab.subplot2grid((4, 2), (2, 0))
        ax5 = mpl.pylab.subplot2grid((4, 2), (2, 1))
        axD = (ax4, ax5)
        Cryo = ('No', 'On')
        for shot, cc, _ax, _ax2, _cr in itertools.izip(
                shotList, colorLS, axL, axD, Cryo):

            Gas = neutrals.Neutrals(shot)
            ax1.plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                     color=cc, lw=3,
                     label=r'Shot # %5i' % shot + ' Cryo ' + _cr)
            ax1.axes.get_xaxis().set_visible(False)
            ax1.set_ylabel(r'D$_2$  [10$^{21}$]')
            ax1.set_xlim([0, 7])
            # this is the line average density
            diag = dd.shotfile('DCN', shot)
            neAvg = diag('H-5')
            diag.close()
            ax12.plot(neAvg.time, neAvg.data/1e19, '-', color=cc, lw=2,
                      label=r'Shot # %5i' % shot + ' Cryo ' + _cr)
            ax12.set_ylim([0, 10])
            ax12.set_xlim([0, 7])
            # load the Li-Beam profiles
            LiBD = dd.shotfile('LIN', shot, experiment='AUGD')
            neLB = LiBD('ne').data
            neLBtime = LiBD('ne').time
            rhoP = LiBD('ne').area
            LiBD.close()
            rhoFake = np.linspace(0.98, 1.06, 50)
            profFake = np.zeros((neLBtime.size, 50))
            for n in range(neLBtime.size):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1]/1e19, s=0)
                profFake[n, :] = S(rhoFake)
            im = _ax.imshow(np.log(profFake.transpose()),
                            origin='lower', aspect='auto', cmap=mpl.cm.viridis,
                            extent=(neLBtime.min(),
                                    neLBtime.max(), 0.98, 1.06),
                            norm=LogNorm(vmin=0.5, vmax=2.5))

            _ax.set_title(r'Shot # % 5i' % shot)

            _ax.set_xlim([1, 6.5])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' % shot)
            _ax.set_ylim([0.99, 1.05])
            try:
                shotfile = dd.shotfile('RIC', shot)
                En = shotfile('Ne_Ant4')
                RicFake = np.zeros((En.time.size, 50))
                for n in range(En.time.size):
                    S = UnivariateSpline(En.area[n, :], En.data[n, :], s=0)
                    RicFake[n, :] = S(rhoFake)

                    im = _ax2.imshow(np.log(profFake.transpose()),
                                     origin='lower', aspect='auto', cmap=mpl.cm.viridis,
                                     extent=(En.time.min(),
                                             En.time.max(), 0.98, 1.06),
                                     norm=LogNorm(vmin=0.5, vmax=2.5))
                    _ax2.set_xlim([1, 6.5])
                    _ax2.set_xlabel(r't [s]')
                    _ax2.set_ylim([0.99, 1.05])
            except:
                pass

        ax2.set_ylabel(r'$\rho_p$')
        ax3.axes.get_yaxis().set_visible(False)
        ax12.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax12.set_ylabel(r'n$_e$ H-5 $[10^{19}$m$^{-3}]$')
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.3])
        cbar = fig.colorbar(im, cax=cbar_ax, format='%1.1f')
        cbar.set_ticks([0.5, 1, 1.5, 2.5])
        cbar.set_label(r'n$_{e}$ [10$^{19}$m$^{-3}$]')
        ax4.set_ylabel(r'$\rho_p$')
        ax5.axes.get_yaxis().set_visible(False)
        mpl.pylab.savefig('../pdfbox/EvolutionEdgeProfiles_' + str(int(shotList[0])) +
                          '_'+str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')
        mpl.pylab.savefig('../pngbox/EvolutionEdgeProfiles_' + str(int(shotList[0])) +
                          '_'+str(int(shotList[1]))+'.png',
                          bbox_to_inches='tight', dpi=300)

    elif selection == 24:
        shotList = (34103, 34102, 34104)
        colorLS = ('#C90015', '#7B0Ce7', '#437356')
        tList = ((2.57, 2.9),
                 (2.57, 3.03, 3.5),
                 (1.83, 2.44, 3.15))
        fig = mpl.pylab.figure(figsize=(16, 17))
        fig.subplots_adjust(hspace=0.25)

        ax1 = mpl.pylab.subplot2grid((4, 3), (0, 0), colspan=3)
        # list of subplots for the profiles of Li-Bea
        axEnL = (mpl.pylab.subplot2grid((4, 3), (1, 0)),
                 mpl.pylab.subplot2grid((4, 3), (1, 1)),
                 mpl.pylab.subplot2grid((4, 3), (1, 2)))

        # list of subplots for the Divertor density profiles
        axDivL = (mpl.pylab.subplot2grid((4, 3), (2, 0)),
                  mpl.pylab.subplot2grid((4, 3), (2, 1)),
                  mpl.pylab.subplot2grid((4, 3), (2, 2)))

        # list of subplots for Lambda profiles
        axLamL = (mpl.pylab.subplot2grid((4, 3), (3, 0)),
                  mpl.pylab.subplot2grid((4, 3), (3, 1)),
                  mpl.pylab.subplot2grid((4, 3), (3, 2)))

        for shot, col, tL in itertools.izip(
                shotList, colorLS, tList):
            # get the current and compute the average current in Ip
            Mag = dd.shotfile('MAG', shot)
            _ii = np.where((Mag('Ipa').time >= 3) &
                           (Mag('Ipa').time <= 3.3))[0]
            ip = Mag('Ipa').data[_ii]/1e6
            Mag.close()
            # this is the line average density
            diag = dd.shotfile('DCN', shot)
            neEdge = diag('H-5')
            diag.close()
            ax1.plot(neEdge.time, neEdge.data/1e20, '-', color=col, lw=2,
                     label=r'Shot # %5i' % shot +
                     ' I$_p$ = %2.1f' % ip.mean() + ' MA')
            ax1.set_ylim([0, 1])
            ax1.set_xlim([0, 4.5])
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
            for t, ax in zip(tL, axEnL):
                _idx = np.where((neLBtime >= t-0.02) & (neLBtime <= t+0.02))[0]
                y = np.zeros((_idx.size, 200))
                for n, _iDummy in zip(_idx, range(_idx.size)):
                    S = UnivariateSpline(rhoP[n, ::-1],
                                         neLB[n, ::-1], s=0)
                    y[_iDummy, :] = S(rhoFake)/S(1)
                _idx = np.where((neEdge.time >= t-0.02) &
                                (neEdge.time <= t+0.02))[0]
                enLabel = neEdge.data[_idx].mean()/1e20
                ax.plot(rhoFake, np.mean(y, axis=0), '-', color=col,
                        lw=3, label=r'$\overline{n_e}$ = %3.2f' % enLabel +
                        ' I$_p$ = %2.1f' % ip.mean() + ' MA')
                ax.fill_between(rhoFake, np.mean(y, axis=0) -
                                np.std(y, axis=0), np.mean(y, axis=0) +
                                np.std(y, axis=0),
                                facecolor=col, edgecolor='none',
                                alpha=0.5)
                ax.set_yscale('log')

            # Now we need to compute the profiles at the target
            Target = langmuir.Target(shot)
            for t, axD, axL in zip(tL, axDivL, axLamL):
                _idx = np.where((neEdge.time >= t-0.02) &
                                (neEdge.time <= t+0.02))[0]
                enLabel = neEdge.data[_idx].mean()/1e20
                rho, en, err = Target.PlotEnProfile(
                    trange=[t-0.015, t+0.015], Plot=False)
                axD.plot(rho, en/1e19, '--o', ms=15, c=col, mec=col,
                         label=r'$\overline{n_e}$ = %3.2f' % enLabel +
                         ' I$_p$ = %2.1f' % ip.mean() + ' MA')
                axD.errorbar(rho, en/1e19, yerr=err/1e19,
                             fmt='none', ecolor=col)
                rhoL, Lambda = Target.computeLambda(
                    trange=[t-0.015, t+0.015], Plot=False)
                axL.plot(rhoL[rhoL < rho.max()], Lambda[rhoL < rho.max()],
                         '-', lw=3, color=col,
                         label=r'$\overline{n_e}$ = %3.2f' % enLabel +
                         ' I$_p$ = %2.1f' % ip.mean() + ' MA')

        leg = ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        for handle, text in zip(leg.legendHandles, leg.get_texts()):
            text.set_color(handle.get_color())
            handle.set_visible(False)
        ax1.set_ylabel(r'$\overline{n_e} H-5 [10^{20}$m$^{-2}]$')
        ax1.set_title(r'I$_p$ scan at constant q$_{95}$')
        ax1.set_ylim([0, 0.6])
        axEnL[0].set_ylabel(r'n$_e$/n$_e(\rho_p = 1)$')
        for i in range(3):
            axEnL[i].axes.get_xaxis().set_visible(False)
            axEnL[i].set_xlim([0.98, 1.04])
            axEnL[i].set_ylim([1e-1, 4])
            axEnL[i].set_yscale('log')
            leg = axEnL[i].legend(loc='best', numpoints=1,
                                  frameon=False, fontsize=14)
            for handle, text in zip(leg.legendHandles, leg.get_texts()):
                text.set_color(handle.get_color())
                handle.set_visible(False)

        axDivL[0].set_ylabel(r'n$_e[10^{19}$m$^{-3}]$')
        for i in range(3):
            axDivL[i].axes.get_xaxis().set_visible(False)
            axDivL[i].set_xlim([0.98, 1.04])
            axDivL[i].set_ylim([0, 6])
            leg = axDivL[i].legend(loc='best', numpoints=1,
                                   frameon=False, fontsize=14)
            for handle, text in zip(leg.legendHandles, leg.get_texts()):
                text.set_color(handle.get_color())
                handle.set_visible(False)
        axLamL[0].set_ylabel(r'$\Lambda_{div}$')
        for i in range(3):
            axLamL[i].set_xlabel(r'$\rho_p$')
            axLamL[i].set_xlim([0.98, 1.04])
            axLamL[i].set_ylim([1e-2, 15])
            axLamL[i].axhline(1, ls='--', color='grey', lw=3)
            leg = axLamL[i].legend(loc='best', numpoints=1,
                                   frameon=False, fontsize=14)
            for handle, text in zip(leg.legendHandles, leg.get_texts()):
                text.set_color(handle.get_color())
                handle.set_visible(False)
            axLamL[i].set_yscale('log')
            axLamL[i].xaxis.set_ticks(np.arange(0.98, 1.05, 0.02))

        mpl.pylab.savefig('../pdfbox/IpConstantQ95_Profiles_UsDiv.pdf',
                          bbox_to_inches='tight')

    elif selection == 25:
        shotList = (34105, 34102, 34106)
        colorLS = ('#C90015', '#7B0Ce7', '#437356')
        tList = ((1.1, 2.83),
                 (1.1, 2.83, 3.41),
                 (1.1, 2.34, 3.09))
        fig = mpl.pylab.figure(figsize=(16, 17))
        fig.subplots_adjust(hspace=0.25)

        ax1 = mpl.pylab.subplot2grid((4, 3), (0, 0), colspan=3)
        # list of subplots for the profiles of Li-Bea
        axEnL = (mpl.pylab.subplot2grid((4, 3), (1, 0)),
                 mpl.pylab.subplot2grid((4, 3), (1, 1)),
                 mpl.pylab.subplot2grid((4, 3), (1, 2)))

        # list of subplots for the Divertor density profiles
        axDivL = (mpl.pylab.subplot2grid((4, 3), (2, 0)),
                  mpl.pylab.subplot2grid((4, 3), (2, 1)),
                  mpl.pylab.subplot2grid((4, 3), (2, 2)))

        # list of subplots for Lambda profiles
        axLamL = (mpl.pylab.subplot2grid((4, 3), (3, 0)),
                  mpl.pylab.subplot2grid((4, 3), (3, 1)),
                  mpl.pylab.subplot2grid((4, 3), (3, 2)))

        for shot, col, tL in itertools.izip(
                shotList, colorLS, tList):
            # get the current and compute the average current in Ip
            Mag = dd.shotfile('MAG', shot)
            _ii = np.where((Mag('Ipa').time >= 3) &
                           (Mag('Ipa').time <= 3.3))[0]
            ip = Mag('Ipa').data[_ii]/1e6
            Mag.close()
            # this is the line average density
            diag = dd.shotfile('DCN', shot)
            neEdge = diag('H-5')
            diag.close()
            ax1.plot(neEdge.time, neEdge.data/1e20, '-', color=col, lw=2,
                     label=r'Shot # %5i' % shot +
                     ' I$_p$ = %2.1f' % ip.mean() + ' MA')
            ax1.set_ylim([0, 1])
            ax1.set_xlim([0, 4.5])
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
            for t, ax in zip(tL, axEnL):
                _idx = np.where((neLBtime >= t-0.02) & (neLBtime <= t+0.02))[0]
                y = np.zeros((_idx.size, 200))
                for n, _iDummy in zip(_idx, range(_idx.size)):
                    S = UnivariateSpline(rhoP[n, ::-1],
                                         neLB[n, ::-1], s=0)
                    y[_iDummy, :] = S(rhoFake)/S(1)
                _idx = np.where((neEdge.time >= t-0.02) &
                                (neEdge.time <= t+0.02))[0]
                enLabel = neEdge.data[_idx].mean()/1e20
                ax.plot(rhoFake, np.mean(y, axis=0), '-', color=col,
                        lw=3, label=r'$\overline{n_e}$ = %3.2f' % enLabel +
                        ' I$_p$ = %2.1f' % ip.mean() + ' MA')
                ax.fill_between(rhoFake, np.mean(y, axis=0) -
                                np.std(y, axis=0), np.mean(y, axis=0) +
                                np.std(y, axis=0),
                                facecolor=col, edgecolor='none',
                                alpha=0.5)
                ax.set_yscale('log')

            # Now we need to compute the profiles at the target
            Target = langmuir.Target(shot)
            for t, axD, axL in zip(tL, axDivL, axLamL):
                _idx = np.where((neEdge.time >= t-0.02) &
                                (neEdge.time <= t+0.02))[0]
                enLabel = neEdge.data[_idx].mean()/1e20
                rho, en, err = Target.PlotEnProfile(
                    trange=[t-0.015, t+0.015], Plot=False)
                axD.plot(rho, en/1e19, '--o', ms=15, c=col, mec=col,
                         label=r'$\overline{n_e}$ = %3.2f' % enLabel +
                         ' I$_p$ = %2.1f' % ip.mean() + ' MA')
                axD.errorbar(rho, en/1e19, yerr=err/1e19,
                             fmt='none', ecolor=col)
                rhoL, Lambda = Target.computeLambda(
                    trange=[t-0.015, t+0.015], Plot=False)
                axL.plot(rhoL[rhoL < rho.max()], Lambda[rhoL < rho.max()],
                         '-', lw=3, color=col,
                         label=r'$\overline{n_e}$ = %3.2f' % enLabel +
                         ' I$_p$ = %2.1f' % ip.mean() + ' MA')

        leg = ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        for handle, text in zip(leg.legendHandles, leg.get_texts()):
            text.set_color(handle.get_color())
            handle.set_visible(False)
        ax1.set_ylabel(r'$\overline{n_e} H-5 [10^{20}$m$^{-2}]$')
        ax1.set_title(r'I$_p$ scan at constant B$_{t}$')
        ax1.set_ylim([0, 0.6])
        axEnL[0].set_ylabel(r'n$_e$/n$_e(\rho_p = 1)$')
        for i in range(3):
            axEnL[i].axes.get_xaxis().set_visible(False)
            axEnL[i].set_xlim([0.98, 1.05])
            axEnL[i].set_ylim([1e-1, 4])
            axEnL[i].set_yscale('log')
            leg = axEnL[i].legend(loc='best', numpoints=1,
                                  frameon=False, fontsize=14)
            for handle, text in zip(leg.legendHandles, leg.get_texts()):
                text.set_color(handle.get_color())
                handle.set_visible(False)
        axDivL[0].set_ylabel(r'n$_e[10^{19}$m$^{-3}]$')
        for i in range(3):
            axDivL[i].axes.get_xaxis().set_visible(False)
            axDivL[i].set_xlim([0.98, 1.05])
            axDivL[i].set_ylim([0, 6])
            leg = axDivL[i].legend(loc='best', numpoints=1,
                                   frameon=False, fontsize=14)
            for handle, text in zip(leg.legendHandles, leg.get_texts()):
                text.set_color(handle.get_color())
                handle.set_visible(False)
        axLamL[0].set_ylabel(r'$\Lambda_{div}$')
        for i in range(3):
            axLamL[i].set_xlabel(r'$\rho_p$')
            axLamL[i].set_xlim([0.98, 1.05])
            axLamL[i].set_ylim([1e-2, 15])
            axLamL[i].axhline(1, ls='--', color='grey', lw=3)
            leg = axLamL[i].legend(loc='best', numpoints=1,
                                   frameon=False, fontsize=14)
            for handle, text in zip(leg.legendHandles, leg.get_texts()):
                text.set_color(handle.get_color())
                handle.set_visible(False)
            axLamL[i].set_yscale('log')
            axLamL[i].xaxis.set_ticks(np.arange(0.98, 1.05, 0.02))
        mpl.pylab.savefig('../pdfbox/IpConstantBt_Profiles_UsDiv.pdf',
                          bbox_to_inches='tight')

    elif selection == 26:
        shotL = ((34103, 34102, 34104),
                 (34105, 34102, 34106))
        TitleL = (r'Constant q$_{95}$', r'Constant B$_t$')
        fig, Ax = mpl.pylab.subplots(figsize=(14, 6),
                                     nrows=1, ncols=2, sharey=True)
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')
        fig.subplots_adjust(bottom=0.17)
        for sL, ax, title in zip(shotL, Ax, TitleL):
            for shot, col, ip in zip(sL, colorLS, ('0.6', '0.8', '1')):
                Eq = eqtools.AUGDDData(shot)
                Eq.remapLCFS()

                # load now the tracer for the same shot all at 3s
                myTracer = get_fieldline_tracer('RK4', machine='AUG', shot=shot,
                                                time=3, interp='quintic', rev_bt=True)
                # height of magnetic axis
                zMAxis = myTracer.eq.axis.__dict__['z']
                # height of Xpoint
                zXPoint = myTracer.eq.xpoints['xpl'].__dict__['z']
                rXPoint = myTracer.eq.xpoints['xpl'].__dict__['r']
                # now determine at the height of the zAxis the R of the LCFS
                _idTime = np.argmin(np.abs(Eq.getTimeBase()-3))
                RLcfs = Eq.getRLCFS()[_idTime, :]
                ZLcfs = Eq.getZLCFS()[_idTime, :]
                # onlye the part greater then xaxis
                ZLcfs = ZLcfs[RLcfs > myTracer.eq.axis.__dict__['r']]
                RLcfs = RLcfs[RLcfs > myTracer.eq.axis.__dict__['r']]
                Rout = RLcfs[np.argmin(np.abs(ZLcfs[~np.isnan(ZLcfs)]-zMAxis))]
                rmin = np.linspace(Rout+0.001, 2.19, num=30)
                # this is R-Rsep
                rMid = rmin-Rout
                # this is Rho
                rho = Eq.rz2psinorm(rmin, np.repeat(
                    zMAxis, rmin.size), 3, sqrt=True)
                # now create the Traces and lines
                fieldLines = [myTracer.trace(r, zMAxis, mxstep=100000,
                                             ds=1e-2, tor_lim=20.0*np.pi) for r in rmin]

                fieldLinesZ = [line.filter(['R', 'Z'],
                                           [[rXPoint, 2], [-10, zXPoint]]) for line in fieldLines]
                Lpar = np.array([])
                for line in fieldLinesZ:
                    try:
                        _dummy = np.abs(line.S[0] - line.S[-1])
                    except:
                        _dummy = np.nan
                    Lpar = np.append(Lpar, _dummy)

                ax.plot(rho, Lpar, '-', color=col, lw=2,
                        label=r'I$_p$ = ' + ip+' MA')
            ax.set_xlim([1, 1.05])
            ax.set_yscale('log')
            ax.set_xlabel(r'$\rho_p$')
            ax.set_title(title)
            ax.legend(loc='best', numpoints=1, frameon=False)

        Ax[1].axes.get_yaxis().set_visible(False)
        Ax[0].set_ylabel(r'L$_{\parallel}$ [m]')
        Ax[0].set_ylim([0.1, 5])
        mpl.pylab.savefig('../pdfbox/IpScanLparallel.pdf',
                          bbox_to_inches='tight')

    elif selection == 27:
        sL = (34103, 34102, 34104)
        fig, Ax = mpl.pylab.subplots(figsize=(6, 10),
                                     nrows=2, ncols=1)
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')
        fig.subplots_adjust(bottom=0.12, right=0.98, wspace=0.2, left=0.25)
        for shot, col, ip in zip(sL, colorLS, ('0.6', '0.8', '1')):
            Eq = eqtools.AUGDDData(shot)
            Eq.remapLCFS()
            # load now the tracer for the same shot all at 3s
            myTracer = get_fieldline_tracer('RK4', machine='AUG', shot=shot,
                                            time=3, interp='quintic', rev_bt=True)
            # height of magnetic axis
            zMAxis = myTracer.eq.axis.__dict__['z']
            # height of Xpoint
            zXPoint = myTracer.eq.xpoints['xpl'].__dict__['z']
            rXPoint = myTracer.eq.xpoints['xpl'].__dict__['r']
            # now determine at the height of the zAxis the R of the LCFS
            _idTime = np.argmin(np.abs(Eq.getTimeBase()-3))
            RLcfs = Eq.getRLCFS()[_idTime, :]
            ZLcfs = Eq.getZLCFS()[_idTime, :]
            # onlye the part greater then xaxis
            ZLcfs = ZLcfs[RLcfs > myTracer.eq.axis.__dict__['r']]
            RLcfs = RLcfs[RLcfs > myTracer.eq.axis.__dict__['r']]
            Rout = RLcfs[np.argmin(np.abs(ZLcfs[~np.isnan(ZLcfs)]-zMAxis))]
            rmin = np.linspace(Rout+0.001, 2.19, num=30)
            # this is R-Rsep
            rMid = rmin-Rout
            # this is Rho
            rho = Eq.rz2psinorm(rmin, np.repeat(
                zMAxis, rmin.size), 3, sqrt=True)
            # now create the Traces and lines
            fieldLines = [myTracer.trace(r, zMAxis, mxstep=100000,
                                         ds=1e-2, tor_lim=20.0*np.pi) for r in rmin]

            fieldLinesZ = [line.filter(['R', 'Z'],
                                       [[rXPoint, 2], [-10, zXPoint]]) for line in fieldLines]
            Lpar = np.array([])
            for line in fieldLinesZ:
                try:
                    _dummy = np.abs(line.S[0] - line.S[-1])
                except:
                    _dummy = np.nan
                Lpar = np.append(Lpar, _dummy)

            Ax[0].contour(myTracer.eq.R, myTracer.eq.Z,
                          myTracer.eq.psiN, np.linspace(0.01, 0.95, num=9),
                          colors=col, linestyles='-', lw=0.5)
            Ax[0].contour(myTracer.eq.R, myTracer.eq.Z, myTracer.eq.psiN, [1],
                          colors=col, linestyle='-', linewidths=2)
            Ax[0].contour(myTracer.eq.R, myTracer.eq.Z, myTracer.eq.psiN,
                          np.linspace(1.01, 1.16, num=5),
                          colors=col, linestyles='--', linewidths=0.5)

            Ax[1].plot(rho, Lpar, '-', color=col, lw=2,
                       label=r'I$_p$ = ' + ip+' MA')

        Ax[1].set_xlim([1, 1.05])
        Ax[1].set_ylim([0.1, 5])
        Ax[1].set_yscale('log')
        Ax[1].set_xlabel(r'$\rho_p$')
        Ax[1].legend(loc='best', numpoints=1, frameon=False)
        Ax[1].set_ylabel(r'L$_{\parallel}$ [m]')

        Ax[0].plot(myTracer.eq.wall['R'], myTracer.eq.wall['Z'], 'k', lw=3)
        Ax[0].set_aspect('equal')
        Ax[0].set_xlabel(r'R')
        Ax[0].set_ylabel(r'Z')

        mpl.pylab.savefig(
            '../pdfbox/EquilibraLparallelConstantQ95.pdf', bbox_to_inches='tight')

    elif selection == 28:
        sL = (34105, 34102, 34106)
        fig, Ax = mpl.pylab.subplots(figsize=(6, 10),
                                     nrows=2, ncols=1)
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')
        fig.subplots_adjust(bottom=0.12, right=0.98, wspace=0.2, left=0.25)
        for shot, col, ip in zip(sL, colorLS, ('0.6', '0.8', '1')):
            Eq = eqtools.AUGDDData(shot)
            Eq.remapLCFS()
            # load now the tracer for the same shot all at 3s
            myTracer = get_fieldline_tracer('RK4', machine='AUG', shot=shot,
                                            time=3, interp='quintic', rev_bt=True)
            # height of magnetic axis
            zMAxis = myTracer.eq.axis.__dict__['z']
            # height of Xpoint
            zXPoint = myTracer.eq.xpoints['xpl'].__dict__['z']
            rXPoint = myTracer.eq.xpoints['xpl'].__dict__['r']
            # now determine at the height of the zAxis the R of the LCFS
            _idTime = np.argmin(np.abs(Eq.getTimeBase()-3))
            RLcfs = Eq.getRLCFS()[_idTime, :]
            ZLcfs = Eq.getZLCFS()[_idTime, :]
            # onlye the part greater then xaxis
            ZLcfs = ZLcfs[RLcfs > myTracer.eq.axis.__dict__['r']]
            RLcfs = RLcfs[RLcfs > myTracer.eq.axis.__dict__['r']]
            Rout = RLcfs[np.argmin(np.abs(ZLcfs[~np.isnan(ZLcfs)]-zMAxis))]
            rmin = np.linspace(Rout+0.001, 2.19, num=30)
            # this is R-Rsep
            rMid = rmin-Rout
            # this is Rho
            rho = Eq.rz2psinorm(rmin, np.repeat(
                zMAxis, rmin.size), 3, sqrt=True)
            # now create the Traces and lines
            fieldLines = [myTracer.trace(r, zMAxis, mxstep=100000,
                                         ds=1e-2, tor_lim=20.0*np.pi) for r in rmin]

            fieldLinesZ = [line.filter(['R', 'Z'],
                                       [[rXPoint, 2], [-10, zXPoint]]) for line in fieldLines]
            Lpar = np.array([])
            for line in fieldLinesZ:
                try:
                    _dummy = np.abs(line.S[0] - line.S[-1])
                except:
                    _dummy = np.nan
                Lpar = np.append(Lpar, _dummy)

            Ax[0].contour(myTracer.eq.R, myTracer.eq.Z,
                          myTracer.eq.psiN, np.linspace(0.01, 0.95, num=9),
                          colors=col, linestyles='-', lw=0.5)
            Ax[0].contour(myTracer.eq.R, myTracer.eq.Z, myTracer.eq.psiN, [1],
                          colors=col, linestyle='-', linewidths=2)
            Ax[0].contour(myTracer.eq.R, myTracer.eq.Z, myTracer.eq.psiN,
                          np.linspace(1.01, 1.16, num=5),
                          colors=col, linestyles='--', linewidths=0.5)

            Ax[1].plot(rho, Lpar, '-', color=col, lw=2,
                       label=r'I$_p$ = ' + ip+' MA')

        Ax[1].set_xlim([1, 1.05])
        Ax[1].set_ylim([0.1, 5])
        Ax[1].set_yscale('log')
        Ax[1].set_xlabel(r'$\rho_p$')
        Ax[1].legend(loc='best', numpoints=1, frameon=False)
        Ax[1].set_ylabel(r'L$_{\parallel}$ [m]')

        Ax[0].plot(myTracer.eq.wall['R'], myTracer.eq.wall['Z'], 'k', lw=3)
        Ax[0].set_aspect('equal')
        Ax[0].set_xlabel(r'R')
        Ax[0].set_ylabel(r'Z')

        mpl.pylab.savefig(
            '../pdfbox/EquilibraLparallelConstantBt.pdf', bbox_to_inches='tight')

    elif selection == 29:
        shotL = (34103, 34102, 34104)
        strokeL = (((1.68, 1.82), (2.889, 3.02), (3.48, 3.56)),
                   ((1.90, 2.00), (3.151, 3.22), (3.69, 3.81)),
                   ((1.90, 2.02), (3.090, 3.22), (3.70, 3.81)))
        IpLabel = ('0.6', '0.8', '1')
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        fig = mpl.pylab.figure(figsize=(16, 14))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        axPdf = (mpl.pylab.subplot2grid((3, 3), (0, 0)),
                 mpl.pylab.subplot2grid((3, 3), (0, 1)),
                 mpl.pylab.subplot2grid((3, 3), (0, 2)))

        axStr = (mpl.pylab.subplot2grid((3, 3), (1, 0)),
                 mpl.pylab.subplot2grid((3, 3), (1, 1)),
                 mpl.pylab.subplot2grid((3, 3), (1, 2)))

        ax = mpl.pylab.subplot2grid((3, 3), (2, 0), colspan=3)

        for shot, strokes, col, ip in zip(shotL, strokeL, colorL, IpLabel):
            Turbo = augFilaments.Filaments(shot, Xprobe=1790)
            Turbo.loadPosition()
            Target = langmuir.Target(shot)
            for time, axP, axS in zip(
                    strokes, axPdf, axStr):
                Turbo.blobAnalysis(Probe='Isat_m06', trange=[time[0], time[1]],
                                   block=[0.012, 0.4])
                # compute the corresponding Lambda at the location of the probe
                rho, Lambda = Target.computeLambda(
                    trange=[time[0], time[1]], Plot=False)
                _iidx = np.where((Turbo.tPos >= time[0]) & (
                    Turbo.tPos <= time[1]))[0]
                rhoProbe = Turbo.rhoProbe[_iidx].mean()
                LambdaProbe = Lambda[np.argmin(np.abs(rho-1.01))]
                # compute the pdf using Scott rule
                h, b = Turbo.blob.pdf(bins='scott', density=True, normed=True)
                axP.plot((b[1:]+b[:-1])/2, h, lw=2, color=col,
                         label=r'I$_p$ = ' + ip +
                         ' MA, $\Lambda_{div} = %3.2f$' % LambdaProbe)

                cs, tau, err = Turbo.blob.cas(Type='thresh', detrend=True)
                axS.plot(tau*1e6, cs, lw=2, color=col,
                         label=r'I$_p$ = ' + ip +
                         ' MA, $\Lambda_{div} = %3.2f$' % LambdaProbe)
                ax.plot(LambdaProbe, Turbo.blob.act*1e6, 'o',
                        ms=15, color=col, label='I$_p$ ' + ip + 'Ma')

        for a, b in zip(axPdf, axStr):
            a.set_xlabel(r'$\tilde{I}_s/\sigma$')
            a.set_ylim([1e-4, 1])
            a.set_yscale('log')
            a.legend(loc='best', numpoints=1, frameon=False, fontsize=12)
            b.set_xlabel(r't[$\mu$s]')
            b.set_xlim([-50, 50])
            b.set_ylim([-0.02, 0.15])
            b.legend(loc='best', numpoints=1, frameon=False, fontsize=12)

        axPdf[1].axes.get_yaxis().set_visible(False)
        axPdf[2].axes.get_yaxis().set_visible(False)
        axPdf[0].set_ylabel(r'Pdf')

        axStr[1].axes.get_yaxis().set_visible(False)
        axStr[2].axes.get_yaxis().set_visible(False)
        axStr[0].set_ylabel(r'$\delta$I$_s$')
        ax.set_xlabel(r'$\Lambda_{div}$ @ $\rho = 1.01$')
        ax.set_ylabel(r'$\tau_{ac}[\mu$s]')
        ax.set_xscale('log')
        ax.set_ylim([0, 200])
        mpl.pylab.savefig('../pdfbox/PdfStructureCurrentScan_ConstantQ95.pdf',
                          bbox_to_inches='tight')

    elif selection == 30:
        shotL = (34105, 34102, 34106)
        strokeL = (((1.90, 2.00), (3.090, 3.22)),
                   ((1.90, 2.00), (3.151, 3.22), (3.69, 3.81)),
                   ((1.90, 2.02), (3.090, 3.22), (3.70, 3.81)))
        IpLabel = ('0.6', '0.8', '1')
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        fig = mpl.pylab.figure(figsize=(16, 14))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        axPdf = (mpl.pylab.subplot2grid((3, 3), (0, 0)),
                 mpl.pylab.subplot2grid((3, 3), (0, 1)),
                 mpl.pylab.subplot2grid((3, 3), (0, 2)))

        axStr = (mpl.pylab.subplot2grid((3, 3), (1, 0)),
                 mpl.pylab.subplot2grid((3, 3), (1, 1)),
                 mpl.pylab.subplot2grid((3, 3), (1, 2)))

        ax = mpl.pylab.subplot2grid((3, 3), (2, 0), colspan=3)

        for shot, strokes, col, ip in zip(shotL, strokeL, colorL, IpLabel):
            Turbo = augFilaments.Filaments(shot, Xprobe=1790)
            Turbo.loadPosition()
            Target = langmuir.Target(shot)
            for time, axP, axS in zip(
                    strokes, axPdf, axStr):
                Turbo.blobAnalysis(Probe='Isat_m06', trange=[time[0], time[1]],
                                   block=[0.012, 0.4])
                # compute the corresponding Lambda at the location of the probe
                rho, Lambda = Target.computeLambda(
                    trange=[time[0], time[1]], Plot=False)
                _iidx = np.where((Turbo.tPos >= time[0]) & (
                    Turbo.tPos <= time[1]))[0]
                rhoProbe = Turbo.rhoProbe[_iidx].mean()
                LambdaProbe = Lambda[np.argmin(np.abs(rho-1.01))]
                # compute the pdf using Scott rule
                h, b = Turbo.blob.pdf(bins='scott', density=True, normed=True)
                axP.plot((b[1:]+b[:-1])/2, h, lw=2, color=col,
                         label=r'I$_p$ = ' + ip +
                         ' MA, $\Lambda_{div} = %3.2f$' % LambdaProbe)

                cs, tau, err = Turbo.blob.cas(Type='thresh', detrend=True)
                axS.plot(tau*1e6, cs, lw=2, color=col,
                         label=r'I$_p$ = ' + ip +
                         ' MA, $\Lambda_{div} = %3.2f$' % LambdaProbe)
                ax.plot(LambdaProbe, Turbo.blob.act*1e6, 'o',
                        ms=15, color=col, label='I$_p$ ' + ip + 'Ma')

        for a, b in zip(axPdf, axStr):
            a.set_xlabel(r'$\tilde{I}_s/\sigma$')
            a.set_ylim([1e-4, 1])
            a.set_yscale('log')
            a.legend(loc='best', numpoints=1, frameon=False, fontsize=12)
            b.set_xlabel(r't[$\mu$s]')
            b.set_xlim([-50, 50])
            b.set_ylim([-0.02, 0.15])
            b.legend(loc='best', numpoints=1, frameon=False, fontsize=12)

        axPdf[1].axes.get_yaxis().set_visible(False)
        axPdf[2].axes.get_yaxis().set_visible(False)
        axPdf[0].set_ylabel(r'Pdf')

        axStr[1].axes.get_yaxis().set_visible(False)
        axStr[2].axes.get_yaxis().set_visible(False)
        axStr[0].set_ylabel(r'$\delta$I$_s$')
        ax.set_xlabel(r'$\Lambda_{div}$ @ $\rho = 1.01$')
        ax.set_ylabel(r'$\tau_{ac}[\mu$s]')
        ax.set_xscale('log')
        ax.set_ylim([0, 200])
        mpl.pylab.savefig('../pdfbox/PdfStructureCurrentScan_ConstantBt.pdf',
                          bbox_to_inches='tight')

    elif selection == 31:
        shotL = (34276, 34278)
        strokeL = ((3.75714, 3.80528), (3.76694, 3.83))
        IpLabel = ('Off', 'On')
        colorL = ('#82A17E', '#1E4682')
        fig, ax = mpl.pylab.subplots(figsize=(8, 14), nrows=3, ncols=1)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        for shot, strokes, col, ip, _idx in zip(shotL, strokeL,
                                                colorL, IpLabel, range(len(ax))):
            Turbo = augFilaments.Filaments(shot, Xprobe=1773)
            Turbo.loadPosition()
            Target = langmuir.Target(shot)
            Turbo.blobAnalysis(Probe='Isat_m06', trange=[strokes[0], strokes[1]],
                               block=[0.012, 1])
            # compute the corresponding Lambda at the location of the probe
            rho, Lambda = Target.computeLambda(
                trange=[time[0], time[1]], Plot=False)
            _iidx = np.where((Turbo.tPos >= time[0]) & (
                Turbo.tPos <= time[1]))[0]
            rhoProbe = Turbo.rhoProbe[_iidx].mean()
            LambdaProbe = Lambda[np.argmin(np.abs(rho-1.01))]
            # compute the pdf using Scott rule
            h, b = Turbo.blob.pdf(bins='scott', density=True, normed=True)
            ax[0].plot((b[1:]+b[:-1])/2, h, lw=2, color=col,
                       label=r'Cryo ' + ip +
                       ' $\Lambda_{div} = %3.2f$' % LambdaProbe)

            cs, tau, err = Turbo.blob.cas(Type='thresh', detrend=True)
            ax[1].plot(tau*1e6, cs, lw=2, color=col,
                       label=r'Cryo ' + ip +
                       ' $\Lambda_{div} = %3.2f$' % LambdaProbe)
            ax[2].plot(LambdaProbe, Turbo.blob.act*1e6, 'o',
                       ms=15, color=col, label='I$_p$ ' + ip + 'Ma')

        ax[0].set_xlabel(r'$\tilde{I}_s/\sigma$')
        ax[0].set_ylim([1e-4, 1])
        ax[0].set_yscale('log')
        ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        ax[1].set_xlabel(r't[$\mu$s]')
        ax[1].set_xlim([-50, 50])
        ax[1].set_ylim([-0.02, 0.15])
        ax[1].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        ax[0].set_ylabel(r'Pdf')
        ax[1].set_ylabel(r'$\delta$I$_s$')
        ax[2].set_xlabel(r'$\Lambda_{div}$ @ $\rho = 1.01$')
        ax[2].set_ylabel(r'$\tau_{ac}[\mu$s]')
        mpl.pylab.savefig('../pdfbox/PdfStructureHmodeCryoOnOff.pdf',
                          bbox_to_inches='tight')

    elif selection == 32:
        shotL = (34276, 34281)
        strokeL = ((3.75714, 3.80528), (3.73724, 3.83))
        IpLabel = ('Off', 'On')
        colorL = ('#82A17E', '#1E4682')
        fig, ax = mpl.pylab.subplots(figsize=(8, 14), nrows=3, ncols=1)
        fig.subplots_adjust(wspace=0.3, hspace=0.3, left=0.2)

        for shot, strokes, col, ip, _idx in zip(shotL, strokeL,
                                                colorL, IpLabel, range(len(ax))):
            Turbo = augFilaments.Filaments(shot, Xprobe=1773)
            Turbo.loadPosition()
            Target = langmuir.Target(shot)
            Turbo.blobAnalysis(Probe='Isat_m06', trange=[strokes[0], strokes[1]],
                               block=[0.012, 1])
            # compute the corresponding Lambda at the location of the probe
            rho, Lambda = Target.computeLambda(trange=[strokes[0],
                                                       strokes[1]], Plot=False)
            _iidx = np.where((Turbo.tPos >= strokes[0]) & (
                Turbo.tPos <= strokes[1]))[0]
            rhoProbe = Turbo.rhoProbe[_iidx].mean()
            LambdaProbe = Lambda[np.argmin(np.abs(rho-1.01))]
            # compute the pdf using Scott rule
            h, b = Turbo.blob.pdf(bins='scott', density=True, normed=True)
            ax[0].plot((b[1:]+b[:-1])/2, h, lw=2, color=col,
                       label=r'Cryo ' + ip +
                       ' $\Lambda_{div} = %3.2f$' % LambdaProbe)

            cs, tau, err = Turbo.blob.cas(Type='thresh', detrend=True)
            ax[1].plot(tau*1e6, cs, lw=2, color=col,
                       label=r'Cryo ' + ip +
                       ' $\Lambda_{div} = %3.2f$' % LambdaProbe)
            ax[2].plot(LambdaProbe, Turbo.blob.act*1e6, 'o',
                       ms=15, color=col, label='I$_p$ ' + ip + 'Ma')

        ax[0].set_xlabel(r'$\tilde{I}_s/\sigma$')
        ax[0].set_ylim([1e-4, 3])
        ax[0].set_yscale('log')
        ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        ax[1].set_xlabel(r't[$\mu$s]')
        ax[1].set_xlim([-50, 50])
        ax[1].set_ylim([-0.02, 0.35])
        ax[1].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        ax[0].set_ylabel(r'Pdf')
        ax[1].set_ylabel(r'$\delta$I$_s$')
        ax[2].set_xlabel(r'$\Lambda_{div}$ @ $\rho = 1.01$')
        ax[2].set_ylabel(r'$\tau_{ac}[\mu$s]')
        ax[2].set_xscale('log')
        mpl.pylab.savefig('../pdfbox/PdfStructureHmodeCryoOnOffMatch.pdf',
                          bbox_to_inches='tight')

    elif selection == 33:
        shotList = (34276, 34277)
        pufL = ('Low Div', 'Up Mid')
        colorList = ('#C90015', '#7B0Ce7')

        rg, zg = map_equ.get_gc()
        fig, ax = mpl.pylab.subplots(figsize=(5, 7), nrows=1, ncols=1)
        fig.subplots_adjust(left=0.2)
        for key in rg.iterkeys():
            ax.plot(rg[key], zg[key], '-k')
        for shot, col in zip(shotList, colorList):
            Eq = equilibrium.equilibrium(device='AUG', time=3, shot=shot)
            ax.contour(Eq.R, Eq.Z, Eq.psiN[:],
                       np.linspace(0, 0.95, 10), colors=col, linestyles='-')
            ax.contour(Eq.R, Eq.Z, Eq.psiN[:],
                       np.linspace(1.01, 1.05, 5), colors=col, linestyles='--')
            ax.contour(Eq.R, Eq.Z, Eq.psiN[:],
                       [1], colors=col, linestyles='-', linewidths=3)

        ax.set_xlabel('R(m)')
        ax.set_ylabel('Z(m)')
        ax.set_aspect('equal')
        ax.arrow(1.25, 1.4, 0, -0.3, lw=4, color=colorList[1])
        ax.arrow(2, -1.4, 0, 0.3, lw=4, color=colorList[0])
        ax.text(2.4, 1.35, str(shotList[0]), color=colorList[0], fontsize=16)
        ax.text(2.4, 1.2, str(shotList[1]), color=colorList[1], fontsize=16)
        mpl.pylab.savefig('../pdfbox/PuffingLocation.pdf',
                          bbox_to_inches='tight')

    elif selection == 34:
        shotList = (34276, 34278)
        pufL = ('Off', 'On')
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(10, 8), nrows=2,
                                     ncols=1, sharex=True)
        for shot, col, _idx, lab in zip(
                shotList, colorList, range(len(shotList)),
                pufL):
            diag = dd.shotfile('MAC', shot)
            Gas = neutrals.Neutrals(shot)
            ax[_idx].plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21, 'k')
            ax[_idx].set_ylabel(r'D$_2$ [10$^{21}$s$^{-1}$]')
            ax[_idx].text(1, 16, 'Cryo ' + lab, fontsize=16)
            T = ax[_idx].twinx()
            T.plot(diag('Ipolsola').time, -diag('Ipolsola').data/1e3,
                   color='red', rasterized=True)
            T.set_ylabel(r'Ipolsola [kA]', color='red')
            T.set_yticks([0, 10, 20, 30])
            T.set_ylim([-10, 30])
            for t in T.yaxis.get_ticklabels():
                t.set_color('red')
        ax[0].axes.get_xaxis().set_visible(False)
        ax[1].set_xlim([0, 7])
        ax[1].set_xlabel(r't[s]')
        mpl.pylab.savefig('../pdfbox/PuffingIpolsola%5i' % shotList[0] +
                          '_%5i' % shotList[1]+'.pdf', bbox_to_inches='tight')

    elif selection == 35:
        shotList = (34276, 34281)
        pufL = ('Off', 'On')
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(10, 10), nrows=2,
                                     ncols=1)
        #fig.subplots_adjust(left=0.17, right=0.8)
        for shot, col, _idx, lab in zip(
                shotList,
                colorList, range(len(shotList)),
                pufL):
            diag = dd.shotfile('MAC', shot)
            Gas = neutrals.Neutrals(shot)
            ax[_idx].plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21, 'k')
            ax[_idx].set_ylabel(r'D$_2$ [10$^{21}$s$^{-1}$]')
            ax[_idx].text(1, 16, 'Cryo ' + lab, fontsize=16)
            T = ax[_idx].twinx()
            T.plot(diag('Ipolsola').time, -diag('Ipolsola').data/1e3, color='red',
                   rasterized=True)
            T.set_ylabel(r'Ipolsola [kA]', color='red')
            T.set_yticks([0, 10, 20, 30])
            T.set_ylim([-10, 30])
            for t in T.yaxis.get_ticklabels():
                t.set_color('red')

        ax[0].axes.get_xaxis().set_visible(False)
        ax[1].set_xlim([0, 7])
        ax[0].set_xlim([0, 7])
        ax[1].set_xlabel(r't[s]')
        mpl.pylab.savefig('../pdfbox/PuffingIpolsola%5i' % shotList[0] +
                          '_%5i' % shotList[1]+'.pdf', bbox_to_inches='tight')

    elif selection == 36:
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        Diods = {'D10': {'Name': 'S2L0A09',
                         'xchord': [1.519, 1.606],
                         'ychord': [-1.131, -1.11]},
                 'D15': {'Name': 'S2L0A14',
                         'xchord': [1.510, 1.629],
                         'ychord': [-1.121, -1.025]},
                 'D17': {'Name': 'S2L1A00',
                         'xchord': [1.437, 2.173],
                         'ychord': [-1.063, -0.282]},
                 'D12': {'Name': 'S2L0A11',
                         'xchord': [1.515, 1.614],
                         'ychord': [-1.127, -1.080]}}
        # create the figure
        fig, ax = mpl.pylab.subplots(figsize=(8, 12),
                                     ncols=1, nrows=5, sharex=True)
        fig.subplots_adjust(wspace=0.05, top=0.98)

        for shot, col, ip in zip(shotList, colorL, currentL):
            # load equilibrium
             # load the H-5
            Dcn = dd.shotfile('DCN', shot)
            H5 = Dcn('H-5')
            # spline evaluation (shorter time faster)
            S = UnivariateSpline(H5.time, H5.data/1e19, s=0)
            Target = langmuir.Target(shot)
            # peak density vs time
            peakTarget = decimate(bottleneck.move_mean(
                np.nanmax(Target.OuterTargetNe, axis=0)/1e20, window=30),
                10, ftype='fir', zero_phase=True)
            tPeak = decimate(Target.time, 10, ftype='fir', zero_phase=True)

            # limit to the time corresponding to 20 ms before the maximum of
            # density i.e. before disruption
            _idx = np.where((tPeak >= 1) & (
                tPeak <= H5.time[np.argmax(H5.data)]-0.02))
            ax[0].plot(S(tPeak[_idx]), peakTarget[_idx], '.', ms=3, color=col,
                       label=r'# %5i' % shot + r' I$_p$ = %2.1f' % ip + ' MA',
                       rasterized=True)
            # now read the signals of AXUV and save them in an array
            XVS = dd.shotfile('XVS', shot)
            for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
                dummy = XVS(Diods[key]['Name'])
                # we need to filter the signal and
                # we can also downsampling the data since there is
                # no need for such a large amount of data
                Fs = 500e3
                sig = bw_filter(dummy.data, 200e3, Fs, 'lowpass', order=6)
                # now downsampling the data two times
                sig = decimate(decimate(sig, 10, ftype='fir', zero_phase=True),
                               10, ftype='fir', zero_phase=True)
                t = decimate(decimate(dummy.time, 10, ftype='fir', zero_phase=True),
                             10, ftype='fir', zero_phase=True)

                _idx = np.where((t >= 1) & (
                    t <= H5.time[np.argmax(H5.data)]-0.02))
                ax[i+1].plot(S(t[_idx]), sig[_idx]/1e3, '.',
                             ms=2, color=col, rasterized=True)

            XVS.close()
            Dcn.close()

        rg, zg = map_equ.get_gc()
        Eq = equilibrium.equilibrium(device='AUG', time=2, shot=shotList[0])
        for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
            inS = inset_axes(ax[i+1], height="50%", width='50%', loc=2)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        np.linspace(1.01, 1.05, 5), colors='grey', linestyles='--',
                        linewidhs=0.7)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        [1], colors='red', linestyles='-', linewidths=1)
            for KK in rg.iterkeys():
                inS.plot(rg[KK], zg[KK], '-', color='grey')
            inS.set_ylim([-1.55, -0.8])
            inS.set_xlim([1.1, 1.95])
            inS.plot(Diods[key]['xchord'], Diods[key]['ychord'], 'k--', lw=2)
            inS.axis('off')
            inS.set_aspect('equal')

        ax[0].axes.get_xaxis().set_visible(False)
        l = ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        for t, col in zip(l.get_texts(), colorL):
            t.set_color(col)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[2].axes.get_xaxis().set_visible(False)
        ax[3].axes.get_xaxis().set_visible(False)
        ax[4].set_xlim([0, 5])
        ax[4].set_xlabel(r'n$_e$ H-5 [10$^{19}$m$^{-2}$]')
        ax[0].set_ylabel(r'$[10^{20}$m$^{-3}]$')
        ax[1].set_ylabel(r'kW/m$^2$')
        ax[2].set_ylabel(r'kW/m$^2$')
        ax[3].set_ylabel(r'kW/m$^2$')
        ax[4].set_ylabel(r'kW/m$^2$')
        mpl.pylab.savefig('../pdfbox/RadiationPeakDensityIpScan_constantQ95.pdf',
                          bbox_to_inches='tight', dpi=400)

        # ----------------------
        # same with just one LoS
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        Diods = {'D10': {'Name': 'S2L0A09',
                         'xchord': [1.519, 1.606],
                         'ychord': [-1.131, -1.11]}}
        # create the figure
        fig, ax = mpl.pylab.subplots(figsize=(6, 8),
                                     ncols=1, nrows=2, sharex=True)
        fig.subplots_adjust(left=0.15, wspace=0.05, top=0.98)

        for shot, col, ip in zip(shotList, colorL, currentL):
            # load equilibrium
             # load the H-5
            Dcn = dd.shotfile('DCN', shot)
            H5 = Dcn('H-5')
            # spline evaluation (shorter time faster)
            S = UnivariateSpline(H5.time, H5.data/1e19, s=0)
            Target = langmuir.Target(shot)
            # peak density vs time
            peakTarget = decimate(bottleneck.move_mean(
                np.nanmax(Target.OuterTargetNe, axis=0)/1e20, window=30),
                10, ftype='fir', zero_phase=True)
            tPeak = decimate(Target.time, 10, ftype='fir', zero_phase=True)

            # limit to the time corresponding to 20 ms before the maximum of
            # density i.e. before disruption
            _idx = np.where((tPeak >= 1) & (
                tPeak <= H5.time[np.argmax(H5.data)]-0.02))
            ax[0].plot(S(tPeak[_idx]), peakTarget[_idx], '.', ms=3, color=col,
                       label=r'# %5i' % shot + r' I$_p$ = %2.1f' % ip + ' MA',
                       rasterized=True)
            # now read the signals of AXUV and save them in an array
            XVS = dd.shotfile('XVS', shot)
            for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
                dummy = XVS(Diods[key]['Name'])
                # we need to filter the signal and
                # we can also downsampling the data since there is
                # no need for such a large amount of data
                Fs = 500e3
                sig = bw_filter(dummy.data, 200e3, Fs, 'lowpass', order=6)
                # now downsampling the data two times
                sig = decimate(decimate(sig, 10, ftype='fir', zero_phase=True),
                               10, ftype='fir', zero_phase=True)
                t = decimate(decimate(dummy.time, 10, ftype='fir', zero_phase=True),
                             10, ftype='fir', zero_phase=True)

                _idx = np.where((t >= 1) & (
                    t <= H5.time[np.argmax(H5.data)]-0.02))
                ax[i+1].plot(S(t[_idx]), sig[_idx]/1e3, '.',
                             ms=2, color=col, rasterized=True)

            XVS.close()
            Dcn.close()

        rg, zg = map_equ.get_gc()
        Eq = equilibrium.equilibrium(device='AUG', time=2, shot=shotList[0])
        for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
            inS = inset_axes(ax[i+1], height="70%", width='70%', loc=2)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        np.linspace(1.01, 1.05, 5), colors='grey', linestyles='--',
                        linewidhs=0.7)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        [1], colors='red', linestyles='-', linewidths=1)
            for KK in rg.iterkeys():
                inS.plot(rg[KK], zg[KK], '-', color='grey')
            inS.set_ylim([-1.5, -0.9])
            inS.set_xlim([1.1, 1.9])
            inS.plot(Diods[key]['xchord'], Diods[key]['ychord'], 'k--', lw=2)
            inS.axis('off')
            inS.set_aspect('equal')

        ax[0].axes.get_xaxis().set_visible(False)
        l = ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        for t, col in zip(l.get_texts(), colorL):
            t.set_color(col)
        ax[1].set_xlim([0, 4])
        ax[1].set_xlabel(r'n$_e$ H-5 [10$^{19}$m$^{-2}$]')
        ax[0].set_ylabel(r'$[10^{20}$m$^{-3}]$')
        ax[1].set_ylabel(r'kW/m$^2$')
        mpl.pylab.savefig('../pdfbox/RadiationPeakDensityIpScan_constantQ95Zoom.pdf',
                          bbox_to_inches='tight', dpi=400)

    elif selection == 37:
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        Diods = {'D10': {'Name': 'S2L0A09',
                         'xchord': [1.519, 1.606],
                         'ychord': [-1.131, -1.11]},
                 'D15': {'Name': 'S2L0A14',
                         'xchord': [1.510, 1.629],
                         'ychord': [-1.121, -1.025]},
                 'D17': {'Name': 'S2L1A00',
                         'xchord': [1.437, 2.173],
                         'ychord': [-1.063, -0.282]},
                 'D12': {'Name': 'S2L0A11',
                         'xchord': [1.515, 1.614],
                         'ychord': [-1.127, -1.080]}}
        # create the figure
        fig, ax = mpl.pylab.subplots(figsize=(8, 12),
                                     ncols=1, nrows=5, sharex=True)
        fig.subplots_adjust(wspace=0.05, top=0.98)
        rg, zg = map_equ.get_gc()
        for shot, col, ip in zip(shotList, colorL, currentL):
            # load equilibrium

            # load the H-5
            Dcn = dd.shotfile('DCN', shot)
            H5 = Dcn('H-5')
            # spline evaluation (shorter time faster)
            S = UnivariateSpline(H5.time, H5.data/1e19, s=0)
            Target = langmuir.Target(shot)
            # peak density vs time
            peakTarget = decimate(bottleneck.move_mean(
                np.nanmax(Target.OuterTargetNe, axis=0)/1e20, window=30),
                10, ftype='fir', zero_phase=True)
            tPeak = decimate(Target.time, 10, ftype='fir', zero_phase=True)

            # limit to the time corresponding to 20 ms before the maximum of
            # density i.e. before disruption
            _idx = np.where((tPeak >= 1) & (
                tPeak <= H5.time[np.argmax(H5.data)]-0.02))
            ax[0].plot(S(tPeak[_idx]), peakTarget[_idx], '.', ms=3, color=col,
                       label=r'# %5i' % shot + r' I$_p$ = %2.1f' % ip + ' MA',
                       rasterized=True)
            # now read the signals of AXUV and save them in an array
            XVS = dd.shotfile('XVS', shot)
            for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
                dummy = XVS(Diods[key]['Name'])
                # we need to filter the signal and
                # we can also downsampling the data since there is
                # no need for such a large amount of data
                Fs = 500e3
                sig = bw_filter(dummy.data, 200e3, Fs, 'lowpass', order=6)
                # now downsampling the data two times
                sig = decimate(decimate(sig, 10, ftype='fir', zero_phase=True),
                               10, ftype='fir', zero_phase=True)
                t = decimate(decimate(dummy.time, 10, ftype='fir', zero_phase=True),
                             10, ftype='fir', zero_phase=True)

                _idx = np.where((t >= 1) & (
                    t <= H5.time[np.argmax(H5.data)]-0.02))
                ax[i+1].plot(S(t[_idx]), sig[_idx]/1e3, '.',
                             ms=2, color=col, rasterized=True)

            XVS.close()
            Dcn.close()

        Eq = equilibrium.equilibrium(device='AUG', time=2, shot=shotList[0])
        for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
            inS = inset_axes(ax[i+1], height="50%", width='50%', loc=2)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        np.linspace(1.01, 1.05, 5), colors='grey', linestyles='--',
                        linewidhs=0.7)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        [1], colors='red', linestyles='-', linewidths=1)
            for KK in rg.iterkeys():
                inS.plot(rg[KK], zg[KK], '-', color='grey')
            inS.set_ylim([-1.55, -0.65])
            inS.set_xlim([1.1, 1.75])
            inS.plot(Diods[key]['xchord'], Diods[key]['ychord'], 'k--', lw=2)
            inS.axis('off')
            inS.set_aspect('equal')

        ax[0].axes.get_xaxis().set_visible(False)
        l = ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        for t, col in zip(l.get_texts(), colorL):
            t.set_color(col)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[2].axes.get_xaxis().set_visible(False)
        ax[3].axes.get_xaxis().set_visible(False)
        ax[4].set_xlim([0, 5])
        ax[4].set_xlabel(r'n$_e$ H-5 [10$^{19}$m$^{-2}$]')
        ax[0].set_ylabel(r'$[10^{20}$m$^{-3}]$')
        ax[1].set_ylabel(r'kW/m$^2$')
        ax[2].set_ylabel(r'kW/m$^2$')
        ax[3].set_ylabel(r'kW/m$^2$')
        ax[4].set_ylabel(r'kW/m$^2$')
        mpl.pylab.savefig('../pdfbox/RadiationPeakDensityIpScan_constantBT.pdf',
                          bbox_to_inches='tight', dpi=400)
        # -----------------
        # Only one LoS
        Diods = {'D10': {'Name': 'S2L0A09',
                         'xchord': [1.519, 1.606],
                         'ychord': [-1.131, -1.11]}}
        # create the figure
        fig, ax = mpl.pylab.subplots(figsize=(6, 8),
                                     ncols=1, nrows=2, sharex=True)
        fig.subplots_adjust(left=0.17, wspace=0.05, top=0.98)

        for shot, col, ip in zip(shotList, colorL, currentL):
            # load equilibrium
             # load the H-5
            Dcn = dd.shotfile('DCN', shot)
            H5 = Dcn('H-5')
            # spline evaluation (shorter time faster)
            S = UnivariateSpline(H5.time, H5.data/1e19, s=0)
            Target = langmuir.Target(shot)
            # peak density vs time
            peakTarget = decimate(bottleneck.move_mean(
                np.nanmax(Target.OuterTargetNe, axis=0)/1e20, window=30),
                10, ftype='fir', zero_phase=True)
            tPeak = decimate(Target.time, 10, ftype='fir', zero_phase=True)

            # limit to the time corresponding to 20 ms before the maximum of
            # density i.e. before disruption
            _idx = np.where((tPeak >= 1) & (
                tPeak <= H5.time[np.argmax(H5.data)]-0.02))
            ax[0].plot(S(tPeak[_idx]), peakTarget[_idx], '.', ms=3, color=col,
                       label=r'# %5i' % shot + r' I$_p$ = %2.1f' % ip + ' MA',
                       rasterized=True)
            # now read the signals of AXUV and save them in an array
            XVS = dd.shotfile('XVS', shot)
            for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
                dummy = XVS(Diods[key]['Name'])
                # we need to filter the signal and
                # we can also downsampling the data since there is
                # no need for such a large amount of data
                Fs = 500e3
                sig = bw_filter(dummy.data, 200e3, Fs, 'lowpass', order=6)
                # now downsampling the data two times
                sig = decimate(decimate(sig, 10, ftype='fir', zero_phase=True),
                               10, ftype='fir', zero_phase=True)
                t = decimate(decimate(dummy.time, 10, ftype='fir', zero_phase=True),
                             10, ftype='fir', zero_phase=True)

                _idx = np.where((t >= 1) & (
                    t <= H5.time[np.argmax(H5.data)]-0.02))
                ax[i+1].plot(S(t[_idx]), sig[_idx]/1e3, '.',
                             ms=2, color=col, rasterized=True)

            XVS.close()
            Dcn.close()

        rg, zg = map_equ.get_gc()
        Eq = equilibrium.equilibrium(device='AUG', time=2, shot=shotList[0])
        for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
            inS = inset_axes(ax[i+1], height="70%", width='70%', loc=2)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        np.linspace(1.01, 1.05, 5), colors='grey', linestyles='--',
                        linewidhs=0.7)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        [1], colors='red', linestyles='-', linewidths=1)
            for KK in rg.iterkeys():
                inS.plot(rg[KK], zg[KK], '-', color='grey')
            inS.set_ylim([-1.5, -0.9])
            inS.set_xlim([1.1, 1.8])
            inS.plot(Diods[key]['xchord'], Diods[key]['ychord'], 'k--', lw=2)
            inS.axis('off')
            inS.set_aspect('equal')

        ax[0].axes.get_xaxis().set_visible(False)
        l = ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        for t, col in zip(l.get_texts(), colorL):
            t.set_color(col)
        ax[1].set_xlim([0, 4])
        ax[1].set_xlabel(r'n$_e$ H-5 [10$^{19}$m$^{-2}$]')
        ax[0].set_ylabel(r'$[10^{20}$m$^{-3}]$')
        ax[1].set_ylabel(r'kW/m$^2$')
        mpl.pylab.savefig('../pdfbox/RadiationPeakDensityIpScan_constantBTZoom.pdf',
                          bbox_to_inches='tight', dpi=400)
    elif selection == 38:
        shotList = (34276, 34277)
        colorL = ('#C90015', '#7B0Ce7')
        pufL = ('Low Div', 'Up Div')
        Diods = {'D10': {'Name': 'S2L0A09',
                         'xchord': [1.519, 1.606],
                         'ychord': [-1.131, -1.11]},
                 'D15': {'Name': 'S2L0A14',
                         'xchord': [1.510, 1.629],
                         'ychord': [-1.121, -1.025]},
                 'D17': {'Name': 'S2L1A00',
                         'xchord': [1.437, 2.173],
                         'ychord': [-1.063, -0.282]},
                 'D12': {'Name': 'S2L0A11',
                         'xchord': [1.515, 1.614],
                         'ychord': [-1.127, -1.080]}}

        # create the figure
        fig, ax = mpl.pylab.subplots(figsize=(8, 12),
                                     ncols=1, nrows=5, sharex=True)
        fig.subplots_adjust(wspace=0.05, top=0.98)
        rg, zg = map_equ.get_gc()
        for shot, col, ip in zip(shotList, colorL, pufL):
            # load equilibrium
            Eq = equilibrium.equilibrium(device='AUG', time=2, shot=shot)

            # load the H-5
            Dcn = dd.shotfile('DCN', shot)
            H5 = Dcn('H-5')
            # spline evaluation (shorter time faster)
            S = UnivariateSpline(H5.time, H5.data/1e19, s=0)
            Target = langmuir.Target(shot)
            # peak density vs time
            peakTarget = decimate(bottleneck.move_mean(
                np.nanmax(Target.OuterTargetNe, axis=0)/1e20, window=30),
                10, ftype='fir', zero_phase=True)
            tPeak = decimate(Target.time, 10, ftype='fir', zero_phase=True)

            # limit to the time corresponding to 20 ms before the maximum of
            # density i.e. before disruption
            _idx = np.where((tPeak >= 1) & (
                tPeak <= H5.time[np.argmax(H5.data)]-0.02))
            ax[0].plot(S(tPeak[_idx]), peakTarget[_idx], '.', ms=3, color=col,
                       label=r'# %5i' % shot + r' Puff from ' + ip,
                       rasterized=True)
            # now read the signals of AXUV and save them in an array
            XVS = dd.shotfile('XVS', shot)
            for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
                dummy = XVS(Diods[key]['Name'])
                # we need to filter the signal and
                # we can also downsampling the data since there is
                # no need for such a large amount of data
                Fs = 500e3
                sig = bw_filter(dummy.data, 200e3, Fs, 'lowpass', order=6)
                # now downsampling the data two times
                sig = decimate(decimate(sig, 10, ftype='fir', zero_phase=True),
                               10, ftype='fir', zero_phase=True)
                t = decimate(decimate(dummy.time, 10, ftype='fir', zero_phase=True),
                             10, ftype='fir', zero_phase=True)

                _idx = np.where((t >= 1) & (
                    t <= H5.time[np.argmax(H5.data)]-0.02))
                ax[i+1].plot(S(t[_idx]), sig[_idx]/1e3, '.',
                             ms=2, color=col, rasterized=True)
            XVS.close()
            Dcn.close()

        Eq = equilibrium.equilibrium(device='AUG', time=2, shot=shotList[0])
        for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
            inS = inset_axes(ax[i+1], height="50%", width='50%', loc=2)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        np.linspace(1.01, 1.05, 5), colors='grey', linestyles='--',
                        linewidhs=0.7)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        [1], colors='red', linestyles='-', linewidths=1)
            for KK in rg.iterkeys():
                inS.plot(rg[KK], zg[KK], '-', color='grey')
            inS.set_ylim([-1.55, -0.65])
            inS.set_xlim([1.1, 1.75])
            inS.plot(Diods[key]['xchord'], Diods[key]['ychord'], 'k--', lw=2)
            inS.axis('off')
            inS.set_aspect('equal')

        ax[0].axes.get_xaxis().set_visible(False)
        l = ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        for t, col in zip(l.get_texts(), colorL):
            t.set_color(col)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[2].axes.get_xaxis().set_visible(False)
        ax[3].axes.get_xaxis().set_visible(False)
        ax[4].set_xlim([0, 5])
        ax[4].set_xlabel(r'n$_e$ H-5 [10$^{19}$m$^{-2}$]')
        ax[0].set_ylabel(r'$[10^{20}$m$^{-3}]$')
        ax[1].set_ylabel(r'kW/m$^2$')
        ax[2].set_ylabel(r'kW/m$^2$')
        ax[3].set_ylabel(r'kW/m$^2$')
        ax[4].set_ylabel(r'kW/m$^2$')
        mpl.pylab.savefig('../pdfbox/RadiationPeakDensityLowUpDivPuff.pdf',
                          bbox_to_inches='tight', dpi=400)

    elif selection == 39:
        shotList = (34276, 34281)
        colorL = ('#C90015', '#7B0Ce7')
        cryoL = ('Off', 'On')
        Diods = {'D10': {'Name': 'S2L0A09',
                         'xchord': [1.519, 1.606],
                         'ychord': [-1.131, -1.11]},
                 'D15': {'Name': 'S2L0A14',
                         'xchord': [1.510, 1.629],
                         'ychord': [-1.121, -1.025]},
                 'D17': {'Name': 'S2L1A00',
                         'xchord': [1.437, 2.173],
                         'ychord': [-1.063, -0.282]},
                 'D12': {'Name': 'S2L0A11',
                         'xchord': [1.515, 1.614],
                         'ychord': [-1.127, -1.080]}}

        # create the figure
        fig, ax = mpl.pylab.subplots(figsize=(8, 12),
                                     ncols=1, nrows=5, sharex=True)
        fig.subplots_adjust(wspace=0.05, top=0.98)
        rg, zg = map_equ.get_gc()
        for shot, col, ip in zip(shotList, colorL, cryoL):
            # load equilibrium
            Eq = equilibrium.equilibrium(device='AUG', time=2, shot=shot)

            # load the H-5
            Dcn = dd.shotfile('DCN', shot)
            H5 = Dcn('H-5')
            # spline evaluation (shorter time faster)
            S = UnivariateSpline(H5.time, H5.data/1e19, s=0)
            Target = langmuir.Target(shot)
            # peak density vs time
            peakTarget = decimate(bottleneck.move_mean(
                np.nanmax(Target.OuterTargetNe, axis=0)/1e20, window=30),
                10, ftype='fir', zero_phase=True)
            tPeak = decimate(Target.time, 10, ftype='fir', zero_phase=True)

            # limit to the time corresponding to 20 ms before the maximum of
            # density i.e. before disruption
            _idx = np.where((tPeak >= 1) & (
                tPeak <= H5.time[np.argmax(H5.data)]-0.02))
            ax[0].plot(S(tPeak[_idx]), peakTarget[_idx], '.', ms=3, color=col,
                       label=r'# %5i' % shot + r' Cryo ' + ip,
                       rasterized=True)
            # now read the signals of AXUV and save them in an array
            XVS = dd.shotfile('XVS', shot)
            for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
                dummy = XVS(Diods[key]['Name'])
                # we need to filter the signal and
                # we can also downsampling the data since there is
                # no need for such a large amount of data
                Fs = 500e3
                sig = bw_filter(dummy.data, 200e3, Fs, 'lowpass', order=6)
                # now downsampling the data two times
                sig = decimate(decimate(sig, 10, ftype='fir', zero_phase=True),
                               10, ftype='fir', zero_phase=True)
                t = decimate(decimate(dummy.time, 10, ftype='fir', zero_phase=True),
                             10, ftype='fir', zero_phase=True)

                _idx = np.where((t >= 1) & (
                    t <= H5.time[np.argmax(H5.data)]-0.02))
                ax[i+1].plot(S(t[_idx]), sig[_idx]/1e3, '.',
                             ms=2, color=col, rasterized=True)
            XVS.close()
            Dcn.close()

        Eq = equilibrium.equilibrium(device='AUG', time=2, shot=shotList[0])
        for key, i in zip(sorted(Diods.keys()), range(len(Diods))):
            inS = inset_axes(ax[i+1], height="50%", width='50%', loc=2)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        np.linspace(1.01, 1.05, 5), colors='grey', linestyles='--',
                        linewidhs=0.7)
            inS.contour(Eq.R, Eq.Z, Eq.psiN[:],
                        [1], colors='red', linestyles='-', linewidths=1)
            for KK in rg.iterkeys():
                inS.plot(rg[KK], zg[KK], '-', color='grey')
            inS.set_ylim([-1.55, -0.65])
            inS.set_xlim([1.1, 1.75])
            inS.plot(Diods[key]['xchord'], Diods[key]['ychord'], 'k--', lw=2)
            inS.axis('off')
            inS.set_aspect('equal')

        ax[0].axes.get_xaxis().set_visible(False)
        l = ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=12)
        for t, col in zip(l.get_texts(), colorL):
            t.set_color(col)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[2].axes.get_xaxis().set_visible(False)
        ax[3].axes.get_xaxis().set_visible(False)
        ax[4].set_xlim([0, 5])
        ax[4].set_xlabel(r'n$_e$ H-5 [10$^{19}$m$^{-2}$]')
        ax[0].set_ylabel(r'$[10^{20}$m$^{-3}]$')
        ax[1].set_ylabel(r'kW/m$^2$')
        ax[2].set_ylabel(r'kW/m$^2$')
        ax[3].set_ylabel(r'kW/m$^2$')
        ax[4].set_ylabel(r'kW/m$^2$')
        mpl.pylab.savefig('../pdfbox/RadiationPeakDensityCryoOnOff.pdf',
                          bbox_to_inches='tight', dpi=400)

    elif selection == 40:
        shotList = (34276, 34278)
        fig = mpl.pylab.figure(figsize=(12, 14))
        fig.subplots_adjust(hspace=0.25, right=0.96, top=0.96)
        ax1 = mpl.pylab.subplot2grid((4, 2), (0, 0), colspan=2)
        ax2 = mpl.pylab.subplot2grid((4, 2), (1, 0), colspan=2)
        ax3 = mpl.pylab.subplot2grid((4, 2), (2, 0))
        ax4 = mpl.pylab.subplot2grid((4, 2), (2, 1))
        ax5 = mpl.pylab.subplot2grid((4, 2), (3, 0))
        ax6 = mpl.pylab.subplot2grid((4, 2), (3, 1))
        axProf = (ax3, ax4)
        axLamb = (ax5, ax6)
        colorLS = ('#C90015', '#7B0Ce7')
        colorL = ('#82A17E', '#1E4682', '#DD6D3D', 'cyan', 'violet')
        Cryo = ('No', 'On')
        df = pd.read_csv('../data/MEM_Topic21.csv')
        for shot, col, Cry, _axP, _axL in zip(
                shotList, colorLS, Cryo, axProf, axLamb):
            Gas = neutrals.Neutrals(shot)
            ax1.plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                     color=col, lw=3, label='# %5i' % shot + ' Cryo ' + Cry)
            diag = dd.shotfile('DCN', shot)
            ax2.plot(diag('H-5').time, diag('H-5').data/1e19, color=col, lw=3)
            LiB = libes.Libes(shot)
            for time, _col in zip(
                    ('1', '2', '3', '4', '5'), colorL):
                try:
                    tmin = df['tmin'+time][df['shot'] == shot].values
                    tmax = df['tmax'+time][df['shot'] == shot].values
                    Data = xray.open_dataarray(
                        '../data/Shot%5i' % shot + '_'+time+'Stroke.nc')
                    _axL.plot(Data.rhoLambda, Data.LambdaProfile,
                              color=_col, lw=3)
                    ax1.axvspan(tmin, tmax, color=_col, edgecolor='white',
                                alpha=0.5)
                    ax2.axvspan(tmin, tmax, color=_col, edgecolor='white',
                                alpha=0.5)
                    p, e = LiB.averageProfile(trange=[tmin, tmax],
                                              interelm=True, threshold=1000)
                    S = UnivariateSpline(LiB.rho, p, s=0)
                    _axP.plot(LiB.rho, p/S(1), '-', lw=2, color=_col)
                    _axP.fill_between(LiB.rho, (p-e)/S(1), (p+e)/S(1), facecolor=_col,
                                      edgecolor='none', alpha=0.5)

                except:
                    print('Not done')
                _axP.set_title('# %5i' % shot)
        ax1.set_xlim([0, 6.5])
        ax1.set_ylabel(r'D$_2$  [10$^{21}$]')
        ax1.legend(loc='best', numpoints=1, frameon=False)
        ax1.axes.get_xaxis().set_visible(False)
        ax2.set_xlim([0, 6.5])
        ax2.set_ylabel(r'n$_e [10^{19}$m$^{-2}]$')
        ax2.set_xlabel(r't [s]')

        ax3.set_ylim([1e-2, 4])
        ax3.set_ylabel(r'n$_e/$n$_e(\rho=1)$')
        ax4.set_ylim([1e-2, 4])
        ax4.axes.get_yaxis().set_visible(False)
        ax4.axes.get_xaxis().set_visible(False)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax3.set_xlim([0.95, 1.1])
        ax4.set_xlim([0.95, 1.1])
        ax5.set_ylim([1e-1, 20])
        ax5.set_ylabel(r'$\Lambda_{div}$')
        ax5.set_xlim([0.95, 1.1])
        ax6.set_xlim([0.95, 1.1])
        ax6.set_ylim([1e-1, 20])
        ax6.axhline(1, ls='--', lw=2)
        ax5.axhline(1, ls='--', lw=2)
        ax6.axes.get_yaxis().set_visible(False)
        ax5.set_yscale('log')
        ax6.set_yscale('log')
        ax5.set_xlabel(r'$\rho$')
        ax6.set_xlabel(r'$\rho$')
        mpl.pylab.savefig('../pdfbox/Shot_%5i' % shotList[0]
                          + '_'+'%5i' % shotList[1]+'_InterELMprofiles.pdf',
                          bbox_to_inches='tight')

    elif selection == 41:
        shotList = (34103, 34102, 34104)
        iPL = ('0.6', '0.8', '1')
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        fig, ax = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.16)
        for shot, col, ip in zip(shotList, colorL, iPL):
            for s in ('1', '2', '3', '4'):
                try:
                    Data = xray.open_dataarray(
                        '../data/Shot%5i' % shot + '_'+s+'Stroke.nc')
                    Size = Data.TauB*np.sqrt(
                        np.power(4e-3/Data.Lags[0], 2) +
                        np.power(6.9e-3/Data.Lags[1], 2))
                    if Size*1e3 > 100:
                        Size /= 2.
                    ax.plot(Data.Lambda, Size*1e3, 'o', color=col, ms=18)
                except:
                    pass
        ax.set_xscale('log')
        ax.text(0.1, 0.9, '0.6 MA', color=colorL[0], transform=ax.transAxes)
        ax.text(0.1, 0.8, '0.8 MA', color=colorL[1], transform=ax.transAxes)
        ax.text(0.1, 0.7, '1 MA', color=colorL[2], transform=ax.transAxes)
        ax.set_title(r'Constant q$_{95}$')
        ax.set_ylabel(r'$\delta_b$ [mm]')
        ax.set_xlabel(r'$\Lambda$ @ $\rho$ = 1.03')
        mpl.pylab.savefig('../pdfbox/BlobSizeCurrentScanConstantQ95.pdf',
                          bbox_to_inches='tight')

    elif selection == 42:
        shotList = (34105, 34102, 34106)
        iPL = ('0.6', '0.8', '1')
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        fig, ax = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.16)
        for shot, col, ip in zip(shotList, colorL, iPL):
            for s in ('1', '2', '3', '4'):
                try:
                    Data = xray.open_dataarray(
                        '../data/Shot%5i' % shot + '_'+s+'Stroke.nc')
                    Size = Data.TauB*np.sqrt(
                        np.power(4e-3/Data.Lags[0], 2) +
                        np.power(6.9e-3/Data.Lags[1], 2))
                    if Size*1e3 > 150:
                        Size /= 2.
                    if np.isfinite(Data.Lambda) is False:
                        L = Data.LambdaProfile[np.nanargmin(
                            np.abs(Data.rhoLambda-1.03))]
                    else:
                        L = Data.Lambda
                    ax.plot(L, Size*1e3, 'o', color=col,
                            ms=18)
                    print Size
                    print Data.Lambda
                    print L
                except:
                    print('not done for shot %5i stroke %1i', shot, s)
                    pass
        ax.set_xscale('log')
        ax.text(0.1, 0.9, '0.6 MA', color=colorL[0], transform=ax.transAxes)
        ax.text(0.1, 0.8, '0.8 MA', color=colorL[1], transform=ax.transAxes)
        ax.text(0.1, 0.7, '1 MA', color=colorL[2], transform=ax.transAxes)
        ax.set_title(r'Constant B$_{t}$')
        ax.set_ylabel(r'$\delta_b$ [mm]')
        ax.set_xlabel(r'$\Lambda$ @ $\rho$ = 1.03')
        mpl.pylab.savefig('../pdfbox/BlobSizeCurrentScanConstantBT.pdf',
                          bbox_to_inches='tight')
    elif selection == 43:
        shotList = (34103, 34104, 34105, 34102, 34106)
        iPL = ('0.6', '0.8', '1')
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        fig, ax = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.16)
        for shot in shotList:
            for s in ('1', '2', '3', '4'):
                try:
                    Data = xray.open_dataarray(
                        '../data/Shot%5i' % shot + '_'+s+'Stroke.nc')
                    Size = Data.TauB*np.sqrt(
                        np.power(4e-3/Data.Lags[0], 2) +
                        np.power(6.9e-3/Data.Lags[1], 2))
                    if Size*1e3 > 150:
                        Size /= 2.
                    if np.isfinite(Data.Lambda) is False:
                        L = Data.LambdaProfile[np.nanargmin(
                            np.abs(Data.rhoLambda-1.03))]
                    else:
                        L = Data.Lambda
                    ax.plot(L, Size*1e3, 'o', color='k', ms=10)
                except:
                    print('not done for shot %5i stroke %1i', shot, s)
                    pass
        ax.set_xscale('log')
        ax.set_ylabel(r'$\delta_b$ [mm]')
        ax.set_xlabel(r'$\Lambda$ @ $\rho$ = 1.03')
        mpl.pylab.savefig('../pdfbox/BlobSizeCurrentScanAll.pdf',
                          bbox_to_inches='tight')
    elif selection == 44:
        shotList = (34276, 34277)
        fig = mpl.pylab.figure(figsize=(12, 14))
        fig.subplots_adjust(hspace=0.25, right=0.96, top=0.96)
        ax1 = mpl.pylab.subplot2grid((4, 2), (0, 0), colspan=2)
        ax2 = mpl.pylab.subplot2grid((4, 2), (1, 0), colspan=2)
        ax3 = mpl.pylab.subplot2grid((4, 2), (2, 0))
        ax4 = mpl.pylab.subplot2grid((4, 2), (2, 1))
        ax5 = mpl.pylab.subplot2grid((4, 2), (3, 0))
        ax6 = mpl.pylab.subplot2grid((4, 2), (3, 1))
        axProf = (ax3, ax4)
        axLamb = (ax5, ax6)
        colorLS = ('#C90015', '#7B0Ce7')
        colorL = ('#82A17E', '#1E4682', '#DD6D3D', 'cyan', 'violet')
        Cryo = ('Low', 'Upp')
        df = pd.read_csv('../data/MEM_Topic21.csv')
        for shot, col, Cry, _axP, _axL in zip(
                shotList, colorLS, Cryo, axProf, axLamb):
            Gas = neutrals.Neutrals(shot)
            ax1.plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                     color=col, lw=3, label='# %5i' % shot + ' Puffing from ' + Cry)
            diag = dd.shotfile('DCN', shot)
            ax2.plot(diag('H-5').time, diag('H-5').data/1e19, color=col, lw=3)
            LiB = libes.Libes(shot)
            if shot == 34276:
                for time, _col in zip(
                        ('1', '2', '3', '4', '5'), colorL):
                    try:
                        tmin = df['tmin'+time][df['shot'] == shot].values
                        tmax = df['tmax'+time][df['shot'] == shot].values
                        Data = xray.open_dataarray(
                            '../data/Shot%5i' % shot + '_'+time+'Stroke.nc')
                        _axL.plot(Data.rhoLambda, Data.LambdaProfile,
                                  color=_col, lw=3)
                        ax1.axvspan(tmin, tmax, color=_col, edgecolor='white',
                                    alpha=0.5)
                        ax2.axvspan(tmin, tmax, color=_col, edgecolor='white',
                                    alpha=0.5)
                        p, e = LiB.averageProfile(trange=[tmin, tmax],
                                                  interelm=True, threshold=1000)
                        S = UnivariateSpline(LiB.rho, p, s=0)
                        _axP.plot(LiB.rho, p/S(1), '-', lw=2, color=_col)
                        _axP.fill_between(LiB.rho, (p-e)/S(1), (p+e)/S(1), facecolor=_col,
                                          edgecolor='none', alpha=0.5)

                    except:
                        print('Not done')
                    _axP.set_title('# %5i' % shot)

        ax1.set_xlim([0, 6.5])
        ax1.set_ylabel(r'D$_2$  [10$^{21}$]')
        ax1.legend(loc='best', numpoints=1, frameon=False)
        ax1.axes.get_xaxis().set_visible(False)
        ax2.set_xlim([0, 6.5])
        ax2.set_ylabel(r'n$_e [10^{19}$m$^{-2}]$')
        ax2.set_xlabel(r't [s]')

        ax3.set_ylim([1e-2, 4])
        ax3.set_ylabel(r'n$_e/$n$_e(\rho=1)$')
        ax4.set_ylim([1e-2, 4])
        ax4.axes.get_yaxis().set_visible(False)
        ax4.axes.get_xaxis().set_visible(False)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax3.set_xlim([0.95, 1.1])
        ax4.set_xlim([0.95, 1.1])
        ax5.set_ylim([1e-1, 20])
        ax5.set_ylabel(r'$\Lambda_{div}$')
        ax5.set_xlim([0.95, 1.1])
        ax6.set_xlim([0.95, 1.1])
        ax6.set_ylim([1e-1, 20])
        ax6.axhline(1, ls='--', lw=2)
        ax5.axhline(1, ls='--', lw=2)
        ax6.axes.get_yaxis().set_visible(False)
        ax5.set_yscale('log')
        ax6.set_yscale('log')
        ax5.set_xlabel(r'$\rho$')
        ax6.set_xlabel(r'$\rho$')
        mpl.pylab.savefig('../pdfbox/Shot_%5i' % shotList[0]
                          + '_'+'%5i' % shotList[1]+'_InterELMprofiles.pdf',
                          bbox_to_inches='tight')

    elif selection == 45:
        shotList = (34278, 34281)
        fig = mpl.pylab.figure(figsize=(12, 14))
        fig.subplots_adjust(hspace=0.25, right=0.96, top=0.96)
        ax1 = mpl.pylab.subplot2grid((4, 2), (0, 0), colspan=2)
        ax2 = mpl.pylab.subplot2grid((4, 2), (1, 0), colspan=2)
        ax3 = mpl.pylab.subplot2grid((4, 2), (2, 0))
        ax4 = mpl.pylab.subplot2grid((4, 2), (2, 1))
        ax5 = mpl.pylab.subplot2grid((4, 2), (3, 0))
        ax6 = mpl.pylab.subplot2grid((4, 2), (3, 1))
        axProf = (ax3, ax4)
        axLamb = (ax5, ax6)
        colorLS = ('#C90015', '#7B0Ce7')
        colorL = ('#82A17E', '#1E4682', '#DD6D3D', 'cyan', 'violet')
        Cryo = ('Low', 'Upp')
        df = pd.read_csv('../data/MEM_Topic21.csv')
        for shot, col, Cry, _axP, _axL in zip(
                shotList, colorLS, Cryo, axProf, axLamb):
            Gas = neutrals.Neutrals(shot)
            ax1.plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                     color=col, lw=3, label='# %5i' % shot)
            diag = dd.shotfile('DCN', shot)
            ax2.plot(diag('H-5').time, diag('H-5').data/1e19, color=col, lw=3)
            LiB = libes.Libes(shot)
            for time, _col in zip(
                    ('1', '2', '3', '4', '5'), colorL):
                try:
                    tmin = df['tmin'+time][df['shot'] == shot].values
                    tmax = df['tmax'+time][df['shot'] == shot].values
                    Data = xray.open_dataarray(
                        '../data/Shot%5i' % shot + '_'+time+'Stroke.nc')
                    _axL.plot(Data.rhoLambda, Data.LambdaProfile,
                              color=_col, lw=3)
                    ax1.axvspan(tmin, tmax, color=_col, edgecolor='white',
                                alpha=0.5)
                    ax2.axvspan(tmin, tmax, color=_col, edgecolor='white',
                                alpha=0.5)
                    p, e = LiB.averageProfile(trange=[tmin, tmax],
                                              interelm=True, threshold=1000)
                    S = UnivariateSpline(LiB.rho, p, s=0)
                    _axP.plot(LiB.rho, p/S(1), '-', lw=2, color=_col)
                    _axP.fill_between(LiB.rho, (p-e)/S(1), (p+e)/S(1), facecolor=_col,
                                      edgecolor='none', alpha=0.5)

                except:
                    print('Not done')
                _axP.set_title('# %5i' % shot)
        ax1.set_xlim([0, 6.5])
        ax1.set_ylabel(r'D$_2$  [10$^{21}$]')
        ax1.legend(loc='best', numpoints=1, frameon=False)
        ax1.axes.get_xaxis().set_visible(False)
        ax2.set_xlim([0, 6.5])
        ax2.set_ylabel(r'n$_e [10^{19}$m$^{-2}]$')
        ax2.set_xlabel(r't [s]')

        ax3.set_ylim([1e-2, 4])
        ax3.set_ylabel(r'n$_e/$n$_e(\rho=1)$')
        ax4.set_ylim([1e-2, 4])
        ax4.axes.get_yaxis().set_visible(False)
        ax4.axes.get_xaxis().set_visible(False)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.set_yscale('log')
        ax4.set_yscale('log')
        ax3.set_xlim([0.95, 1.1])
        ax4.set_xlim([0.95, 1.1])
        ax5.set_ylim([1e-1, 20])
        ax5.set_ylabel(r'$\Lambda_{div}$')
        ax5.set_xlim([0.95, 1.1])
        ax6.set_xlim([0.95, 1.1])
        ax6.set_ylim([1e-1, 20])
        ax6.axhline(1, ls='--', lw=2)
        ax5.axhline(1, ls='--', lw=2)
        ax6.axes.get_yaxis().set_visible(False)
        ax5.set_yscale('log')
        ax6.set_yscale('log')
        ax5.set_xlabel(r'$\rho$')
        ax6.set_xlabel(r'$\rho$')
        mpl.pylab.savefig('../pdfbox/Shot_%5i' % shotList[0]
                          + '_'+'%5i' % shotList[1]+'_InterELMprofiles.pdf',
                          bbox_to_inches='tight')
    elif selection == 46:
        shotList = (34278, 34281)
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(7, 10), nrows=2, ncols=1)
        fig.subplots_adjust(bottom=0.15, hspace=0.25, wspace=0.25)
        for shot, col in zip(shotList, colorList):
            Data = xray.open_dataarray(
                '../data/Shot%5i' % shot + '_5Stroke.nc')
            ax[0].plot(Data.t*1e6, Data.sel(sig='Isat_m06'),
                       color=col, lw=3, label='# %5i' % shot)
            err = Data.err.reshape(3, 501)
            ax[0].fill_between(Data.t*1e6, Data.sel(sig='Isat_m06')-err[0, :],
                               Data.sel(sig='Isat_m06')+err[0, :], edgecolor='white',
                               facecolor=col,
                               alpha=0.5)
            Size = Data.TauB*np.sqrt(
                np.power(4e-3/Data.Lags[0], 2) +
                np.power(6.9e-3/Data.Lags[1], 2))

            ax[1].semilogx(Data.Lambda, Size*1e3, 'o', ms=12, color=col)

        ax[0].set_xlabel(r't [$\mu$s]')
        ax[0].set_ylabel(r'$\delta I_s/\sigma$')
        ax[0].legend(loc=2, numpoints=1, frameon=False)
        ax[1].set_xlabel(r'$\Lambda_{div}$')
        ax[1].set_ylabel(r'$\delta_b$ [mm]')

        mpl.pylab.savefig('../pdfbox/CompareCas%5i' % shotList[0]+'_%5i' % shotList[1]+'.pdf',
                          bbox_to_inches='tight')

        shotList = (34276, 34281)
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(7, 10), nrows=2, ncols=1)
        fig.subplots_adjust(bottom=0.15, hspace=0.25, wspace=0.25)
        for shot, col in zip(shotList, colorList):
            Data = xray.open_dataarray(
                '../data/Shot%5i' % shot + '_3Stroke.nc')
            ax[0].plot(Data.t*1e6, Data.sel(sig='Isat_m06'),
                       color=col, lw=3, label='# %5i' % shot)
            err = Data.err.reshape(3, 501)
            ax[0].fill_between(Data.t*1e6, Data.sel(sig='Isat_m06')-err[0, :],
                               Data.sel(sig='Isat_m06')+err[0, :], edgecolor='white',
                               facecolor=col,
                               alpha=0.5)
            Size = Data.TauB*np.sqrt(
                np.power(4e-3/Data.Lags[0], 2) +
                np.power(6.9e-3/Data.Lags[1], 2))

            ax[1].semilogx(Data.Lambda, Size*1e3, 'o', ms=12, color=col)

        ax[0].set_xlabel(r't [$\mu$s]')
        ax[0].set_ylabel(r'$\delta I_s/\sigma$')
        ax[0].legend(loc=2, numpoints=1, frameon=False)
        ax[1].set_xlabel(r'$\Lambda_{div}$')
        ax[1].set_ylabel(r'$\delta_b$ [mm]')
        mpl.pylab.savefig('../pdfbox/CompareCas%5i' % shotList[0]+'_%5i' % shotList[1]+'.pdf',
                          bbox_to_inches='tight')

    elif selection == 47:
        shotList = (34276, 34278)
        fig = mpl.pylab.figure(figsize=(16, 14))
        fig.subplots_adjust(hspace=0.25, right=0.96, top=0.96)
        ax1 = mpl.pylab.subplot2grid((4, 4), (0, 0), colspan=4)
        ax2 = mpl.pylab.subplot2grid((4, 4), (1, 0), colspan=4)

        ax3 = mpl.pylab.subplot2grid((4, 4), (2, 0))
        ax4 = mpl.pylab.subplot2grid((4, 4), (2, 1))
        ax5 = mpl.pylab.subplot2grid((4, 4), (2, 2))
        ax6 = mpl.pylab.subplot2grid((4, 4), (2, 3))

        ax7 = mpl.pylab.subplot2grid((4, 4), (3, 0))
        ax8 = mpl.pylab.subplot2grid((4, 4), (3, 1))
        ax9 = mpl.pylab.subplot2grid((4, 4), (3, 2))
        ax10 = mpl.pylab.subplot2grid((4, 4), (3, 3))

        axProf = (ax3, ax4, ax5, ax6)
        axLamb = (ax7, ax8, ax9, ax10)
        colorLS = ('#C90015', '#7B0Ce7')
        Cryo = ('No', 'On')
        trangeL = ([3.75, 3.78], [4.24, 4.30], [4.76, 4.80], [5.23, 5.28])
        threshold = ([1000, 500, 400, 300], [1800, 1000, 1100, 1100])

        for shot, col, Cry, _Thr in zip(
                shotList, colorLS, Cryo, threshold):
            Gas = neutrals.Neutrals(shot)
            ax1.plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                     color=col, lw=3, label='# %5i' % shot + ' Cryo ' + Cry)
            diag = dd.shotfile('DCN', shot)
            ax2.plot(diag('H-5').time, diag('H-5').data/1e19, color=col, lw=3)
            LiB = libes.Libes(shot)
            Target = langmuir.Target(shot)
            for time, _axP, _axL, _t in zip(trangeL, axProf, axLamb, _Thr):
                tmin = time[0]
                tmax = time[1]
                p, e, Efold, pN, eN = LiB.averageProfile(trange=[tmin, tmax],
                                                         interelm=True, threshold=_t)
                _axP.plot(LiB.rho, pN, '-', lw=2, color=col)
                _axP.fill_between(LiB.rho, (pN-eN), (pN+eN), facecolor=col,
                                  edgecolor='none', alpha=0.5)
                rhoLambda, Lambda = Target.computeLambda(trange=[tmin, tmax], interelm=True,
                                                         threshold=_t)
                _axL.plot(rhoLambda, Lambda, color=col, lw=3)
                ax1.axvspan(tmin, tmax, color='gray', edgecolor='white',
                            alpha=0.5)
                ax2.axvspan(tmin, tmax, color='gray', edgecolor='white',
                            alpha=0.5)

        ax1.set_xlim([0, 6.5])
        ax1.set_ylabel(r'D$_2$  [10$^{21}$]')
        ax1.legend(loc='best', numpoints=1, frameon=False)
        ax1.axes.get_xaxis().set_visible(False)
        ax2.set_xlim([0, 6.5])
        ax2.set_ylabel(r'n$_e [10^{19}$m$^{-2}]$')
        ax2.set_xlabel(r't [s]')

        for _ax, t in zip(axProf, trangeL):
            _ax.set_yscale('log')
            _ax.set_ylim([1e-2, 4])
            _ax.set_xlim([0.95, 1.1])
            _ax.axes.get_xaxis().set_visible(False)
            _ax.set_title(r't = %4.3f' % np.asarray(t).mean())

        ax3.set_ylabel(r'n$_e/$n$_e(\rho=1)$')
        ax4.axes.get_yaxis().set_visible(False)
        ax5.axes.get_yaxis().set_visible(False)
        ax6.axes.get_yaxis().set_visible(False)

        for _ax in axLamb:
            _ax.set_xlim([0.95, 1.1])
            _ax.set_ylim([1e-1, 20])
            _ax.set_yscale('log')
            _ax.axhline(1, ls='--', color='gray', lw=2)
            _ax.set_xlabel(r'$\rho$')
            _ax.set_xticks([0.96, 1, 1.04, 1.08])

        ax7.set_ylabel(r'$\Lambda_{div}$')
        ax8.axes.get_yaxis().set_visible(False)
        ax9.axes.get_yaxis().set_visible(False)
        ax10.axes.get_yaxis().set_visible(False)

        mpl.pylab.savefig('../pdfbox/Shot_%5i' % shotList[0]
                          + '_'+'%5i' % shotList[1] +
                          '_InterELMprofiles4Panel.pdf',
                          bbox_to_inches='tight')

    elif selection == 48:
        shotList = (34276, 34278)
        fig = mpl.pylab.figure(figsize=(16, 14))
        fig.subplots_adjust(hspace=0.3, right=0.96, top=0.96)
        ax1 = mpl.pylab.subplot2grid((3, 3), (0, 0), colspan=3)
        ax2 = mpl.pylab.subplot2grid((3, 3), (2, 0), colspan=3)
        ax3 = mpl.pylab.subplot2grid((3, 3), (1, 0))
        ax4 = mpl.pylab.subplot2grid((3, 3), (1, 1))
        ax5 = mpl.pylab.subplot2grid((3, 3), (1, 2))

        axCas = (ax3, ax4, ax5)
        colorLS = ('#C90015', '#7B0Ce7')
        Cryo = ('No', 'On')
        df = pd.read_csv('../data/MEM_Topic21.csv')
        for shot, col, Cry in zip(shotList, colorLS, Cryo):
            Gas = neutrals.Neutrals(shot)
            ax1.plot(Gas.gas['D2']['t'], Gas.gas['D2']['data']/1e21,
                     color=col, lw=3, label='# %5i' % shot + ' Cryo ' + Cry)
            Ddf = df[df['shot'] == shot]
            for _s, _ax in zip(
                    np.linspace(1, 3, 3, dtype='int'),
                    axCas):
                Data = xray.open_dataarray(
                    '../data/Shot%5i' % shot + '_%1i' % _s + 'Stroke.nc')
                _ax.plot(Data.t*1e6, Data.sel(sig='Isat_m06'),
                         color=col, lw=3, label='# %5i' % shot)
                err = Data.err.reshape(3, 501)
                _ax.fill_between(Data.t*1e6, Data.sel(sig='Isat_m06')-err[0, :],
                                 Data.sel(sig='Isat_m06')+err[0, :], edgecolor='white',
                                 facecolor=col,
                                 alpha=0.5)
                Size = Data.TauB*np.sqrt(
                    np.power(4e-3/Data.Lags[0], 2) +
                    np.power(6.9e-3/Data.Lags[1], 2))
                SizeErr = Size * np.abs(Data.TauBErr/Data.TauB)

                ax1.axvline((Ddf['tmin%1i' % _s].values[0] + Ddf['tmax%1i' % _s].values[0])/2,
                            color=col, ls='-', lw=2)

                ax2.semilogx(Data.Lambda, Size*1e3, 'o', ms=15, color=col)
                ax2.errorbar(Data.Lambda, Size*1e3, yerr=SizeErr*1e3, fmt='none',
                             ecolor=col, capsize=0)
            # for the shot 34278 where we have additional availabe stroke
            # we also add the rest
            if shot == 34278:
                for _s in np.linspace(4, 6, 3, dtype='int'):
                    Data = xray.open_dataarray(
                        '../data/Shot%5i' % shot + '_%1i' % _s + 'Stroke.nc')
                    Size = Data.TauB*np.sqrt(
                        np.power(4e-3/Data.Lags[0], 2) +
                        np.power(6.9e-3/Data.Lags[1], 2))
                    SizeErr = Size * np.abs(Data.TauBErr/Data.TauB)
                    ax1.axvline((Ddf['tmin%1i' % _s].values[0] + Ddf['tmax%1i' % _s].values[0])/2,
                                color=col, ls='--', lw=2)
                    ax2.semilogx(Data.Lambda, Size*1e3, 'p', ms=15, color=col)
                    ax2.errorbar(Data.Lambda, Size*1e3, yerr=SizeErr*1e4, fmt='none',
                                 ecolor=col, capsize=0)

            else:
                pass

        ax1.set_xlim([0, 6.5])
        ax1.set_ylabel(r'D$_2$  [10$^{21}$]')
        ax1.legend(loc='best', numpoints=1, frameon=False)
        ax1.set_xlabel(r't [s]')
        for _ax in axCas:
            _ax.set_xlim([-120, 120])
            _ax.set_xlabel(r' t[$\mu$s]')
            _ax.set_ylim([-1, 3.5])
        ax3.axes.get_yaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)

        ax2.set_ylim([0.1, 100])
        ax2.set_xlabel(r'$\Lambda_{div}$ @ $\rho=1.03$')
        ax2.set_ylabel(r'$\delta_b$ [mm]')
        mpl.pylab.savefig('../pdfbox/CompareCasSizeShot%5i' % shotList[0]
                          + '_%5i' % shotList[1]+'.pdf', bbox_to_inches='tight')

    elif selection == 49:
        shotCompare = ((34103, 34105), (34106, 34104))
        btList = ((1.9, 2.5), (2.5, 3.1))
        colorList = ('#C90015', '#7B0Ce7')
        ipList = ('0.6', '1')
        fig, ax = mpl.pylab.subplots(
            figsize=(8, 5), nrows=2, ncols=1, sharex=True)
        for _list, _ip, k, _b in zip(shotCompare, ipList,
                                     range(len(colorList)),
                                     btList):
            for shot, col, bt in zip(_list, colorList, _b):
                N = neutrals.Neutrals(shot)
                # load also the edge density
                Tot = dd.shotfile('TOT', shot)
                nnG = Tot('n/nGW').data
                nnGt = Tot('n/nGW').time
                Tot.close()
                #
                S = UnivariateSpline(N.n0Time, N.n0, s=0)
                # determine tmin and tmax as time larger then 0.5 and
                # time lower then  the maximum achieved fueling after 2.5 s
                tmin = N.n0Time.min()
                _idx = np.argmax(N.gas['D2']['data'][
                    ((N.gas['D2']['t'] > 2.5) &
                     (N.gas['D2']['t'] < 5))])
                _i0 = np.argmin(np.abs(N.gas['D2']['t']-2.5))
                tmax = N.gas['D2']['t'][_i0+_idx]
                _idx = np.where((nnGt >= tmin) &
                                (nnGt <= tmax))[0]
                ax[k].semilogy(nnG[_idx], S(nnGt[_idx]),
                               color=col, label=r'# %5i' % shot +
                               r' B$_t$ = %2.1f' % bt)
            ax[k].text(0.1, 0.9, _ip + ' MA', transform=ax[k].transAxes)
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].legend(loc='best', numpoints=1, frameon=False)
        ax[0].set_ylabel(r'n$_0 [10^{19}$m$^{-3}]$')
        ax[1].legend(loc='best', numpoints=1, frameon=False)
        ax[1].set_ylabel(r'n$_0 [10^{19}$m$^{-3}]$')
        ax[1].set_xlabel(r'n/n$_G$')
        ax[1].set_xlim([0.1, 0.9])
        fig.savefig('../pdfbox/NeutralsVsGreenwaldSameCurrentDifferentBt.pdf',
                    bbox_to_inches='tight')
    elif selection == 50:
        # create a figure with neutral density vs edge density and vs normalized
        # greenwald fraction comparing
        # Constant Q95
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        fig, ax = mpl.pylab.subplots(
            figsize=(10, 10), nrows=2, ncols=1, sharex=True)
        fig.subplots_adjust(bottom=0.15)
        fig2, ax2 = mpl.pylab.subplots(figsize=(10, 5), nrows=1, ncols=1)
        fig2.subplots_adjust(bottom=0.15)
        for shot, _ip, col in zip(shotList, currentL, colorL):
            N = neutrals.Neutrals(shot)
            # load also the edge density
            Tot = dd.shotfile('TOT', shot)
            nnG = Tot('n/nGW').data
            nnGt = Tot('n/nGW').time
            Tot.close()
            #
            S = UnivariateSpline(N.n0Time, N.n0, s=0)
            # determine tmin and tmax as time larger then 0.5 and
            # time lower then  the maximum achieved fueling after 2.5 s
            tmin = N.n0Time.min()
            _idx = np.argmax(N.gas['D2']['data'][
                ((N.gas['D2']['t'] > 2.5) &
                 (N.gas['D2']['t'] < 5))])
            _i0 = np.argmin(np.abs(N.gas['D2']['t']-2.5))
            tmax = N.gas['D2']['t'][_i0+_idx]
            _idx = np.where((nnGt >= tmin) &
                            (nnGt <= tmax))[0]
            ax[0].semilogy(nnG[_idx], S(nnGt[_idx]), '.', ms=5,
                           color=col, label=r'# %5i' % shot +
                           r' I$_p$ = %2.1f' % _ip + ' MA')
            ax2.semilogy(nnG[_idx], S(nnGt[_idx]), '.', ms=5,
                         color=col, label=r'# %5i' % shot +
                         r' I$_p$ = %2.1f' % _ip + ' MA')

        ax[0].set_title(r'Constant q$_{95}$')
        ax2.set_title(r'Constant q$_{95}$')
        ax2.set_xlabel(r'n/n$_G$')
        ax2.set_ylabel(r'n$_0[$m$^{-3}]$')
        ax2.set_xlim([0.2, 0.7])
        ax2.set_ylim([5e17, 3e23])
        leg = ax2.legend(loc='best', numpoints=1, frameon=False)
        for t, c in zip(leg.get_texts(), colorL):
            t.set_color(c)
        fig2.savefig('../pdfbox/NeutralsVsGreenwaldConstantQ95.pdf',
                     bbox_to_inches='tight')
        # constant bt
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        fig2, ax2 = mpl.pylab.subplots(figsize=(10, 5), nrows=1, ncols=1)
        fig2.subplots_adjust(bottom=0.15)
        for shot, _ip, col in zip(shotList, currentL, colorL):
            N = neutrals.Neutrals(shot)
            # load also the edge density
            Tot = dd.shotfile('TOT', shot)
            nnG = Tot('n/nGW').data
            nnGt = Tot('n/nGW').time
            Tot.close()
            #
            S = UnivariateSpline(N.n0Time, N.n0, s=0)
            # determine tmin and tmax as time larger then 0.5 and
            # time lower then  the maximum achieved fueling after 2.5 s
            tmin = N.n0Time.min()
            _idx = np.argmax(N.gas['D2']['data'][
                ((N.gas['D2']['t'] > 2.5) &
                 (N.gas['D2']['t'] < 5))])
            _i0 = np.argmin(np.abs(N.gas['D2']['t']-2.5))
            tmax = N.gas['D2']['t'][_i0+_idx]
            _idx = np.where((nnGt >= tmin) &
                            (nnGt <= tmax))[0]
            ax[1].semilogy(nnG[_idx], S(nnGt[_idx]), '.', ms=5,
                           color=col, label=r'# %5i' % shot +
                           r' I$_p$ = %2.1f' % _ip + ' MA')
            ax2.semilogy(nnG[_idx], S(nnGt[_idx]), '.', ms=5,
                         color=col, label=r'# %5i' % shot +
                         r' I$_p$ = %2.1f' % _ip + ' MA')
        ax2.set_title(r'Constant q$_{95}$')
        ax2.set_xlabel(r'n/n$_G$')
        ax2.set_ylabel(r'n$_0[$m$^{-3}]$')
        ax2.set_xlim([0.2, 0.7])
        ax2.set_ylim([5e17, 3e23])
        leg = ax2.legend(loc='best', numpoints=1, frameon=False)
        for t, c in zip(leg.get_texts(), colorL):
            t.set_color(c)
        fig2.savefig('../pdfbox/NeutralsVsGreenwaldConstantBt.pdf',
                     bbox_to_inches='tight')

        ax[1].set_title(r'Constant B$_{t}$')
        ax[0].set_xlim([0.2, 0.7])
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylim([5e17, 3e23])
        ax[0].set_ylabel(r'n$_0[$m$^{-3}]$')
        ax[1].set_xlim([0.2, 0.7])
        ax[1].set_ylim([5e17, 1e23])
        ax[1].set_ylabel(r'n$_0[$m$^{-3}]$')
        ax[1].set_xlabel(r'n/n$_G$')
        leg = ax[1].legend(loc='best', numpoints=1, frameon=False)
        for t, c in zip(leg.get_texts(), colorL):
            t.set_color(c)
        leg = ax[0].legend(loc='best', numpoints=1, frameon=False)
        for t, c in zip(leg.get_texts(), colorL):
            t.set_color(c)
        fig.savefig('../pdfbox/NeutralsVsGreenwaldIpScan.pdf',
                    bbox_to_inches='tight')
    elif selection == 51:
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#01406C', '#F03C07', '#28B799')
        # this is the far and near SOL shoulde amplitude
        # for current scant at consant q95
        fig, ax = mpl.pylab.subplots(figsize=(10, 10),
                                     nrows=2, ncols=1, sharex=True)
        fig.subplots_adjust(bottom=0.15)
        # open a second figure where we will plot also
        # the evolution of lambda_n
        fig2, ax2 = mpl.pylab.subplots(figsize=(10, 10),
                                       nrows=2, ncols=1, sharex=True)
        fig2.subplots_adjust(bottom=0.15, left=0.15)
        for shot, _ip, _idx, col in zip(shotList, currentL,
                                        range(len(shotList)), colorL):
            N = neutrals.Neutrals(shot)
            LiB = libes.Libes(shot)
            dt = 0.1
            LiB.amplitudeShoulder(dt=dt,
                                  reference=[1.2, 1.4], start=N.n0Time.min())
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.02 and 1.02<rho<1.04)
            _idA = np.where((LiB.rhoAmplitude >= 1) &
                            (LiB.rhoAmplitude < 1.03))[0]
            _idB = np.where((LiB.rhoAmplitude >= 1.03) &
                            (LiB.rhoAmplitude < 1.06))[0]
            AmpA = np.nanmean(LiB.Amplitude[:, _idA], axis=1)
            AmpB = np.nanmean(LiB.Amplitude[:, _idB], axis=1)
            AmpAS = np.nanstd(LiB.Amplitude[:, _idA], axis=1)
            AmpBS = np.nanstd(LiB.Amplitude[:, _idB], axis=1)
            # replicate for the Efold length
            EfoldA = np.nanmean(LiB.Efold[:, _idA], axis=1)
            EfoldB = np.nanmean(LiB.Efold[:, _idB], axis=1)
            EfoldAS = np.nanstd(LiB.Efold[:, _idA], axis=1)
            EfoldBS = np.nanstd(LiB.Efold[:, _idB], axis=1)

            # interpolate the neutrals on the same time basis
            S = UnivariateSpline(N.n0Time, N.n0, s=0)
            x, y, e = S(LiB.timeAmplitude[AmpA > 0]
                        ), AmpA[AmpA > 0], AmpAS[AmpA > 0]
            ax[0].errorbar(x[np.argsort(x)],
                           y[np.argsort(x)],
                           yerr=e[np.argsort(x)],
                           fmt='o', ms=8, mec=col, color=col,
                           capsize=0)
            ax[0].text(0.15, 0.85,
                       r'$1\leq \rho< 1.03$',
                       transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude[AmpB > 0]
                        ), AmpB[AmpB > 0], AmpBS[AmpB > 0]
            ax[1].errorbar(x[np.argsort(x)],
                           y[np.argsort(x)],
                           yerr=e[np.argsort(x)],
                           fmt='o', ms=8, mec=col, color=col,
                           capsize=0)
            ax[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                       transform=ax[1].transAxes)

            # same plot but with the efold in cm
            x, y, e = S(LiB.timeAmplitude), EfoldA*1e2, EfoldAS*1e2
            ax2[0].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                        transform=ax2[0].transAxes)
            x, y, e = S(LiB.timeAmplitude), EfoldB*1e2, EfoldBS*1e2
            ax2[1].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                        transform=ax2[1].transAxes)

        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('Amplitude')
        ax[0].set_ylim([0, 0.6])
        ax[0].set_title(r'Constant q$_{95}$')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_ylim([0, 0.6])
        ax[1].set_xlim([1e18, 1e23])
        ax[1].set_xlabel(r'n$_0 [10^{19}$m$^{-3}]$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax[0].text(0.1, 0.7 - _idx * 0.12,
                       r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[0].transAxes, fontsize=14, color=col)
            ax[1].text(0.1, 0.7 - _idx * 0.12,
                       r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[1].transAxes, fontsize=14, color=col)
        fig.savefig('../pdfbox/AmplitudeVsNeutralIpConstantQ95.pdf',
                    bbox_ot_inches='tight')
        # this is for the second of the figure
        ax2[0].set_xscale('log')
        ax2[1].set_xscale('log')
        ax2[0].set_yscale('log')
        ax2[1].set_yscale('log')
        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylabel(r'$\lambda_n [cm]$')
        ax2[0].set_ylim([1, 40])
        ax2[1].set_ylim([1, 40])
        ax2[0].set_title(r'Constant q$_{95}$')
        ax2[1].set_ylabel('$\lambda_n [cm]$')
        ax2[1].set_xlim([1e18, 1e23])
        ax2[1].set_xlabel(r'n$_0 [10^{19}$m$^{-3}]$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax2[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[0].transAxes, fontsize=14, color=col)
            ax2[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[1].transAxes, fontsize=14, color=col)
        fig2.savefig('../pdfbox/EfoldVsNeutralIpConstantQ95.pdf',
                     bbox_ot_inches='tight')

        # constant bt
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#01406C', '#F03C07', '#28B799')

        # this is the far and near SOL shoulde amplitude
        # for current scant at consant q95
        fig, ax = mpl.pylab.subplots(figsize=(10, 10),
                                     nrows=2, ncols=1,
                                     sharex=True)
        fig.subplots_adjust(bottom=0.15)
        fig2, ax2 = mpl.pylab.subplots(figsize=(10, 10),
                                       nrows=2, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(bottom=0.15, left=0.15)
        for shot, _ip, _idx, col in zip(
                shotList, currentL,
                range(len(shotList)), colorL):
            N = neutrals.Neutrals(shot)
            LiB = libes.Libes(shot)
            dt = 0.1
            LiB.amplitudeShoulder(dt=dt, reference=[1.2, 1.4],
                                  start=N.n0Time.min())
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.02 and 1.02<rho<1.04)
            _idA = np.where((LiB.rhoAmplitude >= 1) &
                            (LiB.rhoAmplitude < 1.03))[0]
            _idB = np.where((LiB.rhoAmplitude >= 1.03) &
                            (LiB.rhoAmplitude < 1.06))[0]
            AmpA = np.nanmean(LiB.Amplitude[:, _idA], axis=1)
            AmpB = np.nanmean(LiB.Amplitude[:, _idB], axis=1)
            AmpAS = np.nanstd(LiB.Amplitude[:, _idA], axis=1)
            AmpBS = np.nanstd(LiB.Amplitude[:, _idB], axis=1)
            # replicate for the Efold length
            EfoldA = np.nanmean(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldB = np.nanmean(LiB.Efold[:, _idB], axis=1) * 1e2
            EfoldAS = np.nanstd(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldBS = np.nanstd(LiB.Efold[:, _idB], axis=1) * 1e2
            # interpolate the neutrals on the same time basis
            S = UnivariateSpline(N.n0Time, N.n0, s=0)
            x, y, e = S(LiB.timeAmplitude[AmpA > 0]
                        ),  AmpA[AmpA > 0], AmpAS[AmpA > 0]
            ax[0].errorbar(x[np.argsort(x)],
                           y[np.argsort(x)],
                           yerr=e[np.argsort(x)],
                           fmt='o', ms=8, mec=col, color=col,
                           capsize=0)
            ax[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                       transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude[AmpB > 0]
                        ),  AmpB[AmpB > 0], AmpBS[AmpB > 0]
            ax[1].errorbar(x[np.argsort(x)],
                           y[np.argsort(x)],
                           yerr=e[np.argsort(x)],
                           fmt='o', ms=8, mec=col, color=col,
                           capsize=0)
            ax[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                       transform=ax[1].transAxes)

            # same plot but with the efold in cm
            x, y, e = S(LiB.timeAmplitude), EfoldA, EfoldAS
            ax2[0].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                        transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude), EfoldB, EfoldBS
            ax2[1].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                        transform=ax[1].transAxes)

        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('Amplitude')
        ax[0].set_ylim([0, 0.6])
        ax[0].set_title(r'Constant B$_{t}$')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_ylim([0, 0.6])
        ax[1].set_xlim([1e18, 1e23])
        ax[1].set_xlabel(r'n$_0 [10^{19}$m$^{-3}]$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[0].transAxes, fontsize=14, color=col)
            ax[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[1].transAxes, fontsize=14, color=col)
        fig.savefig('../pdfbox/AmplitudeVsNeutralIpConstantBt.pdf',
                    bbox_ot_inches='tight')

        # this is for the second of the figure
        ax2[0].set_xscale('log')
        ax2[1].set_xscale('log')
        ax2[0].set_yscale('log')
        ax2[1].set_yscale('log')
        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylabel(r'$\lambda_n [cm]$')
        ax2[0].set_title(r'Constant q$_{95}$')
        ax2[1].set_ylabel('$\lambda_n [cm]$')
        ax2[1].set_xlim([1e18, 1e23])
        ax2[1].set_xlabel(r'n$_0 [10^{19}$m$^{-3}]$')
        ax2[1].set_ylim([1, 40])
        ax2[0].set_ylim([1, 40])
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax2[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[0].transAxes, fontsize=14, color=col)
            ax2[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[1].transAxes, fontsize=14, color=col)
        fig2.savefig('../pdfbox/EfoldVsNeutralIpConstantBt.pdf',
                     bbox_ot_inches='tight')

    elif selection == 52:
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#01406C', '#F03C07', '#28B799')
        fig, ax = mpl.pylab.subplots(figsize=(8, 6),
                                     nrows=2, ncols=1,
                                     sharex=True)
        fig.subplots_adjust(bottom=0.15)
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 6),
                                       nrows=2, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(bottom=0.15)
        Directory = '/afs/ipp-garching.mpg.de/home/n/nvianell/analisi/iaea2018/data/aug/'
        for shot, _ip, col in zip(shotList, currentL, colorL):
            # compute the amplitude of the LiB
            LiB = libes.Libes(shot)
            dt = 0.1
            LiB.amplitudeShoulder(dt=dt, reference=[1.2, 1.4], start=1.2)
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.02 and 1.02<rho<1.04)
            _idA = np.where((LiB.rhoAmplitude >= 1) &
                            (LiB.rhoAmplitude < 1.03))[0]
            _idB = np.where((LiB.rhoAmplitude >= 1.03) &
                            (LiB.rhoAmplitude < 1.06))[0]
            AmpA = np.nanmean(LiB.Amplitude[:, _idA], axis=1)
            AmpB = np.nanmean(LiB.Amplitude[:, _idB], axis=1)
            AmpAS = np.nanstd(LiB.Amplitude[:, _idA], axis=1)
            AmpBS = np.nanstd(LiB.Amplitude[:, _idB], axis=1)
            # replicate for the Efold length
            EfoldA = np.nanmean(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldB = np.nanmean(LiB.Efold[:, _idB], axis=1) * 1e2
            EfoldAS = np.nanstd(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldBS = np.nanstd(LiB.Efold[:, _idB], axis=1) * 1e2
            # restore now the value of LambdaDiv from the
            # save data in the IAEA2018 folder
            File = h5py.File(Directory+'Shot%5i' % shot + '.h5')
            _idx = np.where((File['LambdaDivRho'].value >= 1) &
                            (File['LambdaDivRho'].value < 1.02))[0]
            LNear = np.nanmean(File['LambdaDiv'].value[_idx, :],
                               axis=0).ravel()
            _idx = np.where((File['LambdaDivRho'].value >= 1.02) &
                            (File['LambdaDivRho'].value < 1.06))[0]
            LFar = np.nanmean(File['LambdaDiv'].value[_idx, :], axis=0).ravel()
            # Univariate and plot
            S = interp1d(File['LambdaDivTime'].value.ravel(), LNear,
                         kind='linear', fill_value='extrapolate')
            x, y, e = S(LiB.timeAmplitude[AmpA > 0]
                        ),  AmpA[AmpA > 0], AmpAS[AmpA > 0]
            ax[0].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                yerr=e[np.argsort(x)], fmt='o', ms=8,
                mec=col, color=col,
                capsize=0)
            ax[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                       transform=ax[0].transAxes)
            S = interp1d(File['LambdaDivTime'].value.ravel(), LFar,
                         kind='linear', fill_value='extrapolate')
            x, y, e = S(LiB.timeAmplitude[AmpB > 0]
                        ),  AmpB[AmpB > 0], AmpBS[AmpB > 0]
            ax[1].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                yerr=e[np.argsort(x)], fmt='o', ms=8,
                mec=col, color=col,
                capsize=0)
            ax[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                       transform=ax[1].transAxes)
            # ---------
            # now the plot for the efold
            S = interp1d(File['LambdaDivTime'].value.ravel(), LNear,
                         kind='linear', fill_value='extrapolate')
            x, y, e = S(LiB.timeAmplitude),  EfoldA, EfoldAS
            ax2[0].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                yerr=e[np.argsort(x)], fmt='o', ms=8,
                mec=col, color=col,
                capsize=0)
            ax2[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                        transform=ax[0].transAxes)
            S = interp1d(File['LambdaDivTime'].value.ravel(), LFar,
                         kind='linear', fill_value='extrapolate')
            x, y, e = S(LiB.timeAmplitude),  EfoldB, EfoldBS
            ax2[1].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                yerr=e[np.argsort(x)], fmt='o', ms=8,
                mec=col, color=col,
                capsize=0)
            ax2[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                        transform=ax[1].transAxes)
            File.close()
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('Amplitude')
        ax[0].set_ylim([0, 0.6])
        ax[0].set_title(r'Constant q$_{95}$')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_ylim([0, 0.6])
        ax[1].set_xlim([0.05, 25])
        ax[1].set_xlabel(r'$\Lambda_{div}$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax[0].text(0.1, 0.7 - _idx*0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[0].transAxes,
                       fontsize=14, color=col)
            ax[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[1].transAxes,
                       fontsize=14, color=col)
        fig.savefig('../pdfbox/AmplitudeVsLambdaIpConstantQ95.pdf',
                    bbox_ot_inches='tight')
        # second plot
        ax2[0].set_xscale('log')
        ax2[1].set_xscale('log')
        ax2[0].set_yscale('log')
        ax2[1].set_yscale('log')
        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylabel(r'$\lambda_n [cm]$')
        ax2[0].set_title(r'Constant q$_{95}$')
        ax2[1].set_ylabel('$\lambda_n [cm]$')
        ax2[1].set_xlim([0.05, 25])
        ax2[1].set_xlabel(r'$\Lambda_{div}$')
        ax2[1].set_ylim([1, 40])
        ax2[0].set_ylim([1, 40])
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax2[0].text(0.1, 0.7 - _idx*0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[0].transAxes,
                        fontsize=14, color=col)
            ax2[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[1].transAxes,
                        fontsize=14, color=col)
        fig2.savefig('../pdfbox/EfoldVsLambdaIpConstantQ95.pdf',
                     bbox_ot_inches='tight')
        # ----------
        # constant bt
        fig, ax = mpl.pylab.subplots(figsize=(8, 6),
                                     nrows=2, ncols=1, sharex=True)
        fig.subplots_adjust(bottom=0.15)
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        for shot, _ip, col in zip(shotList, currentL, colorL):
            # compute the amplitude of the LiB
            LiB = libes.Libes(shot)
            dt = 0.1
            LiB.amplitudeShoulder(dt=dt, reference=[1.2, 1.4], start=1.2)
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.02 and 1.02<rho<1.04)
            _idA = np.where((LiB.rhoAmplitude >= 1) &
                            (LiB.rhoAmplitude < 1.03))[0]
            _idB = np.where((LiB.rhoAmplitude >= 1.03) &
                            (LiB.rhoAmplitude < 1.06))[0]
            AmpA = np.nanmean(LiB.Amplitude[:, _idA], axis=1)
            AmpB = np.nanmean(LiB.Amplitude[:, _idB], axis=1)
            AmpAS = np.nanstd(LiB.Amplitude[:, _idA], axis=1)
            AmpBS = np.nanstd(LiB.Amplitude[:, _idB], axis=1)
            # replicate for the Efold length
            EfoldA = np.nanmean(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldB = np.nanmean(LiB.Efold[:, _idB], axis=1) * 1e2
            EfoldAS = np.nanstd(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldBS = np.nanstd(LiB.Efold[:, _idB], axis=1) * 1e2
            # restore now the value of LambdaDiv from the
            # save data in the IAEA2018 folder
            File = h5py.File(Directory+'Shot%5i' % shot + '.h5')
            _idx = np.where((File['LambdaDivRho'].value >= 1) &
                            (File['LambdaDivRho'].value < 1.025))[0]
            LNear = np.nanmean(File['LambdaDiv'].value[_idx, :],
                               axis=0).ravel()
            _idx = np.where((File['LambdaDivRho'].value >= 1.025) &
                            (File['LambdaDivRho'].value < 1.06))[0]
            LFar = np.nanmean(File['LambdaDiv'].value[_idx, :], axis=0).ravel()
            # Univariate and plot
            S = interp1d(
                File['LambdaDivTime'].value.ravel(),
                LNear, kind='linear',
                fill_value='extrapolate')
            x, y, e = S(LiB.timeAmplitude[AmpA > 0]
                        ),  AmpA[AmpA > 0], AmpAS[AmpA > 0]
            ax[0].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                e[np.argsort(x)],
                fmt='o', ms=8, color=col, mec=col,
                capsize=0)
            ax[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                       transform=ax[0].transAxes)
            S = interp1d(File['LambdaDivTime'].value.ravel(),
                         LFar, kind='linear',
                         fill_value='extrapolate')
            x, y, e = S(LiB.timeAmplitude[AmpB > 0]
                        ),  AmpB[AmpB > 0], AmpBS[AmpB > 0]
            ax[1].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                e[np.argsort(x)],
                fmt='o', ms=8, color=col, mec=col,
                capsize=0)
            ax[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                       transform=ax[1].transAxes)
            # ---------
            # now the plot for the efold
            S = interp1d(File['LambdaDivTime'].value.ravel(), LNear,
                         kind='linear', fill_value='extrapolate')
            x, y, e = S(LiB.timeAmplitude),  EfoldA, EfoldAS
            ax2[0].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                yerr=e[np.argsort(x)], fmt='o', ms=8,
                mec=col, color=col,
                capsize=0)
            ax2[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                        transform=ax[0].transAxes)
            S = interp1d(File['LambdaDivTime'].value.ravel(), LFar,
                         kind='linear', fill_value='extrapolate')
            x, y, e = S(LiB.timeAmplitude),  EfoldB, EfoldBS
            ax2[1].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                yerr=e[np.argsort(x)], fmt='o', ms=8,
                mec=col, color=col,
                capsize=0)
            ax2[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                        transform=ax[1].transAxes)
            File.close()
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('Amplitude')
        ax[0].set_ylim([0, 0.6])
        ax[0].set_title(r'Constant B$_{t}$')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_ylim([0, 0.6])
        ax[1].set_xlim([0.05, 25])
        ax[1].set_xlabel(r'$\Lambda_{div}$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[0].transAxes, fontsize=14, color=col)
            ax[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[1].transAxes, fontsize=14, color=col)
        fig.savefig('../pdfbox/AmplitudeVsLambdaIpConstantBt.pdf',
                    bbox_ot_inches='tight')
        # second plot
        ax2[0].set_xscale('log')
        ax2[1].set_xscale('log')
        ax2[0].set_yscale('log')
        ax2[1].set_yscale('log')
        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylabel(r'$\lambda_n [cm]$')
        ax2[0].set_title(r'Constant q$_{95}$')
        ax2[1].set_ylabel('$\lambda_n [cm]$')
        ax2[1].set_xlim([0.05, 25])
        ax2[1].set_xlabel(r'$\Lambda_{div}$')
        ax2[0].set_ylim([1, 40])
        ax2[1].set_ylim([1, 40])
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax2[0].text(0.1, 0.7 - _idx*0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[0].transAxes,
                        fontsize=14, color=col)
            ax2[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[1].transAxes,
                        fontsize=14, color=col)
        fig2.savefig('../pdfbox/EfoldVsLambdaIpConstantBt.pdf',
                     bbox_ot_inches='tight')

    elif selection == 53:
        # produce amplitude vs Greenwald fraction
        # for current scan at constant Q95
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#01406C', '#F03C07', '#28B799')
        fig, ax = mpl.pylab.subplots(figsize=(8, 6),
                                     nrows=2, ncols=1,
                                     sharex=True)
        fig.subplots_adjust(bottom=0.15)
        # second plot for the Efold
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 6),
                                       nrows=2, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(bottom=0.15)
        Directory = '/afs/ipp-garching.mpg.de/home/n/nvianell/analisi/iaea2018/data/aug/'
        for shot, _ip, col in zip(shotList, currentL, colorL):
            # compute the amplitude of the LiB
            LiB = libes.Libes(shot)
            dt = 0.1
            LiB.amplitudeShoulder(dt=dt, reference=[1.2, 1.4], start=1.2)
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.02 and 1.02<rho<1.04)
            _idA = np.where((LiB.rhoAmplitude >= 1) &
                            (LiB.rhoAmplitude < 1.03))[0]
            _idB = np.where((LiB.rhoAmplitude >= 1.03) &
                            (LiB.rhoAmplitude < 1.06))[0]
            AmpA = np.nanmean(LiB.Amplitude[:, _idA], axis=1)
            AmpB = np.nanmean(LiB.Amplitude[:, _idB], axis=1)
            AmpAS = np.nanstd(LiB.Amplitude[:, _idA], axis=1)
            AmpBS = np.nanstd(LiB.Amplitude[:, _idB], axis=1)
            # replicate for the Efold length
            EfoldA = np.nanmean(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldB = np.nanmean(LiB.Efold[:, _idB], axis=1) * 1e2
            EfoldAS = np.nanstd(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldBS = np.nanstd(LiB.Efold[:, _idB], axis=1) * 1e2
            # greenwald fraction
            Tot = dd.shotfile('TOT', shot)
            nnG = Tot('n/nGW').data
            nnGt = Tot('n/nGW').time
            Tot.close()
            # Univariate and plot
            S = UnivariateSpline(nnGt, nnG, s=0)
            x, y, e = S(LiB.timeAmplitude[AmpA > 0]
                        ),  AmpA[AmpA > 0], AmpAS[AmpA > 0]
            ax[0].errorbar(x[np.argsort(x)],
                           y[np.argsort(x)],
                           yerr=e[np.argsort(x)],
                           fmt='o', ms=8, mec=col, color=col,
                           capsize=0)
            ax[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                       transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude[AmpB > 0]
                        ),  AmpB[AmpB > 0], AmpBS[AmpB > 0]
            ax[1].errorbar(x[np.argsort(x)],
                           y[np.argsort(x)],
                           yerr=e[np.argsort(x)],
                           fmt='o', ms=8, mec=col, color=col,
                           capsize=0)
            ax[1].text(0.1, 0.85, r'$1.03\leq \rho< 1.06$',
                       transform=ax[1].transAxes)
            # plot for the Efold
            # Univariate and plot
            x, y, e = S(LiB.timeAmplitude), EfoldA, EfoldAS
            ax2[0].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                        transform=ax2[0].transAxes)
            x, y, e = S(LiB.timeAmplitude),  EfoldB, EfoldBS
            ax2[1].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[1].text(0.1, 0.85, r'$1.03\leq \rho< 1.06$',
                        transform=ax2[1].transAxes)

        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('$\lambda_n$ [cm]')
        ax[0].set_yscale('log')
        ax[0].set_title(r'Constant q$_{95}$')
        ax[1].set_ylabel('$\lambda_n$ [cm]')
        ax[1].set_yscale('log')
        ax[1].set_xlim([0.1, 0.7])
        ax[1].set_xlabel(r'n/n$_{G}$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[0].transAxes, fontsize=14, color=col)
            ax[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[1].transAxes, fontsize=14, color=col)
        fig.savefig('../pdfbox/AmplitudeVsGreenwaldIpConstantQ95.pdf',
                    bbox_ot_inches='tight')
        # efold
        ax2[0].set_ylabel('$\lambda_n$ [cm]')
        ax2[0].set_yscale('log')
        ax2[0].set_title(r'Constant q$_{95}$')
        ax2[1].set_ylabel('$\lambda_n$ [cm]')
        ax2[1].set_yscale('log')
        ax2[1].set_xlim([0.1, 0.7])
        ax2[1].set_xlabel(r'n/n$_{G}$')
        ax2[0].set_ylim([1, 40])
        ax2[1].set_ylim([1, 40])
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax2[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[0].transAxes, fontsize=14, color=col)
            ax2[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[1].transAxes, fontsize=14, color=col)
        fig2.savefig('../pdfbox/EfoldVsGreenwaldIpConstantQ95.pdf',
                     bbox_ot_inches='tight')

        # -----------
        # constant bt
        fig, ax = mpl.pylab.subplots(
            figsize=(8, 6),
            nrows=2, ncols=1, sharex=True)
        fig.subplots_adjust(bottom=0.15)
        # this is for the efold
        fig2, ax2 = mpl.pylab.subplots(
            figsize=(8, 6),
            nrows=2, ncols=1, sharex=True)
        fig2.subplots_adjust(bottom=0.15)
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        for shot, _ip, col in zip(shotList, currentL, colorL):
            # compute the amplitude of the LiB
            LiB = libes.Libes(shot)
            dt = 0.1
            LiB.amplitudeShoulder(dt=dt, reference=[1.2, 1.4], start=1.2)
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.02 and 1.02<rho<1.04)
            _idA = np.where((LiB.rhoAmplitude >= 1) &
                            (LiB.rhoAmplitude < 1.03))[0]
            _idB = np.where((LiB.rhoAmplitude >= 1.03) &
                            (LiB.rhoAmplitude < 1.06))[0]
            AmpA = np.nanmean(LiB.Amplitude[:, _idA], axis=1)
            AmpB = np.nanmean(LiB.Amplitude[:, _idB], axis=1)
            AmpAS = np.nanstd(LiB.Amplitude[:, _idA], axis=1)
            AmpBS = np.nanstd(LiB.Amplitude[:, _idB], axis=1)
            # efold
            EfoldA = np.nanmean(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldB = np.nanmean(LiB.Efold[:, _idB], axis=1) * 1e2
            EfoldAS = np.nanstd(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldBS = np.nanstd(LiB.Efold[:, _idB], axis=1) * 1e2
            # greenwald fraction
            Tot = dd.shotfile('TOT', shot)
            nnG = Tot('n/nGW').data
            nnGt = Tot('n/nGW').time
            Tot.close()
            # Univariate and plot
            S = UnivariateSpline(nnGt, nnG, s=0)
            x, y, e = S(LiB.timeAmplitude[AmpA > 0]
                        ),  AmpA[AmpA > 0],  AmpAS[AmpA > 0]
            ax[0].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                yerr=e[np.argsort(x)],
                fmt='o', ms=8,
                color=col, mec=col,
                capsize=0)
            ax[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                       transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude[AmpB > 0]
                        ),  AmpB[AmpB > 0], AmpBS[AmpB > 0]
            ax[1].errorbar(
                x[np.argsort(x)],
                y[np.argsort(x)],
                yerr=e[np.argsort(x)],
                fmt='o', ms=8, color=col,
                capsize=0, mec=col)
            ax[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                       transform=ax[1].transAxes)
            # plot for the Efold
            # Univariate and plot
            x, y, e = S(LiB.timeAmplitude), EfoldA, EfoldAS
            ax2[0].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                        transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude),  EfoldB, EfoldBS
            ax2[1].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[1].text(0.1, 0.85, r'$1.03\leq \rho< 1.06$',
                        transform=ax[1].transAxes)

        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('Amplitude')
        ax[0].set_ylim([0, 0.6])
        ax[0].set_title(r'Constant B$_{t}$')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_ylim([0, 0.6])
        ax[1].set_xlim([0.1, 0.7])
        ax[1].set_xlabel(r'n/n$_G$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[0].transAxes,
                       fontsize=14, color=col)
            ax[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[1].transAxes,
                       fontsize=14, color=col)
        fig.savefig('../pdfbox/AmplitudeVsGreenwaldIpConstantBt.pdf',
                    bbox_ot_inches='tight')

        # efold
        ax2[0].set_ylabel('$\lambda_n$ [cm]')
        ax2[0].set_yscale('log')
        ax2[0].set_title(r'Constant q$_{95}$')
        ax2[1].set_ylabel('$\lambda_n$ [cm]')
        ax2[1].set_yscale('log')
        ax2[1].set_xlim([0.1, 0.7])
        ax2[1].set_xlabel(r'n/n$_{G}$')
        ax2[0].set_ylim([1, 40])
        ax2[1].set_ylim([1, 40])
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax2[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[0].transAxes, fontsize=14, color=col)
            ax2[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[1].transAxes, fontsize=14, color=col)
        fig2.savefig('../pdfbox/EfoldVsGreenwaldIpConstantBt.pdf',
                     bbox_ot_inches='tight')

    elif selection == 54:
        # produce amplitude vs Greenwald fraction
        # for current scan at constant Q95
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#01406C', '#F03C07', '#28B799')
        fig, ax = mpl.pylab.subplots(figsize=(8, 6),
                                     nrows=2, ncols=1,
                                     sharex=True)
        fig.subplots_adjust(bottom=0.15)
        # second plot for the Efold
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 6),
                                       nrows=2, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(bottom=0.15)
        Directory = '/afs/ipp-garching.mpg.de/home/n/nvianell/analisi/iaea2018/data/aug/'
        for shot, _ip, col in zip(shotList, currentL, colorL):
            # compute the amplitude of the LiB
            LiB = libes.Libes(shot)
            dt = 0.1
            LiB.amplitudeShoulder(dt=dt, reference=[1.2, 1.4], start=1.2)
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.02 and 1.02<rho<1.04)
            _idA = np.where((LiB.rhoAmplitude >= 1) &
                            (LiB.rhoAmplitude < 1.03))[0]
            _idB = np.where((LiB.rhoAmplitude >= 1.03) &
                            (LiB.rhoAmplitude < 1.06))[0]
            AmpA = np.nanmean(LiB.Amplitude[:, _idA], axis=1)
            AmpB = np.nanmean(LiB.Amplitude[:, _idB], axis=1)
            AmpAS = np.nanstd(LiB.Amplitude[:, _idA], axis=1)
            AmpBS = np.nanstd(LiB.Amplitude[:, _idB], axis=1)
            # efold
            EfoldA = np.nanmean(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldB = np.nanmean(LiB.Efold[:, _idB], axis=1) * 1e2
            EfoldAS = np.nanstd(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldBS = np.nanstd(LiB.Efold[:, _idB], axis=1) * 1e2
            # edge density
            Tot = dd.shotfile('DCN', shot)
            nnG = Tot('H-5').data/1e19
            nnGt = Tot('H-5').time
            Tot.close()
            # Univariate and plot
            S = UnivariateSpline(nnGt, nnG, s=0)
            x, y, e = S(LiB.timeAmplitude[AmpA > 0]
                        ),  AmpA[AmpA > 0], AmpAS[AmpA > 0]
            ax[0].errorbar(x[np.argsort(x)],
                           y[np.argsort(x)],
                           yerr=e[np.argsort(x)],
                           fmt='o', ms=8, mec=col, color=col,
                           capsize=0)
            ax[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                       transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude[AmpB > 0]
                        ),  AmpB[AmpB > 0], AmpBS[AmpB > 0]
            ax[1].errorbar(x[np.argsort(x)],
                           y[np.argsort(x)],
                           yerr=e[np.argsort(x)],
                           fmt='o', ms=8, mec=col, color=col,
                           capsize=0)
            ax[1].text(0.1, 0.85, r'$1.03\leq \rho< 1.06$',
                       transform=ax[1].transAxes)
            # plot for the Efold
            # Univariate and plot
            x, y, e = S(LiB.timeAmplitude), EfoldA, EfoldAS
            ax2[0].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                        transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude),  EfoldB, EfoldBS
            ax2[1].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[1].text(0.1, 0.85, r'$1.03\leq \rho< 1.06$',
                        transform=ax[1].transAxes)

        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('Amplitude')
        ax[0].set_ylim([0, 0.6])
        ax[0].set_title(r'Constant q$_{95}$')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_ylim([0, 0.6])
        ax[1].set_xlim([0., 4])
        ax[1].set_xlabel(r'n$_{e} H-5 [10^{19}$m$^{-3}]$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax[0].text(0.1, 0.7-_idx*0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[0].transAxes, fontsize=14, color=col)
            ax[1].text(0.1, 0.7-_idx*0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[1].transAxes, fontsize=14, color=col)
        fig.savefig('../pdfbox/AmplitudeVsNe5IpConstantQ95.pdf',
                    bbox_ot_inches='tight')
        # Efold
        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylabel('$\lambda_n$ [cm]')
        ax2[0].set_yscale('log')
        ax2[0].set_title(r'Constant q$_{95}$')
        ax2[1].set_ylabel('$\lambda_n$ [cm]')
        ax2[1].set_yscale('log')
        ax2[1].set_xlim([0., 4])
        ax2[1].set_xlabel(r'n$_{e} H-5 [10^{19}$m$^{-3}]$')
        ax2[0].set_ylim([1, 40])
        ax2[1].set_ylim([1, 40])
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax2[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[0].transAxes, fontsize=14, color=col)
            ax2[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[1].transAxes, fontsize=14, color=col)
        fig2.savefig('../pdfbox/EfoldVsNe5IpConstantQ95.pdf',
                     bbox_ot_inches='tight')
        # -----------
        # constant bt
        fig, ax = mpl.pylab.subplots(figsize=(8, 6),
                                     nrows=2, ncols=1,
                                     sharex=True)
        fig.subplots_adjust(bottom=0.15)
        # second plot for the Efold
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 6),
                                       nrows=2, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(bottom=0.15)
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        for shot, _ip, col in zip(shotList, currentL, colorL):
            # compute the amplitude of the LiB
            LiB = libes.Libes(shot)
            dt = 0.1
            LiB.amplitudeShoulder(dt=dt, reference=[1.2, 1.4], start=1.2)
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.02 and 1.02<rho<1.04)
            _idA = np.where((LiB.rhoAmplitude >= 1) &
                            (LiB.rhoAmplitude < 1.03))[0]
            _idB = np.where((LiB.rhoAmplitude >= 1.03) &
                            (LiB.rhoAmplitude < 1.06))[0]
            AmpA = np.nanmean(LiB.Amplitude[:, _idA], axis=1)
            AmpB = np.nanmean(LiB.Amplitude[:, _idB], axis=1)
            AmpAS = np.nanstd(LiB.Amplitude[:, _idA], axis=1)
            AmpBS = np.nanstd(LiB.Amplitude[:, _idB], axis=1)
            # efold
            EfoldA = np.nanmean(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldB = np.nanmean(LiB.Efold[:, _idB], axis=1) * 1e2
            EfoldAS = np.nanstd(LiB.Efold[:, _idA], axis=1) * 1e2
            EfoldBS = np.nanstd(LiB.Efold[:, _idB], axis=1) * 1e2
            Tot = dd.shotfile('DCN', shot)
            nnG = Tot('H-5').data/1e19
            nnGt = Tot('H-5').time
            Tot.close()
            # Univariate and plot
            S = UnivariateSpline(nnGt, nnG, s=0)
            x, y, e = S(LiB.timeAmplitude[AmpA > 0]
                        ),  AmpA[AmpA > 0],  AmpAS[AmpA > 0]
            ax[0].errorbar(x[np.argsort(x)], y[np.argsort(x)],
                           yerr=e[np.argsort(x)], fmt='o', ms=8,
                           color=col, mec=col,
                           capsize=0)
            ax[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                       transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude[AmpB > 0]
                        ),  AmpB[AmpB > 0], AmpBS[AmpB > 0]
            ax[1].errorbar(x[np.argsort(x)], y[np.argsort(x)],
                           yerr=e[np.argsort(x)], fmt='o', ms=8, color=col,
                           capsize=0, mec=col)
            ax[1].text(0.15, 0.85, r'$1.03\leq \rho< 1.06$',
                       transform=ax[1].transAxes)
            # plot for the Efold
            # Univariate and plot
            x, y, e = S(LiB.timeAmplitude), EfoldA, EfoldAS
            ax2[0].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[0].text(0.15, 0.85, r'$1\leq \rho< 1.03$',
                        transform=ax[0].transAxes)
            x, y, e = S(LiB.timeAmplitude),  EfoldB, EfoldBS
            ax2[1].errorbar(x[np.argsort(x)],
                            y[np.argsort(x)],
                            yerr=e[np.argsort(x)],
                            fmt='o', ms=8, mec=col, color=col,
                            capsize=0)
            ax2[1].text(0.1, 0.85, r'$1.03\leq \rho< 1.06$',
                        transform=ax[1].transAxes)

        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('Amplitude')
        ax[0].set_ylim([0, 0.6])
        ax[0].set_title(r'Constant B$_{t}$')
        ax[1].set_ylabel('Amplitude')
        ax[1].set_ylim([0, 0.6])
        ax[1].set_xlim([0., 4])
        ax[1].set_xlabel(r'n$_{e} H-5 [10^{19}$m$^{-3}]$')
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax[0].text(0.1, 0.7-_idx*0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[0].transAxes,
                       fontsize=14, color=col)
            ax[1].text(0.1, 0.7-_idx*0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                       transform=ax[1].transAxes,
                       fontsize=14, color=col)

        fig.savefig('../pdfbox/AmplitudeVsNe5IpConstantBt.pdf',
                    bbox_ot_inches='tight')

        # Efold
        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylabel('$\lambda_n$ [cm]')
        ax2[0].set_yscale('log')
        ax2[0].set_title(r'Constant q$_{95}$')
        ax2[1].set_ylabel('$\lambda_n$ [cm]')
        ax2[1].set_yscale('log')
        ax2[1].set_xlim([0., 4])
        ax2[0].set_ylim([1, 40])
        ax2[1].set_ylim([1, 40])
        ax2[1].set_xlabel(r'n$_{e} H-5 [10^{19}$m$^{-3}]$')
        ax2[0].set_ylim([1, 40])
        ax2[1].set_ylim([1, 40])
        for _ip, col, _idx in zip(currentL, colorL, range(len(currentL))):
            ax2[0].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[0].transAxes, fontsize=14, color=col)
            ax2[1].text(0.1, 0.7 - _idx * 0.12, r'I$_p$ = %2.1f' % _ip + ' MA',
                        transform=ax2[1].transAxes, fontsize=14, color=col)
        fig2.savefig('../pdfbox/EfoldVsNe5IpConstantBt.pdf',
                     bbox_ot_inches='tight')

    elif selection == 55:
        # create a figure with neutral density vs edge density and vs normalized
        # greenwald fraction comparing
        # Constant Q95
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#01406C', '#F03C07', '#28B799')
        fig2, ax2 = mpl.pylab.subplots(figsize=(10, 5), nrows=1, ncols=1)
        fig2.subplots_adjust(bottom=0.15)
        for shot, _ip, col in zip(shotList, currentL, colorL):
            N = neutrals.Neutrals(shot)
            # load also the edge density
            Tot = dd.shotfile('DCN', shot)
            nnG = Tot('H-5').data/1e19
            nnGt = Tot('H-5').time
            Tot.close()
            #
            S = UnivariateSpline(nnGt, nnG, s=0)
            # determine tmin and tmax as time larger then 0.5 and
            # time lower then  the maximum achieved fueling after 2.5 s
            tmin = N.n0Time.min()
            _idx = np.argmax(N.gas['D2']['data'][
                ((N.gas['D2']['t'] > 2.5) &
                 (N.gas['D2']['t'] < 5))])
            _i0 = np.argmin(np.abs(N.gas['D2']['t']-2.5))
            tmax = N.gas['D2']['t'][_i0+_idx]
            _idx = np.where((N.n0Time >= tmin) &
                            (N.n0Time <= tmax))[0]
            x, y = S(N.n0Time[_idx]), N.n0[_idx]
            ax2.semilogy(x[np.argsort(x)], y[np.argsort(x)], '.', ms=5,
                         color=col, label=r'# %5i' % shot +
                         r' I$_p$ = %2.1f' % _ip + ' MA')

        ax2.set_title(r'Constant q$_{95}$')
        ax2.set_xlabel(r'n$_{e} H-5 [10^{19}$m$^{-3}]$')
        ax2.set_ylabel(r'n$_0[$m$^{-3}]$')
        ax2.set_xlim([0, 4])
        ax2.set_ylim([5e17, 1e23])
        leg = ax2.legend(loc='best', numpoints=1,
                         frameon=False, fontsize=14)
        for t, c in zip(leg.get_texts(), colorL):
            t.set_color(c)
        fig2.savefig('../pdfbox/NeutralsVsNe5IpConstantQ95.pdf',
                     bbox_to_inches='tight')
        # constant bt
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        fig2, ax2 = mpl.pylab.subplots(figsize=(10, 5), nrows=1, ncols=1)
        fig2.subplots_adjust(bottom=0.15)
        for shot, _ip, col in zip(shotList, currentL, colorL):
            N = neutrals.Neutrals(shot)
            # load also the edge density
            Tot = dd.shotfile('DCN', shot)
            nnG = Tot('H-5').data/1e19
            nnGt = Tot('H-5').time
            Tot.close()
            #
            S = UnivariateSpline(nnGt, nnG, s=0)
            # determine tmin and tmax as time larger then 0.5 and
            # time lower then  the maximum achieved fueling after 2.5 s
            tmin = N.n0Time.min()
            _idx = np.argmax(N.gas['D2']['data'][
                ((N.gas['D2']['t'] > 2.5) &
                 (N.gas['D2']['t'] < 5))])
            _i0 = np.argmin(np.abs(N.gas['D2']['t']-2.5))
            tmax = N.gas['D2']['t'][_i0+_idx]
            _idx = np.where((N.n0Time >= tmin) &
                            (N.n0Time <= tmax))[0]
            x, y = S(N.n0Time[_idx]), N.n0[_idx]
            ax2.semilogy(x[np.argsort(x)], y[np.argsort(x)], '.', ms=5,
                         color=col, label=r'# %5i' % shot +
                         r' I$_p$ = %2.1f' % _ip + ' MA')

        ax2.set_title(r'Constant B$_{t}$')
        ax2.set_xlabel(r'n$_{e} H-5 [10^{19}$m$^{-3}]$')
        ax2.set_ylabel(r'n$_0[$m$^{-3}]$')
        ax2.set_xlim([0., 4])
        ax2.set_ylim([5e17, 5e22])
        leg = ax2.legend(loc='best', numpoints=1, frameon=False)
        for t, c in zip(leg.get_texts(), colorL):
            t.set_color(c)
        fig2.savefig('../pdfbox/NeutralsVsNe5IpConstantBt.pdf',
                     bbox_to_inches='tight')

    elif selection == 56:
        shotList = (34276, 34278, 34281)
        colorList = ('#324D5C', '#E37B40', '#60A65F')
        fig, ax = mpl.pylab.subplots(
            figsize=(10, 14), nrows=5, ncols=1, sharex=True)
        fig.subplots_adjust(hspace=0.05, top=0.96, bottom=0.1)
        for shot, col in zip(shotList, colorList):
            diag = dd.shotfile('DCN', shot)('H-5')
            ax[0].plot(diag.time, diag.data/1e19, color=col,
                       lw=3, label='# %5i' % shot)
            diag = dd.shotfile('UVS', shot)('D_tot')
            ax[1].plot(diag.time, diag.data/1e21, color=col, lw=3)
            diag = dd.shotfile('UVS', shot)('N_tot')
            ax[2].plot(diag.time, diag.data/1e21, color=col, lw=3)
            diag = dd.shotfile('IOC', shot)('F01')
            ax[3].plot(diag.time, diag.data/1e22, color=col, lw=3)
            diag = dd.shotfile('MAC', shot)('Tdiv')
            ax[4].plot(diag.time, diag.data, color=col, lw=3)

        for i in range(3):
            ax[i].axes.get_xaxis().set_visible(False)
        label = (r'n$_e$ H-5 [10$^{19}$m$^{-3}$]',
                 r'D$_2 [10^{21}]$',
                 r'N$_2 [10^{21}]$',
                 r'F01 [10$^{22}$]',
                 r'T$_{div}$')
        for i, l in enumerate(label):
            ax[i].set_ylabel(l)
        ax[0].set_ylim([0, 8])
        ax[1].set_ylim([0, 45])
        ax[2].set_ylim([0, 35])
        ax[3].set_ylim([0, 45])
        ax[4].set_ylim([-10, 50])
        ax[4].set_xlim([0, 7])
        ax[4].set_xlabel(r't [s]')
        l = ax[0].legend(loc='best', numpoints=1, markerscale=0,
                         fontsize=16, frameon=False)
        for t, c in zip(l.get_texts(), colorList):
            t.set_color(c)
        fig.savefig('../pdfbox/GeneralComparisonShot%5i' % shotList[0] +
                    '_%5i' % shotList[1] + '_%5i' % shotList[2]+'.pdf',
                    bbox_to_inches='tight')
    elif selection == 57:
        shotList = (34276, 34278, 34281)
        colorList = ('#324D5C', '#E37B40', '#60A65F')
        colorTimeList = ('#FF4F00', '#00A963', '#0080B6')
        fig = mpl.pylab.figure(figsize=(14, 14))
        fig.subplots_adjust(hspace=0.28, right=0.96, top=0.96)
        ax1 = mpl.pylab.subplot2grid((4, 3), (0, 0), colspan=3)
        # these are the panel for upstream profiles
        ax2 = mpl.pylab.subplot2grid((4, 3), (1, 0))
        ax3 = mpl.pylab.subplot2grid((4, 3), (1, 1))
        ax4 = mpl.pylab.subplot2grid((4, 3), (1, 2))
        # these are the panel for target profiles
        ax5 = mpl.pylab.subplot2grid((4, 3), (2, 0))
        ax6 = mpl.pylab.subplot2grid((4, 3), (2, 1))
        ax7 = mpl.pylab.subplot2grid((4, 3), (2, 2))
        # these are the panel for Lambda
        ax8 = mpl.pylab.subplot2grid((4, 3), (3, 0))
        ax9 = mpl.pylab.subplot2grid((4, 3), (3, 1))
        ax10 = mpl.pylab.subplot2grid((4, 3), (3, 2))
        # list of axes
        axProf = (ax2, ax3, ax4)
        axTarg = (ax5, ax6, ax7)
        axLamb = (ax8, ax9, ax10)
        # tranges
        tList = ((3, 3.1), (3.8, 3.9), (5.1, 5.2))
        # now we need to find for each of the time intervals
        # the appropriate threshold for inter-ELM
        thresholdList = ((1000, 500, 250),
                         (1800, 1200, 800),
                         (1000, 500, 250))
        for shot, col, thresh, _axp, _axt, _axl in itertools.izip(
                shotList, colorList, thresholdList, axProf, axTarg, axLamb):
            diag = dd.shotfile('DCN', shot)('H-5')
            ax1.plot(diag.time, diag.data/1e19, color=col,
                     label='# %5i' % shot, lw=3)
            LiB = libes.Libes(shot)
            Target = langmuir.Target(shot)
            for t, _tr, _colorT in zip(
                    tList, thresh, colorTimeList):
                p, e, ef, pN, eN = LiB.averageProfile(
                    trange=[t[0], t[1]],
                    interelm=True, threshold=_tr)
                _axp.semilogy(LiB.rho, pN, color=_colorT, lw=2)
                _axp.errorbar(LiB.rho, pN, yerr=eN, fmt='none',
                              ecolor=_colorT, alpha=0.3)
                rho, en, err = Target.PlotEnProfile(
                    trange=[t[0], t[1]], interelm=True,
                    threshold=_tr, Plot=False)
                _axt.errorbar(rho, en/1e19, yerr=err/1e19/2,
                              fmt='--o', capsize=0, color=_colorT, ms=12,
                              mec=_colorT, alpha=0.5)
                rhoL, Lambda = Target.computeLambda(
                    trange=[t[0], t[1]], interelm=True,
                    threshold=_tr, Plot=False)

                _axl.semilogy(rhoL[rhoL < rho.max()], Lambda[rhoL < rho.max()], '-',
                              color=_colorT, lw=2)

        for _p, _t, _s in zip(axProf, axTarg, shotList):
            _p.set_xlim([0.98, 1.06])
            _t.set_xlim([0.98, 1.06])
            _p.set_ylim([1e-1, 3])
            _t.set_ylim([0, 10])
            _p.axes.get_xaxis().set_visible(False)
            _t.axes.get_xaxis().set_visible(False)
            _p.text(0.1, 0.86, r'#%5i' % _s, transform=_p.transAxes)
            _t.text(0.1, 0.86, r'#%5i' % _s, transform=_t.transAxes)

        for i in np.linspace(1, 2, 2, dtype='int'):
            axProf[i].axes.get_yaxis().set_visible(False)
            axTarg[i].axes.get_yaxis().set_visible(False)
            axLamb[i].axes.get_yaxis().set_visible(False)

        label = (r'n$_e$/n$_e(\rho_p = 1)$',
                 r'n$_e[10^{19}$m$^{-3}]$',
                 r'$\Lambda_{div}$')
        for l, ax in zip(label, (axProf, axTarg, axLamb)):
            ax[0].set_ylabel(l)
        for ax, _s in zip(axLamb, shotList):
            ax.set_ylim([0.08, 30])
            ax.set_xlabel(r'$\rho$')
            ax.set_xlim([0.98, 1.06])
            ax.set_xticks([0.98, 1, 1.02, 1.04])
            ax.axhline(1, ls='--', lw=2, color='gray')
            ax.text(0.1, 0.86, r'#%5i' % _s, transform=ax.transAxes)

        for _t, _c in zip(tList, colorTimeList):
            ax1.axvline((_t[0]+_t[1])/2, ls='-', lw=2, color=_c)

        ax1.set_ylabel(r'n$_e$ H-5 [10$^{19}$]')
        ax1.set_xlabel(r't[s]')
        ax1.set_xlim([0, 7])
        ax1.set_ylim([0, 6])
        leg = ax1.legend(loc='best', numpoints=1, frameon=False)
        for t, c in zip(leg.get_texts(), colorList):
            t.set_color(c)
        fig.savefig('../pdfbox/UpstreamDivertorProfiles%5i' % shotList[0] +
                    '_%5i' % shotList[1] + '_%5i' % shotList[2]+'.pdf',
                    bbox_to_inches='tight')

    elif selection == 58:
        shotList = (34276, 34278, 34281)
        colorList = ('#324D5C', '#E37B40', '#60A65F')
        fig, ax = mpl.pylab.subplots(figsize=(11, 8), nrows=3,
                                     ncols=1, sharex=True)
        for _idx, shot in enumerate(shotList):
            diag = dd.shotfile('UVS', shot)('D_tot')
            ax[_idx].plot(diag.time, diag.data/1e21, 'k')
            ax[_idx].set_ylabel(r'D$_2$ [10$^{21}$s$^{-1}$]')
            ax[_idx].text(0.1, 0.85, '#%5i ' % shot,
                          fontsize=22, transform=ax[_idx].transAxes)
            diag = dd.shotfile('MAC', shot)('Ipolsola')
            T = ax[_idx].twinx()
            T.plot(diag.time, -diag.data/1e3,
                   color='red', rasterized=True)
            T.set_ylabel(r'Ipolsola [kA]', color='red')
            T.set_yticks([0, 10, 20, 30])
            T.set_ylim([-10, 30])
            for t in T.yaxis.get_ticklabels():
                t.set_color('red')
        ax[0].axes.get_xaxis().set_visible(False)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[2].set_xlim([0, 7])
        ax[2].set_xlabel(r't[s]')
        mpl.pylab.savefig('../pdfbox/PuffingIpolsola%5i' % shotList[0] +
                          '_%5i' % shotList[1]+'_%5i' % shotList[2] + '.pdf', bbox_to_inches='tight')

    elif selection == 59:
        # we save in netcdf format the decimated evolution of peak density
        # and XUV diods with corresponding names
        shotList = (34102, 34103, 34104, 34105, 34106)
        Diods = {'D10': {'Name': 'S2L0A09',
                         'xchord': [1.519, 1.606],
                         'ychord': [-1.131, -1.11]},
                 'D17': {'Name': 'S2L1A00',
                         'xchord': [1.437, 2.173],
                         'ychord': [-1.063, -0.282]}}
        tmaxList = (3.87, 3.5, 4.05, 3.21, 4.01)
        for shot, _tmx in zip(shotList, tmaxList):
            # load the greenwald fraction, limit to time greater than 1
            # smooth and decimate
            Tot = dd.shotfile('TOT', shot)
            nGWA = Tot('n/nGW')
            nGW = decimate(bottleneck.move_mean(nGWA.data, window=30), 5, ftype='fir',
                           zero_phase=True)
            tGW = decimate(nGWA.time, 5, ftype='fir', zero_phase=True)
            nGW = nGW[np.where((tGW >= 0.5) & (tGW <= _tmx))[0]]
            tGW = tGW[np.where((tGW >= 0.5) & (tGW <= _tmx))[0]]
            Tot.close()
            # load the target density
            Target = langmuir.Target(shot)
            peakTarget = decimate(bottleneck.move_mean(
                np.nanmax(Target.OuterTargetNe, axis=0)/1e20, window=300),
                10, ftype='fir', zero_phase=True)
            tPeak = decimate(Target.time, 10, ftype='fir', zero_phase=True)
            peakTarget = peakTarget[np.where((tPeak >= tGW.min()) &
                                             (tPeak <= tGW.max()))[0]]
            tPeak = tPeak[np.where((tPeak >= tGW.min()) &
                                   (tPeak <= tGW.max()))[0]]
            # now spline the peak target
            S = UnivariateSpline(
                tPeak[~np.isnan(peakTarget)], peakTarget[~np.isnan(peakTarget)], s=0)

            savedArray = np.vstack((nGW, S(tGW)))
            coords = ['nGw', 'neTarget']
            Dcn = dd.shotfile('DCN', shot)
            dummy = Dcn('H-5')
            H5 = dummy.data[np.where((dummy.time >= tGW.min()) &
                                     (dummy.time <= tGW.max()))[0]]
            tH5 = dummy.time[np.where((dummy.time >= tGW.min()) &
                                      (dummy.time <= tGW.max()))[0]]
            Dcn.close()
            S = UnivariateSpline(tH5, H5/1e19, s=0)
            savedArray = np.vstack((savedArray, S(tGW)))
            coords.append('H-5')
            # and now cycle through the two diods
            XVS = dd.shotfile('XVS', shot)

            for key in Diods.keys():
                dummy = XVS(Diods[key]['Name'])
                Fs = 500e3
                sig = bw_filter(dummy.data, 200e3, Fs, 'lowpass', order=6)
                # now downsampling the data two times
                sig = decimate(decimate(sig, 10, ftype='fir', zero_phase=True),
                               10, ftype='fir', zero_phase=True)
                t = decimate(decimate(dummy.time, 10, ftype='fir', zero_phase=True),
                             10, ftype='fir', zero_phase=True)
                sig = sig[np.where((t >= tGW.min()) & (t <= tGW.max()))[0]]
                t = t[np.where((t >= tGW.min()) & (t <= tGW.max()))[0]]
                S = UnivariateSpline(
                    t[~np.isnan(sig)], sig[~np.isnan(sig)], s=0)
                savedArray = np.vstack((savedArray, S(tGW)))
                coords.append(key)
            df = xray.DataArray(savedArray, coords=[
                                coords, tGW], dims=['sig', 't'])
            df.to_netcdf('../data/Shot%5i' % shot + '_DensityRadiation.nc')
    elif selection == 60:
        shotList = (34102, 34276, 34281)
        colorList = ('#324D5C', '#E37B40', '#60A65F')
        colorTimeList = ('#FF4F00', '#00A963', '#0080B6')
        fig = mpl.pylab.figure(figsize=(14, 14))
        fig.subplots_adjust(hspace=0.28, right=0.96, top=0.96)
        ax1 = mpl.pylab.subplot2grid((4, 3), (0, 0), colspan=3)
        # these are the panel for upstream profiles
        ax2 = mpl.pylab.subplot2grid((4, 3), (1, 0))
        ax3 = mpl.pylab.subplot2grid((4, 3), (1, 1))
        ax4 = mpl.pylab.subplot2grid((4, 3), (1, 2))
        # these are the panel for target profiles
        ax5 = mpl.pylab.subplot2grid((4, 3), (2, 0))
        ax6 = mpl.pylab.subplot2grid((4, 3), (2, 1))
        ax7 = mpl.pylab.subplot2grid((4, 3), (2, 2))
        # these are the panel for Lambda
        ax8 = mpl.pylab.subplot2grid((4, 3), (3, 0))
        ax9 = mpl.pylab.subplot2grid((4, 3), (3, 1))
        ax10 = mpl.pylab.subplot2grid((4, 3), (3, 2))
        # list of axes
        axProf = (ax2, ax3, ax4)
        axTarg = (ax5, ax6, ax7)
        axLamb = (ax8, ax9, ax10)
        # tranges
        tListH = ((3, 3.1), (3.8, 3.9), (5.1, 5.2))
        tListL = ((1.8, 1.9), (3, 3.1), (3.65, 3.75))
        _mode = ('L', 'H', 'H')
        # now we need to find for each of the time intervals
        # the appropriate threshold for inter-ELM
        thresholdList = ((1, 1, 1), 
                         (1000, 500, 250), 
                         (1800, 1200, 800)
                         )
        for shot, col, thresh, _axp, _axt, _axl, _m in itertools.izip(
                shotList, colorList, thresholdList,
                axProf, axTarg, axLamb, _mode):
            diag = dd.shotfile('DCN', shot)('H-5')
            ax1.plot(diag.time, diag.data/1e19, color=col,
                     label='# %5i' % shot, lw=3)
            LiB = libes.Libes(shot)
            Target = langmuir.Target(shot)
            if _m == 'L':
                tList = tListL
            else:
                tList = tListH
            for t, _tr, _colorT in zip(
                    tList, thresh, colorTimeList):
                if _m == 'H':
                    p, e, ef, pN, eN = LiB.averageProfile(
                        trange=[t[0], t[1]],
                        interelm=True, threshold=_tr)
                    _axp.semilogy(LiB.rho, pN, color=_colorT, lw=2)
                    _axp.errorbar(LiB.rho, pN, yerr=eN, fmt='none',
                                  ecolor=_colorT, alpha=0.3)
                    rho, en, err = Target.PlotEnProfile(
                        trange=[t[0], t[1]], interelm=True,
                        threshold=_tr, Plot=False)
                    _axt.errorbar(rho, en/1e19, yerr=err/1e19/2,
                                  fmt='--o', capsize=0, color=_colorT, ms=12,
                                  mec=_colorT, alpha=0.5)
                    rhoL, Lambda = Target.computeLambda(
                        trange=[t[0], t[1]], interelm=True,
                        threshold=_tr, Plot=False)
                    
                    _axl.semilogy(rhoL[rhoL < rho.max()], Lambda[rhoL < rho.max()], '-',
                                  color=_colorT, lw=2)
                else:
                    p, e, ef, pN, eN = LiB.averageProfile(
                        trange=[t[0], t[1]])
                    _axp.semilogy(LiB.rho, pN, color=_colorT, lw=2)
                    _axp.errorbar(LiB.rho, pN, yerr=eN, fmt='none',
                                  ecolor=_colorT, alpha=0.3)
                    rho, en, err = Target.PlotEnProfile(
                        trange=[t[0], t[1]],  Plot=False)
                    _axt.errorbar(rho, en/1e19, yerr=err/1e19/2,
                                  fmt='--o', capsize=0, color=_colorT, ms=12,
                                  mec=_colorT, alpha=0.5)
                    rhoL, Lambda = Target.computeLambda(
                        trange=[t[0], t[1]], Plot=False)
                    
                    _axl.semilogy(rhoL[rhoL < rho.max()], Lambda[rhoL < rho.max()], '-',
                                  color=_colorT, lw=2)


        for _p, _t, _s in zip(axProf, axTarg, shotList):
            _p.set_xlim([0.98, 1.06])
            _t.set_xlim([0.98, 1.06])
            _p.set_ylim([1e-1, 3])
            _t.set_ylim([0, 10])
            _p.axes.get_xaxis().set_visible(False)
            _t.axes.get_xaxis().set_visible(False)
            _p.text(0.1, 0.86, r'#%5i' % _s, transform=_p.transAxes)
            _t.text(0.1, 0.86, r'#%5i' % _s, transform=_t.transAxes)

        for i in np.linspace(1, 2, 2, dtype='int'):
            axProf[i].axes.get_yaxis().set_visible(False)
            axTarg[i].axes.get_yaxis().set_visible(False)
            axLamb[i].axes.get_yaxis().set_visible(False)

        label = (r'n$_e$/n$_e(\rho_p = 1)$',
                 r'n$_e[10^{19}$m$^{-3}]$',
                 r'$\Lambda_{div}$')
        for l, ax in zip(label, (axProf, axTarg, axLamb)):
            ax[0].set_ylabel(l)
        for ax, _s in zip(axLamb, shotList):
            ax.set_ylim([0.01, 30])
            ax.set_xlabel(r'$\rho$')
            ax.set_xlim([0.98, 1.06])
            ax.set_xticks([0.98, 1, 1.02, 1.04])
            ax.axhline(1, ls='--', lw=2, color='gray')
            ax.text(0.1, 0.86, r'#%5i' % _s, transform=ax.transAxes)

        for _m in _mode:
            if _m == 'L':
                tList = tListL
                line = '--'
            else:
                tList = tListH
                line = '-'
            for _t, _c in zip(tList, colorTimeList):
                ax1.axvline((_t[0]+_t[1])/2, ls=line, lw=2, color=_c)

        ax1.set_ylabel(r'n$_e$ H-5 [10$^{19}$]')
        ax1.set_xlabel(r't[s]')
        ax1.set_xlim([0, 7])
        ax1.set_ylim([0, 6])
        leg = ax1.legend(loc='best', numpoints=1, frameon=False)
        for t, c in zip(leg.get_texts(), colorList):
            t.set_color(c)
        fig.savefig('../pdfbox/UpstreamDivertorProfiles%5i' % shotList[0] +
                    '_%5i' % shotList[1] + '_%5i' % shotList[2]+'.pdf',
                    bbox_to_inches='tight')

    elif selection == 99:
        loop = False
    else:
        raw_input("Unknown Option Selected!")

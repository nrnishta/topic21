# script in order to compare heating fueling and equilibria
# for shot at different current/bt/q95 for the proper scan
#from __future__ import print_function
import numpy as np
import sys
import dd
import itertools
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
import pandas as pd
sys.path.append('/afs/ipp/home/n/nvianell/pythonlib/submodules/pycwt/')
sys.path.append('/afs/ipp/home/n/nvianell/analisi/topic21/Codes/python/general/')
sys.path.append('/afs/ipp/home/n/nvianell/analisi/topic21/Codes/python/aug/')
sys.path.append('/afs/ipp/home/n/nvianell/pythonlib/signalprocessing/')
import augFilaments
import neutrals
import peakdetect
from matplotlib.colors import LogNorm
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc("font", size=22)
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Tahoma']})

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
    elif selection == 3:
        shotList = (34103, 34102, 34104)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')        

        fig, ax = mpl.pylab.subplots(figsize=(17, 15),
                                     nrows=3, ncols=2, sharex=True)
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
            ax[1, 0].plot(diag('H-1').time, diag('H-1').data/1e19, color=_col, lw=3)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel(r'$\overline{n}_e$ H-1 [10$^{19}$]')
            ax[1, 0].set_ylim([0, 10])

            ax[2, 0].plot(diag('H-1').time, diag('H-5').data/1e19, color=_col, lw=3)
            ax[2, 0].set_xlabel(r't [s]')
            ax[2, 0].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            diag.close
            ax[2, 0].set_xlim([0, 4.5])
            ax[2, 0].set_ylim([0, 6])

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
            ax[2, 1].set_xlabel(r't [s]')
            ax[2, 1].set_ylabel(r'F01 [10$^{21}$m$^{-2}$s$^{-1}$]')
            diag.close
            ax[2, 1].set_xlim([0, 4.5])
        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        mpl.pylab.savefig('../pdfbox/GeneralIpScanConstantq95.pdf',
                          bbox_to_inches='tight')
            
    elif selection == 4:
        shotList = (34105, 34102, 34106)
        currentL = (0.6, 0.8, 0.99)
        colorL = ('#82A17E', '#1E4682', '#DD6D3D')
        colorLS = ('#C90015', '#7B0Ce7', '#F0CA37')        

        fig, ax = mpl.pylab.subplots(figsize=(17, 15),
                                     nrows=3, ncols=2, sharex=True)
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
            ax[1, 0].plot(diag('H-1').time, diag('H-1').data/1e19, color=_col, lw=3)
            ax[1, 0].axes.get_xaxis().set_visible(False)
            ax[1, 0].set_ylabel(r'$\overline{n}_e$ H-1 [10$^{19}$]')
            ax[1, 0].set_ylim([0, 10])

            ax[2, 0].plot(diag('H-5').time, diag('H-5').data/1e19, color=_col, lw=3)
            ax[2, 0].set_xlabel(r't [s]')
            ax[2, 0].set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
            diag.close
            ax[2, 0].set_xlim([0, 4.5])
            ax[2, 0].set_ylim([0, 6])

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
            ax[2, 1].set_xlabel(r't [s]')
            ax[2, 1].set_ylabel(r'F01 [10$^{21}$m$^{-2}$s$^{-1}$]')
            diag.close
            ax[2, 1].set_xlim([0, 4.5])
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
            ax[2, 0].plot(diag('H-1').time, diag('H-1').data/1e19, color='blue', lw=3)
            ax[2, 0].set_ylabel(r'$\overline{n}_e$ [10$^{19}$]')
            ax[2, 0].set_ylim([0, 10])
            ax[2, 0].plot(diag('H-5').time, diag('H-5').data/1e19, color='orange', lw=3)
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
            diag=dd.shotfile('MAC', shot)
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
            ax[0, 1].plot(diag('H-5').time, diag('H-5').data/1e19, color=_col, lw=3)
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
            diag=dd.shotfile('MAC', shot)
            ax[3, 1].plot(diag('Tdiv').time, diag('Tdiv').data,'--' , color=_col, lw=3,
                          label=r'T$_{div}$')
 
            ax[3, 1].set_xlabel(r't [s]')
            diag.close()

        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].set_ylim([0, 50])
        ax[2, 1].legend(loc='best', numpoints=1, frameon=False)
        mpl.pylab.savefig('../pdfbox/CompareShot33478_34115.pdf',
                          bbox_to_inches='tight')
    elif selection ==7:
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
                     label = r'Shot # %5i' % shot +' I$_p$ = %2.1f' % ip +' MA')
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
            im=_ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto' ,cmap=mpl.cm.viridis,
                          extent=(neLBtime.min(), neLBtime.max(), 0.98, 1.06),
                          norm=LogNorm(vmin=0.5, vmax=1.5))
            _ax.set_xlim([1, 3.9])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' %shot)
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

    elif selection ==8:
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
                     label = r'Shot # %5i' % shot +' I$_p$ = %2.1f' % ip +' MA')
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
            im=_ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto' ,cmap=mpl.cm.viridis,
                          extent=(neLBtime.min(), neLBtime.max(), 0.98, 1.06),
                          norm=LogNorm(vmin=0.5, vmax=1.5))
            _ax.set_xlim([1, 3.9])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' %shot)
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
        

    elif selection ==9:
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
            Probe.blobAnalysis(Probe='Isat_m06', trange=trange, block=[0.015, 0.5])
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
            Probe.blobAnalysis(Probe='Isat_m06', trange=trange, block=[0.015, 0.5])
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
        tmaxL=(6.33, 6.26)
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
            _id=np.where(((Wmhd.time >= 3) & (Wmhd.time <= tm)))[0]
            ax[i].plot(S(Wmhd.time[_id]), Wmhd.data[_id]/1e5)
            ax[i].text(0.7, 0.9, 'Shot # %5i' %shot,
                       transform=ax[i].transAxes)
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel(r'W$_{mhd}[10^5$ J]')
        ax[1].set_ylabel(r'W$_{mhd}[10^5$ J]')
        ax[1].set_xlabel(r'$\overline{n_e}$ H-5 [10$^{19}$m$^{-3}$]')
        ax[1].set_xlim([2, 7])
        mpl.pylab.savefig('../pdfbox/DegradedHMode.pdf', bbox_to_inches='tight')

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
            _id=np.where(((Wmhd.time >= 3) & (Wmhd.time <= tm)))[0]
            ax.plot(S(Wmhd.time[_id]), Wmhd.data[_id]/1e5)
            ax.text(0.6, 0.9, 'Shot # %5i' %shot,
                    transform=ax.transAxes)
            ax.set_ylabel(r'W$_{mhd}[10^5$ J]')
            ax.set_ylabel(r'W$_{mhd}[10^5$ J]')
            ax.set_xlabel(r'$\overline{n_e}$ H-5 [10$^{19}$m$^{-3}$]')
            ax.set_xlim([2, 7])
            mpl.pylab.savefig('../pdfbox/DegradedHModeShot'+str(int(shot))+'.pdf', bbox_to_inches='tight')

    elif selection == 11:
        shotL = (34108, 34115)
        diodsL = ['S2L3A00','S2L3A01','S2L3A02','S2L3A03', 
         'S2L3A12','S2L3A13','S2L3A14','S2L3A15', 
         'S2L3A08','S2L3A09','S2L3A10','S2L3A11', 
         'S2L3A04','S2L3A05','S2L3A06','S2L3A07']
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
            _idx = np.where((t>= 5.5) & (t<= 6.3))[0]
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
            tE = tE[(y>2200)]
            ELMf = 1./np.diff(tE)
            ELMfTime = (tE[1:]+tE[:-1])/2.
            ax.plot(ELMfTime, ELMf, 'o', ms=6, color=_col,
                    alpha=0.5, mec=_col, label=r'Shot # %5i' % shot)
        ax.legend(loc='best', numpoints=1, frameon=True)
        ax.set_xlabel(r't [s]')
        ax.set_ylabel(r'ELM frequency [Hz]')
        mpl.pylab.savefig('../pdfbox/ELMfrequency.pdf',
                          bbox_to_inches='tight')

    elif selection == 99:
        loop = False
    else:
        raw_input("Unknown Option Selected!")
    

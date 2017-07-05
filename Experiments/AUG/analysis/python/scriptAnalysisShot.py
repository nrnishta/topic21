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
    print "13. Check Neutral response to Puffing in H-Mode"
    print "14. Compare Neutral compression with reference 2016"
    print "15. Compare divertor/midplane puffing w/o cryopumps"
    print "16. Compare compression with puffing from divertor/midplane w/o cryompumps"
    print "17. Compare shots same puffing with and wo cryompumps"
    print "18. Compare shots with/wo cryopumps try to match edge density"
    print "19. Compare Li-Beam contour profiles with/wo crypumps when trying to math edge density"
    print "20. Compare Li-Be contour div/midplane puffing"
    print "21. Compare Li-Be contour same fueling with/without cryo"
    print "22. Compare Shot constant q95 same edge density"
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

    elif selection == 13:
        shotL = (34107, 34108, 34115)
        for shot in shotL:
            Gas = neutrals.Neutrals(shot)
            fig, ax = mpl.pylab.subplots(figsize=(6, 12), nrows=4, ncols=1, sharex=True)
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

            ax[3].plot(Gas.signal['F14']['t'], Gas.signal['F01']['data']/Gas.signal['F14']['data'])
            ax[3].set_xlim([0, 7])
            ax[3].text(0.1, 0.8, 'F01/F14 Compression', transform=ax[3].transAxes)
            ax[3].set_ylim([0, 500])
            ax[3].set_xlabel(r't[s]')
            mpl.pylab.savefig('../pdfbox/NeutralCompressionShot' +
                              str(int(shot))+'.pdf', bbox_to_inches='tight')
            
    elif selection == 14:
        shotL = (33478, 34115)
        colorL = ('black', 'red')
        fig, ax = mpl.pylab.subplots(figsize=(6, 12), nrows=4, ncols=1, sharex=True)
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
        pufL = ('Div', 'Mid')
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
            ax[0, 1].plot(diag('H-5').time, diag('H-5').data/1e19, color=_col, lw=3)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            diag.close

            diag = dd.shotfile('TOT', shot)
            ax[1, 1].plot(diag('Wmhd').time, diag('Wmhd').data/1e5, color=_col, lw=3,
                          label=r'# %5i' % shot)
            ax[1, 1].axes.get_xaxis().set_visible(False)
            ax[1, 1].set_ylabel(r'W$_{mhd}$ [10$^5$ J]')
            diag.close()

            try:
                Gas=neutrals.Neutrals(shot)
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
            diag=dd.shotfile('MAC', shot)
            ax[3, 1].plot(diag('Tdiv').time, diag('Tdiv').data,'--' , color=_col, lw=3,
                          label=r'T$_{div}$')
 
            ax[3, 1].set_xlabel(r't [s]')
            diag.close()

        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].set_ylim([-10, 50])
        ax[2, 1].legend(loc='best', numpoints=1, frameon=False)
        mpl.pylab.savefig('../pdfbox/CompareShot'+str(int(shotList[0]))+'_'+
                          str(int(shotList[1]))+'.pdf',
                          bbox_to_inches='tight')
    elif selection == 16:
        shotList = (34276, 34277)
        pufL = ('Div', 'Mid')
        colorList = ('#C90015', '#7B0Ce7')
        fig, ax = mpl.pylab.subplots(figsize=(8, 10), nrows=2, ncols=1, sharex=True)
        fig.subplots_adjust(bottom=0.15)
        for shot, col, p in zip(shotList, colorList, pufL):
            Gas = neutrals.Neutrals(shot)
            ax[0].plot(Gas.signal['F01']['t'],
                    Gas.signal['F01']['data']/Gas.signal['F14']['data'],
                    color=col, lw=2, label=r'Shot # %5i' % shot + 'Puff from '+p)
            Msp = dd.shotfile('MSP', shot)
            Press = 100.*Msp('B25_08Fu').data/(0.0337*np.log10(Msp('B25_08Fu').data*100.)+0.7304)
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
        mpl.pylab.savefig('../pdfbox/CompareCompression'+str(int(shotList[0]))+'_'+
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
            ax[0, 1].plot(diag('H-5').time, diag('H-5').data/1e19, color=_col, lw=3)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            diag.close



            try:
                Gas=neutrals.Neutrals(shot)
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
            Press = 100.*Msp('B25_08Fu').data/(0.0337*np.log10(Msp('B25_08Fu').data*100.)+0.7304)
            ax[3, 1].plot(Msp('B25_08Fu').time, Press, ls='--', color=_col, lw=2, label='B25_08Fu')
            Msp.close()
            try:
                ax[3, 1].plot(Gas.signal['F01']['t'],
                              Gas.signal['F01']['data']/1e22, ls='-', color=_col, lw=3,
                              label=r'F01 [10$^{22}$m$^{-2}$s$^{-1}$]')
            except:
                pass
            diag=dd.shotfile('MAC', shot)
            ax[4, 1].plot(diag('Tdiv').time, diag('Tdiv').data,'-' , color=_col, lw=3,
                          label=r'T$_{div}$')
 
            ax[4, 1].set_xlabel(r't [s]')
            diag.close()

        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[4, 1].set_ylim([-10, 50])
        ax[4, 1].set_ylabel(r'T$_{div}$')
        mpl.pylab.savefig('../pdfbox/CompareShot'+str(int(shotList[0]))+'_'+
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
            ax[0, 1].plot(diag('H-5').time, diag('H-5').data/1e19, color=_col, lw=3)
            ax[0, 1].axes.get_xaxis().set_visible(False)
            diag.close



            try:
                Gas=neutrals.Neutrals(shot)
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
            Press = 100.*Msp('B25_08Fu').data/(0.0337*np.log10(Msp('B25_08Fu').data*100.)+0.7304)
            ax[3, 1].plot(Msp('B25_08Fu').time, Press, ls='--', color=_col, lw=2, label='B25_08Fu')
            Msp.close()
            try:
                ax[3, 1].plot(Gas.signal['F01']['t'],
                              Gas.signal['F01']['data']/1e22, ls='-', color=_col, lw=3,
                              label=r'F01 [10$^{22}$m$^{-2}$s$^{-1}$]')
            except:
                pass
            diag=dd.shotfile('MAC', shot)
            ax[4, 1].plot(diag('Tdiv').time, diag('Tdiv').data,'-' , color=_col, lw=3,
                          label=r'T$_{div}$')
 
            ax[4, 1].set_xlabel(r't [s]')
            diag.close()

        ax[0, 0].legend(loc='best', numpoints=1, frameon=False)
        ax[3, 1].legend(loc='best', numpoints=1, frameon=False)
        ax[4, 1].set_ylim([-10, 50])
        ax[4, 1].set_ylabel(r'T$_{div}$')
        mpl.pylab.savefig('../pdfbox/CompareShot'+str(int(shotList[0]))+'_'+
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
                     label = r'Shot # %5i' % shot +' Cryo ' + _cr)
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
            im=_ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto' ,cmap=mpl.cm.viridis,
                          extent=(neLBtime.min(), neLBtime.max(), 0.98, 1.06),
                          norm=LogNorm(vmin=0.5, vmax=2.5))


            _ax.set_title(r'Shot # % 5i' %shot)

            _ax.set_xlim([1, 6.5])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' %shot)
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
                     label = r'Shot # %5i' % shot +' Cryo ' + _cr)
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
            im=_ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto' ,cmap=mpl.cm.viridis,
                          extent=(neLBtime.min(), neLBtime.max(), 0.98, 1.06),
                          norm=LogNorm(vmin=0.5, vmax=2.5))


            _ax.set_title(r'Shot # % 5i' %shot)

            _ax.set_xlim([1, 6.5])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' %shot)
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

    elif selection == 21:
        shotList = (34276, 34278)
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
                     label = r'Shot # %5i' % shot +' Cryo ' + _cr)
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
            im=_ax.imshow(np.log(profFake.transpose()), origin='lower', aspect='auto' ,
                          cmap=mpl.cm.viridis,
                          extent=(neLBtime.min(), neLBtime.max(), 0.98, 1.06),
                          norm=LogNorm(vmin=0.5, vmax=2.5))


            _ax.set_title(r'Shot # % 5i' %shot)

            _ax.set_xlim([1, 6.5])
            _ax.set_xlabel(r't [s]')
            _ax.set_title(r'Shot # % 5i' %shot)
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
            _idx = np.where((neLBtime >= 1.78) & (neLBtime <=1.82))[0]
            y = np.zeros((_idx.size, 200))
            for n, _iDummy in zip(_idx, range(_idx.size)):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1], s=0)
                y[_iDummy, :] = S(rhoFake)/S(1)
            _idx = np.where((neAvg.time >= 1.78) & (neAvg.time <=1.82))[0]
            _ne = neAvg.data[_idx].mean()/1e20
            _ax.plot(rhoFake, np.mean(y, axis=0), '-', color='k',
                              lw=3, label = r'n$_e$ = %3.2f' % _ne)
            _ax.fill_between(rhoFake, np.mean(y, axis=0)-
                             np.std(y, axis=0), np.mean(y, axis=0)+
                             np.std(y, axis=0),
                             facecolor='grey', edgecolor='none',
                             alpha=0.5)
        
            # repeat and now it will be tmax
            _idx = np.where((neLBtime >= tMax-0.02) & (neLBtime <= tMax+0.02))[0]
            y = np.zeros((_idx.size, 200))
            for n, _iDummy in zip(_idx, range(_idx.size)):
                S = UnivariateSpline(rhoP[n, ::-1],
                                     neLB[n, ::-1], s=0)
                y[_iDummy, :] = S(rhoFake)/S(1)
            _idx = np.where((neAvg.time >= tMax-0.02) & (neAvg.time <= tMax+0.02))[0]
            _ne = neAvg.data[_idx].mean()/1e20
            _ax.plot(rhoFake, np.mean(y, axis=0), '-', color='orange',
                              lw=3, label = r'n$_e$ = %3.2f' % _ne)
            _ax.fill_between(rhoFake, np.mean(y, axis=0)-
                             np.std(y, axis=0), np.mean(y, axis=0)+
                             np.std(y, axis=0),
                             facecolor='orange', edgecolor='none',
                             alpha=0.5)
            _ax.set_yscale('log')
            _ax.set_xlabel(r'$\rho_p$')
            _ax.set_title(r'Shot # % 5i' %shot)
            _ax.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'I$_p$ [MA]')
        ax12.set_ylabel(r'$\overline{n}_e$ H-5 [10$^{20}$]')
        ax13.set_ylabel(r'D$_2 [10^{21}$s$^{-1}]$')
        ax1.set_title(r'I$_p$ scan at constant q$_{95}$')
        ax2.set_ylabel(r'n$_e$/n$_e(\rho_p = 1)$')
        mpl.pylab.savefig('../pdfbox/IpConstantq95_samedensity.pdf',
                          bbox_to_inches='tight')
    elif selection == 99:
        loop = False
    else:
        raw_input("Unknown Option Selected!")
    

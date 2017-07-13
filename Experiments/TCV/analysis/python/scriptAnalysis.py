import numpy as np
import sys
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.signal import find_peaks_cwt
import pandas as pd
import peakdetect
from matplotlib.colors import LogNorm
from boloradiation import Radiation
import eqtools
import MDSplus as mds
import langmuir
from tcv.diag.frp import FastRP
import gauges
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc("font", size=18)
mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Tahoma']})
mpl.rc("lines", linewidth=2)


def print_menu():
    print 30 * "-", "MENU", 30 * "-"
    print "1. General plot with current scan at constant Bt"
    print "2. General plot with current scan at constant q95"
    print "3. Compare profiles scan at constant Bt"
    print "4. Compare profiles scan at constant q95"
    print "5. Compare profiles scan at constat Bt only SOL"
    print "6. Compare profiles scan at constat q95 only SOL"
    print "7. Attempt for low collisionality"
    print "8. Compare plot and profiles at constant q95"
    print "99: End"
    print 67 * "-"


loop = True

while loop:
    print_menu()
    selection = input("Enter your choice [1-99] ")
    if selection == 1:
        shotList = (57425, 57437, 57497)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c')
        # create the plot
        fig, ax = mpl.pylab.subplots(figsize=(16, 12), nrows=4,
                                     ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.2, top=0.98, left=0.1, right=0.98)
        for shot, col in zip(shotList, colorList):
            eq = eqtools.TCVLIUQETree(shot)
            Tree = mds.Tree('tcv_shot', shot)
            iP = mds.Data.compile(r'tcv_ip()').evaluate()
            enAVG = Tree.getNode(r'\results::fir:n_average')
            # load the vloop
            Vloop = Tree.getNode(r'\magnetics::vloop')
            # now load the bolometry radiation
            Bolo = Radiation(shot)
            POhm = Tree.getNode(r'\results::conf:ptot_ohm')
            tagPohm = False
            if POhm.data().mean() == 0:
                POhm = np.abs(iP.data() * Vloop.data())
                tagPohm = True
            # load the fueling
            GasP1 = Tree.getNode(r'\diagz::flux_gaz:piezo_1:flux')
            GasP2 = Tree.getNode(r'\diagz::flux_gaz:piezo_2:flux')
            GasP3 = Tree.getNode(r'\diagz::flux_gaz:piezo_3:flux')
            # load the H-alpha calibrated from the vertical line
            HalphaV = mds.Data.compile(r'pd_calibrated(1)').evaluate()
            # now check if data are available for the LPs so that
            # we can also plot the integrated flux
            try:
                Target = langmuir.LP(shot, Type='floor')
            except:
                pass
            Pressure = gauges.Baratrons(shot)
            ax[0, 0].plot(iP.getDimensionAt().data(),
                          iP.data()/1e6, color=col, label='# %5i' % shot)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_xlim([0, 1.8])
            ax[0, 0].set_ylim([0, 0.4])
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].legend(loc='best', numpoints=1, frameon=False)

            ax[1, 0].plot(eq.getTimeBase(), eq.getQ95(), color=col)
            ax[1, 0].set_ylabel(r'q$_{95}$')
            ax[1, 0].set_xlim([0, 1.8])
            ax[1, 0].axes.get_xaxis().set_visible(False)

            if tagPohm:
                ax[2, 0].plot(
                    iP.getDimensionAt().data(),
                    POhm/1e6, '-', color=col, label='Ohmic')
            else:
                ax[2, 0].plot(POhm.getDimensionAt().data(),
                              POhm.data()/1e6, '-', color=col,
                              label='Ohmic')
            try:
                ax[2, 0].plot(Bolo.time, (Bolo.LfsSol() +
                                          Bolo.LfsLeg())/1e3, '--',
                              color=col, label=r'LFS SOL + Leg')
            except:
                pass
            ax[2, 0].set_ylabel(r'[MW]')
            ax[2, 0].legend(loc='best', numpoints=1, frameon=False)
            ax[2, 0].set_xlim([0, 1.8])
            ax[2, 0].axes.get_xaxis().set_visible(False)
            try:
                ax[3, 0].plot(Pressure.t[Pressure.Compression > 0],
                              Pressure.Compression[Pressure.Compression > 0],
                              color=col)
                ax[3, 0].set_ylabel(r'Neutral Compression')
                ax[3, 0].set_xlim([0, 1.8])
                ax[3, 0].set_ylim([0, 10])
                ax[3, 0].set_xlabel('t[s]')
            except:
                pass
            ax[0, 1].plot(enAVG.getDimensionAt().data(),
                          enAVG.data()/1e19)
            ax[0, 1].set_ylabel(r'$\langle n_e \rangle [10^{19}$m$^{-3}$]')
            ax[0, 1].axes.get_xaxis().set_visible(False)

            ax[1, 1].plot(GasP1.getDimensionAt().data(),
                          GasP1.data(), color=col)
            ax[1, 1].set_ylabel(r'Gas Injection')
            ax[1, 1].axes.get_xaxis().set_visible(False)

            try:
                ax[2, 1].plot(Target.t2, Target.TotalSpIonFlux()/1e27,
                              color=col)
                ax[2, 1].text(
                    0.1, 0.85, r'Total Ion Flux [10$^{27}$s$^{-1}$]',
                    transform=ax[2, 1].transAxes)
                ax[2, 1].set_xlim([0, 1.8])
                ax[2, 1].axes.get_xaxis().set_visible(False)
                ax[2, 1].set_ylim([0, 4.5])
            except:
                pass
            ax[3, 1].plot(HalphaV.getDimensionAt().data(),
                          HalphaV.data(), color=col)
            ax[3, 1].set_ylabel(r'H$_{\alpha}$')
            ax[3, 1].set_xlim([0, 1.8])
            ax[3, 1].set_xlabel('t[s]')
            ax[3, 1].set_ylim([0, 4])
        mpl.pylab.savefig('../pdfbox/CurrentScanConstantBt.pdf',
                          bbox_to_inches='tight')
    elif selection == 2:
        shotList = (57454, 57461, 57497)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c')
        # create the plot
        fig, ax = mpl.pylab.subplots(figsize=(16, 12), nrows=4,
                                     ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.2, top=0.98, left=0.1, right=0.98)
        for shot, col in zip(shotList, colorList):
            eq = eqtools.TCVLIUQETree(shot)
            Tree = mds.Tree('tcv_shot', shot)
            iP = mds.Data.compile(r'tcv_ip()').evaluate()
            enAVG = Tree.getNode(r'\results::fir:n_average')
            # load the vloop
            Vloop = Tree.getNode(r'\magnetics::vloop')
            # now load the bolometry radiation
            Bolo = Radiation(shot)
            POhm = Tree.getNode(r'\results::conf:ptot_ohm')
            tagPohm = False
            if POhm.data().mean() == 0:
                POhm = np.abs(iP.data() * Vloop.data())
                tagPohm = True
            # load the fueling
            GasP1 = Tree.getNode(r'\diagz::flux_gaz:piezo_1:flux')
            GasP2 = Tree.getNode(r'\diagz::flux_gaz:piezo_2:flux')
            GasP3 = Tree.getNode(r'\diagz::flux_gaz:piezo_3:flux')
            # load the H-alpha calibrated from the vertical line
            HalphaV = mds.Data.compile(r'pd_calibrated(1)').evaluate()
            # now check if data are available for the LPs so that
            # we can also plot the integrated flux
            try:
                Target = langmuir.LP(shot, Type='floor')
            except:
                pass
            Pressure = gauges.Baratrons(shot)
            ax[0, 0].plot(iP.getDimensionAt().data(),
                          iP.data()/1e6, color=col, label='# %5i' % shot)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_xlim([0, 1.8])
            ax[0, 0].set_ylim([0, 0.4])
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].legend(loc='best', numpoints=1, frameon=False)

            ax[1, 0].plot(eq.getTimeBase(), eq.getQ95(), color=col)
            ax[1, 0].set_ylabel(r'q$_{95}$')
            ax[1, 0].set_xlim([0, 1.8])
            ax[1, 0].axes.get_xaxis().set_visible(False)

            if tagPohm:
                ax[2, 0].plot(
                    iP.getDimensionAt().data(),
                    POhm/1e6, '-', color=col, label='Ohmic')
            else:
                ax[2, 0].plot(POhm.getDimensionAt().data(),
                              POhm.data()/1e6, '-', color=col,
                              label='Ohmic')
            try:
                ax[2, 0].plot(Bolo.time, (Bolo.LfsLeg() +
                                          Bolo.LfsSol())/1e3, '--',
                              color=col, label=r'LFS SOL + LEG')
            except:
                pass
            ax[2, 0].set_ylabel(r'[MW]')
            ax[2, 0].legend(loc='best', numpoints=1, frameon=False)
            ax[2, 0].set_xlim([0, 1.8])
            ax[2, 0].axes.get_xaxis().set_visible(False)
#            if not Pressure._Midplane:
            ax[3, 0].plot(Pressure._Ttime,
                          Pressure._Target,
                          color=col)
            ax[3, 0].set_ylabel(r'Target Pressure [Pa]')
            ax[3, 0].set_xlim([0, 1.8])
            ax[3, 0].set_xlabel('t[s]')
#                ax[3, 0].set_ylim([0, 10])
#             elif not Pressure._Target:
#                 ax[3, 0].plot(Pressure._Mtime,
#                               Pressure._Midplane,
#                               color=col)
#                 ax[3, 0].set_ylabel(r'Midplane Pressure [Pa]')
#                 ax[3, 0].set_xlim([0, 1.8])
# #                ax[3, 0].set_ylim([0, 10])
#                 ax[3, 0].set_xlabel('t[s]')
#             else:
#                 ax[3, 0].plot(Pressure.t,
#                               Pressure.Compression,
#                               color=col)
#                 ax[3, 0].set_ylabel(r'Compression')
#                 ax[3, 0].set_xlim([0, 1.8])
#                 ax[3, 0].set_ylim([0, 10])
#                 ax[3, 0].set_xlabel('t[s]')
            ax[0, 1].plot(enAVG.getDimensionAt().data(),
                          enAVG.data()/1e19)
            ax[0, 1].set_ylabel(r'$\langle n_e \rangle [10^{19}$m$^{-3}$]')
            ax[0, 1].axes.get_xaxis().set_visible(False)

            ax[1, 1].plot(GasP1.getDimensionAt().data(),
                          GasP1.data(), color=col)
            ax[1, 1].set_ylabel(r'Gas Injection')
            ax[1, 1].axes.get_xaxis().set_visible(False)

            try:
                ax[2, 1].plot(Target.t2, Target.TotalSpIonFlux()/1e27,
                              color=col)
                ax[2, 1].text(
                    0.1, 0.85, r'Total Ion Flux [10$^{27}$s$^{-1}$]',
                    transform=ax[2, 1].transAxes, fontsize=16)
                ax[2, 1].set_xlim([0, 1.8])
                ax[2, 1].axes.get_xaxis().set_visible(False)
                ax[2, 1].set_ylim([0, 5])
            except:
                pass
            ax[3, 1].plot(HalphaV.getDimensionAt().data(),
                          HalphaV.data(), color=col)
            ax[3, 1].set_ylabel(r'H$_{\alpha}$')
            ax[3, 1].set_xlim([0, 1.8])
            ax[3, 1].set_ylim([0, 4])
            ax[3, 1].set_xlabel('t[s]')

        mpl.pylab.savefig('../pdfbox/CurrentScanConstantQ95.pdf',
                          bbox_to_inches='tight')
    elif selection == 3:
        shotList = (57437, 57425, 57497)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c')
        currentList = ('190','240', '330')
        fig, ax = mpl.pylab.subplots(figsize=(16, 10), nrows=2, ncols=3)
        for shot, _col, _idx, _cur in zip(shotList, colorList,
                                          range(len(shotList)),
                                          currentList):
            # read the average density data
            Tree = mds.Tree('tcv_shot', shot)
            enAVG = Tree.getNode(r'\results::fir:n_average')
            # plot the data
            ax[0, _idx].plot(enAVG.getDimensionAt().data(),
                             enAVG.data()/1e19, color='k', lw=2,
                             label=r' I$_p$ = ' + _cur+' kA')
            ax[0, _idx].set_xlabel(r't [s]')
            ax[0, _idx].set_ylim([0, 12])
            ax[0, _idx].legend(loc='best', numpoints=1, frameon=False)
            ax[0, _idx].set_title(r'Shot # %5i' % shot)
            if _idx == 0:
                ax[0, _idx].set_ylabel(
                    r'$\langle$n$_e\rangle [10^{19}$m$^{-3}]$')
            else:
                ax[0, _idx].axes.get_yaxis().set_visible(False)
            # read the equilibrium
            eq = eqtools.TCVLIUQETree(shot)
            # read the thomson data
            rPos = Tree.getNode(r'\diagz::thomson_set_up:radial_pos').data()
            zPos = Tree.getNode(
                r'\diagz::thomson_set_up:vertical_pos').data()
            # now the thomson times
            times = Tree.getNode(r'\results::thomson:times').data()
            # now the thomson raw data
            dataTe = Tree.getNode(r'\results::thomson:te').data()
            errTe = Tree.getNode(r'\results::thomson:te:error_bar').data()
            dataEn = Tree.getNode(r'\results::thomson:ne').data()
            errEn = Tree.getNode(r'\results::thomson:ne:error_bar').data()

            # now read the profile data for these shots for both the
            # strokes, plot the density profiles together with the
            # data from Thomson in the three closests times
            for strokes, cS in zip(('1', '2'),
                                   ('orange', 'blue')):
                # read the data in rho
                doubleP = pd.read_table(
                    '/home/tsui/idl/library/data/double/dpm' +
                    str(int(shot)) + '_' +
                    strokes + '.tab', skiprows=1,
                    header=0)
                t0 = (doubleP['Time(s)'].mean())
                # indicates the density of the strokes
                ax[0, _idx].axvline(t0, ls='--', color=cS)
                x = doubleP['rrsep(m)'][
                    :np.argmin(doubleP['rrsep(m)'])].values
                y = doubleP['Dens(m-3)'][
                    :np.argmin(doubleP['rrsep(m)'])].values/1e19
                eR = doubleP['DensErr(cm-3)'][
                    :np.argmin(doubleP['rrsep(m)'])].values/10
                # now transform from RRsep to Rho using eqtools
                rhoProbe = eq.rmid2psinorm(
                    x+eq.getRmidOutSpline()(
                        doubleP['Time(s)'].mean()),
                    doubleP['Time(s)'].mean(), sqrt=True)
                _SOL = np.where(rhoProbe > 1)[0]
                y = y[_SOL]
                eR = eR[_SOL]
                rhoProbe = rhoProbe[_SOL]
                # now found the values of the thomson closest to the
                # strokes (3 in total)
                _indexThomson = np.argmin(np.abs(times-t0))
                # append totally the rhothomson and
                # the density thomson
                rhoT = np.array([])
                enT = np.array([])
                erT = np.array([])
                for l in (-1, 0, 1):
                    _x = eq.rz2psinorm(rPos, zPos, times[_indexThomson+l],
                                       sqrt=True)
                    _y = dataEn[:, _indexThomson+l]
                    _e = errEn[:, _indexThomson+l]
                    rhoT = np.append(rhoT, _x[_y != -1])
                    erT = np.append(erT, _e[_y != -1])
                    enT = np.append(enT, _y[_y != -1])
                enT = np.asarray(enT[rhoT > 0.97])/1e19
                erT = np.asarray(erT[rhoT > 0.97])/1e19
                rhoT = np.asarray(rhoT[rhoT > 0.97])
                # now the plot and eventually the global spline
                # for the probe limit ourself to the SOL
                # now combine all together and order appropriately
                rho = np.append(rhoT, rhoProbe)
                density = np.append(enT, y)
                error = np.append(erT, eR)
                S = UnivariateSpline(rho[np.argsort(rho)],
                                     density[np.argsort(rho)], ext=0)
                _r = np.linspace(rho.min(), rho.max(), num=100)
                S.set_smoothing_factor(12)

                ax[1, _idx].plot(rhoProbe, y/S(1), 'o', color=cS, ms=10,
                                 alpha=0.4)
                ax[1, _idx].errorbar(rhoProbe, y/S(1),
                                     yerr=eR/S(1), fmt='none',
                                     ecolor=cS, alpha=0.4,
                                     errorevery=4)
                ax[1, _idx].plot(rhoT, enT/S(1), 'p',
                                 color=cS, ms=10, alpha=0.4)
                ax[1, _idx].errorbar(rhoT, enT/S(1), yerr=erT/S(1),
                                     fmt='none', color=cS,
                                     alpha=0.4, errorevery=4)
                ax[1, _idx].plot(_r, S(_r)/S(1), '--', lw=2, color=cS)
                ax[1, _idx].set_xlabel(r'$\rho_{\Psi}$')
                ax[1, _idx].set_xlim([0.96, 1.12])
            if _idx == 0:
                ax[1, _idx].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
            else:
                ax[1, _idx].axes.get_yaxis().set_visible(False)
            ax[1, _idx].set_ylim([0.01, 4])
            ax[1, _idx].set_yscale('log')
        mpl.pylab.savefig('../pdfbox/DensityProfileCurrentScanConstantBt.pdf',
                          bbox_to_inches='tight')
    elif selection == 4:
        shotList = (57461, 57454, 57497)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c')
        currentList = ('190', '240', '330')
        fig, ax = mpl.pylab.subplots(figsize=(16, 10), nrows=2, ncols=3)
        for shot, _col, _idx, _cur in zip(shotList, colorList,
                                          range(len(shotList)),
                                          currentList):
            # read the average density data
            Tree = mds.Tree('tcv_shot', shot)
            enAVG = Tree.getNode(r'\results::fir:n_average')
            # plot the data
            ax[0, _idx].plot(enAVG.getDimensionAt().data(),
                             enAVG.data()/1e19, color='k', lw=2,
                             label=r' I$_p$ = ' + _cur+' kA')
            ax[0, _idx].set_xlabel(r't [s]')
            ax[0, _idx].set_ylim([0, 12])
            ax[0, _idx].legend(loc='best', numpoints=1, frameon=False)
            ax[0, _idx].set_title(r'Shot # %5i' % shot)
            if _idx == 0:
                ax[0, _idx].set_ylabel(
                    r'$\langle$n$_e\rangle [10^{19}$m$^{-3}]$')
            else:
                ax[0, _idx].axes.get_yaxis().set_visible(False)
            # read the equilibrium
            eq = eqtools.TCVLIUQETree(shot)
            # read the thomson data
            rPos = Tree.getNode(r'\diagz::thomson_set_up:radial_pos').data()
            zPos = Tree.getNode(r'\diagz::thomson_set_up:vertical_pos').data()
            # now the thomson times
            times = Tree.getNode(r'\results::thomson:times').data()
            # now the thomson raw data
            dataTe = Tree.getNode(r'\results::thomson:te').data()
            errTe = Tree.getNode(r'\results::thomson:te:error_bar').data()
            dataEn = Tree.getNode(r'\results::thomson:ne').data()
            errEn = Tree.getNode(r'\results::thomson:ne:error_bar').data()

            # now read the profile data for these shots for both the
            # strokes, plot the density profiles together with the
            # data from Thomson in the three closests times
            for strokes, cS in zip(('1', '2'),
                                   ('orange', 'blue')):
                # read the data in rho
                doubleP = pd.read_table(
                    '/home/tsui/idl/library/data/double/dpm' +
                    str(int(shot)) + '_' +
                    strokes + '.tab', skiprows=1,
                    header=0)
                t0 = (doubleP['Time(s)'].mean())
                # indicates the density of the strokes
                ax[0, _idx].axvline(t0, ls='--', color=cS)
                x = doubleP['rrsep(m)'][
                    :np.argmin(doubleP['rrsep(m)'])].values
                y = doubleP['Dens(m-3)'][
                    :np.argmin(doubleP['rrsep(m)'])].values/1e19
                eR = doubleP['DensErr(cm-3)'][
                    :np.argmin(doubleP['rrsep(m)'])].values/10
                # now transform from RRsep to Rho using eqtools
                rhoProbe = eq.rmid2psinorm(
                    x+eq.getRmidOutSpline()(
                        doubleP['Time(s)'].mean()),
                    doubleP['Time(s)'].mean(), sqrt=True)
                _SOL = np.where(rhoProbe > 1)[0]
                y = y[_SOL]
                eR = eR[_SOL]
                rhoProbe = rhoProbe[_SOL]
                # now found the values of the thomson closest to the
                # strokes (3 in total)
                _indexThomson = np.argmin(np.abs(times-t0))
                # append totally the rhothomson and
                # the density thomson
                rhoT = np.array([])
                enT = np.array([])
                erT = np.array([])
                for l in (-1, 0, 1):
                    _x = eq.rz2psinorm(rPos, zPos, times[_indexThomson+l],
                                       sqrt=True)
                    _y = dataEn[:, _indexThomson+l]
                    _e = errEn[:, _indexThomson+l]
                    rhoT = np.append(rhoT, _x[_y != -1])
                    erT = np.append(erT, _e[_y != -1])
                    enT = np.append(enT, _y[_y != -1])
                enT = np.asarray(enT[rhoT > 0.97])/1e19
                erT = np.asarray(erT[rhoT > 0.97])/1e19
                rhoT = np.asarray(rhoT[rhoT > 0.97])
                # now the plot and eventually the global spline
                # for the probe limit ourself to the SOL
                # now combine all together and order appropriately
                rho = np.append(rhoT, rhoProbe)
                density = np.append(enT, y)
                error = np.append(erT, eR)
                S = UnivariateSpline(rho[np.argsort(rho)],
                                     density[np.argsort(rho)], ext=0)
                _r = np.linspace(rho.min(), rho.max(), num=100)
                S.set_smoothing_factor(12)

                ax[1, _idx].plot(rhoProbe, y/S(1), 'o', color=cS, ms=10,
                                 alpha=0.3)
                ax[1, _idx].errorbar(rhoProbe, y/S(1),
                                     yerr=eR/S(1), fmt='none',
                                     ecolor=cS, alpha=0.3,
                                     errorevery=4)
                ax[1, _idx].plot(rhoT, enT/S(1), 'p',
                                 color=cS, ms=10, alpha=0.3)
                ax[1, _idx].errorbar(rhoT, enT/S(1), yerr=erT/S(1),
                                     fmt='none', color=cS,
                                     alpha=0.3, errorevery=4)
                ax[1, _idx].plot(_r, S(_r)/S(1), '--', lw=2, color=cS)
                ax[1, _idx].set_xlabel(r'$\rho_{\Psi}$')
                ax[1, _idx].set_xlim([0.96, 1.12])
            if _idx == 0:
                ax[1, _idx].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
            else:
                ax[1, _idx].axes.get_yaxis().set_visible(False)
            ax[1, _idx].set_ylim([0.01, 4])
            ax[1, _idx].set_yscale('log')
        mpl.pylab.savefig('../pdfbox/DensityProfileCurrentScanConstantQ95.pdf',
                          bbox_to_inches='tight')
    elif selection == 5:
        shotList = (57437, 57425, 57497)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c')
        currentList = ('190','240', '330')
        fig, ax = mpl.pylab.subplots(figsize=(16, 10), nrows=2, ncols=3)
        for shot, _col, _idx, _cur in zip(shotList, colorList,
                                          range(len(shotList)),
                                          currentList):
            # read the average density data
            Tree = mds.Tree('tcv_shot', shot)
            enAVG = Tree.getNode(r'\results::fir:n_average')
            # plot the data
            ax[0, _idx].plot(enAVG.getDimensionAt().data(),
                             enAVG.data()/1e19, color='k', lw=2,
                             label=r' I$_p$ = ' + _cur+' kA')
            ax[0, _idx].set_xlabel(r't [s]')
            ax[0, _idx].set_ylim([0, 12])
            ax[0, _idx].legend(loc='best', numpoints=1, frameon=False)
            ax[0, _idx].set_title(r'Shot # %5i' % shot)
            if _idx == 0:
                ax[0, _idx].set_ylabel(
                    r'$\langle$n$_e\rangle [10^{19}$m$^{-3}]$')
            else:
                ax[0, _idx].axes.get_yaxis().set_visible(False)
            # read the equilibrium
            eq = eqtools.TCVLIUQETree(shot)

            # now read the profile data for these shots for both the
            # strokes, plot the density profiles together with the
            # data from Thomson in the three closests times
            for strokes, cS in zip(('1', '2'),
                                   ('orange', 'blue')):
                # read the data in rho
                doubleP = pd.read_table(
                    '/home/tsui/idl/library/data/double/dpm' +
                    str(int(shot)) + '_' +
                    strokes + '.tab', skiprows=1,
                    header=0)
                t0 = (doubleP['Time(s)'].mean())
                # indicates the density of the strokes
                ax[0, _idx].axvline(t0, ls='--', color=cS)
                x = doubleP['rrsep(m)'][
                    :np.argmin(doubleP['rrsep(m)'])].values
                y = doubleP['Dens(m-3)'][
                    :np.argmin(doubleP['rrsep(m)'])].values/1e19
                eR = doubleP['DensErr(cm-3)'][
                    :np.argmin(doubleP['rrsep(m)'])].values/10
                # now transform from RRsep to Rho using eqtools
                rhoProbe = eq.rmid2psinorm(
                    x+eq.getRmidOutSpline()(
                        doubleP['Time(s)'].mean()),
                    doubleP['Time(s)'].mean(), sqrt=True)
                _SOL = np.where(rhoProbe > 1)[0]
                y = y[_SOL]
                eR = eR[_SOL]
                rhoProbe = rhoProbe[_SOL]
                # now found the values of the thomson closest to the
                # strokes (3 in total)
                # now the plot and eventually the global spline
                # for the probe limit ourself to the SOL
                # now combine all together and order appropriately
                S = UnivariateSpline(rhoProbe[np.argsort(rhoProbe)],
                                     y[np.argsort(rhoProbe)], ext=0)
                _r = np.linspace(rhoProbe.min(), rhoProbe.max(), num=100)
                S.set_smoothing_factor(12)

                ax[1, _idx].plot(rhoProbe, y/S(1), 'o', color=cS, ms=10,
                                 alpha=0.4)
                ax[1, _idx].errorbar(rhoProbe, y/S(1),
                                     yerr=eR/S(1), fmt='none',
                                     ecolor=cS, alpha=0.4, errorevery=3)
                ax[1, _idx].plot(_r, S(_r)/S(1), '--', lw=2, color=cS)
                ax[1, _idx].set_xlabel(r'$\rho_{\Psi}$')
                ax[1, _idx].set_xlim([0.99, 1.12])
            if _idx == 0:
                ax[1, _idx].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
            else:
                ax[1, _idx].axes.get_yaxis().set_visible(False)
            ax[1, _idx].set_ylim([0.01, 4])
            ax[1, _idx].set_yscale('log')
        mpl.pylab.savefig('../pdfbox/DensityProfileCurrentScanConstantBtSOL.pdf',
                          bbox_to_inches='tight')
    elif selection == 6:
        shotList = (57461, 57454, 57497)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c')
        currentList = ('190', '240', '330')
        fig, ax = mpl.pylab.subplots(figsize=(16, 10), nrows=2, ncols=3)
        for shot, _col, _idx, _cur in zip(shotList, colorList,
                                          range(len(shotList)),
                                          currentList):
            # read the average density data
            Tree = mds.Tree('tcv_shot', shot)
            enAVG = Tree.getNode(r'\results::fir:n_average')
            # plot the data
            ax[0, _idx].plot(enAVG.getDimensionAt().data(),
                             enAVG.data()/1e19, color='k', lw=2,
                             label=r' I$_p$ = ' + _cur+' kA')
            ax[0, _idx].set_xlabel(r't [s]')
            ax[0, _idx].set_ylim([0, 12])
            ax[0, _idx].legend(loc='best', numpoints=1, frameon=False)
            ax[0, _idx].set_title(r'Shot # %5i' % shot)
            if _idx == 0:
                ax[0, _idx].set_ylabel(
                    r'$\langle$n$_e\rangle [10^{19}$m$^{-3}]$')
            else:
                ax[0, _idx].axes.get_yaxis().set_visible(False)
            # read the equilibrium
            eq = eqtools.TCVLIUQETree(shot)

            # now read the profile data for these shots for both the
            # strokes, plot the density profiles together with the
            # data from Thomson in the three closests times
            for strokes, cS in zip(('1', '2'),
                                   ('orange', 'blue')):
                # read the data in rho
                doubleP = pd.read_table(
                    '/home/tsui/idl/library/data/double/dpm' +
                    str(int(shot)) + '_' +
                    strokes + '.tab', skiprows=1,
                    header=0)
                t0 = (doubleP['Time(s)'].mean())
                # indicates the density of the strokes
                ax[0, _idx].axvline(t0, ls='--', color=cS)
                x = doubleP['rrsep(m)'][
                    :np.argmin(doubleP['rrsep(m)'])].values
                y = doubleP['Dens(m-3)'][
                    :np.argmin(doubleP['rrsep(m)'])].values/1e19
                eR = doubleP['DensErr(cm-3)'][
                    :np.argmin(doubleP['rrsep(m)'])].values/10
                # now transform from RRsep to Rho using eqtools
                rhoProbe = eq.rmid2psinorm(
                    x+eq.getRmidOutSpline()(
                        doubleP['Time(s)'].mean()),
                    doubleP['Time(s)'].mean(), sqrt=True)
                _SOL = np.where(rhoProbe > 1)[0]
                y = y[_SOL]
                eR = eR[_SOL]
                rhoProbe = rhoProbe[_SOL]
                S = UnivariateSpline(rhoProbe[np.argsort(rhoProbe)],
                                     y[np.argsort(rhoProbe)], ext=0)
                _r = np.linspace(rhoProbe.min(), rhoProbe.max(), num=100)
                S.set_smoothing_factor(12)

                ax[1, _idx].plot(rhoProbe, y/S(1), 'o', color=cS, ms=10,
                                 alpha=0.4)
                ax[1, _idx].errorbar(rhoProbe, y/S(1),
                                     yerr=eR/S(1), fmt='none',
                                     ecolor=cS, alpha=0.4, errorevery=3)
                ax[1, _idx].plot(_r, S(_r)/S(1), '--', lw=2, color=cS)
                ax[1, _idx].set_xlabel(r'$\rho_{\Psi}$')
                ax[1, _idx].set_xlim([0.99, 1.12])
            if _idx == 0:
                ax[1, _idx].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
            else:
                ax[1, _idx].axes.get_yaxis().set_visible(False)
            ax[1, _idx].set_ylim([0.01, 4])
            ax[1, _idx].set_yscale('log')
        mpl.pylab.savefig(
            '../pdfbox/DensityProfileCurrentScanConstantQ95SOL.pdf',
            bbox_to_inches='tight')
    elif selection == 7:
        shotList = (57498, 57499, 57500)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c')
        # create the plot
        fig, ax = mpl.pylab.subplots(figsize=(16, 12), nrows=4,
                                     ncols=2, sharex=True)
        fig.subplots_adjust(hspace=0.2, top=0.98, left=0.1, right=0.98)
        for shot, col in zip(shotList, colorList):
            eq = eqtools.TCVLIUQETree(shot)
            Tree = mds.Tree('tcv_shot', shot)
            iP = mds.Data.compile(r'tcv_ip()').evaluate()
            enAVG = Tree.getNode(r'\results::fir:n_average')
            # load the vloop
            Vloop = Tree.getNode(r'\magnetics::vloop')
            # now load the bolometry radiation
            Bolo = Radiation(shot)
            POhm = Tree.getNode(r'\results::conf:ptot_ohm')
            tagPohm = False
            if POhm.data().mean() == 0:
                POhm = np.abs(iP.data() * Vloop.data())
                tagPohm = True
            # load the fueling
            GasP1 = Tree.getNode(r'\diagz::flux_gaz:piezo_1:flux')
            GasP2 = Tree.getNode(r'\diagz::flux_gaz:piezo_2:flux')
            GasP3 = Tree.getNode(r'\diagz::flux_gaz:piezo_3:flux')
            # load the H-alpha calibrated from the vertical line
            HalphaV = mds.Data.compile(r'pd_calibrated(1)').evaluate()
            # now check if data are available for the LPs so that
            # we can also plot the integrated flux
            try:
                Target = langmuir.LP(shot, Type='floor')
            except:
                pass
            Pressure = gauges.Baratrons(shot)
            ax[0, 0].plot(iP.getDimensionAt().data(),
                          iP.data()/1e6, color=col, label='# %5i' % shot)
            ax[0, 0].set_ylabel(r'I$_p$ [MA]')
            ax[0, 0].set_xlim([0, 1.8])
            ax[0, 0].set_ylim([0, 0.4])
            ax[0, 0].axes.get_xaxis().set_visible(False)
            ax[0, 0].legend(loc='best', numpoints=1, frameon=False)

            ax[1, 0].plot(eq.getTimeBase(), eq.getQ95(), color=col)
            ax[1, 0].set_ylabel(r'q$_{95}$')
            ax[1, 0].set_xlim([0, 1.8])
            ax[1, 0].axes.get_xaxis().set_visible(False)

            if tagPohm:
                ax[2, 0].plot(
                    iP.getDimensionAt().data(),
                    POhm/1e6, '-', color=col, label='Ohmic')
            else:
                ax[2, 0].plot(POhm.getDimensionAt().data(),
                              POhm.data()/1e6, '-', color=col,
                              label='Ohmic')
            try:
                ax[2, 0].plot(Bolo.time, (Bolo.LfsSol() +
                                          Bolo.LfsLeg())/1e3, '--',
                              color=col, label=r'LFS SOL + Leg')
            except:
                pass
            ax[2, 0].set_ylabel(r'[MW]')
            ax[2, 0].legend(loc='best', numpoints=1, frameon=False)
            ax[2, 0].set_xlim([0, 1.8])
            ax[2, 0].axes.get_xaxis().set_visible(False)
            try:
                ax[3, 0].plot(Pressure.t[Pressure.Compression > 0],
                              Pressure.Compression[Pressure.Compression > 0],
                              color=col)
                ax[3, 0].set_ylabel(r'Neutral Compression')
                ax[3, 0].set_xlim([0, 1.8])
                ax[3, 0].set_ylim([0, 10])
                ax[3, 0].set_xlabel('t[s]')
            except:
                pass
            ax[0, 1].plot(enAVG.getDimensionAt().data(),
                          enAVG.data()/1e19)
            ax[0, 1].set_ylabel(r'$\langle n_e \rangle [10^{19}$m$^{-3}$]')
            ax[0, 1].axes.get_xaxis().set_visible(False)

            ax[1, 1].plot(GasP1.getDimensionAt().data(),
                          GasP1.data(), color=col)
            ax[1, 1].set_ylabel(r'Gas Injection')
            ax[1, 1].axes.get_xaxis().set_visible(False)

            try:
                ax[2, 1].plot(Target.t2, Target.TotalSpIonFlux()/1e27,
                              color=col)
                ax[2, 1].text(
                    0.1, 0.85, r'Total Ion Flux [10$^{27}$s$^{-1}$]',
                    transform=ax[2, 1].transAxes)
                ax[2, 1].set_xlim([0, 1.8])
                ax[2, 1].axes.get_xaxis().set_visible(False)
                ax[2, 1].set_ylim([0, 4.5])
            except:
                pass
            ax[3, 1].plot(HalphaV.getDimensionAt().data(),
                          HalphaV.data(), color=col)
            ax[3, 1].set_ylabel(r'H$_{\alpha}$')
            ax[3, 1].set_xlim([0, 1.8])
            ax[3, 1].set_xlabel('t[s]')
            ax[3, 1].set_ylim([0, 4])
        mpl.pylab.savefig('../pdfbox/LowCollisionalityAttempt.pdf',
                          bbox_to_inches='tight')

    elif selection == 8:
        shotL = (57461, 57454, 57497)
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
        for shot, col, _ax in zip(shotL, colorL, axL):
            Tree = mds.Tree('tcv_shot', shot)
            iP = mds.Data.compile(r'tcv_ip()').evaluate()
            ax1.plot(iP.getDimensionAt().data(), iP.data()/1e6,
                     color=col, label=r'# %5i' % shot, lw=3)
            ax1.axes.get_xaxis().set_visible(False)
            ax1.set_xlim([0., 1.7])
            ax1.axes.get_xaxis().set_visible(False)
            ax1.set_ylim([0., 0.4])

            enE = Tree.getNode(r'\results::fir_n_average_array')
            ax12.plot(enE.getDimensionAt().data(), enE.data()[-1, :]/1e20,
                      '-', color=col, lw=3)
            ax12.set_xlim([0, 1.7])
            ax12.set_ylim([0, 1])
            ax12.axes.get_xaxis().set_visible(False)

            GasP1 = Tree.getNode(r'\diagz::flux_gaz:piezo_1:flux')
            ax13.plot(GasP1.getDimensionAt().data(), GasP1.data(),
                      '-', color=col, lw=3)
            ax13.set_ylabel(r'D$_2$')
            ax13.set_xlabel(r't [s]')
            ax13.set_xlim([0.1, 1.7])
            eq = eqtools.TCVLIUQETree(shot)
            # read the thomson data
            rPos = Tree.getNode(r'\diagz::thomson_set_up:radial_pos').data()
            zPos = Tree.getNode(r'\diagz::thomson_set_up:vertical_pos').data()
            # now the thomson times
            times = Tree.getNode(r'\results::thomson:times').data()
            # now the thomson raw data
            dataTe = Tree.getNode(r'\results::thomson:te').data()
            errTe = Tree.getNode(r'\results::thomson:te:error_bar').data()
            dataEn = Tree.getNode(r'\results::thomson:ne').data()
            errEn = Tree.getNode(r'\results::thomson:ne:error_bar').data()

            # now we need the profiles from Thomson and from the probe
            # for the probe we use the newly stored Tree
            filam = mds.Tree('tcv_topic21', shot)
            for plunge, _c in zip(('1', '2'), ('k', 'orange')):
                enProbe = filam.getNode(r'\fp_' + plunge + 'pl_en').data()/1e19
                errProbe = filam.getNode(
                    r'\fp_' + plunge + 'pl_enerr').data()/1e19
                rhoProbe = filam.getNode(r'\fp_' + plunge + 'pl_rho').data()
                t0 = filam.getNode(
                    r'\fp_' + plunge + 'pl_en').getDimensionAt().data().mean()
                # limit to only values of probes into the SOL_geometry
                enProbe = enProbe[np.where(rhoProbe >= 1)[0]]
                errProbe = errProbe[np.where(rhoProbe >= 1)[0]]
                rhoProbe = rhoProbe[np.where(rhoProbe >= 1)[0]]
                _indexThomson = np.argmin(np.abs(times-t0))
                # append totally the rhothomson and
                # the density thomson
                rhoT = np.array([])
                enT = np.array([])
                erT = np.array([])
                for l in (-1, 0, 1):
                    _x = eq.rz2psinorm(rPos, zPos, times[_indexThomson+l],
                                       sqrt=True)
                    _y = dataEn[:, _indexThomson+l]
                    _e = errEn[:, _indexThomson+l]
                    rhoT = np.append(rhoT, _x[_y != -1])
                    erT = np.append(erT, _e[_y != -1])
                    enT = np.append(enT, _y[_y != -1])
                enT = np.asarray(enT[rhoT > 0.985])/1e19
                erT = np.asarray(erT[rhoT > 0.985])/1e19
                rhoT = np.asarray(rhoT[rhoT > 0.985])
                rho = np.append(rhoT, rhoProbe)
                error = np.append(erT, errProbe)
                density = np.append(enT, enProbe)
                S = UnivariateSpline(rho[np.argsort(rho)],
                                     density[np.argsort(rho)], ext=0)
                _r = np.linspace(rho.min(), rho.max(), num=100)
                S.set_smoothing_factor(12)
                # find the corresponding mean values of edge density
                _idx = np.where((enE.getDimensionAt().data() >= t0-0.01) &
                                (enE.getDimensionAt().data() <= t0+0.01))[0]
                _n = enE.data()[-1, _idx].mean()/1e19
                _ax.plot(rhoProbe, enProbe/S(1), 'o', color=_c,
                         ms=10, alpha=0.3)
                _ax.errorbar(rhoProbe, enProbe/S(1),
                             yerr=errProbe/S(1), fmt='none',
                             ecolor=_c, alpha=0.3,
                             errorevery=4)
                _ax.plot(rhoT, enT/S(1), 'p',
                         color=_c, ms=10, alpha=0.3)
                _ax.errorbar(rhoT, enT/S(1), yerr=erT/S(1),
                             fmt='none', color=_c,
                             alpha=0.3, errorevery=4)
                _ax.plot(_r, S(_r)/S(1), '--', lw=2,
                         color=_c, label=r'n$_e$ = %3.2f' % _n)
            _ax.set_ylim([0.01, 3])
            _ax.set_yscale('log')
            _ax.set_xlabel(r'$\rho_{\Psi}$')
            _ax.set_xlim([0.98, 1.08])
            _ax.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
            _ax.set_title(r'# %5i' % shot)
        ax1.legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        ax1.set_ylabel(r'I$_p$ [MA]')
        ax12.set_ylabel(r'$\overline{n}_e$ edge [10$^{20}$]')
        ax13.set_ylabel(r'D$_2 [10^{21}$s$^{-1}]$')
        ax1.set_title(r'I$_p$ scan at constant q$_{95}$')
        ax2.set_ylabel(r'n$_e$/n$_e(\rho_p = 1)$')
        ax3.axes.get_yaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)
        mpl.pylab.savefig('../pdfbox/IpConstantq95_samedensity.pdf',
                          bbox_to_inches='tight')
        mpl.pylab.savefig('../pngbox/IpConstantq95_samedensity.png',
                          bbox_to_inches='tight', dpi=300)
ssh
    elif selection == 99:
        loop = False
    else:
        raw_input("Unknown Option Selected!")

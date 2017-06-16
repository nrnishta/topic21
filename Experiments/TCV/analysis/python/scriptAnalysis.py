import numpy as np
import sys
import itertools
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
        fig.subplots_adjust(hspace=0.2, top=0.98, left=0.17, right=0.98)
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
                ax[2, 0].plot(Bolo.time, (Bolo.LfsSol() + Bolo.LfsLeg())/1e3, '--',
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
        fig.subplots_adjust(hspace=0.2, top=0.98, left=0.17, right=0.98)
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
        pass
    elif selection == 4:
        pass
    elif selection == 99:
        loop = False
    else:
        raw_input("Unknown Option Selected!")

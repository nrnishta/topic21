import matplotlib as mpl
import MDSplus as mds
import pandas as pd
import numpy as np
from tcv.diag.frp import FastRP
from scipy.interpolate import UnivariateSpline
from scipy import constants
mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
mpl.rc("font", size=18)
mpl.rc("lines", linewidth=2)


def print_menu():
    print 30 * "-", "MENU", 30 * "-"
    print "1. Current scan from Topic 25"
    print "2. Loop on Topic 25 Upstream profile evolution"
    print "3. Plot proposed density ramp"
    print "99: End"
    print 67 * "-"


loop = True
while loop:
    print_menu()
    selection = input("Enter your choice [1-99] ")
    if selection == 1:
        shotList = (57082, 57086, 57087, 57088, 57089)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                     '#17becf')
        fig, ax = mpl.pylab.subplots(figsize=(10, 14), nrows=3, ncols=1)
        fig.subplots_adjust(hspace=0.2, top=0.98, left=0.17, right=0.98)
        for shot, col in zip(shotList, colorList):
            Tree = mds.Tree('tcv_shot', shot)
            # retrieve the current
            iP = Tree.getNode(r'\magnetics::iplasma:trapeze').data()
            iPt = Tree.getNode(
                r'\magnetics::iplasma:trapeze').getDimensionAt().data()
            ax[0].plot(iPt, iP/1e3, lw=3, color=col,
                       label=r'# %5i' % shot)
            ax[0].set_xlim([0, 2])
            ax[0].set_ylim([0, 450])
            ax[0].axes.get_xaxis().set_visible(False)
            # get the density in 10^19
            eN = Tree.getNode(r'\results::fir:n_average').data()/1e19
            eNT = Tree.getNode(
                r'\results::fir:n_average').getDimensionAt().data()
            ax[1].plot(eNT, eN, lw=3, color=col)
            ax[1].set_xlim([0, 2])
            ax[1].set_ylim([0, 15])
            # get the UnivariateSpline represenation of the density
            enS = UnivariateSpline(eNT, eN, s=0)
            # now retrieve the information for the jSat from Langmuir
            R, Z = zip(Tree.getNode(r'\results::langmuir:pos').data())
            R = np.asarray(R).ravel()
            Z = np.asarray(Z).ravel()
            _idxBottom = np.where(Z == -0.75)[0]
            Rb = R[_idxBottom]
            # limit the computation to the bottom probes
            jSat = Tree.getNode(
                r'\results::langmuir.jsat2').data()[:, _idxBottom]
            jSatT = Tree.getNode(r'\results::langmuir:time2').data()
            Tree.quit()
            intFlux = np.zeros(jSatT.size)
            for i in range(jSatT.size):
                _x = Rb
                _y = jSat[i, :]
                # eliminate the NaN
                _dummy = np.vstack((_x, _y)).transpose()
                _dummy = _dummy[~np.isnan(_dummy).any(1)]
                _x = _dummy[:, 0]
                _y = _dummy[:, 1][np.argsort(_x)]
                _x = np.sort(_x)
                intFlux[i] = np.trapz(_y/constants.elementary_charge*1e4,
                                      x=2*np.pi*_x)
            _idxTime = np.where(jSatT <= eNT.max())[0]
            ax[2].plot(enS(jSatT[_idxTime]),
                       intFlux[_idxTime]/1e27, color=col)

        ax[0].set_ylabel(r'I$_p$ [kA]')
        ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=16)
        ax[1].set_xlabel(r't[s]')
        ax[1].set_ylabel(r'$\langle$n$_e\rangle$ [10$^{19}$m$^{-3}$]')
        ax[2].set_xlabel(r'$\langle$n$_e\rangle$ [10$^{19}$m$^{-3}$]')
        ax[2].set_ylabel(r'Ion Flux [10$^{27}$ ion/s]')
        mpl.pylab.savefig('../pdfbox/CurrentScanTopic25.pdf',
                          bbox_to_inches='tight')

    if selection == 2:
        # now we still cycle over the same shots but we check for
        # the existence of processed data from FastProbe
        shotList = (57082, 57086, 57087, 57088, 57089)
        for shot in shotList:
            fig, ax = mpl.pylab.subplots(figsize=(10, 14), nrows=3, ncols=1)
            fig.subplots_adjust(hspace=0.2, left=0.18, top=0.91, right=0.98)
            Tree = mds.Tree('tcv_shot', shot)
            iP = Tree.getNode(r'\magnetics::iplasma:trapeze').data()
            iPt = Tree.getNode(
                r'\magnetics::iplasma:trapeze').getDimensionAt().data()
            ax[0].plot(iPt, iP/1e3, lw=3)
            ax[0].set_title(r'# %5i' % shot)
            ax[0].set_xlim([0, 2])
            ax[0].set_ylim([0, 450])
            ax[0].axes.get_xaxis().set_visible(False)
            eN = Tree.getNode(r'\results::fir:n_average').data()/1e19
            eNT = Tree.getNode(
                r'\results::fir:n_average').getDimensionAt().data()
            ax[1].plot(eNT, eN, lw=3)
            ax[1].set_xlim([0, 2])
            ax[1].set_ylim([0, 15])
            ax[1].set_xlabel(r't[s]')
            ax[1].set_ylabel(r'$\langle$n$_e\rangle$ [10$^{19}$m$^{-3}$]')
            Tree.quit()
            colL = ('#1f77b4', '#ff7f0e')
            plL = ('1', '2')
            for p, c in zip(plL, colL):
                try:
                    doubleP = pd.read_table(
                        '/home/tsui/idl/library/data/double/dpm' +
                        str(int(shot)) + '_' + p +'.tab', skiprows=1, header=0)
                    xO = doubleP['rrsep(m)'][
                        :np.argmin(doubleP['rrsep(m)'])]
                    yO = doubleP['Dens(m-3)'][
                        :np.argmin(doubleP['rrsep(m)'])]/1e19
                    Profile = FastRP._getprofileR(xO, yO, npoint=30)
                    x = Profile.rho.values
                    y = Profile.values
                    err = Profile.err
                    spline = UnivariateSpline(x, y, ext=0)
                    xFake = np.linspace(0, 0.035, 50)
                    ax[2].plot(x*1e2, y, 'o', mfc=c, mec='white',
                               markersize=16)
                    ax[2].errorbar(x*1e2, y, yerr=err, ecolor=c, fmt='none')
                    ax[2].plot(xFake*1e2, spline(xFake), '--', color=c)
                    ax[2].set_xlabel(r'R-R$_s$ [cm]')
                    ax[2].set_ylabel(r'n$_e [10^{19}$m$^{-3}]$')
                    ax[2].set_yscale('log')
                    ax[2].set_xlim([0, 4])
                    ax[1].axvline(doubleP['Time(s)'].mean(), ls='--', color=c)
                except:
                    pass
            mpl.pylab.savefig('../pdfbox/UpstreamProfileShot_'
                              + str(int(shot)) + '.pdf',
                              bbox_to_inches='tight')
    elif selection == 3:
        tFake = np.asarray([0, 0.2, 0.4, 0.54, 1.6])
        enFake = np.asarray([0, 2.8, 3.6, 3.6, 11])
        shotL = (57082, 57088, 51178, 57089)
        iPL = (330, 245, 160, 185)
        fig, ax = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig.subplots_adjust(left=0.17, top=0.98, bottom=0.16)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728')
        for shot, col, ip in zip(shotL, colorList, iPL):
            Tree = mds.Tree('tcv_shot', shot)
            eN = Tree.getNode(r'\results::fir:n_average').data()/1e19
            eNT = Tree.getNode(
                r'\results::fir:n_average').getDimensionAt().data()
            ax.plot(eNT, eN, lw=2, color=col, label=r'# %5i' % shot +
                    r' I$_p$ = %3i' % ip)
        ax.plot(tFake, enFake, 'k--', lw=3)
        ax.set_xlabel(r't[s]')
        ax.set_ylabel(r'$\langle$n$_e\rangle$ [10$^{19}$m$^{-3}$]')
        ax.legend(loc='best', numpoints=1, frameon=False)
        mpl.pylab.savefig('../pdfbox/ProposedDensityRamp.pdf',
                          bbox_to_inches='tight')
        
    elif selection == 99:
        loop = False
    else:
        raw_input("Unknown Option Selected!")

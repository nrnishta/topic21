# summary plot to show the equilibrium, the reproducibility of the
#
from __future__ import print_function
import timeit
from boloradiation import Radiation
import matplotlib as mpl
import numpy as np
import eqtools
import MDSplus as mds
from scipy.interpolate import UnivariateSpline
import langmuir
import pandas as pd
from tcv.diag.frp import FastRP
# first of all choose the shot you need
try:
    shot = np.uint(raw_input('Enter shot number: '))
except:
    print('Wrong shot number')

# load the equilibrium
tstart = timeit.default_timer()
# code you want to evaluate
eq = eqtools.TCVLIUQETree(shot)
tEnd = timeit.default_timer() - tstart
print('Time for loading equilibrium %3.1f' % tEnd)
# load the psi and the grid
rGrid = eq.getRGrid(length_unit='m')
zGrid = eq.getZGrid(length_unit='m')
psi = eq.getFluxGrid()
tPsi = eq.getTimeBase()
# load the ip and the density
Tree = mds.Tree('tcv_shot', shot)
tstart = timeit.default_timer()
iP = mds.Data.compile(r'tcv_ip()').evaluate()
tEnd = timeit.default_timer() - tstart
print('Time for reading Ip %3.1f' % tEnd)
enAVG = Tree.getNode(r'\results::fir:n_average')
# load the vloop
Vloop = Tree.getNode(r'\magnetics::vloop')
# now load the bolometry radiation
tstart = timeit.default_timer()
Bolo = Radiation(shot)
tEnd = timeit.default_timer() - tstart
print('Time for reading Bolo %3.1f' % tEnd)
# now load the thomson scattering
rPos = Tree.getNode(r'\diagz::thomson_set_up:radial_pos').data()
zPos = Tree.getNode(r'\diagz::thomson_set_up:vertical_pos').data()
# now the thomson times
times = Tree.getNode(r'\results::thomson:times').data()
# now the thomson raw data
dataTe = Tree.getNode(r'\results::thomson:te').data()
errTe = Tree.getNode(r'\results::thomson:te:error_bar').data()
dataEn = Tree.getNode(r'\results::thomson:ne').data()
errEn = Tree.getNode(r'\results::thomson:ne:error_bar').data()
# we now try to perform for each time the spline profile
# so that we then create a uniformly spaced rho spaced from 0.7 to 1.14 in
# rho
tstart = timeit.default_timer()
rhoFake = np.linspace(0.8, 1.08, 50)
profS = np.zeros((dataEn.shape[1], rhoFake.size))
for t, _idx in zip(times, range(times.size)):
    rho = eq.rz2psinorm(rPos, zPos, t, sqrt=True)
    mask = np.where(dataEn[:, _idx] != -1)[0]
    if (mask.size > 0) & (mask.size < dataEn.shape[0]):
        _y = dataEn[mask, _idx]
        _e = errEn[mask, _idx]
        rho = rho[mask]
        S = UnivariateSpline(rho, _y, w=1./_e**2,
                             s=None, ext=0, k=4)
        profS[_idx, :] = S(rhoFake)

# now load the power
try:
    PEcrh = Tree.getNode(r'\results::toray.input:p_gyro[*, 10]')
except:
    PEcrh = None
POhm = Tree.getNode(r'\results::conf:ptot_ohm')
tagPohm = False
if POhm.data().mean() == 0:
    POhm = np.abs(iP.data() * Vloop.data())
    tagPohm = True
# load the data from NBH
PNbi = Tree.getNode(r'\atlas::nbh.data.main_adc:data')
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

# now the appropriate plot
fig, ax = mpl.pylab.subplots(figsize=(16, 8), nrows=4, ncols=2,
                             sharex=True)
fig.subplots_adjust(right=0.7, bottom=0.15, left=0.07, top=0.98)
ax[0, 0].plot(iP.getDimensionAt().data(), iP.data()/1e6)
ax[0, 0].set_ylabel(r'I$_p$ [MA]')
ax[0, 0].set_xlim([0, 2])
ax[0, 0].set_ylim([0, 0.4])
ax[0, 0].axes.get_xaxis().set_visible(False)
if tagPohm:
    ax[1, 0].plot(iP.getDimensionAt().data(), POhm/1e6, 'k', label='Ohmic')
else:
    ax[1, 0].plot(POhm.getDimensionAt().data(),
                  POhm.data()/1e6, 'k', label='Ohmic')
if PEcrh:
    ax[1, 0].plot(PEcrh.getDimensionAt().data(),
                  PEcrh.data()/1e6, 'r', label='ECRH')
try:
    ax[1, 0].plot(PNbi.getDimensionAt().data(),
                  PNbi.data()/1e6, 'r', label='NBI')
except:
    pass
ax[1, 0].set_ylabel(r'[MW]')
ax[1, 0].legend(loc='best', numpoints=1, frameon=False)
ax[1, 0].set_xlim([0, 2])
ax[1, 0].axes.get_xaxis().set_visible(False)
try:
    ax[2, 0].plot(Bolo.time, Bolo.core()/1e6, '-', label=r'Core')
    ax[2, 0].plot(Bolo.time, Bolo.LfsLeg()/1e6, '-', label=r'LFS leg')
    ax[2, 0].plot(Bolo.time, Bolo.lfsSol()/1e6, '-', label=r'LFS SOL')
    ax[2, 0].set_ylabel(r'[MW]')
    ax[2, 0].legend(loc='best', numpoints=1, frameon=False)
    ax[2, 0].set_xlim([0, 2])
    ax[2, 0].axes.get_xaxis().set_visible(False)
except:
    pass
ax[3, 0].plot(HalphaV.getDimensionAt().data(),
              HalphaV.data())
ax[3, 0].set_ylabel(r'H$_{\alpha}$')
ax[3, 0].set_xlim([0, 2])

ax[0, 1].plot(enAVG.getDimensionAt().data(),
              enAVG.data()/1e19)
ax[0, 1].set_ylabel(r'$\langle n_e \rangle [10^{19}$m$^{-3}$]')
ax[0, 1].axes.get_xaxis().set_visible(False)

ax[1, 1].plot(GasP1.getDimensionAt().data(),
              GasP1.data(), 'k', label='Valves # 1')
ax[1, 1].plot(GasP2.getDimensionAt().data(),
              GasP2.data(), 'r', label='Valves # 2')
ax[1, 1].plot(GasP3.getDimensionAt().data(),
              GasP3.data(), 'b', label='Valves # 3')
ax[1, 1].legend(loc='best', numpoints=1, frameon=False)
ax[1, 1].set_ylabel(r'')
ax[1, 1].axes.get_xaxis().set_visible(False)

try:
    ax[2, 1].plot(Target.t2, Target.TotalSpIonFlux()/1e17)
    ax[2, 1].set_ylabel(r'Total Ion Flux [s$^{-1}$]')
    ax[2, 1].set_xlim([0, 2])
    ax[2, 1].axes.get_xaxis().set_visible(False)
except:
    pass
ax[3, 1].imshow(profS.transpose(), aspect='auto', origin='lower',
                vmin=5e18, vmax=1.5e19, extent=(times.min(), times.max(),
                                                rhoFake.min(), rhoFake.max()))
ax[3, 1].set_ylim([0.98, 1.06])
ax[3, 1].set_ylabel(r'$\rho$')
ax[3, 1].set_xlim([0, 2])

# create the panel for the equilibrium
R, Z = zip(Tree.getNode(
    r'\results::langmuir:pos').data())
R = np.asarray(R).ravel()
Z = np.asarray(Z).ravel()

tilesP, vesselP = eq.getMachineCrossSectionPatch()
psiS1ax = mpl.pylab.axes([0.75, 0.55, 0.25, 0.4])
tS1 = 1
arg0 = np.nanargmin(np.abs(tPsi-tS1))
psiS1ax.contour(rGrid, zGrid, -psi[arg0], 50, colors='k', linewidths=0.7)
psiS1ax.add_patch(tilesP)
psiS1ax.add_patch(vesselP)
psiS1ax.set_title(r'Shot # ' + str(shot) + ' @ ' +
                  "%1.3f" % tS1)
psiS1ax.set_aspect('equal')
psiS1ax.plot(R, Z, '.', ms=10)
# now we can try to have the profiles from thomson averaging
# at least 200 ms around 0.6 and around 1.3 s
# now check the existence of the processed profile
try:
    doubleP = pd.read_table('/home/tsui/idl/library/data/double/dpm' +
                            str(int(shot))+'_1.tab', skiprows=1,
                            header=0)
    t0 = (doubleP['Time(s)'].mean())
except:
    doubleP = None
    t0 = 0.7

_idx = np.argmin(np.abs(times - t0))
thPlot = mpl.pylab.axes([0.75, 0.1, 0.2, 0.4])

rho = eq.rz2psinorm(rPos, zPos, times[_idx], sqrt=True)
thPlot.plot(rho[dataEn[:, _idx] != -1],
            dataEn[dataEn[:, _idx] != -1, _idx]/1e19,
            'o', color='orange', markersize=10, label=r't @ 0.7')
thPlot.plot(rhoFake, profS[_idx, :]/1e19, '--', color='orange')
if doubleP is not None:
    # just keep the insertion of the probe
    xO = doubleP['rrsep(m)'][:np.argmin(doubleP['rrsep(m)'])].values
    yO = doubleP['Dens(m-3)'][
        :np.argmin(doubleP['rrsep(m)'])].values/1e19
    eRO = doubleP['DensErr(cm-3)'][
        :np.argmin(doubleP['rrsep(m)'])].values/10.
    Profile = FastRP._getprofileR(xO, yO, npoint=35)
    x = Profile.rho.values
    y = Profile.values
    eR = Profile.err
    # now transform from RRsep to Rho using eqtools
    rhoProbe = eq.rmid2psinorm(x+eq.getRmidOutSpline()(
        doubleP['Time(s)'].mean()),
                               doubleP['Time(s)'].mean(), sqrt=True)
    thPlot.plot(rhoProbe, y, 's', markersize=10, color='orange')
    thPlot.errorbar(rhoProbe, y, yerr=eR, fmt='none', ecolor='orange')

try:
    doubleP = pd.read_table('/home/tsui/idl/library/data/double/dpm' +
                            str(int(shot))+'_2.tab', skiprows=1,
                            header=0)
    t0 = (doubleP['Time(s)'].mean())
except:
    doubleP = None
    t0 = 1.3

_ii2 = np.argmin(np.abs(times-t0))
rho2 = eq.rz2psinorm(rPos, zPos, times[_ii2], sqrt=True)
thPlot.plot(rho2[dataEn[:, _ii2] != -1],
            dataEn[dataEn[:, _ii2] != -1, _ii2]/1e19,
            'o', color='blue', markersize=10, label=r't @ 01.3')
thPlot.set_xlabel(r'$\rho$', fontsize=18)
thPlot.set_ylabel(r'n$_e$ [10$^{19}$m$^{-3}$]', fontsize=18)
thPlot.set_xlim([0.9, 1.06])
thPlot.legend(loc='best', numpoints=1, frameon=False)
thPlot.set_yscale('log')
thPlot.plot(rhoFake, profS[_ii2, :]/1e19, '--', color='blue')
if doubleP is not None:
    # just keep the insertion of the probe
    xO = doubleP['rrsep(m)'][:np.argmin(doubleP['rrsep(m)'])].values
    yO = doubleP['Dens(m-3)'][
        :np.argmin(doubleP['rrsep(m)'])].values/1e19
    eRO = doubleP['DensErr(cm-3)'][
        :np.argmin(doubleP['rrsep(m)'])].values/10.
    Profile = FastRP._getprofileR(xO, yO, npoint=35)
    x = Profile.rho.values
    y = Profile.values
    eR = Profile.err
    # now transform from RRsep to Rho using eqtools
    rhoProbe = eq.rmid2psinorm(x+eq.getRmidOutSpline()(
        doubleP['Time(s)'].mean()),
                               doubleP['Time(s)'].mean(), sqrt=True)
    thPlot.plot(rhoProbe, y, 's', markersize=10, color='blue')
    thPlot.errorbar(rhoProbe, y, yerr=eR, fmt='none', ecolor='blue')
mpl.pylab.show()


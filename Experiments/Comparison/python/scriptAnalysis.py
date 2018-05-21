from __future__ import print_function
import numpy as np
import matplotlib as mpl
import xarray as xray
import h5py
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import hilbert
import pandas as pd
from os.path import expanduser
from lmfit.models import GaussianModel
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc("font", size=18)
mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
mpl.rc("lines", linewidth=2)
homeDir = expanduser("~")
EuroFusionDir = homeDir + "/Dropbox/Work/Collaborations/EUROFusion/"
Data = pd.read_csv(
    EuroFusionDir +
    'MST1-2017-2018/Topic-21/Repository/Experiments/TCV/data/BlobDatabase.csv')
DataTCV = Data[((Data['Rho'] > 1.025) &
                (Data['Rho'] <= 1.05) &
                (Data['Conf'] == 'LSN') &
                (Data['Lambda Div'] < 40) &
                (Data['Blob Size [rhos]'] > 1))]
# limit the otlier of the poloidal velocity
DataTCV = DataTCV[DataTCV['vP'].abs() <= DataTCV['vP'].std()]


def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("1. Target density and radiation vs Density Constant Bt")
    print("2. Target density and radiation vs Greenwald Fraction Constant Bt")
    print("3. Target density and radiation vs Density Constant q95")
    print("4. Target density and radiation vs Greenwald constant q95")
    print('5. Upstream and target profiles Constant Bt')
    print('6. Upstream and target profiles Constant q95')
    print('7. Example of shoulder amplitude')
    print('8. Statistics at constant Bt')
    print('9. Statistics at constant q95')
    print('10. Shoulder and Target density evolution Constant Bt')
    print('11. Shoulder and Target density evolution Constant q95')
    print("12. Efold vs Lambda same fueling with/wo cryopumps")
    print("13. Efold vs Lambda different fueling with/wo cryopumps")
    print("14. Efold vs Lambda H-Mode/L-Mode")
    print("15. CAS example combined")
    print("99: End")
    print(67 * "-")


loop = True

while loop:
    print_menu()
    selection = input("Enter your choice [1-99] ")
    if selection == 1:
        shotAug = (34105, 34102, 34106)
        iPAug = (0.6, 0.8, 1)
        shotTcv = (57437, 57425, 57497)
        iPTcv = (.19, 0.245, 0.33)
        # now the plot target peak density and radiation vs density
        fig, ax = mpl.pylab.subplots(figsize=(10, 7), nrows=2, ncols=2)
        fig.subplots_adjust(wspace=0.25, hspace=0.05,
                            bottom=0.15, left=0.15, right=0.98)
        colorList = ('#01406C', '#F03C07', '#28B799')
        for i, (shot, ip, col) in enumerate(zip(shotAug, iPAug, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../AUG/analysis/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            ax[0, 0].plot(df.sel(sig='H-5')[:-10]/10,
                          df.sel(sig='neTarget')[:-10], '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
            ax[1, 0].plot(df.sel(sig='H-5')[:-10]/10,
                          df.sel(sig='D10')[:-10]/1e3, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
        ax[0, 0].set_xlim([0.1, 0.5])
        ax[0, 0].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 0].axes.get_xaxis().set_visible(False)
        ax[0, 0].set_title(r'AUG')
        ax[1, 0].set_xlim([0.1, 0.5])
        ax[1, 0].set_ylabel(r'kw/m$^2$')
        ax[1, 0].set_xlabel(r'n$_e^{Edge}[10^{20}$m$^{-2}]$')
        leg = ax[0, 0].legend(loc='best', numpoints=1,
                              frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        for i, (shot, ip, col) in enumerate(zip(shotTcv, iPTcv, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../TCV/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            ax[0, 1].plot(df.sel(sig='en'),
                          df.sel(sig='neMaxTarget')/10, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
            ax[1, 1].plot(df.sel(sig='en'),
                          df.sel(sig='Bolo')/1e3, '.',  color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
        ax[0, 1].set_xlim([0.1, 1.2])
        ax[0, 1].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_title(r'TCV')
        ax[1, 1].set_xlim([0.1, 1.2])
        ax[1, 1].set_ylabel(r'kW/m$^{2}$')
        ax[1, 1].set_xlabel(r'$\langle$n$_e\rangle[10^{20}$m$^{-3}]$')
        ax[1, 1].set_ylim([0, 100])
        leg = ax[0, 1].legend(loc='best', numpoints=1,
                              frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)
        fig.savefig('../pdfbox/TargetDensityRadiationVsDensityConstantBt.pdf',
                    bbox_to_inches='tight', dpi=300)
    elif selection == 2:
        shotAug = (34105, 34102, 34106)
        iPAug = (0.6, 0.8, 1)
        shotTcv = (57437, 57425, 57497)
        iPTcv = (.19, 0.245, 0.33)
        # now the plot target peak density and radiation vs density
        fig, ax = mpl.pylab.subplots(figsize=(10, 7), nrows=2, ncols=2)
        fig.subplots_adjust(wspace=0.25, hspace=0.06,
                            bottom=0.15, left=0.15, right=0.97)
        colorList = ('#01406C', '#F03C07', '#28B799')
        for i, (shot, ip, col) in enumerate(zip(shotAug, iPAug, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../AUG/analysis/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            ax[0, 0].plot(df.sel(sig='nGw')[:-10],
                          df.sel(sig='neTarget')[:-10], '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
            ax[1, 0].plot(df.sel(sig='nGw')[:-10],
                          df.sel(sig='D10')[:-10]/1e3, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
        ax[0, 0].set_xlim([0.1, 1])
        ax[0, 0].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 0].axes.get_xaxis().set_visible(False)
        ax[0, 0].set_title(r'AUG')
        ax[1, 0].set_xlim([0.1, 1])
        ax[1, 0].set_ylabel(r'kw/m$^2$')
        ax[1, 0].set_xlabel(r'n$_e$/n$_G$')
        leg = ax[0, 0].legend(loc='best', numpoints=1,
                              frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        for i, (shot, ip, col) in enumerate(zip(shotTcv, iPTcv, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../TCV/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            ax[0, 1].plot(df.sel(sig='n/nG'),
                          df.sel(sig='neMaxTarget')/10, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
            ax[1, 1].plot(df.sel(sig='n/nG'),
                          df.sel(sig='Bolo')/1e3, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
        ax[0, 1].set_xlim([0.1, 1.])
        ax[0, 1].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_title(r'TCV')
        ax[1, 1].set_xlim([0.1, 1.])
        ax[1, 1].set_ylabel(r'kW/m$^{2}$')
        ax[1, 1].set_xlabel(r'n$_e$/n$_G$')
        ax[1, 1].set_ylim([0, 100])
        leg = ax[0, 1].legend(loc='best', numpoints=1,
                              frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)
        fig.savefig(
            '../pdfbox/TargetDensityRadiationVsGreenwaldConstantBt.pdf',
            bbox_to_inches='tight', dpi=300)

    elif selection == 3:
        shotAug = (34103, 34102, 34104)
        iPAug = (0.6, 0.8, 1)
        shotTcv = (57461, 57454, 57497)
        iPTcv = (.19, 0.245, 0.33)
        # now the plot target peak density and radiation vs density
        fig, ax = mpl.pylab.subplots(figsize=(10, 7), nrows=2, ncols=2)
        fig.subplots_adjust(wspace=0.25, hspace=0.05,
                            bottom=0.15, left=0.15, right=0.98)
        colorList = ('#01406C', '#F03C07', '#28B799')
        for i, (shot, ip, col) in enumerate(zip(shotAug, iPAug, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../AUG/analysis/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            ax[0, 0].plot(df.sel(sig='H-5')[:-10]/10,
                          df.sel(sig='neTarget')[:-10], '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
            ax[1, 0].plot(df.sel(sig='H-5')[:-10]/10,
                          df.sel(sig='D10')[:-10]/1e3, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
        ax[0, 0].set_xlim([0.1, 0.5])
        ax[0, 0].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 0].axes.get_xaxis().set_visible(False)
        ax[0, 0].set_title(r'AUG')
        ax[1, 0].set_xlim([0.1, 0.5])
        ax[1, 0].set_ylabel(r'kw/m$^2$')
        ax[1, 0].set_xlabel(r'n$_e^{Edge}[10^{20}$m$^{-2}]$')
        leg = ax[0, 0].legend(loc='best', numpoints=1,
                              frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        for i, (shot, ip, col) in enumerate(zip(shotTcv, iPTcv, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../TCV/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            ax[0, 1].plot(df.sel(sig='en'),
                          df.sel(sig='neMaxTarget')/10, '.',  color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
            ax[1, 1].plot(df.sel(sig='en'),
                          df.sel(sig='Bolo')/1e3, '.',  color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                          rasterized=True)
        ax[0, 1].set_xlim([0.1, 1.2])
        ax[0, 1].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_title(r'TCV')
        ax[1, 1].set_xlim([0.1, 1.2])
        ax[1, 1].set_ylabel(r'kW/m$^{2}$')
        ax[1, 1].set_xlabel(r'$\langle$n$_e\rangle[10^{20}$m$^{-3}]$')
        ax[1, 1].set_ylim([0, 100])
        leg = ax[0, 1].legend(loc='best', numpoints=1,
                              frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)
        fig.savefig('../pdfbox/TargetDensityRadiationVsDensityConstantQ95.pdf',
                    bbox_to_inches='tight', dpi=300)

    elif selection == 4:
        shotAug = (34103, 34102, 34104)
        iPAug = (0.6, 0.8, 1)
        shotTcv = (57461, 57454, 57497)
        iPTcv = (.19, 0.245, 0.33)
        # now the plot target peak density and radiation vs density
        fig, ax = mpl.pylab.subplots(figsize=(10, 7), nrows=2, ncols=2)
        fig.subplots_adjust(wspace=0.25, hspace=0.06,
                            bottom=0.15, left=0.15, right=0.97)
        colorList = ('#01406C', '#F03C07', '#28B799')
        for i, (shot, ip, col) in enumerate(zip(shotAug, iPAug, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../AUG/analysis/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            ax[0, 0].plot(df.sel(sig='nGw')[:-10],
                          df.sel(sig='neTarget')[:-10], '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1, 0].plot(df.sel(sig='nGw')[:-10],
                          df.sel(sig='D10')[:-10]/1e3, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
        ax[0, 0].set_xlim([0.1, 1])
        ax[0, 0].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 0].axes.get_xaxis().set_visible(False)
        ax[0, 0].set_title(r'AUG')
        ax[1, 0].set_xlim([0.1, 1])
        ax[1, 0].set_ylabel(r'kw/m$^2$')
        ax[1, 0].set_xlabel(r'n$_e$/n$_G$')
        leg = ax[0, 0].legend(loc='best', numpoints=1,
                              frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        for i, (shot, ip, col) in enumerate(zip(shotTcv, iPTcv, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../TCV/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            ax[0, 1].plot(df.sel(sig='n/nG'),
                          df.sel(sig='neMaxTarget')/10, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1, 1].plot(df.sel(sig='n/nG'),
                          df.sel(sig='Bolo')/1e3, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
        ax[0, 1].set_xlim([0.1, 1.])
        ax[0, 1].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_title(r'TCV')
        ax[1, 1].set_xlim([0.1, 1.])
        ax[1, 1].set_ylabel(r'kW/m$^{2}$')
        ax[1, 1].set_xlabel(r'n$_e$/n$_G$')
        ax[1, 1].set_ylim([0, 100])
        leg = ax[0, 1].legend(loc='best', numpoints=1,
                              frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)
        fig.savefig(
            '../pdfbox/TargetDensityRadiationVsGreenwaldConstantQ95.pdf',
            bbox_to_inches='tight')

    elif selection == 5:
        # open the appropriate figure
        fig, ax = mpl.pylab.subplots(
            figsize=(11, 12), nrows=3, ncols=2,
            sharex=True)
        fig.subplots_adjust(wspace=0.25, top=0.95, bottom=0.1, right=0.99)
        # shotlist for AUGD
        shotListA = (34105, 34102, 34106)
        tListA = (2.83, 2.83, 2.34)
        # shotlist for TCV
        shotListT = (57089, 57088, 52062)
        # colorList
        colorList = ('#01406C', '#F03C07', '#28B799')
        # current list AUGD
        ipListA = ('0.6', '0.8', '1')
        ipListT = ('0.185', '0.24', '0.34')
        for shot, t, c, _ip in zip(shotListA, tListA, colorList, ipListA):
            # open the HDF file
            DirectoryAug = homeDir+"/Documents/" \
                "Fisica/Conferences/IAEA/iaea2018/data/aug/"
            File = h5py.File(DirectoryAug + 'Shot%5i' % shot + '.h5', 'r')
            neLBtime = File['timeLiB'].value
            # limit to the interval
            _idx = np.where((neLBtime >= t-0.02) & (neLBtime <= t+0.02))[0]
            _duMean = np.nanmean(File['LiB'].value[_idx, :], axis=0)
            _duStd = np.nanstd(File['LiB'].value[_idx, :], axis=0)
            _idx = np.where((File['enTime'].value >= t-0.02) &
                            (File['enTime'].value <= t+0.02))[0]
            enLabel = np.nanmean(File['en'].value[_idx])/1e19
            _idx = np.where((File['enGTime'].value >= t-0.02) &
                            (File['enGTime'].value <= t+0.02))[0]
            nng = np.nanmean(File['enG'].value[_idx])
            # spline for normalization
            S = UnivariateSpline(File['rhoLiB'].value, _duMean, s=0)
            # upstream profile normalized at the separatrix
            ax[0, 0].plot(File['rhoLiB'].value, _duMean/S(1), color=c, lw=3,
                          label=r'$\overline{n_e}$ = %3.2f' % (enLabel/10) +
                          ' I$_p$ = ' + str(_ip) +
                          r' MA')
            ax[0, 0].fill_between(File['rhoLiB'].value,
                                  (_duMean-_duStd)/S(1),
                                  (_duMean+_duStd)/S(1),
                                  facecolor=c, edgecolor='none',
                                  alpha=0.5)
            # now the target profiles
            _idx = np.where((File['timeTarget'].value > t-0.01) &
                            (File['timeTarget'].value < t+0.01))[0]
            xx = File['rhoTarget'].value[:, _idx]
            yy = File['neTarget'].value[:, _idx]/1e19
            ax[1, 0].errorbar(
                np.nanmean(xx, axis=1)[np.argsort(np.nanmean(xx, axis=1))],
                np.nanmean(yy, axis=1)[
                    np.argsort(np.nanmean(xx, axis=1))],
                fmt='-o', ms=12,
                color=c, alpha=0.7, yerr=np.nanstd(yy, axis=1),
                label=r'n/n$_G$ = %3.2f' % nng)

            # now the values of Lambda
            _idx = np.where(
                (File['LambdaDivTime'].value >= t-0.03) &
                (File['LambdaDivTime'].value <= t+0.03))[0]
            ax[2, 0].errorbar(
                File['LambdaDivRho'].value,
                np.nanmean(File['LambdaDiv'].value[:, _idx], axis=1),
                yerr=np.nanstd(File['LambdaDiv'].value[:, _idx]*3, axis=1),
                fmt='-', lw=3, color=c, alpha=0.5)
            File.close()
        ax[2, 0].set_xlim([0.96, 1.06])
        ax[1, 0].set_xlim([0.96, 1.06])
        ax[0, 0].set_xlim([0.96, 1.06])
        ax[2, 0].set_xlabel(r'$\rho$')
        ax[2, 0].set_ylabel(r'$\Lambda_{div}$')
        ax[2, 0].set_yscale('log')
        ax[2, 0].axhline(1, ls='--', color='grey', lw=2)
        ax[2, 0].set_ylim([5e-2, 30])
        ax[1, 0].axes.get_xaxis().set_visible(False)
        ax[1, 0].set_ylabel(r'n$_e^{t} [10^{19}$m$^{-3}$]')
        ax[0, 0].axes.get_xaxis().set_visible(False)
        ax[0, 0].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
        ax[0, 0].set_title('AUG constant B$_t$')
        ax[0, 0].set_yscale('log')
        ax[0, 0].set_ylim([0.05, 3])
        leg = ax[0, 0].legend(loc='best',
                              numpoints=1, frameon=False,
                              fontsize=14)
        for handle, text in zip(leg.legendHandles, leg.get_texts()):
            text.set_color(handle.get_color())
            handle.set_visible(False)
        leg = ax[1, 0].legend(loc='best',
                              numpoints=1, frameon=False,
                              fontsize=14)
        for handle, text, c in zip(
                leg.legendHandles, leg.get_texts(), colorList):
            text.set_color(c)
            handle.set_visible(False)
        # Now TCV
        DirectoryTcv = homeDir + "/Documents/Fisica/Conferences/" \
            "IAEA/iaea2018/data/tcv/"
        File = h5py.File(DirectoryTcv + 'ProfilesConstantBt.h5', 'r')
        for shot, col, _ip in zip(
                shotListT, colorList, ipListT):
            Up = File['%5i/Upstream' % shot]
            _norm = Up['yFit'].value[np.argmin(np.abs(Up['xFit'].value - 1))]
            ax[0, 1].errorbar(Up['enX'].value,
                              Up['enY'].value/_norm,
                              xerr=Up['enEx'].value,
                              yerr=Up['enEy'].value/_norm,
                              fmt='o', ms=11, color=col, alpha=0.5,
                              label=r'$\overline{n}_e$ = 0.55, I$_p$ = ' +
                              str(_ip)+' MA')
            ax[0, 1].plot(Up['xFit'].value,
                          Up['yFit'].value/_norm, '-', color=col, lw=3)
            ax[0, 1].fill_between(Up['xFit'].value,
                                  (Up['yFit'].value-Up['eFit'].value)/_norm,
                                  (Up['yFit'].value+Up['eFit'].value)/_norm,
                                  facecolor=col,
                                  alpha=0.3, edgecolor='white')
            # Target
            nng = 0.55/(float(_ip)/(np.pi*0.25**2))

            Target = File['%5i/Target' % shot]
            ax[1, 1].plot(Target['enX'].value, Target['enY'].value, 'o',
                          ms=5, color=col, alpha=0.2,
                          label=r'n/n$_G$ = % 3.2f' % nng)
            ax[1, 1].plot(Target['xFit'].value, Target['yFit'].value, '-',
                          lw=3, color=col,)
            ax[1, 1].fill_between(Target['xFit'].value,
                                  (Target['yFit'].value-Target['eFit'].value),
                                  (Target['yFit'].value+Target['eFit'].value),
                                  color=col, alpha=0.3)
            # Lambda
            ax[2, 1].errorbar(
                Target['xLambda'].value,
                Target['yLambda'].value, yerr=Target['eLambda'].value,
                fmt='-',
                lw=3, color=col, alpha=0.5)
        File.close()
        ax[2, 1].set_xlim([0.96, 1.06])
        ax[1, 1].set_xlim([0.96, 1.06])
        ax[0, 1].set_xlim([0.96, 1.06])
        ax[2, 1].set_xlabel(r'$\rho$')
        ax[2, 1].set_ylabel(r'$\Lambda_{div}$')
        ax[2, 1].set_yscale('log')
        ax[2, 1].set_ylim([5e-2, 30])
        ax[2, 1].axhline(1, ls='--', color='grey', lw=2)
        ax[1, 1].axes.get_xaxis().set_visible(False)
        ax[1, 1].set_ylabel(r'n$_e^{t} [10^{19}$m$^{-3}$]')
        ax[1, 1].set_ylim([-0.1, 2])
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
        ax[0, 1].set_title('TCV constant B$_t$')
        ax[0, 1].set_yscale('log')
        ax[0, 1].set_ylim([0.05, 3])
        leg = ax[0, 1].legend(loc='best',
                              numpoints=1, frameon=False,
                              fontsize=14)
        for handle, text, color in zip(
                leg.legendHandles, leg.get_texts(), colorList):
            text.set_color(color)
            handle.set_visible(False)
        leg = ax[1, 1].legend(loc='best',
                              numpoints=1, frameon=False,
                              fontsize=14)
        for handle, text, color in zip(
                leg.legendHandles, leg.get_texts(), colorList):
            text.set_color(color)
            handle.set_visible(False)

        fig.savefig('../pdfbox/UpstreamTargetProfilesConstantBt.pdf',
                    bbox_to_inches='tight')

    elif selection == 6:
        fig, ax = mpl.pylab.subplots(
            figsize=(11, 12), nrows=3, ncols=2,
            sharex=True)
        fig.subplots_adjust(wspace=0.25, top=0.95, bottom=0.1, right=0.99)
        # shotlist for AUGD
        shotListA = (34103, 34102, 34104)
        tListA = (2.9, 3.03, 2.44)
        # shotlist for TCV
        shotListT = (57454, 57497)
        # colorList
        colorList = ('#01406C', '#F03C07', '#28B799')
        # current list AUGD
        ipListA = ('0.6', '0.8', '1')
        ipListT = ('0.24', '0.34')
        for shot, t, c, _ip in zip(shotListA, tListA, colorList, ipListA):
            # open the HDF file
            DirectoryAug = '/Users/vianello/Documents/Fisica/Conferences/IAEA/iaea2018/data/aug/'
            File = h5py.File(DirectoryAug + 'Shot%5i' % shot + '.h5', 'r')
            neLBtime = File['timeLiB'].value
            # limit to the interval
            _idx = np.where((neLBtime >= t-0.02) & (neLBtime <= t+0.02))[0]
            _duMean = np.nanmean(File['LiB'].value[_idx, :], axis=0)
            _duStd = np.nanstd(File['LiB'].value[_idx, :], axis=0)
            _idx = np.where((File['enTime'].value >= t-0.02) &
                            (File['enTime'].value <= t+0.02))[0]
            enLabel = np.nanmean(File['en'].value[_idx])/1e19
            _idx = np.where((File['enGTime'].value >= t-0.02) &
                            (File['enGTime'].value <= t+0.02))[0]
            nng = np.nanmean(File['enG'].value[_idx])
            # spline for normalization
            S = UnivariateSpline(File['rhoLiB'].value, _duMean, s=0)
            # upstream profile normalized at the separatrix
            ax[0, 0].plot(File['rhoLiB'].value, _duMean/S(1), color=c, lw=3,
                          label=r'$\overline{n_e}$ = %3.2f' % (enLabel/10) +
                          ' I$_p$ = ' + str(_ip) +
                          r' MA')
            ax[0, 0].fill_between(File['rhoLiB'].value,
                                  (_duMean-_duStd)/S(1),
                                  (_duMean+_duStd)/S(1),
                                  facecolor=c, edgecolor='none',
                                  alpha=0.5)
            # now the target profiles
            _idx = np.where((File['timeTarget'].value > t-0.01) &
                            (File['timeTarget'].value < t+0.01))[0]
            xx = File['rhoTarget'].value[:, _idx]
            yy = File['neTarget'].value[:, _idx]/1e19
            ax[1, 0].errorbar(
                np.nanmean(xx, axis=1)[np.argsort(np.nanmean(xx, axis=1))],
                np.nanmean(yy, axis=1)[
                    np.argsort(np.nanmean(xx, axis=1))],
                fmt='-o', ms=12,
                color=c, alpha=0.7, yerr=np.nanstd(yy, axis=1),
                label=r'n/n$_G$ = %3.2f' % nng)

            # now the values of Lambda
            _idx = np.where(
                (File['LambdaDivTime'].value >= t-0.03) &
                (File['LambdaDivTime'].value <= t+0.03))[0]
            ax[2, 0].errorbar(
                File['LambdaDivRho'].value,
                np.nanmean(File['LambdaDiv'].value[:, _idx], axis=1),
                yerr=np.nanstd(File['LambdaDiv'].value[:, _idx]*3, axis=1),
                fmt='-', lw=3, color=c, alpha=0.5)
            File.close()
        ax[2, 0].set_xlim([0.96, 1.06])
        ax[1, 0].set_xlim([0.96, 1.06])
        ax[0, 0].set_xlim([0.96, 1.06])
        ax[2, 0].set_xlabel(r'$\rho$')
        ax[2, 0].set_ylabel(r'$\Lambda_{div}$')
        ax[2, 0].set_yscale('log')
        ax[2, 0].axhline(1, ls='--', color='grey', lw=2)
        ax[2, 0].set_ylim([5e-2, 30])
        ax[1, 0].axes.get_xaxis().set_visible(False)
        ax[1, 0].set_ylabel(r'n$_e^{t} [10^{19}$m$^{-3}$]')
        ax[0, 0].axes.get_xaxis().set_visible(False)
        ax[0, 0].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
        ax[0, 0].set_title('AUG constant q$_{95}$')
        ax[0, 0].set_yscale('log')
        ax[0, 0].set_ylim([0.05, 3])
        ax[1, 0].set_ylim([-0.05, 4.5])
        leg = ax[0, 0].legend(loc='best',
                              numpoints=1, frameon=False,
                              fontsize=14)
        for handle, text in zip(leg.legendHandles, leg.get_texts()):
            text.set_color(handle.get_color())
            handle.set_visible(False)
        leg = ax[1, 0].legend(loc='best',
                              numpoints=1, frameon=False,
                              fontsize=14)
        for handle, text, c in zip(
                leg.legendHandles, leg.get_texts(), colorList):
            text.set_color(c)
            handle.set_visible(False)
        # Now TCV
        DirectoryTcv = "/Users/vianello/Documents/Fisica/Conferences/" \
                       "IAEA/iaea2018/data/tcv/"
        File = h5py.File(DirectoryTcv + 'ProfilesConstantQ95.h5', 'r')
        for shot, col, _ip, en, nng in zip(
                shotListT, colorList, ipListT,
                (9.81, 9.27), (0.8, 0.54)):
            Up = File['%5i/Plunge2/Upstream' % shot]
            _norm = Up['yFit'].value[np.argmin(np.abs(Up['xFit'].value - 1))]
            ax[0, 1].errorbar(Up['enX'].value,
                              Up['enY'].value/_norm,
                              xerr=Up['enEx'].value,
                              yerr=Up['enEy'].value/_norm,
                              fmt='o', ms=11, color=col, alpha=0.5,
                              label=r'$\overline{n}_e$ = %3.2f' % (en/10.) +
                              ', I$_p$ = ' + str(_ip)+' MA')
            ax[0, 1].plot(Up['xFit'].value,
                          Up['yFit'].value/_norm, '-', color=col, lw=3)
            ax[0, 1].fill_between(Up['xFit'].value,
                                  (Up['yFit'].value-Up['eFit'].value)/_norm,
                                  (Up['yFit'].value+Up['eFit'].value)/_norm,
                                  facecolor=col,
                                  alpha=0.3, edgecolor='white')
            # Target
            Target = File['%5i/Plunge2/Target' % shot]
            ax[1, 1].plot(Target['enX'].value, Target['enY'].value, 'o',
                          ms=5, color=col, alpha=0.2,
                          label=r'n/n$_G$ = % 3.2f' % nng)
            ax[1, 1].plot(Target['xFit'].value, Target['yFit'].value, '-',
                          lw=3, color=col,)
            ax[1, 1].fill_between(Target['xFit'].value,
                                  (Target['yFit'].value-Target['eFit'].value),
                                  (Target['yFit'].value+Target['eFit'].value),
                                  color=col, alpha=0.3)
            # Lambda
            ax[2, 1].errorbar(
                Target['xLambda'].value,
                Target['yLambda'].value, yerr=Target['eLambda'].value,
                fmt='-',
                lw=3, color=col, alpha=0.5)
        File.close()
        ax[2, 1].set_xlim([0.96, 1.06])
        ax[1, 1].set_xlim([0.96, 1.06])
        ax[0, 1].set_xlim([0.96, 1.06])
        ax[2, 1].set_xlabel(r'$\rho$')
        ax[2, 1].set_ylabel(r'$\Lambda_{div}$')
        ax[2, 1].set_yscale('log')
        ax[2, 1].set_ylim([5e-2, 30])
        ax[2, 1].axhline(1, ls='--', color='grey', lw=2)
        ax[1, 1].axes.get_xaxis().set_visible(False)
        ax[1, 1].set_ylabel(r'n$_e^{t} [10^{19}$m$^{-3}$]')
        ax[1, 1].set_ylim([-0.1, 3])
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
        ax[0, 1].set_title('TCV constant q$_{95}$')
        ax[0, 1].set_yscale('log')
        ax[0, 1].set_ylim([-0.05, 4.5])
        leg = ax[0, 1].legend(loc='best',
                              numpoints=1, frameon=False,
                              fontsize=14)
        for handle, text, color in zip(
                leg.legendHandles, leg.get_texts(), colorList):
            text.set_color(color)
            handle.set_visible(False)
        leg = ax[1, 1].legend(loc='best',
                              numpoints=1, frameon=False,
                              fontsize=14)
        for handle, text, color in zip(
                leg.legendHandles, leg.get_texts(), colorList):
            text.set_color(color)
            handle.set_visible(False)

        fig.savefig('../pdfbox/UpstreamTargetProfilesConstantQ95.pdf',
                    bbox_to_inches='tight')
    elif selection == 7:
        DirectoryAug = '/Users/vianello/Documents/Fisica/Conferences/IAEA/iaea2018/data/aug/'
        File = h5py.File(DirectoryAug + 'Shot34102.h5', 'r')
        LiBN = File['LiBNorm'].value
        rho = File['rhoLiB'].value
        time = File['timeLiB'].value
        _idx = np.where((time >= 1.2) & (time <= 1.3))[0]
        pNA = np.nanmean(LiBN[_idx, :], axis=0)
        eNA = np.nanstd(LiBN[_idx, :], axis=0)
        _idx = np.where((time >= 3.1) & (time <= 3.2))[0]
        pNB = np.nanmean(LiBN[_idx, :], axis=0)
        eNB = np.nanstd(LiBN[_idx, :], axis=0)
        fig, ax = mpl.pylab.subplots(figsize=(7, 7),
                                     nrows=2, ncols=1, sharex=True)
        fig.subplots_adjust(bottom=0.18, left=0.18, top=0.98, hspace=0.06)
        _idx = np.where(rho >= 0.995)[0]
        Amplitude = pNB-pNA
        errAmplitude = np.sqrt(np.power(eNB, 2) + np.power(eNA, 2))
        ax[0].plot(rho[_idx], pNA[_idx], '-', lw=4, color='#F0CA4D')
        ax[0].fill_between(rho[_idx], pNA[_idx]-eNA[_idx],
                           pNA[_idx]+eNA[_idx], color='#F0CA4D',
                           edgecolor='none',
                           alpha=0.2)
        ax[0].plot(rho[_idx], pNB[_idx], '-', lw=4, color='#F53855')
        ax[0].fill_between(rho[_idx], pNB[_idx]-eNB[_idx],
                           pNB[_idx]+eNB[_idx],
                           color='#F53855', edgecolor='none',
                           alpha=0.2)
        ax[0].fill_between(rho[_idx], pNA[_idx], pNB[_idx],
                           color='gray', edgecolor='gray', alpha=0.4)
        ax[0].set_xlim([1, 1.05])
        ax[0].set_ylim([1e-1, 2])
        ax[0].axes.get_xaxis().set_visible(False)

        ax[0].set_ylabel(r'n$_e$/n$_e(\rho=1)$')
        ax[0].set_yscale('log')

        ax[1].plot(rho[_idx], pNB[_idx]-pNA[_idx], 'k', lw=2)
        ax[1].fill_between(rho[_idx], Amplitude[_idx]-errAmplitude[_idx],
                           Amplitude[_idx]+errAmplitude[_idx], color='gray',
                           edgecolor='none', alpha=0.2)
        ax[1].set_xlabel(r'$\rho$')
        ax[1].set_ylabel(r'Amplitude')
        ax[1].set_ylim([0, 0.5])
        ax[1].axvline(1.03, ls='--', color='green', lw=2)
        ax[0].axvline(1.03, ls='--', color='green', lw=2)

        fig.savefig('../pdfbox/ExampleShoulderAmplitude.pdf',
                    bbox_to_inches='tight')

    elif selection == 8:
        shotListA = (34105, 34102, 34106)
        btAsdex = (0.6, 0.8, 1)
        colorList = ('#01406C', '#F03C07', '#28B799')
        Directory = "/Users/vianello/Desktop/Topic-21/" \
                    "Experiments/AUG/analysis/data/"
        # this is the figure of Blob-size vs Lambda
        fig, ax = mpl.pylab.subplots(figsize=(8, 8), nrows=2, ncols=1,
                                     sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.15, left=0.17, hspace=0.05, top=0.97)
        # this is the figure of Efold vs Lambda
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 8), nrows=2, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(bottom=0.15, left=0.17, hspace=0.05, top=0.97)
        # this is the figure of Efold vs Blob-size
        fig3, ax3 = mpl.pylab.subplots(figsize=(8, 8), nrows=2, ncols=1,
                                       sharex=True)
        fig3.subplots_adjust(bottom=0.15, left=0.17, hspace=0.05, top=0.97)
        # this is the cycle throug AUG shots
        # first the plot for AUG
        for shot, col in zip(shotListA, colorList):
            for s in ('1', '2', '3', '4', '5'):
                try:
                    Data = xray.open_dataarray(Directory +
                                               'Shot%5i' % shot +
                                               '_'+s+'Stroke.nc')
                    Vperp = Data.Vperp
                    Size = 0.5*Data.TauB*np.abs(Vperp)/1e-4
                    SizeErr = np.sqrt(
                        np.power(Data.Vperp*Data.TauBErr, 2) +
                        np.power(Data.TauB*Data.VperpErr, 2))/2/1e-4
                    if SizeErr > 0.5*Size:
                        SizeErr = 0.5*Size
                    _idx = np.where((Data.rhoLambda >= Data.Rho-0.03) &
                                    (Data.rhoLambda <= Data.Rho+0.03))[0]
                    _idx2 = np.where((Data.rhoLiB >= Data.Rho-0.03) &
                                     (Data.rhoLiB <= Data.Rho+0.03))[0]
                    print(('Size for shot {} plunge {} is {} +/- {}').format(
                        shot, s, Size, SizeErr))
                    print(('Lambda for shot {} plunge {} is {} +/- {}').format(
                        shot, s,
                        np.nanmean(Data.LambdaProfile[_idx]),
                        np.nanstd(Data.LambdaProfile[_idx])))
                    print(('Efold for shot {} plunge {} is {} +/- {}').format(
                        shot, s,
                        np.nanmean(Data.Efold[_idx2])*1e2,
                        np.nanstd(Data.Efold[_idx2])*1e2))
                    # Blob vs Lambda
                    ax[0].errorbar(
                        np.nanmean(Data.LambdaProfile[_idx]), Size,
                        fmt='o', yerr=SizeErr,
                        xerr=np.nanstd(Data.LambdaProfile[_idx]),
                        ms=15, color=col, alpha=0.7, label='AUG')
                    # Efold Vs Lambda
                    ax2[0].errorbar(
                        np.nanmean(Data.LambdaProfile[_idx]),
                        np.nanmean(Data.Efold[_idx2])*1e2,
                        fmt='o', yerr=np.nanstd(Data.Efold[_idx2])*1e2,
                        xerr=np.nanstd(Data.LambdaProfile[_idx]),
                        ms=15, color=col, alpha=0.7, label='AUG')
                    # Efold vs Blob
                    ax3[0].errorbar(
                        Size, np.nanmean(Data.Efold[_idx2])*1e2,
                        xerr=SizeErr, yerr=np.nanstd(Data.Efold[_idx2])*1e2,
                        fmt='o', ms=15, color=col, alpha=0.7, label='AUG')
                except:
                    print(('not done for shot {} stroke {}').format(shot, s))
                    pass
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel(r'$\delta_b [\rho_s]$')
        ax[0].text(0.9, 0.87, 'AUG', transform=ax[0].transAxes)
        # Efold vs Lambda
        ax2[0].set_xscale('log')
        ax2[0].set_yscale('log')
        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylabel(r'$\lambda_n [$cm$]$')
        ax2[0].text(0.9, 0.87, 'AUG', transform=ax2[0].transAxes)
        ax2[0].set_ylim([1, 50])
        # Efold vs Blob-size
        ax3[0].set_xscale('log')
        ax3[0].set_yscale('log')
        ax3[0].axes.get_xaxis().set_visible(False)
        ax3[0].set_ylabel(r'$\lambda_n [$cm$]$')
        ax3[0].text(0.9, 0.87, 'AUG', transform=ax3[0].transAxes)
        ax3[0].set_ylim([1, 50])
        for ip, c, _idx in zip(btAsdex, colorList, range(len(btAsdex))):
            ax[0].text(0.05, 0.8 - 0.13 * _idx, r'I$_p$=%2.1f' % ip + ' MA',
                       color=c, transform=ax[0].transAxes)
            ax2[0].text(0.05, 0.8 - 0.13 * _idx, r'I$_p$=%2.1f' % ip + ' MA',
                        color=c, transform=ax2[0].transAxes)
            ax3[0].text(0.05, 0.8 - 0.13 * _idx, r'I$_p$=%2.1f' % ip + ' MA',
                        color=c, transform=ax3[0].transAxes)

        # now we group according to Ip also the data for TCV
        ranges = ((100, 200), (200, 300), (300, 400))
        DataUsed = DataTCV[DataTCV['Bt'].abs() > 1.41]
        for r, c in zip(ranges, colorList):
            _dummy = DataUsed[((DataUsed['Ip'].abs() >= r[0]) &
                               (DataUsed['Ip'].abs() < r[1]))]
            ax[1].errorbar(_dummy['Lambda Div'],
                           _dummy['Blob Size [rhos]']/2,
                           xerr=_dummy['Lambda Div Err'],
                           yerr=_dummy['Blob size Err [rhos]']/2, fmt='o',
                           ms=15, color=c, alpha=0.3)
            # Efold vs Lambda
            ax2[1].errorbar(_dummy['Lambda Div'],
                            _dummy['Efold']*1e2,
                            xerr=_dummy['Lambda Div Err'],
                            yerr=_dummy['EfoldErr']*1e2, fmt='o',
                            ms=15, color=c, alpha=0.3)
            # Efold vs Blob size
            ax3[1].errorbar(_dummy['Blob Size [rhos]']/2,
                            _dummy['Efold']*1e2,
                            xerr=_dummy['Blob size Err [rhos]']/2,
                            yerr=_dummy['EfoldErr']*1e2, fmt='o',
                            ms=15, color=c, alpha=0.3)

        ax[1].set_xscale('log')
        ax[1].set_xlim([0.01, 200])
        ax[1].set_ylabel(r'$\delta_b [\rho_s]$')
        ax[1].text(0.9, 0.87, 'TCV', transform=ax[1].transAxes)
        ax[1].set_xlabel(r'$\Lambda_{div}$')
        ax[1].set_ylim([1, 250])
        ax[1].set_yscale('log')
        # Efold vs Lambda
        ax2[1].set_xscale('log')
        ax2[1].set_xlim([0.01, 200])
        ax2[1].set_ylabel(r'$\lambda_n [$cm$]$')
        ax2[1].text(0.9, 0.87, 'TCV', transform=ax2[1].transAxes)
        ax2[1].set_xlabel(r'$\Lambda_{div}$')
        ax2[1].set_ylim([0.1, 200])
        ax2[1].set_yscale('log')

        ax3[1].set_xscale('log')
        ax3[1].set_xlim([1, 200])
        ax3[1].set_xlabel(r'$\delta_b [\rho_s]$')
        ax3[1].text(0.9, 0.87, 'TCV', transform=ax3[1].transAxes)
        ax3[1].set_ylabel(r'$\lambda_n [$cm$]$')
        ax3[1].set_ylim([0.1, 200])
        ax3[1].set_yscale('log')

        for r, c, _idx in zip(ranges, colorList, range(len(btAsdex))):
            ax[1].text(0.05, 0.8 - 0.13 * _idx,
                       str(r[0]) + r'$\leq $I$_p$ < %3i' % r[1] + ' kA',
                       color=c, transform=ax[1].transAxes)
            ax2[1].text(0.05, 0.8 - 0.13 * _idx,
                        str(r[0]) + r'$\leq $I$_p$ < %3i' % r[1] + ' kA',
                        color=c, transform=ax2[1].transAxes)
            ax3[1].text(0.6, 0.77 - 0.13 * _idx,
                        str(r[0]) + r'$\leq $I$_p$ < %3i' % r[1] + ' kA',
                        color=c, transform=ax3[1].transAxes)

        fig.savefig('../pdfbox/BlobLambdaConstantBt.pdf',
                    bbox_to_inchest='tight')
        fig2.savefig('../pdfbox/EfoldLambdaConstantBt.pdf',
                     bbox_to_inchest='tight')
        fig3.savefig('../pdfbox/EfoldBlobConstantBt.pdf',
                     bbox_to_inchest='tight')

    elif selection == 9:
        shotListA = (34103, 34102, 34104)
        btAsdex = (0.6, 0.8, 1)
        colorList = ('#01406C', '#F03C07', '#28B799')
        Directory = "/Users/vianello/Desktop/Topic-21/" \
                    "Experiments/AUG/analysis/data/"
        # this is the figure of Blob-size vs Lambda
        fig, ax = mpl.pylab.subplots(figsize=(8, 8), nrows=2, ncols=1,
                                     sharex=True, sharey=True)
        fig.subplots_adjust(bottom=0.15, left=0.17, hspace=0.05, top=0.97)
        # this is the figure of Efold vs Lambda
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 8), nrows=2, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(bottom=0.15, left=0.17, hspace=0.05, top=0.97)
        # this is the figure of Efold vs Blob-size
        fig3, ax3 = mpl.pylab.subplots(figsize=(8, 8), nrows=2, ncols=1,
                                       sharex=True)
        fig3.subplots_adjust(bottom=0.15, left=0.17, hspace=0.05, top=0.97)
        # this is the cycle throug AUG shots
        # first the plot for AUG
        for shot, col in zip(shotListA, colorList):
            for s in ('1', '2', '3', '4', '5'):
                try:
                    Data = xray.open_dataarray(Directory +
                                               'Shot%5i' % shot +
                                               '_'+s+'Stroke.nc')
                    Vperp = Data.Vperp
                    Size = 0.5*Data.TauB*np.abs(Vperp)/1e-4
                    SizeErr = np.sqrt(
                        np.power(Data.Vperp*Data.TauBErr, 2) +
                        np.power(Data.TauB*Data.VperpErr, 2))/2/1e-4
                    if SizeErr > 0.5*Size:
                        SizeErr = 0.5*Size
                    _idx = np.where((Data.rhoLambda >= Data.Rho-0.03) &
                                    (Data.rhoLambda <= Data.Rho+0.03))[0]
                    _idx2 = np.where((Data.rhoLiB >= Data.Rho-0.03) &
                                     (Data.rhoLiB <= Data.Rho+0.03))[0]
                    print(('Size for shot {} plunge {} is {} +/- {}').format(
                        shot, s, Size, SizeErr))
                    print(('Lambda for shot {} plunge {} is {} +/- {}').format(
                        shot, s,
                        np.nanmean(Data.LambdaProfile[_idx]),
                        np.nanstd(Data.LambdaProfile[_idx])))
                    print(('Efold for shot {} plunge {} is {} +/- {}').format(
                        shot, s,
                        np.nanmean(Data.Efold[_idx2])*1e2,
                        np.nanstd(Data.Efold[_idx2])*1e2))
                    # Blob vs Lambda
                    ax[0].errorbar(
                        np.nanmean(Data.LambdaProfile[_idx]), Size,
                        fmt='o', yerr=SizeErr,
                        xerr=np.nanstd(Data.LambdaProfile[_idx]),
                        ms=15, color=col, alpha=0.7, label='AUG')
                    # Efold Vs Lambda
                    ax2[0].errorbar(
                        np.nanmean(Data.LambdaProfile[_idx]),
                        np.nanmean(Data.Efold[_idx2])*1e2,
                        fmt='o', yerr=np.nanstd(Data.Efold[_idx2])*1e2,
                        xerr=np.nanstd(Data.LambdaProfile[_idx]),
                        ms=15, color=col, alpha=0.7, label='AUG')
                    # Efold vs Blob
                    ax3[0].errorbar(
                        Size, np.nanmean(Data.Efold[_idx2])*1e2,
                        xerr=SizeErr, yerr=np.nanstd(Data.Efold[_idx2])*1e2,
                        fmt='o', ms=15, color=col, alpha=0.7, label='AUG')
                except:
                    print(('not done for shot {} stroke {}').format(shot, s))
                    pass
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel(r'$\delta_b [\rho_s]$')
        ax[0].text(0.9, 0.87, 'AUG', transform=ax[0].transAxes)
        # Efold vs Lambda
        ax2[0].set_xscale('log')
        ax2[0].set_yscale('log')
        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylabel(r'$\lambda_n [$cm$]$')
        ax2[0].text(0.9, 0.87, 'AUG', transform=ax2[0].transAxes)
        ax2[0].set_ylim([1, 50])
        # Efold vs Blob-size
        ax3[0].set_xscale('log')
        ax3[0].set_yscale('log')
        ax3[0].axes.get_xaxis().set_visible(False)
        ax3[0].set_ylabel(r'$\lambda_n [$cm$]$')
        ax3[0].text(0.9, 0.87, 'AUG', transform=ax3[0].transAxes)
        ax3[0].set_ylim([1, 50])
        for ip, c, _idx in zip(btAsdex, colorList, range(len(btAsdex))):
            ax[0].text(0.05, 0.8 - 0.13 * _idx, r'I$_p$=%2.1f' % ip + ' MA',
                       color=c, transform=ax[0].transAxes)
            ax2[0].text(0.05, 0.8 - 0.13 * _idx, r'I$_p$=%2.1f' % ip + ' MA',
                        color=c, transform=ax2[0].transAxes)
            ax3[0].text(0.05, 0.8 - 0.13 * _idx, r'I$_p$=%2.1f' % ip + ' MA',
                        color=c, transform=ax3[0].transAxes)

        # now we group according to Ip also the data for TCV
        ranges = ((100, 200), (200, 300), (300, 400))
        shotList = [57450, 57454, 57459, 57461, 57497]
        DataUsed = DataTCV[DataTCV['Shots'].isin(shotList)]
        for r, c in zip(ranges, colorList):
            _dummy = DataUsed[((DataUsed['Ip'].abs() >= r[0]) &
                               (DataUsed['Ip'].abs() < r[1]))]
            ax[1].errorbar(_dummy['Lambda Div'],
                           _dummy['Blob Size [rhos]']/2,
                           xerr=_dummy['Lambda Div Err'],
                           yerr=_dummy['Blob size Err [rhos]']/2, fmt='o',
                           ms=15, color=c, alpha=0.3)
            # Efold vs Lambda
            if r[0] == 100:
                ax2[1].errorbar(_dummy['Lambda Div'],
                                _dummy['Efold GPR']*1e2,
                                xerr=_dummy['Lambda Div Err'],
                                yerr=_dummy['EfoldErr']*1e2, fmt='o',
                                ms=15, color=c, alpha=0.3)
                # Efold vs Blob size
                ax3[1].errorbar(_dummy['Blob Size [rhos]']/2,
                                _dummy['Efold GPR']*1e2,
                                xerr=_dummy['Blob size Err [rhos]']/2,
                                yerr=_dummy['EfoldErr']*1e2, fmt='o',
                                ms=15, color=c, alpha=0.3)
            else:
                ax2[1].errorbar(_dummy['Lambda Div'],
                                _dummy['Efold']*1e2,
                                xerr=_dummy['Lambda Div Err'],
                                yerr=_dummy['EfoldErr']*1e2, fmt='o',
                                ms=15, color=c, alpha=0.3)
                # Efold vs Blob size
                ax3[1].errorbar(_dummy['Blob Size [rhos]']/2,
                                _dummy['Efold']*1e2,
                                xerr=_dummy['Blob size Err [rhos]']/2,
                                yerr=_dummy['EfoldErr']*1e2, fmt='o',
                                ms=15, color=c, alpha=0.3)

        ax[1].set_xscale('log')
        ax[1].set_xlim([0.01, 200])
        ax[1].set_ylabel(r'$\delta_b [\rho_s]$')
        ax[1].text(0.9, 0.87, 'TCV', transform=ax[1].transAxes)
        ax[1].set_xlabel(r'$\Lambda_{div}$')
        ax[1].set_ylim([0.1, 250])
        ax[1].set_yscale('log')
        # Efold vs Lambda
        ax2[1].set_xscale('log')
        ax2[1].set_xlim([0.01, 200])
        ax2[1].set_ylabel(r'$\lambda_n [$cm$]$')
        ax2[1].text(0.9, 0.87, 'TCV', transform=ax2[1].transAxes)
        ax2[1].set_xlabel(r'$\Lambda_{div}$')
        ax2[1].set_ylim([0.1, 200])
        ax2[1].set_yscale('log')

        ax3[1].set_xscale('log')
        ax3[1].set_xlim([0.1, 200])
        ax3[1].set_xlabel(r'$\delta_b [\rho_s]$')
        ax3[1].text(0.9, 0.87, 'TCV', transform=ax3[1].transAxes)
        ax3[1].set_ylabel(r'$\lambda_n [$cm$]$')
        ax3[1].set_ylim([0.1, 200])
        ax3[1].set_yscale('log')

        for r, c, _idx in zip(ranges, colorList, range(len(btAsdex))):
            ax[1].text(0.05, 0.8 - 0.13 * _idx,
                       str(r[0]) + r'$\leq $I$_p$ < %3i' % r[1] + ' kA',
                       color=c, transform=ax[1].transAxes)
            ax2[1].text(0.05, 0.8 - 0.13 * _idx,
                        str(r[0]) + r'$\leq $I$_p$ < %3i' % r[1] + ' kA',
                        color=c, transform=ax2[1].transAxes)
            ax3[1].text(0.6, 0.77 - 0.13 * _idx,
                        str(r[0]) + r'$\leq $I$_p$ < %3i' % r[1] + ' kA',
                        color=c, transform=ax3[1].transAxes)

        fig.savefig('../pdfbox/BlobLambdaConstantQ95.pdf',
                    bbox_to_inchest='tight')
        fig2.savefig('../pdfbox/EfoldLambdaConstantQ95.pdf',
                     bbox_to_inchest='tight')
        fig3.savefig('../pdfbox/EfoldBlobConstantQ95.pdf',
                     bbox_to_inchest='tight')

    elif selection == 10:
        shotListA = (34105, 34102, 34106)
        colorList = ('#01406C', '#F03C07', '#28B799')
        iPAug = (0.6, 0.8, 1)
        DirectoryAug = "/Users/vianello/Documents/Fisica/"\
                       "Conferences/IAEA/iaea2018/data/aug/"
        # this is the plot with amplitude, target vs H-5
        fig, ax = mpl.pylab.subplots(figsize=(10, 9), nrows=3,
                                     ncols=1, sharex=True)
        fig.subplots_adjust(hspace=0.08,
                            bottom=0.15, left=0.15,
                            right=0.98)
        # this is the plot of the amplitude, target vs nGw
        fig2, ax2 = mpl.pylab.subplots(figsize=(10, 9),
                                       nrows=3, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(hspace=0.08,
                             bottom=0.15, left=0.15, right=0.98)
        # this is the plot if shoulder vs LambdaDiv
        fig3, ax3 = mpl.pylab.subplots(figsize=(10, 6),
                                       nrows=2, ncols=1, sharex=True)
        fig3.subplots_adjust(hspace=0.08,
                             bottom=0.15, left=0.15, right=0.98)

        for i, (shot, ip, col) in enumerate(zip(shotListA, iPAug, colorList)):
            df = xray.open_dataarray(
                '../../AUG/analysis/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            File = h5py.File(DirectoryAug + 'Shot{}.h5'.format(shot), 'r')
            LiBN = File['LiBNorm'].value
            rho = File['rhoLiB'].value
            time = File['timeLiB'].value
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.03 and 1.03<rho<1.06) wrt initial perio [1.2, 1.4]
            if shot == 34105:
                _trange = [0.7, 0.8]
            else:
                _trange = [1.2, 1.4]
            _idx = np.where((time >= _trange[0]) & (time <= _trange[1]))[0]
            Reference = np.nanmean(LiBN[_idx, :], axis=0)
            errReference = np.nanstd(LiBN[_idx, :], axis=0)
            LiBN = LiBN[np.where(time > _trange[1])[0], :]
            time = time[np.where(time > _trange[1])[0]]
            _npoint = int(np.floor((time.max()-time.min()))/0.02)
            Split = np.asarray(np.array_split(LiBN, _npoint, axis=0))
            Amplitude = np.asarray(
                [np.nanmean(p, axis=0)-Reference for p in Split])
            ErrAmplitude = np.asarray([
                np.sqrt(np.power(np.nanstd(p, axis=0), 2) +
                        np.power(errReference, 2)) for p in Split])
            _ = np.asarray(np.array_split(time, _npoint))
            timeSplit = np.asarray([np.mean(k) for k in _])
            aNear = np.nanmean(Amplitude[:, np.where(
                (rho >= 1) & (rho < 1.03))[0]], axis=1)
            aFar = np.nanmean(Amplitude[:, np.where(
                (rho >= 1.03) & (rho < 1.06))[0]], axis=1)
            aNearS = np.nanstd(Amplitude[:, np.where(
                (rho >= 1) & (rho < 1.03))[0]], axis=1)
            aFarS = np.nanstd(Amplitude[:, np.where(
                (rho >= 1.03) & (rho < 1.06))[0]], axis=1)
            # we now limit all the quantities in netcdf to
            # time greater than the minimum of
            # the timeSplit and interpolate
            _dummy = df.where((df.t >= timeSplit.min()) &
                              (df.t <= timeSplit.max()), drop=True)
            # this is the plot vs edge density
            S = interp1d(
                _dummy.t.values, _dummy.sel(sig='H-5').values,
                kind='linear', fill_value='extrapolate')
            ax[0].errorbar(S(timeSplit), aNear, yerr=aNearS,
                           fmt='o', color=col,
                           label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                           rasterized=True)
            ax[1].errorbar(S(timeSplit), aFar, yerr=aFarS, fmt='o', color=col,
                           label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                           rasterized=True)
            ax[2].plot(_dummy.sel(sig='H-5'),
                       _dummy.sel(sig='neTarget'), 'o', color=col,
                       label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                       rasterized=True)
            # this is the plot vs greenwald
            S = interp1d(
                _dummy.t.values, _dummy.sel(sig='nGw').values,
                kind='linear', fill_value='extrapolate')
            ax2[0].errorbar(S(timeSplit), aNear, yerr=aNearS,
                            fmt='o', color=col,
                            label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                            rasterized=True)
            ax2[1].errorbar(S(timeSplit), aFar, yerr=aFarS, fmt='o', color=col,
                            label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                            rasterized=True)
            ax2[2].plot(_dummy.sel(sig='nGw'),
                        _dummy.sel(sig='neTarget'), 'o', color=col,
                        label=r'#%5i' % shot + ' %3.2f' % ip + ' MA',
                        rasterized=True)

            # now load the Lambda Div from saved h5 file
            File = h5py.File(DirectoryAug+'Shot{}.h5'.format(shot))

            _idx = np.where((File['LambdaDivRho'].value >= 1) &
                            (File['LambdaDivRho'].value < 1.03))[0]
            LNear = np.nanmean(File['LambdaDiv'].value[_idx, :],
                               axis=0).ravel()
            _idx = np.where((File['LambdaDivRho'].value >= 1.03) &
                            (File['LambdaDivRho'].value < 1.06))[0]
            LFar = np.nanmean(File['LambdaDiv'].value[_idx, :], axis=0).ravel()
            LTime = File['LambdaDivTime'].value.ravel()
            LNear = LNear[np.where((LTime >= timeSplit.min()) &
                                   (LTime <= timeSplit.max()))[0]]
            LFar = LFar[np.where((LTime >= timeSplit.min()) &
                                 (LTime <= timeSplit.max()))[0]]
            LTime = LTime[np.where((LTime >= timeSplit.min()) &
                                   (LTime <= timeSplit.max()))[0]]
            if shot == 34104:
                LNear = LNear[np.where(LTime <= 4)[0]]
                LFar = LFar[np.where(LTime <= 4)[0]]
                LTime = LTime[np.where(LTime <= 4)[0]]

            S = interp1d(LTime, LNear, kind='linear', fill_value='extrapolate')
            ax3[0].errorbar(S(timeSplit[np.where(timeSplit <= 4.)[0]]),
                            aNear[np.where(timeSplit <= 4.)[0]],
                            yerr=aNearS[np.where(timeSplit <= 4.)[0]],
                            fmt='o', color=col,
                            label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            S = interp1d(LTime, LFar, kind='linear', fill_value='extrapolate')
            ax3[1].errorbar(S(timeSplit[np.where(timeSplit <= 4.)[0]]),
                            aFar[np.where(timeSplit <= 4.)[0]],
                            yerr=aFarS[np.where(timeSplit <= 4.)[0]],
                            fmt='o', color=col,
                            label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')

        ax[0].axes.get_xaxis().set_visible(False)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[0].set_ylim([0, 1])
        ax[1].set_ylim([0, 1])
        ax[2].set_ylim([0, 1])
        ax[2].set_xlim([0.5, 4])
        ax[2].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0].set_ylabel(r'Amp [a.u]')
        ax[0].text(0.5, 0.8, r'$1\leq \rho < 1.03$',
                   transform=ax[0].transAxes)
        ax[1].set_ylabel(r'Amp [a.u.]')
        ax[1].text(0.5, 0.8, r'$1.03\leq \rho < 1.06$',
                   transform=ax[1].transAxes)
        ax[2].set_xlabel(r'n$_e^{Edge}[10^{20}$m$^{-2}]$')
        leg = ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[1].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylim([0, 1])
        ax2[1].set_ylim([0, 1])
        ax2[2].set_ylim([0, 1])
        ax2[2].set_xlim([0.25, 0.7])
        ax2[2].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax2[0].set_ylabel(r'Amp [a.u]')
        ax2[0].text(0.5, 0.8, r'$1\leq \rho < 1.03$',
                    transform=ax2[0].transAxes)
        ax2[1].set_ylabel(r'Amp [a.u.]')
        ax2[1].text(0.5, 0.8, r'$1.03\leq \rho < 1.06$',
                    transform=ax2[1].transAxes)
        ax2[2].set_xlabel(r'n$_e$/n$_G$')
        leg = ax2[0].legend(loc='best', numpoints=1,
                            frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        ax3[0].axes.get_xaxis().set_visible(False)
        ax3[0].set_ylim([0, 1])
        ax3[1].set_ylim([0, 1])
        ax3[1].set_xlim([0.1, 30])
        ax3[0].set_xscale('log')
        ax3[1].set_xscale('log')
        ax3[0].set_ylabel(r'Amp [a.u]')
        ax3[0].text(0.2, 0.3, r'$1\leq \rho < 1.03$',
                    transform=ax3[0].transAxes)
        ax3[1].set_ylabel(r'Amp [a.u.]')
        ax3[1].text(0.2, 0.3, r'$1.03\leq \rho < 1.06$',
                    transform=ax3[1].transAxes)
        ax3[1].set_xlabel(r'$\Lambda_{div}$')
        leg = ax3[0].legend(loc='best', numpoints=1,
                            frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        fig.savefig('../pdfbox/AmplitudeTargetVsDensityConstantBt.pdf',
                    bbox_to_inches='tight', dpi=300)
        fig2.savefig(
            '../pdfbox/AmplitudeTargetVsGreenwaldConstantBt.pdf',
            bbox_to_inches='tight', dpi=300)
        fig3.savefig('../pdfbox/AmplitudeVsLambdaConstantBt.pdf',
                     bbox_to_inches='tight', dpi=300)

    elif selection == 11:
        shotListA = (34103, 34102, 34104)
        colorList = ('#01406C', '#F03C07', '#28B799')
        iPAug = (0.6, 0.8, 1)
        DirectoryAug = homeDir + "/Documents/Fisica/"\
            "Conferences/IAEA/iaea2018/data/aug/"
        # this is the plot with amplitude, target vs H-5
        fig, ax = mpl.pylab.subplots(figsize=(10, 9), nrows=3,
                                     ncols=1, sharex=True)
        fig.subplots_adjust(hspace=0.08,
                            bottom=0.15, left=0.15,
                            right=0.98)
        # this is the plot of the amplitude, target vs nGw
        fig2, ax2 = mpl.pylab.subplots(figsize=(10, 9),
                                       nrows=3, ncols=1,
                                       sharex=True)
        fig2.subplots_adjust(hspace=0.08,
                             bottom=0.15, left=0.15, right=0.98)
        # this is the plot if shoulder vs LambdaDiv
        fig3, ax3 = mpl.pylab.subplots(figsize=(10, 6),
                                       nrows=2, ncols=1, sharex=True)
        fig3.subplots_adjust(hspace=0.08,
                             bottom=0.15, left=0.15, right=0.98)

        for i, (shot, ip, col) in enumerate(zip(shotListA, iPAug, colorList)):
            df = xray.open_dataarray(
                '../../AUG/analysis/data/Shot%5i' % shot +
                '_DensityRadiation.nc')
            File = h5py.File(DirectoryAug + 'Shot{}.h5'.format(shot), 'r')
            LiBN = File['LiBNorm'].value
            rho = File['rhoLiB'].value
            time = File['timeLiB'].value
            # now integrate in the near and far SOL defining respectively for
            # (1<rho<1.03 and 1.03<rho<1.06) wrt initial perio [1.2, 1.4]
            if shot == 34103:
                _trange = [0.7, 0.8]
            else:
                _trange = [1.2, 1.4]
            _idx = np.where((time >= _trange[0]) & (time <= _trange[1]))[0]
            Reference = np.nanmean(LiBN[_idx, :], axis=0)
            errReference = np.nanstd(LiBN[_idx, :], axis=0)
            LiBN = LiBN[np.where(time > _trange[1])[0], :]
            time = time[np.where(time > _trange[1])[0]]
            _npoint = int(np.floor((time.max()-time.min()))/0.02)
            Split = np.asarray(np.array_split(LiBN, _npoint, axis=0))
            Amplitude = np.asarray(
                [np.nanmean(p, axis=0)-Reference for p in Split])
            ErrAmplitude = np.asarray([
                np.sqrt(np.power(np.nanstd(p, axis=0), 2) +
                        np.power(errReference, 2)) for p in Split])
            _ = np.asarray(np.array_split(time, _npoint))
            timeSplit = np.asarray([np.mean(k) for k in _])
            aNear = np.nanmean(Amplitude[:, np.where(
                (rho >= 1) & (rho < 1.03))[0]], axis=1)
            aFar = np.nanmean(Amplitude[:, np.where(
                (rho >= 1.03) & (rho < 1.06))[0]], axis=1)
            aNearS = np.nanstd(Amplitude[:, np.where(
                (rho >= 1) & (rho < 1.03))[0]], axis=1)
            aFarS = np.nanstd(Amplitude[:, np.where(
                (rho >= 1.03) & (rho < 1.06))[0]], axis=1)
            # we now limit all the quantities in netcdf to
            # time greater than the minimum of
            # the timeSplit and interpolate
            _dummy = df.where((df.t >= timeSplit.min()) &
                              (df.t <= timeSplit.max()), drop=True)
            # this is the plot vs edge density
            S = interp1d(
                _dummy.t.values, _dummy.sel(sig='H-5').values,
                kind='linear', fill_value='extrapolate')
            ax[0].errorbar(S(timeSplit), aNear, yerr=aNearS,
                           fmt='o', color=col,
                           label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1].errorbar(S(timeSplit), aFar, yerr=aFarS, fmt='o', color=col,
                           label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[2].plot(_dummy.sel(sig='H-5'),
                       _dummy.sel(sig='neTarget'), 'o', color=col,
                       label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            # this is the plot vs greenwald
            S = interp1d(
                _dummy.t.values, _dummy.sel(sig='nGw').values,
                kind='linear', fill_value='extrapolate')
            ax2[0].errorbar(S(timeSplit), aNear, yerr=aNearS,
                            fmt='o', color=col,
                            label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax2[1].errorbar(S(timeSplit), aFar, yerr=aFarS, fmt='o', color=col,
                            label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax2[2].plot(_dummy.sel(sig='nGw'),
                        _dummy.sel(sig='neTarget'), 'o', color=col,
                        label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')

            # now load the Lambda Div from saved h5 file
            File = h5py.File(DirectoryAug+'Shot{}.h5'.format(shot))
            _idx = np.where((File['LambdaDivRho'].value >= 1) &
                            (File['LambdaDivRho'].value < 1.03))[0]
            LNear = np.nanmean(File['LambdaDiv'].value[_idx, :],
                               axis=0).ravel()
            _idx = np.where((File['LambdaDivRho'].value >= 1.03) &
                            (File['LambdaDivRho'].value < 1.06))[0]
            LFar = np.nanmean(File['LambdaDiv'].value[_idx, :], axis=0).ravel()
            LTime = File['LambdaDivTime'].value.ravel()
            LNear = LNear[np.where((LTime >= timeSplit.min()) &
                                   (LTime <= timeSplit.max()))[0]]
            LFar = LFar[np.where((LTime >= timeSplit.min()) &
                                 (LTime <= timeSplit.max()))[0]]
            LTime = LTime[np.where((LTime >= timeSplit.min()) &
                                   (LTime <= timeSplit.max()))[0]]

            S = interp1d(LTime, LNear, kind='linear', fill_value='extrapolate')
            ax3[0].errorbar(S(timeSplit[np.where(timeSplit <= 4.)[0]]),
                            aNear[np.where(timeSplit <= 4.)[0]],
                            yerr=aNearS[np.where(timeSplit <= 4.)[0]],
                            fmt='o', color=col,
                            label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            S = interp1d(LTime, LFar, kind='linear', fill_value='extrapolate')
            ax3[1].errorbar(S(timeSplit), aFar, yerr=aFarS, fmt='o', color=col,
                            label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')

        ax[0].axes.get_xaxis().set_visible(False)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[0].set_ylim([0, 1])
        ax[1].set_ylim([0, 1])
        ax[2].set_ylim([0, 1])
        ax[2].set_xlim([0.5, 4])
        ax[2].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0].set_ylabel(r'Amp [a.u]')
        ax[0].text(0.5, 0.8, r'$1\leq \rho < 1.03$',
                   transform=ax[0].transAxes)
        ax[1].set_ylabel(r'Amp [a.u.]')
        ax[1].text(0.5, 0.8, r'$1.03\leq \rho < 1.06$',
                   transform=ax[1].transAxes)
        ax[2].set_xlabel(r'n$_e^{Edge}[10^{20}$m$^{-2}]$')
        leg = ax[0].legend(loc='best', numpoints=1, frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        ax2[0].axes.get_xaxis().set_visible(False)
        ax2[1].axes.get_xaxis().set_visible(False)
        ax2[0].set_ylim([0, 1])
        ax2[1].set_ylim([0, 1])
        ax2[2].set_ylim([0, 1])
        ax2[2].set_xlim([0.25, 0.7])
        ax2[2].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax2[0].set_ylabel(r'Amp [a.u]')
        ax2[0].text(0.5, 0.8, r'$1\leq \rho < 1.03$',
                    transform=ax2[0].transAxes)
        ax2[1].set_ylabel(r'Amp [a.u.]')
        ax2[1].text(0.5, 0.8, r'$1.03\leq \rho < 1.06$',
                    transform=ax2[1].transAxes)
        ax2[2].set_xlabel(r'n$_e$/n$_G$')
        leg = ax2[0].legend(loc='best', numpoints=1,
                            frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        ax3[0].axes.get_xaxis().set_visible(False)
        ax3[0].set_ylim([0, 1])
        ax3[1].set_ylim([0, 1])
        ax3[1].set_xlim([0.1, 30])
        ax3[0].set_xscale('log')
        ax3[1].set_xscale('log')
        ax3[0].set_ylabel(r'Amp [a.u]')
        ax3[0].text(0.2, 0.3, r'$1\leq \rho < 1.03$',
                    transform=ax3[0].transAxes)
        ax3[1].set_ylabel(r'Amp [a.u.]')
        ax3[1].text(0.2, 0.3, r'$1.03\leq \rho < 1.06$',
                    transform=ax3[1].transAxes)
        ax3[1].set_xlabel(r'$\Lambda_{div}$')
        leg = ax3[0].legend(loc='best', numpoints=1,
                            frameon=False, fontsize=14)
        for text, color in zip(leg.get_texts(), colorList):
            text.set_color(color)

        fig.savefig('../pdfbox/AmplitudeTargetVsDensityConstantQ95.pdf',
                    bbox_to_inches='tight')
        fig2.savefig(
            '../pdfbox/AmplitudeTargetVsGreenwaldConstantQ95.pdf',
            bbox_to_inches='tight')
        fig3.savefig('../pdfbox/AmplitudeVsLambdaConstantQ95.pdf',
                     bbox_to_inches='tight')

    elif selection == 12:
        shotListA = (34276, 34278, 34102)
        cryo = ('#34276 H-Mode cryo off',
                '#34278 H-Mode cryon on',
                '#34102 L-mode')
#        colorList = ('#324D5C', '#E37B40', '#60A65F')
        colorList = ('#324D5C', '#E37B40', 'k')
        Directory = '../../AUG/analysis/data/'
        # figure with Efold vs blob Size
        fig, ax = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.17, left=0.17, hspace=0.05, top=0.9)
        # figure for Blob vs Lambda
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig2.subplots_adjust(bottom=0.17, left=0.17, hspace=0.05, top=0.9)
        # first the plot for AUG
        for shot, col in zip(shotListA, colorList):
            for s in ('1', '2', '3', '4', '5'):
                try:
                    Data = xray.open_dataarray(Directory +
                                               'Shot%5i' % shot +
                                               '_'+s+'Stroke.nc')
                    Vperp = Data.Vperp
                    Size = 0.5*Data.TauB*np.abs(Vperp)/1e-4
                    SizeErr = np.sqrt(
                        np.power(Data.Vperp*Data.TauBErr, 2) +
                        np.power(Data.TauB*Data.VperpErr, 2))/2/1e-4
                    if SizeErr > 0.5*Size:
                        SizeErr = 0.5*Size
                    _idx = np.where((Data.rhoLambda >= Data.Rho-0.03) &
                                    (Data.rhoLambda <= Data.Rho+0.03))[0]
                    _idx2 = np.where((Data.rhoLiB >= 1.02-0.01) &
                                     (Data.rhoLiB <= 1.02+0.01))[0]
                    eFold = np.nanmean(Data.Efold[_idx2])*1e2

                    ax.errorbar(Size, np.nanmean(Data.Efold[_idx2])*1e2,
                                xerr=SizeErr,
                                yerr=np.nanstd(Data.Efold[_idx2])*1e2,
                                fmt='o', color=col, ms=15, alpha=0.6)
                    ax2.errorbar(np.nanmean(Data.LambdaProfile[_idx]),
                                 Size, yerr=SizeErr,
                                 xerr=np.nanstd(Data.LambdaProfile[_idx]),
                                 fmt='o',
                                 color=col, ms=15, alpha=0.6)
                except:
                    print(('not done for shot {} stroke {}').format(shot, s))
                    pass

        # figure with Efold vs Blob-size
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.5, 500])
        ax.set_ylim([1, 40])
        ax.set_xlabel(r'$\delta_b [\rho_s]$')
        ax.set_ylabel(r'$\lambda_n$ [cm]')
        ax.set_title('Same fueling')
        for cr, c, _idx in zip(cryo, colorList, range(len(cryo))):
            ax.text(0.01, 0.8 - 0.1 * _idx, cr,
                    color=c, transform=ax.transAxes, fontsize=12)
        fig.savefig('../pdfbox/EfoldSizeShots%5i'
                    % shotListA[0] +
                    '_%5i' % shotListA[1] + '.pdf', bbox_to_inches='tight')

        # figure with Blob size vs Lambda
        ax2.set_xscale('log')
        ax2.set_ylim([0., 200])
        ax2.set_ylabel(r'$\delta_b [\rho_s]$')
        ax2.set_xlabel(r'$\Lambda_{div}$')
        ax2.set_title('Same fueling')
        for cr, c, _idx in zip(cryo, colorList, range(len(cryo))):
            ax2.text(0.05, 0.8 - 0.1 * _idx, cr,
                     color=c, transform=ax2.transAxes, fontsize=12)
        fig2.savefig('../pdfbox/BlobLambdaShots%5i'
                     % shotListA[0] +
                     '_%5i' % shotListA[1] + '.pdf', bbox_to_inches='tight')

    elif selection == 13:
        shotListA = (34276, 34281, 34102)
        cryo = ('# 34276 H-Mode cryo off',
                '# 34281 H-Mode cryon on',
                '# 34102 L-mode')
        colorList = ('#324D5C', '#E37B40', '#60A65F')
        Directory = '/Users/vianello/Desktop/Topic-21/Experiments/AUG/analysis/data/'
        # figure with Efold vs blob Size
        fig, ax = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.17, left=0.17, hspace=0.05, top=0.9)
        # figure for Blob vs Lambda
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig2.subplots_adjust(bottom=0.17, left=0.17, hspace=0.05, top=0.9)
        # first the plot for AUG
        for shot, col in zip(shotListA, colorList):
            for s in ('1', '2', '3', '4', '5'):
                try:
                    Data = xray.open_dataarray(Directory +
                                               'Shot%5i' % shot +
                                               '_'+s+'Stroke.nc')
                    Vperp = Data.Vperp
                    Size = 0.5*Data.TauB*np.abs(Vperp)/1e-4
                    SizeErr = np.sqrt(
                        np.power(Data.Vperp*Data.TauBErr, 2) +
                        np.power(Data.TauB*Data.VperpErr, 2))/2/1e-4
                    if SizeErr > 0.5*Size:
                        SizeErr = 0.5*Size
                    _idx = np.where((Data.rhoLambda >= Data.Rho-0.03) &
                                    (Data.rhoLambda <= Data.Rho+0.03))[0]
                    _idx2 = np.where((Data.rhoLiB >= 1.02-0.02) &
                                     (Data.rhoLiB <= 1.02+0.02))[0]
                    eFold = np.nanmean(Data.Efold[_idx2])*1e2

                    ax.errorbar(Size, np.nanmean(Data.Efold[_idx2])*1e2,
                                xerr=SizeErr,
                                yerr=np.nanstd(Data.Efold[_idx2])*1e2,
                                fmt='o', color=col, ms=15, alpha=0.6)
                    ax2.errorbar(np.nanmean(Data.LambdaProfile[_idx]),
                                 Size, yerr=SizeErr,
                                 xerr=np.nanstd(Data.LambdaProfile[_idx]),
                                 fmt='o',
                                 color=col, ms=15, alpha=0.6)
                except:
                    print(('not done for shot {} stroke {}').format(shot, s))
                    pass

        # figure with Efold vs Blob-size
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.5, 500])
        ax.set_ylim([1, 40])
        ax.set_xlabel(r'$\delta_b [\rho_s]$')
        ax.set_ylabel(r'$\lambda_n$ [cm]')
        ax.set_title('Different fueling')
        for cr, c, _idx in zip(cryo, colorList, range(len(cryo))):
            ax.text(0.01, 0.8 - 0.1 * _idx, cr,
                    color=c, transform=ax.transAxes,
                    fontsize=12)
        fig.savefig('../pdfbox/EfoldSizeShots%5i'
                    % shotListA[0] +
                    '_%5i' % shotListA[1] + '.pdf', bbox_to_inches='tight')

        # figure with Blob size vs Lambda
        ax2.set_xscale('log')
        ax2.set_ylim([0., 200])
        ax2.set_ylabel(r'$\delta_b [\rho_s]$')
        ax2.set_xlabel(r'$\Lambda_{div}$')
        ax2.set_title('Different fueling')
        for cr, c, _idx in zip(cryo, colorList, range(len(cryo))):
            ax2.text(0.05, 0.8 - 0.1 * _idx, cr,
                     color=c, transform=ax2.transAxes,
                     fontsize=12)
        fig2.savefig('../pdfbox/BlobLambdaShots%5i'
                     % shotListA[0] +
                     '_%5i' % shotListA[1] + '.pdf', bbox_to_inches='tight')

    elif selection == 14:
        shotListA = (34276, 34278, 34281, 34102)
        cryo = ('# 34276 H-Mode cryo off',
                '# 34278 H-Mode cryo on same fueling',
                '# 34281 H-Mode cryo different fueling',
                '# 34102 L-mode')
        colorList = ('#324D5C', '#E37B40', '#60A65F', 'k')
        Directory = '../../AUG/analysis/data/'
        # figure with Efold vs blob Size
        fig, ax = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig.subplots_adjust(bottom=0.17, left=0.17, hspace=0.05, top=0.9)
        # figure for Blob vs Lambda
        fig2, ax2 = mpl.pylab.subplots(figsize=(8, 5), nrows=1, ncols=1)
        fig2.subplots_adjust(bottom=0.17, left=0.17, hspace=0.05, top=0.9)
        # first the plot for AUG
        for shot, col in zip(shotListA, colorList):
            for s in ('1', '2', '3', '4', '5'):
                try:
                    Data = xray.open_dataarray(Directory +
                                               'Shot%5i' % shot +
                                               '_'+s+'Stroke.nc')
                    Vperp = Data.Vperp
                    Size = 0.5*Data.TauB*np.abs(Vperp)/1e-4
                    SizeErr = np.sqrt(
                        np.power(Data.Vperp*Data.TauBErr, 2) +
                        np.power(Data.TauB*Data.VperpErr, 2))/2/1e-4
                    if SizeErr > 0.5*Size:
                        SizeErr = 0.5*Size
                    _idx = np.where((Data.rhoLambda >= Data.Rho-0.03) &
                                    (Data.rhoLambda <= Data.Rho+0.03))[0]
                    _idx2 = np.where((Data.rhoLiB >= 1.02-0.02) &
                                     (Data.rhoLiB <= 1.02+0.02))[0]
                    eFold = np.nanmean(Data.Efold[_idx2])*1e2

                    ax.errorbar(Size, np.nanmean(Data.Efold[_idx2])*1e2,
                                xerr=SizeErr,
                                yerr=np.nanstd(Data.Efold[_idx2])*1e2,
                                fmt='o', color=col, ms=15, alpha=0.6)
                    ax2.errorbar(np.nanmean(Data.LambdaProfile[_idx]),
                                 Size, yerr=SizeErr,
                                 xerr=np.nanstd(Data.LambdaProfile[_idx]),
                                 fmt='o',
                                 color=col, ms=15, alpha=0.6)
                except:
                    print(('not done for shot {} stroke {}').format(shot, s))
                    pass

        # figure with Efold vs Blob-size
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([0.5, 500])
        ax.set_ylim([1, 20])
        ax.set_xlabel(r'$\delta_b [\rho_s]$')
        ax.set_ylabel(r'$\lambda_n$ [cm]')
        ax.set_title('Different fueling')
        for cr, c, _idx in zip(cryo, colorList, range(len(cryo))):
            ax.text(0.01, 0.8 - 0.1 * _idx, cr,
                    color=c, transform=ax.transAxes,
                    fontsize=11)
        fig.savefig('../pdfbox/EfoldSizeShots%5i'
                    % shotListA[0] +
                    '_%5i' % shotListA[1] +
                    '_%5i' % shotListA[2] + '.pdf', bbox_to_inches='tight')

        # figure with Blob size vs Lambda
        ax2.set_xscale('log')
        ax2.set_ylim([0, 200])
        #        ax2.set_xlim([0.01, 30])
        ax2.set_ylabel(r'$\delta_b [\rho_s]$')
        ax2.set_xlabel(r'$\Lambda_{div}$')
        ax2.set_title('Different fueling')
        for cr, c, _idx in zip(cryo, colorList, range(len(cryo))):
            ax2.text(0.05, 0.8 - 0.1 * _idx, cr,
                     color=c, transform=ax2.transAxes,
                     fontsize=12)
        fig2.savefig('../pdfbox/BlobLambdaShots%5i'
                     % shotListA[0] +
                     '_%5i' % shotListA[1] +
                     '_%5i' % shotListA[2] + '.pdf', bbox_to_inches='tight')

    elif selection == 15:
        DataTCV = xray.open_dataarray(
            '../../TCV/analysis/data/BlobShot54873_plunge1_rrsep1cm.nc')
        DataTCVn = xray.open_dataarray(
            '../../TCV/analysis/data/BlobShot54873_plunge1_rrsep1cm_norm.nc')
        DataAUG = xray.open_dataarray(
            '../../AUG/analysis/data/Shot34102_1Stroke.nc')
        # now the 2 panel with CAS and FWHM
        fig, ax = mpl.pylab.subplots(figsize=(10, 7), nrows=2, ncols=2)
        # AUG part
        ax[0, 0].plot(DataAUG.t*1e6, DataAUG.sel(sig='Isat_m06'), '-k', lw=2)
        ax[0, 0].fill_between(DataAUG.t*1e6,
                              DataAUG.sel(sig='Isat_m06') -
                              DataAUG.err.reshape(3, 501)[0, :],
                              DataAUG.sel(sig='Isat_m06') +
                              DataAUG.err.reshape(
                                  3, 501)[0, :], facecolor='gray',
                              alpha=0.5, edgecolor='white')
        # compute the FWHM
        _x, _y = DataAUG.t.values, DataAUG.sel(sig='Isat_m06').values
        _y -= _y.min()
        S = UnivariateSpline(_x, _y-_y.max()/2, s=0)
        r1, r2 = S.roots()
        ax[0, 0].axvspan(r1*1e6, r2*1e6, facecolor='yellow', alpha=0.5)
        # now the others two
        colorList = ('#01406C', '#F03C07', '#28B799')
        ax[1, 0].plot(DataAUG.t*1e6, DataAUG.sel(sig='Isat_m07'), '-',
                      color=colorList[1], lw=2, label=r'J$_s^{z}$')
        ax[1, 0].fill_between(DataAUG.t*1e6,
                              DataAUG.sel(sig='Isat_m07') -
                              DataAUG.err.reshape(3, 501)[2, :],
                              DataAUG.sel(sig='Isat_m07') +
                              DataAUG.err.reshape(
                                  3, 501)[2, :], facecolor='#F03C07', edgecolor='white',
                              alpha=0.5)
        ax[1, 0].plot(DataAUG.t*1e6, DataAUG.sel(sig='Isat_m10'), '-',
                      color=colorList[2], lw=2, label=r'J$_s^{r}$')
        ax[1, 0].fill_between(DataAUG.t*1e6,
                              DataAUG.sel(sig='Isat_m10') -
                              DataAUG.err.reshape(3, 501)[1, :],
                              DataAUG.sel(sig='Isat_m10') +
                              DataAUG.err.reshape(
                                  3, 501)[1, :], facecolor='#28B799',
                              alpha=0.5, edgecolor='white')

        ax[0, 0].set_title('AUG')
        ax[0, 0].set_ylabel(r'$\delta$J$_s [\sigma]$')
        ax[0, 0].axes.get_xaxis().set_visible(False)
        ax[0, 0].set_xlim([-100, 100])
        ax[1, 0].set_xlim([-100, 100])
        ax[1, 0].set_xlabel(r't [$\mu$s]')
        ax[1, 0].set_ylabel(r'$\delta$J$_s [\sigma]$')
        leg = ax[1, 0].legend(loc='best', numpoints=1, frameon=False)
        for t, c in zip(leg.get_texts(), ('#F03C07', '#28B799')):
            t.set_color(c)

        # ----------
        # now the TCV part
        ax[0, 1].plot(DataTCVn.t*1e6, DataTCVn.sel(sig='Is'), '-k', lw=2)
        ax[0, 1].fill_between(DataTCVn.t*1e6,
                              DataTCVn.sel(sig='Is') -
                              DataTCVn.err.reshape(8, 501)[0, :],
                              DataTCVn.sel(sig='Is') +
                              DataTCVn.err.reshape(
                                  8, 501)[0, :], facecolor='gray',
                              alpha=0.5, edgecolor='white')
        # compute the FWHM
        _x, _y = DataTCVn.t.values, DataTCVn.sel(sig='Is').values
        _y -= _y.min()
        S = UnivariateSpline(_x, _y-_y.max()/2, s=0)
        r1, r2 = S.roots()
        ax[0, 1].axvspan(r1*1e6, r2*1e6, facecolor='yellow', alpha=0.5)
        # now the others two
        colorList = ('#01406C', '#F03C07', '#28B799')
        ax[1, 1].plot(DataTCV.t*1e6, DataTCV.sel(sig='Epol')/1e3, '-',
                      color=colorList[1], lw=2)
        ax[1, 1].fill_between(DataTCV.t*1e6,
                              DataTCV.sel(sig='Epol')/1e3 -
                              DataTCV.err.reshape(8, 501)[1, :]/1e3,
                              DataTCV.sel(sig='Epol')/1e3 +
                              DataTCV.err.reshape(
                                  8, 501)[1, :]/1e3, facecolor='#F03C07',
                              alpha=0.5, edgecolor='white')

        ax[0, 1].set_title('TCV')
        ax[0, 1].set_ylabel(r'$\delta$J$_s [\sigma]$')
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_xlim([-100, 100])
        ax[1, 1].set_xlim([-100, 100])
        ax[1, 1].set_xlabel(r't [$\mu$s]')
        ax[1, 1].set_ylabel(r'E$_{\theta}$[kV/m]')
        fig.savefig('../pdfbox/ExampleCas.pdf', bbox_to_inches='tight')

        # now example of CROSS-Correlation computation
        # AUGD
        fig, ax = mpl.pylab.subplots(figsize=(8, 8), nrows=2, ncols=1,
                                     sharex=True)
        fig.subplots_adjust(hspace=0.05, right=0.83, left=0.2, top=0.95)
        a = DataAUG.sel(sig='Isat_m07').values
        b = DataAUG.sel(sig='Isat_m06').values
        xcor = np.correlate(a, b, mode='same')
        xcor /= np.sqrt(np.dot(a, a)*np.dot(b, b))
        lag = np.arange(xcor.size, dtype='float')-xcor.size/2
        # in microsecond
        lag *= 0.4
        ax[0].plot(lag, xcor, 'k-', lw=2)
        axt = ax[0].twinx()
        h = hilbert(xcor)
        axt.plot(lag, np.imag(h), 'r-')
        ax[0].text(0.1, 0.9, 'AUG', transform=ax[0].transAxes)
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].set_ylabel('Correlation')
        axt.set_ylabel('Im (H(C))', color='red')
        axt.spines['right'].set_color('red')
        axt.xaxis.label.set_color('red')
        axt.tick_params(axis='y', colors='red')
        # TCV
        a = DataTCV.sel(sig='VFM_1').values
        b = DataTCV.sel(sig='VFT_1').values
        xcor = np.correlate(a, b, mode='same')
        xcor /= np.sqrt(np.dot(a, a)*np.dot(b, b))
        lag = np.arange(xcor.size, dtype='float')-xcor.size/2
        lag *= 0.4
        ax[1].plot(lag, xcor, 'k-', lw=2)
        mod = GaussianModel()
        pars = mod.guess(xcor, x=lag)
        pars['sigma'].set(value=10, vary=True)
        out = mod.fit(xcor, pars, x=lag)
        ax[1].plot(lag, out.best_fit, 'r--')
        ax[1].text(0.1, 0.9, 'TCV', transform=ax[1].transAxes)
        ax[1].set_xlabel(r't [$\mu$s]')
        ax[1].set_ylabel('Correlation')
        fig.savefig('../pdfbox/ExampleCorrelation.pdf',
                    bbox_to_inches='tight')
    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

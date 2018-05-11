from __future__ import print_function
import numpy as np
import matplotlib as mpl
import xarray as xray
import h5py
from scipy.interpolate import UnivariateSpline
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc("font", size=18)
mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
mpl.rc("lines", linewidth=2)


def print_menu():
    print(30 * "-", "MENU", 30 * "-")
    print("1. Target density and radiation vs Density Constant Bt")
    print("2. Target density and radiation vs Greenwald Fraction Constant Bt")
    print("3. Target density and radiation vs Density Constant q95")
    print("4. Target density and radiation vs Greenwald constant q95")
    print('5. Upstream and target profiles Constant Bt')
    print('6. Upstream and target profiles Constant q95')
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
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1, 0].plot(df.sel(sig='H-5')[:-10]/10,
                          df.sel(sig='D10')[:-10]/1e3, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
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
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1, 1].plot(df.sel(sig='en'),
                          df.sel(sig='Bolo')/1e3, '.',  color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
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
                    bbox_to_inches='tight')
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
            '../pdfbox/TargetDensityRadiationVsGreenwaldConstantBt.pdf',
            bbox_to_inches='tight')

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
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1, 0].plot(df.sel(sig='H-5')[:-10]/10,
                          df.sel(sig='D10')[:-10]/1e3, '.', color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
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
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1, 1].plot(df.sel(sig='en'),
                          df.sel(sig='Bolo')/1e3, '.',  color=col,
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
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
                    bbox_to_inches='tight')

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
        DirectoryTcv = "/Users/vianello/Documents/Fisica/Conferences/" \
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

    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

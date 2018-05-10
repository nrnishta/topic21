from __future__ import print_function
import numpy as np
import matplotlib as mpl
import pandas as pd
import MDSplus as mds
import xarray as xray
import smooth
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF,
                                              ConstantKernel,
                                              WhiteKernel)
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
                          df.sel(sig='neMaxTarget')/10, '.',
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1, 1].plot(df.sel(sig='en'),
                          df.sel(sig='Bolo')/1e3, '.',
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
        ax[0, 1].set_xlim([0.1, 1.2])
        ax[0, 1].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_title(r'TCV')
        ax[1, 1].set_xlim([0.1, 1.2])
        ax[1, 1].set_ylabel(r'kW/m$^{2}$')
        ax[1, 1].set_xlabel(r'$\langle$n$_e[10^{20}$m$^{-3}]\rangle$')
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
                          df.sel(sig='neMaxTarget')/10, '.',
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
            ax[1, 1].plot(df.sel(sig='en'),
                          df.sel(sig='Bolo')/1e3, '.',
                          label=r'#%5i' % shot + ' %3.2f' % ip + ' MA')
        ax[0, 1].set_xlim([0.1, 1.2])
        ax[0, 1].set_ylabel(r'n$^{peak}_{target} [10^{20}$m$^{-3}]$')
        ax[0, 1].axes.get_xaxis().set_visible(False)
        ax[0, 1].set_title(r'TCV')
        ax[1, 1].set_xlim([0.1, 1.2])
        ax[1, 1].set_ylabel(r'kW/m$^{2}$')
        ax[1, 1].set_xlabel(r'$\langle$n$_e[10^{20}$m$^{-3}]\rangle$')
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

    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

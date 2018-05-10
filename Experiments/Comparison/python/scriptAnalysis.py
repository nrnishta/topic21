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
        fig.subplots_adjust(wspace=0.25, hspace=0.05)
        colorList = ('#01406C', '#F03C07', '#28B799')
        for i, (shot, ip, col) in enumerate(zip(shotAug, iPAug, colorList)):
            # load the data
            df = xray.open_dataarray(
                '../../AUG/analysis/data/Shot%5i' % shot + '_DensityRadiation.nc')
            ax[0, 0].p
    elif selection == 2:
        pass
    elif selection == 3:
        pass
    elif selection == 4:
        pass
    elif selection == 99:
        loop = False
    else:
        input("Unknown Option Selected!")

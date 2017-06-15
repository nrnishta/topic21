from __future__ import print_function
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
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc("font", size=22)
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Tahoma']})

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
        shotList = (57425, 57437)
        colorList = ('#1f77b4', '#ff7f0e', '#2ca02c')
        # create the plot
        fig, ax = mpl.pylab.subplot(figsize=(16, 10), nrows=)
    if selection == 2:
        pass
    if selection == 3:
        pass
    if selection == 4:
        pass
    elif selection == 99:
        loop = False
    else:
        raw_input("Unknown Option Selected!")

"""
Script reads the processed profiles of ICRF reflectometry system (RIC)
from AUG data bese.

Note:
    To use the script from a command line use argparse notation:
    e.g. to get a profile in a shot number 34283 from antenna 4 
    at time instance 2.5 sec type following in a command line:

    python get_ricProfiles.py -s 34283 -a 4 -t 2.5


Author:
    Egor Seliunin
    eseliunin@ipfn.tecnico.ulisboa.pt
"""

import dd
import findindx
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_ric_profile(shotnumber, antenna, time, accuracy=1e-4):
    """
    Script to read the RIC profiles from data base

    Input:
        shotnumber: int,
                    Number of the requiring shot
        antenna: int,
                 Number of requiring antenna.
                 Note: there are 3 antennas in RIC with following 
                       numbers: 1 : upper plane, 
                                4 : mid plane,
                                8 : lower plane.
        time: float,
              Time instance of a profile in [s]
        accuracy: float,
                  time accuracy to find a value in a time base in 
                  order to get the index
        
    Output:
        p: dictionary,
           Dictionary, that contains the following keys:
           time: float 
                 time instance of a profile 
           area: numpy array, float
                 rho poloidal base of a profile
           data: numpy array, float
                 electron density array
           persistance: float,
                        number of frequency sweeps that was used
                        to calculate the profile
    """

    # Antenna name

    antenna = "Ne_Ant" + str(antenna)

    # Getting data routin

    shotfile = dd.shotfile('RIC', shotnumber)
    ne = shotfile(antenna)
    shotfile.close()

    # Find time index

    (indx1, indy) = findindx.findin(ne.time, time, accuracy)

    # Generating the time, area and data

    time = ne.time[indx1]
    area = ne.area[indx1,:]
    data = ne.data[indx1,:]

    # Constructing dictionary

    p = {"time":time,"area":area, "data":data}
    return p

def plot_ric_profile(shotnumber, antenna, time, accuracy=1e-4):
    """
    Script to plot a profile
    """
    p = get_ric_profile(shotnumber, antenna, time, accuracy=1e-4)
    plt.plot(p["area"], p["data"])
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"n$_e$ [m$^{-3}$]")
    # real time of the profile up to ms
    time = np.trunc(p["time"]*1e3)*1e-3
    title = "#" + str(shotnumber) + ", RIC Antenna " + str(antenna) + \
            ", t = " + str(time) + " s"
    plt.title(title)
    plt.show()




if __name__=="__main__":
    # Use argparse to call the script with input arguments from a command line
    parser = argparse.ArgumentParser(description='Plot electron density \
                                     profile, calculated from RIC data ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s','--shotnumber',
                        help='Shotnumber, required = %(required)s',
                        required=True)
    parser.add_argument('-a','--antenna',
                        help='Antenna, required = %(required)s',
                        required=True)
    parser.add_argument('-t','--time',
                        help='Time instance of requiring profile, \
                              required = %(required)s',
                              required=True)
    parser.add_argument('-acc','--accuracy',
                        help='Time accuracy, required = %(required)s',
                        required=False, default=1e-4)

    args = parser.parse_args()

    shotnumber = int(args.shotnumber)
    antenna = int(args.antenna)
    time = float(args.time)
    accuracy = float(args.accuracy)
    plot_ric_profile(shotnumber, antenna, time, accuracy=accuracy)


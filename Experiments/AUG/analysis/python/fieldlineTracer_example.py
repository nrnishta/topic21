#!/usr/bin/env python

"""
Example script running a field line tracer

Nick Walkden, May 2015
"""

from cyfieldlineTracer import get_fieldline_tracer
from copy import deepcopy as copy
import numpy as np

#loading from a MAST shot
my_tracer =  get_fieldline_tracer('RK4', gfile='/home/nwalkden/gfiles/sxd2010.geqdsk',revBt=False, interp='quintic') #RK4Tracer(gfile='/home/nwalkden/gfiles/sxd2010.geqdsk')

# The tracer is now ready to be used. Note that you can access
# equilibrium quantities using my_tracer.eq which is an equilibrium object

# To trace a field line, call the trace method
my_fieldline = my_tracer.trace(1.36,0.0,mxstep=10000,ds=1e-2,tor_lim=20.0*np.pi)
# Here the first two arguments are the R and Z starting points 
# mxstep is the maximum number of steps to take
# ds is the distance between each step along the field line

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

# Rotating the field line in the toroidal direction is done
# via the method rotateToroidal
my_fieldline_rotated = my_tracer.trace(1.425,0.0,mxstep=100000,ds=1e-2)#copy(my_fieldline)
my_fieldline_rotated.rotate_toroidal(np.pi)

ax1 = fig.add_subplot(121,aspect='equal')
ax1.plot(my_fieldline.R,my_fieldline.Z,'-ro')
ax1.plot(my_tracer.eq.wall['R'],my_tracer.eq.wall['Z'],'k')

ax2 = fig.add_subplot(122,projection='3d',aspect='equal')

ax2.plot(my_fieldline.X,my_fieldline.Y,my_fieldline.Z,'r')
ax2.plot(my_fieldline_rotated.X,my_fieldline_rotated.Y,my_fieldline_rotated.Z,'g')

#Now apply filtering to remove parts of the fieldline not wanted

my_filtered_fieldline = my_fieldline.filter('phi',[0.0,np.pi])
ax2.plot(my_filtered_fieldline.X,my_filtered_fieldline.Y,my_filtered_fieldline.Z,'b')
ax1.plot(my_filtered_fieldline.R,my_filtered_fieldline.Z,'-bx')

ax2.set_zlim3d(-2.0,2.0)
ax2.set_xlim3d(-2.0,2.0)
ax2.set_ylim3d(-2.0,2.0)
plt.show()

#Get the connection length

#Get Radius and parallel distance, S, at both ends of the fieldline

S0,S1 = my_fieldline.S[0],my_fieldline.S[-1]
R0,R1 = my_fieldline.R[0],my_fieldline.R[-1]

print('R = '+str(R0)+', S = '+str(S0))
print('R = '+str(R1)+', S = '+str(S1))

#Note this works because we started the trace at the midplane, so S = 0 corresponds to the midplane
#If we wanted to do, for example, the X-point to target connection length we could do

new_fieldline = my_fieldline.filter('Z',[-10.0,-1.47]) #Get just the bit of the fieldline below the X-point

Lpar = np.abs(new_fieldline.S[0] - new_fieldline.S[-1])

print(Lpar)



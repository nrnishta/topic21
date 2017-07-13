"""
Script to check the consistency of RK4 integrator from Nick
with respect to FLT code from SOL_geometry
"""
from copy import deepcopy as copy
import numpy as np
import eqtools
import sys
import MDSplus as mds
sys.path.append(
    '/home/vianello/work/topic21/Codes/python/general/cyFieldlineTracer/')
sys.path.append(
    '/home/vianello/work/topic21/Codes/python/general/pyEquilibrium/')
from cyfieldlineTracer import get_fieldline_tracer
import equilibrium

Eq = eqtools.TCVLIUQETree(57418)
# build an array of position and corresponding
# r-rsep at 1s
Rarray = np.linspace(1.105, 1.14, num=10)
rmid = Eq.rz2rmid(Rarray,
                  np.repeat(0, Rarray.size), 1)-Eq.getRmidOutSpline()(1)
# hard coded as eqtools does not work remotely
rmid = np.asarray([0.00827948, 0.01215342, 0.01603019, 0.01990912,
                   0.02378965, 0.02767118, 0.03155308, 0.03543471,
                   0.03931538, 0.0431944])
# loading from TCV saved gfile
Tracer = get_fieldline_tracer(
    'RK4', machine='TCV', shot=57418, time=1, remote=False,
    interp='quintic') 
# now build a list of field lines as a 
# To trace a field line, call the trace method
my_fieldlines = [Tracer.trace(r,0.0,mxstep=10000,ds=1e-2,tor_lim=20.0*np.pi) for r in Rarray]
# Here the first two arguments are the R and Z starting points 
# mxstep is the maximum number of steps to take
# ds is the distance between each step along the field line

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

# Rotating the field line in the toroidal direction is done
# via the method rotateToroidal

ax1 = fig.add_subplot(121,aspect='equal')
for lin in my_fieldlines:
    ax1.plot(lin.R,lin.Z,'--')
ax1.plot(Tracer.eq.wall['R'],Tracer.eq.wall['Z'],'k')

ax2 = fig.add_subplot(122,projection='3d',aspect='equal')
for lin in my_fieldlines:
    ax2.plot(lin.X,lin.Y,lin.Z,'--')

ax2.set_zlim3d(-2.0,2.0)
ax2.set_xlim3d(-2.0,2.0)
ax2.set_ylim3d(-2.0,2.0)
plt.show()


#Note this works because we started the trace at the midplane, so S = 0 corresponds to the midplane
#If we wanted to do, for example, the X-point to target connection length we could do
# now the list of field line masked below the Xpoint
new_fieldline = [line.filter('Z',[-0.75,-0.367]) for line in my_fieldlines]

Lpar = np.array([])
for line in new_fieldline:
    try:
        _dummy = np.abs(line.S[0] - line.S[-1])
    except:
        _dummy = np.nan
    Lpar = np.append(Lpar, _dummy)

Tree = mds.Tree('tcv_topic21', 57418)
Lp = Tree.getNode(r'\LPDIVX').data()
tLp = Tree.getNode(r'\LPDIVX').getDimensionAt(0).data()
rrLp = Tree.getNode(r'\LPDIVX').getDimensionAt(1).data()
Tree.quit()
_idx = np.argmin(np.abs(tLp-1))
fig, ax = mpl.pylab.subplots(nrows=1, ncols=1)
ax.plot(rrLp, Lp[_idx, :])
ax.plot(rmid, Lpar)
plt.show()


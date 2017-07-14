"""
Script to compute the appropriate Lparallel using cythonized version
of FLT code. We will compute the parallel connection length from the
height of magnetic axis and a series of points for which we will compute
both the parallel connection length from midplane to LFS and from
Xpoint height to the LFS. We will also save the distance as R-Rsep and
rho_p
"""

from cyfieldlineTracer import get_fieldline_tracer
from copy import deepcopy as copy
import numpy as np
import eqtools
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

shotList = (34102, 34103, 34104, 34105, 34106)
for shot in shotList:
    Eq = eqtools.AUGDDData(shot)
    Eq.remapLCFS()
    
    # load now the tracer for the same shot all at 3s
    myTracer = get_fieldline_tracer('RK4', machine='AUG', shot=shot, time=3, interp='quintic', rev_bt=True)
    # height of magnetic axis
    zMAxis = myTracer.eq.axis.__dict__['z']
    # height of Xpoint
    zXPoint = myTracer.eq.xpoints['xpl'].__dict__['z']
    rXPoint = myTracer.eq.xpoints['xpl'].__dict__['r']
    # now determine at the height of the zAxis the R of the LCFS
    _idTime = np.argmin(np.abs(Eq.getTimeBase()-3))
    RLcfs = Eq.getRLCFS()[_idTime, :]
    ZLcfs = Eq.getZLCFS()[_idTime, :]
    # onlye the part greater then xaxis
    ZLcfs = ZLcfs[RLcfs > myTracer.eq.axis.__dict__['r']]
    RLcfs = RLcfs[RLcfs > myTracer.eq.axis.__dict__['r']]
    Rout = RLcfs[np.argmin(np.abs(ZLcfs[~np.isnan(ZLcfs)]-zMAxis))]
    rmin = np.linspace(Rout+0.003, 2.19, num=25)
    # this is R-Rsep
    rMid = rmin-Rout
    # this is Rho
    rho = Eq.rz2psinorm(rmin, np.repeat(zMAxis, rmin.size), 3, sqrt=True)
    # now create the Traces and lines
    fieldLines = [myTracer.trace(r, zMAxis, mxstep=100000, ds=1e-2, tor_lim=20.0*np.pi) for r in rmin]
    # create and save the figure
    fig = plt.figure(figsize=(18, 10))
    ax1 = fig.add_subplot(121,aspect='equal')
    # plot the separatrix
    ax1.contour(myTracer.eq.R, myTracer.eq.Z, myTracer.eq.psiN, [1], linestyles='--', colors='k', lw=2)
    ax1.plot(myTracer.eq.wall['R'],myTracer.eq.wall['Z'],'k', lw=3)
    for line in fieldLines:
        ax1.plot(line.R,line.Z,'-', lw=0.7)

    ax2 = fig.add_subplot(122,projection='3d',aspect='equal')
    for line in fieldLines:
        ax2.plot(line.X,line.Y,line.Z,'-', lw=0.7)

    ax2.set_zlim3d(-2.0,2.0)
    ax2.set_xlim3d(-2.0,2.0)
    ax2.set_ylim3d(-2.0,2.0)
    mpl.pylab.savefig('../pdfbox/Shot%5i' % shot +'_fieldline.pdf', bbox_to_inches='tight')

    
    
    fieldLinesZ = [line.filter(['R', 'Z'],
                               [[rXPoint, 2], [-10, zXPoint]]) for line in fieldLines]
    Lpar =np.array([])
    for line in fieldLinesZ:
        try:
            _dummy = np.abs(line.S[0] - line.S[-1])
        except:
            _dummy = np.nan
        Lpar = np.append(Lpar, _dummy)

    # compute also the other one
    LparAll = np.array([])
    for line in fieldLines:
        try:
            _dummy = np.abs(line.S[0] - line.S[-1])
        except:
            _dummy = np.nan
        LparAll = np.append(LparAll, _dummy)

    np.savetxt('../equilibriadata/LparallelShot%5i' % shot+'.txt', np.c_[rMid, rho, LparAll, Lpar])






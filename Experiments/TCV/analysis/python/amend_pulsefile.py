import topic21Mds
import MDSplus as mds
from os import listdir
from os.path import isfile, join
import warnings
import numpy as np
mypath = '/home/vianello/work/topic21/Experiments/TCV/data/tree'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
shots = []
for f in onlyfiles:
    try:
        shots.append(int(f[12:17]))
    except:
        pass

shots = np.unique(np.asarray(shots)).astype('int')
for shot in shots:
    # first of all verify that the data are written
    # in the pulse file
    Tcv = mds.Tree('tcv_shot', shot)
    try:
        r = Tcv.getNode(r'\results::fp:r_1').data()
        Tag = True
        warnings.warn('Original tree written')
        Out = topic21Mds.Tree(shot, amend=True)
        Out.toMds()
        warnings.warn('Topic21 tree amended for shot %5i' % shot)
    except:
        Tag = False
        warnings.warn('Original tree not written')

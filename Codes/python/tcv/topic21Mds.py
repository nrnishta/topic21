# this script create the appropriate MDSplus tree
# containing the relevant quantities which are computed
# for the analysis of Topic 21 experiment. In particular we
# will create an MDStree which will contain the following
# quantities
# 1. Density and Temperature upstream profiles as a function of rho and R-Rsep
# 2. Lambda as a function of rho and Remapped upstream profiles
#    using SOL_geometry and processed quantities from LP mdsplus Tree.
#    This will be computed with the same time
#    basis of LP
# 3. Evaluate of Lparallel up to X-point and to the Divertor
# 4. Evaluate appropriate positions in rho for the probehead
# The computation of the parallel connection length will be done through a
# matlab subprocess which save a dummy data which will be
# afterwards stored in the
# appropriate MDSplus tree

import MDSplus as mds
from scipy.interpolate import UnivariateSpline
import subprocess

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

import numpy
import MDSplus as mds
from scipy import interpolate
from scipy import constants
from scipy import io
import langmuir
import eqtools
import pandas as pd


class Tree(object):
    """
    Create the appropriate Tree for processed results concerning
    Topic 21 Experiment
    """

    def __init__(self, shot, Gas='D2'):
        self.shot = shot
        self.gas = 'D2'
        if Gas == 'D2':
            self.Z = 1
            self.mu = 2
        elif Gas == 'H':
            self.Z = 1
            self.mu = 1
        elif Gas == 'He':
            self.Z = 4
            self.mu = 4
        else:
            print('Gas not found, assuming Deuterium')
            self.gas = 'D2'
            self.Z = 1
            self.mu = 2
        self.TcvTree = mds.Tree('tcv_shot', self.shot)
        # load the equilibrium which will be used
        # to compute rho from rmid for the Lparallel
        self.Eq = eqtools.TCVLIUQETree(self.shot)
        # load the data for langmuir
        self._Target = langmuir.LP(self.shot, Type='floor')
        # load the data for the connection length
        self._loadConnection()
        # compute the Lambda
        self._computeLambda()
        # load the profiles from reciprocating
        self._loadUpstream()

    def _loadLangmuir(self):
        """
        Load the target profiles from tcv_shot MDSplus Tree
        """
        self.enT = self.TcvTree.getNode(
            r'\results::langmuir:four_par_fit:dens').data()
        self.teT = self.TcvTree.getNode(
            r'\results::langmuir:four_par_fit:te').data()
        self.rhoT = self.TcvTree.getNode(
            r'\results::langmuir:rho_psi').data()
        self.RRsepT = self.TcvTree.getNode(
            r'\results::langmuir:dsep_mid').data()
        self.time = self.TcvTree.getNode(
            r'\results::langmuir:time').data()

    def _loadConnection(self):
        """
        Load the already saved connection length file which
        have the same time basis of the langmuir one
        """

        data = io.loadmat(
            '/home/vianello/work/topic21/Experiments/TCV/analysis/data/connectionlength' +
            str(int(self.shot))+'mat.mat')
        self.LpU = data['lParUp']
        self.LpX = data['lParDiv']
        self.Lp_drUs = data['drUs'].ravel()
        self.LpTime = data['time'].ravel()
        self.LpRho = numpy.asarray([
            self.Eq.rmid2psinorm(self.Eq.getRmidOutSpline()(t)+self.Lp_drUs,
                                 t, sqrt=True) for t in self.LpTime])

    def _computeLambda(self):
        """
        Compute the Lambda Div with the two cases for the
        parallel connection length (up to the Midplane and up to
        the X-point). It will be computed only for R-Rsep > 0
        """

        # if the number of point is exactly the same
        if self.LpTime.size == self._Target.t.size:
            self.LambdaDivU = numpy.zeros((self.LpTime.size,
                                           self.Lp_drUs.size))
            self.LambdaDivX = numpy.zeros((self.LpTime.size,
                                           self.Lp_drUs.size))
            for _idx in range(self.LpTime.size):
                try:
                    print('Analysis on time %4.3f' % self.LpTime[_idx])
                    # interp1d the density profile
                    _dummy = numpy.vstack((self._Target.RUpStream[_idx, :],
                                           self._Target.en[_idx, :])).transpose()
                    _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
                    x = _dummy[:, 0]
                    y = _dummy[:, 1]
                    neInt = interpolate.interp1d(
                        x, y,
                        fill_value='extrapolate')(self.Lp_drUs)
                    # do the same for the the temperature
                    _dummy = numpy.vstack((self._Target.RUpStream[_idx, :],
                                           self._Target.te[_idx, :])).transpose()
                    _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
                    x = _dummy[:, 0]
                    y = _dummy[:, 1]
                    teInt = interpolate.interp1d(
                        x, y,
                        fill_value='extrapolate')(self.Lp_drUs)

                    nuEi = 5e-11*neInt/(teInt**1.5)
                    Cs = numpy.sqrt(2 * constants.e * teInt /
                                    (self.Z*constants.proton_mass))
                    self.LambdaDivU[_idx, :] = (
                        nuEi * self.LpU[_idx, :] * constants.electron_mass /
                        (self.Z*constants.proton_mass*Cs))
                    self.LambdaDivX[_idx, :] = (
                        nuEi * self.LpX[_idx, :] * constants.electron_mass /
                        (self.Z*constants.proton_mass*Cs))
                except:
                    print('Not worked for time %4.3f' % self.LpTime[_idx])
            else:
                print('number of point different not yet implemented')

    def _loadUpstream(self):
        """
        Load the profiles saved in tabular data files. They can be 
        saved afterwards into the appropriate pulse file
        """
        try:
            doubleP = pd.read_table(
                '/home/tsui/idl/library/data/double/dpm' +
                str(int(self.shot))+'_1.tab',
                skiprows=1, header=0)
            self.enU1 = doubleP['Dens(m-3)'].values
            self.teU1 = doubleP['Temp(eV)'].values
            # we convert in A/m^2
            self.jsU1 = doubleP['Jsat(A/cm^2)'].values*1e4
            # we convert in (m^-2)
            self.enU1Err = doubleP['DensErr(cm-3)'].values*1e18
            self.teU1Err = doubleP['TempErr(eV)'].values
            self.vfTU1 = doubleP['Vft'].values
            self.vfMU1 = doubleP['Vfm'].values
            self.vfBU1 = doubleP['Vfb'].values
            self.drUsU1 = doubleP['rrsep(m)'].values
            self.tU1 = doubleP['Time(s)'].values
            # we need to convert to rho using eqtools since
            # the saved data are incorrect for the reverse
            # field case
            self.rhoU1 = numpy.asarray([
                self.Eq.rmid2psinorm(self.Eq.getRmidOutSpline()(t)+x,
                                     t, sqrt=True) for t, x in
                zip(self.tU1, self.drUsU1)])
            self._Plunge1 = True
        except:
            print('Plunge 1 not found for shot %5i' % self.shot)
            self._Plunge1 = False
        # repeat for the second plunge
        try:
            doubleP = pd.read_table(
                '/home/tsui/idl/library/data/double/dpm' +
                str(int(self.shot))+'_2.tab',
                skiprows=1, header=0)
            self.enU2 = doubleP['Dens(m-3)'].values
            self.teU2 = doubleP['Temp(eV)'].values
            # we convert in A/m^2
            self.jsU2 = doubleP['Jsat(A/cm^2)'].values*1e4
            # we convert in (m^-2)
            self.enU2Err = doubleP['DensErr(cm-3)'].values*1e18
            self.teU2Err = doubleP['TempErr(eV)'].values
            self.vfTU2 = doubleP['Vft'].values
            self.vfMU2 = doubleP['Vfm'].values
            self.vfBU2 = doubleP['Vfb'].values
            self.drUsU2 = doubleP['rrsep(m)'].values
            self.tU2 = doubleP['Time(s)'].values
            # we need to convert to rho using eqtools since
            # the saved data are incorrect for the reverse
            # field case
            self.rhoU2 = numpy.asarray([
                self.Eq.rmid2psinorm(self.Eq.getRmidOutSpline()(t)+x,
                                     t, sqrt=True) for t, x in
                zip(self.tU2, self.drUsU2)])
            self._Plunge2 = True
        except:
            print('Plunge 1 not found for shot %5i' % self.shot)
            self._Plunge2 = False

    def toMds(self):

        Model = mds.Tree('tcv_topic21')
        Model.createPulse(self.shot)
        del Model
        self.saveTree = mds.Tree('tcv_topic21', self.shot)
        self._Lp2Mds()
        self._Lambda2Mds()
        self._Fp2Mds()
        self.saveTree.quit()

    def _Lp2Mds(self):
        """
        Inner method to write parallel connection length
        on MDSplus nodes
        """

        dummy = self.saveTree.getNode(r'\LPDIVU')
        dummy.putData(
            mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2, $3)",
                             self.LpU, self.LpTime,
                             self.Lp_drUs))
        dummy.setUnits('m')

        dummy = self.saveTree.getNode(r'\LPDIVX')
        dummy.putData(
            mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2, $3)",
                             self.LpX, self.LpTime,
                             self.Lp_drUs))
        dummy.setUnits('m')

        dummy = self.saveTree.getNode(r'\LPRHO')
        dummy.putData(
            mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2, $3)",
                             self.LpRho, self.LpTime,
                             self.Lp_drUs))
        dummy.setUnits('m')

    def _Lambda2Mds(self):
        """
        Store the computed Lambda on appropriate MDSplus tree
        """
        dummy = self.saveTree.getNode(r'\LDIVU')
        dummy.putData(
            mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2, $3)",
                             self.LambdaDivU, self.LpTime,
                             self.Lp_drUs))

        dummy = self.saveTree.getNode(r'\LDIVX')
        dummy.putData(
            mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2, $3)",
                             self.LambdaDivX, self.LpTime,
                             self.Lp_drUs))

        dummy = self.saveTree.getNode(r'\LRHO')
        dummy.putData(
            mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2, $3)",
                             self.LpRho, self.LpTime,
                             self.Lp_drUs))

    def _Fp2Mds(self):
        """
        Store the Fast Reciprocating probe into the
        appropriate MDSplus Tree
        """

        # density
        if self._Plunge1:
            dummy = self.saveTree.getNode(r'\FP_1PL_EN')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.enU1, self.tU1))
            dummy.setUnits('m-3')

            dummy = self.saveTree.getNode(r'\FP_1PL_ENERR')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.enU1Err, self.tU1))
            dummy.setUnits('m-3')
            # temperature
            dummy = self.saveTree.getNode(r'\FP_1PL_TE')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.teU1, self.tU1))
            dummy.setUnits('eV')

            dummy = self.saveTree.getNode(r'\FP_1PL_TEERR')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.teU1Err, self.tU1))
            dummy.setUnits('eV')
            # jsat
            dummy = self.saveTree.getNode(r'\FP_1PL_JS')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.jsU1, self.tU1))
            dummy.setUnits('Am-2')
            # floating potentials
            dummy = self.saveTree.getNode(r'\FP_1PL_VFT')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.vfTU1, self.tU1))
            dummy.setUnits('V')

            dummy = self.saveTree.getNode(r'\FP_1PL_VFM')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.vfMU1, self.tU1))
            dummy.setUnits('V')

            dummy = self.saveTree.getNode(r'\FP_1PL_VFB')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.vfBU1, self.tU1))
            dummy.setUnits('V')
            # rho
            dummy = self.saveTree.getNode(r'\FP_1PL_RHO')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.rhoU1, self.tU1))
            # R-Rsep
            dummy = self.saveTree.getNode(r'\FP_1PL_RRSEP')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.drUsU1, self.tU1))
            dummy.setUnits('m')

        if self._Plunge2:
            dummy = self.saveTree.getNode(r'\FP_2PL_EN')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.enU2, self.tU2))
            dummy.setUnits('m-3')

            dummy = self.saveTree.getNode(r'\FP_2PL_ENERR')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.enU2Err, self.tU2))
            dummy.setUnits('m-3')
            # temperature
            dummy = self.saveTree.getNode(r'\FP_2PL_TE')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.teU2, self.tU2))
            dummy.setUnits('eV')

            dummy = self.saveTree.getNode(r'\FP_2PL_TEERR')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.teU2Err, self.tU2))
            dummy.setUnits('eV')
            # jsat
            dummy = self.saveTree.getNode(r'\FP_2PL_JS')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.jsU2, self.tU2))
            dummy.setUnits('Am-2')
            # floating potentials
            dummy = self.saveTree.getNode(r'\FP_2PL_VFT')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.vfTU2, self.tU2))
            dummy.setUnits('V')

            dummy = self.saveTree.getNode(r'\FP_2PL_VFM')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.vfMU2, self.tU2))
            dummy.setUnits('V')

            dummy = self.saveTree.getNode(r'\FP_2PL_VFB')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.vfBU2, self.tU2))
            dummy.setUnits('V')
            # rho
            dummy = self.saveTree.getNode(r'\FP_2PL_RHO')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.rhoU2, self.tU2))
            # R-Rsep
            dummy = self.saveTree.getNode(r'\FP_2PL_RRSEP')
            dummy.putData(
                mds.Data.compile("BUILD_SIGNAL(($VALUE), $1, $2)",
                                 self.drUsU2, self.tU2))
            dummy.setUnits('m')

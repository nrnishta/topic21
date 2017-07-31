import dd
import equilibrium
from XVL import XVL
import matplotlib as mpl
import map_equ
import numpy as np
import sys
sys.path.append('/afs/ipp/u/nvianell/pythonlib/submodules/palettable')
import palettable


class Stark(object):
    """
    Class to load and analyze the Density as estimated from
    Stark broadening effect in the inner and outer divertor.
    The inner divertor are the signal 

    """
    def __init__(self, shot, time=3):

        self.shot = shot
        self.time=time
        self._Dcn = dd.shotfile('DCN', self.shot)
        self._Uvs = dd.shotfile('UVS', self.shot)
        # load the equilibrium
        self.Eq = equilibrium.equilibrium(device='AUG', shot=self.shot, time=self.time)
        # we use M. Cavedon routine to restore
        # the data.
        self._xvs = XVL('Ne', 'ROV', self.shot)
        self.time = self._xvs.idl.time
        self._Ne = self._xvs.idl.data
        self._LosInfo = self._xvs.idl.los_info
        # we build an appropriate dictionary which for each name save the
        # diagnostic the channel and the LoS information. For plottin the
        # lines we use a general dictionary and then we math it according to self._Los['losname']
        self._AllLos = {'ROV-01':{'R1':1.440, 'Z1':-1.192, 'Phi1': 112.453,
                                  'R2': 1.581, 'Z2':-1.201, 'Phi2': 112.464},
                        'ROV-02':{'R1':1.440, 'Z1':-1.192, 'Phi1': 112.453,
                                  'R2': 1.584, 'Z2':-1.190, 'Phi2': 112.461},
                        'ROV-03':{'R1':1.440, 'Z1':-1.192, 'Phi1': 112.453,
                                  'R2': 1.587, 'Z2':-1.177, 'Phi2': 112.458},
                        'ROV-04':{'R1':1.440, 'Z1':-1.192, 'Phi1': 112.453,
                                  'R2': 1.591, 'Z2':-1.164, 'Phi2': 112.451},
                        'ROV-05':{'R1':1.440, 'Z1':-1.192, 'Phi1': 112.453,
                                  'R2': 1.595, 'Z2':-1.150, 'Phi2': 112.451},
                        'ROV-06':{'R1':1.440, 'Z1':-1.192, 'Phi1': 112.453,
                                  'R2': 1.599, 'Z2':-1.135, 'Phi2': 112.447},
                        'ROV-07':{'R1':1.440, 'Z1':-1.192, 'Phi1': 112.453,
                                  'R2': 1.603, 'Z2':-1.118, 'Phi2': 112.424},
                        'ROV-08':{'R1':1.405, 'Z1':-1.151, 'Phi1': 112.282,
                                  'R2': 1.604, 'Z2':-1.115, 'Phi2': 112.440},
                        'ROV-09':{'R1':1.405, 'Z1':-1.151, 'Phi1': 112.282,
                                  'R2': 1.609, 'Z2':-1.096, 'Phi2': 112.440},
                        'ROV-10':{'R1':1.405, 'Z1':-1.151, 'Phi1': 112.282,
                                  'R2': 1.615, 'Z2':-1.074, 'Phi2': 112.440},
                        'ROV-11':{'R1':1.405, 'Z1':-1.151, 'Phi1': 112.282,
                                  'R2': 1.621, 'Z2':-1.050, 'Phi2': 112.434},
                        'ROV-12':{'R1':1.405, 'Z1':-1.151, 'Phi1': 112.282,
                                  'R2': 1.629, 'Z2':-1.021, 'Phi2': 112.450}, 
                        'ROV-13':{'R1':1.405, 'Z1':-1.151, 'Phi1': 112.282,
                                  'R2': 1.639, 'Z2':-0.987, 'Phi2': 112.436}, 
                        'ROV-14':{'R1':1.405, 'Z1':-1.151, 'Phi1': 112.282,
                                  'R2': 1.656, 'Z2':-0.941, 'Phi2': 112.444}}                     
        self.Los = {}
        for name, i in zip(self._LosInfo['losname'], range(len(self._LosInfo['losname']))):
            self.Los[name] = {'R1': self._AllLos[name]['R1'],
                              'Z1': self._AllLos[name]['Z1'],
                              'Phi1': self._AllLos[name]['Phi1'],
                              'R2': self._AllLos[name]['R2'],
                              'Z2': self._AllLos[name]['Z2'],
                              'Phi2': self._AllLos[name]['Phi2'],
                              'S': self._LosInfo['x'][i],
                              'data': self._Ne[i]}

    def plotGeometry(self, time=3):
        """
        Plot the geometry of the different LOS in the divertor region
        with the corresponding equilibrium.
        """
        self.Eq.set_time(time)
        rVessel, zVessel = map_equ.get_gc()
        fig, ax = mpl.pylab.subplots(figsize=(10, 8), nrows=1, ncols=1)
        fig.subplots_adjust(right=0.8)
        for key in rVessel.iterkeys():
            ax.plot(rVessel[key], zVessel[key], 'k', lw=1.5)
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN, np.linspace(0.1, 0.95, num=10),
                   colors='grey', linestyles='-')
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN, [1],colors= 'r', linewidths=2)
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN, np.linspace(1.01, 1.05, num=4),
                   colors='grey', linestyles='--')
        # adess facciamo il ciclo sulle LoS salvate e
        # costruiamo una legenda a dx
        ax.set_color_cycle(palettable.tableau.Tableau_20.mpl_colors)
        for name in self.Los.iterkeys():
            ax.plot([self.Los[name]['R1'], self.Los[name]['R2']],
                    [self.Los[name]['Z1'], self.Los[name]['Z2']],
                    label=name, lw=2)
        ax.legend(loc='upper center', numpoints=1, frameon=False,
                  bbox_to_anchor=(0.5, 1.2), fontsize=14,
                  ncol=3)
        ax.set_aspect('equal')
        ax.set_xlim([0.7, 2.])
        ax.set_ylim([-1.5, -0.5])
        

    def plotTimeTraces(self, trange=[0, 7]):
        """
        This produce a plot of all the collected signal
        divided into two groups together with the evolution of
        H-5N and of the fueling. This will be combined also
        with the equilibrium with the same color code of the LoS

        """


        fig = mpl.pylab.figure(figsize=(14, 10))
        ax = mpl.pylab.subplot2grid((4, 2), (0, 0), rowspan=2)
        self.Eq.set_time(3)
        rVessel, zVessel = map_equ.get_gc()
        for key in rVessel.iterkeys():
            ax.plot(rVessel[key], zVessel[key], 'k', lw=1.5)
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN, np.linspace(0.1, 0.95, num=10),
                   colors='grey', linestyles='-')
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN, [1],colors= 'r', linewidths=2)
        ax.contour(self.Eq.R, self.Eq.Z, self.Eq.psiN, np.linspace(1.01, 1.05, num=4),
                   colors='grey', linestyles='--')
        # adess facciamo il ciclo sulle LoS salvate e
        # costruiamo una legenda a dx
        ax.set_color_cycle(palettable.tableau.Tableau_20.mpl_colors)
        for name in self.Los.iterkeys():
            ax.plot([self.Los[name]['R1'], self.Los[name]['R2']],
                    [self.Los[name]['Z1'], self.Los[name]['Z2']],
                    label=name, lw=2)
        ax.legend(loc='upper center', numpoints=1, frameon=False,
                  bbox_to_anchor=(0.5, 1.2), fontsize=14,
                  ncol=3)
        ax.set_aspect('equal')
        ax.set_xlim([0.7, 2.])
        ax.set_ylim([-1.5, -0.5])

        # now plot the rest including fueling and density
        ax1 = mpl.pylab.subplot2grid((4, 2), (0, 1))
        ax1.plot(self._Dcn('H-5').time, self._Dcn('H-5').data/1e19, 'k', lw=2,
                 label=r'# %5i' % self.shot)
        ax1.set_ylabel(r'$\overline{n}_e$ H-5 [10$^{19}$]')
        ax1.set_xlabel(r't [s]')
        ax1.set_xlim(trange)

        ax2 = mpl.pylab.subplot2grid((4, 2), (1, 1))
        ax2.plot(self._Uvs('D_tot').time, self._Uvs('D_tot').data/1e21, 'k', lw=2,
                 label=r'# %5i' % self.shot)
        ax2.set_ylabel(r'D$_2 [10^{21}$ e/s]')
        ax2.set_xlabel(r't [s]')
        ax2.set_xlim(trange)

        ax3 = mpl.pylab.subplot2grid((4, 2), (2, 1), rowspan=2)
        ax3.set_color_cycle(palettable.tableau.Tableau_20.mpl_colors)
        for name in self.Los.iterkeys():
            ax3.plot(self.time,
                    self.Los[name]['data']/1e20,
                    label=name, lw=2)
        ax3.set_xlabel('t[s]')
        ax3.set_ylabel(r'n$_e [10^{20}$ m$^{-3}$]')
        ax3.set_ylim([0, 3])
        ax3.set_xlim(trange)

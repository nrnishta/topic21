import numpy
from scipy import io
from scipy import interpolate
from scipy import constants
import numpy as np
import MDSplus as mds
from cyfieldlineTracer import get_fieldline_tracer
import eqtools

class LP:
    def __init__(self, shot, Type='floor'):
        """
        Python class to read and compute appropriate
        profiles of LPs. It compute appropriately also
        divertor collisionality using the already stored
        parallel connection length (or computing if not
        found). It use the parallel connection length
        up to the position of the X-point according to the
        definition of Myra

        :param shot:
            Shot number
        :param Type:
            Type of probes used.
            Possible values are 'floor','HFSwall','LFSwall'
        """

        self.shot = shot
        self.type = Type
        # open the appropriate Tree and get the available probes and
        # locations
        self._Tree = mds.Tree('tcv_shot', self.shot)
        # these are the probes number used in the present shot
        self.Probes = self._Tree.getNode(
            r'\results::langmuir:probes').data()
        # load the position of the langmuir probes
        self.pos = io.loadmat(r'/home/vianello/work/tcv15.2.2.3/' +
                              'data/lp_codes/LPposit.mat')['lppos']
        # we now have properly defined all the quantities needed
        self._defineProbe(type=self.type)
        # appropriate remapping upstream
        self.RemapUpstream()
        # load the equilibrium
        self.Eqm = eqtools.TCVLIUQETree(self.shot)
        # try to open the filament Tree otherwise we will
        # need to compute using Field Line Tracing
        try:
            self._filament = mds.Tree('tcv_topic21', self.shot)
            self._tagF = True
        except:
            self._tagF = False
            print('Filament Tree not found')

    def _defineProbe(self, type='floor'):
        """
        Choose the appropriate type of probes which will
        be remapped through the type keyword. Possibilities
        are 'floor', 'HFSwall','LFSwall' with obvious
        meanings
        """
        # now we read the position for each of the probe in
        # term of (R, Z)
        R, Z = zip(self._Tree.getNode(
            r'\results::langmuir:pos').data())
        R = np.asarray(R).ravel()
        Z = np.asarray(Z).ravel()
        self.t = self._Tree.getNode(
            r'\results::langmuir:time').data()
        self.t2 = self._Tree.getNode(
            r'\results::langmuir:time2').data()
        if type == 'floor':
            # limit to the floor probe
            self._idx = np.where(Z == -0.75)[0]
        elif type == 'HFSwall':
            self._idx = np.where(R == 0.624)[0]
        elif type == 'LFSwall':
            self._idx = np.where(R == 1.136)[0]
        else:
            print('Not implemented')

        self.en = self._Tree.getNode(
           r'\results::langmuir:four_par_fit:dens').data()[:, self._idx]
        self.te = self._Tree.getNode(
            r'\results::langmuir:four_par_fit:Te').data()[:, self._idx]
        self.angle = self._Tree.getNode(
            r'\results::langmuir:area').data()[:, self._idx]
        self.jSat2 = self._Tree.getNode(
            r'\results::langmuir:jsat2').data()[:, self._idx]
        self.pPer = self._Tree.getNode(
            r'\results::langmuir:four_par_fit:P_perp').data()[:, self._idx]
        self.R = R[self._idx]
        self.Z = Z[self._idx]

    def RemapUpstream(self):
        # first of all trasform remap the R, Z to
        # Rmid

        self.RUpStream = self._Tree.getNode(
            r'\results::langmuir:dsep_mid').data()[:, self._idx]
        self.RUpStream2 = self._Tree.getNode(
            r'\results::langmuir:dsep_mid2').data()[:, self._idx]
        self.Rho = self._Tree.getNode(
            r'\results::langmuir:rho_psi').data()[:, self._idx]
        self.Rho2 = self._Tree.getNode(
            r'\results::langmuir:rho_psi2').data()[:, self._idx]

    def UpStreamProfile(self, trange=[0.6, 0.8]):
        """
        In this way we define the profile in a given
        time range upstream remapped
        """
        # now we only need to average appropriately
        # taking into account that at each time
        # step we have different remapping
        rOut = numpy.asarray([])
        neOut = numpy.asarray([])
        teOut = numpy.asarray([])
        rhoOut = numpy.asarray([])
        for r in range(self.R.size):
            _idx = ((self.t >= trange[0]) &
                    (self.t <= trange[1])).nonzero()[0]
            for i in _idx:
                neOut = numpy.append(neOut, self.en[_idx, r])
                teOut = numpy.append(teOut, self.te[_idx, r])
                rOut = numpy.append(rOut,
                                    self.RUpStream[_idx, r])
                rhoOut = numpy.append(rhoOut,
                                      self.Rho[_idx, r])
        # we include also an univariate spline interpolation
        # and we decide to output a dictionary containing the
        # values
        _dummy = numpy.vstack((rOut, neOut)).transpose()
        _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
        x = _dummy[:, 0]
        y = _dummy[:, 1]
        neInt = interpolate.interp1d(x, y, fill_value='extrapolate')

        _dummy = numpy.vstack((rOut, teOut)).transpose()
        _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
        x = _dummy[:, 0]
        y = _dummy[:, 1]
        teInt = interpolate.interp1d(x, y, fill_value='extrapolate')

        out = dict([('r', rOut),
                    ('en', neOut),
                    ('te', teOut),
                    ('neInt', neInt),
                    ('teInt', teInt),
                    ('rho', rhoOut)])
        return out

    def Lambda(self, gas='D2', trange=[0.6, 0.8]):
        """
        Compute the Lambda divertor profile given the
        array of connection length, the gas and the trange
        to perform the computation of the profile. It uses 

        """

        # load the Lp profiles and average over the same range
        if self._tagF:
            print('Using Lambda from Tree')
            LpN = self._filament.getNode(r'\LDIVX')
            LpT = LpN.getDimensionAt(0).data()
            _idx = np.where(((LpT >= trange[0]) &
                            (LpT <= trange[1])))[0]
            xCl = LpN.getDimensionAt(1).data()
            Lambda = np.mean(LpN.data()[_idx,:],axis=0)
        else:
            if gas == 'D2':
                Z = 1
            elif gas == 'H':
                Z = 1
            elif gas == 'He':
                Z = 4
            else:
                print('Gas not found')
            out = self.UpStreamProfile(trange=trange)
            xCl, Lp = self._computeLpar(trange=trange)
            neInt = out['neInt'](xCl)
            teInt = out['teInt'](xCl)
            nuEi = 5e-11*neInt/(teInt**1.5)
            Cs = numpy.sqrt(2 * constants.e * teInt /
                                        (Z*constants.proton_mass))
            Lambda = (nuEi * Lp * constants.electron_mass /
                          (Z*constants.proton_mass*Cs))

        return Lambda, xCl

    def TotalSpIonFlux(self):
        """
        Method for the computation of the total ion flux
        to the Strike point chosen (if floor is chosen it uses the
        second strike point)
        """
        # now we compute the flux
        intFlux = numpy.zeros(self.t2.size)
        for i in range(self.t2.size):
            _x = self.R
            _y = self.jSat2[i, :]/constants.elementary_charge*1e4
            # eliminate the NaN
            _dummy = numpy.vstack((_x, _y)).transpose()
            _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
            _x = _dummy[:, 0]
            _y = _dummy[:, 1][numpy.argsort(_x)]
            _x = numpy.sort(_x)
            intFlux[i] = numpy.trapz(_y, x=2*numpy.pi*_x)
        return intFlux

    def _computeLpar(self, trange=[0.8, 1], Plot=False, Type='floor'):
        """
        Method for the computation of the parallel connection length
        using the cyFieldLine Tracer Code. Presently it is
        implemented only for the lenght in the Divertor region

        :param trange:
            2D array indicating the time minium and maximum
        :param Plot:
            Boolean in case I want a plot
        :param Type:
            The only Possible value so far is 'floor'
        :return:
        """

        self.Tracer = get_fieldline_tracer('RK4', machine='TCV', remote=True,
                                           shot=self.shot, time=t0,
                                           interp='quintic', rev_bt=True)
        if Type is 'floor':
            # height of magnetic axis
            zMAxis = Tracer.eq.axis.__dict__['z']
            # height of Xpoint
            zXPoint = Tracer.eq.xpoints['xp1'].__dict__['z'][0]
            rXPoint = Tracer.eq.xpoints['xpl1'].__dict__['r'][0]
            # now determine at the height of the zAxis the R of the LCFS
            Boundary = Tracer.eq.get_fluxsurface(1.0)
            close('all')
            RLcfs = Boundary.R
            ZLcfs = Boundary.Z
            # only the part greater then xaxis
            ZLcfs = ZLcfs[RLcfs > self.axis.__dict__['r']]
            RLcfs = RLcfs[RLcfs > self.axis.__dict__['r']]
            Rout = RLcfs[np.argmin(np.abs(ZLcfs[~np.isnan(ZLcfs)]-zMAxis))]
            rmin = np.linspace(Rout+0.001, 2.19, num=30)
            # this is the R-Rsep
            rMid = rmin-Rout
            # get the corrisponding Rho
            rho = self.Eqm.rz2rho(rmin, np.repeat(zMAxis, rmin.size), self._time,
                                  extrapolate=True).squeeze()

            # compute the field lines
            fieldLines = [self.Tracer.trace(r, zMAxis, mxstep=100000, ds=1e-2, tor_lim=20.0*np.pi) for r in rmin]
            # compute the parallel connection length from the divertor plate to the X-point
            fieldLinesZ = [line.filter(['R', 'Z'],
                                       [[rXPoint, 2], [-10, zXPoint]]) for line in fieldLines]
            Lpar =np.array([])
            for line in fieldLinesZ:
                try:
                    _dummy = np.abs(line.S[0] - line.S[-1])
                except:
                    _dummy = np.nan
                Lpar = np.append(Lpar, _dummy)

            # we then remove the temporary created G-file
            return rMid, Lpar
        else:
            print('Only outer divertor implemented so far')
            pass

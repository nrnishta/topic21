import MDSplus as mds
import matplotlib as mpl
import scipy
import numpy
import eqtools


class Radiation:
    """
    Python class to integrate the radiation
    in different region of the plasmas
    which need to eventually decided.
    Inputs:
    -------
       shot = shot number

    Methods:
    -------
       Core = Integrate the radiation in the region enclosed
              within the LCFS
       LfsLeg = Integrate the radiation in the region below the
              x-point and enclosed in the region at the LFS
              with respect to the linear interpolation between
              the X-point and the Magnetic-Axis
       lfsSol = Radiation in the LFS SOL (above the X-point up to the
                horizontal position of the Magnetic Axis)
       hfsSOl = Radiation in the HFS SOL (above the X-point up to the
                horizontal position of the Magnetic Axis)
       pFr = Radiation in the Private Flux Region
       Sol = Radiation in the SOL (not in the PFR)
       plotRegion = Plot the different regions at a given time

    Attributes:
    ----------
       shot = Shot which has been analized

    Dependencies:
    ----------
       matplotlib
       MDSplus
       numpy
    """

    def __init__(self, shot):

        self.shot = shot
        # the equilibrium
        self.eq = eqtools.TCVLIUQETree(self.shot)
        # now open the three and read the important results
        self.tree = mds.Tree('tcv_shot', self.shot)
        R = [0.624, 0.624, 0.666, 0.672, 0.965, 0.971, 1.136, 1.136,
             0.971, 0.965, 0.672, 0.666, 0.624, 0.624, 0.624]
	Z = [0.697,0.704, 0.75, 0.75, 0.75, 0.747, 
             0.55, -0.55, -0.747, -0.75, -0.75,
             -0.75, -0.704, -0.697, 0.697]
        try:
            # remember that we have all at LIUQE times
            pRadN = self.tree.getNode(r'\results::bolo:emissivity')
            self.time = pRadN.getDimensionAt(2).data()
            self.R = pRadN.getDimensionAt(0).data()
            self.Z = pRadN.getDimensionAt(1).data()
            self.pRad = pRadN.data()
            # now add the part on the equilibrium
            self.rxP = self.tree.getNode(r'\results::r_xpts').data()
            self.zxP = self.tree.getNode(r'\results::z_xpts').data()
            self.actX = self.tree.getNode(
                r'\results::indx_act_xp').data().astype('int')-1
            self.rmAx = self.tree.getNode(r'\results::r_axis').data()
            self.zmAx = self.tree.getNode(r'\results::z_axis').data()
            self.rLcfs = self.tree.getNode(r'\results::r_contour').data()
            self.zLcfs = self.tree.getNode(r'\results::z_contour').data()
            self.rLim = numpy.asarray(R)
            self.zLim = numpy.asarray(Z)
            self.tree.quit()
            # determine the grid of psiNormalize
            self.psiN = self.eq.rz2psinorm(self.R, self.Z, self.time,
                                           make_grid=True, each_t=True)
        except:
            print('Prad node not filled')

    def _pathCore(self, tidx):
        """
        Define the matplotlib path of the LCFS at a given time
        """
        line = numpy.vstack([self.rLcfs[tidx, :],
                             self.zLcfs[tidx, :]]).transpose()
        # ensure to eliminate the NaN
        line = line[~numpy.isnan(line).any(1)]
        # build the path
        path = mpl.path.Path(line)
        return path

    def _pathOleg(self, tidx):
        """
        Define the matplotlib path of the so-called Outer-leg at a given
        time
        """

        # we must find the linear interpolation
        # between the axctive X-point and
        # the magnetic axis and the position at the
        # intercept at the vessel
        p = numpy.polyfit([self.rxP[tidx, self.actX[tidx]],
                           self.rmAx[tidx]],
                          [self.zxP[tidx, self.actX[tidx]],
                           self.zmAx[tidx]], 1)
        pF = numpy.poly1d(p)
        # find the position at the minimum
        intHfs = pF(self.rLim.min())
        # find the closest point to the
        # now the polynomial at the height of zXP
        p = numpy.polyfit([self.rxP[tidx, self.actX[tidx]],
                           self.rLim.max()],
                          [self.zxP[tidx, self.actX[tidx]],
                           self.zxP[tidx, self.actX[tidx]]], 1)
        intLfs = numpy.poly1d(p)(self.rLim.max())
        # find the minimum height
        mn = numpy.min([intHfs, intLfs])
        # and build the path remember they are anti-clockwise
        coords = [r for r in zip(self.rLim[self.zLim < mn],
                                 self.zLim[self.zLim < mn])]
        # and now add the points
        coords.append((self.rLim.min(), intHfs))
        coords.append((self.rxP[tidx, self.actX[tidx]],
                       self.zxP[tidx, self.actX[tidx]]))
        coords.append((self.rLim.max(), intLfs))
        # now we build the path
        codes = [mpl.path.Path.MOVETO] + \
                (len(coords)-1) * [mpl.path.Path.LINETO]
        path = mpl.path.Path(coords, codes, closed=True)
        return path

    def _pathLfsSol(self, tidx):
        """
        Define the path corresponding to Outer SOL
        at a given time instant in the time grid
        """
        # eliminate all the NaN from the rLcfs and zLcfs
        _dummy = numpy.vstack((self.rLcfs[tidx], self.zLcfs[tidx])).transpose()
        _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
        yAll = _dummy[:, 1]
        xAll = _dummy[:, 0]
        x = xAll[yAll > self.zmAx[tidx]]
        y = yAll[yAll > self.zmAx[tidx]]

        # the point you are interested is close
        # to the radial position of the Xpoint
        _d = numpy.nanargmin(numpy.abs(x -
                                       self.rxP[tidx, self.actX[tidx]]))
        # find the equivalent to all the the point in the LCFS
        _dummy = ((xAll == x[_d]) & (yAll == y[_d]))
        _dA = _dummy[0].astype('int')
        _dX = numpy.argmin(numpy.abs(xAll -
                                     self.rxP[tidx, self.actX[tidx]]))
        xn = xAll[_dA:_dX]
        yn = yAll[_dA:_dX]
#         # these are the vertices

        coords = [(x[_d], y[_d]),
                  (x[_d], self.zLim.max())]
        coords.extend([r for r in zip(
                self.rLim[((self.rLim > self.rmAx[tidx]) &
                           (self.zLim > self.zxP[tidx,
                                                 self.actX[tidx]]))],
                self.zLim[((self.rLim > self.rmAx[tidx]) &
                           (self.zLim > self.zxP[tidx,
                                                 self.actX[tidx]]))])])
        coords.append((self.rLim.max(),
                       self.zxP[tidx, self.actX[tidx]]))
        coords.append((self.rxP[tidx, self.actX[tidx]],
                       self.zxP[tidx, self.actX[tidx]]))
        coords.extend([r for r in zip(xn[::-1], yn[::-1])])
        codes = [mpl.path.Path.MOVETO] + \
                (len(coords)-1) * [mpl.path.Path.LINETO]
        path = mpl.path.Path(coords, codes, closed=True)
        return path

    def _pathHfsSol(self, tidx):
        """
        Define the path for the inner SOL

        """
        # eliminate all the NaN from the rLcfs and zLcfs
        _dummy = numpy.vstack((self.rLcfs[tidx], self.zLcfs[tidx])).transpose()
        _dummy = _dummy[~numpy.isnan(_dummy).any(1)]
        yAll = _dummy[:, 1]
        xAll = _dummy[:, 0]
        x = xAll[yAll > self.zmAx[tidx]]
        y = yAll[yAll > self.zmAx[tidx]]

        # the point you are interested is close
        # to the radial position of the magnetic axis
        _d = numpy.nanargmin(numpy.abs(x -
                                       self.rmAx[tidx]))
        # find the equivalent to all the the point in the LCFS
        _dummy = ((xAll == x[_d]) & (yAll == y[_d]))
        _dX = numpy.argmin(numpy.abs(xAll -
                                     self.rxP[tidx, self.actX[tidx]]))
        xn = xAll[_dX:]
        yn = yAll[_dX:]
        coords = [(x[_d], y[_d]),
                  (x[_d], self.zLim.max())]
        _xv = self.rLim[((self.rLim < x[_d]) &
                         (self.zLim > self.zxP[tidx,
                                               self.actX[tidx]]))]
        _yv = self.zLim[((self.rLim < x[_d]) &
                         (self.zLim > self.zxP[tidx,
                                               self.actX[tidx]]))]
        _ii = numpy.argsort(_xv)[::-1]
        coords.extend([r for r in zip(_xv[_ii], _yv[_ii])])
        coords.append((self.rLim.min(),
                       self.zxP[tidx, self.actX[tidx]]))
        coords.append((self.rxP[tidx, self.actX[tidx]],
                       self.zxP[tidx, self.actX[tidx]]))
        coords.extend([r for r in zip(xn, yn)])
        codes = [mpl.path.Path.MOVETO] + \
                (len(coords)-1) * [mpl.path.Path.LINETO]
        path = mpl.path.Path(coords, codes, closed=True)
        return path

    def _pathPrivate(self, tidx):
        """
        Define the path for the almost private flux region
        """
        p = numpy.polyfit([self.rxP[tidx, self.actX[tidx]],
                           self.rmAx[tidx]],
                          [self.zxP[tidx, self.actX[tidx]],
                           self.zmAx[tidx]], 1)
        pF = numpy.poly1d(p)
        # find the position at the minimum
        intHfs = pF(self.rLim.min())
        coords = [(self.rLim.min(), intHfs),
                  (self.rLim.min(), self.zxP[tidx, self.actX[tidx]]),
                  (self.rxP[tidx, self.actX[tidx]],
                   self.zxP[tidx, self.actX[tidx]])]
        codes = [mpl.path.Path.MOVETO] + \
                (len(coords)-1) * [mpl.path.Path.LINETO]
        path = mpl.path.Path(coords, codes, closed=True)
        return path

    def _pathAboveXpMaLine(self, tidx):

        """
        Give the path defining the closed polygon
        above the line passing through X-point and
        Magnetic Axis
        """
        p = numpy.polyfit([self.rxP[tidx, self.actX[tidx]],
                           self.rmAx[tidx]],
                          [self.zxP[tidx, self.actX[tidx]],
                           self.zmAx[tidx]], 1)
        pF = numpy.poly1d(p)
        coords = ((self.rLim.min(), pF(self.rLim.min())),
                  (self.rLim.max(), pF(self.rLim.max())),
                  (self.rLim.max(), self.zLim.max()),
                  (self.rLim.min(), self.zLim.max()))
        path = mpl.path.Path(coords)
        return path

    def _pathBelowXpMaLine(self, tidx):

        """
        Give the path defining the closed polygon
        above the line passing through X-point and
        Magnetic Axis
        """
        p = numpy.polyfit([self.rxP[tidx, self.actX[tidx]],
                           self.rmAx[tidx]],
                          [self.zxP[tidx, self.actX[tidx]],
                           self.zmAx[tidx]], 1)
        pF = numpy.poly1d(p)
        coords = ((self.rLim.min(), pF(self.rLim.min())),
                  (self.rLim.max(), pF(self.rLim.max())),
                  (self.rLim.max(), self.zLim.min()),
                  (self.rLim.min(), self.zLim.min()))
        path = mpl.path.Path(coords)
        return path

    def Core(self):
        """
        This is the method which computes the radiation in the
        core region using a loop over time
        with the condition on psiN

        """
        # this is what we have in output
        emissivity = numpy.zeros(self.time.size)
        _dz = (self.Z.max()-self.Z.min())/(self.Z.size-1)
        _dx = (self.R.max()-self.R.min())/(self.R.size-1)
        for tidx in range(self.time.size):
            _dpsiN = self.psiN[tidx, :, :]
            mask = (_dpsiN < 1.)
            mask[self.Z < self.zxP[tidx, self.actX[tidx]], :] = False
            _dummy = self.pRad[tidx, :, :]*mask
            emissivity[tidx] = numpy.trapz(2 * scipy.pi *
                                           numpy.trapz(
                                               _dummy,
                                               dx=_dz, axis=0),
                                           dx=_dx)
        return emissivity

    def LfsSol(self):
        """
        This is the method which computes the radiation in the
        core region using a loop over time
        and the definition of _pathCore method

        """
        # this is what we have in output
        emissivity = numpy.zeros(self.time.size)
        _dz = (self.Z.max()-self.Z.min())/(self.Z.size-1)
        _dx = (self.R.max()-self.R.min())/(self.R.size-1)
        for tidx in range(self.time.size):
            mask = ((self.psiN[tidx, :, :] > 1) &
                    (self.psiN[tidx, :, :] < 1.4))
            mask[self.Z < self.zxP[tidx, self.actX[tidx]], :] = False
            mask[:, self.R < self.rxP[tidx, self.actX[tidx]]] = False
            # now integrate
            emissivity[tidx] = numpy.trapz(2 * scipy.pi *
                                           numpy.trapz(
                                               self.pRad[tidx, :, :]*mask,
                                               dx=_dz, axis=0),
                                           dx=_dx)

        return emissivity

    def LfsLeg(self):
        """
        Outer leg radiation

        """

        emissivity = numpy.zeros(self.time.size)
        _dz = (self.Z.max()-self.Z.min())/(self.Z.size-1)
        _dx = (self.R.max()-self.R.min())/(self.R.size-1)
        for tidx in range(self.time.size):
            mask = ((self.psiN[tidx, :, :] > 1) &
                    (self.psiN[tidx, :, :] < 1.4))
            mask[self.Z > self.zxP[tidx, self.actX[tidx]], :] = False
            mask[:, self.R < self.rxP[tidx, self.actX[tidx]]] = False
            # now integrate
            emissivity[tidx] = numpy.trapz(2 * scipy.pi *
                                           numpy.trapz(
                                               self.pRad[tidx, :, :]*mask,
                                               dx=_dz, axis=0),
                                           dx=_dx)

        return emissivity

    def HfsSolLeg(self):
        """
        Outer Sol + Leg radiation

        """

        emissivity = numpy.zeros(self.time.size)
        _dz = (self.Z.max()-self.Z.min())/(self.Z.size-1)
        _dx = (self.R.max()-self.R.min())/(self.R.size-1)
        for tidx in range(self.time.size):
            mask = ((self.psiN[tidx, :, :] > 1) &
                    (self.psiN[tidx, :, :] < 1.4))
            mask[:, self.R > self.rxP[tidx, self.actX[tidx]]] = False
            # now integrate
            emissivity[tidx] = numpy.trapz(2 * scipy.pi *
                                           numpy.trapz(
                                               self.pRad[tidx, :, :]*mask,
                                               dx=_dz, axis=0),
                                           dx=_dx)

        return emissivity

    def PrivateFlux(self):
        emissivity = numpy.zeros(self.time.size)
        _dz = (self.Z.max()-self.Z.min())/(self.Z.size-1)
        _dx = (self.R.max()-self.R.min())/(self.R.size-1)
        for tidx in range(self.time.size):
            _dpsiN = self.psiN[tidx, :, :]
            mask = (_dpsiN < 1.)
            mask[self.Z > self.zxP[tidx, self.actX[tidx]], :] = False
            _dummy = self.pRad[tidx, :, :]*mask
            emissivity[tidx] = numpy.trapz(2 * scipy.pi *
                                           numpy.trapz(
                                               _dummy,
                                               dx=_dz, axis=0),
                                           dx=_dx)
        return emissivity

    def PrivateFluxLfs(self):
        emissivity = numpy.zeros(self.time.size)
        _dz = (self.Z.max()-self.Z.min())/(self.Z.size-1)
        _dx = (self.R.max()-self.R.min())/(self.R.size-1)
        # we must find the linear interpolation
        # between the axctive X-point and
        # the magnetic axis and the position at the
        # intercept at the vessel
        X, Y = numpy.meshgrid(self.R, self.Z)
        XY = numpy.dstack((X, Y)).reshape((-1, 2))
        for tidx in range(self.time.size):
            _dpsiN = self.psiN[tidx, :, :]
            maskB = (_dpsiN < 1.)
            maskB[self.Z > self.zxP[tidx, self.actX[tidx]], :] = False
            try:
                path = self._pathBelowXpMaLine(tidx)
                maskA = path.contains_points(XY).reshape(X.shape)
            except:
                maskA = numpy.ones(X.shape, dtype=bool)
                # this is the mask for the Private Flux
            mask = maskA & maskB
            _dummy = self.pRad[tidx, :, :]*mask
            emissivity[tidx] = numpy.trapz(2 * scipy.pi *
                                           numpy.trapz(
                                               _dummy,
                                               dx=_dz, axis=0),
                                           dx=_dx)
        return emissivity

    def PrivateFluxHfs(self):
        emissivity = numpy.zeros(self.time.size)
        _dz = (self.Z.max()-self.Z.min())/(self.Z.size-1)
        _dx = (self.R.max()-self.R.min())/(self.R.size-1)
        # we must find the linear interpolation
        # between the axctive X-point and
        # the magnetic axis and the position at the
        # intercept at the vessel
        X, Y = numpy.meshgrid(self.R, self.Z)
        XY = numpy.dstack((X, Y)).reshape((-1, 2))
        for tidx in range(self.time.size):
            _dpsiN = self.psiN[tidx, :, :]
            maskB = (_dpsiN < 1.)
            maskB[self.Z > self.zxP[tidx, self.actX[tidx]], :] = False
            try:
                path = self._pathAboveXpMaLine(tidx)
                maskA = path.contains_points(XY).reshape(X.shape)
            except:
                maskA = numpy.ones(X.shape, dtype=bool)
                # this is the mask for the Private Flux
            mask = maskA & maskB
            _dummy = self.pRad[tidx, :, :]*mask
            emissivity[tidx] = numpy.trapz(2 * scipy.pi *
                                           numpy.trapz(
                                               _dummy,
                                               dx=_dz, axis=0),
                                           dx=_dx)
        return emissivity

    def Xpoint(self, radius=0.05):
        """
        Define the radiation in a circula region
        around the X-point which can be defined with a
        parameter radius (default is 2cm)
        """
        emissivity = numpy.zeros(self.time.size)
        _dz = (self.Z.max()-self.Z.min())/(self.Z.size-1)
        _dx = (self.R.max()-self.R.min())/(self.R.size-1)
        # intercept at the vessel
        X, Y = numpy.meshgrid(self.R, self.Z)
        for tidx in range(self.time.size):
            # find the radial and vertical position
            # of the xpoint
            xP = self.rxP[tidx, self.actX[tidx]]
            zP = self.zxP[tidx, self.actX[tidx]]
            if (numpy.isnan(xP)) or (numpy.isnan(zP)):
                emissivity[tidx] = numpy.nan
            else:
                distance = numpy.sqrt((X-xP)**2 + (Y-zP)**2)
                _dummy = numpy.ma.masked_where(distance > radius,
                                               self.pRad[tidx, :, :])
                emissivity[tidx] = numpy.trapz(2 * scipy.pi *
                                               numpy.trapz(
                                                   _dummy,
                                                   dx=_dz, axis=0),
                                               dx=_dx)
        return emissivity

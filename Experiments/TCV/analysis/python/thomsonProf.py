import tcv
from tcv.diag.general import General
import eqtools
import matplotlib as mpl
import numpy
import scipy
import MDSplus as mds


def arrowRespond(slider, event):
    if event.key == 'right':
        slider.set_val(min(slider.val + 1, slider.valmax))
    if event.key == 'left':
        slider.set_val(max(slider.val - 1, slider.valmin))


# routine to create a plot for thomson profile
# as a function of rho
# with slices
# first of all load the equilibriym
try:
    shot = numpy.uint(raw_input('Enter shot number: '))
except:
    print('Wrong shot number ')
remote = True

eq = eqtools.TCVLIUQETree(shot)
# load the data from thomson, rawdata and
Tree = mds.Tree('tcv_shot', shot)
rPos = Tree.getNode(
    r'\diagz::thomson_set_up:radial_pos').data()
zPos = Tree.getNode(
    r'\diagz::thomson_set_up:vertical_pos').data()
# now the thomson times
times = Tree.getNode(r'\results::thomson:times').data()
# now the thomson raw data
dataTe = Tree.getNode(r'\results::thomson:te').data()
errTe = Tree.getNode(r'\results::thomson:te:error_bar').data()
dataEn = Tree.getNode(r'\results::thomson:ne').data()
errEn = Tree.getNode(r'\results::thomson:ne:error_bar').data()
# now if available load the profiles of thomson from proffit
try:
    profTe = Tree.getNode(r'\results::thomson.profiles.auto:te').data()
    profEn = Tree.getNode(r'\results::thomson.profiles.auto:ne').data()
    rhoPr = Tree.getNode(r'\results::thomson.profiles.auto:rho').data()
    timePr = Tree.getNode(r'\results::thomson.profiles.auto:time').data()
    # get the normalization factor between the FIR and the Thomson
    # limit to the timing of the Prof
    dataTe = dataTe[((times >= timePr.min()) & (times <= timePr.max())), :]
    dataEn = dataEn[((times >= timePr.min()) &
                     (times <= timePr.max())), :]
    errTe = errTe[((times >= timePr.min()) & (times <= timePr.max())), :]
    errEn = errEn[((times >= timePr.min()) & (times <= timePr.max())), :]
    # limit also the timing
    times = timePr
    print 'Loaded also the fitted profiles '
    profit = True
except:
    print 'Fitted profiles not found '
    profit = False
# load the line average density
enAvg = General.neline(shot)
# now the plot. We copy the way it works in
psiRZ = eq.getFluxGrid()
rGrid = eq.getRGrid(length_unit='m')
zGrid = eq.getZGrid(length_unit='m')
t = eq.getTimeBase()
RLCFS = eq.getRLCFS(length_unit='m')
ZLCFS = eq.getZLCFS(length_unit='m')
limx, limy = eq.getMachineCrossSection()
# the patches for the plot
tilesP, vesselP = eq.getMachineCrossSectionPatch()

# create the appropriate Figure
mainPlot = mpl.pyplot.figure(figsize=(15, 10))
psi = mpl.pylab.axes([0.1, 0.25, 0.4, 0.7])
psi.set_aspect('equal')
psi.add_patch(tilesP)
psi.add_patch(vesselP)
psi.set_xlim([0.6, 1.2])
psi.set_ylim([-0.8, 0.8])
# first dummy plot of the psi
CS = psi.contour(rGrid, zGrid, psiRZ[0], 50, colors = 'k')
# add the time slider
timeSliderSub = mpl.pylab.axes(
    [0.1, 0.1, 0.4, 0.025], axisbg = 'lightgoldenrodyellow')
# add the plot of the line integrated density
enPlot = mpl.pylab.axes([0.55, 0.25, 0.4, 0.32])
enPlot.plot(enAvg[enAvg.dims[0]],enAvg.values/1e19, 'b')
enPlot.set_ylabel(r'n$_e$ [10$^{19}$ fringes]', fontsize = 16)
# add a dashed line with the first time of thomson data chosen
axv = enPlot.axvline(times[0], linestyle ='--', color = 'orange')
# add a dashed line with the first time of Liuqe reconstruction

# now we add the panel with the profile of thomson density
thPlot = mpl.pylab.axes([0.55, 0.63, 0.4, 0.32])
# we recompute in psirho the position of the Thomson data at the first point

rho = eq.rz2psinorm(rPos, zPos, times[0], sqrt=True)
pr, = thPlot.plot(rho[dataEn[:, 0] != -1],
                  dataEn[dataEn[:, 0] != -1, 0]/1e19, 'o',
                  color='orange', markersize = 10)
thPlot.set_ylim([0, dataEn.max()/1e19])
thPlot.set_xlim([0.7, 1.2])
thPlot.axvline(1, linestyle='--', color='black')
thPlot.set_xlabel(r'$\rho$')
thPlot.set_ylabel(r'n$_e$ [10$^{19}$ m$^{-3}$]')
if profit == True:
    prF, = thPlot.plot(rhoPr, profEn[:, 0], '--b', linewidth=1.2)
# now we start to update the slider goes for
# the thomson profile and update
# consequently the psi with the closest value
def updateTime(val):
    # clear the figure
    # enPlot.clear()
    psi.clear()

    # check the timining on the slider
    t_idx = int(timeSlider.val)
    # redo the plot for the psi
    # first find the closest index to the t_idx
    indPsi = numpy.argmin(numpy.abs(eq.getTimeBase()-times[t_idx]))
    psi.set_xlim([0.5, 1.2])
    psi.set_ylim([-0.8, 0.8])
    psi.set_title('LIUQE Reconstruction, $t = %(t).2f$ s' % {'t': eq.getTimeBase()[indPsi]})
    psi.set_xlabel('$R$ [m]')
    psi.set_ylabel('$Z$ [m]')
    # # catch NaNs separating disjoint sections of R,ZLCFS in mask
    maskarr = scipy.where(scipy.logical_or(RLCFS[indPsi] > 0.0,
                                           scipy.isnan(RLCFS[indPsi])))
    RLCFSframe = RLCFS[indPsi, maskarr[0]]
    ZLCFSframe = ZLCFS[indPsi, maskarr[0]]
    psi.plot(RLCFSframe, ZLCFSframe, 'r', lw=3, zorder=3)
    psi.contour(rGrid, zGrid, psiRZ[indPsi], 50, colors='k')
    psi.add_patch(tilesP)
    psi.add_patch(vesselP)
    psi.set_xlim([0.5, 1.2])
    psi.set_ylim([-0.8, 0.8])

    # redo the plot for the density
    # enPlot.plot(enAvg[enAvg.dims[0]],enAvg.values/1e19, 'b')
    # enPlot.set_ylabel(r'n$_e$ [10$^{19}$ fringes]', fontsize = 16)
    # # add a dashed line with the first time of thomson data chosen
    # enPlot.axvline(times[t_idx], linestyle ='--', color = 'orange')
    axv.set_xdata(times[t_idx])
    # redo the plot for the profile
    rho = eq.rz2psinorm(rPos, zPos, times[t_idx],sqrt=True)
    pr.set_xdata(rho[dataEn[:, t_idx] != -1])
    pr.set_ydata(dataEn[dataEn[:, t_idx] != -1, t_idx]/1e19)
    thPlot.set_ylim([0, dataEn[:, t_idx].max()/1e19])
    mainPlot.canvas.draw_idle()


timeSlider = mpl.widgets.Slider(timeSliderSub, 't index', 0, len(times) - 1, valinit = 0, valfmt = "%d")
timeSlider.on_changed(updateTime)
#updateTime(0)

mpl.pyplot.ion()
mainPlot.show()

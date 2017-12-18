# script to evaluate the blob size at different position
# for all the shots so far included in the Topic-21

import tcvFilaments
import langmuir
import numpy as np
import MDSplus as mds
import pandas as pd
import warnings
import numpy as np
from os import listdir
from os.path import isfile, join
mypath = '/home/vianello/NoTivoli/work/topic21/Experiments/TCV/data/tree'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
shots = []
for f in onlyfiles:
    try:
        shots.append(int(f[12:17]))
    except:
        pass

shotList = np.unique(np.asarray(shots)).astype('int')
# limit to L-Mode plasmas
shotList = shotList[np.where(shotList < 58640)[0]]
Shots = np.asarray([])
# quantity
Ip = np.asarray([])
Rho = np.asarray([])
AvDens = np.asarray([])
LambdaDiv = np.asarray([])
ThetaDiv = np.asarray([])
Tau = np.asarray([])
vR = np.asarray([])
vP = np.asarray([])
vPExB = np.asarray([])
Rhos = np.asarray([])
Cs = np.asarray([])
Size = np.asarray([])
# error
IpErr = np.asarray([])
AvDensErr = np.asarray([])
LambdaDivErr = np.asarray([])
ThetaDivErr = np.asarray([])
TauErr = np.asarray([])
vRErr = np.asarray([])
vPErr = np.asarray([])
vPExBErr = np.asarray([])
RhosErr = np.asarray([])
SizeErr = np.asarray([])
Efold = np.asarray([])
EfoldErr = np.asarray([])
for shot in shotList:
    Tree = mds.Tree('tcv_shot', shot)
    iP = mds.Data.compile(r'tcv_ip()').evaluate()
    iPTime = iP.getDimensionAt().data()
    enAVG = Tree.getNode(r'\results::fir:n_average')
    enAVGTime = enAVG.getDimensionAt().data()
    Data = tcvFilaments.Turbo(shot)
    for plunge in (1, 2):
        for r in np.arange(0, 0.025, 0.005):
            try:
                Blob = Data.blob(
                    plunge=plunge,
                    rrsep=[r, r+0.005],
                    iwin=75, rmsNorm=True,
                    detrend=True)
                Found = True
            except:
                Found = False
                print('Not computed for shot %5i' % shot +
                      'plunge %1i' % plunge)

            if Found:    
                if np.isfinite(Blob.vAutoP):
                    _size = Blob.FWHM*np.sqrt(
                        Blob.vrExB**2 + Blob.vpExB**2)/Blob.rhos
                    _dSize = _size*np.sqrt(
                        (Blob.FWHMerr/Blob.FWHM)**2 +
                        (Blob.vrExBerr/Blob.vrExB)**2 +
                        (Blob.vAutoPErr/Blob.vAutoP)**2)
                else:
                    _size = Blob.FWHM*np.sqrt(
                        Blob.vrExB**2 + Blob.vpExB**2)/Blob.rhos
                    _dSize = _size*np.sqrt(
                        (Blob.FWHMerr/Blob.FWHM)**2 +
                        (Blob.vrExBerr/Blob.vrExB)**2 +
                        (Blob.vAutoPErr/Blob.vpExB)**2)

                Shots = np.append(Shots, shot)
                Ip = np.append(Ip, np.abs(
                    iP.data()[
                        np.where(
                            ((iPTime >= Blob.tmin - 0.1) &
                             (iPTime <= Blob.tmax + 0.1)))[0]]).mean())
                AvDens = np.append(
                    AvDens,
                    enAVG.data()[
                        np.where(
                            ((enAVGTime >= Blob.tmin - 0.1) &
                             (enAVGTime <= Blob.tmax + 0.1)))[0]].mean())
                IpErr = np.append(IpErr, np.abs(
                    iP.data()[
                        np.where(
                            ((iPTime >= Blob.tmin - 0.1) &
                             (iPTime <= Blob.tmax + 0.1)))[0]]).std())
                AvDensErr = np.append(
                    AvDensErr,
                    enAVG.data()[
                        np.where(
                            ((enAVGTime >= Blob.tmin - 0.1) &
                             (enAVGTime <= Blob.tmax + 0.1)))[0]].std())
                Rho = np.append(Rho, Blob.Rho)
                LambdaDiv = np.append(LambdaDiv, Blob.LambdaDiv)
                ThetaDiv = np.append(ThetaDiv, Blob.ThetaDiv)
                Tau = np.append(Tau, Blob.FWHM)
                vR = np.append(vR, Blob.vrExB)
                vP = np.append(vP, Blob.vAutoP)
                vPExB = np.append(vPExB, Blob.vpExB)
                Rhos = np.append(Rhos, Blob.rhos)
                Size = np.append(Size, _size)
                Cs = np.append(Cs, Blob.Cs)
                Efold = np.append(Efold, Blob.Efold)
                # errors 
                LambdaDivErr = np.append(LambdaDivErr,
                                         Blob.LambdaDivErr)
                ThetaDivErr = np.append(ThetaDivErr, Blob.ThetaDivErr)
                TauErr = np.append(TauErr, Blob.FWHMerr)
                vRErr = np.append(vRErr, Blob.vrExBerr)
                vPErr = np.append(vPErr, Blob.vAutoPErr)
                vPExBErr = np.append(vPExBErr, Blob.vpExBerr)
                RhosErr = np.append(RhosErr, Blob.drhos)
                SizeErr = np.append(SizeErr, _dSize)
                EfoldErr = np.append(EfoldErr, Blob.EfoldErr)
                print('Computed for Shot %5i' % shot +' Plunge %1i' % plunge)
    Tree.quit()

# now we try to save appropriate pandas dataframe
outdict = {'Shots': Shots,
           'Ip': Ip,
           '<n_e>': AvDens, 'Rho': Rho, 'Lambda Div': LambdaDiv,
           'Theta Div':ThetaDiv, 'Blob Size [rhos]': Size,
           'Tau': Tau, 'vR': vR, 'vP': vP, 'vPExB': vPExB,
           'Rhos':Rhos, 'Cs':Cs,
           'Ip Err': IpErr, '<n_e> Err': AvDensErr,
           'Lambda Div Err':LambdaDivErr, 'Theta Div Err':ThetaDivErr,
           'Blob size Err [rhos]':SizeErr, 'Tau Err':TauErr, 'vR Err':vRErr,
           'vP Err':vPErr, 'vPExB Err':vPExBErr, 'Rhos Err':RhosErr,
           'Efold':Efold, 'EfoldErr':EfoldErr}
df = pd.DataFrame.from_dict(outdict)
df['Z'] = np.repeat(1, df.index.size)
df['Mu'] = np.repeat(2, df.index.size)
df['Conf'] = np.repeat('LSN', df.index.size)
# change only for thos in DN
shotDN = (58611, 58614, 58623, 58624)
for ss in shotDN:
    df['Conf'][df['Shots'] == ss] = 'DN'
# load existing database and merge them
df.to_csv('../../data/BlobDatabase.csv')

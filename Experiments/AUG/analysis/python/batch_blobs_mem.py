import pandas as pd
import augFilaments
import numpy as np
df = pd.read_csv('../data/MEM_Topic21.csv')
shotList = df['shot'].values
shotList = shotList[shotList> 34276]
for shot in shotList:
    D = df[df['shot'] == shot]
    # first load the augFilaments data
    Data = augFilaments.Filaments(shot, Xprobe=D['X'].values)
    # then iterate with check on the number of strokes
    for strokes in ('1', '2', '3', '4', '5', '6'):
        # this is the check in case the timing is defined
        if not np.isnan(D['tmin'+strokes].values[0]):
            print('Evaluating Data for shot %5i ' %shot + ' strokes number %1i' % int(strokes))
            tmin = D['tmin'+strokes].values[0]
            tmax = D['tmax'+strokes].values[0]
            # now distinguish the case with interELM or ELM
            if np.isnan(D['Threshold_' + strokes].values[0]):
                # no interelm
                print('Evaluating without ELM for shot %5i' %shot +
                      ' strokes number %1i' % int(strokes))
                out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                        block=[0.015, 1.5], normalize=True, detrend=True,
                                        trange=[tmin, tmax])
                out.to_netcdf('../data/Shot%5i' % shot + '_'+strokes+'Stroke.nc')
            else:
                print('Evaluating with ELM for shot %5i' %shot +
                      ' strokes number %1i' % int(strokes))
                thr = D['Threshold_'+strokes].values[0]
                out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                        block=[0.015, 1.5], normalize=True, detrend=True,
                                        trange=[tmin, tmax], interelm=True, threshold=thr)
                out.to_netcdf('../data/Shot%5i' % shot + '_'+strokes+'Stroke.nc')
        else:
            print('No strokes %1i ' % int(strokes) + 'for shot %5i' % shot)

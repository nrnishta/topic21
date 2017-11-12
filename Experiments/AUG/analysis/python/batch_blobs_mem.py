import pandas as pd
import augFilaments
import numpy as np
df = pd.read_csv('../data/MEM_Topic21.csv')
shotList = df['shot'].values
shotList = shotList[shotList> 34279]
for shot in shotList:
    # first load
    Data = augFilaments.Filaments(shot, Xprobe=df['X'][df['shot'] == shot].values)
    # first interval
    if shot <= 34108:
        # first one
        tmin = df['tmin1'][df['shot'] == shot].values
        tmax = df['tmax1'][df['shot'] == shot].values
        out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                block=[0.015, 1.5], normalize=True, detrend=True,
                                trange=[tmin, tmax])
        out.to_netcdf('../data/Shot%5i' % shot + '_1Stroke.nc')
        # second one
        tmin = df['tmin2'][df['shot'] == shot].values
        tmax = df['tmax2'][df['shot'] == shot].values
        out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                block=[0.015, 1.5], normalize=True, detrend=True,
                                trange=[tmin, tmax])
        out.to_netcdf('../data/Shot%5i' % shot + '_2Stroke.nc')
        # third one
        tmin = df['tmin3'][df['shot'] == shot].values
        tmax = df['tmax3'][df['shot'] == shot].values
        out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                block=[0.015, 1.5], normalize=True, detrend=True,
                                trange=[tmin, tmax])
        out.to_netcdf('../data/Shot%5i' % shot + '_3Stroke.nc')
        # 4 stroke
        if not np.isnan(df['tmin4'][df['shot'] == shot].values):
            tmin = df['tmin4'][df['shot'] == shot].values
            tmax = df['tmax4'][df['shot'] == shot].values
            out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                    block=[0.015, 1.5], normalize=True, detrend=True,
                                    trange=[tmin, tmax])
            out.to_netcdf('../data/Shot%5i' % shot + '_4Stroke.nc')
        # 5 stroke
        if not np.isnan(df['tmin5'][df['shot'] == shot].values):
            tmin = df['tmin5'][df['shot'] == shot].values
            tmax = df['tmax5'][df['shot'] == shot].values
            out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                    block=[0.015, 1.5], normalize=True, detrend=True,
                                    trange=[tmin, tmax])
            out.to_netcdf('../data/Shot%5i' % shot + '_5Stroke.nc')
        # 6 stroke
        if not np.isnan(df['tmin6'][df['shot'] == shot].values):
            tmin = df['tmin6'][df['shot'] == shot].values
            tmax = df['tmax6'][df['shot'] == shot].values
            out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                    block=[0.015, 1.5], normalize=True, detrend=True,
                                    trange=[tmin, tmax])
            out.to_netcdf('../data/Shot%5i' % shot + '_6Stroke.nc')
    # H-Mode
    else:
        # first one
        tmin = df['tmin1'][df['shot'] == shot].values
        tmax = df['tmax1'][df['shot'] == shot].values
        out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                block=[0.015, 1.5], normalize=True, detrend=True,
                                trange=[tmin, tmax], interELM=True, threshold=500)
        out.to_netcdf('../data/Shot%5i' % shot + '_1Stroke.nc')
        # second one
        tmin = df['tmin2'][df['shot'] == shot].values
        tmax = df['tmax2'][df['shot'] == shot].values
        out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                block=[0.015, 1.5], normalize=True, detrend=True,
                                trange=[tmin, tmax], interELM=True, threshold=500)
        out.to_netcdf('../data/Shot%5i' % shot + '_2Stroke.nc')
        # third one
        tmin = df['tmin3'][df['shot'] == shot].values
        tmax = df['tmax3'][df['shot'] == shot].values
        out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                block=[0.015, 1.5], normalize=True, detrend=True,
                                trange=[tmin, tmax], interELM=True, threshold=500)
        out.to_netcdf('../data/Shot%5i' % shot + '_3Stroke.nc')
        # 4 stroke
        if not np.isnan(df['tmin4'][df['shot'] == shot].values):
            tmin = df['tmin4'][df['shot'] == shot].values
            tmax = df['tmax4'][df['shot'] == shot].values
            out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                    block=[0.015, 1.5], normalize=True, detrend=True,
                                    trange=[tmin, tmax], interELM=True, threshold=500)
            out.to_netcdf('../data/Shot%5i' % shot + '_4Stroke.nc')
        # 5 stroke
        if not np.isnan(df['tmin5'][df['shot'] == shot].values):
            tmin = df['tmin5'][df['shot'] == shot].values
            tmax = df['tmax5'][df['shot'] == shot].values
            if shot == 34279:
                lim = 600
            else:
                lim = 300
            out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                    block=[0.015, 1.5], normalize=True, detrend=True,
                                    trange=[tmin, tmax], interELM=True, threshold=lim)
            out.to_netcdf('../data/Shot%5i' % shot + '_5Stroke.nc')
        # 6 stroke
        if not np.isnan(df['tmin6'][df['shot'] == shot].values):
            tmin = df['tmin6'][df['shot'] == shot].values
            tmax = df['tmax6'][df['shot'] == shot].values
            if shot == 34279:
                lim = 600
            else:
                lim = 300
            out = Data.blobAnalysis(Probe='Isat_m06', otherProbe=['Isat_m10', 'Isat_m07'],
                                    block=[0.015, 1.5], normalize=True, detrend=True,
                                    trange=[tmin, tmax], interELM=True, threshold=lim)
            out.to_netcdf('../data/Shot%5i' % shot + '_6Stroke.nc')

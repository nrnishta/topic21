import numpy as np
import matplotlib as mpl
from scipy.signal import savgol_filter
from time_series_tools import identify_bursts2
import dd
import pandas as pd
df = pd.read_csv('../data/MEM_Topic21.csv')
shotList = df['shot'].values
shotList = shotList[shotList >= 34107]

for shot in shotList:
    D = df[df['shot'] == shot]
    tminL = np.asarray([D['tmin%i' % k].values[0] for k in np.linspace(1, 6, 5, dtype='int')])
    tmaxL = np.asarray([D['tmax%i' % k].values[0] for k in np.linspace(1, 6, 5, dtype='int')])    
    thrL =  np.asarray([D['Threshold_%i' % k].values[0] for k in np.linspace(1, 6, 5, dtype='int')])
    # avoid all the nan
    tmaxL = tmaxL[np.isfinite(tminL)]
    thrL = thrL[np.isfinite(tminL)]
    tminL = tminL[np.isfinite(tminL)]
    Mac = dd.shotfile('MAC', shot, experiment='AUGD')
    Ipol = Mac('Ipolsoli')
    fig, ax = mpl.pylab.subplots(figsize=(10, 14), nrows=5, ncols=1)
    fig.subplots_adjust(wspace=0.2)
    for _tmn, _tmx, _thr, _iax in zip(tminL, tmaxL, thrL, range(len(tminL))):
        _idx = np.where((Ipol.time >= _tmn) & (Ipol.time <= _tmx))[0]
        t = Ipol.time[_idx]
        y = Ipol.data[_idx]
        yS = savgol_filter(y, 501, 3)

        ax[_iax].plot(t, y, 'gray', alpha=0.5)
        ax[_iax].plot(t, yS)
        ax[_iax].axhline(_thr, ls='--', color='b', lw=2)
        ax[_iax].set_xlabel(r't [s]')
        ax[_iax].set_ylabel(r'Ipolsoli')
        if np.isfinite(_thr):
            window, _a, _b, _c = identify_bursts2(yS, _thr)
            _idx, _idy = list(zip(*window))
            tBegElm = t[np.asarray(_idx)]
            tEndElm = t[np.asarray(_idy)]
            for i in range(tBegElm.size-1):
                if yS[0] < _thr:
                    _aa = np.where((t >= tEndElm[i]) & (t <= tBegElm[i+1]))[0]
                else:
                    _aa = np.where((t >= tBegElm[i]) & (t <= tEndElm[i]))[0]

                ax[_iax].plot(t[_aa], y[_aa], 'orange', alpha=0.7)

    ax[0].set_title(r'# %5i' % shot)
    fig.savefig('../pdfbox/ExampleInteELMShot%5i' % shot +'.pdf', bbox_to_inches='tight')

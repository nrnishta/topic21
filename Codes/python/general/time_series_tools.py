import numpy as np

def get_moments(series,moments=(1,2,3,4)):
    """
    Wrapper around scipy to produce time series moments
    
    args:
        series --> 1D time series

    kwargs:
        moments --> tuple containing order of all moments to be returned

    return:
        moments --> list of moments of time-series
                    NOTE: order of list is always consequative
    """
    from scipy.stats import describe
    nobs,minmax,mean,var,skew,kurt = describe(series)
    returns = []
    if 1 in moments:
        returns.append(mean)
    if 2 in moments:
        returns.append(var)
    if 3 in moments:
        returns.append(skew)
    if 4 in moments:
        returns.append(kurt)
    return returns

def identify_bursts2(x,thresh,analyse=True):
    """
    
    Function that identifies windows in the time series
    where a burst with an amplitude above the threshold 
    occurs.
    
    args:
        x --> 1D time series to be analysed
        thresh --> theshold above which bursts will be detected

    kwargs:
        analyse = True,False    provide additional analysis of the timeseries

    returns: 
        windows --> a list of tuples that contain indices that
                    bracket each burst detected

    if analyse is True, then also return
        N_bursts --> total number of burst detections   
        burst_ratio --> N_samples/N_bursts
        av_window --> mean window size of a burst
    """
    crossings = np.where(np.diff(np.signbit(x-thresh)))[0]
    windows = list(zip(crossings[::2],crossings[1::2]+1))

    if analyse:
        N_bursts = len(windows)
        burst_ratio = len(list(x))/N_bursts
        av_window = np.mean([y - x for x,y in windows])
        return windows,N_bursts,burst_ratio,av_window
    else:
        return windows

def conditionally_average(series,nw,thresh,baseline=None,verbose=True,full_output=False,plot=False):
    """
    Function to average the wave-form of bursts in a time-series
    above some threshold.

    args:
        series --> the 1D time-series
        nw --> window size to conduct the average around
        thresh --> threshold to identify bursts

    kwargs:
        verbose = True --> Print additional info to screen
        baseline = None --> if set to a number, treat baseline as an initial threshold, then search for windows containing features above thresh
        plot = False --> If True plot each individual profile and show
        full_output = False --> output av, amps and inds

    return:
        av --> average burst profile
        amps --> amplitudes of detected filaments, only returned if full_output=True
        inds --> indices of detected filaments, only returned if full_output=True
    """

    #Need window to be an odd number
    if nw%2 == 0: nw += 1

    #Identify bursts
    if baseline is None:
        windows,Nbursts,ratio,av_width = identify_bursts2(series,thresh,analyse=True)
    else:
        windows_bl,Nbursts,ratio,av_width = identify_bursts2(series,baseline,analyse=True)
        windows = [window for window in windows_bl if np.max(series[window[0]:window[1]]) > thresh]


    import matplotlib.pyplot as plt
    if verbose:
        print("####### conditional averaging info #######")
        print("\tBursts identified: \t"+str(Nbursts))
        print("\tSamples per burst: \t"+str(ratio))
        print("\tAverage burst window: \t"+str(av_width))

    av = np.zeros(nw)
    amps = []
    inds = []
    for window in windows:
        #print(window)
        try:
            ind_max = np.where(series[window[0]:window[1]]==np.max(series[window[0]:window[1]]))[0][0]
            av += series[window[0] + ind_max - (nw-1)/2 : window[0] + ind_max + (nw-1)/2 + 1]/np.max(series[window[0] + ind_max - (nw-1)/2 : window[0] + ind_max + (nw-1)/2 + 1])
            if plot: plt.plot(series[window[0] + ind_max - (nw-1)/2 : window[0] + ind_max + (nw-1)/2 + 1]/np.max(series[window[0] + ind_max - (nw-1)/2 : window[0] + ind_max + (nw-1)/2 + 1]),'x',alpha=0.1)
            if full_output:
                amps.append(series[window[0]+ind_max])
                inds.append(window[0]+ind_max)
        except:
            pass
    av /= len(windows)
    if plot: plt.show()
    if verbose:
        print("\tWave-form duration: \t"+str(np.sum((av/np.max(av))[np.where(av/np.max(av) > 0.1)])))
    if full_output:
        return av,amps,inds
    else:
        return av

def coarse_grain(signal,factor,dx=None,integrate=False):
    if not integrate:
        new_sig = signal[::factor]
    else:
        if factor % 2 == 0: nx = factor + 1
        else: nx = factor
        new_sig = np.zeros(signal[::factor].shape)
        if dx is None:
            dx = np.ones(signal.shape)/(factor + 1*(nx%2))
        new_sig += (signal*dx)[::factor]
        for i in np.int32(np.arange((nx-1)/2) + 1):
            new_sig += np.roll(signal*dx,i)[::factor]
            new_sig += np.roll(signal*dx,-i)[::factor]

    return new_sig

def signed_diff(signal,window):
    #renormalize signal
    signal = signal - np.mean(signal)
    signal = signal/np.std(signal)

    snf = np.zeros(signal.shape[0])
    for i in np.arange(window):
        snf -= np.roll(signal,i) + np.roll(signal,-i)
    snf /= 2*window
    snf += signal
    return snf

def significance(signal,window):
    #renormalize signal
    signal = signal - np.mean(signal)
    signal = signal/np.std(signal)
    prevnegs = np.zeros(signal.shape[0])
    prevposs = np.zeros(signal.shape[0])
    snf = np.zeros(signal.shape[0])
    for i in np.arange(1,window):
        negs = signal - np.roll(signal,i)
        poss = signal - np.roll(signal,-i)
        prevnegs[np.where(negs>prevnegs)] = negs[np.where(negs>prevnegs)]
        prevposs[np.where(poss>prevposs)] = poss[np.where(poss>prevposs)]
    snf = 0.5*(prevnegs + prevposs)
    return snf
    
def compute_waiting_times(inds):
    #Input is inds from the burst identification algorithm
    return [inds[i+1] - inds[i] for i in np.arange(len(inds)-1)]

def acf(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    xx = x-x.mean()
    r = np.correlate(xx, xx, mode = 'full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

    
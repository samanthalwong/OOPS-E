import numpy as np
from matplotlib import pyplot as plt
from astropy.timeseries import LombScargle
from scipy import stats
from datetime import date
from astropy.time import Time
from scipy.interpolate import splev, splrep
from datetime import date
import pint.models as models
from scipy.stats import combine_pvalues
from scipy.optimize import curve_fit

def timecuts(file,data):
    """
    Applies timecuts (including pixel suppression) and mean subtracts + flips data. Takes raw ECM
    timeseries and returns a timeseries ready for processing.

    Parameters
    ----------
    file: a timecuts SSV file with columns [start stop]
    data: an array or tuple with [time, 1tel data]

    Returns
    -------
    time: the time array with time cuts
    tel: the telescope data array with time cuts, mean subtraction, and flipping
    """

    time,tel = data
    start,stop = np.genfromtxt(file,delimiter=' ',unpack=True)
    for i in range(len(start)):
        tel = tel[(time<start[i]) | (time>stop[i])]
        time = time[(time<start[i]) | (time>stop[i])]
    tel = -(tel - np.mean(tel))
    return time, tel

def timecuts_from_array(array,data):
    """
    Applies timecuts (including pixel suppression) and mean subtracts + flips data. Takes raw ECM
    timeseries and returns a timeseries ready for processing.

    Parameters
    ----------
    array: an array of timecuts with [(start1,stop1),(start2,stop2),...]
    data: an array or tuple with [time, 1tel data]

    Returns
    -------
    time: the time array with time cuts
    tel: the telescope data array with time cuts, mean subtraction, and flipping
    """
    time,tel = data
   
    for cut in array:
        start,stop = cut
        tel = tel[(time<start) | (time>stop)]
        time = time[(time<start) | (time>stop)]
    tel = -(tel - np.mean(tel))
    return time, tel

def stitch_runs(data,start,stop):
    time,tel = data
    tel = np.hstack((tel[(time<start)],tel[(time>stop)]))
    time = np.hstack((time[(time<start)],time[(time>stop)]))
    return time,tel

def cut_middle(file,data):
    time,tel = data
    start,stop = np.genfromtxt(file,delimiter=' ',unpack=True)
    for i in range(len(start)):
        tel = tel[(time<start[i]) | (time>stop[i])]
        time = time[(time<start[i]) | (time>stop[i])]
    return time, tel

def get_spacing(sampling, length):
    """
    Gets L-S spacing

    Parameters
    ----------
    sampling: ECM sampling rate in Hz
    length: length of data array

    Returns
    -------
    spacing: spacing for L-S
    """
    fftfreq = np.fft.rfftfreq(length,1/sampling)
    spacing = fftfreq[1]-fftfreq[0]
    return spacing

def chunk_data(nchunks,dur,data,time,sampling=2400):
    """
    Converts continuous data into nchunks smaller runs

    Parameters
    ----------
    nchunks: number of sub-runs
    dur: the duration of each sub-run
    data: a single telescope data array
    time: time array

    Returns
    -------
    chunks: an array of sub-run arrays
    chunk_times: the time arrays of each sub-run
    chunk_spacing: the L-S spacing for the sub-runs
    """
    chunks = []
    last_idx = 0
    tmin = time[0]

    for i in range(1,nchunks+1):
        idx = np.where(time >= i*dur + tmin)[0][0]
        chunks.append([data[last_idx:idx]] - np.mean([data[last_idx:idx]]))
        last_idx = idx
        
    chunk_times = 1/sampling * np.arange(0,len(chunks[0][0]))
    chunk_spacing = get_spacing(sampling,len(chunks[0][0]))
    return chunks, chunk_times, chunk_spacing

def estimate_period(run_date, f, fdot, mjd):
    """
    Estimates the pulsar period at the time of observations

    Parameters
    ----------
    run_date: the date of the observations as an astropy date object (date(yyyy,mm,dd))
    f: the pulsar frequency [Hz]
    fdot: the first derivative of the pulsar frequency {s/s}
    mjd: the MJD associated with the pulsar ephemeris

    Returns
    -------
    f_new: the estimated frequency
    """

    iso = date.isoformat(run_date)
    today_mjd = Time(iso).mjd
    dt = (today_mjd-mjd)*24*60*60
    f_new = f + fdot*dt
    return f_new

def clean_spl(time,signal,sigma=6,deg=10,plot=True):
    xnew = np.linspace(time[0],time[-1],deg)
    spl = splrep(time,signal)
    if plot:
        plt.plot(time,signal)
    clean = splev(xnew,spl)
    if plot:
        plt.plot(xnew,splev(xnew,spl))
        plt.show()
    splnew = splrep(xnew,clean)
    clean2 = splev(time,splnew)
    resid = signal-clean2
    thres = np.median(resid)
    if plot:
        plt.plot(time,resid)
        plt.axhline(thres,color='r')
        plt.axhline(thres+sigma*np.std(resid),color='k')
        plt.axhline(thres-sigma*np.std(resid),color='k')
        plt.show()
    std = np.std(resid)
    resid[resid > thres+(sigma*std)] = thres+(sigma*std)
    resid[resid < thres-(sigma*std)] = thres-(sigma*std)

    return resid

def calc_p(time,signal,ephemeris,tel,spacing,shift=0,samp=2400,plot=True):
    ps = np.array(())
    pvals = np.array(())
        
    p = ephemeris

    frequency,power = LombScargle(time, signal,1/samp).autopower(minimum_frequency=p-4,maximum_frequency=p+4,samples_per_peak=1)
        
    maxf = 4 #for high f
    minf = 0.5 #for high f

    badfreqs = [15,30,60,120,42.5,41.6]
                        
    off = np.where(((frequency < p + maxf) & (frequency > p + minf)) | ((frequency > p - maxf) & (frequency < p - minf)))[0]

    #mask out known noise peaks
    maskf = np.ones(len(off),dtype=bool)
    for bad in badfreqs:
        #print(frequency[np.where((frequency > bad - 0.1) & (frequency < bad + 0.1))[0]])
        maskf[np.where((frequency[off] > bad - 0.1) & (frequency[off] < bad + 0.1))[0]] = 0

    off = off[maskf]

    #print('period = ',p+shift)
    on = np.where((frequency > p + shift - spacing/2) & (frequency < p + shift + spacing/2))[0]
    if len(on) > 1:
        print('Too many values in ON region')
        return 1

    norm = np.std(power[off])/2
        
    mask = np.zeros(len(frequency))
    mask[off] = 1
                        
    P = stats.chi2.sf(max(power[on])/norm, 2)
    
    if plot:
            plt.clf()
            plt.figure(figsize=(8,6))
            plt.plot(frequency,power,'k')
            plt.axvline(p,color='k',alpha=0.2,linestyle='--',label='Pulse Frequency')
            plt.axvspan(frequency[on][0],frequency[on][-1],alpha=0.5,color='g',label='ON region')
            plt.fill_between(frequency, 0, max(power), where=mask, color='r', alpha=0.5,label='OFF region')
            plt.ticklabel_format(useOffset=False)
            plt.title(f'T{tel} L-S Periodogram')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [A.U.]')
            #plt.xlim(p-0.5, p+0.5)

            plt.legend()
            plt.show()
            
            plt.figure(figsize=(8,6))
            plt.plot(frequency,power,'k')
            plt.axvline(p,color='k',alpha=0.2,linestyle='--',label='Pulse Frequency')
            plt.axvspan(frequency[on][0],frequency[on][-1],alpha=0.5,color='g',label='ON region')
            plt.fill_between(frequency, 0, max(power), where=mask, color='r', alpha=0.5,label='OFF region')
            plt.ticklabel_format(useOffset=False)
            plt.title(f'T{tel} L-S Periodogram')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [A.U.]')
            plt.xlim(p-spacing,p+spacing)
            plt.legend()
            plt.show()

    return P

def gumbel_CDF(x,mu=3.72,b=1.83):
    z = ((x - mu)/b)
    return np.exp(-np.exp(-z))

def calc_p_gumball(time,signal,ephemeris,tel,spacing,samp=2400,numpoints=6,plot=True):
    ps = np.array(())
    pvals = np.array(())
        
    p = ephemeris

    frequency,power = LombScargle(time, signal,1/samp).autopower(minimum_frequency=p-4,maximum_frequency=p+4,samples_per_peak=1)
        
    maxf = 4 #for high f
    minf = 0.5 #for high f

    badfreqs = [15,30,60,120,42.5,41.6]
                        
    off = np.where(((frequency < p + maxf) & (frequency > p + minf)) | ((frequency > p - maxf) & (frequency < p - minf)))[0]

    #mask out known noise peaks
    maskf = np.ones(len(off),dtype=bool)
    for bad in badfreqs:
        #print(frequency[np.where((frequency > bad - 0.1) & (frequency < bad + 0.1))[0]])
        maskf[np.where((frequency[off] > bad - 0.1) & (frequency[off] < bad + 0.1))[0]] = 0

    off = off[maskf]

    on = np.where((frequency > p - spacing * (numpoints/2)) & (frequency < p + spacing*(numpoints/2)))[0]
    #on_small = np.where((frequency > p + shift - spacing) & (frequency < p + shift + spacing))[0]
    
    norm = np.std(power[off])/2
        
    mask = np.zeros(len(frequency))
    mask[off] = 1
    
    pval = max(power[on])
    #single_pval = max(power[on_small])
    mu = 2.02632506 * np.log(numpoints)-0.1137353
    beta = 1.47166634 * (numpoints/(1+numpoints))+0.53850982
    P = 1-gumbel_CDF(pval/norm,mu=mu,b=beta)
    
    if plot:
            plt.clf()
            plt.figure(figsize=(8,6))
            plt.plot(frequency,power,'k')
            plt.axvline(p,color='k',alpha=0.2,linestyle='--',label='Pulse Frequency')
            plt.axvspan(frequency[on][0],frequency[on][-1],alpha=0.5,color='g',label='ON region')
            plt.fill_between(frequency, 0, max(power), where=mask, color='r', alpha=0.5,label='OFF region')
            plt.ticklabel_format(useOffset=False)
            plt.title(f'T{tel} L-S Periodogram')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [A.U.]')
            #plt.xlim(p-0.5, p+0.5)

            plt.legend()
            plt.show()
            
            plt.figure(figsize=(8,6))
            plt.plot(frequency,power,'k',marker='o')
            plt.axvline(p,color='k',alpha=0.2,linestyle='--',label='Pulse Frequency')
            plt.axvspan(frequency[on][0],frequency[on][-1],alpha=0.5,color='g',label='ON region')
            plt.fill_between(frequency, 0, max(power), where=mask, color='r', alpha=0.5,label='OFF region')
            #plt.plot(freqfit,gauss(freqfit,*popt),label='Gaussian Fit')

            plt.ticklabel_format(useOffset=False)
            plt.title(f'T{tel} L-S Periodogram')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [A.U.]')
            plt.xlim(p-spacing*(numpoints/2)-spacing,p+spacing*(numpoints/2))
            plt.legend()
            plt.show()

    return P

def calc_fourier(time,signal,ephemeris,tel,spacing,samp=2400,numpoints=6):
    ps = np.array(())
    pvals = np.array(())
        
    p = ephemeris
        
    maxf = 4 #for high f
    minf = 0.5 #for high f

    badfreqs = [15,30,60,120,42.5,41.6]
    
    fft = np.fft.rfft(signal,norm='ortho')
    frequency = np.fft.rfftfreq(len(signal),1/samp)
                        
    off = np.where(((frequency < p + maxf) & (frequency > p + minf)) | ((frequency > p - maxf) & (frequency < p - minf)))[0]

    #mask out known noise peaks
    maskf = np.ones(len(off),dtype=bool)
    for bad in badfreqs:
        #print(frequency[np.where((frequency > bad - 0.1) & (frequency < bad + 0.1))[0]])
        maskf[np.where((frequency[off] > bad - 0.1) & (frequency[off] < bad + 0.1))[0]] = 0

    off = off[maskf]

    on = np.where((frequency > p - spacing * (numpoints/2)) & (frequency < p + spacing*(numpoints/2)))[0]
    #on_small = np.where((frequency > p + shift - spacing) & (frequency < p + shift + spacing))[0]
    
    return fft[on], fft[off]

def calc_p_crab(time,signal,ephemeris,tel,spacing,shift=0,samp=2400,plot=True):
    ps = np.array(())
    pvals = np.array(())
        
    p = ephemeris

    frequency,power = LombScargle(time, signal,1/samp).autopower(minimum_frequency=p-4,maximum_frequency=p+4,samples_per_peak=1)
        
    maxf = 4 #for high f
    minf = 0.5 #for high f

    badfreqs = [15,60,120]
                        
    off = np.where(((frequency < p + maxf) & (frequency > p + minf)) | ((frequency > p - maxf) & (frequency < p - minf)))[0]

    #mask out known noise peaks
    maskf = np.ones(len(off),dtype=bool)
    for bad in badfreqs:
        #print(frequency[np.where((frequency > bad - 0.1) & (frequency < bad + 0.1))[0]])
        maskf[np.where((frequency[off] > bad - 0.1) & (frequency[off] < bad + 0.1))[0]] = 0

    off = off[maskf]

    #print('period = ',p+shift)
    on = np.where(power == max(power))[0]
    if len(on) > 1:
        print('Too many values in ON region')
        return 1

    norm = np.std(power[off])/2
        
    mask = np.zeros(len(frequency))
    mask[off] = 1
                        
    P = stats.chi2.sf(max(power)/norm, 2)
    
    if plot:
            plt.clf()
            plt.figure(figsize=(8,6))
            plt.plot(frequency,power,'k')
            plt.axvline(p,color='k',alpha=0.2,linestyle='--',label='Pulse Frequency')
            plt.axvspan(frequency[on][0],frequency[on][-1],alpha=0.5,color='g',label='ON region')
            plt.fill_between(frequency, 0, max(power), where=mask, color='r', alpha=0.5,label='OFF region')
            plt.ticklabel_format(useOffset=False)
            plt.title(f'T{tel} L-S Periodogram')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [A.U.]')
            #plt.xlim(p-0.5, p+0.5)

            plt.legend()
            plt.show()
            
            plt.figure(figsize=(8,6))
            plt.plot(frequency,power,'k')
            plt.axvline(p,color='k',alpha=0.2,linestyle='--',label='Pulse Frequency')
            plt.axvspan(frequency[on][0],frequency[on][-1],alpha=0.5,color='g',label='ON region')
            plt.fill_between(frequency, 0, max(power), where=mask, color='r', alpha=0.5,label='OFF region')
            plt.ticklabel_format(useOffset=False)
            plt.title(f'T{tel} L-S Periodogram')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [A.U.]')
            plt.xlim(p-spacing,p+spacing)
            plt.legend()
            plt.show()


    return P, frequency[on], max(power)/np.std(power[off])

def calc_sigma(pvals):
    #p_combined = 0
    #for p in pvals:  
    #    p_combined += np.log(p)
    #p_combined = -2*p_combined
    p_combined = combine_pvalues(pvals,method='fisher').statistic
    overall_p = 1-stats.chi2.sf(p_combined, len(pvals)*2)
    #print(overall_p)
    #return stats.norm.ppf(1-overall_p)
    return stats.norm.ppf(overall_p)

def plot_cum_sig(pvals,chunk_dur):
    cumulative = []
    #ts = stats.chi2.sf(-2*np.log(pvals[0]),1)
    #cumulative.append(stats.norm.ppf(ts))
    for i,p in enumerate(pvals):
        cumulative.append(calc_sigma(pvals[:i]))
        #print(calc_sigma(pvals[:i]))
    cumulative.append(calc_sigma(pvals))

    time = np.arange(0,chunk_dur * (len(pvals)+1),chunk_dur)
    plt.plot(time,cumulative,'darkseagreen',marker='o',ls='-')
    plt.xlabel('Time (h)')
    plt.ylabel('Significance (sigma)')
    return

def get_p(file,date,diff=False):

    '''
    Uses PINT to get an up-to-date spin period

    Parameters
    ----------
    par: a .par or .eph file with F0, F1, F2, and PEPOCH
    date: a string formatted as 'yyyy-mm-dd hh:mm:ss' (time is optional)

    Returns
    -------
    p: period valid on date
    diff: difference between original spin period and new spin period
    '''

    m = models.get_model(file)

    F0_original = m.F0.quantity
    t = Time(date,scale='utc')
    date = t.mjd

    m.change_pepoch(date)
    if diff:
        return m.F0.quantity.value, (F0_original - m.F0.quantity).value
    else:
        return m.F0.quantity.value
    

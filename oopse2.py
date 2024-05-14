import numpy as np
from matplotlib import pyplot as plt
from astropy.timeseries import LombScargle
from scipy import stats
from datetime import date
from astropy.time import Time
from scipy.interpolate import splev, splrep

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
    splnew = splrep(xnew,clean)
    clean2 = splev(time,splnew)
    #plt.plot(time,signal2-clean2)
    resid = signal-clean2
    thres = np.median(clean2)
    if plot:
        plt.axhline(thres,color='r')
        plt.axhline(thres+2*np.std(resid),color='k')
        plt.axhline(thres-2*np.std(resid),color='k')
        plt.show()
    resid[resid > thres+(sigma*np.std(resid))] = thres+(sigma*np.std(resid))
    resid[resid < thres-(sigma*np.std(resid))] = thres-(sigma*np.std(resid))

    return resid

def calc_p(time,signal,ephemeris,tel,spacing,shift=0,samp=2400,plot=True):
    ps = np.array(())
    pvals = np.array(())
        
    p = ephemeris

    frequency,power = LombScargle(time, signal,1/2400).autopower(minimum_frequency=p-4,maximum_frequency=p+4,samples_per_peak=1)
        
    maxf = 4 #for high f
    minf = 0.5 #for high f
                        
    off = np.where(((frequency < p + maxf) & (frequency > p + minf)) | ((frequency > p - maxf) & (frequency < p - minf)))[0]
    
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
            plt.figure(figsize=(6,4))
            plt.plot(frequency,power,'k')
            plt.axvline(p,color='k',alpha=0.2,linestyle='--',label='Pulse Frequency')
            plt.axvspan(frequency[on][0],frequency[on][-1],alpha=0.5,color='g',label='ON region')
            plt.fill_between(frequency, 0, max(power), where=mask, color='r', alpha=0.5,label='OFF region')
            plt.ticklabel_format(useOffset=False)
            plt.title(f'T{tel} L-S Periodogram')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [A.U.]')
            #plt.xlim(p-0.5, p+0.5)

            #plt.legend()
            plt.show()

    return P

def calc_sigma(pvals):
    p_combined = 0
    for p in pvals:  
        p_combined += np.log(p)
    p_combined = -2*p_combined
    overall_p = stats.chi2.sf(p_combined, len(pvals)*2)
    #print(overall_p)
    return stats.norm.ppf(1-overall_p)

def plot_cum_sig(pvals,chunk_dur):
    cumulative = []
    #ts = stats.chi2.sf(-2*np.log(pvals[0]),1)
    #cumulative.append(stats.norm.ppf(ts))
    for i,p in enumerate(pvals):
        cumulative.append(calc_sigma(pvals[:i]))
        #print(calc_sigma(pvals[:i]))
    cumulative.append(calc_sigma(pvals))

    time = np.arange(0,chunk_dur * (len(pvals)+1),chunk_dur)
    plt.plot(time,cumulative,'k',marker='o',ls='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Sigma')
    return
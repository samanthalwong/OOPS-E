import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from astropy.timeseries import LombScargle
import yaml
import sys
from datetime import datetime
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm

matplotlib.use('Agg')

def get_ephemeris(date, time):

    """
    :param date: Run date
    :param time: Run time
    :return: Estimated ephemeris interpolated from Jodrell Bank values
    """
    from astropy.time import Time, TimeDelta
    from astropy.timeseries import TimeSeries
    import urllib.request
    import urllib
    from urllib.request import urlopen

    #convert date + time to mjd
    datetime = date[:4] + '-' + date[4:6] + '-' + date[6:] + 'T' + time
    datetime = Time(datetime,format='isot',scale='utc')
    datetime = datetime.mjd

    mjd = np.array(())
    p = np.array(())
    p_dot = np.array(())
    for line in urllib.request.urlopen("https://www.jb.man.ac.uk/pulsar/crab/all.gro"):
        line = line.decode('utf-8').split(' ')
        if line == ['\n']:
            continue
        mjd = np.append(mjd, line[9])
        p = np.append(p, line[13])
        p_dot = np.append(p_dot, line[14])

    mjd = mjd.astype(float)
    p = p.astype(float)
    p_dot = np.char.split(p_dot, 'D')
    p_dot = p_dot.tolist()
    p_dot = np.array(p_dot)
    pdot = p_dot[:, 0].astype(float) * 1e-10

    if date == 0:
        return p[-1]

    else:
        if datetime >= mjd[-1]:
            nearest = mjd[-1]
            nearest = Time(nearest,format='mjd')
            idx = -1
        else:
            nearest = mjd[np.where(mjd < datetime)[0][-1]]
            idx = np.where(mjd < datetime)[0][-1]

    nearest = Time(nearest, format='mjd')
    datetime = Time(datetime,format='mjd')

    diff = (datetime - nearest).sec

    ephemeris = float(p[idx]) + (diff) * float(pdot[idx])

    return ephemeris

def read_files(dir,date,sample,tels=3):

    """
    :param date: Run date
    :param sample: Sample rate for T2/T3/T4 - T1 is 1200 by default
    :param tels: # of telescopes used (3 by default since T1 usually saturates)
    :return: time,signals arrays of time and signal values for each telescope (single time array for T2/T3/T4)
    """
    dir = str(dir)
    date = str(date)

    t2 = dir + "/" + date + "-Crab-T2.csv"
    t3 = dir + "/" + date + "-Crab-T3.csv"
    t4 = dir + "/" + date + "-Crab-T4.csv"

    print('Reading T2')
    time2, signal2 = np.genfromtxt(t2, delimiter=',', unpack=True, usecols=(0, 1), skip_header=3)
    print('Reading T3')
    time3, signal3 = np.genfromtxt(t3, delimiter=',', unpack=True, usecols=(0, 1), skip_header=3)
    print('Reading T4')
    time4, signal4 = np.genfromtxt(t4, delimiter=',', unpack=True, usecols=(0, 1), skip_header=3)

    time =  1./sample * np.arange(0,len(signal2))

    if tels == 4:
        print('Reading T1')
        t1 = "/raid/romulus/swong/ecm/d" + date + "/" + date + "-Crab-T1.csv"
        time1, signal1 = np.genfromtxt(t1, delimiter=',', unpack=True, usecols=(0, 1), skip_header=3)
        sample1 = 1200
        time1 = 1. / sample1 * np.arange(0, len(time2))
        time = [time1,time]
        signals = [signal1,signal2, signal3, signal4]

    else:
        signals = [signal2, signal3, signal4]

    return time,signals

def calc_sig(signal,time,ephemeris):
    """
    :param signal: Array of signal values to L-S
    :param time: Array of time values
    :param ephemeris: Pulsar ephemeris at time of observation (using get_ephemeris is recommended)
    :return: sigma (float): significance of S/N
             peak (float): peak L-S power
             peak_freq (float): frequency of peak (used for calculating drift b/w tels)
             noise (array): array of noise values (for use in stacked significance)
    """
    frequency, power = LombScargle(time, signal, 1000).autopower(minimum_frequency=25, maximum_frequency=35,
                                                                     samples_per_peak=10)
    peak = max(power)
    noise = power[((frequency < ephemeris + 0.6) & (frequency > ephemeris + 0.1)) | (
                (frequency > ephemeris - 0.6) & (frequency < ephemeris - 0.1))]
    sigma = peak/np.std(noise)
    peak_freq = frequency[np.where(power == peak)[0][0]]
    return sigma, peak, peak_freq, noise

def cumulative_sig(s2,s3,s4,time,n,tmin,name,plot=True):
    """
    :param s2: T2 signal (array)
    :param s3: T3 signal (array)
    :param s4: T4 signal (array)
    :param time: Time array (float)
    :param n: Number of points at which significance is calculated
    :param tmin: Run start time (from parameter file) (float)
    :param name: Pulsar name (string)
    :param plot: If true, saves cumulative significance plot (boolean)
    :return: times,sigs arrays of times and significances at each time interval
    """

    print(f'Starting Cumulative Significance Calculation - {n} datapoints requested')
    detect = 0.9999994
    times = np.array(())
    significance = np.array(())

    interval = int(len(time)/(n+1))
    for index,t in tqdm(enumerate(time)):
        if (index < 100) or (index%interval != 0):
            continue
        if index == 0:
            continue

        frequency2, power2 = LombScargle(time[:index], s2[:index], 1000).autopower(minimum_frequency=25, maximum_frequency=35,
                                                                     samples_per_peak=10)
        frequency3, power3 = LombScargle(time[:index], s3[:index], 1000).autopower(minimum_frequency=25, maximum_frequency=35,
                                                                   samples_per_peak=10)
        frequency4, power4 = LombScargle(time[:index], s4[:index], 1000).autopower(minimum_frequency=25, maximum_frequency=35,
                                                                   samples_per_peak=10)
        peak2 = max(power2)
        peak3 = max(power3)
        peak4 = max(power4)

        noise2 = power2[((frequency2 < ephemeris + 0.6) & (frequency2 > ephemeris + 0.1)) | (
                (frequency2 > ephemeris - 0.6) & (frequency2 < ephemeris - 0.1))]
        noise3 = power3[((frequency3 < ephemeris + 0.6) & (frequency3 > ephemeris + 0.1)) | (
                (frequency3 > ephemeris - 0.6) & (frequency3 < ephemeris - 0.1))]
        noise4 = power4[((frequency4 < ephemeris + 0.6) & (frequency4 > ephemeris + 0.1)) | (
                (frequency4 > ephemeris - 0.6) & (frequency4 < ephemeris - 0.1))]

        noise = noise2 + noise3 + noise4
        sig = (peak2 + peak3 + peak4)/np.std(noise)
        significance = np.append(significance,sig)
        times = np.append(times,t-tmin)

    if plot:
        from scipy.optimize import curve_fit
        def linear_fit(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(linear_fit, times, significance)
        xplot = np.linspace(-5, max(times), times.size)

        plt.plot(times,significance,'ko')
        plt.plot(xplot, linear_fit(xplot, *popt), 'g')
        plt.axhline(5, color='r', ls='--', label=r'5$\sigma$')

        plt.legend()
        plt.ylabel('Significance [σ]')
        plt.xlabel('Elapsed Time [s]')
        plt.title(f'{name} 3-Tel Cumulative Significance')
        plt.grid(which='major')
        plt.grid(which='minor')
        plt.savefig('csig.png',format='png')

    return times, significance

def tukey_window(n,alpha=1/50):
    return sp.signal.tukey(n,alpha=alpha)

def whiten(signal,ephemeris,sample):
    from scipy.ndimage import gaussian_filter1d
    fftfreqs = np.fft.rfftfreq(len(signal), d=1/sample)
    win = tukey_window(len(signal2)) * signal2
    ft_windowed = np.fft.rfft(signal * win)  # window our strain data
    ps = np.abs(ft_windowed) ** 2  # take the ps of windowed data
    buff = 0.1 #buffer around ephemeris that won't be smoothed
    temp = gaussian_filter1d(ps[:len(ps) // int(len(ps) / np.where(fftfreqs >= ephemeris - buff)[0][0])], 10,
                             mode='nearest') #left of ephemeris
    fill = np.append(temp, ps[len(ps) // int(len(ps) / np.where(fftfreqs >= ephemeris - buff)[0][0]):len(ps) // int(
        len(ps) / np.where(fftfreqs >= ephemeris + buff)[0][0])]) #buffer region
    temp2 = gaussian_filter1d(ps[len(ps) // int(len(ps) / np.where(fftfreqs >= ephemeris + buff)[0][0]):], 10,
                              mode='nearest') #right of ephemeris
    smooth = np.append(fill, temp2)
    smooth_fft = np.fft.rfft(smooth)
    norm = 1./np.sqrt(1./(1/sample*2))
    white = smooth_fft / np.sqrt(psd) * norm
    return np.fft.irfft(white)

def phase_fold(signal,time,ephemeris,nbins,name,plot=True):
    """
    :param signal: Pulsar signal array (can be pre-whitened)
    :param time: Time array
    :param ephemeris: Ephemeris to phase fold over (recommended to use get_ephemeris)
    :param nbins: Number of bins in phasogram
    :param plot: (boolean) If true, plot results
    :param name: Pulsar name
    :return: bins, signal, error
    """

    def phase_bin(signal, phases, bins):
        counts_bin = np.zeros(len(bins))
        std_bin = np.zeros(len(bins))
        sum_bin = np.zeros(len(bins))
        for i in range(1, len(bins)):
            idx = np.where((phases >= bins[i - 1]) & (phases < bins[i]))[0]
            std_bin[i - 1] = np.std(signal[idx])
            sum_bin[i - 1] = np.mean(signal[idx])
            counts_bin[i - 1] = len(signal[idx])
        return counts_bin, std_bin, sum_bin


    bins = np.linspace(0, 1, nbins)
    period = 1/ephemeris
    phase = np.abs((time) / period) - np.abs(np.floor((time) / period))
    counts, std, sig = phase_bin(signal, phase, bins)
    err = std / np.sqrt(counts)

    if plot:
        plt.errorbar(bins, sig, yerr=err, fmt='.', color='k')
        plt.title(f'{name} Phasogram')
        plt.ylabel('Voltage [V]')
        plt.xlabel('Phase')
        plt.grid()
        plt.savefig('phasogram.png',format='png')

    return bins, sig, err


print('===================================================')
print('===================================================')
print('========== OOPS-E Pulsar Analysis Script ==========')
print('============ (C) 2023 Samantha Wong ===============')
print('===================================================')
print('===================================================')

#Read in configuration parameters
config = sys.argv[1]
cumulative = sys.argv[2]
phasogram = sys.argv[3]

with open(config, 'r') as file:
    params = yaml.load(file,Loader=yaml.FullLoader)

name = str(params['name']) #pulsar name
date = params['date'] #run date
run_time = params['time'] #run start time
dur = params['duration'] #run duration - unused but could be used to improve ephemeris
sample = params['sample'] #sample rate (usually 2400 unless specified otherwise)
ntel = params['ntel'] #number of telescopes to use (usually 3)

#log file
logfile = name + '_' + str(date) + '_' + str(sample) + '_log.txt'
log = open(logfile,"w")

log.write('OOPS-E Pulsar Analysis Script\n')
log.write('Samantha Wong 2023\n')
log.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
log.write('\n')
log.write(f'RUN DATE: {date}\n')
log.write(f'RUN TIME: {run_time} UTC\n')
log.write(f'SAMPLE RATE: {sample} samples/s\n')
log.write(f'# TELS: {ntel}\n')
log.write(f'------------------------------\n')
log.write('\n')

#Read files - extract signals and make time arrays
times, signals = read_files(dir,date,sample,ntel)
times = np.array(times)

#Apply time cuts
start = params['start']
stop = params['stop']

if ntel == 3:
    signal2 = signals[0]
    signal3 = signals[1]
    signal4 = signals[2]

    signal2 = -signal2[np.where((times > start) & (times < stop))]
    mean2 = np.mean(signal2)
    signal2 = signal2-mean2

    signal3 = -signal3[np.where((times > start) & (times < stop))]
    mean3 = np.mean(signal3)
    signal3 = signal3-mean3

    signal4 = -signal4[np.where((times > start) & (times < stop))]
    mean4 = np.mean(signal4)
    signal4 = signal4-mean4

    times = times[np.where((times > start) & (times < stop))]

if ntel == 4:
    signal1 = signals[:0]
    signal2 = signals[:1]
    signal3 = signals[:2]
    signal4 = signals[:3]

    time1 = times[:0]
    time = times[:1]

    t1_start = params['start']['t1']
    t1_stop = params['stop']['t1']
    signal1 = -signal1[np.where((time1 > t1_start) & (time1 < t1_stop))]
    mean1 = np.mean(signal1)
    signal1 = signal1 - mean1

#significance calculations
print('Calculating Significance')

ephemeris = get_ephemeris(str(date),str(run_time))
print(f'Ephemeris on {date} at {run_time} is {ephemeris}')

print('Lomb-Scargling T2')
sig2,peak2,peak_freq2,noise2 = calc_sig(signal2,times,ephemeris)
print('Lomb-Scargling T3')
sig3,peak3,peak_freq3,noise3 = calc_sig(signal3,times,ephemeris)
print('Lomb-Scargling T4')
sig4,peak4,peak_freq4,noise4 = calc_sig(signal4,times,ephemeris)

print(f'T2 Significance: {sig2} sigma')
print(f'T3 Significance: {sig3} sigma')
print(f'T4 Significance: {sig4} sigma')

log.write('=========== Significances ===========\n')
log.write(f'T2 Significance: {sig2} sigma\n')
log.write(f'T3 Significance: {sig3} sigma\n')
log.write(f'T4 Significance: {sig4} sigma\n')

tot_noise = noise2 + noise3 + noise4
std_tot = np.std(tot_noise)
sig_tot = (peak2 + peak3 + peak4)/std_tot

print(f'3-Tel Stacked Significance: {sig_tot} sigma')
log.write(f'3-Tel Stacked Significance: {sig_tot} sigma\n')
log.write('\n')

#write offsets to log for comparison b/w crab runs
log.write('=========== Offsets ===========\n')
log.write(f'T2 peak @ {peak_freq2} Hz\n')
log.write(f'T3 peak @ {peak_freq3} Hz\n')
log.write(f'T4 peak @ {peak_freq4} Hz\n')

if cumulative: #calculates + plots cumulative significance
    csig_times, csig_sigs = cumulative_sig(signal2,signal3,signal4,times,10,start,name,plot=True)
    np.savetxt(name+'_csig_data.txt',np.c_[csig_times,csig_sigs])

if phasogram: #plots phasogram
    nbins = 100
    bins,folded_signal,fs_err = phase_fold(signal2,times,ephemeris,nbins,name,plot=True)
    np.savetxt(name+'_phase_fold.txt',np.c_[bins,folded_signal,fs_err])

log.close()
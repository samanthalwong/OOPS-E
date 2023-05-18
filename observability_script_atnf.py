import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
import scipy as sp
from scipy import stats
from scipy.interpolate import CubicSpline as spl
from scipy.signal.windows import tukey
import sys
import math

def calc_sig(time,signal,ephemeris,tel,plot=True):
    harmonics = 3
    sigmas = np.array(())
    for h in range(harmonics):
        p = ephemeris * (h+1)
        
        frequency, power = LombScargle(time, signal, 1000).autopower(minimum_frequency= p-1,
                                                                             maximum_frequency= p+1,
                                                                             samples_per_peak=10)

        peak = max(power[np.where((frequency >= p - 0.001) & (frequency <= p + 0.001))])
        peak_freq = frequency[np.where(power ==  peak)]

        noise = power[((frequency < p + 0.6) & (frequency > p + 0.1)) | ((frequency > p - 0.6) & (frequency < p - 0.1))]

        sigma = peak / np.std(noise)

        sigmas = np.append(sigmas,sigma)

    tot_sig = 0
    for i in sigmas:
        tot_sig = tot_sig + i**2

    tot_sig = np.sqrt(tot_sig)

    return tot_sig,sigmas

# set up ECM-like sampling for 10h
samp = 1200.
arr_len = 36000. * samp

time = 1./samp * np.arange(0,arr_len)

#read in background pixel
pix = np.genfromtxt('/raid/romulus/swong/ecm/d20221125/20221125-Crab_1-T3.csv',usecols=(2),delimiter=',',unpack=True)

#normalize
pix = pix[::2] - np.mean(pix[::2])
time_pix = 1./samp * np.arange(0,len(pix))

#fit spline to background pixel PS
ps = np.abs(np.fft.rfft(pix))**2
fftfreqs = np.fft.rfftfreq(len(pix),1/1200)

splinefit = spl(fftfreqs,ps)
fftfreqs_sim = np.fft.rfftfreq(int(arr_len),1./1200.)

model_ps = splinefit(fftfreqs_sim)

ftnoise2 = splinefit(fftfreqs_sim)
ftnoise3 = splinefit(fftfreqs_sim)
ftnoise4 = splinefit(fftfreqs_sim)

win = tukey(len(ftnoise2),alpha=0.1)

#create noise
noise2 = np.fft.irfft(ftnoise2*win)
noise3 = np.fft.irfft(ftnoise2*win)
noise4 = np.fft.irfft(ftnoise2*win)

noise2 = noise2 + np.random.normal(loc=0, scale=0.0001, size=len(time))
noise3 = noise3 + np.random.normal(loc=0, scale=0.0001, size=len(time))
noise4 = noise4 + np.random.normal(loc=0, scale=0.0001, size=len(time))

#create template pulsars
periods = np.logspace(-2.7,1.06,10) #general range of ATNF pulsars
mag = np.linspace(14.8,40,10) #general range according to formula + ECM calibration
width = np.logspace(-1.3,2.87,10) #from ATNF; ms

#create gaussian pulse profiles
def gauss(x,a,b,c,d):
    return a*np.exp((-(x-b)**2)/(2*c**2)) + d

def exponential(t,amp,tref,trise,tdecay,c):
    out = np.zeros(len(t))
    mask = t < tref
    out[mask] = amp * np.exp((np.abs(t[mask]-tref)/trise))+c
    out[~mask] = amp * np.exp((np.abs(t[~mask]-tref)/tdecay))+c
    return out

#loop through file
file = open('atnf_observability.txt','r')
for i in file.readlines():
    name = i.split()[0]
    p = float(i.split()[1])
    w = float(i.split()[2])
    m = float(i.split()[3])
    
    if 1/p < 1:
        continue
    
    if math.isnan(w):
        w = 31 #mean width
    if math.isinf(w):
        w = 31

    #define phase array
    phases = np.abs((time)/p) - np.abs(np.floor((time)/p))

    def smeared_gaussian(x,a,b,c,d,decay=-(1/349)/p):
        f = gauss(x,a,b,c,d)
        g = exponential(x,a,b,-1e-16,decay,d)
        F = np.fft.rfft(f)
        G = np.fft.rfft(g)
        return np.fft.irfft(F*G)

    a = 10**((-0.40355447*m)+2.27458167)
    
    sigma = ((w/1000)/p)/2.355 
    fake_signal = smeared_gaussian(phases,a,0.5,w,0)
    summed_signal2 = fake_signal + noise2
    summed_signal3 = fake_signal + noise3
    summed_signal4 = fake_signal + noise4

    #digitization
    step = 1.22e-5 #G100
    edges = np.arange(-1,1,step)

    kodkod2 = np.zeros(summed_signal2.size)
    kodkod3 = np.zeros(summed_signal3.size)
    kodkod4 = np.zeros(summed_signal4.size)

    dig_bins2 = np.digitize(summed_signal2,edges,right=True)
    dig_bins3 = np.digitize(summed_signal3,edges,right=True)
    dig_bins4 = np.digitize(summed_signal4,edges,right=True)

    for i,b in enumerate(dig_bins2):
        kodkod2[i] = edges[b-1]
        
    for i,b in enumerate(dig_bins3):
        kodkod3[i] = edges[b-1]
        
    for i,b in enumerate(dig_bins4):
        kodkod4[i] = edges[b-1]

    #subtract background PS
    ps_kodkod2 = np.abs(np.fft.rfft(kodkod2))**2
    ps_kodkod3 = np.abs(np.fft.rfft(kodkod3))**2
    ps_kodkod4 = np.abs(np.fft.rfft(kodkod4))**2

    diff2 = ps_kodkod2-model_ps
    diff3 = ps_kodkod3-model_ps
    diff4 = ps_kodkod4-model_ps

    win = tukey(len(ps_kodkod2),alpha=0.01)

    clean_sig2 = np.fft.irfft(win*diff2)
    clean_sig3 = np.fft.irfft(win*diff3)
    clean_sig4 = np.fft.irfft(win*diff4)
    
    #signal SNR
    tot2c,sigs2c = calc_sig(time,clean_sig2,1/p,2)
    tot3c,sigs3c = calc_sig(time,clean_sig3,1/p,3)
    tot4c,sigs4c = calc_sig(time,clean_sig4,1/p,4)

    combined = np.sqrt(tot2c**2 + tot3c**2 + tot4c**2)

    #write output
    with open('observability_results_atnf.txt', 'a') as f:
        f.write(f'{name} {p} {m} {w} {combined}\n')            
    f.close()




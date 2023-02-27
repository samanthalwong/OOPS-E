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
import os
import oopse

print('===================================================')
print('===================================================')
print('========== OOPS-E Pulsar Analysis Script ==========')
print('============= Samantha Wong (2023) ================')
print('===================================================')
print('===================================================')

#Read in configuration parameters
config = sys.argv[1]
cumulative = bool(int(sys.argv[2]))
phasogram = bool(int(sys.argv[3]))
data_dir = sys.argv[4]
out_dir = sys.argv[5]

with open(config, 'r') as file:
    params = yaml.load(file,Loader=yaml.FullLoader)

name = str(params['name']) #pulsar name
date = params['date'] #run date
run_time = params['time'] #run start time
dur = params['duration'] #run duration - unused but could be used to improve ephemeris
sample = params['sample'] #sample rate (usually 2400 unless specified otherwise)
ntel = params['ntel'] #number of telescopes to use (usually 3)
nruns = int(params['nruns']) #number of runs taken on [date]
runnum = params['runnum'] #run number if > 1 runs taken on [date]
nbins = int(params['nbins']) #number of bins for phase folding

#create out directory if it doesn't exist
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

#log file
logfile = out_dir + '/' + name + '_' + str(date) + '_' + str(sample) + '_log.txt'
log = open(logfile,"w")

log.write('OOPS-E Pulsar Analysis Script\n')
log.write('Developed by Samantha Wong 2023\n')
log.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
log.write('\n')
log.write(f'RUN DATE: {date}\n')
log.write(f'RUN TIME: {run_time} UTC\n')
log.write(f'SAMPLE RATE: {sample} samples/s\n')
log.write(f'# TELS: {ntel}\n')
log.write(f'------------------------------\n')
log.write('\n')

#Read files - extract signals and make time arrays
times, signals = oopse.read_files(data_dir,date,float(sample),nruns,runnum,ntel)
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

#plot data
oopse.plot_raw(signal2,times,out_dir,date,runnum,2)
oopse.plot_raw(signal3,times,out_dir,date,runnum,3)
oopse.plot_raw(signal4,times,out_dir,date,runnum,4)

#significance calculations
print('Calculating Significance')

ephemeris = oopse.get_ephemeris(str(date),str(run_time))
print(f'Ephemeris on {date} at {run_time} is {ephemeris}')

print('Lomb-Scargling T2')
sig2,peak2,peak_freq2,noise2 = oopse.calc_sig(signal2,times,ephemeris,2,out_dir,runnum,plot=True)
print('Lomb-Scargling T3')
sig3,peak3,peak_freq3,noise3 = oopse.calc_sig(signal3,times,ephemeris,3,out_dir,runnum,plot=True)
print('Lomb-Scargling T4')
sig4,peak4,peak_freq4,noise4 = oopse.calc_sig(signal4,times,ephemeris,4,out_dir,runnum,plot=True)

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

#write noise and peak values to .txt file for combined run analysis
peaks = np.array([peak2,peak3,peak4])
peaks = str(peaks)
np.savetxt(out_dir + '/' + str(date) +'_peaknoise.txt',np.c_[noise2,noise3,noise4],header=peaks)

print(f'3-Tel Stacked Significance: {sig_tot} sigma')
log.write(f'3-Tel Stacked Significance: {sig_tot} sigma\n')
log.write('\n')

#write offsets to log for comparison b/w crab runs
log.write('=========== Offsets ===========\n')
log.write(f'T2 peak @ {peak_freq2} Hz\n')
log.write(f'T3 peak @ {peak_freq3} Hz\n')
log.write(f'T4 peak @ {peak_freq4} Hz\n')

if cumulative: #calculates + plots cumulative significance
    csig_times, csig_sigs = oopse.cumulative_sig(signal2,signal3,signal4,times,10,start,name,out_dir,ephemeris,runnum,plot=True)
    np.savetxt(out_dir + '/' + name+'_csig_data.txt',np.c_[csig_times,csig_sigs])

if phasogram: #plots phasogram
    bins2,folded_signal2,fs_err2 = oopse.phase_fold(signal2,times,peak_freq2,nbins,name,out_dir,'T2',runnum,plot=True)
    bins3,folded_signal3,fs_err3 = oopse.phase_fold(signal3,times,peak_freq3,nbins,name,out_dir,'T3',runnum,plot=True)
    bins4,folded_signal4,fs_err4 = oopse.phase_fold(signal4,times,peak_freq4,nbins,name,out_dir,'T4',runnum,plot=True)

    np.savetxt(out_dir + '/phase_fold.txt',np.c_[bins2,folded_signal2,fs_err2,folded_signal3,fs_err3,folded_signal4,fs_err4])

log.close()
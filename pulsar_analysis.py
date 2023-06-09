import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from astropy.timeseries import LombScargle
import yaml
import sys
from datetime import datetime
import os
import oopse

print('===================================================')
print('===================================================')
print('========== OOPS-E Pulsar Analysis Script ==========')
print('============= Samantha Wong (2023) ================')
print('===================================================')
print('===================================================')

# Read in configuration parameters
config = sys.argv[1]
cumulative = bool(int(sys.argv[2]))
phasogram = bool(int(sys.argv[3]))
data_dir = sys.argv[4]
out_dir = sys.argv[5]

with open(config, 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

name = str(params['name'])  # pulsar name
ephemeris = float(params['ephemeris']) #pulsar ephemeris - this can eventually be migrated to a .par file
date = params['date']  # run date
run_time = params['time']  # run start time
dur = params['duration']  # run duration - unused but could be used to improve ephemeris
sample = params['sample']  # sample rate (usually 2400 unless specified otherwise)
ntel = params['ntel']  # number of telescopes to use (usually 3)
nbins = int(params['nbins'])  # number of bins for phase folding
file1 = str(params['file1']) #name of t1 file
file2 = str(params['file2']) #name of t2 file
file3 = str(params['file3']) #name of t3 file
file4 = str(params['file4']) #name of t4 file
filename = str(params['fname']) #name for output files

# create out directory if it doesn't exist
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

logfile = out_dir + '/' + filename + '_log.txt'
log = open(logfile, "w")

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

# Read files - extract signals and make time arrays
time2,sig2 = oopse.read_file(file2)
time3,sig3 = oopse.read_file(file3)
time4,sig4 = oopse.read_file(file4)

#calculate p values of peaks around ephemeris frequency
p2,min2,eph2 = oopse.calc_sig(time2,sig2,ephemeris,2)
p3,min3,eph3 = oopse.calc_sig(time3,sig3,ephemeris,3)
p4,min4,eph4 = oopse.calc_sig(time4,sig4,ephemeris,4)

#use T1 if unsaturated
if ntel == 4:
    time1,sig1 = oopse.read_file(file1)
    p1,min1,eph1 = oopse.calc_sig(time1,sig1,ephemeris,1)
    pvals = [p1,p2,p3,p4]
else:
    pvals = [p2,p3,p4]

log.write(f'P Values: {pvals}\n')

#calculate significance from p values
sigs = []
for i in len(pvals):
    sigs[i] = oopse.calc_sigma(pvals[i])
log.write(f'Individual Telescope Significances: f{sigs}\n')

significance = oopse.combine_sig(pvals)
log.write(f'Total Significance: {significance}\n')

#calculate cumulative significance
if cumulative:
    time_new = np.array(())
    cum_sigs = np.array(())
    for index, t in enumerate(time2):
        if ((index < 100) or (index % 500000 != 0)):
            continue
        if (index == 0):
            continue

        p2c, min2c, eph2c = oopse.calc_sig(time2[:index], sig2[:index], ephemeris, 2)
        p3c, min3c, eph3c = oopse.calc_sig(time3[:index], sig3[:index], ephemeris, 3)
        p4c, min4c, eph4c = oopse.calc_sig(time4[:index], sig4[:index], ephemeris, 4)

        if ntel == 4:
            p1c, min1c, eph1c = oopse.calc_sig(time1[:index], sig1[:index], ephemeris, 1)
            pvalsc = [p1c,p2c,p3c,p4c]
        else:
            pvalsc = [p2c,p3c,p4c]

        cum_sig = oopse.combine_sig(pvalsc)

        time_new = np.append(time_new,t)
        cum_sigs = np.append(cum_sigs, cum_sig)

    np.savetxt(f'{filename}_cumulative_sig.txt',np.c_[time_new,cum_sigs])

if phasogram:
    bins2, sig2, err2 = oopse.phase_fold(time2, sig2, 1 / eph2, nbins)
    bins3, sig3, err3 = oopse.phase_fold(time3, sig3, 1 / eph3, nbins)
    bins4, sig4, err4 = oopse.phase_fold(time4, sig4, 1 / eph4, nbins)

    if ntel == 4:
        bins1, sig1, err1 = oopse.phase_fold(time1, sig1, 1 /eph1, nbins)
        np.savetxt(f'{filename}_phase_info.txt', np.c_[bins2,sig1,err1, sig2, err2, sig3, err3, sig4, err4])

    np.savetxt(f'{filename}_phase_info.txt',np.c_[bins2,sig2,err2,sig3,err3,sig4,err4])


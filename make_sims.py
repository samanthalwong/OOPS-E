import numpy as np
from astropy.io import fits
import sys

import oopse

name = str(sys.argv[1])
period = float(sys.argv[2])
width = float(sys.argv[3])
mag = float(sys.argv[4])
samp = int(sys.argv[5])
outfile = str(sys.argv[6])

#read in background data
hdul1 = fits.open('t1.fits')
hdul2 = fits.open('t2.fits')
hdul3 = fits.open('t3.fits')
hdul4 = fits.open('t4.fits')

sig1 = hdul1[1].data['signal']
sig2 = hdul2[1].data['signal']
sig3 = hdul3[1].data['signal']
sig4 = hdul4[1].data['signal']

sig2 = sig2[::2]
sig3 = sig3[::2]
sig4 = sig4[::2]

sig1 = sig1[:len(sig3)]
sig2 = sig2[:len(sig3)]
sig4 = sig4[:len(sig3)]

#normalize
sig1 = (sig1 - np.mean(sig1))
sig2 = (sig2 - np.mean(sig2))
sig3 = (sig3 - np.mean(sig3))
sig4 = (sig4 - np.mean(sig4))

time = 1./samp * np.arange(0,len(sig3))

on1 = oopse.undigitize(sig1)
on2 = oopse.undigitize(sig2)
on3 = oopse.undigitize(sig3)
on4 = oopse.undigitize(sig4)

#create template pulsaar - this stage is slow due to convolution

if period > 5e-2: #add random phase for long period pulsars
    m1 = np.random.uniform(0,0.5)
    m2 = np.random.uniform(0, 0.5)
    m3 = np.random.uniform(0, 0.5)
    m4 = np.random.uniform(0, 0.5)

else:
    m1 = m2 = m3 = m4 = 0

phases = np.abs((time)/p) - np.abs(np.floor((time)/p))

a = 10**((-0.40355447*mag)+2.27458167)
sigma = width/2.355

fake_signal1 = oopse.smeared_gaussian(phases,a,m1,sigma,0)
fake_signal2 = oopse.smeared_gaussian(phases,a,m2,sigma,0)
fake_signal3 = oopse.smeared_gaussian(phases,a,m3,sigma,0)
fake_signal4 = oopse.smeared_gaussian(phases,a,m4,sigma,0)

summed_signal1 = fake_signal1 + on1
summed_signal2 = fake_signal2 + on2
summed_signal3 = fake_signal3 + on3
summed_signal4 = fake_signal4 + on4

#redigitize
kodkod1 = oopse.digitize(summed_signal1)
kodkod2 = oopse.digitize(summed_signal2)
kodkod3 = oopse.digitize(summed_signal3)
kodkod4 = oopse.digitize(summed_signal4)

ephemeris = 1/period

p1,min1,eph1 = oopse.calc_sig(time,kodkod1,ephemeris,1)
p2,min2,eph2 = oopse.calc_sig(time,kodkod2,ephemeris,2)
p3,min3,eph3 = oopse.calc_sig(time,kodkod3,ephemeris,3)
p4,min4,eph4 = oopse.calc_sig(time,kodkod4,ephemeris,4)

combined = oopse.combine_sig([p1,p2,p3,p4])

with open(outfile, 'a') as f:
    f.write(f'{name} {period} {mag} {width} {combined}\n')
f.close()
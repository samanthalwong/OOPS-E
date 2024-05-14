# OOPS-E

OOPS-E is an anlayis and simulations package designed for the VERITAS enhanced current monitor (ECM) optical data. 

## Installation
Coming soon! Just git clone and run the `pulsar_analysis.py` script to get started.

## Usage

### Run parameter files

OOPS-E analysis requires a .yaml parameter file for each run with the following lines:

* `name`: Pulsar name
* `ephemeris`: Up-to-date pulsar ephemeris (eventually .par files will be integrated to help with this)
*  `date`: Run date
*  `time`: Run start time (UTC)
*  `duration`: Run duration
*  `sample`: Sampling rate (usually 2400 HzNo, sometimes 1200 Hz for T1)
*  `ntel`: Number of telescopes included in the analysis (see note 1 below RE: T1)
*  `nbins`: Number of bins for phase folding
*  `file1`...`file4`: Name of T1 file (need one for each telescope)
*  `filename`: Output file name

Your data files should already be cropped with time cuts applied to pixel suppressions and any instrumental or non-astrophysical effects (meteors are fine). 

You should ensure that no telescopes are saturated prior to analysis. If only T1 is saturated, enter `ntel = 3` in your run parameter file.

### Simulations

Simulations require a blank field file output to FITS format.

### Notes
Note 1: Runs with T1 sampling at a different rate than T2/T3/T4 are currently unsupported

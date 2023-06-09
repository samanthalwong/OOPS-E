import numpy as np
from matplotlib import pyplot as plt
from astropy.timeseries import LombScargle
import scipy.stats as st


def plot_raw(signal, time, dir, date, runnum, tel):
    """
    :param signal: raw signal data
    :param time: time array
    :param dir: output directory
    :param date: run date
    :param runnum: run number if # runs taken on [date] > 1
    :param tel: Telescope #
    """
    plt.clf()
    plt.plot(time, signal, 'k.')
    plt.xlabel('Time')
    plt.ylabel('Voltage [V]')
    plt.title(f'T{tel} Data')
    if runnum >= 1:
        plt.savefig(dir + '/' + str(date) + '_' + str(runnum) + '_T' + str(tel) + '_rawdata.png', format='png')
    else:
        plt.savefig(dir + '/' + str(date) + str(tel) + '_rawdata.png', format='png')
    return


def read_file(filename, sample=2400):
    """
    :param filename: .csv file to read
    :param sample: sample rate (default = 2400 Hz)
    :return: time,signals arrays of time and signal values for each telescope (single time array for T2/T3/T4)
    """

    signal = np.genfromtxt(filename, delimiter=',', unpack=True, usecols=(1), skip_header=3)

    time = 1. / sample * np.arange(0, len(signal))

    return time, signal


def undigitize(array):
    return array + np.random.uniform((-1.22e-5) / 2, (1.22e-5) / 2, len(array))


def digitize(array, step=1.22e-5):
    edges = np.arange(-1, 1, step)
    arr = np.zeros(len(array))
    dig_bins = np.digitize(array, edges, right=True)
    for i, b in enumerate(dig_bins):
        arr[i] = edges[b - 1]
    return arr


def gauss(x, a, b, c, d):
    return a * np.exp((-(x - b) ** 2) / (2 * c ** 2)) + d


def exponential(t, amp, tref, trise, tdecay, c):
    out = np.zeros(len(t))
    mask = t < tref
    out[mask] = amp * np.exp((np.abs(t[mask] - tref) / trise)) + c
    out[~mask] = amp * np.exp((np.abs(t[~mask] - tref) / tdecay)) + c
    return out


def smeared_gaussian(x, a, b, c, d, decay=-(1 / 349) / p):
    f = gauss(x, a, b, c, d)
    f = f / np.sum(f)
    g = exponential(x, a, b, -1e-16, decay, d)
    g = g / np.sum(g)
    F = np.fft.rfft(f)
    G = np.fft.rfft(g)
    conv = np.fft.irfft(F * G)
    conv = conv / np.max(conv)
    conv = conv * a
    return conv


def inject_signal(noise_arr, p, mag, width, time, noise):
    a = 10 ** ((-0.40355447 * mag) + 2.27458167)
    sigma = (width / 2.355)
    m = np.random.uniform(0.2, 0.9)
    phases = np.abs((time) / p) - np.abs(np.floor((time) / p))
    fake_signal = smeared_gaussian(phases, a, m, sigma, 0)
    summed_signal = fake_signal + noise
    digitized_signal = digitize(summed_signal)
    return (digitized_signal)


from tqdm.auto import tqdm


def combine_sig(P):
    pval = 1
    for p in P:
        pval = pval * np.exp(-p)
    sig = st.norm.ppf(1 - pval, loc=0, scale=1)
    return sig


def calc_sig(time, signal, ephemeris, tel, err=1e-3, plot=True):
    # err is the error on the ephemeris, which is used to calcuate the on region
    harmonics = 1
    ps = np.array(())
    for h in tqdm(range(harmonics)):
        p = ephemeris * (h + 1)
        print(f'Starting harmonic {h}:')
        # get local average power level & use to whiten

        frequency, power = LombScargle(time, signal).autopower(minimum_frequency=p - 1,
                                                               maximum_frequency=p + 1,
                                                               samples_per_peak=10,
                                                               nyquist_factor=2)
        on = np.where((frequency >= p - err) & (frequency <= p + err))[0]

        off = \
            np.where(((frequency < p + 1.1) & (frequency > p + 0.1)) | ((frequency > p - 1.1) & (frequency < p - 0.1)))[
                0]
        norm = np.median(power[off]) / np.log(2)

        ln_prob = power[on] / norm

        P = np.max(ln_prob)

        peak_freq = frequency[np.where(power[on] == np.max(power[on]))]

        def sig(x):
            out = (1.0 / np.sqrt(2.0 * np.pi)) * (np.exp(-(x ** 2.) / 2.))
            return out

        def f(sigval, p):
            integ = integrate.quad(sig, sigval, np.inf)
            return np.abs(integ[0] - p)

        def trials(pcheck, n):
            return (1 - ((1 - pcheck) ** (1. / n)))

        pcheck = 5.733e-7 / 2  # 5 sigma the division by two is for the two sides

        result = optimize.minimize_scalar(f, args=trials(pcheck, float(len(on))))

        min_sigma = result.x

        print(f'P: {P}')
        print(f'Threshold significance: {min_sigma}')

        if plot:
            plt.clf()
            plt.plot(frequency, power)
            plt.axvline(p, color='k', alpha=0.2, linestyle='--', label='Ephemeris')
            plt.axvspan(frequency[off][0], frequency[off][int(len(off) / 2) - 1], alpha=0.5, color='r',
                        label='Noise Region')
            plt.axvspan(frequency[off][int(len(off) / 2)], frequency[off][-1], alpha=0.5, color='r')

            plt.title(f'T{tel} Lomb-Scargle Periodogram')
            plt.savefig()

    return P, min_sigma, peak_freq


def calc_sigma(pval):
    sig = st.norm.ppf(1 - pval, loc=0, scale=1)
    return sig


def phase_bin(signal,phases,bins):
    counts_bin = np.zeros(len(bins))
    std_bin = np.zeros(len(bins))
    sum_bin = np.zeros(len(bins))
    for i in range(1, len(bins)):
        idx = np.where((phases >= bins[i - 1]) & (phases < bins[i]))[0]
        std_bin[i - 1] = np.std(signal[idx])
        sum_bin[i - 1] = np.mean(signal[idx])
        counts_bin[i - 1] = len(signal[idx])
    return counts_bin, std_bin, sum_bin

def phase_fold(time,signal,period,nbins):
    bins = np.linspace(0,1,nbins)
    phase = np.abs((time) / period) - np.abs(np.floor((time) / period))
    counts, std, sig = phase_bin(signal, phase, bins)
    err = std/np.sqrt(counts)
    return bins, sig, err

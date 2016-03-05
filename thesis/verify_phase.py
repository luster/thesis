"""this script is meant to verify that the noise phase will give rise to a
fairly clean sounding recording. if it sounds like shit, we're going to convert
the NN model to a pseudo-complex one that can run on the GPU. if the recording is
fairly clean, we'll probably stick with the current magnitude/power spectrogram
model.
"""
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import os
from util import *
from config import *
import scikits.audiolab
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def normalize(this, against):
    return this / max(abs(this)) * max(abs(against))

# path to signal
fname = 'my_perfect_clean_guitar_sound.wav'
sig_path = os.path.join(data_folder, fname)
noise_fname = 'golf_club_bar_lunch_time.wav'
noise_path = os.path.join(data_folder, noise_fname)

# load signal
sig = load_soundfile(sig_path, 0, 10)
noise = load_soundfile(noise_path, 0, 10)

baseline_mse = None
def save_sound(loosely_snr):
    global baseline_mse
    # corrupt a copy of signal
    # loosely_snr = 0.1
    noisy = sig + loosely_snr * noise

    # get magnitude of clean, phase of noisy
    mag_clean, phase_clean = standard_specgram(sig)
    _, phase_noisy = standard_specgram(noisy)

    # istft using clean mag, noisy phase
    out_questionable = calculate_time_signal(mag_clean, phase_noisy)

    # istft using clean mag, clean phase for comparison
    out_shouldnt_be_questionable = calculate_time_signal(mag_clean, phase_clean)

    # normalize to original signal
    out_shouldnt_be_questionable = normalize(out_shouldnt_be_questionable, sig)
    baseline_mse = mean_squared_error(sig, out_shouldnt_be_questionable)
    print 'baseline mse: ', baseline_mse
    print 'mse: ', mean_squared_error(sig, out_questionable)
    print 'snr: ', 1./loosely_snr**2
    print 'snr: ', 10*np.log10(1./loosely_snr**2), ' dB'

    # save files to wav
    scikits.audiolab.wavwrite(noisy, 'phase_test/out_%s.wav' % 'noisy', fs=srate, enc='pcm16')
    scikits.audiolab.wavwrite(out_questionable, 'phase_test/out_%s.wav' % 'questionable', fs=srate, enc='pcm16')
    scikits.audiolab.wavwrite(out_shouldnt_be_questionable, 'phase_test/out_%s.wav' % 'noquestions', fs=srate, enc='pcm16')

# plt.figure()
# plt.plot(sig)
# plt.plot(out_shouldnt_be_questionable)
# plt.legend('sig', 'test')
# plt.show()

def generate_plot(snrs, axis_func=plt.plot):
    """loop over SNRs, compute MSE using noisy phase

    Arguments:
        snrs {list} -- arraylike of SNRs
    """
    mses = []
    for snr in snrs:
        noisy = sig + snr * noise
        mag_clean, phase_clean = standard_specgram(sig)
        _, phase_noisy = standard_specgram(noisy)
        out_questionable = calculate_time_signal(mag_clean, phase_noisy)
        # out_questionable = normalize(out_questionable, out_shouldnt_be_questionable)
        # mse = mean_squared_error(out_shouldnt_be_questionable, out_questionable)
        out_questionable = normalize(out_questionable, sig)
        mse = mean_squared_error(sig, out_questionable)
        print 'snr: ', snr, ' mse: ', mse
        mses.append(mse)
    plt.figure()
    xaxis = -20. * np.log10(snrs)
    axis_func(xaxis, mses)
    axis_func(xaxis, baseline_mse*np.ones(snrs.shape))
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE')
    plt.title('MSE of Noisy Phase Signal w.r.t. Clean Phase Signal Reconstruction')
    plt.legend(['MSE', 'Baseline'])
    plt.show()

if __name__ == '__main__':
    import sys
    save_sound(float(sys.argv[1]))
    # generate_plot(np.logspace(0,4,25)*0.01, plt.semilogy)



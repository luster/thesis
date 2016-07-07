from __future__ import division

import os
from glob import glob
from random import randint
from random import uniform
import numpy as np
from numpy import complex64
from numpy import float32
import theano
import theano.tensor as T
import lasagne
from cfg import *
from util import ISTFT
from util import load_soundfile
from util import standard_specgram
from util import stft
from scikits.audiolab import wavwrite

import matplotlib
# http://www.astrobetter.com/plotting-to-a-file-in-python/
matplotlib.use('PDF')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})

background = 1.
foreground = 0.


def create_simpler_data():
    x_signal = load_soundfile(signal_files, 0, 10)
    x_noise = load_soundfile(noise_files, 0, 10)
    return x_signal, x_noise


def load_soundfiles(signal, noise):
    x_signal = load_soundfile(signal, 0)
    x_noise = load_soundfile(noise, 0)
    return x_signal, x_noise


def build_dataset_one_signal_frame(x_signal, x_noise, framelength, k, num_minibatches,
    minibatch_size, specbinnum, numtimebins, n_noise_only_examples, signal_only=False):
    """
        x_signal: CLEAN "one frame" --> equivalent length to one spectrogram
        x_noise: NOISE longer noise-only signal
        k: linear value of SNR
    """
    n_start = int(len(x_noise)/2)

    def _norm_signal(x):
        # x -= np.mean(x)
        return 0.5 * x / max(abs(x))

    def _avg_energy_scale(x, y):
        pwr_x = np.sum(x**2)
        pwr_y = np.sum(y[n_start:n_start+len(x)]**2)
        return np.sqrt(pwr_y/pwr_x)

    # import ipdb; ipdb.set_trace()
    # override to use just sine waves for now
    n = np.linspace(0,len(x_signal),len(x_signal))
    x_signal = np.sin(2.*np.pi*440./44100. * n)
    # plt.figure()
    # plt.plot(x_signal[0:100])
    # plt.savefig('x_sig_test.png')

    # override to use just awgn for now as well
    x_noise = np.random.normal(0, 1, len(x_noise))

    dtype = theano.config.floatX

    scale_factor = _avg_energy_scale(x_signal, x_noise)
    x_signal = _norm_signal(scale_factor * x_signal)
    x_noise = _norm_signal(x_noise)
    x_clean = np.copy(x_signal)

    # prevent clipping
    if k < 1:
        x_signal = x_signal + k * x_noise[n_start:n_start+len(x_signal)]
	x_noise = k * x_noise[0:n_start]
    else:
        x_signal = 1/k * x_signal + x_noise[n_start:n_start+len(x_signal)]
	x_noise = x_noise[0:n_start]

    noise_real, noise_imag = stft(x_noise)
    signal_real, signal_imag = stft(x_signal)
    clean_real, clean_imag = stft(x_clean)

    training_data = np.zeros((num_minibatches, minibatch_size, 2, specbinnum, numtimebins), dtype=dtype)
    training_labels = np.zeros((num_minibatches, minibatch_size), dtype=dtype)
    num_time_samples = int(framelength/2 * (numtimebins + 1))

    noise_minibatch_range = range(n_noise_only_examples)

    for which_training_batch in range(num_minibatches):
        for which_training_datum in range(minibatch_size):
            if which_training_datum in noise_minibatch_range and not signal_only:
                re = noise_real
                im = noise_imag
                label = background
                startindex = np.random.randint(re.shape[1] - numtimebins)
            else:
                re = signal_real
                im = signal_imag
                label = foreground
                startindex = 0
            time_start = int(framelength/2 * (startindex + 1) - framelength/2)
            time_end = time_start + num_time_samples

            training_data[which_training_batch, which_training_datum, 0, :, :] = re[:, startindex:startindex+numtimebins]
            training_data[which_training_batch, which_training_datum, 1, :, :] = im[:, startindex:startindex+numtimebins]
            training_labels[which_training_batch, which_training_datum] = label

    return {
        'training_labels': training_labels.reshape(
            training_labels.shape[0], training_labels.shape[1], 1, 1, 1),
        'training_data': training_data,
        'clean_real': clean_real,
        'clean_imag': clean_imag,
        'signal_real': signal_real,
        'signal_imag': signal_imag,
        'noise_real': noise_real,
        'noise_imag': noise_imag,
        'clean_time_signal': x_clean,
        'noisy_time_signal': x_signal,
    }

if __name__ == '__main__':
    pass

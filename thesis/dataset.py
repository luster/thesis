from __future__ import division

import os

from random import randint, uniform
from glob import glob

import numpy as np
from numpy import float32, complex64

import theano
import theano.tensor as T

import lasagne

from cfg import *
from util import standard_specgram, load_soundfile, stft

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
    minibatch_size, specbinnum, numtimebins, n_noise_only_examples):
    """
        x_signal: CLEAN "one frame" --> equivalent length to one spectrogram
        x_noise: NOISE longer noise-only signal
        k: linear value of SNR
    """
    def _norm_signal(x):
        x -= np.mean(x)
        return 0.5 * x / max(abs(x))

    def _avg_energy_scale(x, y):
        pwr_x = np.sum(x**2)
        pwr_y = np.sum(y[0:len(x)]**2)
        return np.sqrt(pwr_y/pwr_x)

    dtype = theano.config.floatX

    scale_factor = _avg_energy_scale(x_signal, x_noise)
    x_signal = _norm_signal(scale_factor * x_signal)
    x_noise = _norm_signal(x_noise)
    x_clean = np.copy(x_signal)

    # prevent clipping
    if k < 1:
        x_signal = x_signal + k * x_noise[0:len(x_signal)]
    else:
        x_signal = 1/k * x_signal + x_noise[0:len(x_signal)]
    x_noise = x_noise[len(x_signal)+1:]

    noise_real, noise_imag = stft(x_noise)
    signal_real, signal_imag = stft(x_signal)
    clean_real, clean_imag = stft(x_clean)

    training_data = np.zeros((num_minibatches, minibatch_size, 2, specbinnum, numtimebins), dtype=dtype)
    training_labels = np.zeros((num_minibatches, minibatch_size), dtype=dtype)
    num_time_samples = int(framelength/2 * (numtimebins + 1))

    noise_minibatch_range = range(n_noise_only_examples)

    for which_training_batch in range(num_minibatches):
        for which_training_datum in range(minibatch_size):
            if which_training_datum in noise_minibatch_range:
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
        'training_labels': training_labels,
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
    data, labels, noisegram, signalgram, x_noise, x_signal, noise_phasegram, signal_phasegram = build_dataset()

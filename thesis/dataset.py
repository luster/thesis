from __future__ import division

import os

from random import randint, uniform
from glob import glob

import numpy as np
from numpy import float32, complex64

import theano
import theano.tensor as T

import lasagne

from config import *
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


def build_dataset3(x_signal, x_noise, sec_of_audio, k, training_data_size,
    minibatch_size, specbinnum, numtimebins, n_noise_only_examples, index=0):

    dtype = theano.config.floatX
    freq_transform = standard_specgram

    x_clean = np.copy(x_signal)

    # need a slice of x_signal and x_noise to add, obviously of the same size
    if len(x_signal) > len(x_noise):
        indx = len(x_noise)
    else:
        indx = len(x_signal)

    if indx > sec_of_audio*srate:
        start = np.random.randint(indx - sec_of_audio*srate)
        end = start + sec_of_audio*srate + 1
    else:
        raise ValueError('not enough audio data')

    x_signal = x_signal[start:end]
    x_noise = x_noise[start:end]
    x_clean = x_clean[start:end]

    # prevent clipping
    if k < 1:
        x_signal = x_signal + k * x_noise
    else:
        x_signal = 1/k * x_signal + x_noise
    noise_specgram, noise_phasegram = freq_transform(x_noise)
    signal_specgram, signal_phasegram = freq_transform(x_signal)
    clean_specgram, clean_phasegram = freq_transform(x_clean)

    training_data_magnitude = np.zeros((training_data_size, minibatch_size, 1, specbinnum, numtimebins), dtype=dtype)
    training_data_phase = np.zeros((training_data_size, minibatch_size, 1, specbinnum, numtimebins), dtype=dtype)
    training_labels = np.zeros((training_data_size, minibatch_size), dtype=dtype)
    num_time_samples = int(audioframe_len/2 * (numtimebins + 1))
    # training_data_time = np.zeros((training_data_size, minibatch_size, 1, 1, num_time_samples))

    noise_minibatch_range = range(n_noise_only_examples)

    for which_training_batch in range(training_data_size):
        for which_training_datum in range(minibatch_size):
            if which_training_datum in noise_minibatch_range:
                specgram = noise_specgram
                phasegram = noise_phasegram
                label = background
                timesig = x_noise
            else:
                specgram = signal_specgram
                phasegram = signal_phasegram
                label = foreground
                timesig = x_signal
            startindex = np.random.randint(specgram.shape[1] - numtimebins)
            time_start = int(audioframe_len/2 * (startindex + 1) - audioframe_len/2)
            time_end = time_start + num_time_samples

            training_data_magnitude[which_training_batch, which_training_datum, :, :, :] = specgram[:, startindex:startindex+numtimebins]
            training_data_phase[which_training_batch, which_training_datum, :, :, :] = phasegram[:, startindex:startindex+numtimebins]
            training_labels[which_training_batch, which_training_datum] = label
            # import ipdb; ipdb.set_trace()
            # training_data_time[which_training_batch, which_training_datum, :, :, :] = timesig[time_start:time_end]

    return {
        'training_labels': training_labels,
        'training_data_magnitude': training_data_magnitude,
        'training_data_phase': training_data_phase,
        # 'training_data_time': training_data_time,
        'clean_magnitude': clean_specgram,
        'clean_phase': clean_phasegram,
        'signal_magnitude': signal_specgram,
        'signal_phase': signal_phasegram,
        'noise_magnitude': noise_specgram,
        'noise_phase': noise_phasegram,
        'clean_time_signal': x_clean,
        'noisy_time_signal': x_signal,
    }


def build_dataset_one_signal_frame(x_signal, x_noise, k, num_minibatches,
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
    freq_transform = standard_specgram

    scale_factor = _avg_energy_scale(x_signal, x_noise)
    x_signal = _norm_signal(scale_factor * x_signal)
    x_noise = _norm_signal(x_noise)
    x_clean = np.copy(x_signal)

    #x_signal = x_signal[start:end]
    #x_noise = x_noise[start:end]
    #x_clean = x_clean[start:end]

    # prevent clipping
    if k < 1:
        x_signal = x_signal + k * x_noise[0:len(x_signal)]
    else:
        x_signal = 1/k * x_signal + x_noise[0:len(x_signal)]
    x_noise = x_noise[len(x_signal)+1:]

    noise_specgram, noise_phasegram = freq_transform(x_noise)
    signal_specgram, signal_phasegram = freq_transform(x_signal)
    clean_specgram, clean_phasegram = freq_transform(x_clean)

    training_data_magnitude = np.zeros((num_minibatches, minibatch_size, 1, specbinnum, numtimebins), dtype=dtype)
    training_data_phase = np.zeros((num_minibatches, minibatch_size, 1, specbinnum, numtimebins), dtype=dtype)
    training_labels = np.zeros((num_minibatches, minibatch_size), dtype=dtype)
    num_time_samples = int(audioframe_len/2 * (numtimebins + 1))

    noise_minibatch_range = range(n_noise_only_examples)

    for which_training_batch in range(num_minibatches):
        for which_training_datum in range(minibatch_size):
            if which_training_datum in noise_minibatch_range:
                specgram = noise_specgram
                phasegram = noise_phasegram
                label = background
                startindex = np.random.randint(specgram.shape[1] - numtimebins)
            else:
                specgram = signal_specgram
                phasegram = signal_phasegram
                label = foreground
                startindex = 0
            time_start = int(audioframe_len/2 * (startindex + 1) - audioframe_len/2)
            time_end = time_start + num_time_samples

            training_data_magnitude[which_training_batch, which_training_datum, :, :, :] = specgram[:, startindex:startindex+numtimebins]
            training_data_phase[which_training_batch, which_training_datum, :, :, :] = phasegram[:, startindex:startindex+numtimebins]
            training_labels[which_training_batch, which_training_datum] = label

    return {
        'training_labels': training_labels,
        'training_data_magnitude': training_data_magnitude,
        'training_data_phase': training_data_phase,
        'clean_magnitude': clean_specgram,
        'clean_phase': clean_phasegram,
        'signal_magnitude': signal_specgram,
        'signal_phase': signal_phasegram,
        'noise_magnitude': noise_specgram,
        'noise_phase': noise_phasegram,
        'clean_time_signal': x_clean,
        'noisy_time_signal': x_signal,
    }

if __name__ == '__main__':
    data, labels, noisegram, signalgram, x_noise, x_signal, noise_phasegram, signal_phasegram = build_dataset()

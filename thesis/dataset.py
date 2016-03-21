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


def build_dataset(use_stft=False, use_simpler_data=False, k=0.5):
    if use_stft:
        # FIXME: this still doesn't work
        dtype = complex64
        freq_transform = stft
    else:
        dtype = theano.config.floatX
        freq_transform = standard_specgram

    if use_one_file and not use_simpler_data:
        x_signal = load_soundfile(signal_files, 0)
        x_noise = load_soundfile(noise_files, 0)
    else:
        x_signal, x_noise = create_simpler_data()
        x_clean = np.copy(x_signal)
        # TODO: need clean, noisy, and noise to properly evaluate
        x_signal = x_signal + k * x_noise
    noise_specgram, noise_phasegram = freq_transform(x_noise)
    signal_specgram, signal_phasegram = freq_transform(x_signal)

    training_data = np.zeros((training_data_size, minibatch_size, 1, specbinnum, numtimebins), dtype=dtype)
    training_labels = np.zeros((training_data_size, minibatch_size), dtype=dtype)

    noise_minibatch_range = range(n_noise_only_examples)

    for which_training_batch in range(training_data_size):
        for which_training_datum in range(minibatch_size):
            if which_training_datum in noise_minibatch_range:
                specgram = noise_specgram
                label = background
            else:
                specgram = signal_specgram
                label = foreground
            startindex = np.random.randint(specgram.shape[1]-numtimebins)
            training_data[which_training_batch, which_training_datum, :, :, :] = specgram[:, startindex:startindex+numtimebins]
            training_labels[which_training_batch, which_training_datum] = label
    return (training_data, training_labels,
        noise_specgram, signal_specgram,
        x_noise, x_signal,
        noise_phasegram, signal_phasegram)

def build_dataset2(use_stft=False, use_simpler_data=False, k=0.5, training_data_size=128, minibatch_size=16, specbinnum=128, numtimebins=512, n_noise_only_examples=4):
    if use_stft:
        # FIXME: this still doesn't work
        dtype = complex64
        freq_transform = stft
    else:
        dtype = theano.config.floatX
        freq_transform = standard_specgram

    if use_one_file and not use_simpler_data:
        x_signal = load_soundfile(signal_files, 0)
        x_noise = load_soundfile(noise_files, 0)
    else:
        x_signal, x_noise = create_simpler_data()
        x_clean = np.copy(x_signal)
        # prevent clipping conditions
        if k < 1:
            x_signal = x_signal + k * x_noise
        else:
            x_signal = 1/k * x_signal + x_noise
    noise_specgram, noise_phasegram = freq_transform(x_noise)
    signal_specgram, signal_phasegram = freq_transform(x_signal)
    clean_specgram, clean_phasegram = freq_transform(x_clean)

    training_data_magnitude = np.zeros((training_data_size, minibatch_size, 1, specbinnum, numtimebins), dtype=dtype)
    training_data_phase = np.copy(training_data_magnitude)
    training_labels = np.zeros((training_data_size, minibatch_size), dtype=dtype)

    noise_minibatch_range = range(n_noise_only_examples)

    for which_training_batch in range(training_data_size):
        for which_training_datum in range(minibatch_size):
            if which_training_datum in noise_minibatch_range:
                specgram = noise_specgram
                phasegram = noise_phasegram
                label = background
            else:
                specgram = signal_specgram
                phasegram = signal_phasegram
                label = foreground
            startindex = np.random.randint(specgram.shape[1] - numtimebins)
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


def load_soundfiles(signal, noise):
    x_signal = load_soundfile(signal, 0)
    x_noise = load_soundfile(noise, 0)
    return x_signal, x_noise


def build_dataset3(x_signal, x_noise, sec_of_audio=20, k=0.5, training_data_size=128,
    minibatch_size=16, specbinnum=128, numtimebins=512, n_noise_only_examples=4, index=0):

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
    training_data_phase = np.copy(training_data_magnitude)
    training_labels = np.zeros((training_data_size, minibatch_size), dtype=dtype)

    noise_minibatch_range = range(n_noise_only_examples)

    for which_training_batch in range(training_data_size):
        for which_training_datum in range(minibatch_size):
            if which_training_datum in noise_minibatch_range:
                specgram = noise_specgram
                phasegram = noise_phasegram
                label = background
            else:
                specgram = signal_specgram
                phasegram = signal_phasegram
                label = foreground
            startindex = np.random.randint(specgram.shape[1] - numtimebins)
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

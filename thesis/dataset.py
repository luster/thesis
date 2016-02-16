import os

from glob import glob

import numpy as np
from numpy import float32, complex64

import theano
import theano.tensor as T

import lasagne

from config import (
    lambduh,
    minibatch_size,
    hop,
    n_freq_bins,
    n_iterations,
    noise_only_fraction,
    n_noise_only_examples,
    specgram_timeframes,
    fft_bins,
    n_latents,
    conv_filter_length,
    maxpooling_downsample_factor,
    noise_files,
    signal_files,
    use_one_file,
    specbinnum,
    numtimebins,
    training_data_size,
    use_complex,
)
from util import standard_specgram, load_soundfile, stft

background = 1.
foreground = 0.


def create_simple_data():
    # create a sum of sine waves, add bg noise
    return None, None


def build_dataset(use_stft=False, use_simple_data=False):
    if use_stft:
        # FIXME: this still doesn't work
        dtype = complex64
        freq_transform = stft
    else:
        dtype = theano.config.floatX
        freq_transform = standard_specgram

    if use_one_file and not use_simple_data:
        x_signal = load_soundfile(signal_files, 0)
        x_noise = load_soundfile(noise_files, 0)
    else:
        x_signal, x_noise = create_simple_data()
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


if __name__ == '__main__':
    data, labels, noisegram, signalgram, x_noise, x_signal, noise_phasegram, signal_phasegram = build_dataset()

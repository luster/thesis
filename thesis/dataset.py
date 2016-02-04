import os

from glob import glob

import numpy as np
from numpy import float32

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
)
from util import standard_specgram, load_soundfile

background = 0
foreground = 1

training_data_size = 100

def build_dataset():
    if use_one_file:
        noise_specgram = standard_specgram(load_soundfile(noise_files, 0))
        signal_specgram = standard_specgram(load_soundfile(signal_files, 0))

    training_data = np.zeros((training_data_size, minibatch_size, 1, specbinnum, numtimebins), dtype=float32)
    training_labels = np.zeros((training_data_size, minibatch_size))

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
    return training_data, training_labels


if __name__ == '__main__':
    data = build_dataset()

import os

from glob import glob

import numpy as np
import theano
import theano.tensor as T

import lasagne

numepochs = 512

lambduh = 0.75  # lambda
minibatch_size = 16
hop = 0.5
n_freq_bins = 32
n_iterations = 10**6
noise_only_fraction = 0.25

srate = 44100
wavdownsample = 1

n_noise_only_examples = int(noise_only_fraction * minibatch_size)
background = 0
foreground = 1

fft_bins = 128
audioframe_len = 512
audioframe_stride = audioframe_len/2
specbinlow = 0
specbinnum = fft_bins
numtimebins = 160 # 128 # 48 # NOTE that this size needs really to be compatible with downsampling (maxpooling) steps if you use them.

specgram_timeframes = 512
n_latents = 32
numfilters = 6
conv_filter_length = 9  # time frames
maxpooling_downsample_factor = 16

# train with AdaDelta to control SGD learning rates
# no dropout
# init tensor of filters as set of K random orth unit vectors of length MH, M x H x K
# single layer autoencoder
# data normalization at input layer, use of mu and sigma


data_folder = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data'))
concert_prefix = 'gj'
noise = 'noise'
signal = 'song'
concert_folder = os.path.join(data_folder, '{}_mono'.format(concert_prefix))
noise_pattern = os.path.join(concert_folder, '{}_{}*.wav'.format(concert_prefix, noise))
signal_pattern = os.path.join(concert_folder, '{}_{}*.wav'.format(concert_prefix, signal))
noise_files = glob(noise_pattern)
signal_files = glob(signal_pattern)

use_one_file = True
if use_one_file:
    noise_files = noise_files[0]
    signal_files = signal_files[0]

use_maxpool = True

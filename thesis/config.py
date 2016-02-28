import os

from glob import glob

import numpy as np
import theano
import theano.tensor as T

import lasagne

use_complex = False
if use_complex:
    dtype = np.complex64
else:
    dtype = theano.config.floatX

training_data_size = 768
numepochs = 256

lambduh = 0.75  # lambda
minibatch_size = 16
hop = 0.5
n_freq_bins = 32
n_iterations = 10**6
noise_only_fraction = 0.25

srate = 44100
wavdownsample = 1

n_noise_only_examples = int(noise_only_fraction * minibatch_size)

fft_bins = 512
audioframe_len = 512
audioframe_stride = int(audioframe_len/2)
specbinlow = 0
specbinnum = fft_bins
numtimebins = 512 # 128 # 48 # NOTE that this size needs really to be compatible with downsampling (maxpooling) steps if you use them.

specgram_timeframes = 512
n_latents = 32
n_background_latents = int(0.25 * n_latents)
numfilters = 32
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

use_simpler_data = True
if use_simpler_data:
    noise_file = 'golf_club_bar_lunch_time.wav'
    sig_file = 'golf_club_bar_lunch_time_AND_guitar.wav'
    noise_files = glob(os.path.join(data_folder, noise_file))[0]
    signal_files = glob(os.path.join(data_folder, sig_file))[0]
    print noise_files, signal_files

use_maxpool = True

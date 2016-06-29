minibatches = 1
examples_per_minibatch = 16
framelength = 128
overlap = framelength/2
freq_bins = 128
time_bins = 64

percent_background_latents = 0.25
percent_noise_only_examples = 0.5

lambduh = 0.75
lambduh_finetune = 8

fs = 44100

niter_pretrain = 200
niter_finetune = 200

import lasagne
get_output = lasagne.layers.get_output
get_all_params = lasagne.layers.get_all_params


# calculated based on params
n_noise_only_examples = int(
    percent_noise_only_examples * examples_per_minibatch)

import theano
dtype = theano.config.floatX

snr = 20  # db

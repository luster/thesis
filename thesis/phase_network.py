"""Build time domain network

input: minibatches x minibatchsize x 1 x framelength x 1
latentspace: minibatches x minibatchsize x
output: minibatches x minibatchsize x 1 x framelength x 1

"""
from __future__ import division

import scikits.audiolab
import numpy as np
import sys
from datetime import datetime

import lasagne
import theano
import theano.tensor as T

import matplotlib
# http://www.astrobetter.com/plotting-to-a-file-in-python/
matplotlib.use('PDF')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})

from config import audioframe_len, specbinnum, srate
from dataset import build_dataset3, load_soundfiles
from build_networks import dtype, PartitionedAutoencoder
from util import calculate_time_signal
from sklearn.metrics import mean_squared_error

from lasagne.layers import batch_norm, DenseLayer
from lasagne.nonlinearities import elu, tanh, identity
from conv_layer import custom_convlayer_2
from norm_layer import NormalisationLayer
from build_networks import PartitionedAutoencoder
from build_networks import ZeroOutBackgroundLatentsLayer


def normalize(this, against):
    return this / max(abs(this)) * max(abs(against))


def phase_activation(x):
    return T.tanh(x) * np.pi


class PartitionedAutoencoderForPhase(PartitionedAutoencoder):

    def add_layer(self, network, in_chans, out_chans, nonlinearity=elu):
        print in_chans, out_chans
        network, _ = custom_convlayer_2(network, in_num_chans=in_chans, out_num_chans=out_chans, nonlinearity=elu)
        network = batch_norm(network)
        return network

    def initialize_network(self):
        network = lasagne.layers.InputLayer((None, 1, self.specbinnum, self.numtimebins), self.input_var)
        network = NormalisationLayer(network, self.specbinnum)
        self.normlayer = network

        input_output_pairs = self.get_layer_sizes()

        for in_chans, out_chans in input_output_pairs:
            network = self.add_layer(network, in_chans, out_chans)
        self.latents = network
        network = ZeroOutBackgroundLatentsLayer(self.latents,
            mp_down_factor=self.mp_down_factor,
            numfilters=self.numfilters,
            numtimebins=self.numtimebins,
            background_latents_factor=self.background_latents_factor,
            use_maxpool=self.use_maxpool)
        reversed_network_sizes = list(reversed(network_sizes))
        unfolded_input_output_pairs = zip(reversed_network_sizes[0:-1], reversed_network_sizes[1:])
        for in_chans, out_chans in unfolded_input_output_pairs[0:-1]:
            network = self.add_layer(network, in_chans, out_chans)
        # last layer we do separately
        in_chans, out_chans = unfolded_input_output_pairs[-1]
        network, _ = custom_convlayer_2(network, in_num_chans=in_chans, out_num_chans=out_chans, nonlinearity=softplus)
        network = batch_norm(network)

        self.network = network

    # def loss_func(self, lambduh=0.75):
    #     prediction = self.get_output()
    #     loss = lasagne.objectives.squared_error(prediction, self.input_time_var)
    #     regularization_term = self.soft_output_var * ((self.C_mat * lasagne.layers.get_output(self.latents)).mean())**2
    #     loss = (loss.mean() + lambduh/self.mean_C * regularization_term).mean()
    #     return loss

class PhaseNeuralNet(PartitionedAutoencoder):

    def initialize_network(self):
        network = lasagne.layers.InputLayer((None, 1, self.specbinnum, self.numtimebins), self.input_var)


        self.network = network

    def loss_func(self, lambduh=0.5):
        pass

    def train_fn_slim(self, updates='adadelta'):
        loss = self.loss_func()
        update_args = {
            'adadelta': (lasagne.updates.adadelta, {
                'learning_rate': 0.01, 'rho': 0.4, 'epsilon': 1e-6,
            }),
            'adam': (lasagne.updates.adam, {},),
        }[updates]
        update_func, update_params = update_args[0], update_args[1]

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = update_func(loss, params, **update_params)
        train_fn = theano.function([self.input_var, self.soft_output_var],
            loss, updates=updates,
            allow_input_downcast=True,
        )
        return train_fn


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
from lasagne.nonlinearities import elu, tanh
from conv_layer import custom_convlayer_2
from norm_layer import NormalisationLayer
from build_networks import PartitionedAutoencoder
from build_networks import ZeroOutBackgroundLatentsLayer


def normalize(this, against):
    return this / max(abs(this)) * max(abs(against))


def phase_activation(x):
    return T.tanh(x) * np.pi


class PartitionedAutoencoderForPhase(PartitionedAutoencoder):

    def initialize_network(self):
        network = lasagne.layers.InputLayer((None, 1, self.specbinnum, self.numtimebins), self.input_var)
        network = NormalisationLayer(network, self.specbinnum)
        self.normlayer = network

        # intermediate layer size
        ils = int((self.specbinnum + self.numfilters) / 2)
        network, _ = custom_convlayer_2(network, in_num_chans=self.specbinnum, out_num_chans=ils)
        network = batch_norm(network)
        network, _ = custom_convlayer_2(network, in_num_chans=ils, out_num_chans=self.numfilters)
        network = lasagne.layers.NonlinearityLayer(network, nonlinearity=elu)
        # if self.use_maxpool:
        #     mp_down_factor = self.maxpooling_downsample_factor
        #     network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, self.mp_down_factor), stride=(1, self.mp_down_factor))
        #     maxpool_layer = network
        self.latents = network
        network = ZeroOutBackgroundLatentsLayer(self.latents,
            mp_down_factor=self.mp_down_factor,
            numfilters=self.numfilters,
            numtimebins=self.numtimebins,
            background_latents_factor=self.background_latents_factor,
            use_maxpool=self.use_maxpool)
        # if self.use_maxpool:
        #     network = lasagne.layers.InverseLayer(network, maxpool_layer)
        network, _ = custom_convlayer_2(network, in_num_chans=self.numfilters, out_num_chans=ils)
        network = batch_norm(network)
        network, _ = custom_convlayer_2(network, in_num_chans=ils, out_num_chans=self.specbinnum, nonlinearity=phase_activation)
        network = batch_norm(network)

        self.network = network

    # def loss_func(self, lambduh=0.75):
    #     prediction = self.get_output()
    #     loss = lasagne.objectives.squared_error(prediction, self.input_time_var)
    #     regularization_term = self.soft_output_var * ((self.C_mat * lasagne.layers.get_output(self.latents)).mean())**2
    #     loss = (loss.mean() + lambduh/self.mean_C * regularization_term).mean()
    #     return loss
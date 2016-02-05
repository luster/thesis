# Borrowed from Dan Stowell via https://github.com/danstowell/autoencoder-specgram
import numpy as np

import theano
import theano.tensor as T

from lasagne.layers.base import Layer
from lasagne.nonlinearities import leaky_rectify
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import very_leaky_rectify

from config import *

featframe_len = 9  # ???

def custom_convlayer(network, in_num_chans, out_num_chans):
    "Applies our special padding and reshaping to do 1D convolution on 2D data"
    network = lasagne.layers.PadLayer(network, width=(featframe_len-1)/2, batch_ndim=3) # NOTE: the "batch_ndim" is used to stop batch dims being padded, but here ALSO to skip first data dim
    print("shape after pad layer: %s" % str(network.output_shape))
    network = lasagne.layers.Conv2DLayer(network, out_num_chans, (in_num_chans, featframe_len), stride=(1,1), pad=0, nonlinearity=very_leaky_rectify, W=lasagne.init.Orthogonal()) # we pad "manually" in order to do it in one dimension only
    filters = network.W
    network = lasagne.layers.ReshapeLayer(network, ([0], [2], [1], [3])) # reinterpret channels as rows
    print("shape after conv layer: %s" % str(network.output_shape))
    return network, filters

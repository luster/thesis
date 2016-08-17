"""this file/script helps clean up the current partitioned autoencoder network
so that another can be made to train to correct phase noise
"""
from datetime import datetime
import lasagne
import numpy as np
import theano
import theano.tensor as T

from lasagne.layers import batch_norm
from lasagne.nonlinearities import rectify, elu, softplus
import scikits.audiolab
from sklearn.metrics import mean_squared_error


dtype = theano.config.floatX

batchsize = 64
framelen = 1024
shape = (batchsize,framelen)
srate = 16000
latentsize = 2000
background_latents_factor = 0.25
minibatch_noise_only_factor = 0.25
n_noise_only_examples = int(minibatch_noise_only_factor * batchsize)
n_background_latents = int(background_latents_factor * latentsize)
lambduh = 0.75


class ZeroOutBackgroundLatentsLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(ZeroOutBackgroundLatentsLayer, self).__init__(incoming)
        n_background_latents = int(kwargs.get('background_latents_factor') * framelen)

        mask = np.ones(shape)
        mask[:, 0:n_background_latents] = 0
        self.mask = theano.shared(mask, borrow=True)

    def get_output_for(self, input_data, reconstruct=False, **kwargs):
        if reconstruct:
            return self.mask * input_data
        return input_data


def dan_net():
    # net
    x = T.matrix('X')  # input
    y = T.matrix('Y')  # soft label
    network = lasagne.layers.InputLayer(shape, x)
    network = lasagne.layers.DenseLayer(network, latentsize, nonlinearity=lasagne.nonlinearities.rectify)
    latents = network
    network = ZeroOutBackgroundLatentsLayer(latents, background_latents_factor=background_latents_factor)
    network = lasagne.layers.DenseLayer(network, shape[0], nonlinearity=lasagne.nonlinearities.identity)

    # loss
    C = np.zeros((batchsize,latentsize))
    C[0:n_noise_only_examples, n_background_latents + 1:] = 1
    C_mat = theano.shared(np.asarray(C, dtype=dtype), borrow=True)
    mean_C = theano.shared(C.mean(), borrow=True)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, x)
    square_term = loss.mean()
    regularization_term = y * ((C_mat * lasagne.layers.get_output(latents)).mean())**2
    loss = (loss.mean() + lambduh/mean_C * regularization_term).mean()

    # training compilation
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adamax(loss, params)
    train_fn = theano.function([x,y], loss, updates=updates)

    # inference compilation
    predict_fn = theano.function([x], lasagne.layers.get_output(network, deterministic=True, reconstruct=True))

    #
    # other objectives
    #
    def do_stuff(network, latents, predict_fn):
        pass

    return network, latents, loss, square_term, regularization_term.mean(), train_fn, predict_fn, do_stuff

def dan_main(args):
    network, latents, loss, square_loss, reg_loss, train_fn, predict_fn, do_stuff = dan_net()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, help='niter', default=100)
    args = parser.parse_args()
    dan_main(args)
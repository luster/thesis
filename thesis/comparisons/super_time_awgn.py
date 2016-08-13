from __future__ import division

import numpy as np
import theano
import theano.tensor as T
import lasagne

nonlin = lasagne.nonlinearities
layers = lasagne.layers
batch_norm = lasange.layers.batch_norm
dtype = theano.config.floatX

default_params = {
    'batchsize': 16,
    'framelen': 1024,
    'srate': 16000,
    'window': np.hanning,
    'pct': 0.5,
    'reg': False,
}
default_params['fftlen'] = default_params['framelen']


def simple_autoencoder(params=default_params):
    # network
    shape = (params.get('batchsize'), params.get('framelen'),)
    x = T.matrix('x')  # dirty
    s = T.matrix('s')  # clean
    in_layer = batch_norm(layers.InputLayer(shape, x))
    bottlesize = int(shape[1]/2)
    h1 = batch_norm(layers.DenseLayer(in_layer, bottlesize, nonlinearity=nonlin.leaky_rectify))
    x_hat = layers.DenseLayer(h1, shape[1], nonlinearity=nonlin.identity)

    return x_hat, x, s

def loss(x_hat, x, s, reg=False):
    prediction = layers.get_output(x_hat)
    loss = lasagne.objectives.squared_error(prediction, s)
    if reg:
        reg = 2 * (1e-5 * lasagne.regularization.regularize_network_params(x_hat, lasagne.regularization.l2) + \
              1e-6 * lasagne.regularization.regularize_network_params(x_hat, lasagne.regularization.l1))
        loss = loss + reg
    return loss

def train(x_hat, x, s, loss):
    params = lasagne.layers.get_all_params(autoencoder, trainable=True)
    updates = lasagne.updates.adamax(loss, params)
    train_fn = theano.function([x,s], loss, updates=updates)
    return train_fn


def get_minibatch(params=default_params, sample=False):
    # sample=False --> for training
    # sample=True --> for testing (continuous samples in minibatch)
    srate = default_params.get('srate')
    batchsize = default_params.get('batchsize')
    framelen = default_params.get('framelen')
    pct = default_params.get('pct')
    snr = default_params.get('snr')  # dB

    def _sin_f(a, f, srate, n, phase):
        return a * np.sin(2*np.pi*f/srate*n+phase)

    def _noise_var(input_signal, snr_db):
        avg_energy = sum(input_signal * input_signal)/len(input_signal)
        snr_lin = 10**(snr_db/10)
        return avg_energy/snr_lin

    if sample:
        n = np.linspace(0, batchsize * framelen - 1, batchsize * framelen)
        phase = np.random.uniform(0.0, 2*np.pi)
        amp = np.random.uniform(0.35, 0.6)
    else:
        n = np.tile(np.linspace(0, framelen-1, framelen), (batchsize,1))
        phase = np.tile(np.random.uniform(0.0, 2*np.pi, batchsize), (framelen, 1)).transpose()
        amp = np.tile(np.random.uniform(0.35, 0.65, batchsize), (framelen,1)).transpose()
    # clean = amp * np.sin(2 * np.pi * f / srate * n + phase)
    clean = _sin_f(amp,441,srate,n,phase) + _sin_f(amp, 635.25,srate,n,phase) + _sin_f(amp,528,srate,n,phase) + _sin_f(amp,880,srate,n,phase)

    # corrupt with gaussian noise
    import ipdb; ipdb.set_trace()
    # noise_var = _noise_var(clean[], snr)
    noise = np.random.normal(0, noise_var, clean.shape)
    noisy = clean + noise

    if sample:
        noisy = np.array([noisy[i:i+framelen] for i in xrange(0, len(noisy), int(pct*framelen))][0:batchsize])
        clean = np.array([clean[i:i+framelen] for i in xrange(0, len(clean), int(pct*framelen))][0:batchsize])

    return clean.astype(dtype), noisy.astype(dtype), n, None

def sim():
    # get network
    x_hat, x, s = simple_autoencoder()
    # get loss function
    loss = loss(x_hat, x, s, reg=False)
    # get train function
    train_fn = train(x_hat, x, s, loss)



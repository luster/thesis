from __future__ import division

import numpy as np
import theano
import theano.tensor as T
import lasagne
from scikits.audiolab import wavwrite
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from autoencoder import mod_relu

get_output = lasagne.layers.get_output
nonlin = lasagne.nonlinearities
layers = lasagne.layers
batch_norm = lasagne.layers.batch_norm
dtype = theano.config.floatX

default_params = {
    'batchsize': 64,
    'framelen': 1024,
    'srate': 16000,
    'window': np.hanning,
    'pct': 0.5,
    'reg': False,
    'snr': 100,  # dB
    'niter': 800,
}
default_params['fftlen'] = default_params['framelen']


def simple_autoencoder(params=default_params):
    # network
    shape = (params.get('batchsize'), params.get('framelen'),)
    x = T.matrix('x')  # dirty
    s = T.matrix('s')  # clean
    in_layer = layers.InputLayer(shape, x)
    bottlesize = int(shape[1]/2)
    print 'bottle: ', bottlesize
    h1 = layers.DenseLayer(in_layer, bottlesize, nonlinearity=mod_relu)
    x_hat = layers.DenseLayer(h1, shape[1], nonlinearity=nonlin.identity)

    return x_hat, x, s

def loss(x_hat, x, s, reg=False):
    prediction = layers.get_output(x_hat)
    loss_fn = lasagne.objectives.squared_error(prediction, s)
    if reg:
        reg = (1e-6 * lasagne.regularization.regularize_network_params(x_hat, lasagne.regularization.l2) + \
              1e-7 * lasagne.regularization.regularize_network_params(x_hat, lasagne.regularization.l1))
        loss_fn = loss_fn + reg
    return loss_fn.mean()

def train(x_hat, x, s, loss):
    params = lasagne.layers.get_all_params(x_hat, trainable=True)
    updates = lasagne.updates.adam(loss, params)
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

    def _noise_var(clean, snr_db):
        # we use one noise variance per minibatch
        avg_energy = np.sum(clean*clean)/clean.size
        snr_lin = 10**(snr_db/10)
        noise_var = avg_energy / snr_lin
        # print 'noise variance for minibatch: ', noise_var
        return noise_var

    if sample:
        n = np.linspace(0, batchsize * framelen - 1, batchsize * framelen)
        phase = np.random.uniform(0.0, 2*np.pi)
        amp = np.random.uniform(0.35, 0.6)
    else:
        n = np.tile(np.linspace(0, framelen-1, framelen), (batchsize,1))
        phase = np.tile(np.random.uniform(0.0, 2*np.pi, batchsize), (framelen, 1)).transpose()
        amp = np.tile(np.random.uniform(0.35, 0.65, batchsize), (framelen,1)).transpose()
    clean = _sin_f(amp,441,srate,n,phase) + _sin_f(amp, 635.25,srate,n,phase) + _sin_f(amp,528,srate,n,phase) + _sin_f(amp,880,srate,n,phase)

    # corrupt with gaussian noise
    noise_var = _noise_var(clean, snr)
    print 'noise var =', noise_var
    noise = np.random.normal(0, noise_var, clean.shape)
    noisy = clean + noise

    if sample:
        w = np.hanning(framelen)
        noisy = np.array([w*noisy[i:i+framelen] for i in xrange(0, len(noisy), int(pct*framelen))][0:batchsize])
        clean = np.array([w*clean[i:i+framelen] for i in xrange(0, len(clean), int(pct*framelen))][0:batchsize])
    else:
        w = np.tile(np.hanning(framelen), (batchsize,1))
        clean = clean * w
        noisy = noisy * w

    return clean.astype(dtype), noisy.astype(dtype), n

def run(x_hat, x, s, loss, train_fn, params=default_params):
    predict_fn = theano.function([x], get_output(x_hat))
    niter = params.get('niter')
    snrs = [0,6,12,18]

    loss_plots = []
    for snr in snrs:
        params['snr'] = snr
        loss_plot = []
        for i in xrange(niter):
            clean, noisy, n = get_minibatch(params, sample=False)
            l = train_fn(noisy, clean)
            loss_plot.append(l)
            print i, 'loss = ', l
        loss_plots.append(loss_plot)
    # test/plot
    # clean, noisy, n = get_minibatch(params, sample=True)
    plt.figure()
    for snr, loss_plot in zip(snrs, loss_plots):
        plt.semilogy(loss_plot)
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs. Number of Iterations')
    plt.legend(['{} dB'.format(str(i)) for i in snrs])
    plt.savefig('loss.pdf', format='pdf')


def sim():
    # get network
    x_hat, x, s = simple_autoencoder()
    # get loss function
    loss_fn = loss(x_hat, x, s, reg=False)
    # get train function
    train_fn = train(x_hat, x, s, loss_fn)
    # run and collect
    run(x_hat, x, s, loss_fn, train_fn)

if __name__ == '__main__':
    sim()


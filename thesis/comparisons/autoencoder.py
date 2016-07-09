from __future__ import division
# different networks (autoencoder, conv autoencoder, recurrent)
# different signals (sine, recording)
# different noises (awgn, crowd)
# different domains (time, freq)

import lasagne
import theano
import theano.tensor as T
import numpy as np
from scikits.audiolab import wavwrite
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


dtype = theano.config.floatX
batchsize = 64
framelen = 441
srate = 44100

batch_norm = lasagne.layers.batch_norm

def autoencoder(params):
    # network
    shape = (batchsize, framelen)
    x = T.matrix('x')  # dirty
    s = T.matrix('s')  # clean
    in_layer = batch_norm(lasagne.layers.InputLayer(shape, x))
    h1 = batch_norm(lasagne.layers.DenseLayer(in_layer, 400, nonlinearity=lasagne.nonlinearities.leaky_rectify))
    h2 = batch_norm(lasagne.layers.DenseLayer(h1, 330, nonlinearity=lasagne.nonlinearities.leaky_rectify))
    h3 = batch_norm(lasagne.layers.DenseLayer(h2, 300, nonlinearity=lasagne.nonlinearities.leaky_rectify))
    h4 = batch_norm(lasagne.layers.DenseLayer(h3, 270, nonlinearity=lasagne.nonlinearities.leaky_rectify))
    bottle = h4
    d4 = batch_norm(lasagne.layers.DenseLayer(h4, 300, nonlinearity=lasagne.nonlinearities.leaky_rectify))
    d3 = batch_norm(lasagne.layers.DenseLayer(d4, 330, nonlinearity=lasagne.nonlinearities.leaky_rectify))
    d2 = batch_norm(lasagne.layers.DenseLayer(d3, 400, nonlinearity=lasagne.nonlinearities.leaky_rectify))
    x_hat = batch_norm(lasagne.layers.DenseLayer(d2, framelen, nonlinearity=lasagne.nonlinearities.identity))

    # loss function
    prediction = lasagne.layers.get_output(x_hat)
    loss = lasagne.objectives.squared_error(prediction, s)
    reg = 2 * (1e-5 * lasagne.regularization.regularize_network_params(x_hat, lasagne.regularization.l2) + \
          1e-6 * lasagne.regularization.regularize_network_params(x_hat, lasagne.regularization.l1))
    loss = loss + reg
    return x_hat, x, s, loss.mean(), reg.mean(), prediction


def train(autoencoder, x, s, loss):
    params = lasagne.layers.get_all_params(autoencoder, trainable=True)
    updates = lasagne.updates.adam(loss, params)
    train_fn = theano.function([x,s], loss, updates=updates)
    return train_fn


def gen_data(sample=False):
    f = 440
    if sample:
        n = np.linspace(0, batchsize * framelen - 1, batchsize * framelen)
        phase = np.random.uniform(0.0, 2*np.pi)
        amp = np.random.uniform(0.35, 0.6)
    else:
        n = np.tile(np.linspace(0, 442, framelen), (batchsize,1))
        phase = np.tile(np.random.uniform(0.0, 2*np.pi, batchsize), (framelen, 1)).transpose()
        amp = np.tile(np.random.uniform(0.35, 0.65, batchsize), (framelen,1)).transpose()
    clean = amp * np.sin(2 * np.pi * f / srate * n + phase)
    # window
    # clean = np.hamming(framelen) * clean

    # corrupt with gaussian noise
    noise = np.random.normal(0, 1e-5, clean.shape)
    noisy = clean + noise

    if sample:
        noisy = noisy.reshape(batchsize, framelen)
        clean = clean.reshape(batchsize, framelen)

    return clean.astype(dtype), noisy.astype(dtype)


if __name__ == "__main__":
    a, x, s, loss, reg, x_hat = autoencoder({})
    train_fn = train(a,x,s,loss)
    loss_mse = theano.function([x, s], loss)
    loss_reg = theano.function([], reg)
    lmse = []
    lreg = []
    predict_fn = theano.function([x], x_hat)
    clean, noisy = gen_data()
    # wavwrite(clean[1,:], 'fig/s.wav', fs=srate, enc='pcm16')
    for i in xrange(2000):
        clean, noisy = gen_data()
        loss = train_fn(noisy, clean)
        lmse.append(loss_mse(noisy, clean))
        lreg.append(loss_reg())
        print i, loss
    clean, noisy = gen_data(sample=True)
    cleaned_up = predict_fn(noisy)
    cleaned_up = cleaned_up.reshape(batchsize * framelen)
    # mse calculation
    mse = mean_squared_error(cleaned_up, clean.reshape(batchsize * framelen))
    print 'mse ', mse
    wavwrite(clean.reshape(batchsize * framelen), 'fig/s.wav', fs=srate, enc='pcm16')
    wavwrite(noisy.reshape(batchsize * framelen), 'fig/xn.wav', fs=srate, enc='pcm16')
    wavwrite(cleaned_up, 'fig/x.wav', fs=srate, enc='pcm16')
    plt.figure()
    plt.subplot(311)
    plt.plot(clean[0,:])
    plt.plot(noisy[0,:])
    # plt.plot(noisy[1,:])
    plt.plot(cleaned_up[0:framelen])
    plt.subplot(312)
    plt.plot(lmse)
    plt.semilogy(lmse)
    plt.subplot(313)
    plt.plot(lreg)
    plt.semilogy(lreg)
    plt.savefig('fig/x.svg', format='svg')


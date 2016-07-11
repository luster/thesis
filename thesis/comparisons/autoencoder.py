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

def mod_relu(x):
    eps = 1e-5
    return T.switch(x > eps, x, -eps/(x-1-eps))

fftlen = 1024
framelen = fftlen
# overlap = int(fraelen/2)

def paris_net(params):
    shape = (batchsize, fftlen)
    x = T.matrix('x')  # dirty
    s = T.matrix('s')  # clean
    in_layer = batch_norm(lasagne.layers.InputLayer(shape, x))
    h1 = lasagne.layers.DenseLayer(in_layer, 2000, nonlinearity=mod_relu)

    # loss function
    prediction = lasagne.layers.get_output(h1)
    loss = lasagne.objectives.squared_error(prediction, s)
    return h1, x, s, loss.mean(), None, prediction


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


from numpy import complex64
import scipy
def stft(x, framelen, overlap=int(0.25*framelen)):
    w = scipy.hamming(framelen)
    X = np.array([scipy.fft(w*x[i:i+framelen], freq_bins)
                     for i in range(0, len(x)-framelen, overlap)], dtype=complex64)
    X = np.transpose(X)
    return np.abs(X), np.angle(X)


def gen_freq_data(sample=False):
    clean, noisy = gen_data(sample)
    import ipdb; ipdb.set_trace()
    # get FFTs
    clean_stft = stft(clean, framelen)  # mag, phase
    noisy_stft = stft(noisy, framelen)  # mag, phase
    return clean_stft, noisy_stft  # (mag, phase), (mag, phase)

def istft(X, framelen, overlap=int(0.25*framelen)):
    time_bins = X.shape[1]  # or 0? /shrug
    x = scipy.zeros(framelen/2*(time_bins + 1))
    w = scipy.hamming(framelen)
    for n,i in enumerate(range(0, len(x)-framelen, overlap)):
        x[i:i+framelen] += scipy.real(scipy.ifft(X[:, n], framelen))
    return x

def ISTFT(mag, phase):
    stft = mag * np.exp(1j*phase)
    return istft(stft, framelen)

def paris_main(params):
    a, x, s, loss, _, x_hat = paris_net({})
    train_fn = train(a,x,s,loss)
    lmse = []
    predict_fn = theano.function([x], x_hat)
    for i in xrange(1000):
        clean, noisy = gen_freq_data()
        import ipdb; ipdb.set_trace()
        loss = train_fn(noisy[0], clean[0])
        lmse.append(loss)
        print i, loss
    clean, noisy = gen_freq_data(sample=True)
    cleaned_up = predict_fn(noisy[0])
    cleaned_up_time = ISTFT(cleaned_up, noisy[1])
    clean_time = ISTFT(clean[0], clean[1])
    mse = mean_squared_error(cleaned_up_time, clean_time)
    print 'mse ', mse
    wavwrite(cleaned_up_time, 'paris/x.wav', fs=srate, enc='pcm16')
    plt.figure()
    plt.subplot(211)
    plt.plot(cleaned_up_time)
    plt.plot(clean_time)
    plt.subplot(212)
    plt.semilogy(lmse)
    plt.savefig('paris/x.svg', format='svg')


if __name__ == "__main__":
    paris_main({})
    # a, x, s, loss, reg, x_hat = autoencoder({})
    # train_fn = train(a,x,s,loss)
    # loss_mse = theano.function([x, s], loss)
    # loss_reg = theano.function([], reg)
    # lmse = []
    # lreg = []
    # predict_fn = theano.function([x], x_hat)
    # # clean, noisy = gen_data()
    # # wavwrite(clean[1,:], 'fig/s.wav', fs=srate, enc='pcm16')
    # for i in xrange(2000):
    #     clean, noisy = gen_data()
    #     loss = train_fn(noisy, clean)
    #     lmse.append(loss_mse(noisy, clean))
    #     lreg.append(loss_reg())
    #     print i, loss
    # clean, noisy = gen_data(sample=True)
    # cleaned_up = predict_fn(noisy)
    # cleaned_up = cleaned_up.reshape(batchsize * framelen)
    # # mse calculation
    # mse = mean_squared_error(cleaned_up, clean.reshape(batchsize * framelen))
    # print 'mse ', mse
    # wavwrite(clean.reshape(batchsize * framelen), 'fig/s.wav', fs=srate, enc='pcm16')
    # wavwrite(noisy.reshape(batchsize * framelen), 'fig/xn.wav', fs=srate, enc='pcm16')
    # wavwrite(cleaned_up, 'fig/x.wav', fs=srate, enc='pcm16')
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(clean[0,:])
    # plt.plot(noisy[0,:])
    # # plt.plot(noisy[1,:])
    # plt.plot(cleaned_up[0:framelen])
    # plt.subplot(312)
    # plt.plot(lmse)
    # plt.semilogy(lmse)
    # plt.subplot(313)
    # plt.plot(lreg)
    # plt.semilogy(lreg)
    # plt.savefig('fig/x.svg', format='svg')


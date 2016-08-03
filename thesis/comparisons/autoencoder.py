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
srate = 16000
pct = 0.25  # overlap
fftlen = 1024
framelen = fftlen
# overlap = int(framelen/2)

batch_norm = lasagne.layers.batch_norm

def mod_relu(x):
    eps = 1e-5
    return T.switch(x > eps, x, -eps/(x-1-eps))


def normalize(x):
    return x / max(abs(x))


def paris_net(params):
    shape = (batchsize, fftlen)
    x = T.matrix('x')  # dirty
    s = T.matrix('s')  # clean
    in_layer = batch_norm(lasagne.layers.InputLayer(shape, x))
    h1 = lasagne.layers.DenseLayer(in_layer, 2000, nonlinearity=mod_relu)
    h1 = lasagne.layers.DenseLayer(h1, fftlen, nonlinearity=lasagne.nonlinearities.identity)

    # loss function
    prediction = lasagne.layers.get_output(h1)
    loss = lasagne.objectives.squared_error(prediction, s)
    return h1, x, s, loss.mean(), None, prediction


class RecombineLayer(lasagne.layers.ElemwiseMergeLayer):
    def __init__(self, incoming, **kwargs):
        super(RecombineLayer, self).__init__(incoming)

    def get_output_for(self, input_data, reconstruct=False, **kwargs):
        if reconstruct:
            return self.mask * input_data
        return input_data


def curro_net(params):
    shape = (batchsize, framelen)
    x = T.matrix('x')  # dirty input
    label = T.matrix('label')  # noise OR signal/noise
    in_layer = batch_norm(lasagne.layers.InputLayer(shape, x))
    layersizes=1024
    h1 = batch_norm(lasagne.layers.DenseLayer(in_layer, layersizes, nonlinearity=mod_relu))
    h2 = batch_norm(lasagne.layers.DenseLayer(h1, layersizes, nonlinearity=mod_relu))
    h3 = batch_norm(lasagne.layers.DenseLayer(h2, layersizes, nonlinearity=mod_relu))
    f = h3  # at this point, first half is signal, second half is noise
    f_sig = lasagne.layers.SliceLayer(f, indices=slice(0,int(layersizes/2)), axis=-1)
    sig_d3 = lasagne.layers.DenseLayer(f_sig, 768, nonlinearity=mod_relu)
    d3_W = sig_d3.W
    d3_b = sig_d3.b
    sig_d3 = batch_norm(sig_d3)
    sig_d2 = lasagne.layers.DenseLayer(sig_d3, layersizes, nonlinearity=mod_relu)
    d2_W = sig_d2.W
    d2_b = sig_d2.b
    sig_d2 = batch_norm(sig_d2)
    g_sig = batch_norm(lasagne.layers.DenseLayer(sig_d2, framelen, nonlinearity=lasagne.nonlinearities.identity))

    f_noi = lasagne.layers.SliceLayer(f, indices=slice(int(layersizes/2),layersizes), axis=-1)
    noi_d3 = lasagne.layers.DenseLayer(f_noi, 768, W=d3_W, b=d3_b, nonlinearity=mod_relu)
    noi_d3 = batch_norm(noi_d3)
    noi_d2 = lasagne.layers.DenseLayer(noi_d3, layersizes, W=d2_W, b=d2_b, nonlinearity=mod_relu)
    noi_d2 = batch_norm(noi_d2)
    g_noi = batch_norm(lasagne.layers.DenseLayer(noi_d2, framelen, nonlinearity=lasagne.nonlinearities.identity))

    # TODO: recombine network?

    prediction_sig = lasagne.layers.get_output(g_sig)
    prediction_noi = lasagne.layers.get_output(g_noi)
    # label is 1 for signal, 0 for noise
    prediction = label * prediction_sig + prediction_noi
    loss = lasagne.objectives.squared_error(prediction, x)

    return g_sig, g_noi, x, label, loss.mean(), None, prediction


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
    def _sin_f(a, f, srate, n, phase):
        return a * np.sin(2*np.pi*f/srate*n+phase)

    f = 440
    if sample:
        n = np.linspace(0, batchsize * framelen - 1, batchsize * framelen)
        phase = np.random.uniform(0.0, 2*np.pi)
        amp = np.random.uniform(0.35, 0.6)
    else:
        n = np.tile(np.linspace(0, framelen-1, framelen), (batchsize,1))
        phase = np.tile(np.random.uniform(0.0, 2*np.pi, batchsize), (framelen, 1)).transpose()
        amp = np.tile(np.random.uniform(0.35, 0.65, batchsize), (framelen,1)).transpose()
    # clean = amp * np.sin(2 * np.pi * f / srate * n + phase)
    clean = _sin_f(amp,441,srate,n,phase) +  _sin_f(amp, 650,srate,n,phase) + _sin_f(amp,545,srate,n,phase)

    # corrupt with gaussian noise
    noise = np.random.normal(0, 1e-5, clean.shape)
    noisy = clean + noise

    if sample:
        noisy = np.array([noisy[i:i+framelen] for i in xrange(0, len(noisy), int(pct*framelen))][0:batchsize])
        clean = np.array([clean[i:i+framelen] for i in xrange(0, len(clean), int(pct*framelen))][0:batchsize])
        #noisy = noisy.reshape(batchsize, framelen)
        #clean = clean.reshape(batchsize, framelen)

    return clean.astype(dtype), noisy.astype(dtype), n, None


def gen_batch_half_noisy_half_noise(sample=False):
    f = 440
    if sample:
        n = np.linspace(0, batchsize * framelen - 1, batchsize * framelen)
        phase = np.random.uniform(0.0, 2*np.pi)
        amp = np.random.uniform(0.35, 0.6)
    else:
        n = np.tile(np.linspace(0, framelen-1, framelen), (batchsize,1))
        phase = np.tile(np.random.uniform(0.0, 2*np.pi, batchsize), (framelen, 1)).transpose()
        amp = np.tile(np.random.uniform(0.35, 0.65, batchsize), (framelen,1)).transpose()
    clean = amp * np.sin(2 * np.pi * f / srate * n + phase)
    # window
    # clean = np.hanning(framelen) * clean

    # corrupt with gaussian noise
    noise = np.random.normal(0, 1e-8, clean.shape)
    noisy = clean + noise

    if sample:
        noisy = np.array([noisy[i:i+framelen] for i in xrange(0, len(noisy), int(pct*framelen))][0:batchsize])
        clean = np.array([clean[i:i+framelen] for i in xrange(0, len(clean), int(pct*framelen))][0:batchsize])

    if not sample:
        labels = np.zeros((batchsize,1))
        labels[0:int(batchsize/2)]=1
    else:
        # assuming "noisy" example for sample, not noise example
        labels = np.ones((batchsize,1))
    labels = np.tile(labels, (1,framelen))

    return clean.astype(dtype), noisy.astype(dtype), n, labels.astype(dtype)


from numpy import complex64
import scipy
def stft(x, framelen, overlap=int(pct*framelen)):
    w = scipy.hanning(framelen)
    X = np.array([scipy.fft(w*x[i:i+framelen], freq_bins)
                     for i in range(0, len(x)-framelen, overlap)], dtype=complex64)
    X = np.transpose(X)
    return np.abs(X), np.angle(X)


def fft(x, fftlen):
    w = np.tile((scipy.hanning(fftlen)), (batchsize, 1))
    X = scipy.fft(w*x, fftlen, axis=-1)
    return np.abs(X).astype(dtype), np.angle(X).astype(dtype)


def gen_freq_data(sample=False, gen_data_fn=gen_data):
    # for training, use FFTs of any frames
    # for testing, use FFTs of frames with 25% overlap for proper reconstruction
    clean, noisy, n, labels = gen_data_fn(sample)
    # get FFTs
    clean_stft = fft(clean, fftlen)  # mag, phase
    noisy_stft = fft(noisy, fftlen)  # mag, phase
    return clean_stft, noisy_stft, n, labels  # (mag, phase), (mag, phase)

def istft(X, framelen):
    frames_avg = int(1/pct)  # 4 in this case
    # no avg first, 
    overlap = int(pct * framelen)
    #x = scipy.zeros(int(framelen/2*(time_bins + 1)))
    x = scipy.zeros(int(X.shape[1]*(X.shape[0]*pct+1-pct)))
    for n,i in enumerate(range(0, len(x)-framelen, overlap)):
        x[i:i+framelen] += scipy.real(scipy.ifft(X[n, :]))
    return x

def ISTFT(mag, phase, framelen):
    stft = mag * np.exp(1j*phase)
    # return np.fft.ifft(stft, framelen)
    return istft(stft, framelen)

def paris_main(params):
    a, x, s, loss, _, x_hat = paris_net({})
    train_fn = train(a,x,s,loss)
    lmse = []
    predict_fn = theano.function([x], x_hat)
    for i in xrange(params.get('niter')):
        clean, noisy, n, _ = gen_freq_data()
        loss = train_fn(noisy[0], clean[0])
        lmse.append(loss)
        print i, loss
    clean, noisy, n, _ = gen_freq_data(sample=True)
    cleaned_up = predict_fn(noisy[0])
    cleaned_up_time = normalize(ISTFT(cleaned_up, noisy[1], fftlen))
    clean_time = normalize(ISTFT(clean[0], clean[1], fftlen))
    mse = mean_squared_error(cleaned_up_time, clean_time)
    print 'mse ', mse
    wavwrite(cleaned_up_time/max(abs(cleaned_up_time)), 'paris/xhat.wav', fs=srate, enc='pcm16')
    wavwrite(clean_time/max(abs(clean_time)), 'paris/x.wav', fs=srate, enc='pcm16')
    plt.figure()
    plt.subplot(411)
    #plt.plot(cleaned_up_time[0:fftlen*2])
    #plt.plot(clean_time[0:fftlen*2])
    plt.plot(cleaned_up_time[1000:1250])
    plt.plot(clean_time[1000:1250])
    plt.subplot(412)
    plt.semilogy(lmse)
    plt.subplot(413)
    plt.plot(clean[0][0,:])
    plt.subplot(414)
    plt.plot(np.unwrap(clean[1][0,:]))
    plt.savefig('paris/x.svg', format='svg')


def curro_main(params):
    g_sig, g_noi, x, s, loss, _, x_hat = curro_net({})
    train_fn = train(g_sig,x,s,loss)
    lmse = []
    predict_fn = theano.function([x], lasagne.layers.get_output(g_sig))
    for i in xrange(params.get('niter')):
        clean, noisy, n, labels = gen_freq_data(sample=False, gen_data_fn=gen_batch_half_noisy_half_noise)
        loss = train_fn(noisy[0], labels)
        lmse.append(loss)
        print i, loss
    clean, noisy, n, labels = gen_freq_data(sample=True, gen_data_fn=gen_batch_half_noisy_half_noise)
    cleaned_up = predict_fn(noisy[0])
    cleaned_up_time = normalize(ISTFT(cleaned_up, noisy[1], fftlen))
    clean_time = normalize(ISTFT(clean[0], clean[1], fftlen))
    mse = mean_squared_error(cleaned_up_time, clean_time)
    print 'mse ', mse
    wavwrite(cleaned_up_time/max(abs(cleaned_up_time)), 'curro/xhat.wav', fs=srate, enc='pcm16')
    wavwrite(clean_time/max(abs(clean_time)), 'curro/x.wav', fs=srate, enc='pcm16')
    plt.figure()
    plt.subplot(411)
    plt.plot(cleaned_up_time[0:fftlen*2])
    plt.plot(clean_time[0:fftlen*2])
    plt.subplot(412)
    plt.semilogy(lmse)
    plt.subplot(413)
    plt.plot(clean[0][0,:])
    plt.subplot(414)
    plt.plot(np.unwrap(clean[1][0,:]))
    plt.savefig('curro/x.svg', format='svg')


if __name__ == "__main__":
    import sys
    niter = int(sys.argv[1])
    paris_main({'niter': niter})
#     # a, x, s, loss, reg, x_hat = autoencoder({})
#     a, x, s, loss, _, x_hat = curro_net({})
#     train_fn = train(a,x,s,loss)
#     loss_mse = theano.function([x, s], loss)
#     # loss_reg = theano.function([], reg)
#     lmse = []
#     # lreg = []
#     predict_fn = theano.function([x,s], x_hat)
#     # clean, noisy = gen_data()
#     # wavwrite(clean[1,:], 'fig/s.wav', fs=srate, enc='pcm16')
#     for i in xrange(niter):
#         clean, noisy, _, labels = gen_freq_data(sample=False, gen_data_fn=gen_batch_half_noisy_half_noise)
#         loss = train_fn(noisy, labels)
#         lmse.append(loss)
#         # lmse.append(loss_mse(noisy, clean))
#         # lreg.append(loss_reg())
#         print i, loss
#     clean, noisy, n, labels = gen_batch_half_noisy_half_noise(sample=True)
#     cleaned_up = predict_fn(noisy, labels)
#     cleaned_up = cleaned_up.reshape(batchsize * framelen)
#     # mse calculation
#     mse = mean_squared_error(cleaned_up, clean.reshape(batchsize * framelen))
#     print 'mse ', mse
#     wavwrite(clean.reshape(batchsize * framelen), 'fig/s.wav', fs=srate, enc='pcm16')
#     wavwrite(noisy.reshape(batchsize * framelen), 'fig/xn.wav', fs=srate, enc='pcm16')
#     wavwrite(cleaned_up, 'fig/x.wav', fs=srate, enc='pcm16')
#     plt.figure()
#     plt.subplot(211)
#     # plt.plot(n, clean.reshape(batchsize * framelen))
#     # plt.plot(n, noisy.reshape(batchsize * framelen))
#     # plt.plot(n, cleaned_up)
#     plt.plot(n[0:framelen*2],clean[0:2,:].reshape(-1))
#     plt.plot(n[0:framelen*2],noisy[0:2,:].reshape(-1))
#     plt.plot(n[0:framelen*2],cleaned_up[0:framelen*2])
#     # plt.plot(n[0:framelen],cleaned_up[0:framelen])
#     plt.subplot(212)
#     plt.plot(lmse)
#     plt.semilogy(lmse)
#     # plt.subplot(313)
#     # plt.plot(lreg)
#     # plt.semilogy(lreg)
#     plt.savefig('fig/x.svg', format='svg')
# 

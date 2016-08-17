from __future__ import division
# different networks (autoencoder, conv autoencoder, recurrent)
# different signals (sine, recording)
# different noises (awgn, crowd)
# different domains (time, freq)
from numpy import complex64
import scipy
import lasagne
import theano
import theano.tensor as T
import numpy as np
from scikits.audiolab import wavwrite
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


dtype = theano.config.floatX
batchsize = 256
# framelen = 441
srate = 16000
pct = 0.5  # overlap
fftlen = 1024
framelen = fftlen
# overlap = int(framelen/2)

# dan-specific
shape = (batchsize,framelen)
latentsize = 2000
background_latents_factor = 0.5
minibatch_noise_only_factor = 0.25
n_noise_only_examples = int(minibatch_noise_only_factor * batchsize)
n_background_latents = int(background_latents_factor * latentsize)
lambduh = 0.0

batch_norm = lasagne.layers.batch_norm

def mod_relu(x):
    eps = 1e-5
    return T.switch(x > eps, x, -eps/(x-1-eps))

def normalize(x):
    return x / max(abs(x))

def snr_after(x, x_hat):
    return np.var(x)/np.var(x-x_hat)


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
    network = batch_norm(lasagne.layers.InputLayer(shape, x))
    print network.output_shape
    network = lasagne.layers.DenseLayer(network, latentsize, nonlinearity=mod_relu)
    print network.output_shape
    latents = network
    network = ZeroOutBackgroundLatentsLayer(latents, background_latents_factor=background_latents_factor)
    network = lasagne.layers.DenseLayer(network, shape[1], nonlinearity=lasagne.nonlinearities.identity)
    print network.output_shape

    # loss
    C = np.zeros((batchsize,latentsize))
    C[0:n_noise_only_examples, n_background_latents + 1:] = 1
    C_mat = theano.shared(np.asarray(C, dtype=dtype), borrow=True)
    mean_C = theano.shared(C.mean(), borrow=True)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, x)
    square_term = loss.mean()
    regularization_term = lambduh/mean_C * y * ((C_mat * lasagne.layers.get_output(latents)).mean())**2
    loss = (loss.mean() + regularization_term).mean()

    # training compilation
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params)
    train_fn = theano.function([x,y], loss, updates=updates)

    # inference compilation
    predict_fn = theano.function([x], lasagne.layers.get_output(network, deterministic=True, reconstruct=True))

    #
    # other objectives
    #
    square_term = theano.function([x], square_term)
    regularization_term = theano.function([x,y], regularization_term.mean())

    def do_stuff(network, latents, predict_fn):
        pass

    return network, latents, loss, square_term, regularization_term, train_fn, predict_fn, do_stuff

def dan_main(params):
    network, latents, loss, square_loss, reg_loss, train_fn, predict_fn, do_stuff = dan_net()
    lmse = []
    lsq = []
    lreg = []
    for i in xrange(params.niter):
        clean, noisy, n, labels = gen_freq_data(sample=False, gen_data_fn=gen_batch_half_noisy_half_noise)
        # swap 0 and 1 since for dan net, 0 is signal and 1 is background
        labels = np.expand_dims(np.abs(labels-1).astype(dtype)[:,1], axis=1)
        # labels = np.abs(labels-1).astype(dtype)

        loss = train_fn(noisy[0], labels)
        lmse.append(loss)

        loss_lsq = square_loss(noisy[0])
        lsq.append(loss_lsq)

        loss_reg = reg_loss(noisy[0], labels)
        lreg.append(loss_reg)
        print loss, '\t', loss_lsq, '\t', loss_reg


    plt.figure()
    plt.semilogy(lmse)
    plt.savefig('dan/loss.pdf', format='pdf')

    plt.figure()
    plt.semilogy(lsq)
    plt.savefig('dan/lsq.pdf', format='pdf')

    plt.figure()
    plt.semilogy(lreg)
    plt.savefig('dan/lreg.pdf', format='pdf')

    plt.figure()
    plt.semilogy(lmse)
    plt.semilogy(lsq)
    plt.semilogy(lreg)
    plt.legend(['overall loss', 'squared error loss', 'regularization loss'])
    plt.savefig('dan/losses.svg', format='svg')


def paris_net(params):
    shape = (batchsize, fftlen)
    x = T.matrix('x')  # dirty
    s = T.matrix('s')  # clean
    in_layer = batch_norm(lasagne.layers.InputLayer(shape, x))
    h1 = batch_norm(lasagne.layers.DenseLayer(in_layer, 2000, nonlinearity=mod_relu))
    h1 = lasagne.layers.DenseLayer(h1, fftlen, nonlinearity=lasagne.nonlinearities.identity)

    # loss function
    prediction = lasagne.layers.get_output(h1)
    loss = lasagne.objectives.squared_error(prediction, s)
    return h1, x, s, loss.mean(), None, prediction

def curro_net(params):
    # input
    shape = (batchsize, framelen)
    x = T.matrix('x')  # dirty input
    label = T.matrix('label')  # noise OR signal/noise

    nonlin = mod_relu

    # network
    # in_layer = batch_norm(lasagne.layers.InputLayer(shape, x))  # batch norm or no?
    in_layer = lasagne.layers.InputLayer(shape, x) # batch norm or no?
    layersizes = 1024*2
    h1 = lasagne.layers.DenseLayer(in_layer, layersizes, nonlinearity=nonlin)
    h2 = lasagne.layers.DenseLayer(h1, layersizes, nonlinearity=nonlin)
    h3 = lasagne.layers.DenseLayer(h2, layersizes, nonlinearity=nonlin)
    f = h3  # at this point, first half is signal, second half is noise

    # signal split
    f_sig = lasagne.layers.SliceLayer(f, indices=slice(0,int(layersizes/2)), axis=-1)
    print 'sig split size: ', lasagne.layers.get_output_shape(f_sig)
    sig_d3 = lasagne.layers.DenseLayer(f_sig, framelen, nonlinearity=nonlin)
    # save parameters for noise split
    d3_W = sig_d3.W
    d3_b = sig_d3.b
    sig_d2 = lasagne.layers.DenseLayer(sig_d3, framelen, nonlinearity=nonlin)
    d2_W = sig_d2.W
    d2_b = sig_d2.b
    g_sig = lasagne.layers.DenseLayer(sig_d2, framelen, nonlinearity=lasagne.nonlinearities.identity)
    gs_W = g_sig.W
    gs_b = g_sig.b

    f_noi = lasagne.layers.SliceLayer(f, indices=slice(int(layersizes/2),layersizes), axis=-1)
    print 'noisy split size: ', lasagne.layers.get_output_shape(f_noi)
    noi_d3 = lasagne.layers.DenseLayer(f_noi, framelen, W=d3_W, b=d3_b, nonlinearity=nonlin)
    noi_d2 = lasagne.layers.DenseLayer(noi_d3, framelen, W=d2_W, b=d2_b, nonlinearity=nonlin)
    g_noi = lasagne.layers.DenseLayer(noi_d2, framelen, W=gs_W, b=gs_b, nonlinearity=lasagne.nonlinearities.identity)

    out_layer = lasagne.layers.ElemwiseSumLayer([g_sig,g_noi])

    prediction_sig = lasagne.layers.get_output(g_sig)
    prediction_noi = lasagne.layers.get_output(g_noi)
    # label is 1 for signal, 0 for noise
    prediction = label * prediction_sig + prediction_noi
    loss = lasagne.objectives.squared_error(prediction, x)
    loss_sig = lasagne.objectives.squared_error(prediction_sig, x)
    loss_noi = lasagne.objectives.squared_error(prediction_noi, x)

    return out_layer, g_sig, x, label, loss.mean(), g_noi, prediction, loss_sig, loss_noi

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
    print params
    #updates = lasagne.updates.adam(loss, params)
    updates = lasagne.updates.adamax(loss, params)
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
    clean = _sin_f(amp,441,srate,n,phase) +  _sin_f(amp, 635.25,srate,n,phase) + _sin_f(amp,528,srate,n,phase) + _sin_f(amp,880,srate,n,phase)

    # corrupt with gaussian noise
    noise = np.random.normal(0, 1e-1, clean.shape)
    noisy = clean + noise

    if sample:
        noisy = np.array([noisy[i:i+framelen] for i in xrange(0, len(noisy), int(pct*framelen))][0:batchsize])
        clean = np.array([clean[i:i+framelen] for i in xrange(0, len(clean), int(pct*framelen))][0:batchsize])
        #noisy = noisy.reshape(batchsize, framelen)
        #clean = clean.reshape(batchsize, framelen)

    return clean.astype(dtype), noisy.astype(dtype), n, None

def gen_batch_half_noisy_half_noise(sample=False):
    nop = minibatch_noise_only_factor  # noise only percentage of minibatch
    snr = 100  # dB
    f = 440
    if sample:
        n = np.linspace(0, batchsize * framelen - 1, batchsize * framelen)
        phase = np.random.uniform(0.0, 2*np.pi)
        amp = np.random.uniform(0.35, 0.6)
        clean = amp * np.sin(2 * np.pi * f / srate * n + phase)
    else:
        n = np.tile(np.linspace(0, framelen-1, framelen), (batchsize,1))
        phase = np.tile(np.random.uniform(0.0, 2*np.pi, batchsize), (framelen, 1)).transpose()
        amp = np.tile(np.random.uniform(0.35, 0.65, batchsize), (framelen,1)).transpose()
        clean = amp * np.sin(2 * np.pi * f / srate * n + phase)
        clean[0:int(batchsize*nop),:] = 0

    def _noise_var(clean, snr_db):
        # we use one noise variance per minibatch
        avg_energy = np.sum(clean*clean)/clean.size
        snr_lin = 10**(snr_db/10)
        noise_var = avg_energy / snr_lin
        # print 'noise variance for minibatch: ', noise_var
        return noise_var

    # corrupt with gaussian noise
    if not sample:
        noise_var = _noise_var(clean[int(batchsize*nop):,:], snr)
    else:
        noise_var = _noise_var(clean[int(batchsize*nop):], snr)
    noise = np.random.normal(0, noise_var, clean.shape)
    noisy = clean + noise

    if sample:
        noisy = np.array([noisy[i:i+framelen] for i in xrange(0, len(noisy), int(pct*framelen))][0:batchsize])
        clean = np.array([clean[i:i+framelen] for i in xrange(0, len(clean), int(pct*framelen))][0:batchsize])

    if not sample:
        labels = np.ones((batchsize,1))
        labels[0:int(batchsize*nop)]=0
        # labels = np.zeros((batchsize,1))
        # labels[0:int(batchsize*nop)]=1
    else:
        # assuming "noisy" example for sample, not noise example
        labels = np.ones((batchsize,1))
        # labels = np.zeros((batchsize,1))
    labels = np.tile(labels, (1,framelen))

    return clean.astype(dtype), noisy.astype(dtype), n, labels.astype(dtype)

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
    for i in xrange(params.niter):
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
    wavwrite(normalize(cleaned_up_time), 'paris/xhat.wav', fs=srate, enc='pcm16')
    wavwrite(normalize(clean_time), 'paris/x.wav', fs=srate, enc='pcm16')
    noisy_time = normalize(ISTFT(noisy[0], noisy[1], fftlen))
    wavwrite(normalize(noisy_time), 'paris/n.wav', fs=srate, enc='pcm16')
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
    g_sig, g_sig_for_real, x, s, loss, g_noi_for_real, x_hat, loss_sig, loss_noi = curro_net({})
    train_fn = train(g_sig,x,s,loss)
    train_sig = theano.function([x], loss_sig.mean())
    train_noi = theano.function([x], loss_noi.mean())
    lmse = []
    lsig = []
    lnoi = []
    predict_fn = theano.function([x], lasagne.layers.get_output(g_sig_for_real, deterministic=True))
    predict_fn_noi = theano.function([x], lasagne.layers.get_output(g_noi_for_real, deterministic=True))
    both = theano.function([x], lasagne.layers.get_output(g_sig, deterministic=True))
    for i in xrange(params.niter):
        clean, noisy, n, labels = gen_freq_data(sample=False, gen_data_fn=gen_batch_half_noisy_half_noise)
        loss = train_fn(noisy[0], labels)
        lmse.append(loss)

        loss1 = train_sig(noisy[0])
        lsig.append(loss1)

        loss2 = train_noi(noisy[0])
        lnoi.append(loss2)

        print i, loss, loss1, loss2
    clean, noisy, n, labels = gen_freq_data(sample=True, gen_data_fn=gen_batch_half_noisy_half_noise)

    cleaned_up = predict_fn(noisy[0])
    noisy_reconstructed = predict_fn_noi(noisy[0])
    both_ffts = both(noisy[0])

    cleaned_up_time = normalize(ISTFT(cleaned_up, noisy[1], fftlen))
    clean_time = normalize(ISTFT(clean[0], clean[1], fftlen))
    noisy_reconstructed = normalize(ISTFT(noisy_reconstructed, noisy[1], fftlen))
    both_time = normalize(ISTFT(both_ffts, noisy[1], fftlen))

    mse = mean_squared_error(cleaned_up_time, clean_time)
    mse_noi = mean_squared_error(noisy_reconstructed, clean_time)
    mse_both = mean_squared_error(both_time, clean_time)
    #print 'baseline mse', mean_squared_error()  TODO: mse
    print 'mse ', mse
    print 'mse of noisy half ', mse_noi
    print 'mse of combined (both) ', mse_both
    wavwrite(normalize(cleaned_up_time), 'curro/xhat.wav', fs=srate, enc='pcm16')
    wavwrite(normalize(clean_time), 'curro/x.wav', fs=srate, enc='pcm16')
    wavwrite(normalize(noisy_reconstructed), 'curro/nxhat.wav', fs=srate, enc='pcm16')
    wavwrite(normalize(both_time), 'curro/both.wav', fs=srate, enc='pcm16')
    plt.figure()
    plt.subplot(511)
    plt.plot(clean_time[0:fftlen*3])
    plt.plot(cleaned_up_time[0:fftlen*3])
    plt.subplot(512)
    plt.semilogy(lmse)
    plt.subplot(513)
    #plt.plot(cleaned_up[0,:])
    plt.semilogy(np.abs(np.fft.fft(np.blackman(cleaned_up_time.size)*cleaned_up_time)))
    plt.subplot(514)
    plt.plot(np.unwrap(noisy[1][0,:]))
    plt.subplot(515)
    plt.plot(noisy_reconstructed[0:fftlen*3])
    plt.savefig('curro/x.svg', format='svg')
    plt.figure()
    plt.plot(lsig)
    plt.plot(lnoi)
    plt.legend(['sig', 'noi'])
    plt.savefig('curro/split.svg', format='svg')


def sim_():
    # a, x, s, loss, reg, x_hat = autoencoder({})
    a, x, s, loss, _, x_hat = curro_net({})
    train_fn = train(a,x,s,loss)
    loss_mse = theano.function([x, s], loss)
    # loss_reg = theano.function([], reg)
    lmse = []
    # lreg = []
    predict_fn = theano.function([x,s], x_hat)
    # clean, noisy = gen_data()
    # wavwrite(clean[1,:], 'fig/s.wav', fs=srate, enc='pcm16')
    for i in xrange(niter):
        clean, noisy, _, labels = gen_freq_data(sample=False, gen_data_fn=gen_batch_half_noisy_half_noise)
        loss = train_fn(noisy, labels)
        lmse.append(loss)
        # lmse.append(loss_mse(noisy, clean))
        # lreg.append(loss_reg())
        print i, loss
    clean, noisy, n, labels = gen_batch_half_noisy_half_noise(sample=True)
    cleaned_up = predict_fn(noisy, labels)
    cleaned_up = cleaned_up.reshape(batchsize * framelen)
    # mse calculation
    mse = mean_squared_error(cleaned_up, clean.reshape(batchsize * framelen))
    print 'mse ', mse
    wavwrite(clean.reshape(batchsize * framelen), 'fig/s.wav', fs=srate, enc='pcm16')
    wavwrite(noisy.reshape(batchsize * framelen), 'fig/xn.wav', fs=srate, enc='pcm16')
    wavwrite(cleaned_up, 'fig/x.wav', fs=srate, enc='pcm16')
    plt.figure()
    plt.subplot(211)
    # plt.plot(n, clean.reshape(batchsize * framelen))
    # plt.plot(n, noisy.reshape(batchsize * framelen))
    # plt.plot(n, cleaned_up)
    plt.plot(n[0:framelen*2],clean[0:2,:].reshape(-1))
    plt.plot(n[0:framelen*2],noisy[0:2,:].reshape(-1))
    plt.plot(n[0:framelen*2],cleaned_up[0:framelen*2])
    # plt.plot(n[0:framelen],cleaned_up[0:framelen])
    plt.subplot(212)
    plt.plot(lmse)
    plt.semilogy(lmse)
    # plt.subplot(313)
    # plt.plot(lreg)
    # plt.semilogy(lreg)
    plt.savefig('fig/x.svg', format='svg')


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('net', type=str, help='super, paris, dan, or curro', default='super')
    parser.add_argument('-n', '--niter', type=int, help='number of iterations', default=100)
    args = parser.parse_args()
    mapping = {
        'super': autoencoder,
        'paris': paris_main,
        'dan': dan_main,
        'curro': curro_main,
    }
    mapping[args.net](args)

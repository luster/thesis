import os
from os.path import join
import time
import numpy as np
from datetime import datetime
import pytz
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import batch_norm
from scikits.audiolab import wavwrite
from sklearn.metrics import mean_squared_error

dtype = theano.config.floatX

from cfg import *
from dataset import load_soundfiles, build_dataset_one_signal_frame
from util import ISTFT, normalize
n_noise_only_examples = int(percent_noise_only_examples * examples_per_minibatch)

get_output = lasagne.layers.get_output
get_all_params = lasagne.layers.get_all_params


class ZeroOutBackgroundLatentsLayer(lasagne.layers.Layer):

    def __init__(self, incoming, **kwargs):
        super(ZeroOutBackgroundLatentsLayer, self).__init__(incoming)
        percent_background_latents = kwargs.get('percent_background_latents')
        sh = list(incoming.output_shape)
        sh[0] = 1
        mask = np.ones(sh)
        n = int(percent_background_latents * mask.shape[1])
        mask[:, 0:n, :, :] = 0
        self.mask = theano.shared(mask, borrow=True)
        self.n_background_latents = n
        print self.output_shape

    def get_output_for(self, input_data, reconstruct=False, **kwargs):
        if reconstruct:
            return self.mask * input_data
        return input_data

    @property
    def n(self):
        return self.n_background_latents

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        self._C = C

    @property
    def mean_C(self):
        return self._mean_C

    @mean_C.setter
    def mean_C(self, mean_C):
        self._mean_C = mean_C


def conv2d(incoming, numfilters, shape, stride=(1, 1,)):
    out = lasagne.layers.Conv2DLayer(incoming, numfilters, shape, stride=stride, pad=0, W=lasagne.init.GlorotUniform(
    ), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)
    out = batch_norm(out)
    print out.output_shape
    return out


def deconv2d(incoming, numfilters, shape, stride=(1, 1)):
    out = lasagne.layers.TransposedConv2DLayer(incoming, numfilters, shape,
        stride=stride, crop=0, untie_biases=False,
        W=lasagne.init.GlorotUniform(),
        b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.rectify)
    out = batch_norm(out)
    print out.output_shape
    return out


def build_network(X, shape, percent_background_latents):
    sh = list(shape)
    sh[0] = None
    inlayer = batch_norm(lasagne.layers.InputLayer(sh, X))
    print inlayer.output_shape
    h0 = conv2d(inlayer, 16, (8, 1), (1, 1))
    h1 = conv2d(h0, 16, (8, 1), (2, 1))
    h2 = conv2d(h1, 32, (1, 8), (1, 1))
    h3 = conv2d(h2, 32, (1, 8), (1, 2))
    h4 = conv2d(h3, 64, (8, 1), (2, 1))
    h5 = conv2d(h4, 64, (1, 8), (1, 2))
    print 'latents'
    latents = ZeroOutBackgroundLatentsLayer(
        h5, percent_background_latents=percent_background_latents,
        examples_per_minibatch=examples_per_minibatch)
    print 'back up'
    d4 = deconv2d(latents, 64, (1, 9), (1, 2))
    d3 = deconv2d(d4, 32, (9, 1), (2, 1))
    d2 = deconv2d(d3, 32, (1, 9), (1, 2))
    d1 = deconv2d(d2, 16, (1, 8), (1, 1))
    d0 = deconv2d(d1, 16, (8, 1), (2, 1))
    network = deconv2d(d0, 2, (9, 1), (1, 1))
    return network, latents


def make_c_matrix(latents, n_noise_only_examples, minibatches):
    sizeof_c = list(lasagne.layers.get_output_shape(latents))
    sizeof_c[0] = minibatches
    C = np.zeros(sizeof_c)
    C[0:n_noise_only_examples, :, latents.n+1:, :] = 1
    C_mat = theano.shared(np.asarray(C, dtype=dtype), borrow=True)
    mean_C = theano.shared(C.mean(), borrow=True)
    return C_mat, mean_C


def loss_func(X, y, network, latents, C, mean_C, lambduh=0.75):
    prediction = get_output(network)
    loss = lasagne.objectives.squared_error(prediction, X)
    regularization_term = y * ((C * get_output(latents)).mean())**2
    loss = (loss.mean() + lambduh/mean_C * regularization_term).mean()
    return loss


def pretrain_fn(X, y, network, loss):
    params = get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params)
    pretrain_fn = theano.function([X, y], loss, updates=updates)
    return pretrain_fn


def get_sample_data(signal, noise, framelength, k, minibatches, examples_per_minibatch, freq_bins, time_bins, n_noise_only_examples):
    dataset = build_dataset_one_signal_frame(
        signal, noise,
        framelength, k,
        minibatches, examples_per_minibatch, freq_bins, time_bins,
        n_noise_only_examples)
    idx = 0
    signal = np.array(dataset['training_data'][:, idx:idx+time_bins], dtype)
    start = int(framelength/2 * (idx + 1) - framelength/2)
    end = int(start + framelength/2 * (time_bins + 1))
    noisy = ISTFT(dataset['noise_real'], dataset['noise_imag'])
    clean = ISTFT(dataset['clean_real'], dataset['clean_imag'])
    Scc = normalize(clean, dataset['clean_time_signal'])[start:end]
    baseline_mse = mean_squared_error(dataset['clean_time_signal'][start:end], Scc)
    print 'baseline mse: ', baseline_mse
    sample = np.zeros((1, 2, freq_bins, time_bins))
    sample[:, 0, :, :] = dataset['signal_real'][:, idx:idx+time_bins]
    sample[:, 1, :, :] = dataset['signal_imag'][:, idx:idx+time_bins]
    dataset.update({
        'Scc': Scc,
        'clean': clean,
        'noisy': noisy,
        'sample': sample,
    })
    return dataset


def main(*args, **kwargs):
    X = T.tensor4('X')
    y = T.matrix('y')
    shape = (examples_per_minibatch, 2, freq_bins, time_bins)
    network, latents = build_network(X, shape, percent_background_latents)
    C, mean_C = make_c_matrix(latents, n_noise_only_examples, minibatches)
    loss = loss_func(X, y, network, latents, C, mean_C, lambduh)
    train_fn = pretrain_fn(X, y, network, loss)

    prediction = get_output(network, deterministic=True, reconstruct=True)
    predict_fn = theano.function([X], prediction)

    # load data
    snr = -3
    k = 10. ** (snr/20.)
    x_path = '../data/moonlight_sample.wav'
    n_path = '../data/golf_club_bar_lunch_time.wav'
    signal, noise = load_soundfiles(x_path, n_path)
    niter = 100

    sample_data = get_sample_data(signal, noise,
        framelength, k,
        minibatches, examples_per_minibatch, freq_bins, time_bins,
        n_noise_only_examples)

    p = join('sim', datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d_%H-%M'))
    if not os.path.exists(p):
        os.makedirs(p)
        os.makedirs(join(p, 'wav'))
    wavwrite(sample_data['Scc'], join(p, 'wav/Scc.wav'), fs=fs, enc='pcm16')
    wavwrite(sample_data['noisy'], join(p, 'wav/noisy.wav'), fs=fs, enc='pcm16')


    for i in range(niter):
        dataset = build_dataset_one_signal_frame(
            signal, noise,
            framelength, k,
            minibatches, examples_per_minibatch, freq_bins, time_bins,
            n_noise_only_examples)

        loss = 0
        for batch_idx in range(minibatches):
            ts = time.time()
            l = train_fn(
                dataset['training_data'][batch_idx, :, :, :, :],
                dataset['training_labels'][batch_idx, :, :],
            )
            loss += l
            te = time.time()
            print 'loss: %.3f iter %d/%d/%d/%d took %.3f sec' % (l, batch_idx+1, minibatches, i, niter, te-ts)
        print loss/minibatches

        if i % 5 == 0:
            X_hat = predict_fn(sample_data['sample'])
            x_hat = ISTFT(X_hat[:,0,:,:], X_hat[:,1,:,:])
            mse = mean_squared_error(sample_data['Scc'], x_hat)
            print 'mse: %.3E' % mse
            wavwrite(x_hat, join(p, 'wav/xhat.wav'), fs=fs, enc='pcm16')
            # save model
            # plots



    # create back-prop net

    # train

if __name__ == '__main__':
    main()

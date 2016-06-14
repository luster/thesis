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
n_noise_only_examples = int(
    percent_noise_only_examples * examples_per_minibatch)

get_output = lasagne.layers.get_output
get_all_params = lasagne.layers.get_all_params


class FineTuneLayer(lasagne.layers.Layer):

    def __init__(self, incoming, delta=lasagne.init.Constant(), **kwargs):
        super(FineTuneLayer, self).__init__(incoming)
        self.shape = list(incoming.output_shape)
        self.shape[0] = examples_per_minibatch
        self.delta = self.add_param(delta, self.shape, name='delta', trainable=True, finetune=True)
        print self.output_shape

    def get_output_for(self, input_data, pretrain=True, one=False, **kwargs):
        if pretrain and not one:
            return input_data + 0.0 * self.delta
        elif pretrain and one:
            return input_data + 0.0 * self.delta[0, :, :, :]
        elif not pretrain and not one:
            return input_data + self.delta
        else:
            return input_data + self.delta[0, :, :, :]


def finetune_loss_func(X, latents):
    n = latents.n
    f_x_tilde = get_output(latents, pretrain=False)
    f_xtilde_sig = f_x_tilde[:, n+1:, :, :]
    f_xtilde_noise = f_x_tilde[:, 0:n, :, :]
    f_x_sig = get_output(latents, pretrain=True)[:, n+1:, :, :]
    sig = lasagne.objectives.squared_error(f_xtilde_sig, f_x_sig).mean()
    noise = (f_xtilde_noise**2).mean()
    print 'sig:', sig.eval(), 'noise:', noise.eval()
    return sig + noise


def finetune_train_fn(X, network, loss):
    params = get_all_params(network, trainable=True, finetune=True)
    print 'finetune params', params
    updates = lasagne.updates.adadelta(loss, params)
    train_fn = theano.function([X], loss, updates=updates)
    return train_fn


class ZeroOutBackgroundLatentsLayer(lasagne.layers.Layer):

    def __init__(self, incoming, **kwargs):
        super(ZeroOutBackgroundLatentsLayer, self).__init__(incoming)
        percent_background_latents = kwargs.get('percent_background_latents')
        sh = list(incoming.output_shape)
        sh[0] = 1
        mask = np.ones(sh, dtype)
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
    finetune_layer = FineTuneLayer(inlayer)
    h0 = conv2d(finetune_layer, 16, (8, 1), (1, 1))
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
    return network, latents, finetune_layer


def make_c_matrix(latents, n_noise_only_examples, minibatches):
    sizeof_c = list(lasagne.layers.get_output_shape(latents))
    sizeof_c[0] = minibatches
    C = np.zeros(sizeof_c, dtype)
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
    params = get_all_params(network, trainable=True, finetune=False)
    updates = lasagne.updates.adadelta(loss, params)
    pretrain_fn = theano.function([X, y], loss, updates=updates)
    return pretrain_fn


def get_sample_data(signal, noise, framelength, k, minibatches,
    examples_per_minibatch, freq_bins, time_bins, n_noise_only_examples):

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
    baseline_mse = mean_squared_error(
        dataset['clean_time_signal'][start:end], Scc)
    print 'baseline mse: %.3E' % baseline_mse
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
    network, latents, finetune_layer = build_network(X, shape, percent_background_latents)
    C, mean_C = make_c_matrix(latents, n_noise_only_examples, minibatches)
    loss = loss_func(X, y, network, latents, C, mean_C, lambduh)
    train_fn = pretrain_fn(X, y, network, loss)

    prediction = get_output(network, deterministic=True, reconstruct=True, pretrain=True, one=True)
    predict_fn = theano.function([X], prediction, allow_input_downcast=True)

    # load data
    snr = -1
    k = 10. ** (-snr/10.); print k
    x_path = '../data/moonlight_sample.wav'
    n_path = '../data/golf_club_bar_lunch_time.wav'
    signal, noise = load_soundfiles(x_path, n_path)

    sample_data = get_sample_data(signal, noise,
                                  framelength, k,
                                  minibatches, examples_per_minibatch, freq_bins, time_bins,
                                  n_noise_only_examples)

    p = join('sim', datetime.now(
        pytz.timezone('America/New_York')).strftime('%Y-%m-%d_%H-%M'))
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

        if True:
            X_hat = predict_fn(sample_data['sample'])
            x_hat = ISTFT(X_hat[:, 0, :, :], X_hat[:, 1, :, :])
            mse = mean_squared_error(sample_data['Scc'], x_hat)
            print 'mse: %.3E' % mse
            wavwrite(x_hat, join(p, 'wav/xhat.wav'), fs=fs, enc='pcm16')
            # save model
            # plots

    # create back-prop net
    # finetune_network = build_finetune_network(X, shape, latents)
    finetune_loss = finetune_loss_func(X, latents)
    ft_train_fn = finetune_train_fn(X, latents, finetune_loss)

    finetune_prediction = get_output(finetune_layer, deterministic=True, pretrain=False, one=True)
    finetune_predict_fn = theano.function([X], finetune_prediction, allow_input_downcast=True)

    # train
    for i in range(niter):
        dataset = build_dataset_one_signal_frame(
            signal, noise,
            framelength, k,
            minibatches, examples_per_minibatch, freq_bins, time_bins,
            n_noise_only_examples, signal_only=True)

        loss = 0
        for batch_idx in range(minibatches):
            ts = time.time()
            l = ft_train_fn(
                dataset['training_data'][batch_idx, :, :, :, :]
            )
            loss += l
            te = time.time()
            print 'loss: %.3f iter %d/%d/%d/%d took %.3f sec' % (l, batch_idx+1, minibatches, i, niter, te-ts)
        print loss/minibatches
        print np.mean(finetune_layer.delta.eval())

        if True:
            X_hat = finetune_predict_fn(sample_data['sample'])
            x_hat = ISTFT(X_hat[:, 0, :, :], X_hat[:, 1, :, :])
            mse = mean_squared_error(sample_data['Scc'], x_hat)
            print 'finetune mse: %.3E' % mse
            wavwrite(x_hat, join(p, 'wav/fine_xhat.wav'), fs=fs, enc='pcm16')
            wtf = ISTFT(sample_data['sample'][:,0,:,:], sample_data['sample'][:,1,:,:])
            wavwrite(wtf, join(p, 'wav/wtf.wav'), fs=fs, enc='pcm16')
            # save model
            # plots


if __name__ == '__main__':
    main()

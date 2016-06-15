import lasagne
import numpy as np
import os
import pytz
import theano
import theano.tensor as T
import time

from datetime import datetime
from lasagne.layers import batch_norm
from os.path import join
from scikits.audiolab import wavwrite
from sklearn.metrics import mean_squared_error
from slack import post_slack

from cfg import *
from dataset import build_dataset_one_signal_frame
from dataset import load_soundfiles
from finetune import FineTuneLayer
from finetune import finetune_loss_func
from finetune import finetune_train_fn
from latents import ZeroOutBackgroundLatentsLayer
from util import ISTFT
from util import normalize


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
    finetune_layer = FineTuneLayer(inlayer, delta=lasagne.init.Normal())
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


def make_c_matrix(latents, n_noise_only_examples, examples_per_minibatch):
    sizeof_c = list(lasagne.layers.get_output_shape(latents))
    sizeof_c[0] = examples_per_minibatch
    C = np.zeros(sizeof_c, dtype)
    C[0:n_noise_only_examples, latents.n+1:, :, :] = 1
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

    def _construct_sample(real, imag):
        sample = np.zeros((1,2,freq_bins,time_bins))
        sample[:,0,:,:] = real[:,0:0+time_bins]
        sample[:,1,:,:] = imag[:,0:0+time_bins]
        return sample

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
    sample = _construct_sample(dataset['signal_real'], dataset['signal_imag'])
    only_noise = _construct_sample(dataset['noise_real'], dataset['noise_imag'])
    dataset.update({
        'Scc': Scc,
        'clean': clean,
        'noisy': noisy,
        'sample': sample,
        'only_noise': only_noise,
    })
    return dataset


def main(*args, **kwargs):
    post_slack('starting sim')
    stime = time.time()
    X = T.tensor4('X')
    y = T.matrix('y')
    shape = (examples_per_minibatch, 2, freq_bins, time_bins)
    network, latents, finetune_layer = build_network(X, shape, percent_background_latents)
    C, mean_C = make_c_matrix(latents, n_noise_only_examples, examples_per_minibatch)
    loss = loss_func(X, y, network, latents, C, mean_C, lambduh)
    train_fn = pretrain_fn(X, y, network, loss)

    # X_hat calculation
    prediction = get_output(network, deterministic=True, reconstruct=True, pretrain=True, one=True)
    predict_fn = theano.function([X], prediction, allow_input_downcast=True)

    # latents calculation f(X)
    f_x = get_output(latents, deterministic=True, pretrain=True, one=True)
    f_x = theano.function([X], f_x, allow_input_downcast=True)

    # load data
    snr = -1
    k = 10. ** (-snr/10.)
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
        os.makedirs(join(p, 'npz'))
    wavwrite(sample_data['Scc'], join(p, 'wav/Scc.wav'), fs=fs, enc='pcm16')
    wavwrite(sample_data['noisy'], join(p, 'wav/noisy.wav'), fs=fs, enc='pcm16')
    iter_fname = os.path.join(p, 'graphs.txt')

    #################################################################
    #                           PRETRAIN                            #
    #################################################################
    for i in range(niter_pretrain):
        if i % 100 == 0 and i != 0:
            post_slack('pretrain: iter %d of %d, avg loss @ %.4E, mse @ %.4E' % (i+1,niter_pretrain,loss/minibatches,mse))
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
            print 'loss: %.3f iter %d/%d/%d/%d took %.3f sec' % (l, batch_idx+1, minibatches, i, niter_pretrain, te-ts)
        print loss/minibatches

        if True:
            fx = f_x(sample_data['only_noise'])
            print 'avg noise power: %s, avg signal power: %s' % (
                np.sum(fx[:,0:latents.n,:,:]**2)/latents.n,
                np.sum(fx[:,latens.n+1:,:,:]**2)/(fx.shape[1]-latents.n)
            )
            X_hat = predict_fn(sample_data['sample'])
            x_hat = ISTFT(X_hat[:, 0, :, :], X_hat[:, 1, :, :])
            mse = mean_squared_error(sample_data['Scc'], x_hat)
            print 'mse: %.3E' % mse
            wavwrite(x_hat, join(p, 'wav/xhat.wav'), fs=fs, enc='pcm16')
            # save model
            np.savez(join(p,'npz/pt_network.npz'), *lasagne.layers.get_all_param_values(network))
            np.savez(join(p,'npz/pt_latents.npz'), *lasagne.layers.get_all_param_values(latents))
            np.savez(join(p,'npz/pt_finetune_layer.npz'), *lasagne.layers.get_all_param_values(finetune_layer))
            # plots
            with open(iter_fname, 'a') as f:
                line = '{},{},{},{}\n'.format(i, loss/minibatches, mse, 'pretrain')
                f.write(line)

    # create back-prop net
    # finetune_network = build_finetune_network(X, shape, latents)
    finetune_loss, sig_loss, noise_loss = finetune_loss_func(X, latents)
    # TODO: DO SOMETHING WITH SIG_LOSS AND NOISE_LOSS
    ft_train_fn = finetune_train_fn(X, latents, finetune_loss)

    finetune_prediction = get_output(finetune_layer, deterministic=True, pretrain=False, one=True)
    finetune_predict_fn = theano.function([X], finetune_prediction, allow_input_downcast=True)

    #################################################################
    #                           FINETUNE                            #
    #################################################################
    for i in range(niter_finetune):
        if i % 100 == 0 and i != 0:
            post_slack('finetune: iter %d of %d, avg loss @ %.4E, mse @ %.4E' % (i+1, niter_finetune, loss/minibatches, mse))
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
            print 'loss: %.3f iter %d/%d/%d/%d took %.3f sec' % (l, batch_idx+1, minibatches, i, niter_finetune, te-ts)
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
            with open(iter_fname, 'a') as f:
                line = '{},{},{},{}\n'.format(i, loss/minibatches, mse, 'finetune')
                f.write(line)
            # save model
            np.savez(join(p,'npz/ft_network.npz'), *lasagne.layers.get_all_param_values(network))
            np.savez(join(p,'npz/ft_latents.npz'), *lasagne.layers.get_all_param_values(latents))
            np.savez(join(p,'npz/ft_finetune_layer.npz'), *lasagne.layers.get_all_param_values(finetune_layer))
            # plots

    ttime = time.time()
    post_slack('done with sim, total time: %.3f min' % ((ttime-stime)/60.))


if __name__ == '__main__':
    main()

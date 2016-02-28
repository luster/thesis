# Borrowed from Dan Stowell via https://github.com/danstowell/autoencoder-specgram
#
# Unusual things about this implementation:
#  * Data is not pre-whitened, instead we use a custom layer (NormalisationLayer) to normalise the mean-and-variance of the data for us. This is because I want the spectrogram to be normalised when it is input but not normalised when it is output.
#  * It's a convolutional net but only along the time axis; along the frequency axis it's fully-connected.
from __future__ import division

import scikits.audiolab
import numpy as np
import sys
from datetime import datetime

import lasagne
import theano
import theano.tensor as T

from lasagne.nonlinearities import leaky_rectify
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import very_leaky_rectify
from numpy import complex64
from numpy import float32

try:
    from lasagne.layers import InverseLayer as _
    use_maxpool = True
except ImportError:
    print("""**********************
        WARNING: InverseLayer not found in Lasagne. Please use a more recent version of Lasagne.
        WARNING: We'll deactivate the maxpooling part of the network (since we can't use InverseLayer to undo it)""")
    use_maxpool = False

import matplotlib
matplotlib.use('PDF') # http://www.astrobetter.com/plotting-to-a-file-in-python/
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})

from config import *
from dataset import build_dataset
from dataset import training_data_size
# from plot import plot_probedata
import util

from conv_layer import custom_convlayer
from norm_layer import NormalisationLayer
from util import istft


if use_complex:
    input_var = T.ctensor4('X')
else:
    input_var = T.tensor4('X')
# note that in general, the main data tensors will have these axes:
#   - minibatchsize
#   - numchannels (always 1 for us, since spectrograms)
#   - numfilters (or specbinnum for input)
#   - numtimebins
soft_output_var = T.matrix('y')
idx = T.iscalar()  # index to a [mini]batch


network = lasagne.layers.InputLayer((None, 1, specbinnum, numtimebins), input_var)
# dims: minibatchsize x channels x rows x cols
print "shape after input layer: %s" % str(network.output_shape)
# normalize layer
#  -- note that we deliberately normalise the input but do not undo that at the output.
#  -- note that the normalisation params are not set by the training procedure, they need to be set before training begins.
network = NormalisationLayer(network, specbinnum)
normlayer = network  # need to keep reference to set its parameters
# convolutional layer
network, filters_enc = custom_convlayer(network, in_num_chans=specbinnum, out_num_chans=numfilters)
network = lasagne.layers.NonlinearityLayer(network, nonlinearity=rectify)  # standard rectify since nonnegative target
# maxpool layer
#   NOTE: here we're using max-pooling, along the time axis only, and then
#   using Lasagne's "InverseLayer" to undo the maxpooling in one-hot fashion.
#   There's a side-effect of this: if you use *overlapping* maxpooling windows,
#   the InverseLayer may behave slightly unexpectedly, adding some points with
#   double magnitude. It's OK here since we're not overlapping the windows
if use_maxpool:
    mp_down_factor = maxpooling_downsample_factor
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, mp_down_factor), stride=(1, mp_down_factor))
    maxpool_layer = network  # need to keep reference
# the "middle" of the autoencoder
latents = network  # might want to inspect and/or regularize these too
print lasagne.layers.get_output_shape(latents)


class ZeroOutForegroundLatentsLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(ZeroOutForegroundLatentsLayer, self).__init__(incoming, **kwargs)
        sizeof_C = list(lasagne.layers.get_output_shape(latents))
        mask = np.ones((1, 1, numfilters, numtimebins/mp_down_factor))
        mask[:, :, 0:n_background_latents, :] = 0
        print np.squeeze(mask)
        self.mask = theano.shared(mask, borrow=False)

    def get_output_for(self, input_data, reconstruct=False, **kwargs):
        if reconstruct:
            return self.mask * input_data
        return input_data

network = ZeroOutForegroundLatentsLayer(latents)

# start to unwrap starting here
if use_maxpool:
    network = lasagne.layers.InverseLayer(network, maxpool_layer)
network, filters_dec = custom_convlayer(network, in_num_chans=numfilters, out_num_chans=specbinnum)

# loss function, predictions
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, input_var)
sizeof_C = list(lasagne.layers.get_output_shape(latents))
sizeof_C[0] = minibatch_size
C = np.zeros(sizeof_C)
C[0:n_noise_only_examples, :, n_background_latents+1:, :] = 1
C_mat = theano.shared(np.asarray(C, dtype=dtype), borrow=False)
mean_C = theano.shared(C.mean(), borrow=False)

regularization_term = soft_output_var * ((C_mat * lasagne.layers.get_output(latents)).mean())**2
loss = (loss.mean() + lambduh/mean_C * regularization_term).mean()
if use_complex:
    loss = abs(loss)

# build dataset
training_data, training_labels, noise_specgram, signal_specgram, x_noise, x_signal, noise_phasegram, signal_phasegram = build_dataset(use_stft=use_complex, use_simpler_data=use_simpler_data)

examplegram_startindex = 99
time_startindex = audioframe_len/2 * (examplegram_startindex + 1) - audioframe_len/2
time_endindex = time_startindex + audioframe_len/2 * (numtimebins + 1) + 1


def calculate_time_signal(magnitudegram, phasegram):
    stft = magnitudegram * np.exp(1j * phasegram)
    return istft(np.squeeze(stft), None)


# plot_probedata_data = None
def plot_probedata(gram_name, outpostfix, plottitle=None, compute_time_signal=True):
    """Visualises the network behaviour.
    NOTE: currently accesses globals. Should really be passed in the network, filters etc"""
    # global plot_probedata_data
    if gram_name == 'noise':
        sig = x_noise
        gram = noise_specgram
        phase = noise_phasegram
    elif gram_name == 'signal':
        sig = x_signal
        gram = signal_specgram
        phase = signal_phasegram
    else:
        raise Exception('invalid gram_name, %s' % gram)

    if plottitle==None:
        plottitle = outpostfix

    # if np.shape(plot_probedata_data)==():
    plot_probedata_data = np.array([[gram[:, examplegram_startindex:examplegram_startindex+numtimebins]]], dtype)

    test_prediction = lasagne.layers.get_output(network, deterministic=True, reconstruct=True)
    test_latents = lasagne.layers.get_output(latents, deterministic=True)
    predict_fn = theano.function([input_var], test_prediction)
    latents_fn = theano.function([input_var], test_latents)
    prediction = predict_fn(plot_probedata_data)
    latentsval = latents_fn(plot_probedata_data)

    n_plots = 3
    if compute_time_signal:
        n_plots = 4
        reconstructed_stft = prediction * np.exp(1j*phase[:, examplegram_startindex : examplegram_startindex + numtimebins])
        reconstructed = istft(np.squeeze(reconstructed_stft), sig)
        original_stft = plot_probedata_data * np.exp(1j * phase[:, examplegram_startindex : examplegram_startindex + numtimebins])
        original = istft(np.squeeze(original_stft), sig)
        real_original = sig[time_startindex : time_endindex]

    if False:
        print("Probedata  has shape %s and meanabs %g" % ( plot_probedata_data.shape, np.mean(np.abs(plot_probedata_data ))))
        print("Latents has shape %s and meanabs %g" % (latentsval.shape, np.mean(np.abs(latentsval))))
        print("Prediction has shape %s and meanabs %g" % (prediction.shape, np.mean(np.abs(prediction))))
        print("Ratio %g" % (np.mean(np.abs(prediction)) / np.mean(np.abs(plot_probedata_data))))

    util.mkdir_p('pdf')
    pdf = PdfPages('pdf/%s_autoenc_probe_%s.pdf' % (gram_name, outpostfix))
    plt.figure(frameon=False)
    #
    plt.subplot(n_plots, 1, 1)
    plotdata = plot_probedata_data[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Input')
    plt.title("%s" % (plottitle))
    #
    plt.subplot(n_plots, 1, 2)
    plotdata = latentsval[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Latents')
    #
    plt.subplot(n_plots, 1, 3)
    plotdata = prediction[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Output')
    #
    # ##
    # for filtvar, filtlbl, isenc in [
    #     (filters_enc, 'encoding', True),
    #     (filters_dec, 'decoding', False),
    #         ]:
    #     plt.figure(frameon=False)
    #     vals = filtvar.get_value()
    #     vlim = np.max(np.abs(vals))
    #     for whichfilt in range(numfilters):
    #         plt.subplot(3, 8, whichfilt+1)
    #         # NOTE: for encoding/decoding filters, we grab the "slice" of interest from the tensor in different ways: different axes, and flipped.
    #         if isenc:
    #             plotdata = vals[numfilters - (1 + whichfilt), 0, ::-1, ::-1]
    #         else:
    #             plotdata = vals[:, 0, whichfilt, :]

    #         plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-vlim, vmax=vlim)
    #         plt.xticks([])
    #         if whichfilt == 0:
    #             plt.title("%s filters (%s)" % (filtlbl, outpostfix))
    #         else:
    #             plt.yticks([])
    #     pdf.savefig()
    #     plt.close()

    if compute_time_signal:
        plt.subplot(n_plots, 1, 4)
        plotdata = reconstructed
        # plt.plot(real_original, color='b', label='original signal')  # this signal is too big compared to the normalized ones
        plt.plot(original, color='k', label='original')
        plt.plot(plotdata, color='r', label='reconstructed')
        plt.legend()
        plt.ylabel('Output')
    #
    # plt.close()
    ##
    pdf.savefig()
    plt.close()
    pdf.close()

    # if outpostfix == 'trained' and compute_time_signal:
    if compute_time_signal:
        specgram_ = np.array([[gram[:, examplegram_startindex : examplegram_startindex + numtimebins]]], dtype)
        predicted_gram_ = predict_fn(specgram_)
        phasegram_ = phase[:, examplegram_startindex : examplegram_startindex + numtimebins]

        output_ = calculate_time_signal(predicted_gram_, phasegram_)
        # save to wav
        scikits.audiolab.wavwrite(output_, 'wav/out_%s.wav' % gram_name, fs=srate, enc='pcm16')
    return


if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--')

    cts = True
    plot_probedata('noise', 'init', compute_time_signal=cts)
    plot_probedata('signal', 'init', compute_time_signal=cts)

    # reshape data because of 3rd dim needing to be 1
    training_labels_shared = theano.shared(training_labels.reshape(training_labels.shape[0], training_labels.shape[1], 1), borrow=False)

    training_data_shared = theano.shared(np.asarray(training_data, dtype=dtype), borrow=False)

    # pretrain setup
    normlayer.set_normalisation(training_data)
    indx = theano.shared(0)

    # training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params, learning_rate=1.0, rho=0.95, epsilon=1e-6)
    updates[indx] = indx + 1

    train_fn = theano.function([], loss, updates=updates,
        givens={
            input_var: training_data_shared[indx, :, :, :, :],
            soft_output_var: training_labels_shared[indx, :, :],
        },
        allow_input_downcast=True,
    )

    dt = datetime.now().strftime('%Y%m%d%H%M')
    for epoch in range(numepochs):
        loss = 0
        indx.set_value(0)
        for batch_idx in range(training_data_size):
            loss += train_fn()
        lossreadout = loss / len(training_data)
        infostring = "Epoch %d/%d: Loss %g" % (epoch, numepochs, lossreadout)
        print infostring
        if epoch == 0 or epoch == numepochs - 1 or (2 ** int(np.log2(epoch)) == epoch) or epoch % 50 == 0:
            plot_probedata('noise', 'progress', plottitle="progress (%s)" % infostring, compute_time_signal=cts)
            plot_probedata('signal', 'progress', plottitle="progress (%s)" % infostring, compute_time_signal=cts)
            np.savez('npz/network_%s_epoch%s.npz' % (dt, epoch), *lasagne.layers.get_all_param_values(network))
            np.savez('npz/latents_%s_epoch%s.npz' % (dt, epoch), *lasagne.layers.get_all_param_values(latents))

    plot_probedata('trained', plottitle="trained (%d epochs; Loss %g, )" % (numepochs, lossreadout), compute_time_signal=cts)

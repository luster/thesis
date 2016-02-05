# Borrowed from Dan Stowell via https://github.com/danstowell/autoencoder-specgram
#
# Unusual things about this implementation:
#  * Data is not pre-whitened, instead we use a custom layer (NormalisationLayer) to normalise the mean-and-variance of the data for us. This is because I want the spectrogram to be normalised when it is input but not normalised when it is output.
#  * It's a convolutional net but only along the time axis; along the frequency axis it's fully-connected.

import numpy as np

import lasagne
import theano
import theano.tensor as T
#import downhill
from lasagne.nonlinearities import leaky_rectify
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import very_leaky_rectify
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
#matplotlib.use('PDF') # http://www.astrobetter.com/plotting-to-a-file-in-python/
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})

from config import *
from dataset import build_dataset
# from plot import plot_probedata
from norm_layer import NormalisationLayer
from conv_layer import custom_convlayer
import util

input_var = T.tensor4('X')
# note that in general, the main data tensors will have these axes:
#   - minibatchsize
#   - numchannels (always 1 for us, since spectrograms)
#   - numfilters (or specbinnum for input)
#   - numtimebins

network = lasagne.layers.InputLayer((None, 1, specbinnum, numtimebins), input_var)

# normalize layer
#  -- note that we deliberately normalise the input but do not undo that at the output.
#  -- note that the normalisation params are not set by the training procedure, they need to be set before training begins.
network = NormalisationLayer(network, specbinnum)
normlayer = network  # need to keep reference to set its parameters

# convolutional layer
network, filters_enc = custom_convlayer(network, in_num_chans=specbinnum, out_num_chans=numfilters)

# maxpool layer
#   NOTE: here we're using max-pooling, along the time axis only, and then
#   using Lasagne's "InverseLayer" to undo the maxpooling in one-hot fashion.
#   There's a side-effect of this: if you use *overlapping* maxpooling windows,
#   the InverseLayer may behave slightly unexpectedly, adding some points with
#   double magnitude. It's OK here since we're not overlapping the windows
if use_maxpool:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,2), stride=(1,2))
    maxpool_layer = network  # need to keep reference

# the "middle" of the autoencoder
latents = network  # might want to inspect and/or regularize these too

# start to unwrap starting here
if use_maxpool:
    network = lasagne.layers.InverseLayer(network, maxpool_layer)

network, filters_dec = custom_convlayer(network, in_num_chans=numfilters, out_num_chans=specbinnum)
network = lasagne.layers.NonlinearityLayer(network, nonlinearity=rectify)  # standard rectify since nonnegative target


# loss function, predictions
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, input_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

# build dataset
training_data, training_labels, noise_specgram, signal_specgram = build_dataset()


# TODO: wtf is going on with these relative imports???

examplegram_startindex = 500

plot_probedata_data = None
def plot_probedata(outpostfix, plottitle=None):
    """Visualises the network behaviour.
    NOTE: currently accesses globals. Should really be passed in the network, filters etc"""
    global plot_probedata_data

    if plottitle==None:
        plottitle = outpostfix

    if np.shape(plot_probedata_data)==():
        plot_probedata_data = np.array([[signal_specgram[:, examplegram_startindex:examplegram_startindex+numtimebins]]], float32)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_latents    = lasagne.layers.get_output(latents, deterministic=True)
    predict_fn = theano.function([input_var], test_prediction)
    latents_fn = theano.function([input_var], test_latents)
    prediction = predict_fn(plot_probedata_data)
    latentsval = latents_fn(plot_probedata_data)
    if False:
        print("Probedata  has shape %s and meanabs %g" % ( plot_probedata_data.shape, np.mean(np.abs(plot_probedata_data ))))
        print("Latents has shape %s and meanabs %g" % (latentsval.shape, np.mean(np.abs(latentsval))))
        print("Prediction has shape %s and meanabs %g" % (prediction.shape, np.mean(np.abs(prediction))))
        print("Ratio %g" % (np.mean(np.abs(prediction)) / np.mean(np.abs(plot_probedata_data))))

    util.mkdir_p('pdf')
    pdf = PdfPages('pdf/autoenc_probe_%s.pdf' % outpostfix)
    plt.figure(frameon=False)
    #
    plt.subplot(3, 1, 1)
    plotdata = plot_probedata_data[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Input')
    plt.title("%s" % (plottitle))
    #
    plt.subplot(3, 1, 2)
    plotdata = latentsval[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Latents')
    #
    plt.subplot(3, 1, 3)
    plotdata = prediction[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Output')
    #
    pdf.savefig()
    plt.close()
    ##
    for filtvar, filtlbl, isenc in [
        (filters_enc, 'encoding', True),
        (filters_dec, 'decoding', False),
            ]:
        plt.figure(frameon=False)
        vals = filtvar.get_value()
        #print("        %s filters have shape %s" % (filtlbl, vals.shape))
        vlim = np.max(np.abs(vals))
        for whichfilt in range(numfilters):
            plt.subplot(3, 8, whichfilt+1)
            # NOTE: for encoding/decoding filters, we grab the "slice" of interest from the tensor in different ways: different axes, and flipped.
            if isenc:
                plotdata = vals[numfilters-(1+whichfilt),0,::-1,::-1]
            else:
                plotdata = vals[:,0,whichfilt,:]

            plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-vlim, vmax=vlim)
            plt.xticks([])
            if whichfilt==0:
                plt.title("%s filters (%s)" % (filtlbl, outpostfix))
            else:
                plt.yticks([])

        pdf.savefig()
        plt.close()
    ##
    pdf.close()

plot_probedata('init')



if True:
    # pretrain setup
    normlayer.set_normalisation(training_data)

    # training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    train_fn = theano.function([input_var], loss, updates=updates)

    for epoch in range(numepochs):
        loss = 0
        for input_batch in training_data:
            loss += train_fn(input_batch)
        if epoch == 0 or epoch == numepochs - 1 or (2 ** int(np.log2(epoch)) == epoch):
            lossreadout = loss / len(training_data)
            infostring = "Epoch %d/%d: Loss %g" % (epoch, numepochs, lossreadout)
            print infostring
            plot_probedata('progress', plottitle="progress (%s)" % infostring)

    plot_probedata('trained', plottitle="trained (%d epochs; Loss %g)" % (numepochs, lossreadout))


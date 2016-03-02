"""this file/script helps clean up the current partitioned autoencoder network
so that another can be made to train to correct phase noise
"""
from datetime import datetime
import lasagne
import numpy as np
import theano
import theano.tensor as T

from conv_layer import custom_convlayer
from lasagne.nonlinearities import rectify
from norm_layer import NormalisationLayer
from dataset import build_dataset2

dtype = theano.config.floatX


class ZeroOutBackgroundLatentsLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(ZeroOutBackgroundLatentsLayer, self).__init__(incoming)
        mp_down_factor = kwargs.get('mp_down_factor')
        numfilters = kwargs.get('numfilters')
        numtimebins = kwargs.get('numtimebins')
        n_background_latents = int(kwargs.get('background_latents_factor') * numfilters)
        use_maxpool = kwargs.get('use_maxpool')

        if use_maxpool:
            mask = np.ones((1, 1, numfilters, numtimebins/mp_down_factor))
        else:
            mask = np.ones((1, 1, numfilters, numtimebins))
        mask[:, :, 0:n_background_latents, :] = 0
        print np.squeeze(mask)
        self.mask = theano.shared(mask, borrow=False)

    def get_output_for(self, input_data, reconstruct=False, **kwargs):
        if reconstruct:
            return self.mask * input_data
        return input_data


class PartitionedAutoencoder(object):
    """create a partitioned autoencoder, using the following params:

    specbinnum: number of frequency bins, default is nFFT coefficients
    numtimebins: number of time slices in each spectrogram
    numfilters: number of convolutional filters to use
    use_maxpool: boolean, if you want to maxpool the filter outputs
    mp_down_factor: maxpooling downsample factor, along time axis
    background_latents_factor: percentage of background latents (0-1)

    """
    def __init__(self, minibatch_size=16, specbinnum=128, numtimebins=512,
        numfilters=32, use_maxpool=False, mp_down_factor=16,
        background_latents_factor=0.25, n_noise_only_examples=4):

        self.minibatch_size = minibatch_size
        self.specbinnum = specbinnum
        self.numtimebins = numtimebins
        self.numfilters = numfilters
        self.background_latents_factor = background_latents_factor
        self.n_background_latents = int(background_latents_factor * numfilters)

        # theano variables
        self.input_var = T.tensor4('X')
        self.soft_output_var = T.matrix('y')
        self.idx = T.iscalar()

        # build network
        network = lasagne.layers.InputLayer((None, 1, specbinnum, numtimebins), self.input_var)
        network = NormalisationLayer(network, specbinnum)
        normlayer = network
        network, _ = custom_convlayer(network, in_num_chans=specbinnum, out_num_chans=numfilters)
        network = lasagne.layers.NonlinearityLayer(network, nonlinearity=rectify)
        if use_maxpool:
            mp_down_factor = maxpooling_downsample_factor
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, mp_down_factor), stride=(1, mp_down_factor))
            maxpool_layer = network
        latents = network
        network = ZeroOutBackgroundLatentsLayer(latents,
            mp_down_factor=mp_down_factor,
            numfilters=self.numfilters,
            numtimebins=self.numtimebins,
            background_latents_factor=self.background_latents_factor,
            use_maxpool=use_maxpool)
        if use_maxpool:
            network = lasagne.layers.InverseLayer(network, maxpool_layer)
        network, _ = custom_convlayer(network, in_num_chans=numfilters, out_num_chans=specbinnum)

        # the network
        self.network = network
        self.normlayer = normlayer
        self.latents = latents

        # initialize loss function
        self.loss = self.loss_func(n_noise_only_examples)

    def get_output(self):
        return lasagne.layers.get_output(self.network)

    def loss_func(self, n_noise_only_examples, lambduh=0.75):
        prediction = self.get_output()
        loss = lasagne.objectives.squared_error(prediction, self.input_var)
        sizeof_C = list(lasagne.layers.get_output_shape(self.latents))
        sizeof_C[0] = self.minibatch_size
        C = np.zeros(sizeof_C)
        C[0:n_noise_only_examples, :, self.n_background_latents + 1:, :] = 1
        C_mat = theano.shared(np.asarray(C, dtype=dtype), borrow=False)
        mean_C = theano.shared(C.mean(), borrow=False)

        regularization_term = self.soft_output_var * ((C_mat * lasagne.layers.get_output(self.latents)).mean())**2
        loss = (loss.mean() + lambduh/mean_C * regularization_term).mean()
        return loss

    def train_fn(self, training_data, training_labels, updates='adadelta'):
        training_labels_shared = theano.shared(training_labels.reshape(training_labels.shape[0], training_labels.shape[1], 1), borrow=False)
        training_data_shared = theano.shared(np.asarray(training_data, dtype=dtype), borrow=False)
        self.normlayer.set_normalisation(training_data)

        indx = theano.shared(0)
        update_args = {
            'adadelta': (lasagne.updates.adadelta, {
                'learning_rate': 0.01, 'rho': 0.4, 'epsilon': 1e-6,
            }),
        }[updates]
        update_func, update_params = update_args[0], update_args[1]

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = update_func(self.loss, params, **update_params)
        updates[indx] = indx + 1
        train_fn = theano.function([], self.loss, updates=updates,
            givens={
                self.input_var: training_data_shared[indx, :, :, :, :],
                self.soft_output_var: training_labels_shared[indx, :, :],
            },
            allow_input_downcast=True,
        )
        return indx, train_fn

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=64)
    parser.add_argument('-u', '--updates', type=str, default='adadelta')
    parser.add_argument('-m', '--minibatches', type=int, default=128)
    parser.add_argument('-b', '--minibatchsize', type=int, default=16)
    parser.add_argument('-k', '--snr', type=float, default=0.5)
    parser.add_argument('-t', '--timesignal', type=bool, default=True)
    args = parser.parse_args()

    cts = args.timesignal
    numepochs = args.epochs

    print 'creating partitioned autoencoder'
    pa = PartitionedAutoencoder()

    print 'building dataset'
    training_data, training_labels, noise_specgram, signal_specgram, \
    x_noise, x_signal, noise_phasegram, signal_phasegram = build_dataset2(
        use_stft=False, use_simpler_data=True, k=args.snr, training_data_size=args.minibatches,
        minibatch_size=args.minibatchsize, specbinnum=pa.specbinnum, numtimebins=pa.numtimebins,
        n_noise_only_examples=args.minibatchsize/4)

    print 'initialize train fn'
    indx, train_fn = pa.train_fn(training_data, training_labels, 'adadelta')

    dt = datetime.now().strftime('%Y%m%d%H%M')

    print 'training network'
    for epoch in xrange(numepochs):
        loss = 0
        indx.set_value(0)
        for batch_idx in range(args.minibatches):
            loss += train_fn()
        lossreadout = loss / len(training_data)
        infostring = "Epoch %d/%d: Loss %g" % (epoch, numepochs, lossreadout)
        print infostring
        if epoch == 0 or epoch == numepochs - 1 or (2 ** int(np.log2(epoch)) == epoch) or epoch % 50 == 0:
            plot_probedata('noise', 'progress', plottitle="progress (%s)" % infostring, compute_time_signal=cts)
            plot_probedata('signal', 'progress', plottitle="progress (%s)" % infostring, compute_time_signal=cts)
            np.savez('npz/network_%s_epoch%s.npz' % (dt, epoch), *lasagne.layers.get_all_param_values(self.network))
            np.savez('npz/latents_%s_epoch%s.npz' % (dt, epoch), *lasagne.layers.get_all_param_values(self.latents))

    plot_probedata('noise', 'trained', plottitle="trained (%d epochs; Loss %g, )" % (numepochs, lossreadout), compute_time_signal=cts)
    plot_probedata('signal', 'trained', plottitle="trained (%d epochs; Loss %g, )" % (numepochs, lossreadout), compute_time_signal=cts)
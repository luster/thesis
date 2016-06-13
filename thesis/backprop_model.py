import numpy as np
import theano
import theano.tensor as T
import lasagne

dtype = theano.config.floatX

from cfg import *
n_noise_only_examples = int(percent_noise_only_examples * examples_per_minibatch)

get_output = lasagne.layers.get_output
get_all_params = lasagne.layers.get_all_params


class ZeroOutBackgroundLatentsLayer(lasagne.layers.Layer):

    def __init__(self, incoming, **kwargs):
        super(ZeroOutBackgroundLatentsLayer, self).__init__(incoming)
        percent_background_latents = kwargs.get('percent_background_latents')
        mask = np.ones(incoming.output_shape)
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
    print out.output_shape
    return out


def deconv2d(incoming, numfilters, shape, stride=(1, 1)):
    out = lasagne.layers.TransposedConv2DLayer(incoming, numfilters, shape,
        stride=stride, crop=0, untie_biases=False,
        W=lasagne.init.GlorotUniform(),
        b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.rectify)
    print out.output_shape
    return out


def build_network(X, shape, percent_background_latents):
    inlayer = lasagne.layers.InputLayer(shape, X)
    print inlayer.output_shape
    h0 = conv2d(inlayer, 16, (8, 1), (1, 1))
    h1 = conv2d(h0, 16, (8, 1), (2, 1))
    h2 = conv2d(h1, 32, (1, 8), (1, 1))
    h3 = conv2d(h2, 32, (1, 8), (1, 2))
    h4 = conv2d(h3, 64, (8, 1), (2, 1))
    h5 = conv2d(h4, 64, (1, 8), (1, 2))
    print 'latents'
    latents = ZeroOutBackgroundLatentsLayer(
        h5, percent_background_latents=percent_background_latents)
    print 'back up'
    d4 = deconv2d(latents, 64, (1, 9), (1, 2))
    d3 = deconv2d(d4, 32, (9, 1), (2, 1))
    d2 = deconv2d(d3, 32, (1, 9), (1, 2))
    d1 = deconv2d(d2, 16, (1, 8), (1, 1))
    d0 = deconv2d(d1, 16, (8, 1), (2, 1))
    x_hat = deconv2d(d0, 2, (9, 1), (1, 1))
    return x_hat, latents


def make_c_matrix(latents, n_noise_only_examples, minibatches):
    sizeof_c = list(lasagne.layers.get_output_shape(latents))
    sizeof_c[0] = minibatches
    C = np.zeros(sizeof_c)
    C[0:n_noise_only_examples, :, latents.n+1:, :] = 1
    C_mat = theano.shared(np.asarray(C, dtype=dtype), borrow=True)
    mean_C = theano.shared(C.mean(), borrow=True)
    return C_mat, mean_C


def loss(X, y, network, latents, C, mean_C, lambduh=0.75):
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


def main(*args, **kwargs):
    X = T.tensor4('X')
    y = T.matrix('y')
    shape = (examples_per_minibatch, 2, freq_bins, time_bins)
    x_hat, latents = build_network(X, shape, percent_background_latents)
    C, mean_C = make_c_matrix(latents, n_noise_only_examples, minibatches)
    loss = loss(X, y, x_hat, latents, C, mean_C, lambduh)
    pretrain_fn = pretrain_fn(X, y, x_hat, loss)

    # load data
    snr = -3
    k = 10. ** (snr/20.)
    x_path = '../data/moonlight_sample.wav'
    n_path = '../data/golf_club_bar_lunch_time.wav'
    from dataset import load_soundfiles, build_dataset_one_signal_frame
    signal, noise = load_soundfiles(x_path, n_path)
    dataset = build_dataset_one_signal_frame(
        signal, noise,
        framelength, k,
        minibatches, examples_per_minibatch, freq_bins, time_bins,
        n_noise_only_examples)


    # train

    # create back-prop net

    # train

if __name__ == '__main__':
    main()

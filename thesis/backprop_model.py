import numpy as np
import theano
import theano.tensor as T
import lasagne

# config
minibatches = 16
examples_per_minibatch = 16
freq_bins = 256
time_bins = 256

class ZeroOutBackgroundLatentsLayer(lasagne.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(ZeroOutBackgroundLatentsLayer, self).__init__(incoming)
        percent_background_latents = kwargs.get('percent_background_latents')
        mask = np.ones(incoming.output_shape)
        n = int(percent_background_latents * mask.shape[1])
        mask[:, 0:n, :, :] = 0
        self.mask = theano.shared(mask, borrow=True)
        print self.output_shape

    def get_output_for(self, input_data, reconstruct=False, **kwargs):
        if reconstruct:
            return self.mask * input_data
        return input_data

def conv2d(incoming, numfilters, shape, stride=(1,1,)):
    out = lasagne.layers.Conv2DLayer(incoming, numfilters, shape, stride=stride, pad=0, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)
    print out.output_shape
    return out

def deconv2d(incoming, numfilters, shape, stride=(1,1)):
    out = lasagne.layers.TransposedConv2DLayer(incoming, numfilters, shape, stride=stride, crop=0, untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify)
    print out.output_shape
    return out

def build_network(X, shape):
    inlayer = lasagne.layers.InputLayer(shape, X)
    print inlayer.output_shape
    h0 = conv2d(inlayer, 16, (8,1), (1,1))
    h1 = conv2d(h0, 16, (8,1), (2,1))
    h2 = conv2d(h1, 32, (1,8), (1,1))
    h3 = conv2d(h2, 32, (1,8), (1,2))
    h4 = conv2d(h3, 64, (8,1), (2,1))
    h5 = conv2d(h4, 64, (1,8), (1,2))
    print 'latents'
    latents = ZeroOutBackgroundLatentsLayer(h5, percent_background_latents=0.25)
    print 'back up'
    d4 = deconv2d(latents, 64, (1,9), (1,2))
    d3 = deconv2d(d4, 32, (9,1), (2,1))
    d2 = deconv2d(d3, 32, (1,9), (1,2))
    d1 = deconv2d(d2, 16, (1,8), (1,1))
    d0 = deconv2d(d1, 16, (8,1), (2,1))
    x_hat = deconv2d(d0, 2, (9,1), (1,1))
    return x_hat, latents

# create pre-train net
X = T.tensor4('X')
y = T.matrix('y')
shape = (examples_per_minibatch, 2, freq_bins, time_bins)
x_hat, latents = build_network(X, shape)

# make C matrix, C_mean
pass

# define loss function
def loss(X, y, network, latents, lambduh=0.75):
    prediction = network.get_output()
    loss = lasagne.objectives.squared_error(prediction, X)
    regularization_term = y * ((C * latents.get_output()).mean())**2
    loss = (loss.mean() + lambduh/mean_C * regularization_term).mean()

    return loss

# load data

# train

# create back-prop net

# train
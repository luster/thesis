from __future__ import division

import math
import matplotlib.pyplot as P
import numpy as np
import numpy.matlib
import os
import theano
import theano.tensor as T

from dA import dA
from scipy.io import wavfile
from theano.tensor.shared_randomstreams import RandomStreams


N_obs = 10000
here = os.path.dirname(__file__)
fname = os.path.join(here, 'data', 'sin_440_100msec.wav')
fs, x = wavfile.read(fname)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# normalize to +/- 1
x = x/max(abs(x))

# take only a small portion of the signal - one frame
msec = 50.
n = fs / 1000. * msec
x = x[0:n]
X = np.matlib.repmat(x, N_obs, 1)

# add noise
noise = np.random.normal(0, 0.1, (N_obs, len(x)))
s = X + noise
test_s = x + np.random.normal(0, 0.1, len(x))

# shift to 0-1
def scale(x, a, b):
    return (x-x.min())*(b-a)/(x.max()-x.min()) + a
test_s = scale(test_s, 0., 1.)
s = scale(s, 0., 1.)

# window
# hamming = np.hamming(len(x))
# s = s * hamming

# configure datasets with theano
borrow = True
training_set_x = theano.shared(np.asarray(s, dtype=theano.config.floatX),
                               borrow=borrow)
training_set_y = theano.shared(np.asarray(np.zeros((N_obs, 1)), dtype=theano.config.floatX),
                               borrow=borrow)
training_set_y = T.cast(training_set_y, 'int32')
# test_s = theano.shared(np.asarray(test_s, dtype=theano.config.floatX),
                               # borrow=borrow)

# initialize and train autoencoder
def train_autoencoder(training_set_x):
    learning_rate = 0.1
    training_epochs = 15
    batch_size = 20

    numpy_rng = np.random.RandomState(123)
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(training_set_x.get_value(borrow=True).shape[0] / batch_size)

    da = dA(
        numpy_rng = numpy_rng,
        theano_rng = None,
        input = x,
        n_visible = n,
        n_hidden = 500
    )

    cost, updates = da.get_cost_updates(
        corruption_level = 0.,
        learning_rate = learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: training_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    # train
    for epoch in xrange(training_epochs):
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, np.mean(c)

    return da

da = train_autoencoder(training_set_x)

# test autoencoder
YY = sigmoid(
    np.dot(da.W.get_value(borrow=True).T, test_s) + da.b.get_value(borrow=True)
)
ZZ = sigmoid(
    np.dot(da.W.get_value(borrow=True), YY) + da.b_prime.get_value(borrow=True)
)

test_s = scale(test_s, -1., 1.)
ZZ = scale(ZZ, -1., 1.)

# metric - mse
# s = s/np.hamming(len(s))
baseline_mse = ((test_s - x)**2).mean()
autoencode_mse = ((ZZ - x)**2).mean()

print baseline_mse
print autoencode_mse

P.plot(ZZ)
P.show()

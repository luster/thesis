from __future__ import division

import matplotlib.pyplot as P
import numpy as np
import numpy.matlib
import os

from dA import dA
from scipy.io import wavfile
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.tensor as T

N_obs = 100

here = os.path.dirname(__file__)
fname = os.path.join(here, 'data', 'sin_440_100msec.wav')
fs, x = wavfile.read(fname)

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

# initialize and train autoencoder
def autoencoder(training_set_x):
    learning_rate = 0.1
    training_epochs = 15
    batch_size = 20

    numpy_rng = np.random.RandomState(123)
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')
    # x = T.vector('x')

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

    import pdb; pdb.set_trace()


da = autoencoder(training_set_x)

# test autoencoder


# metric - mse
# s = s/np.hamming(len(s))
# baseline_mse = ((s - x)**2).mean()

# print baseline_mse

# P.plot(s)
# P.show()

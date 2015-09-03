from __future__ import division

import math
import matplotlib.pyplot as P
import numpy as np
import numpy.matlib
import os
import theano
import theano.tensor as T
from utils import *
from train import *

from dA.dA import dA
from scipy.io import wavfile
from theano.tensor.shared_randomstreams import RandomStreams


N = 1000
here = os.path.dirname(__file__)
fname = os.path.join(here, 'data', 'santa_clip.wav')
fs, x = wavfile.read(fname)

x = normalize(x)
x, n = get_first_frame(x, fs)
s = make_observation_matrix(x)
s = scale(s, 0., 1.)

# configure datasets with theano
params = {
    'learning_rate': 0.1,
    'training_epochs': 20,
    'batch_size': 20,
    'n_visible': n,
    'n_hidden': 900,
}
training_set_x = theano.shared(np.asarray(s, dtype=theano.config.floatX), borrow=True)
da = train_autoencoder(training_set_x, **params)

# test autoencoder
test_s = make_test_signal(x)
test_s = scale(test_s, 0., 1.)
test_y = da.get_hidden_values(test_s)
test_z = da.get_reconstructed_input(test_y).eval()

# metric - mse
test_s = scale(test_s, -1., 1.)
test_z = scale(test_z, -1., 1.)
baseline_mse = mse(x, test_s)  #((test_s - x)**2).mean()
autoencode_mse = mse(x, test_z)  #((test_z - x)**2).mean()
print "base mse: %.2E" % baseline_mse
print "mse: %.2E" % autoencode_mse
print "BETTER" if autoencode_mse < baseline_mse else "WORSE"

# plots
fig1 = P.figure()
ax1 = fig1.add_subplot(211)
ax1.plot(test_z)
ax1.plot(x)
ax2 = fig1.add_subplot(212)
ax2.plot(test_s)
ax2.plot(x)

P.show()

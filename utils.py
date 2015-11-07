from __future__ import division

import numpy as np

# generate a sine wave
def gen_signal(t, f, fs):
    return np.sin(2. * np.pi * f / fs * t)

# generate sum of signals
def sum_signals(*args):
    return sum(args)

# sigmoid function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# normalize to +/- 1
def normalize(x):
    # normalize to +/- 1
    return x / max(abs(x))

# scale (usually to 0-1)
def scale(x, a, b):
    return (x - x.min()) * (b - a) / (x.max() - x.min()) + a

# take only a small portion of the signal - one frame
def get_first_frame(x, fs, msec=50.):
    n = fs / 1000. * msec
    return x[0:n], n

def mse(x, y):
    return ((x-y)**2).mean()

# repmat signal and add noise
def make_observation_matrix(x, N=1000, std=0.01):
    L = len(x)
    X = np.matlib.repmat(x, N, 1)
    noise = np.random.normal(0, std, (N, L))
    return X + noise

def make_test_signal(x, std=0.01):
    return x + np.random.normal(0, std, len(x))


class Signal(object):

    def __init__(self, signal, samples=None):
        if samples:
            self.x = np.array(signal)
        self.x = np.array(signal)

    def normalize(self):
        return self.x / max(abs(self.x))

    def scale(self, a, b):
        return (self.x - self.x.min()) * (b - a) / (self.x.max() - self.x.min()) + a

    def mse(self, est):
        return ((self.x - est)**2).mean()


if __name__ == "__main__":
    x = Signal([1,2,3,4,5])
    x.normalize()
    print x.orig
    print x.x

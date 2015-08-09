from __future__ import division

import numpy as np

# signal
# slice
# normalize
# noise
# scale
# mse

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


class Signal(object):

    def __init__(self, signal, samples=None):
        if samples:
            self.x = np.array(signal)[]
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
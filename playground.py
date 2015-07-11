import theano.tensor as T
import lasagne
from math import pi, sin, log, exp
import pylab as P

class TestSignalGenerator(object):
    def __init__(self):
        self.kinds = {
            'sineSweep': self.sine_sweep
        }

    def __call__(self, kind, *args, **kwargs):
        return self.kinds[kind](*args, **kwargs)

    def sine_sweep(self, f_start, f_end, interval, n_steps):
        x = []
        y = []
        b = log(f_end/f_start) / interval
        a = 2 * pi * f_start / b
        for i in range(n_steps):
            delta = i / float(n_steps)
            t = interval * delta
            g_t = a * exp(b * t)
            x.append(t)
            y.append(3 * sin(g_t))
        return x, y

if __name__ == '__main__':
    tsg = TestSignalGenerator()
    kw = {
        'f_start': 20,
        'f_end': 20000,
        'interval': 10,
        'n_steps': 400000
    }
    x, y = tsg('sineSweep', **kw)



    # n_observations = 100
    # obs_len = 50

    # l_in = lasagne.layers.InputLayer((n_observations, obs_len))
    # l_hidden = lasagne.layers.DenseLayer(l_in, num_units=200)
    # l_out = lasagne.layers.DenseLayer(l_hidden, num_units=10,
    #                                   nonlinearity=T.nnet.softmax)


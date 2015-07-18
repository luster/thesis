import lasagne
import matplotlib.pyplot as P
import theano.tensor as T

from math import exp
from math import log
from math import pi
from math import sin


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


class Plotter(object):
    def __init__(self):
        self.ptypes = {
            'basic': self.basic_plot
        }

    def __call__(self, x, y, ptype='basic', *args, **kwargs):
        return self.ptypes[ptype](x, y, *args, **kwargs)

    def basic_plot(self, x, y):
        P.plot(x, y)


if __name__ == '__main__':
    tsg = TestSignalGenerator()
    p = Plotter()
    kw = {
        'f_start': 20,
        'f_end': 20000,
        'interval': 10,
        'n_steps': 441000
    }
    x, y = tsg('sineSweep', **kw)
    p(x, y)
    P.show()

    # n_observations = 100
    # obs_len = 50



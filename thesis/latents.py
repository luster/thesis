import lasagne
import numpy as np
import theano

from cfg import *

class ZeroOutBackgroundLatentsLayer(lasagne.layers.Layer):

    def __init__(self, incoming, **kwargs):
        super(ZeroOutBackgroundLatentsLayer, self).__init__(incoming)
        percent_background_latents = kwargs.get('percent_background_latents')
        sh = list(incoming.output_shape)
        sh[0] = 1
        mask = np.ones(sh, dtype)
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



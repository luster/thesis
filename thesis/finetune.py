import lasagne
from cfg import *


class FineTuneLayer(lasagne.layers.Layer):

    def __init__(self, incoming, delta=lasagne.init.Constant(), **kwargs):
        super(FineTuneLayer, self).__init__(incoming)
        self.shape = list(incoming.output_shape)
        self.shape[0] = examples_per_minibatch
        self.delta = self.add_param(delta, self.shape, name='delta', trainable=True, finetune=True)
        print self.output_shape

    def get_output_for(self, input_data, pretrain=True, one=False, **kwargs):
        if pretrain and not one:
            # bypass finetune params
            return input_data + 0.0 * self.delta
        elif pretrain and one:
            # bypass finetune params, but with 1 example instead of a minibatch
            return input_data + 0.0 * self.delta[0, :, :, :]
        elif not pretrain and not one:
            # use finetune params
            return input_data + self.delta
        else:
            # use finetune params, but with 1 example instead of a minibatch
            return input_data + self.delta[0, :, :, :]


def finetune_loss_func(X, latents, lambduh=8):
    n = latents.n
    f_x_tilde = get_output(latents, pretrain=False)
    f_xtilde_sig = f_x_tilde[:, n+1:, :, :]
    f_xtilde_noise = f_x_tilde[:, 0:n, :, :]
    f_x_sig = get_output(latents, pretrain=True)[:, n+1:, :, :]
    sig = lasagne.objectives.squared_error(f_xtilde_sig, f_x_sig).mean()
    noise = (f_xtilde_noise**2).mean()
    return sig + lambduh * noise, sig, noise


def finetune_train_fn(X, network, loss):
    params = get_all_params(network, trainable=True, finetune=True)
    print 'finetune params', params
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.05, momentum=0.95)
    train_fn = theano.function([X], loss, updates=updates)
    return train_fn

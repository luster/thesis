import theano.tensor as T

n_observations = 100
obs_len = 50

l_in = lasagne.layers.InputLayer((n_observations, obs_len))
l_hidden = lasagne.layers.DenseLayer(l_in, num_units=200)
l_out = lasagne.layers.DenseLayer(l_hidden, num_units=10,
                                  nonlinearity=T.nnet.softmax)


import numpy as np
import theano
import theano.tensor as T
from dA.dA import dA

# initialize and train autoencoder
def train_autoencoder(training_set_x, **kwargs):
    # input parameters
    learning_rate = kwargs.pop('learning_rate')
    training_epochs = kwargs.pop('training_epochs')
    batch_size = kwargs.pop('batch_size')
    n_visible = kwargs.pop('n_visible')
    n_hidden = kwargs.pop('n_hidden')
    corruption_level = kwargs.pop('corruption_level')

    numpy_rng = np.random.RandomState(123)
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(training_set_x.get_value(borrow=True).shape[0] / batch_size)

    da = dA(
        numpy_rng = numpy_rng,
        theano_rng = None,
        input = x,
        n_visible = n_visible,
        n_hidden = n_hidden
    )

    cost, updates = da.get_cost_updates(
        corruption_level = corruption_level,
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

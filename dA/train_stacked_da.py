import numpy as np
import theano
import theano.tensor as T
import timeit
from SdA import SdA
import sys
import os

def train_stacked_da(datasets, **kwargs):
    training_set_x, validate_set_x, test_set_x = datasets

    # input parameters
    pretrain_lr = kwargs.pop('pretrain_lr')
    finetune_lr = kwargs.pop('finetune_lr')
    pretraining_epochs = kwargs.pop('pretraining_epochs')
    training_epochs = kwargs.pop('training_epochs')
    batch_size = kwargs.pop('batch_size')
    n_visible = kwargs.pop('n_visible')
    n_hidden = kwargs.pop('n_hidden')  # list
    corruption_levels = kwargs.pop('corruption_levels')  # list

    numpy_rng = np.random.RandomState(123)
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(training_set_x.get_value(borrow=True).shape[0] / batch_size)

    print '... building the model'

    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_visible,
        hidden_layers_sizes=n_hidden,
        n_outs=n_visible,
        corruption_levels=corruption_levels
    )

    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(
        train_set_x=training_set_x,
        batch_size=batch_size
    )

    print '... pre-training the model'

    start_time = timeit.default_timer()
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](
                    index=batch_index,
                    corruption=corruption_levels[i],
                    lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)
    end_time = timeit.default_timer()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    print '... getting the finetuning functions'
    train_fn = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print '... finetuning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches
    patience_increase = 2.  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative imprv of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go thru this many minibatches b4 checking network on validation set; in this case
    # we check every epoch

    best_validation_loss = np.inf
    test_score = 0.

    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

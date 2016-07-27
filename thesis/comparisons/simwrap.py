#!/usr/bin/env python

from sklearn.metrics import mean_squared_error as mse

DEFAULT_INPUT = None  # TODO: sine wave wrapper - random freq/phase/amp
DEFAULT_NETWORK = None  # TODO: create a 1-2 layer neural network
DEFAULT_NUMBER_ITERATIONS = 100
DEFAULT_METRICS = [mse,]

class SimWrapper(object):
    """Basic wrapper for running a simulation.

    Should this be implemented as an abstract class?

    Simulation Properties:
    - Input
      - 1D/2D? Time/Freq?
      - Signal - generate or load from file?
      - Noise - generate or load from file? SNR?
    - Network --> relative sizes or preset?
      - Network architecture
      - Loss function
      - Other hyperparameters
    - Output from network (defined by network)
      - Boilerplate get_output(network) function
    - Postprocessing steps?
      - Rescale signal?
      - Overlap-add?
    - Training criteria
      - Number of iterations/epochs
    - Evaluation criteria
      - MSE, SNR

    Where to store?
    What kind of metadata set on the simulation?
    
    """

    def __init__(self):
        self._input = DEFAULT_INPUT
        self._network = DEFAULT_NETWORK

        self.niter = DEFAULT_NUMBER_ITERATIONS
        self.metrics = []

    def network(self, net):
        self._network = net
        return self

    def _gen_batch(self):
        # return a minibatch
        pass

    def _train_fn(self, batch):
        # return loss
        pass

    def _run_iter(self, i):
        # generate batch
        batch = self._gen_batch()
        loss = self._train_fn(batch)
        print i+1, loss

    def _run(self):
        for i in xrange(self.niter):
            self._run_iter(i)
        return None

    def run(self):
        return self._run()

def validate_args(args):
    import os
    if not os.path.isfile(args.input):
        if args.input not in ['sin', 'sin440']:
            raise Exception('Invalid input: %s' % args.input)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT)
    parser.add_argument('--network', type=str, default=DEFAULT_NETWORK)
    args = parser.parse_args()
    validate_args(args)
    SimWrapper().network(1).run()

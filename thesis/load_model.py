import numpy as np

"""depending on which model, need to do one of the following:
        1) import the relevant code and build the same network
            a) load the model parameters and then set them
        2) pickle the current network along with the params
"""

import lasagne
from time_domain import TimeDomainOutputPartitionedAutoencoder
from config import specbinnum

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('-m', '--minibatches', type=int, default=96)
    args = parser.parse_args()

    network = TimeDomainOutputPartitionedAutoencoder(num_minibatches=args.minibatches, specbinnum=specbinnum)
    with np.load(args.file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

import numpy as np

"""depending on which model, need to do one of the following:
        1) import the relevant code and build the same network
            a) load the model parameters and then set them
        2) pickle the current network along with the params
"""

import lasagne
import theano

import matplotlib.pyplot as plt

from build_networks import PartitionedAutoencoder
from config import specbinnum
from phase_network import PartitionedAutoencoderForPhase
from util import calculate_time_signal
from util import load_soundfile
from util import standard_specgram


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mag', type=str)
    parser.add_argument('phase', type=str)
    parser.add_argument('-m', '--minibatches', type=int, default=24)
    parser.add_argument('-b', '--minibatchsize', type=int, default=32)
    parser.add_argument('-t', '--timebins', type=int, default=512)
    parser.add_argument('-k', '--snr', type=float, default=0.5)
    args = parser.parse_args()

    pa_mag = PartitionedAutoencoder(num_minibatches=args.minibatches,
        minibatch_size=args.minibatchsize,
        specbinnum=specbinnum,
        numtimebins=args.timebins,
        numfilters=128,
        use_maxpool=False,
        mp_down_factor=16,
        background_latents_factor=0.25,
        n_noise_only_examples=int(0.25*args.minibatchsize))
    pa_phase = PartitionedAutoencoderForPhase(num_minibatches=args.minibatches,
        minibatch_size=args.minibatchsize,
        specbinnum=specbinnum,
        numtimebins=args.timebins,
        numfilters=128,
        use_maxpool=False,
        mp_down_factor=16,
        background_latents_factor=0.25,
        n_noise_only_examples=int(0.25*args.minibatchsize))
    with np.load(args.mag) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(pa_mag.network, param_values)
    with np.load(args.phase) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(pa_phase.network, param_values)

    test_prediction_mag = lasagne.layers.get_output(pa_mag.network, deterministic=True, reconstruct=True)
    test_prediction_phase = lasagne.layers.get_output(pa_phase.network, deterministic=True, reconstruct=True)
    predict_fn_mag = theano.function([pa_mag.input_var], test_prediction_mag)
    predict_fn_phase = theano.function([pa_phase.input_var], test_prediction_phase)

    # create test signal or load dataset for evaluation
    k = 10 ** (-np.array(args.snr) / 20.)
    fs = 44100
    ls = 5*fs-1
    n = np.linspace(0, ls, ls+1)
    signal = np.sin(2*np.pi*440./fs*n)
    noise = load_soundfile('../data/chon/noise_44.wav', 0, 5)[0:ls+1]
    signal = signal/sum(signal**2)
    noise = noise/sum(noise**2)
    x = signal + k * noise

    s_mag, s_phase = standard_specgram(signal)
    s_mag = s_mag[:, 0:512].reshape(1,1,256,512)
    s_phase = s_phase[:, 0:512].reshape(1,1,256,512)
    x_mag, x_phase = standard_specgram(x)
    x_mag = x_mag[:, 0:512].reshape(1,1,256,512)
    x_phase = x_phase[:, 0:512].reshape(1,1,256,512)

    Xmag_hat = predict_fn_mag(x_mag)
    Xphase_hat = predict_fn_phase(x_phase)

    # reconstructions
    Xdd = calculate_time_signal(Xmag_hat, Xphase_hat)
    Xdc = calculate_time_signal(Xmag_hat, s_phase)
    Xcc = calculate_time_signal(s_mag, s_phase)
    Xdx = calculate_time_signal(Xmag_hat, x_phase)

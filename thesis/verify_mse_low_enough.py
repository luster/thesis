#!/usr/bin/env python

import numpy as np
import scikits.audiolab
from util import load_soundfile
from sklearn.metrics import mean_squared_error


mse_noisy = 0.026509613988
mse_sdc = 0.00439247718653
mse_scd = 0.0192693287771
mse_sdd = 0.0164271486726



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--signal', type=str, default='../data/chon/signal_44.wav')
    args = parser.parse_args()

    signal = load_soundfile(args.signal, 0, 10)
    noise = np.random.normal(0, .07, signal.shape)
    y = signal + noise
    mse = mean_squared_error(signal, y)
    print mse

    scikits.audiolab.wavwrite(signal, 'chon_sig.wav', fs=44100, enc='pcm16')
    scikits.audiolab.wavwrite(y, 'chon_noisy.wav', fs=44100, enc='pcm16')

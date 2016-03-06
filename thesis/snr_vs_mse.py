from __future__ import division

import scikits.audiolab
import numpy as np
import sys
from datetime import datetime

import lasagne
import theano
import theano.tensor as T

import matplotlib
# http://www.astrobetter.com/plotting-to-a-file-in-python/
matplotlib.use('PDF')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})

from config import audioframe_len
from dataset import build_dataset2
from build_networks import dtype, PartitionedAutoencoder
from util import calculate_time_signal
from sklearn.metrics import mean_squared_error


def normalize(this, against):
    return this / max(abs(this)) * max(abs(against))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=64)
    parser.add_argument('-u', '--updates', type=str, default='adadelta')
    parser.add_argument('-m', '--minibatches', type=int, default=128)
    parser.add_argument('-b', '--minibatchsize', type=int, default=16)
    parser.add_argument('-k', '--snr', type=float, nargs='+', default=[-6,-3,0,3,6,9,12])
    parser.add_argument('-t', '--timesignal', type=bool, default=True)
    args = parser.parse_args()

    cts = args.timesignal
    numepochs = args.epochs
    k_values = 10 ** (-np.array(args.snr) / 20.)

    mse_cc = []
    mse_dc = []
    mse_cd = []
    mse_dd = []
    for k in k_values:
        print 'k = ', k
        # create network(s)
        pa_mag = PartitionedAutoencoder()
        pa_phase = PartitionedAutoencoder()

        # make dataset
        dataset = build_dataset2(use_stft=False, use_simpler_data=True,
            k=k, training_data_size=args.minibatches,
            minibatch_size=args.minibatchsize, specbinnum=pa_mag.specbinnum,
            numtimebins=pa_mag.numtimebins,
            n_noise_only_examples=int(args.minibatchsize / 4))

        # vars from dataset
        training_labels = dataset['training_labels']
        data_len = len(training_labels)

        # clean signal (baseline)
        idx = 105
        start = audioframe_len/2 * (idx + 1) - audioframe_len/2
        end = start + audioframe_len/2 * (pa_mag.numtimebins + 1)  #+ 1
        clean = calculate_time_signal(dataset['clean_magnitude'], dataset['clean_phase'])
        Scc = normalize(clean, dataset['clean_time_signal'])[start:end]
        baseline_mse = mean_squared_error(dataset['clean_time_signal'][start:end], Scc)
        print 'baseline mse: ', baseline_mse
        mse_cc.append(baseline_mse)

        # normalize/get train functions
        indx_mag, train_fn_mag = pa_mag.train_fn(dataset['training_data_magnitude'], training_labels, 'adadelta')
        indx_phase, train_fn_phase = pa_phase.train_fn(dataset['training_data_phase'], training_labels, 'adadelta')

        # reconstruction functions
        test_prediction_mag = lasagne.layers.get_output(pa_mag.network, deterministic=True, reconstruct=True)
        test_prediction_phase = lasagne.layers.get_output(pa_phase.network, deterministic=True, reconstruct=True)
        test_latents_mag = lasagne.layers.get_output(pa_mag.latents, deterministic=True)
        test_latents_phase = lasagne.layers.get_output(pa_phase.latents, deterministic=True)
        predict_fn_mag = theano.function([pa_mag.input_var], test_prediction_mag)
        predict_fn_phase = theano.function([pa_phase.input_var], test_prediction_phase)
        latents_fn_mag = theano.function([pa_mag.input_var], test_latents_mag)
        latents_fn_phase = theano.function([pa_phase.input_var], test_latents_phase)

        # data sample for reconstruction
        sample_mag = np.array([[dataset['signal_magnitude'][:, idx:idx+pa_mag.numtimebins]]], dtype)
        sample_phase = np.array([[dataset['signal_phase'][:, idx:idx+pa_mag.numtimebins]]], dtype)

        # train network(s)
        for epoch in xrange(numepochs):
            loss_mag = 0
            loss_phase = 0
            indx_mag.set_value(0)
            indx_phase.set_value(0)
            for batch_idx in range(args.minibatches):
                loss_mag += train_fn_mag()
                loss_phase += train_fn_phase()
            lossreadout_mag = loss_mag / data_len
            lossreadout_phase = loss_phase / data_len
            infostring = "Epoch %d/%d: mag Loss %g, phase loss %g" % (epoch, numepochs, lossreadout_mag, lossreadout_phase)
            print infostring
            if epoch == 0 or epoch == numepochs - 1 or (2 ** int(np.log2(epoch)) == epoch) or epoch % 50 == 0:
                """generate 4 time signals using networks:
                        Sdc: denoised mag, clean phase
                        Scd: clean mag, denoised phase
                        Sdd: denoised mag, denoised phase
                    using these signals, compute MSE with respect to baseline
                """
                prediction_mag = predict_fn_mag(sample_mag)
                prediction_phase = predict_fn_phase(sample_phase)
                Sdc = normalize(calculate_time_signal(prediction_mag, dataset['clean_phase'][:, idx:idx+pa_mag.numtimebins]), Scc)
                Scd = normalize(calculate_time_signal(dataset['clean_magnitude'][:, idx:idx+pa_mag.numtimebins], prediction_phase), Scc)
                Sdd = normalize(calculate_time_signal(prediction_mag, prediction_phase), Scc)
                print '\tMSE Sdc: ', mean_squared_error(Scc, Sdc)
                print '\tMSE Scd: ', mean_squared_error(Scc, Scd)
                print '\tMSE Sdd: ', mean_squared_error(Scc, Sdd)

                latentsval_phase = latents_fn_phase(sample_phase)
                latentsval_mag = latents_fn_mag(sample_mag)
        Sdc = normalize(calculate_time_signal(prediction_mag, dataset['clean_phase'][:, idx:idx+pa_mag.numtimebins]), Scc)
        Scd = normalize(calculate_time_signal(dataset['clean_magnitude'][:, idx:idx+pa_mag.numtimebins], prediction_phase), Scc)
        Sdd = normalize(calculate_time_signal(prediction_mag, prediction_phase), Scc)
        mse_dc.append(mean_squared_error(Scc, Sdc))
        mse_cd.append(mean_squared_error(Scc, Scd))
        mse_dd.append(mean_squared_error(Scc, Sdd))
    print mse_cc
    print mse_dc
    print mse_cd
    print mse_dd

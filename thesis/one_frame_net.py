from __future__ import division

from datetime import datetime
import pytz
from pprint import pprint

import scikits.audiolab
import numpy as np
import os
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

from config import audioframe_len, specbinnum, srate
from dataset import build_dataset_one_signal_frame, load_soundfiles
from build_networks import dtype, PartitionedAutoencoder
from phase_network import PartitionedAutoencoderForPhase
from util import calculate_time_signal
from sklearn.metrics import mean_squared_error


def normalize(this, against):
    return this / max(abs(this)) * max(abs(against))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--niter', type=int, default=1024)
    parser.add_argument('-e', '--epochs', type=int, default=16)
    parser.add_argument('-m', '--minibatches', type=int, default=192)
    parser.add_argument('-b', '--minibatchsize', type=int, default=16)
    parser.add_argument('-k', '--snr', type=float, nargs='+', default=[-3,0,3,6,9,12])
    parser.add_argument('-f', '--timebins', type=int, default=512)
    parser.add_argument('-s', '--signal', type=str, default='../data/chon/signal_44.wav')
    parser.add_argument('-n', '--noise', type=str, default='../data/chon/noise_44.wav')
    args = parser.parse_args()

    numepochs = args.epochs
    k_values = 10 ** (-np.array(args.snr) / 20.)
    signal, noise = load_soundfiles(args.signal, args.noise)

    # create network(s)
    pa_mag = PartitionedAutoencoder(num_minibatches=args.minibatches,
        minibatch_size=args.minibatchsize,
        specbinnum=specbinnum,
        numtimebins=args.timebins,
        numfilters=64,
        use_maxpool=False,
        mp_down_factor=16,
        background_latents_factor=0.25,
        n_noise_only_examples=int(0.25*args.minibatchsize))
    pa_phase = PartitionedAutoencoderForPhase(num_minibatches=args.minibatches,
        minibatch_size=args.minibatchsize,
        specbinnum=specbinnum,
        numtimebins=args.timebins,
        numfilters=64,
        use_maxpool=False,
        mp_down_factor=16,
        background_latents_factor=0.25,
        n_noise_only_examples=int(0.25*args.minibatchsize))

    folder = os.path.join('sim', datetime.now(pytz.timezone('America/New_York')).strftime('%Y-%m-%d_%H-%M'))
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, 'config.txt'), 'a') as f:
        f.write('pa_mag.__dict__ : \n')
        pprint(pa_mag.__dict__, stream=f)
        f.write('args : \n')
        pprint(args.__dict__, stream=f)
    print pa_mag.__dict__

    iter_fname = os.path.join(folder, 'graphs.txt')  # iter, loss_mag, loss_phase, Sdc MSE, Sdd MSE

    mse_cc = []  # clean reconstruction
    mse_dc = []  # denoised magnitude, clean phase
    mse_cd = []  # clean magnitude, denoised phase
    mse_dd = []  # denoised magnitude, denoised phase
    mse_noisy = []  # baseline mse, Y = S + N w.r.t. S
    for snr_idx, k in enumerate(k_values):

        mse_dc.append(0)
        mse_cd.append(0)
        mse_dd.append(0)
        mse_noisy.append(0)
        # reinitialize networks
        if snr_idx > 0:
            pa_mag.initialize_network()
            pa_phase.initialize_network()

        print 'SNR = ', args.snr[snr_idx]

        # make dataset
        dataset_ = build_dataset_one_signal_frame(signal, noise, k=k,
            num_minibatches=args.minibatches,
            minibatch_size=args.minibatchsize, specbinnum=pa_mag.specbinnum,
            numtimebins=pa_mag.numtimebins,
            n_noise_only_examples=int(args.minibatchsize / 4))

        # vars from dataset
        training_labels = dataset_['training_labels']
        data_len = len(training_labels)

        # clean signal (baseline)
        idx = 0
        start = int(audioframe_len/2 * (idx + 1) - audioframe_len/2)
        end = int(start + audioframe_len/2 * (pa_mag.numtimebins + 1))  #+ 1
        clean = calculate_time_signal(dataset_['clean_magnitude'], dataset_['clean_phase'])
        Scc = normalize(clean, dataset_['clean_time_signal'])[start:end]
        baseline_mse = mean_squared_error(dataset_['clean_time_signal'][start:end], Scc)
        print 'baseline mse: ', baseline_mse
        mse_cc.append(baseline_mse)

        # noisy time signal
        noisy = dataset_['noisy_time_signal'][start:end]
        noisy = normalize(noisy, Scc)

        # normalize/get train functions
        training_labels = training_labels.reshape(training_labels.shape[0], training_labels.shape[1], 1)

        # reconstruction functions
        test_prediction_mag = lasagne.layers.get_output(pa_mag.network, deterministic=True, reconstruct=True)
        test_prediction_phase = lasagne.layers.get_output(pa_phase.network, deterministic=True, reconstruct=True)
        test_latents_mag = lasagne.layers.get_output(pa_mag.latents, deterministic=True)
        test_latents_phase = lasagne.layers.get_output(pa_phase.latents, deterministic=True)
        predict_fn_mag = theano.function([pa_mag.input_var], test_prediction_mag)
        predict_fn_phase = theano.function([pa_phase.input_var], test_prediction_phase)
        # latents_fn_mag = theano.function([pa_mag.input_var], test_latents_mag)
        # latents_fn_phase = theano.function([pa_phase.input_var], test_latents_phase)

        # data sample for reconstruction
        sample_mag = np.array([[dataset_['signal_magnitude'][:, idx:idx+pa_mag.numtimebins]]], dtype)
        sample_phase = np.array([[dataset_['signal_phase'][:, idx:idx+pa_mag.numtimebins]]], dtype)

        train_fn_mag = pa_mag.train_fn_slim('adam')
        train_fn_phase = pa_phase.train_fn_slim('adam')

        niter = args.niter
        for _iter in range(niter):
            dataset = build_dataset_one_signal_frame(signal, noise, k=k,
                num_minibatches=args.minibatches,
                minibatch_size=args.minibatchsize, specbinnum=pa_mag.specbinnum,
                numtimebins=pa_mag.numtimebins,
                n_noise_only_examples=int(args.minibatchsize / 4))
            # normalize
            pa_mag.normalize_batches(dataset['training_data_magnitude'])
            pa_phase.normalize_batches(dataset['training_data_phase'])

            # train network(s)
            loss_mag = 0
            loss_phase = 0

            for batch_idx in range(args.minibatches):
                to_print = 'SNR = {}, Starting dataset {}/{}, batch {}/{}'.format(args.snr[snr_idx], _iter+1, niter, batch_idx+1, args.minibatches)
                print to_print
                loss_mag += train_fn_mag(dataset['training_data_magnitude'][batch_idx, :, :, :, :],
                    training_labels[batch_idx, :, :])
                loss_phase += train_fn_phase(dataset['training_data_phase'][batch_idx, :, :, :, :],
                    training_labels[batch_idx, :, :])
            lossreadout_mag = loss_mag / data_len
            lossreadout_phase = loss_phase / data_len
            infostring = "mag Loss %g, phase loss %g" % (lossreadout_mag, lossreadout_phase)
            print infostring
            if _iter == 0 or _iter == numepochs - 1 or (2 ** int(np.log2(_iter)) == _iter) or _iter % 50 == 0:
                """generate 4 time signals using networks:
                        Sdc: denoised mag, clean phase
                        Scd: clean mag, denoised phase
                        Sdd: denoised mag, denoised phase
                    using these signals, compute MSE with respect to baseline
                """
                prediction_mag = predict_fn_mag(sample_mag)
                prediction_phase = predict_fn_phase(sample_phase)
                Snoisy = normalize(calculate_time_signal(dataset_['noise_magnitude'][:, idx:idx+pa_mag.numtimebins], dataset_['noise_phase'][:, idx:idx+pa_mag.numtimebins]), Scc)
                Sdc = normalize(calculate_time_signal(prediction_mag, dataset_['clean_phase'][:, idx:idx+pa_mag.numtimebins]), Scc)
                Scd = normalize(calculate_time_signal(dataset_['clean_magnitude'][:, idx:idx+pa_mag.numtimebins], prediction_phase), Scc)
                Sdd = normalize(calculate_time_signal(prediction_mag, prediction_phase), Scc)

                print 'baseline mse: ', baseline_mse
                msesdc = mean_squared_error(Scc, Sdc)
                msesdd = mean_squared_error(Scc, Sdd)
                print '\tMSE noisy: ', mean_squared_error(Scc, Snoisy)
                print '\tMSE Sdc: ', msesdc
                print '\tMSE Scd: ', mean_squared_error(Scc, Scd)
                print '\tMSE Sdd: ', msesdd

                with open(iter_fname, 'a') as f:
                    line = "{},{},{},{},{}\n".format(_iter, lossreadout_mag, lossreadout_phase, msesdc, msesdd)
                    f.write(line)

                # create dir first
                if not os.path.exists(os.path.join(folder, 'wav')):
                    os.makedirs(os.path.join(folder, 'wav'))
                scikits.audiolab.wavwrite(noisy, os.path.join(folder,'wav/out_noisy_%s.wav') % args.snr[snr_idx], fs=srate, enc='pcm16')
                scikits.audiolab.wavwrite(Scc, os.path.join(folder,'wav/out_Scc_%s.wav') % args.snr[snr_idx], fs=srate, enc='pcm16')
                scikits.audiolab.wavwrite(Sdc, os.path.join(folder,'wav/out_Sdc_%s.wav') % args.snr[snr_idx], fs=srate, enc='pcm16')
                scikits.audiolab.wavwrite(Scd, os.path.join(folder,'wav/out_Scd_%s.wav') % args.snr[snr_idx], fs=srate, enc='pcm16')
                scikits.audiolab.wavwrite(Sdd, os.path.join(folder,'wav/out_Sdd_%s.wav') % args.snr[snr_idx], fs=srate, enc='pcm16')
                # add mses to lists for plotting
                mse_dc[snr_idx] = mean_squared_error(Scc, Sdc)
                mse_cd[snr_idx] = mean_squared_error(Scc, Scd)
                mse_dd[snr_idx] = mean_squared_error(Scc, Sdd)
                mse_noisy[snr_idx] = mean_squared_error(Scc, noisy)
                # save model
                if not os.path.exists(os.path.join(folder, 'npz')):
                    os.makedirs(os.path.join(folder, 'npz'))
                np.savez(os.path.join(folder,'npz/network_mag_snr_%s.npz') % args.snr[snr_idx], *lasagne.layers.get_all_param_values(pa_mag.network))
                np.savez(os.path.join(folder,'npz/latents_mag_snr_%s.npz') % args.snr[snr_idx], *lasagne.layers.get_all_param_values(pa_mag.latents))
                np.savez(os.path.join(folder,'npz/network_phase_snr_%s.npz') % args.snr[snr_idx], *lasagne.layers.get_all_param_values(pa_phase.network))
                np.savez(os.path.join(folder,'npz/latents_phase_snr_%s.npz') % args.snr[snr_idx], *lasagne.layers.get_all_param_values(pa_phase.latents))

                print mse_cc
                print mse_dc
                print mse_cd
                print mse_dd
                print mse_noisy
    with open(os.path.join(folder, 'config.txt'), 'a') as f:
        f.write('mse_cc: \n')
        pprint(mse_cc, stream=f)
        f.write('mse_dc: \n')
        pprint(mse_dc, stream=f)
        f.write('mse_cd: \n')
        pprint(mse_cd, stream=f)
        f.write('mse_dd: \n')
        pprint(mse_dd, stream=f)
        f.write('mse_noisy: \n')
        pprint(mse_noisy, stream=f)
    # plot MSE vs. SNR for various reconstructions
    plt.figure()
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE')
    plt.title('MSE vs. SNR for various signal reconstruction methods')
    plt.plot(args.snr, mse_cc)
    plt.plot(args.snr, mse_dc)
    plt.plot(args.snr, mse_cd)
    plt.plot(args.snr, mse_dd)
    plt.plot(args.snr, mse_noisy)
    plt.legend(['Baseline', 'Denoised Mag, Clean Phase', 'Clean Mag, Denoised Phase', 'Denoised Mag, Denoised Phase', 'Noisy'], loc=3)
    plt.savefig(os.path.join(folder, 'snr_mse.png'))
    plt.savefig(os.path.join(folder, 'snr_mse.pdf'))

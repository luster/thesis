"""Build time domain network

input: minibatches x minibatchsize x 1 x framelength x 1
latentspace: minibatches x minibatchsize x
output: minibatches x minibatchsize x 1 x framelength x 1

"""
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

from config import audioframe_len, specbinnum, srate
from dataset import build_dataset3, load_soundfiles
from build_networks import dtype, PartitionedAutoencoder
from util import calculate_time_signal
from sklearn.metrics import mean_squared_error

from lasagne.layers import batch_norm, DenseLayer
from lasagne.nonlinearities import elu
from conv_layer import custom_convlayer_2
from norm_layer import NormalisationLayer
from build_networks import PartitionedAutoencoder
from build_networks import ZeroOutBackgroundLatentsLayer


def normalize(this, against):
    return this / max(abs(this)) * max(abs(against))


class TimeDomainOutputPartitionedAutoencoder(PartitionedAutoencoder):

    def initialize_network(self):
        """This network uses an extra convolutional layer, utilizing ELU instead
        of ReLU or leaky ReLU from before. Also, we attempt to output the signal
        directly in the time domain by outputting the correct size output signal.
        """
        # intermediate layer size
        ils = int((self.specbinnum + self.numfilters) / 2)

        network = lasagne.layers.InputLayer((None, 1, self.specbinnum, self.numtimebins), self.input_var)

        network = NormalisationLayer(network, self.specbinnum)
        self.normlayer = network

        network, _ = custom_convlayer_2(network, in_num_chans=self.specbinnum, out_num_chans=ils)
        network = batch_norm(network)
        network, _ = custom_convlayer_2(network, in_num_chans=ils, out_num_chans=self.numfilters)
        network = batch_norm(network)

        network = lasagne.layers.NonlinearityLayer(network, nonlinearity=elu)
        self.latents = network
        network = ZeroOutBackgroundLatentsLayer(self.latents,
            mp_down_factor=self.mp_down_factor,
            numfilters=self.numfilters,
            numtimebins=self.numtimebins,
            background_latents_factor=self.background_latents_factor,
            use_maxpool=self.use_maxpool)
        network, _ = custom_convlayer_2(network, in_num_chans=self.numfilters, out_num_chans=ils)
        network = batch_norm(network)
        network, _ = custom_convlayer_2(network, in_num_chans=ils, out_num_chans=self.specbinnum)
        network = batch_norm(network)

        # output_size
        num_time_samples = int(audioframe_len/2 * (self.numtimebins + 1))
        # network = batch_norm(DenseLayer(network, num_time_samples))  # MemoryError
        network, _ = custom_convlayer_2(network, in_num_chans=self.specbinnum, out_num_chans=num_time_samples)
        network, _ = batch_norm(network)
        network, _ = custom_convlayer_2(network, in_num_chans=num_time_samples, out_num_chans=1)
        network, _ = batch_norm(network)

        self.network = network

    def loss_func(self, lambduh=0.75):
        prediction = self.get_output()
        loss = lasagne.objectives.squared_error(prediction, self.input_time_var)
        regularization_term = self.soft_output_var * ((self.C_mat * lasagne.layers.get_output(self.latents)).mean())**2
        loss = (loss.mean() + lambduh/self.mean_C * regularization_term).mean()
        return loss

    def train_fn_slim(self, updates='adadelta'):
        loss = self.loss_func()
        update_args = {
            'adadelta': (lasagne.updates.adadelta, {
                'learning_rate': 0.01, 'rho': 0.4, 'epsilon': 1e-6,
            }),
            'adam': (lasagne.updates.adam, {},),
        }[updates]
        update_func, update_params = update_args[0], update_args[1]

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = update_func(loss, params, **update_params)
        train_fn = theano.function([self.input_var, self.input_time_var, self.soft_output_var],
            loss, updates=updates,
            allow_input_downcast=True,
        )
        return train_fn


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--niter', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=int, default=8)
    parser.add_argument('-u', '--updates', type=str, default='adam')
    parser.add_argument('-m', '--minibatches', type=int, default=96)
    parser.add_argument('-b', '--minibatchsize', type=int, default=16)
    parser.add_argument('-k', '--snr', type=float, nargs='+', default=[-3,0,3,6,9,12])
    parser.add_argument('-t', '--timesignal', type=bool, default=True)
    parser.add_argument('-s', '--signal', type=str, default='../data/chon/signal_44.wav')
    parser.add_argument('-n', '--noise', type=str, default='../data/chon/noise_44.wav')
    args = parser.parse_args()

    numepochs = args.epochs
    k_values = 10 ** (-np.array(args.snr) / 20.)
    signal, noise = load_soundfiles(args.signal, args.noise)

    # create network(s)
    pa_mag = TimeDomainOutputPartitionedAutoencoder(num_minibatches=args.minibatches, specbinnum=specbinnum)
    print pa_mag.__dict__

    mse_cc = []  # clean reconstruction
    mse_dc = []  # denoised magnitude, clean phase
    mse_cd = []  # clean magnitude, denoised phase
    mse_dd = []  # denoised magnitude, denoised phase
    mse_noisy = []  # baseline mse, Y = S + N w.r.t. S
    for snr_idx, k in enumerate(k_values):
        # reinitialize networks
        if snr_idx > 0:
            pa_mag.initialize_network()

        print 'SNR = ', args.snr[snr_idx]

        # make dataset
        dataset_ = build_dataset3(signal, noise, sec_of_audio=30, k=k,
            training_data_size=args.minibatches,
            minibatch_size=args.minibatchsize, specbinnum=pa_mag.specbinnum,
            numtimebins=pa_mag.numtimebins,
            n_noise_only_examples=int(args.minibatchsize / 4), index=0)

        # vars from dataset
        training_labels = dataset_['training_labels']
        data_len = len(training_labels)

        # clean signal (baseline)
        idx = 105
        start = audioframe_len/2 * (idx + 1) - audioframe_len/2
        end = start + audioframe_len/2 * (pa_mag.numtimebins + 1)  #+ 1
        num_time_samples = int(end-start)
        clean = calculate_time_signal(dataset_['clean_magnitude'], dataset_['clean_phase'])
        Scc = normalize(clean, dataset_['clean_time_signal'])[start:end]
        import ipdb; ipdb.set_trace()
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
        test_latents_mag = lasagne.layers.get_output(pa_mag.latents, deterministic=True)
        predict_fn_mag = theano.function([pa_mag.input_var], test_prediction_mag)
        latents_fn_mag = theano.function([pa_mag.input_var], test_latents_mag)

        # data sample for reconstruction
        sample_mag = np.array([[dataset_['signal_magnitude'][:, idx:idx+pa_mag.numtimebins]]], dtype)

        train_fn_mag = pa_mag.train_fn_slim('adam')

        niter = args.niter
        for it_ in range(niter):
            dataset = build_dataset3(signal, noise, sec_of_audio=30, k=k,
                training_data_size=args.minibatches,
                minibatch_size=args.minibatchsize, specbinnum=pa_mag.specbinnum,
                numtimebins=pa_mag.numtimebins,
                n_noise_only_examples=int(args.minibatchsize / 4), index=0)
            # normalize
            pa_mag.normalize_batches(dataset['training_data_magnitude'])

            # train network(s)
            for epoch in xrange(numepochs):
                loss_mag = 0
                loss_phase = 0
                # indx_mag.set_value(0)
                # indx_phase.set_value(0)

                for batch_idx in range(args.minibatches):
                    print 'SNR = {}, Starting dataset {}/{}, epoch {}/{}, batch {}/{}'.format(args.snr[snr_idx], it_+1, niter, epoch+1, numepochs, batch_idx+1, args.minibatches)
                    loss_mag += train_fn_mag(dataset['training_data_magnitude'][batch_idx, :, :, :, :],
                        dataset['training_data_time'][batch_idx, :, :, :],
                        training_labels[batch_idx, :, :])
                lossreadout_mag = loss_mag / data_len
                infostring = "Epoch %d/%d: mag Loss %g" % (epoch, numepochs, lossreadout_mag)
                print infostring
                if epoch == 0 or epoch == numepochs - 1 or (2 ** int(np.log2(epoch)) == epoch) or epoch % 50 == 0:
                    """generate 4 time signals using networks:
                            Sdc: denoised mag, clean phase
                            Scd: clean mag, denoised phase
                            Sdd: denoised mag, denoised phase
                        using these signals, compute MSE with respect to baseline
                    """
                    prediction_mag = predict_fn_mag(sample_mag)
                    # Snoisy = normalize(calculate_time_signal(dataset_['noise_magnitude'][:, idx:idx+pa_mag.numtimebins], dataset_['noise_phase'][:, idx:idx+pa_mag.numtimebins]), Scc)
                    # Sdc = normalize(calculate_time_signal(prediction_mag, dataset_['clean_phase'][:, idx:idx+pa_mag.numtimebins]), Scc)
                    # Scd = normalize(calculate_time_signal(dataset_['clean_magnitude'][:, idx:idx+pa_mag.numtimebins], prediction_phase), Scc)
                    # Sdd = normalize(calculate_time_signal(prediction_mag, prediction_phase), Scc)

                    # print 'baseline mse: ', baseline_mse
                    # print '\tMSE noisy: ', mean_squared_error(Scc, Snoisy)
                    # print '\tMSE Sdc: ', mean_squared_error(Scc, Sdc)
                    # print '\tMSE Scd: ', mean_squared_error(Scc, Scd)
                    # print '\tMSE Sdd: ', mean_squared_error(Scc, Sdd)

                    latentsval_mag = latents_fn_mag(sample_mag)

        # normalize signals with respect to clean reconstruction
        # Sdc = normalize(calculate_time_signal(prediction_mag, dataset_['clean_phase'][:, idx:idx+pa_mag.numtimebins]), Scc)
        # Scd = normalize(calculate_time_signal(dataset_['clean_magnitude'][:, idx:idx+pa_mag.numtimebins], prediction_phase), Scc)
        # Sdd = normalize(calculate_time_signal(prediction_mag, prediction_phase), Scc)
        # save wav files
        # scikits.audiolab.wavwrite(noisy, 'wav/out_noisy_%s.wav' % args.snr[snr_idx], fs=srate, enc='pcm16')
        # scikits.audiolab.wavwrite(Scc, 'wav/out_Scc_%s.wav' % args.snr[snr_idx], fs=srate, enc='pcm16')
        # scikits.audiolab.wavwrite(Sdc, 'wav/out_Sdc_%s.wav' % args.snr[snr_idx], fs=srate, enc='pcm16')
        # scikits.audiolab.wavwrite(Scd, 'wav/out_Scd_%s.wav' % args.snr[snr_idx], fs=srate, enc='pcm16')
        # scikits.audiolab.wavwrite(Sdd, 'wav/out_Sdd_%s.wav' % args.snr[snr_idx], fs=srate, enc='pcm16')
        # add mses to lists for plotting
        # mse_dc.append(mean_squared_error(Scc, Sdc))
        # mse_cd.append(mean_squared_error(Scc, Scd))
        # mse_dd.append(mean_squared_error(Scc, Sdd))
        # mse_noisy.append(mean_squared_error(Scc, noisy))
        # save model
        np.savez('npz/network_mag_snr_%s.npz' % args.snr[snr_idx], *lasagne.layers.get_all_param_values(pa_mag.network))
        np.savez('npz/latents_mag_snr_%s.npz' % args.snr[snr_idx], *lasagne.layers.get_all_param_values(pa_mag.latents))
        np.savez('npz/network_phase_snr_%s.npz' % args.snr[snr_idx], *lasagne.layers.get_all_param_values(pa_phase.network))
        np.savez('npz/latents_phase_snr_%s.npz' % args.snr[snr_idx], *lasagne.layers.get_all_param_values(pa_phase.latents))

    print mse_cc
    print mse_dc
    print mse_cd
    print mse_dd
    print mse_noisy
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
    plt.savefig('snr_mse.png')
    plt.savefig('snr_mse.pdf')
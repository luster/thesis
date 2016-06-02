import numpy as np

import lasagne
import theano
import theano.tensor as T
#import downhill
from lasagne.nonlinearities import leaky_rectify
from lasagne.nonlinearities import rectify
from lasagne.nonlinearities import very_leaky_rectify
from numpy import float32

try:
    from lasagne.layers import InverseLayer as _
    use_maxpool = True
except ImportError:
    print("""**********************
        WARNING: InverseLayer not found in Lasagne. Please use a more recent version of Lasagne.
        WARNING: We'll deactivate the maxpooling part of the network (since we can't use InverseLayer to undo it)""")
    use_maxpool = False

import matplotlib
#matplotlib.use('PDF') # http://www.astrobetter.com/plotting-to-a-file-in-python/
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})

from config import *

# from network import input_var, signal_specgram, network


def make_plots(gram_name, outpostfix, plottitle=None, compute_time_signal=True):
    if gram_name == 'noise':
        sig = x_noise
        gram = noise_specgram
        phase = noise_phasegram
    elif gram_name == 'signal':
        sig = x_signal
        gram = signal_specgram
        phase = signal_phasegram
    else:
        raise Exception('invalid gram_name, %s' % gram)

    if plottitle==None:
        plottitle = outpostfix

    # if np.shape(plot_probedata_data)==():
    plot_probedata_data = np.array([[gram[:, examplegram_startindex:examplegram_startindex+numtimebins]]], dtype)

    test_prediction = lasagne.layers.get_output(network, deterministic=True, reconstruct=True)
    test_latents = lasagne.layers.get_output(latents, deterministic=True)
    predict_fn = theano.function([input_var], test_prediction)
    latents_fn = theano.function([input_var], test_latents)
    prediction = predict_fn(plot_probedata_data)
    latentsval = latents_fn(plot_probedata_data)

    n_plots = 3
    if compute_time_signal:
        n_plots = 4
        reconstructed_stft = prediction * np.exp(1j*phase[:, examplegram_startindex : examplegram_startindex + numtimebins])
        reconstructed = istft(np.squeeze(reconstructed_stft), sig)
        original_stft = plot_probedata_data * np.exp(1j * phase[:, examplegram_startindex : examplegram_startindex + numtimebins])
        original = istft(np.squeeze(original_stft), sig)
        real_original = sig[time_startindex : time_endindex]

    if False:
        print("Probedata  has shape %s and meanabs %g" % ( plot_probedata_data.shape, np.mean(np.abs(plot_probedata_data ))))
        print("Latents has shape %s and meanabs %g" % (latentsval.shape, np.mean(np.abs(latentsval))))
        print("Prediction has shape %s and meanabs %g" % (prediction.shape, np.mean(np.abs(prediction))))
        print("Ratio %g" % (np.mean(np.abs(prediction)) / np.mean(np.abs(plot_probedata_data))))

    util.mkdir_p('pdf')
    pdf = PdfPages('pdf/%s_autoenc_probe_%s.pdf' % (gram_name, outpostfix))
    plt.figure(frameon=False)
    #
    plt.subplot(n_plots, 1, 1)
    plotdata = plot_probedata_data[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Input')
    plt.title("%s" % (plottitle))
    #
    plt.subplot(n_plots, 1, 2)
    plotdata = latentsval[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Latents')
    #
    plt.subplot(n_plots, 1, 3)
    plotdata = prediction[0,0,:,:]
    plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
    plt.ylabel('Output')

    if compute_time_signal:
        plt.subplot(n_plots, 1, 4)
        plotdata = reconstructed
        # plt.plot(real_original, color='b', label='original signal')  # this signal is too big compared to the normalized ones
        plt.plot(original, color='k', label='original')
        plt.plot(plotdata, color='r', label='reconstructed')
        plt.legend()
        plt.ylabel('Output')
    #
    # plt.close()
    ##
    pdf.savefig()
    plt.close()
    pdf.close()

    # if outpostfix == 'trained' and compute_time_signal:
    if compute_time_signal:
        specgram_ = np.array([[gram[:, examplegram_startindex : examplegram_startindex + numtimebins]]], dtype)
        predicted_gram_ = predict_fn(specgram_)
        phasegram_ = phase[:, examplegram_startindex : examplegram_startindex + numtimebins]

        output_ = calculate_time_signal(predicted_gram_, phasegram_)
        # save to wav
        scikits.audiolab.wavwrite(output_, 'wav/out_%s.wav' % gram_name, fs=srate, enc='pcm16')
    return

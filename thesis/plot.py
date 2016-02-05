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
from dataset import build_dataset

# from network import input_var, signal_specgram, network


# examplegram_startindex = 500

# plot_probedata_data = None
# def plot_probedata(outpostfix, plottitle=None):
#     """Visualises the network behaviour.
#     NOTE: currently accesses globals. Should really be passed in the network, filters etc"""
#     global plot_probedata_data

#     if plottitle==None:
#         plottitle = outpostfix

#     if np.shape(plot_probedata_data)==():
#         plot_probedata_data = np.array([[signal_specgram[:, examplegram_startindex:examplegram_startindex+numtimebins]]], float32)

#     test_prediction = lasagne.layers.get_output(network, deterministic=True)
#     test_latents    = lasagne.layers.get_output(latents, deterministic=True)
#     predict_fn = theano.function([input_var], test_prediction)
#     latents_fn = theano.function([input_var], test_latents)
#     prediction = predict_fn(plot_probedata_data)
#     latentsval = latents_fn(plot_probedata_data)
#     if False:
#         print("Probedata  has shape %s and meanabs %g" % ( plot_probedata_data.shape, np.mean(np.abs(plot_probedata_data ))))
#         print("Latents has shape %s and meanabs %g" % (latentsval.shape, np.mean(np.abs(latentsval))))
#         print("Prediction has shape %s and meanabs %g" % (prediction.shape, np.mean(np.abs(prediction))))
#         print("Ratio %g" % (np.mean(np.abs(prediction)) / np.mean(np.abs(plot_probedata_data))))

#     util.mkdir_p('pdf')
#     pdf = PdfPages('pdf/autoenc_probe_%s.pdf' % outpostfix)
#     plt.figure(frameon=False)
#     #
#     plt.subplot(3, 1, 1)
#     plotdata = plot_probedata_data[0,0,:,:]
#     plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
#     plt.ylabel('Input')
#     plt.title("%s" % (plottitle))
#     #
#     plt.subplot(3, 1, 2)
#     plotdata = latentsval[0,0,:,:]
#     plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
#     plt.ylabel('Latents')
#     #
#     plt.subplot(3, 1, 3)
#     plotdata = prediction[0,0,:,:]
#     plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
#     plt.ylabel('Output')
#     #
#     pdf.savefig()
#     plt.close()
#     ##
#     for filtvar, filtlbl, isenc in [
#         (filters_enc, 'encoding', True),
#         (filters_dec, 'decoding', False),
#             ]:
#         plt.figure(frameon=False)
#         vals = filtvar.get_value()
#         #print("        %s filters have shape %s" % (filtlbl, vals.shape))
#         vlim = np.max(np.abs(vals))
#         for whichfilt in range(numfilters):
#             plt.subplot(3, 8, whichfilt+1)
#             # NOTE: for encoding/decoding filters, we grab the "slice" of interest from the tensor in different ways: different axes, and flipped.
#             if isenc:
#                 plotdata = vals[numfilters-(1+whichfilt),0,::-1,::-1]
#             else:
#                 plotdata = vals[:,0,whichfilt,:]

#             plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-vlim, vmax=vlim)
#             plt.xticks([])
#             if whichfilt==0:
#                 plt.title("%s filters (%s)" % (filtlbl, outpostfix))
#             else:
#                 plt.yticks([])

#         pdf.savefig()
#         plt.close()
#     ##
#     pdf.close()

# plot_probedata('init')

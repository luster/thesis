# Borrowed from Dan Stowell via https://github.com/danstowell/autoencoder-specgram
import numpy as np
from numpy import float32

import os
import errno
from scikits.audiolab import Sndfile
from scikits.audiolab import Format

from matplotlib.mlab import specgram

from config import *

def standard_specgram(signal):#, audioframe_len, audioframe_stride, specbinlow, specbinnum):
    "Return specgram matrix, made using the audio-layer config"
    return np.array(specgram(signal,
        NFFT=audioframe_len,
        noverlap=audioframe_len - audioframe_stride,
        window=np.hamming(audioframe_len))[0][specbinlow:specbinlow + specbinnum, :], dtype=float32)


def load_soundfile(inwavpath, startpossecs, maxdursecs=None):
    """Loads audio data, optionally limiting to a specified start position and duration.
    Must be SINGLE-CHANNEL and matching our desired sample-rate."""
    framelen = 4096
    hopspls = framelen
    unhopspls = framelen - hopspls
    if (framelen % wavdownsample) != 0:
        raise ValueError("framelen needs to be a multiple of wavdownsample: %i, %i" % (
            framelen, wavdownsample))
    if (hopspls % wavdownsample) != 0:
        raise ValueError("hopspls  needs to be a multiple of wavdownsample: %i, %i" % (
            hopspls, wavdownsample))
    if maxdursecs == None:
        maxdursecs = 9999
    sf = Sndfile(inwavpath, "r")
    splsread = 0
    framesread = 0
    if sf.channels != 1:
        raise ValueError(
            "Sound file %s has multiple channels (%i) - mono required." % (inwavpath, sf.channels))
    timemax_spls = int(maxdursecs * sf.samplerate)
    if sf.samplerate != (srate * wavdownsample):
        raise ValueError(
            "Sample rate mismatch: we expect %g, file has %g" % (srate, sf.samplerate))
    if startpossecs > 0:
        # note: returns IOError if beyond the end
        sf.seek(startpossecs * sf.samplerate)
    audiodata = np.array([], dtype=np.float32)
    while(True):
        try:
            if splsread == 0:
                chunk = sf.read_frames(framelen)[::wavdownsample]
                splsread += framelen
            else:
                chunk = np.hstack(
                    (chunk[:unhopspls], sf.read_frames(hopspls)[::wavdownsample]))
                splsread += hopspls
            framesread += 1
            if framesread % 25000 == 0:
                print("Read %i frames" % framesread)
            if len(chunk) != (framelen / wavdownsample):
                print("Not read sufficient samples - returning")
                break
            chunk = np.array(chunk, dtype=np.float32)
            audiodata = np.hstack((audiodata, chunk))
            if splsread >= timemax_spls:
                break
        except RuntimeError:
            break
    sf.close()
    return audiodata


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':
    y = standard_specgram(load_soundfile(signal_files, 0))
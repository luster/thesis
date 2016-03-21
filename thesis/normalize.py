"""normalize signals to have unit power

scale WAV file contents by the square of the sum of the power such that
    Sum over the signal squared = 1
"""
from __future__ import division
import os
import numpy as np
import scikits.audiolab
from util import load_soundfile

def normalize_wav_file(signal, noise, fs):
    s = load_soundfile(signal, 0)
    n = load_soundfile(noise, 0)

    power_s = sum(i**2 for i in s)/len(s)
    power_n = sum(i**2 for i in n)/len(n)
    print power_s, power_n
    n_out = n * np.sqrt(power_s/power_n)

    head, tail = os.path.split(noise)
    out_fname = os.path.join(
        head,
        "normalized_{}".format(tail),
    )
    scikits.audiolab.wavwrite(n_out, out_fname, fs=fs, enc='pcm16')


if __name__ == "__main__":
    import sys
    fs = 22050
    normalize_wav_file(sys.argv[1], sys.argv[2], fs)

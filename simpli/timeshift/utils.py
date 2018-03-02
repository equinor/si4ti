# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os

import numpy as np
import numpy.fft as fft
import segyio

def frequency_spectrum(n, dt = 1):
    return np.fft.fftfreq(n, dt)

def angular_frequency(traces, samples, dt = 1):
    """ MxN angular frequency (M = traces-per-inline, N = samples-per-trace)

    In order to derive and/or transform the input data line-by-line, it's
    useful to define the omega as a traces-by-samples matrix, where every row
    in the matrix is the repeated 2πf vector, where f is the frequency spectrum
    of the signal (number-of-samples)
    """
    m, n = traces, samples
    return np.tile(2 * np.pi * frequency_spectrum(n, dt), (m, 1))

def derive(signal, omega):
    """ D = iωF[d(t)]
        
        where D is the data-derivatived = data/signal, t = trace

        It takes the entire omega (given by the angular_frequency function) as
        an argument, in order to not recompute it on every invocation (it's
        shared across all data derivations in this application).
    """

    # the results from this function differe by a whopping 1e-5 from the
    # reference solution, but the reference solution's signal -> fft -> ifft
    # (without multiplication with omega) alone gives a difference of this
    # magnitude.  when the fft is done with double precision, both in numpy and
    # the reference program, this inaccuracy drops to 1e-14.

    ff = 1j * omega * fft.fft(signal.T)
    return fft.ifft(ff).real.T 

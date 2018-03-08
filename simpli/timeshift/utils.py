# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os

import numpy as np
import numpy.fft as fft
import segyio

from .bspline import bspline as bs

def bspline(samples, density, degree):
    """ bspline on matrix form

    Compute a MxN bspline matrix, where N = samples-per-trace, where the basis
    functions are all normalised
    """
    step = 1 / density
    middle = (samples + 1) / 2
    fst = np.arange(middle, 1 / density, -step)[::-1]
    lst = np.arange(middle, (1 + samples) - 1/density, step)[1:]
    knots = np.concatenate([fst, lst])
    knots = np.asarray(knots, dtype='single')

    rows = len(knots) + degree + 1
    output = np.zeros([rows, samples], dtype='single')

    bs(knots, output, samples, degree)

    # normalise
    output = output.T / (np.ones(output.shape[0]) * np.sum(output, axis=1))
    return output.T


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

def constraints(spline, vertical_smoothing, horizontal_smoothing):
    """ Smoothing constraints

        MxM matrix (M = number of spline functions) containing the vertical
        smoothing and central component of the horizontal smoothing.
    """
    Ipm = spline.dot(spline.T)
    tracelen = spline.shape[1]

    ones = np.ones(tracelen)
    D = (-np.diag(ones) + np.diag(ones[:-1], k = 1))[:-1].T
    Dpm = spline.dot(D.dot(D.T)).dot(spline.T)

    return (horizontal_smoothing * Ipm) + (vertical_smoothing * Dpm)

def linear_operator(derived, spline):
    """ Compute the linear operator

        MxM matrix (M = number of spline functions).
    """
    D = np.diag(derived)
    F = spline
    lt = np.matmul(F, D)
    return np.matmul(lt, lt.T)

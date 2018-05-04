# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os

import numpy as np
import numpy.fft as fft
import segyio

from .ts import bspline as bs
from .ts import derive as deriv
from .ts import fftfreq
from .ts import angfreq
from .ts import knotvec
from .ts import spline

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

    knots = knotvec(samples, density, np.zeros(len(knots), dtype='single'))

    rows = len(knots) + degree + 1

    output = np.zeros([samples, rows], order = 'F', dtype='single')
    return spline(samples, density, degree, output).T

    # normalise
    output = output / (np.ones(output.shape[1]) * np.sum(output, axis=0))
    return output.T


def frequency_spectrum(n, dt = 1):
    output = np.zeros(n, dtype='single')
    return fftfreq(n, dt, output)

def angular_frequency(traces, samples, dt = 1):
    """ MxN angular frequency (M = traces-per-inline, N = samples-per-trace)

    In order to derive and/or transform the input data line-by-line, it's
    useful to define the omega as a traces-by-samples matrix, where every row
    in the matrix is the repeated 2πf vector, where f is the frequency spectrum
    of the signal (number-of-samples)
    """
    m, n = traces, samples
    return np.tile(angfreq(n, dt, np.zeros(samples, dtype='single')), (m, 1))

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

    signal = np.asarray(signal, dtype='single')
    omega  = np.asarray(omega,  dtype='single')
    output = np.zeros(signal.shape, order='F', dtype='single')

    return deriv(signal, omega, output)

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

def solution(derived, delta, spline):
    """ Solution (right-hand-side of the system)

        N dimensional column vector (N = number of spline functions)
    """
    D = np.diag(derived)
    B = spline
    return np.matmul(B, D).dot(delta)

def normalize_surveys(surveys):
    if len(surveys) < 2:
        raise ValueError('Must have at least 2 surveys to normalize')
    ref_ilines, ref_xlines = surveys[0].ilines, surveys[0].xlines
    for survey in surveys:
        if not np.array_equal(survey.ilines, ref_ilines):
            msg = [
                'All surveys must have the same inlines.',
                '{}.ilines: {},'.format(survey.filename, survey.ilines),
                '{}.ilines: {}'.format(surveys[0].filename, ref_ilines),
            ]
            raise ValueError('\n'.join(msg))

        if not np.array_equal(survey.xlines, ref_xlines):
            msg = [
                'All surveys must have the same crosslines.',
                '{}.xlines: {},'.format(survey.filename, survey.xlines),
                '{}.xlines: {}'.format(surveys[0].filename, ref_xlines),
            ]
            raise ValueError('\n'.join(msg))

    acc = 0
    for survey in surveys:
        srv = np.abs(survey.trace.raw[:])
        acc += srv.sum() / (srv != 0).sum()

    # TODO: name/figure out why multiply-by-30
    return (acc * 30) / len(surveys)

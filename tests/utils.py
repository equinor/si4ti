import os

import pytest
import hypothesis
import hypothesis.strategies as st

import numpy as np

import simpli.timeshift as ts

# reference frequencies are obtained by running the reference program with
# various inputs and recording the outpud
reference_freqs = {
    10: {
        # n even, dt both prime and float
        1: [ 0,
             0.1000,
             0.2000,
             0.3000,
             0.4000,
            -0.5000,
            -0.4000,
            -0.3000,
            -0.2000,
            -0.1000,
        ],
        0.53: [ 0,
                0.1887,
                0.3774,
                0.5660,
                0.7547,
                -0.9434,
                -0.7547,
                -0.5660,
                -0.3774,
                -0.1887,
        ],
        7: [ 0,
             0.0143,
             0.0286,
             0.0429,
             0.0571,
            -0.0714,
            -0.0571,
            -0.0429,
            -0.0286,
            -0.0143
        ],
    },

    # n odd, dt float both larger and smaller than 1
    11: {
        1: [ 0,
             0.0909,
             0.1818,
             0.2727,
             0.3636,
             0.4545,
            -0.4545,
            -0.3636,
            -0.2727,
            -0.1818,
            -0.0909,
        ],
        .233: [ 0,
                0.3902,
                0.7803,
                1.1705,
                1.5607,
                1.9508,
               -1.9508,
               -1.5607,
               -1.1705,
               -0.7803,
               -0.3902,
        ],
        3.71: [ 0,
                0.0245,
                0.0490,
                0.0735,
                0.0980,
                0.1225,
               -0.1225,
               -0.0980,
               -0.0735,
               -0.0490,
               -0.0245,
        ],
    },

    # dt > n
    5: {
        10: [ 0,
              0.0200,
              0.0400,
             -0.0400,
             -0.0200,
        ],
        11: [ 0,
              0.0182,
              0.0364,
             -0.0364,
             -0.0182,
        ],
        13.7: [ 0,
                0.0146,
                0.0292,
               -0.0292,
               -0.0146
        ],
    },
}

def test_frequencies():
    freq = ts.frequency_spectrum

    assert np.allclose(freq(10, 1),    reference_freqs[10][1],    atol = 1e-4)
    assert np.allclose(freq(10, 0.53), reference_freqs[10][0.53], atol = 1e-4)
    assert np.allclose(freq(10, 7),    reference_freqs[10][7],    atol = 1e-4)

    assert np.allclose(freq(11, 1),    reference_freqs[11][1],    atol = 1e-4)
    assert np.allclose(freq(11, .233), reference_freqs[11][.233], atol = 1e-4)
    assert np.allclose(freq(11, 3.71), reference_freqs[11][3.71], atol = 1e-4)

    assert np.allclose(freq(5, 10),    reference_freqs[5][10],    atol = 1e-4)
    assert np.allclose(freq(5, 11),    reference_freqs[5][11],    atol = 1e-4)
    assert np.allclose(freq(5, 13.7),  reference_freqs[5][13.7],  atol = 1e-4)

@hypothesis.given(st.integers(3, 100), st.floats(0.01, 10))
def test_fuzzy_freqs(n, dt):
    freqs = ts.frequency_spectrum(n, dt)
    assert len(freqs) >= 1
    assert min(freqs) < 0
    assert max(freqs) > 0

def test_derivation():
    refdir = os.path.join(os.path.dirname(__file__), 'derivs')
    with np.load(os.path.join(refdir, 'deriv1.npz') ) as reference:
        signal, ref = reference['source'], reference['derived']
        omega = ts.angular_frequency(signal.shape[1], signal.shape[0])
        derived = ts.derive(signal, omega)
        assert np.allclose(derived, ref, atol = 1e-5)

    with np.load(os.path.join(refdir, 'deriv2.npz') ) as reference:
        signal, ref = reference['source'], reference['derived']
        omega = ts.angular_frequency(signal.shape[1], signal.shape[0])
        derived = ts.derive(signal, omega)
        assert np.allclose(derived, ref, atol = 1e-5)

    with np.load(os.path.join(refdir, 'deriv3.npz') ) as reference:
        signal, ref = reference['source'], reference['derived']
        omega = ts.angular_frequency(signal.shape[1], signal.shape[0])
        derived = ts.derive(signal, omega)
        assert np.allclose(derived, ref, atol = 1e-5)

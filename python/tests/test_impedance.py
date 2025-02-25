from __future__ import annotations

import numpy as np
import pytest
import xtgeo  # type: ignore[import-untyped]

# from si4ti import impedanct, ImpedanceOptions
from si4ti import compute_impedance  # type: ignore[import-not-found]

INPUT_FILES = [
    "../test-data/vint0.sgy",
    "../test-data/vint1.sgy",
    "../test-data/vint2.sgy",
]


@pytest.fixture
def input_cubes():
    return [xtgeo.cube_from_file(filename) for filename in INPUT_FILES]


def test_playing_with_cubes(input_cubes):
    # import os
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    print([c.values.shape for c in input_cubes])
    print([np.mean(c.values) for c in input_cubes])
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes)
    print([c.values.shape for c in relAI_cubes])
    print([np.mean(c.values) for c in relAI_cubes])
    print([c.values.shape for c in dsyn_cubes])
    print([np.mean(c.values) for c in dsyn_cubes])


def test_timevarying_wavelet_default_options(input_cubes):
    # import os
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # print([c.values.shape for c in input_cubes])
    # print([np.mean(c.values) for c in input_cubes])
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes)
    print("relAI cubes")
    print([c.values.shape for c in relAI_cubes])
    print([np.mean(c.values) for c in relAI_cubes])
    print("dsyn cubes")
    print([c.values.shape for c in dsyn_cubes])
    print([np.mean(c.values) for c in dsyn_cubes])

    ref_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    print([c.values.shape for c in ref_relAI_cubes])
    print("Reference relAI cubes")
    print([np.mean(c.values) for c in ref_relAI_cubes])
    # for input_cube, relAI_cube in zip(input_cubes, relAI_cubes)

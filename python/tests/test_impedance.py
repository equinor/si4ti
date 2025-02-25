from __future__ import annotations

import numpy as np
import pytest
import xtgeo  # type: ignore[import-untyped]

# from si4ti import impedanct, ImpedanceOptions
from si4ti import ImpedanceOptions, compute_impedance  # type: ignore[import-not-found]

INPUT_FILES = [
    "../test-data/vint0.sgy",
    "../test-data/vint1.sgy",
    "../test-data/vint2.sgy",
]


@pytest.fixture
def input_cubes():
    return [xtgeo.cube_from_file(filename) for filename in INPUT_FILES]


def compare_cubes(bases, references, avg_diff=8e-3, max_diff=8e-3):
    if len(bases) != len(references):
        msg = "Nonmatching number of base and reference cubes"
        raise ValueError(msg)

    for base, reference in zip(bases, references):
        diff = np.abs(base.values - reference.values)
        s = np.sum(sum(diff)) / np.sum(np.sum(reference.values))

        print(f"Avg: {s}, Max: {diff.max()}")
        if s > avg_diff:
            msg = f"Error: Average too high {s}"
            raise ValueError(msg)

        if not diff.max() < max_diff:
            msg = (
                "Error: Maximum absolute difference between cubes too high "
                f"{diff.max()} >= {max_diff}"
            )
            raise ValueError(msg)


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


def test_timeinvariant_wavelet_default_options(input_cubes):
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes)

    ref_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tinw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    ref_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tinw-dsyn-{i}-ref.sgy")
        for i in range(3)
    ]

    compare_cubes(relAI_cubes, ref_relAI_cubes)
    compare_cubes(dsyn_cubes, ref_dsyn_cubes)


def test_timevarying_wavelet_default_options(input_cubes):
    options = ImpedanceOptions()
    options.tv_wavelet = True
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes, options)

    ref_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    ref_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-dsyn-{i}-ref.sgy") for i in range(3)
    ]

    compare_cubes(relAI_cubes, ref_relAI_cubes)
    compare_cubes(dsyn_cubes, ref_dsyn_cubes)


def test_timevarying_wavelet_segmented(input_cubes):
    options = ImpedanceOptions()
    options.segments = 2
    options.max_iter = 3
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes, options)

    ref_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-segmented-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    ref_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-segmented-dsyn-{i}-ref.sgy")
        for i in range(3)
    ]

    compare_cubes(relAI_cubes, ref_relAI_cubes, avg_diff=28)
    compare_cubes(dsyn_cubes, ref_dsyn_cubes, avg_diff=28)

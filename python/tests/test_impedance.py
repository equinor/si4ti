from __future__ import annotations

import numpy.testing as npt
import pytest
import xtgeo  # type: ignore[import-untyped]

from si4ti import compute_impedance  # type: ignore[import-not-found]


@pytest.fixture
def input_cubes():
    INPUT_FILES = [
        "../test-data/vint0.sgy",
        "../test-data/vint1.sgy",
        "../test-data/vint2.sgy",
    ]
    return [xtgeo.cube_from_file(filename) for filename in INPUT_FILES]


def compare_cubes(actuals, references, rtol=1e-8, atol=1e-3, strict=True):
    for actual, expected in zip(actuals, references):
        npt.assert_allclose(
            actual.values, expected.values, rtol=rtol, atol=atol, strict=strict
        )


def test_timevarying_wavelet_default_options(input_cubes):
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes, tv_wavelet=True)

    expected_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    expected_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-dsyn-{i}-ref.sgy") for i in range(3)
    ]

    compare_cubes(relAI_cubes, expected_relAI_cubes)
    compare_cubes(dsyn_cubes, expected_dsyn_cubes)


def test_timeinvariant_wavelet_default_options(input_cubes):
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes)

    expected_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tinw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    expected_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tinw-dsyn-{i}-ref.sgy")
        for i in range(3)
    ]

    compare_cubes(relAI_cubes, expected_relAI_cubes)
    compare_cubes(dsyn_cubes, expected_dsyn_cubes)


def test_timevarying_wavelet_segmented(input_cubes):
    relAI_cubes, dsyn_cubes = compute_impedance(
        input_cubes,
        tv_wavelet=True,
        segments=2,
        max_iter=3,
    )

    expected_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-segmented-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    expected_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-segmented-dsyn-{i}-ref.sgy")
        for i in range(3)
    ]

    compare_cubes(relAI_cubes, expected_relAI_cubes)
    compare_cubes(dsyn_cubes, expected_dsyn_cubes)

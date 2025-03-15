from __future__ import annotations

import numpy as np
import pytest
import xtgeo  # type: ignore[import-untyped]
from si4ti import compute_impedance


@pytest.fixture
def input_cubes() -> list[xtgeo.Cubes]:
    INPUT_FILES = [
        "../test-data/vint0.sgy",
        "../test-data/vint1.sgy",
        "../test-data/vint2.sgy",
    ]
    return [xtgeo.cube_from_file(filename) for filename in INPUT_FILES]


@pytest.fixture
def input_cubes_xlinesorted(input_cubes: list[xtgeo.Cubes]) -> list[xtgeo.Cubes]:
    for c in input_cubes:
        c.swapaxes()
    return input_cubes


def compare_cubes(
    actuals: list[xtgeo.Cubes],
    references: list[xtgeo.Cubes],
    max_diff_bound: float = 4e-4,
    avg_diff_bound: float = 8e-3,
    # rtol: float = 1e-4,
    # atol: float = 4e-4,
    strict: bool = True,
) -> None:
    for actual, expected in zip(actuals, references):
        diff = actual.values - expected.values
        # max_diff = np.max(np.abs(diff))
        assert diff.max() <= max_diff_bound, (
            f"Max difference too high: {diff.max()} > {max_diff_bound}"
        )

        # s = np.abs(np.sum(np.sum(diff)) / np.sum(np.sum(expected.values)))
        s = abs(sum(diff.ravel()) / sum(expected.values.ravel()))
        assert s <= avg_diff_bound, (
            f"Average difference too high: {s} > {avg_diff_bound}"
        )

        np.testing.assert_allclose(
            actual.values, expected.values, rtol=0, atol=max_diff_bound, strict=strict
        )

        # np.testing.assert_allclose(
        #    actual.values, expected.values, rtol=rtol, atol=0, strict=strict
        # )


@pytest.mark.limit_memory("10.5 MB")
def test_timevarying_wavelet_default_options(input_cubes: list[xtgeo.Cubes]) -> None:
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


@pytest.mark.limit_memory("10.5 MB")
def test_timevarying_wavelet_xlinesorted_default_options(
    input_cubes_xlinesorted: list[xtgeo.Cubes],
) -> None:
    relAI_cubes, dsyn_cubes = compute_impedance(
        input_cubes_xlinesorted, tv_wavelet=True
    )

    expected_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    for c in expected_relAI_cubes:
        c.swapaxes()
    expected_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-dsyn-{i}-ref.sgy") for i in range(3)
    ]
    for c in expected_dsyn_cubes:
        c.swapaxes()

    compare_cubes(relAI_cubes, expected_relAI_cubes)
    compare_cubes(dsyn_cubes, expected_dsyn_cubes)


@pytest.mark.limit_memory("9.8 MB")
def test_timeinvariant_wavelet_default_options(input_cubes: list[xtgeo.Cubes]) -> None:
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


@pytest.mark.limit_memory("9.6 MB")
def test_timevarying_wavelet_segmented_xlinesorted(
    input_cubes_xlinesorted: list[xtgeo.Cubes],
) -> None:
    relAI_cubes, dsyn_cubes = compute_impedance(
        input_cubes_xlinesorted,
        tv_wavelet=True,
        segments=2,
        max_iter=3,
    )

    expected_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-segmented-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    for c in expected_relAI_cubes:
        c.swapaxes()
    expected_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-segmented-dsyn-{i}-ref.sgy")
        for i in range(3)
    ]
    for c in expected_dsyn_cubes:
        c.swapaxes()

    compare_cubes(relAI_cubes, expected_relAI_cubes)
    compare_cubes(dsyn_cubes, expected_dsyn_cubes)


@pytest.mark.limit_memory("9.6 MB")
def test_timevarying_wavelet_segmented(input_cubes: list[xtgeo.Cubes]) -> None:
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


# @pytest.fixture
# def large_cubes() -> list[xtgeo.Cubes]:
#    INPUT_FILES = [
#        "/Users/AEJ/projects/timeshift/cpp/datasets/sleipner4d/94-2001-processing/data/94p01ful.sgy",
#        "/Users/AEJ/projects/timeshift/cpp/datasets/sleipner4d/99-2001-processing/data/99p01ful.sgy",
#        "/Users/AEJ/projects/timeshift/cpp/datasets/sleipner4d/01-2001-processing/data/01p01ful.sgy",
#    ]
#    return [xtgeo.cube_from_file(filename) for filename in INPUT_FILES]
#
#
# def test_timeinvariant_wavelet_default_options_large_cubes(
#    large_cubes: list[xtgeo.Cubes],
# ) -> None:
#    relAI_cubes, dsyn_cubes = compute_impedance(
#        large_cubes,
#    )

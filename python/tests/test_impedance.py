import typing

import pytest
import xtgeo  # type: ignore[import-untyped]
from si4ti import compute_impedance

import segyio


@pytest.fixture
def input_cubes() -> list[xtgeo.Cube]:
    INPUT_FILES = [
        "../test-data/vint0.sgy",
        "../test-data/vint1.sgy",
        "../test-data/vint2.sgy",
    ]
    return [xtgeo.cube_from_file(filename) for filename in INPUT_FILES]


@typing.no_type_check
@pytest.fixture(scope="session")
def input_cubes_crosslinesorted(tmp_path_factory) -> list[xtgeo.Cube]:
    """Creates input cubes with crossline sorting in the same spirit as the
    tests for the command line interface tools.

    We cannot cannot inform xtgeo about the sorting when creating an xtgeo
    cube. Therefore, we create temporary SEG-Y files with crossline sorting.
    """
    INPUT_FILES = [
        "../test-data/vint0.sgy",
        "../test-data/vint1.sgy",
        "../test-data/vint2.sgy",
    ]

    crossline_sorted_dir = tmp_path_factory.mktemp("crossline_sorted_files")

    cubes = []

    for f in INPUT_FILES:
        dst_path = crossline_sorted_dir / f.split("/")[-1]

        with segyio.open(f, iline=193, xline=189) as src:
            spec = segyio.spec()
            spec.sorting = src.sorting
            spec.format = src.format
            spec.samples = src.samples
            spec.ilines = src.ilines
            spec.xlines = src.xlines
            spec.offsets = src.offsets
            spec.ext_headers = src.ext_headers
            spec.endian = src.endian
            with segyio.create(dst_path, spec) as dst:
                dst.text[0] = src.text[0]
                dst.bin = src.bin
                dst.header = src.header
                dst.trace = src.trace

        cubes.append(xtgeo.cube_from_file(dst_path))

    return cubes


def assert_cubes_equal(
    actuals: list[xtgeo.Cube],
    references: list[xtgeo.Cube],
    max_diff_bound: float = 8e-3,
    avg_diff_bound: float = 8e-3,
) -> None:
    """Checks for equality of computed cubes and reference cubes in the
    same spirit as tests of the command line tool, see `diff.py`."""
    for actual, expected in zip(actuals, references):
        diff = abs(actual.values - expected.values)
        assert diff.max() <= max_diff_bound, (
            f"Max difference too high: {diff.max()} > {max_diff_bound}"
        )

        s = abs(sum(diff.ravel()) / sum(expected.values.ravel()))
        assert s <= avg_diff_bound, (
            f"Average difference too high: {s} > {avg_diff_bound}"
        )


@pytest.mark.limit_memory("10.5 MB")
def test_timevarying_wavelet_default_options(input_cubes: list[xtgeo.Cube]) -> None:
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes, tv_wavelet=True)

    expected_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    expected_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-dsyn-{i}-ref.sgy") for i in range(3)
    ]

    assert_cubes_equal(relAI_cubes, expected_relAI_cubes)
    assert_cubes_equal(dsyn_cubes, expected_dsyn_cubes)


@pytest.mark.limit_memory("10.5 MB")
def test_timevarying_wavelet_crosslinesorted_default_options(
    input_cubes_crosslinesorted: list[xtgeo.Cube],
) -> None:
    relAI_cubes, dsyn_cubes = compute_impedance(
        input_cubes_crosslinesorted, tv_wavelet=True
    )

    expected_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    expected_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tvw-dsyn-{i}-ref.sgy") for i in range(3)
    ]

    assert_cubes_equal(relAI_cubes, expected_relAI_cubes)
    assert_cubes_equal(dsyn_cubes, expected_dsyn_cubes)


@pytest.mark.limit_memory("9.8 MB")
def test_timeinvariant_wavelet_default_options(input_cubes: list[xtgeo.Cube]) -> None:
    relAI_cubes, dsyn_cubes = compute_impedance(input_cubes)

    expected_relAI_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tinw-relAI-{i}-ref.sgy")
        for i in range(3)
    ]
    expected_dsyn_cubes = [
        xtgeo.cube_from_file(f"../test-data/imp-tinw-dsyn-{i}-ref.sgy")
        for i in range(3)
    ]

    assert_cubes_equal(relAI_cubes, expected_relAI_cubes)
    assert_cubes_equal(dsyn_cubes, expected_dsyn_cubes)


@pytest.mark.limit_memory("9.6 MB")
def test_timevarying_wavelet_segmented_crosslinesorted(
    input_cubes_crosslinesorted: list[xtgeo.Cube],
) -> None:
    relAI_cubes, dsyn_cubes = compute_impedance(
        input_cubes_crosslinesorted,
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

    assert_cubes_equal(relAI_cubes, expected_relAI_cubes)
    assert_cubes_equal(dsyn_cubes, expected_dsyn_cubes)


@pytest.mark.limit_memory("9.6 MB")
def test_timevarying_wavelet_segmented(input_cubes: list[xtgeo.Cube]) -> None:
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

    assert_cubes_equal(relAI_cubes, expected_relAI_cubes)
    assert_cubes_equal(dsyn_cubes, expected_dsyn_cubes)

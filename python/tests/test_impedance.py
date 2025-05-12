import pytest
import xtgeo
from si4ti import compute_impedance


@pytest.fixture
def input_cubes() -> list[xtgeo.Cube]:
    INPUT_FILES = [
        "../test-data/vint0.sgy",
        "../test-data/vint1.sgy",
        "../test-data/vint2.sgy",
    ]
    return [xtgeo.cube_from_file(filename) for filename in INPUT_FILES]


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


@pytest.mark.limit_memory("11.5 MB")
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


@pytest.mark.limit_memory("10.8 MB")
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


@pytest.mark.limit_memory("10.6 MB")
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

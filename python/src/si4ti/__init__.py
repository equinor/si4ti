"""si4ti: Seismic Inversion for Time-lapse Interpretation"""

__all__ = [
    "InvalidShapeError",
    "__doc__",
    "compute_impedance",
]

import numpy as np
import numpy.typing as npt
import xtgeo  # type: ignore[import-untyped]

from ._si4ti_python import (  # type: ignore[import-not-found]
    ImpedanceOptions,
    impedance,
)


class InvalidShapeError(Exception):
    pass


def _assert_all_cube_shapes_equal(input_cubes: list[xtgeo.Cube]) -> None:
    for c in input_cubes:
        if c.values.shape != input_cubes[0].values.shape:
            msg = (
                "Input cubes must all equal shape, i.e., the have the same "
                "number of inlines, crosslines and traces"
            )
            raise InvalidShapeError(msg)


def _numpy_arrays_to_xtgeo_cubes(
    input_cubes: list[xtgeo.Cube], numpy_arrays: list[npt.NDArray[np.float32]]
) -> list[xtgeo.Cube]:
    xtgeo_cubes = []
    for input_cube, values in zip(input_cubes, numpy_arrays):
        cube = xtgeo.Cube(
            xori=input_cube.xori,
            yori=input_cube.yori,
            zori=input_cube.zori,
            xinc=input_cube.xinc,
            yinc=input_cube.yinc,
            zinc=input_cube.zinc,
            ncol=input_cube.ncol,
            nrow=input_cube.nrow,
            nlay=input_cube.nlay,
            rotation=input_cube.rotation,
            values=values,
            ilines=input_cube.ilines,
            xlines=input_cube.xlines,
            yflip=input_cube.yflip,
        )

        xtgeo_cubes.append(cube)

    return xtgeo_cubes


def compute_impedance(
    input_cubes: list[xtgeo.Cube],
    inverse_polarity: bool = False,
    segments: int = 1,
    overlap: int = -1,
    tv_wavelet: bool = False,
    damping_3D: float = 0.0001,
    damping_4D: float = 0.0001,
    latsmooth_3D: float = 0.05,
    latsmooth_4D: float = 4.0,
    max_iter: int = 50,
) -> tuple[list[xtgeo.Cube], list[xtgeo.Cube]]:
    """Compute impedance from provided cubes and parameters

    Parameters
    ----------
    input_cubes : list[xtgeo.Cube]
        List of input cubes
    inverse-polarity : bool, optional
        Invert polarity of the data, by default False
    segments : int, optional
        Data domain splitting will be performed. Takes the number of segments
        as an argument, by default 1
    overlap : int, optional
        Number of inlines (crosslines if crossline sorted) overlap between
        segments when performing data domain splitting. Defaults to maximum
        number of iterations of the linear solver, by default -1
    tv_wavelet : bool, optional
        Use windowed time-varying wavelet. If set to false, time-invariant
        wavelet is used, by default False
    damping_3D : float, optional
        Constrains the relative acoustic impedance on each vintage so it does
        not deviate too much from zero, by default 0.0001
    damping_4D : float, optional
        constraint on the difference in relative acoustic impedance between
        vintages, by default 0.0001
    latsmooth_3D : float, optional
        Horizontal smoothing factor, by default 0.05
    latsmooth_4D : float, optional
        4D extension of the horizontal smoothing. This will give preference to
        a solution with similar lateral smoothness at corresponding points on
        the vintages, by default 4.0
    max_iter : int, optional
        Maximum number of iterations for liner solver, by default 50

    Returns
    -------
    Relative acoustic impedances and XXX: tuple[list[xtgeo.Cube], list[xtgeo.Cube]]
        Tuple of relAI and dsyn cubes
    """
    _assert_all_cube_shapes_equal(input_cubes)

    options = ImpedanceOptions()
    options.polarity = -1 if inverse_polarity else 1
    options.segments = segments
    options.overlap = overlap
    options.tv_wavelet = tv_wavelet
    options.damping_3D = damping_3D
    options.damping_4D = damping_4D
    options.latsmooth_3D = latsmooth_3D
    options.latsmooth_4D = latsmooth_4D
    options.max_iter = max_iter

    cubes_as_numpy_arrays = [c.values for c in input_cubes]
    relAI_cubes, dsyn_cubes = impedance(cubes_as_numpy_arrays, options)

    relAI_xtgeo_cubes = _numpy_arrays_to_xtgeo_cubes(input_cubes, relAI_cubes)
    dsyn_xtgeo_cubes = _numpy_arrays_to_xtgeo_cubes(input_cubes, dsyn_cubes)

    return (relAI_xtgeo_cubes, dsyn_xtgeo_cubes)

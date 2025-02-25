from __future__ import annotations

import numpy as np
import xtgeo  # type: ignore[import-untyped]

from ._core import (  # type: ignore[import-not-found]
    ImpedanceOptions,
    __version__,
    add,
    impedance,
    subtract,
)

default_options = ImpedanceOptions()


def compute_impedance(
    input_cubes: list[xtgeo.Cube], options: ImpedanceOptions = default_options
) -> tuple[list[xtgeo.Cube], list[xtgeo.Cube]]:
    """Compute impedance from input cubes"""
    # return _compute_impedance(input_cubes, options)
    cubes_as_numpy_arrays = [c.values for c in input_cubes]
    relAI_cubes, dsyn_cubes = impedance(cubes_as_numpy_arrays, options)

    def convert_cubes_to_xtgeo(
        input_cubes: list[xtgeo.Cube], cubes_to_convert: list[np.ndarray]
    ) -> list[xtgeo.Cube]:
        xtgeo_cubes = []
        for input_cube, values in zip(input_cubes, cubes_to_convert):
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
                filesrc="si4ti impedance computation",
                yflip=input_cube.yflip,
            )

            xtgeo_cubes.append(cube)

        return xtgeo_cubes

    relAI_xtgeo_cubes = convert_cubes_to_xtgeo(input_cubes, relAI_cubes)
    dsyn_xtgeo_cubes = convert_cubes_to_xtgeo(input_cubes, dsyn_cubes)
    return (relAI_xtgeo_cubes, dsyn_xtgeo_cubes)


def py_add(i, j):
    return add(i, j)


def py_subtract(i, j):
    return subtract(i, j)


__all__ = [
    "ImpedanceOptions",
    # TODO: Find out if we need the `__doc__` property
    # "__doc__",
    "__version__",
    "compute_impedance",
    "py_add",
    "py_subtract",
]

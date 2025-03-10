# si4ti - Python bindings

These are the Python bindings of [si4ti](https://github.com/equinor/si4ti):
> si4ti is a LGPL licensed seismic inversion tool for monitoring effects in 4D
> seismic from changes in acoustic properties of a reservoir."

The bindings currently only implement a version of si4ti's impedance
calculation. The timeshift calculation is not included in the Python bindings.


## Installation

It is easiest to install the bindings via `pip`. Note, that only a limit number
of platforms are supported at the moment.

## Building from source

### Dependencies

-   OpenMP
-   Eigen3
-   pybind
-   Optional: FFTW3 with single precision (`float`) interface

These are all compile-time dependencies. Once the Python package was built, the
dependencies are not needed anymore.

Note: OpenMP might be needed after all.

## Runtime

### Dependencies

-   NumPy
-   xtgeo

## Testing

### Dependencies

-   NumPy
-   pytest
-   pytest-memray
-   xtgeo

### Installation via pip (developer mode)

We need to specify the installation directory of the dependencies if they are
not installed in a path where CMake picks up the corresponding CMake
configuration files. On my machines I need to specify the paths fo:

-   OpenMP
-   pybind11

I need a command like this on MacOS:

```text
pip install -Ccmake.define.CMAKE_PREFIX_PATH="$(pybind11-config --cmakedir);/opt/homebrew/opt/libomp" -e . --verbose
```

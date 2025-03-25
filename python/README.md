# si4ti - Python interface

These are the Python interface of [si4ti](https://github.com/equinor/si4ti):
> si4ti is a LGPL licensed seismic inversion tool for monitoring effects in 4D
> seismic from changes in acoustic properties of a reservoir."

The interface currently only implement a version of si4ti's impedance
calculation. The timeshift calculation is not included in the Python interface.

## Installation

It is easiest to install the interface via `pip`. Note, that only a limit
number of platforms are supported at the moment.

## Building from source

### Dependencies

-   OpenMP
-   Eigen3
-   Optional: FFTW3 with single precision (`float`) interface

pybind11 is used to create Python bindings to the C++ core code, but does not
need to be installed as it is included as submodule in the repository and as
source code in the source distribution (sdist).

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

I need a command like this on MacOS:

```text
pip install -Ccmake.define.CMAKE_PREFIX_PATH="/opt/homebrew/opt/libomp" -e . --verbose
```

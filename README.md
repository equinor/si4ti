## Introduction ##
si4ti is a LGPL licensed seismic inversion tool for monitoring effects in 4D
seismic from changes in acoustic properties of a reservoir.

### Timeshift ###
si4ti timeshift computes the dynamic timeshift between traces of different
datasets (vintages). It uses vertical and horizontal smoothing constraints, and
no prior information.

### Impedance ###
si4ti impedance uses a statistical wavelet and lateral, horizontal and 4D
smoothing for computing the relative acoustic impedance of a set of vintages.

## Installation ##
Pre-built executables for Linux can be downloaded from
[here](https://github.com/equinor/si4ti/releases).

## Build from source ##
To build si4ti you need:
 * A C++11 compatible compiler (tested on gcc)
 * [CMake](https://cmake.org) version 3.5 or greater
 * [Eigen3](https://eigen.tuxfamily.org) version 3.3.4 or greater
 * [OpenMP](https://www.openmp.org)
 * [segyio](https://github.com/equinor/segyio) run cmake with EXPERIMENTAL=ON to get the required C++ headers
 * [fftw](https://www.fftw.org) If built with USE_FFTW=True

To build the documentation you also need:
 * [sphinx](https://pypi.org/project/Sphinx)

To build and install si4ti run the following commands in your console:

```bash
git clone https://github.com/equinor/si4ti
mkdir si4ti/build
cd si4ti/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
make install
```

To build the documentation run:
```bash
cmake .. -DBUILD_DOC=TRUE
make doc
```

## Usage ##
For more information on how to run the programs:
```bash
timeshift --help
impedance --help
apply-timeshift --help
```

## Python interface ##
si4ti provides a Python interface for the impedance calculations. No Python
interface for the timeshift is provided.

### Instalation from PyPI ###

Pre-built wheels are available for the following platforms:

TBD

#### Dependencies ####

The wheel should be self contained.

#### Installation ####

The pre-built wheels can be installed via `pip`:
interface via

```text
pip install si4ti
```

### Installation from source ###

You can install si4ti via the source distribution (sdist) or the repository.
This allows to compile the package with FFTW3 support as well as platform
specific optimisation which may improve the performance.

#### Dependencies ####

-   C++ compiler
-   CMake
-   OpenMP
-   Eigen3
-   FFTW3 with single precision (`float`) interface (optional)

If you want to run the tests, which is highly recommended, you also need

-   pytest
-   pytest-memray
-   NumPy
-   xtgeo

#### Installation ####

One can specify compile options directly during the `pip` invocation.

The following command builds and installs the Python interface with OpenMP and
FFTW (both installed via Homebrew) from the root of the repository. It also
enables platform specific optimisation (`-march=native`) and strict warnings
(`-Wall -pedantic`).

```text
pip install -Ccmake.define.CMAKE_PREFIX_PATH="/opt/homebrew/opt/libomp" \
    -Ccmake.define.USE_FFTW=ON \
    -Ccmake.define.CMAKE_CXX_FLAGS="-march=native -Wall -pedantic" \
    .
```

### Usage ###

The interface is inspired by the command line interface. Please check the
Python help for details. You can find the Python help via

```python
import si4ti
help(si4ti)
```

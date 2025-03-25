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
We provide two build flags that let you turn on and off the timeshift
(`BUILD_TIMESHIFT`) and impedance (`BUILD_IMPEDANCE`) tools. By default, both
tools are built.

To build si4ti you need:
 * A C++11 compatible compiler (tested on gcc)
 * [CMake](https://cmake.org) version 3.15 or greater
 * [Eigen3](https://eigen.tuxfamily.org) version 3.3.4 or greater
 * [OpenMP](https://www.openmp.org)
 * [segyio](https://github.com/equinor/segyio) run cmake with `EXPERIMENTAL=ON`
   to get the required C++ headers
 * [fftw](https://www.fftw.org) If built with `USE_FFTW=True`
 * [Boost](https://www.boost.org) `math` module version 1.76 or greater if
   `BUILD_TIMESHIFT=ON`

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

## Python bindings ##
si4ti provides a Python bindings for the impedance calculations. No Python
interface for the timeshift is provided.

### Installation ###
Pre-built wheels are available for the following platforms for Python 3.9 up to
3.12:

 * `manylinux_2_28` for `x86_64`
 * MacOS X 14.0 and newer for `arm64`
 * MacOS X 13.0 and newer for `x86_64`

The pre-built wheels can be installed via `pip`

```bash
pip install si4ti
```

### Build from source ###
You can install si4ti via the source distribution (sdist) or the Git
repository. This allows to compile the package with FFTW3 support as well as
platform specific optimisation which may improve the performance.

During compilation, you need the following dependencies.
 * A C++11 compatible compiler (tested on gcc and clang)
 * Python 3.9 or greater including the development headers
 * [CMake](https://cmake.org) version 3.15 or greater
 * [Eigen3](https://eigen.tuxfamily.org) version 3.3.4 or greater
 * [OpenMP](https://www.openmp.org)
 * [scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/)
   version 0.1 or greater, if build via `pip`
 * [fftw](https://www.fftw.org) with single precision (`float`) interface, if
   built with `USE_FFTW=True`

At runtime, you additionally need
 * [NumPy](https://numpy.org)
 * [xtgeo](https://xtgeo.readthedocs.io/en/stable/)

If you want to run the tests, which is highly recommended, you also need
 * [pytest](https://docs.pytest.org/en/stable/)
 * [pytest-memray](https://pytest-memray.readthedocs.io/en/latest/)

#### Compilation via CMake ####
The Python bindings can be compiled directly via the CMake interface following
the instructions above by setting the `BUILD_PYTHON` parameter to `ON`. The
command line from above would become

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_PYTHON=ON
```

If you only need the Python bindings and not the command line tools, you can
deactivate them from the build using the options mentioned above. In this case,
you can also deactivate the tests for the command line tools:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_PYTHON=ON \
  -DBUILD_TIMESHIFT=OFF \
  -DBUILD_IMPEDANCE=OFF \
  -DBUILD_TESTING=OFF
```

#### Compilation via pip ####
Compilation of the Python bindings via `pip` only builds the bindings and not
the command line tools. One can specify compile options directly during the
`pip` invocation similarly to the build process using CMake directly. The
following command builds and installs the Python bindings with OpenMP and FFTW
(both installed via Homebrew) from the root of the repository. It also enables
platform specific optimisation (`-march=native`) and strict warnings (`-Wall
-pedantic`).

```bash
pip install -Ccmake.define.CMAKE_PREFIX_PATH="/opt/homebrew/opt/libomp" \
    -Ccmake.define.USE_FFTW=ON \
    -Ccmake.define.CMAKE_CXX_FLAGS="-march=native -Wall -pedantic" \
    .
```

### Running Python bindings tests ###
After successful compilation and installation, navigate to the `python/`
directory and run pytest. For this to work, the `si4ti` package must be
available in your current Python environment.

```bash
cd python
python -m pytest --memray tests
```

### Usage ###
The interface is inspired by the command line interface. Please check the
Python help for details. You can find the Python help via

```python
import si4ti
help(si4ti)
```

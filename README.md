## Introduction ##
si4ti is a GPL licensed seismic inversion tool for monitoring effects in 4D 
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
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
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

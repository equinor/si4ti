#!/usr/bin/env bash

#PREFIX=$HOME/software/segyio-master-experimental-debug
PREFIX=$HOME/projects/github/si4ti/si4ti-install
BUILD_TYPE="Release"

OPENMP_DIR="/opt/homebrew/opt/libomp"
PYBIND_DIR=$(pybind11-config --cmakedir)

# CMake paths need to be separated by semicolons
CMAKE_PREFIX_PATH="${OPENMP_DIR};${PYBIND_DIR}"
echo $CMAKE_PREFIX_PATH

rm -rf build
mkdir build
pushd build

cmake ..  --log-level=VERBOSE -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" -DUSE_FFTW=ON

make -j7
make diff
make install

popd

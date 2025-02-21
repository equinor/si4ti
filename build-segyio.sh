#!/usr/bin/env bash

#PREFIX=$HOME/software/segyio-master-experimental-debug
PREFIX=$HOME/projects/github/si4ti/py312-segyio-master-experimental-debug
BUILD_TYPE="Release"

pushd segyio
rm -rf build
mkdir build
pushd build

PYBIND_DIR=$(pybind11-config --cmakedir)

cmake .. -DCMAKE_INSTALL_PREFIX=${PREFIX} -DEXPERIMENTAL=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DREQUIRE_PYTHON=ON -DBUILD_SHARED_LIBS=ON
make -j7
make test
make install

popd
popd

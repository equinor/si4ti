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

cmake ..  --log-level=VERBOSE -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DBUILD_SHARED_LIBS=ON -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" -DUSE_FFTW=ON -DSI4TI_BUILD_TIMESHIFT=ON -DSI4TI_PYTHON_BINDINGS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j7
make diff

echo "Running cppcheck"
cppcheck --enable=style,portability,performance,warning \
	--library=posix \
	--inline-suppr \
	--project=compile_commands.json \
	--error-exitcode=1 \
	--suppressions-list=../cppcheck/suppressions.txt \
	--xml \
	--check-level=exhaustive

make install

popd

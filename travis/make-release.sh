#! /usr/bin/env bash

set -e

mkdir segyio
pushd segyio

cmake -DBUILD_SHARED_LIBS=OFF     \
      -DCMAKE_BUILD_TYPE=Release  \
      -DBUILD_PYTHON=OFF          \
      -DBUILD_TESTING=OFF         \
      -DEXPERIMENTAL=ON           \
      -DSEGYIO_NO_GIT_VER=ON      \
      /io/segyio
make
make install
popd

mkdir eigen
pushd eigen
cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release /io/eigen
make
make install
popd

mkdir cppcheck
pushd cppcheck
cmake -DCMAKE_BUILD_TYPE=Release /io/cppcheck-src/cppcheck
make
make install
popd

mkdir build-x86-64
pushd build-x86-64
cmake -DBUILD_SHARED_LIBS=OFF               \
      -DCMAKE_BUILD_TYPE=Release            \
      -DCMAKE_INSTALL_PREFIX=/io/x86-64     \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON    \
      /io
make
cppcheck --enable=style,portability,performance,warning    \
         --library=posix                                   \
         --inline-suppr                                    \
         --project=compile_commands.json                   \
         --error-exitcode=1                                \
         --suppressions-list=/io/cppcheck/suppressions.txt \
         --xml
make test
make diff
make install
popd

mkdir build-skylake
pushd build-skylake
cmake -DBUILD_SHARED_LIBS=OFF               \
      -DCMAKE_BUILD_TYPE=Release            \
      -DCMAKE_CXX_FLAGS=-march=skylake      \
      -DCMAKE_INSTALL_PREFIX=/io/skylake    \
      /io
make
make install
popd

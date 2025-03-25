cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(si4ti VERSION 0.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(BUILD_TESTING "Build tests" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)

option(BUILD_TIMESHIFT "Build timeshift command line tool" ON)
option(BUILD_IMPEDANCE "Build impedace command line tool" ON)
option(BUILD_DOC "Build documentation" OFF)
option(USE_FFTW "Build with fftw" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)

if (BUILD_TIMESHIFT)
    find_package(Boost 1.76 REQUIRED CONFIG)
endif()

if(BUILD_TIMESHIFT OR BUILD_IMPEDANCE)
    find_package(segyio REQUIRED)
endif()

if(USE_FFTW)
    # TODO: If we can rely on the presence of FFTW installations that provide
    # `.cmake` files for FFTW we can use `find_package(FFTW REQUIRED)`. This
    # will simplify the linking. The currently available installations on RHEL
    # do not provide such CMake files.
    find_library(fftw3 NAMES fftw3 REQUIRED)
    find_library(fftw3f NAMES fftw3f REQUIRED)
    # The FFTW3 header is required by Eigen3 if `USE_FFTW` is enabled. Eigen3
    # will not look for this headers itself. We must add the header explicitly
    # if we do not use `find_package(FFTW) to compile on platforms where FFTW
    # is not installed into the standard include path of the compiler, e.g.,
    # when using homebrew.
    find_path(fftw3_includes NAMES "fftw3.h" REQUIRED)
endif()

if(BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    add_subdirectory(external/pybind11)
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
    if (NOT (BUILD_IMPEDANCE OR BUILD_TIMESHIFT))
        message(WARNING "Requested to build tests, but timeshift and impedance command line tool builds are disabled. Nothing will be build.")
    endif()
    include(CTest)
    enable_testing()
    add_subdirectory(external/catch2)
    add_custom_target(diff)
endif()

if(BUILD_TIMESHIFT)
    add_subdirectory(timeshift)
endif()

if(BUILD_IMPEDANCE)
    add_subdirectory(impedance)
endif()

add_subdirectory(doc)

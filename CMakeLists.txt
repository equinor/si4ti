cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(si4ti VERSION 0.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(BUILD_TESTING "Build tests" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.76 REQUIRED)
find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(segyio REQUIRED)
# TODO: If we can rely on the presence of FFTW installations that provide
# `.cmake` files for FFTW we can use `find_package(FFTW REQUIRED)`. This will
# simplify the linking. The currently available installations on RHEL do not
# provide such CMake files.
find_library(fftw3 NAMES fftw3 REQUIRED)
find_library(fftw3f NAMES fftw3f REQUIRED)
# The FFTW3 header is required by Eigen3 if `USE_FFTW` is enabled. Eigen3 will
# not look for this headers itself. We must add the header explicitly if we do
# not use `find_package(FFTW) to compile on platforms where FFTW is not
# installed into the standard include path of the compiler, e.g., when using
# homebrew.
find_path(fftw3_includes NAMES "fftw3.h" REQUIRED)

option(BUILD_DOC "Build documentation" OFF)
option(USE_FFTW "Build with fftw" OFF)

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(external/catch2)
    add_custom_target(diff)
endif()

add_subdirectory(timeshift)
add_subdirectory(doc)
add_subdirectory(impedance)

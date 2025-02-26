cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(si4ti VERSION 0.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(BUILD_TESTING "Build tests" ON)
option(PYTHON_BINDINGS "Build Python bindings" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.76 REQUIRED CONFIG)
find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(segyio REQUIRED)

option(BUILD_DOC "Build documentation" OFF)
option(USE_FFTW "Build with fftw" OFF)

if(USE_FFTW)
    find_library(fftw3 NAMES fftw3 REQUIRED)
    find_library(fftw3f NAMES fftw3f REQUIRED)
    find_path(fftw3_includes NAMES "fftw3.h" REQUIRED)
endif()

if(PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(external/catch2)
    add_custom_target(diff)
endif()

add_subdirectory(timeshift)
add_subdirectory(doc)
add_subdirectory(impedance)

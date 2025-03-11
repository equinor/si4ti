cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(si4ti VERSION 0.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(BUILD_TESTING "Build tests" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)

option(BUILD_TIMESHIFT "Build timeshift code" ON)
option(BUILD_IMPEDANCE "Build impedace code" ON)

option(BUILD_DOC "Build documentation" OFF)
option(USE_FFTW "Build with fftw" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)

if (BUILD_TIMESHIFT)
    find_package(Boost 1.76 REQUIRED CONFIG)
    # From Boost 1.82.0 Boost.math requires C++14 support
    if (Boost_VERSION VERSION_GREATER_EQUAL 1.82.0)
        set(CMAKE_CXX_STANDARD 14)
    endif()
endif()

if(BUILD_TIMESHIFT OR BUILD_IMPEDANCE)
    find_package(segyio REQUIRED)
endif()

if(USE_FFTW)
    find_library(fftw3 NAMES fftw3 REQUIRED)
    find_library(fftw3f NAMES fftw3f REQUIRED)
    find_path(fftw3_includes NAMES "fftw3.h" REQUIRED)
endif()

if(BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    add_subdirectory(external/pybind11)
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
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

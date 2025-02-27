cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(si4ti VERSION 0.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(BUILD_TESTING "Build tests" ON)
option(SI4TI_PYTHON_BINDINGS "Build Python bindings" OFF)

option(SI4TI_BUILD_TIMESHIFT "Build timeshift code" ON)
option(SI4TI_BUILD_IMPEDANCE "Build impedace code" ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)

if(SI4TI_BUILD_TIMESHIFT OR SI4TI_BUILD_IMPEDANCE)
    find_package(segyio REQUIRED)
endif()

option(BUILD_DOC "Build documentation" OFF)

option(USE_FFTW "Build with fftw" OFF)
option(SI4TI_USE_FFTW "Build with fftw" OFF)

if (USE_FFTW)
    message(DEPRECATION "Flag 'USE_FFTW' is deprecated. Use 'SI4TI_USE_FFTW' instead")
    set(SI4TI_USE_FFTW ${USE_FFTW})
endif()


if(SI4TI_USE_FFTW)
    find_library(fftw3 NAMES fftw3 REQUIRED)
    find_library(fftw3f NAMES fftw3f REQUIRED)
    find_path(fftw3_includes NAMES "fftw3.h" REQUIRED)
endif()

if(SI4TI_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(external/catch2)
    add_custom_target(diff)
endif()

if(SI4TI_BUILD_TIMESHIFT)
    add_subdirectory(timeshift)
endif()

if(SI4TI_BUILD_IMPEDANCE)
    add_subdirectory(impedance)
endif()

add_subdirectory(doc)

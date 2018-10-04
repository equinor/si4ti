cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(simpli VERSION 0.0.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(segyio REQUIRED)
find_library(fftw3 NAMES fftw3 REQUIRED)
find_library(fftw3f NAMES fftw3f REQUIRED)

option(BUILD_DOC "Build documentation" OFF)

add_custom_target(diff)

enable_testing()
add_subdirectory(external/catch2)
add_subdirectory(timeshift)
add_subdirectory(doc)
add_subdirectory(impedance)

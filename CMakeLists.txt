cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(simpli VERSION 0.0.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

find_package(Eigen3 REQUIRED)
find_library(fftw3f NAMES fftw3f REQUIRED)

enable_testing()
add_subdirectory(timeshift)

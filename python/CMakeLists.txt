cmake_minimum_required(VERSION 3.15...3.26)
message("Project name: ${SKBUILD_PROJECT_NAME}")
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)

option(USE_FFTW "Build with fftw" OFF)

if(USE_FFTW)
    find_library(fftw3 NAMES fftw3 REQUIRED)
    find_library(fftw3f NAMES fftw3f REQUIRED)
    find_path(fftw3_includes NAMES "fftw3.h" REQUIRED)
endif()

pybind11_add_module(_core MODULE src/main.cpp)
target_include_directories(_core PRIVATE ${CMAKE_SOURCE_DIR}/../impedance/src
)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
if (USE_FFTW)
    target_link_libraries(_core PRIVATE ${fftw3f} ${fftw3})
    target_include_directories(_core PRIVATE ${fftw3_includes})
    target_compile_definitions(_core PRIVATE EIGEN_FFTW_DEFAULT=1)
endif()
add_compile_definitions(MUTE_PROGRESS)

install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})

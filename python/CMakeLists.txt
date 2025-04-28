cmake_minimum_required(VERSION 3.15...3.27)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

if (NOT DEFINED SKBUILD)
    message(FATAL_ERROR "This project is intended to be built with scikit-build. Please use scikit-build to build this project.")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include(${CMAKE_SOURCE_DIR}/../cmake/FindImpedanceDependencies.cmake)

pybind11_add_module(_si4ti_python MODULE src/python_interface.cpp)
target_link_libraries(_si4ti_python PRIVATE pybind11::headers)
target_include_directories(_si4ti_python PRIVATE ${CMAKE_SOURCE_DIR}/../impedance/src
)
target_link_libraries(_si4ti_python PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
if (USE_FFTW)
    target_link_libraries(_si4ti_python PRIVATE ${fftw3f} ${fftw3})
    target_include_directories(_si4ti_python PRIVATE ${fftw3_includes})
    target_compile_definitions(_si4ti_python PRIVATE EIGEN_FFTW_DEFAULT=1)
endif()
target_compile_definitions(_si4ti_python PRIVATE MUTE_PROGRESS)


install(TARGETS _si4ti_python DESTINATION si4ti)

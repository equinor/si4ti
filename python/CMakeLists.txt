cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

include(${CMAKE_SOURCE_DIR}/../cmake/CommonConfiguration.cmake)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_si4ti_python MODULE src/python_interface.cpp)
target_link_libraries(_si4ti_python PRIVATE pybind11::headers)
target_include_directories(_si4ti_python PRIVATE ${CMAKE_SOURCE_DIR}/../impedance/src)
target_link_libraries(_si4ti_python PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
if (USE_FFTW)
    target_link_libraries(_si4ti_python PRIVATE ${fftw3f} ${fftw3})
    target_include_directories(_si4ti_python PRIVATE ${fftw3_includes})
    target_compile_definitions(_si4ti_python PRIVATE EIGEN_FFTW_DEFAULT=1)
endif()
target_compile_definitions(_si4ti_python PRIVATE MUTE_PROGRESS)

install(TARGETS _si4ti_python DESTINATION si4ti)

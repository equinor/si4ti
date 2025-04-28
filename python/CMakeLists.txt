#cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
cmake_minimum_required(VERSION 3.15...3.27)
#project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
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

find_package(Eigen3 3.3.4 REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)

option(USE_FFTW "Build with fftw" OFF)

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


set(PYTHON_INSTALL_DIR .)
install(TARGETS _si4ti_python DESTINATION si4ti)

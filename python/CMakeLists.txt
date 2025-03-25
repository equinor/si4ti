project(impedance-python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pybind11_add_module(_si4ti_python MODULE src/python_interface.cpp)
target_include_directories(_si4ti_python PRIVATE ${CMAKE_SOURCE_DIR}/impedance/src
)
target_link_libraries(_si4ti_python PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
if (USE_FFTW)
    target_link_libraries(_si4ti_python PRIVATE ${fftw3f} ${fftw3})
    target_include_directories(_si4ti_python PRIVATE ${fftw3_includes})
    target_compile_definitions(_si4ti_python PRIVATE EIGEN_FFTW_DEFAULT=1)
endif()
add_compile_definitions(MUTE_PROGRESS)

if(DEFINED SKBUILD_PROJECT_NAME)
    set(PYTHON_INSTALL_DIR .)
elseif(NOT DEFINED PYTHON_INSTALL_DIR)
    set(PYTHON_INSTALL_DIR lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/${CMAKE_PROJECT_NAME})
endif()

install(TARGETS _si4ti_python DESTINATION ${PYTHON_INSTALL_DIR} COMPONENT python)
install(FILES src/si4ti/__init__.py DESTINATION ${PYTHON_INSTALL_DIR} COMPONENT python)

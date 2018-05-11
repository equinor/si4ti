project(simpli-timeshift LANGUAGES CXX)

add_executable(timeshift src/timeshift.cpp)
target_link_libraries(timeshift ${fftw3}
                                ${fftw3f}
                                ${cnpy}
                                Eigen3::Eigen
                                segyio-shared
)
target_compile_definitions(timeshift PRIVATE EIGEN_FFTW_DEFAULT=1)

add_executable(tests test/testsuite.cpp)
target_include_directories(tests PRIVATE include)
target_link_libraries(tests catch2)
add_test(NAME test COMMAND tests)

project(simpli-timeshift LANGUAGES CXX)

add_executable(timeshift src/timeshift.cpp)
target_link_libraries(timeshift ${fftw3f}
                                ${fftw3}
                                Eigen3::Eigen
                                segyio-shared
)
target_compile_definitions(timeshift PRIVATE EIGEN_FFTW_DEFAULT=1)

add_executable(tests test/testsuite.cpp
                     test/timeshift.cpp
)
target_include_directories(tests PRIVATE src)
target_link_libraries(tests catch2 Eigen3::Eigen)
target_compile_definitions(tests PRIVATE EIGEN_FFTW_DEFAULT=1)
add_test(NAME test COMMAND tests)

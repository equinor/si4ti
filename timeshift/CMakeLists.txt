project(simpli-timeshift LANGUAGES CXX)

add_executable(timeshift src/timeshift.cpp)
target_link_libraries(timeshift fftw3f Eigen3::Eigen segyio-shared)
target_compile_definitions(timeshift PRIVATE EIGEN_FFT_DEFAULT=1)

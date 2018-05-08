project(simpli-timeshift LANGUAGES CXX)

add_executable(timeshift src/timeshift.cpp)
target_link_libraries(timeshift fftw3 Eigen3::Eigen segyio-shared)
target_compile_definitions(timeshift PRIVATE EIGEN_FFTW_DEFAULT=1)

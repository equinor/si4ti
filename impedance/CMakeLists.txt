project(simpli-impedance LANGUAGES CXX)

add_executable(impedance src/impedance.cpp)
target_link_libraries(impedance ${fftw3f}
                                ${fftw3}
                                Eigen3::Eigen
                                segyio
                                ${OpenMP_EXE_LINKER_FLAGS}
                                ${OpenMP_CXX_FLAGS}
)

target_compile_options(impedance
    PRIVATE ${OpenMP_CXX_FLAGS}
)
target_compile_definitions(impedance
    PRIVATE EIGEN_FFTW_DEFAULT=1
)

if(NOT BUILD_TESTING)
    return()
endif()

add_executable(impedance_tests test/testsuite.cpp
                               test/impedance.cpp
)
target_include_directories(impedance_tests PRIVATE src)
target_link_libraries(impedance_tests catch2
                                      ${fftw3f}
                                      ${fftw3}
                                      Eigen3::Eigen
                                      segyio
                                      ${OpenMP_EXE_LINKER_FLAGS}
                                      ${OpenMP_CXX_FLAGS}
)

target_compile_options(impedance_tests
    PRIVATE ${OpenMP_CXX_FLAGS}
)
target_compile_definitions(impedance_tests
    PRIVATE EIGEN_FFTW_DEFAULT=1
)

add_test(NAME test COMMAND impedance_tests)

configure_file(test/vint1.sgy test-data/vintage1.sgy COPYONLY)
configure_file(test/vint2.sgy test-data/vintage2.sgy COPYONLY)
configure_file(test/vint3.sgy test-data/vintage3.sgy COPYONLY)

configure_file(test/timevaryingwavelet.csv test-data/timevaryingwavelet.csv COPYONLY)

configure_file(test/forward_operator_tvw1.csv test-data/forward_operator_tvw1.csv COPYONLY)
configure_file(test/forward_operator_tvw2.csv test-data/forward_operator_tvw2.csv COPYONLY)
configure_file(test/forward_operator_tvw3.csv test-data/forward_operator_tvw3.csv COPYONLY)

configure_file(test/forward_operator_tinw1.csv test-data/forward_operator_tinw1.csv COPYONLY)
configure_file(test/forward_operator_tinw2.csv test-data/forward_operator_tinw2.csv COPYONLY)
configure_file(test/forward_operator_tinw3.csv test-data/forward_operator_tinw3.csv COPYONLY)

configure_file(test/inverse_operator_tvw1.csv test-data/inverse_operator_tvw1.csv COPYONLY)
configure_file(test/inverse_operator_tvw2.csv test-data/inverse_operator_tvw2.csv COPYONLY)
configure_file(test/inverse_operator_tvw3.csv test-data/inverse_operator_tvw3.csv COPYONLY)

configure_file(test/inverse_operator_tinw1.csv test-data/inverse_operator_tinw1.csv COPYONLY)
configure_file(test/inverse_operator_tinw2.csv test-data/inverse_operator_tinw2.csv COPYONLY)
configure_file(test/inverse_operator_tinw3.csv test-data/inverse_operator_tinw3.csv COPYONLY)

configure_file(test/solution_1D_tvw.csv test-data/solution_1D_tvw.csv COPYONLY)
configure_file(test/solution_1D_tinw.csv test-data/solution_1D_tinw.csv COPYONLY)

configure_file(test/solution_tvw.csv test-data/solution_tvw.csv COPYONLY)

configure_file(test/solution_1D_tvw_segmented.csv test-data/solution_1D_tvw_segmented.csv COPYONLY)
configure_file(test/solution_tvw_segmented.csv test-data/solution_tvw_segmented.csv COPYONLY)

configure_file(test/relAI-0.sgy test-data/relAI-0.sgy COPYONLY)
configure_file(test/relAI-1.sgy test-data/relAI-1.sgy COPYONLY)
configure_file(test/relAI-2.sgy test-data/relAI-2.sgy COPYONLY)

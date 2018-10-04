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

#
# Integration test suite
#

add_custom_target(diff-impedance)
add_dependencies(diff diff-impedance)

find_program(python python3)
if(NOT python)
    message(WARNING "Could not find a python3 executable, "
                    "diff-targets not built"
    )
    return()
endif()

list(APPEND cropped-files
    ${CMAKE_CURRENT_SOURCE_DIR}/../92NmoUpd_8-16stkEps_985_1281-cropped.sgy
    ${CMAKE_CURRENT_SOURCE_DIR}/../01NmoUpd_8-16stkEps_985_1281-cropped.sgy
    ${CMAKE_CURRENT_SOURCE_DIR}/../06NmoUpd_8-16stkEps_985_1281-cropped.sgy
)

add_custom_target(imp-tvw-cropped
    COMMENT "Running impedance, time varying wavelet (cropped)"
    COMMAND impedance -i 5 -x 21 --timevarying-wavelet ${cropped-files}
    DEPENDS impedance
)

add_custom_target(imp-tvw-diff-cropped
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py -a 28
        --iline 5 --xline 21
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-0-tvw.sgy relAI-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-1-tvw.sgy relAI-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-2-tvw.sgy relAI-2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-0-tvw.sgy dsyn-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-1-tvw.sgy dsyn-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-2-tvw.sgy dsyn-2.sgy
    COMMENT "Analysing impedance diff image (time varying wavelet)"
    DEPENDS imp-tvw-cropped
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_custom_target(imp-tinw-cropped
    COMMENT "Running impedance, time invariant wavelet (cropped)"
    COMMAND impedance -i 5 -x 21 ${cropped-files}
    DEPENDS imp-tvw-diff-cropped
)

add_custom_target(imp-tinw-diff-cropped
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py -a 4.4
        --iline 5 --xline 21
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-0-tinw.sgy relAI-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-1-tinw.sgy relAI-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-2-tinw.sgy relAI-2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-0-tinw.sgy dsyn-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-1-tinw.sgy dsyn-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-2-tinw.sgy dsyn-2.sgy
    COMMENT "Analysing impedance diff image (time invarant wavelet)"
    DEPENDS imp-tinw-cropped
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_custom_target(imp-segmented-cropped
    COMMENT "Running impedance, segmented (cropped)"
    COMMAND impedance -i 5 -x 21 --timevarying-wavelet --segments 2 --max-iter 3 ${cropped-files}
    DEPENDS imp-tinw-diff-cropped
)

add_custom_target(imp-segmented-diff-cropped
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py -a 28
        --iline 5 --xline 21
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-0-segmented.sgy relAI-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-1-segmented.sgy relAI-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-relAI-ref-2-segmented.sgy relAI-2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-0-segmented.sgy dsyn-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-1-segmented.sgy dsyn-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-dsyn-ref-2-segmented.sgy dsyn-2.sgy
    COMMENT "Analysing impedance diff image (segmented)"
    DEPENDS imp-segmented-cropped
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_dependencies(diff-impedance imp-tvw-diff-cropped imp-tinw-diff-cropped
                 imp-segmented-diff-cropped)

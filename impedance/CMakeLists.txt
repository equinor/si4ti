project(si4ti-impedance LANGUAGES CXX)

add_executable(impedance src/impedance.cpp)
target_link_libraries(impedance Eigen3::Eigen
                                segyio::segyio
                                OpenMP::OpenMP_CXX
)

if(USE_FFTW)
    target_link_libraries(impedance ${fftw3f} ${fftw3})
    target_include_directories(impedance PRIVATE ${fftw3_includes})

    target_compile_definitions(impedance
        PRIVATE EIGEN_FFTW_DEFAULT=1
    )
endif()

install(TARGETS impedance
        EXPORT impedance
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(NOT BUILD_TESTING)
    return()
endif()

add_executable(impedance_tests test/testsuite.cpp
                               test/impedance.cpp
)
target_include_directories(impedance_tests PRIVATE src)
target_link_libraries(impedance_tests catch2
                                      Eigen3::Eigen
                                      segyio::segyio
                                      OpenMP::OpenMP_CXX
)
target_compile_definitions(impedance_tests PRIVATE MUTE_PROGRESS)

if(USE_FFTW)
    target_link_libraries(impedance_tests ${fftw3f} ${fftw3})
    target_include_directories(impedance_tests PRIVATE ${fftw3_includes})

    target_compile_definitions(impedance_tests
        PRIVATE EIGEN_FFTW_DEFAULT=1
    )
endif()

add_test(NAME test COMMAND impedance_tests)

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

list(APPEND segy-files
    ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/vint0.sgy
    ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/vint1.sgy
    ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/vint2.sgy
)

add_custom_target(imp-tvw
    COMMENT "Running impedance, time varying wavelet"
    COMMAND impedance --timevarying-wavelet ${segy-files} 1>/dev/null
    DEPENDS impedance
)

add_custom_target(imp-tvw-diff
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py --atol=3e-4
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-relAI-0-ref.sgy relAI-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-relAI-1-ref.sgy relAI-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-relAI-2-ref.sgy relAI-2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-dsyn-0-ref.sgy dsyn-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-dsyn-1-ref.sgy dsyn-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-dsyn-2-ref.sgy dsyn-2.sgy
    COMMENT "Analysing impedance diff image (time varying wavelet)"
    DEPENDS imp-tvw
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_custom_target(imp-tinw
    COMMENT "Running impedance, time invariant wavelet (cropped)"
    COMMAND impedance ${segy-files} 1>/dev/null
    DEPENDS imp-tvw-diff
)

add_custom_target(imp-tinw-diff
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py --atol=5e-5
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-relAI-0-ref.sgy relAI-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-relAI-1-ref.sgy relAI-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-relAI-2-ref.sgy relAI-2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-dsyn-0-ref.sgy dsyn-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-dsyn-1-ref.sgy dsyn-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-dsyn-2-ref.sgy dsyn-2.sgy
    COMMENT "Analysing impedance diff image (time invarant wavelet)"
    DEPENDS imp-tinw
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_custom_target(imp-segmented
    COMMENT "Running impedance, segmented"
    COMMAND impedance --timevarying-wavelet --segments 2 --max-iter 3 ${segy-files} 1>/dev/null
    DEPENDS imp-tinw-diff
)

add_custom_target(imp-custom-filenames
    COMMENT "Running impedance, time varying wavelet"
    COMMAND impedance --output-files r1.sgy r2.sgy r3.sgy d1.sgy d2.sgy d3.sgy -- ${segy-files} 1>/dev/null
    DEPENDS impedance
)

add_custom_target(imp-custom-filenames-diff
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py --atol=3e-5
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-relAI-0-ref.sgy r1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-relAI-1-ref.sgy r2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-relAI-2-ref.sgy r3.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-dsyn-0-ref.sgy d1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-dsyn-1-ref.sgy d2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tinw-dsyn-2-ref.sgy d3.sgy
    COMMENT "Analysing impedance diff image (time varying wavelet)"
    DEPENDS imp-custom-filenames
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_custom_target(imp-segmented-diff
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py --atol=4e-4
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-relAI-0-ref.sgy relAI-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-relAI-1-ref.sgy relAI-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-relAI-2-ref.sgy relAI-2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-dsyn-0-ref.sgy dsyn-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-dsyn-1-ref.sgy dsyn-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-dsyn-2-ref.sgy dsyn-2.sgy
    COMMENT "Analysing impedance diff image"
    DEPENDS imp-segmented
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_custom_target(xlinesorted
    COMMENT "Running impedance, (crossline sorted)"
    COMMAND impedance -i 193 -x 189 --timevarying-wavelet ${segy-files} 1>/dev/null
    DEPENDS impedance
)

add_custom_target(xlinesorted-diff
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py --atol=3e-4
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-relAI-0-ref.sgy relAI-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-relAI-1-ref.sgy relAI-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-relAI-2-ref.sgy relAI-2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-dsyn-0-ref.sgy dsyn-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-dsyn-1-ref.sgy dsyn-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-tvw-dsyn-2-ref.sgy dsyn-2.sgy
    COMMENT "Analysing impedance diff image (crossline sorted)"
    DEPENDS xlinesorted
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_custom_target(imp-segmented-xlinesorted
    COMMENT "Running impedance, segmented"
    COMMAND impedance --timevarying-wavelet --segments 2 --max-iter 3 ${segy-files} 1>/dev/null
    DEPENDS imp-tinw-diff
)

add_custom_target(imp-segmented-xlinesorted-diff
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py --atol=4e-4
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-relAI-0-ref.sgy relAI-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-relAI-1-ref.sgy relAI-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-relAI-2-ref.sgy relAI-2.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-dsyn-0-ref.sgy dsyn-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-dsyn-1-ref.sgy dsyn-1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/imp-segmented-dsyn-2-ref.sgy dsyn-2.sgy
    COMMENT "Analysing impedance diff image (segmented)"
    DEPENDS imp-segmented-xlinesorted
            ${CMAKE_CURRENT_SOURCE_DIR}/../timeshift/diff.py
)

add_dependencies(diff-impedance imp-tvw-diff imp-tinw-diff imp-segmented-diff
                 imp-custom-filenames-diff xlinesorted-diff
                 imp-segmented-xlinesorted-diff
)

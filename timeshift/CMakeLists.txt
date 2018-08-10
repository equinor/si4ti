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
target_link_libraries(tests catch2
                            ${fftw3f}
                            ${fftw3}
                            Eigen3::Eigen
                            segyio-shared
)
target_link_libraries(tests catch2 Eigen3::Eigen ${fftw3f} ${fftw3}
segyio-shared)
target_compile_definitions(tests PRIVATE EIGEN_FFTW_DEFAULT=1)
add_test(NAME test COMMAND tests)

#
# Benchmarks and integration test suite
#

add_custom_target(diff)

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

add_custom_target(ts-cropped
    COMMENT "Running timeshift (cropped)"
    COMMAND timeshift -i 5 -x 21 ${cropped-files}
    DEPENDS timeshift
)

add_custom_target(diff-cropped
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py -r
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-ref-0.sgy timeshift-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-ref-1.sgy timeshift-1.sgy
    COMMENT "Analysing cropped diff image"
    DEPENDS ts-cropped
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_custom_target(ts-cropped-4Dcorr
    COMMENT "Running 4D corrected timeshift (cropped)"
    COMMAND timeshift -c -i 5 -x 21 -p ts-4Dcorr ${cropped-files}
    DEPENDS timeshift
)

add_custom_target(diff-cropped-4Dcorr
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py -r -a 9e-3
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-4Dcorr-ref-0.sgy ts-4Dcorr-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-4Dcorr-ref-1.sgy ts-4Dcorr-1.sgy
    COMMENT "Analysing 4D corrected cropped diff image"
    DEPENDS ts-cropped-4Dcorr
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_custom_target(ts-cropped-cumulative
    COMMENT "Running cumulative timeshift (cropped)"
    COMMAND timeshift -s -i 5 -x 21 -p ts-cropped ${cropped-files}
    DEPENDS timeshift
)

add_custom_target(diff-cropped-cumulative
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py -r
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-cumulative-ref-0.sgy ts-cropped-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-cumulative-ref-1.sgy ts-cropped-1.sgy
    COMMENT "Analysing cumulative cropped diff image"
    DEPENDS ts-cropped-cumulative
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_custom_target(ts-cropped-custom_norm
    COMMENT "Running timeshift with custom normalization"
    COMMAND timeshift -i 5 -x 21 -A 12.72372 -p ts-cust_norm ${cropped-files}
    DEPENDS timeshift
)

add_custom_target(diff-cropped-custom_norm
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py -r -a 8e-7 -m 9e-7
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-custom_norm-ref-0.sgy ts-cust_norm-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../cropped-custom_norm-ref-1.sgy ts-cust_norm-1.sgy
    COMMENT "Analysing timeshift with custom normalization"
    DEPENDS ts-cropped-custom_norm
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_dependencies(diff diff-cropped diff-cropped-4Dcorr diff-cropped-cumulative
                 diff-cropped-custom_norm)

find_program(valgrind valgrind)
if(NOT valgrind)
    message(WARNING "Could not find valgrind, memcheck/callgrind not built")
    return()
endif()

add_custom_target(memcheck
    COMMAND ${valgrind} --leak-check=full
        $<TARGET_FILE:timeshift> -i 5 -x 21 ${cropped-files}
    COMMENT "Running memcheck"
    DEPENDS timeshift ${cropped-files}
)

add_custom_command(
    OUTPUT callgrind.out.timeshift
    COMMAND ${valgrind} --tool=callgrind
                        --callgrind-out-file=callgrind.out.timeshift
        $<TARGET_FILE:timeshift> -i 5 -x 21 ${cropped-files}
        COMMENT "Running callgrind"
    DEPENDS timeshift ${cropped-files}
)

add_custom_target(callgrind DEPENDS callgrind.out.timeshift)

find_program(kcachegrind kcachegrind)
add_custom_target(showgrind
    COMMAND ${kcachegrind} callgrind.out.timeshift
    DEPENDS callgrind
)

configure_file(test/vintage1.sgy test-data/vintage1.sgy COPYONLY)
configure_file(test/vintage2.sgy test-data/vintage2.sgy COPYONLY)
configure_file(test/vintage3.sgy test-data/vintage3.sgy COPYONLY)

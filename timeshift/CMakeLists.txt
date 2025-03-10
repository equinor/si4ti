project(si4ti-timeshift LANGUAGES CXX)

add_executable(timeshift src/timeshift.cpp)
target_link_libraries(timeshift Eigen3::Eigen
                                segyio::segyio
                                OpenMP::OpenMP_CXX
                                Boost::boost
)

if(USE_FFTW)
    target_link_libraries(timeshift ${fftw3f} ${fftw3})
    target_include_directories(timeshift PRIVATE ${fftw3_includes})

    target_compile_definitions(timeshift
        PRIVATE EIGEN_FFTW_DEFAULT=1
    )
endif()

add_executable(apply_timeshift src/apply_timeshift.cpp)
target_link_libraries(apply_timeshift Eigen3::Eigen
                                      segyio::segyio
                                      Boost::boost
                                      OpenMP::OpenMP_CXX
)

install(TARGETS timeshift
                apply_timeshift
        EXPORT timeshift
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(NOT BUILD_TESTING)
    return()
endif()

add_executable(tests test/testsuite.cpp
                     test/timeshift.cpp
)
target_include_directories(tests PRIVATE src)
target_link_libraries(tests catch2
                            Eigen3::Eigen
                            segyio::segyio
                            OpenMP::OpenMP_CXX
                            Boost::boost
)
target_compile_definitions(tests PRIVATE MUTE_PROGRESS)

if(USE_FFTW)
    target_link_libraries(tests ${fftw3f} ${fftw3})
    target_include_directories(tests PRIVATE ${fftw3_includes})

    target_compile_definitions(tests
        PRIVATE EIGEN_FFTW_DEFAULT=1
    )
endif()

add_test(NAME test COMMAND tests)

configure_file(test/vintage1.sgy test-data/vintage1.sgy COPYONLY)
configure_file(test/vintage2.sgy test-data/vintage2.sgy COPYONLY)
configure_file(test/vintage3.sgy test-data/vintage3.sgy COPYONLY)

#
# Benchmarks and integration test suite
#

add_custom_target(diff-timeshift)
add_dependencies(diff diff-timeshift)

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

add_custom_target(ts
    COMMENT "Running timeshift"
    COMMAND timeshift ${segy-files} 1>/dev/null
    DEPENDS timeshift
)

add_custom_target(diff-ts
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/timeshift-0-ref.sgy timeshift-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/timeshift-1-ref.sgy timeshift-1.sgy
    COMMENT "Analysing diff image"
    DEPENDS ts
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_custom_target(ts-4Dcorr
    COMMENT "Running 4D corrected timeshift"
    COMMAND timeshift -c -p ts-4Dcorr ${segy-files} 1>/dev/null
    DEPENDS timeshift
)

add_custom_target(diff-4Dcorr
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py -a 9e-3
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/ts-4Dcorr-0-ref.sgy ts-4Dcorr-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/ts-4Dcorr-1-ref.sgy ts-4Dcorr-1.sgy
    COMMENT "Analysing 4D corrected diff image"
    DEPENDS ts-4Dcorr
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_custom_target(ts-cumulative
    COMMENT "Running cumulative timeshift"
    COMMAND timeshift -s -p ts-cumulative ${segy-files} 1>/dev/null
    DEPENDS timeshift
)

add_custom_target(diff-cumulative
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/ts-cumulative-0-ref.sgy ts-cumulative-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/ts-cumulative-1-ref.sgy ts-cumulative-1.sgy
    COMMENT "Analysing cumulative diff image"
    DEPENDS ts-cumulative
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_custom_target(ts-custom_norm
    COMMENT "Running timeshift with custom normalization"
    COMMAND timeshift -A 12.72372 -p ts-cust_norm ${segy-files} 1>/dev/null
    DEPENDS timeshift
)

add_custom_target(diff-custom_norm
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py -a 6e-5 -m 9e-6
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/ts-cust_norm-0-ref.sgy ts-cust_norm-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/ts-cust_norm-1-ref.sgy ts-cust_norm-1.sgy
    COMMENT "Analysing timeshift with custom normalization"
    DEPENDS ts-custom_norm
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_custom_target(ts-user-specified-output-files
    COMMENT "Running timeshift (cropped)"
    COMMAND timeshift --output-files out1.sgy out2.sgy -- ${segy-files} 1>/dev/null
    DEPENDS timeshift
)

add_custom_target(diff-user-specified-output-files
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/timeshift-0-ref.sgy out1.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/timeshift-1-ref.sgy out2.sgy
    COMMENT "Analysing diff image"
    DEPENDS ts-user-specified-output-files
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_custom_target(ts-crosslinesorted
    COMMENT "Running timeshift crosslinesorted"
    COMMAND timeshift -i 193 -x 189 -p crosslinesorted ${segy-files} 1>/dev/null
    DEPENDS timeshift
)

add_custom_target(diff-crosslinesorted
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/timeshift-0-ref.sgy timeshift-0.sgy
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/timeshift-1-ref.sgy timeshift-1.sgy
    COMMENT "Analysing diff image crosslinesorted"
    DEPENDS ts-crosslinesorted
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

list(GET segy-files 1 M1)
add_custom_target(apply-ts
    COMMENT "Running apply timeshift"
    COMMAND apply_timeshift ${M1} timeshift-1.sgy
    DEPENDS ts
)

add_custom_target(diff-apply-ts
    COMMAND ${python} ${CMAKE_CURRENT_SOURCE_DIR}/diff.py -a 3e-4 -m 6e-5
        ${CMAKE_CURRENT_SOURCE_DIR}/../test-data/shifted-ref.sgy shifted.sgy
    COMMENT "Analysing applied timeshift"
    DEPENDS apply-ts
            ${CMAKE_CURRENT_SOURCE_DIR}/diff.py
)

add_dependencies(diff-timeshift diff-ts diff-4Dcorr diff-cumulative
                 diff-custom_norm diff-apply-ts diff-user-specified-output-files
                 diff-crosslinesorted)

find_program(valgrind valgrind)
if(NOT valgrind)
    message(WARNING "Could not find valgrind, memcheck/callgrind not built")
    return()
endif()

add_custom_target(memcheck
    COMMAND ${valgrind} --leak-check=full
        $<TARGET_FILE:timeshift> ${segy-files}
    COMMENT "Running memcheck"
    DEPENDS timeshift ${cropped-files}
)

add_custom_command(
    OUTPUT callgrind.out.timeshift
    COMMAND ${valgrind} --tool=callgrind
                        --callgrind-out-file=callgrind.out.timeshift
        $<TARGET_FILE:timeshift> ${segy-files}
        COMMENT "Running callgrind"
    DEPENDS timeshift ${cropped-files}
)

add_custom_target(callgrind DEPENDS callgrind.out.timeshift)

find_program(kcachegrind kcachegrind)
add_custom_target(showgrind
    COMMAND ${kcachegrind} callgrind.out.timeshift
    DEPENDS callgrind
)

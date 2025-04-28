cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(si4ti VERSION 0.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(BUILD_TESTING "Build tests" ON)

option(BUILD_TIMESHIFT "Build timeshift command line tool" ON)
option(BUILD_IMPEDANCE "Build impedace command line tool" ON)
option(BUILD_DOC "Build documentation" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(BUILD_TESTING)
    if (NOT (BUILD_IMPEDANCE OR BUILD_TIMESHIFT))
        message(WARNING "Requested to build tests, but timeshift and impedance command line tool builds are disabled. Nothing will be build.")
    endif()
    include(CTest)
    enable_testing()
    add_subdirectory(external/catch2)
    add_custom_target(diff)
endif()

if(BUILD_TIMESHIFT)
    add_subdirectory(timeshift)
endif()

if(BUILD_IMPEDANCE)
    add_subdirectory(impedance)
endif()

add_subdirectory(doc)

project(simpli-doc)

if(NOT BUILD_DOC)
    return()
endif()

find_program(sphinx sphinx-build)

if(NOT sphinx)
    message(WARNING "Could not find sphinx, skipping documentation")
    set(BUILD_DOC OFF)
    return()
endif()

add_custom_target(doc
    COMMAND ${sphinx}
        -d ${CMAKE_CURRENT_BINARY_DIR}/.doctrees
        ${SPHINX_ARGS}
        ${CMAKE_CURRENT_SOURCE_DIR}     #input dir
        ${CMAKE_CURRENT_BINARY_DIR}/doc #output dir
    DEPENDS conf.py
            index.rst
            theory.rst 
    COMMENT "Building documentation with sphinx"
)

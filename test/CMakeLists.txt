function (add_wmma_test)
    cmake_parse_arguments(
        "PARGS"
        ""
        "NAME"
        "FILES"
        ${ARGN})

    string(TOUPPER ${PARGS_NAME} _name)

    anydsl_runtime_wrap(${_name}_OBJ
        FRONTEND "artic"
        NAME ${PARGS_NAME}
        FILES ${BACKEND_FILES} ${PARGS_FILES}
    )

    add_executable(${PARGS_NAME} ${${_name}_OBJ})
    target_link_libraries(${PARGS_NAME} ${AnyDSL_runtime_LIBRARIES})

    set_property(TARGET ${PARGS_NAME}
        APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${PARGS_NAME}.nvvm ${PARGS_NAME}.nvvm.bc cache)
    add_test(NAME ${PARGS_NAME}_test COMMAND ${PARGS_NAME})
endfunction()

add_wmma_test(NAME simple FILES common.art simple.art)
add_wmma_test(NAME layout FILES common.art layout.art)

file(GLOB GENERATED_TESTS generated/*.art)
if(NOT GENERATED_TESTS)
    message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
endif()
foreach(_case ${GENERATED_TESTS})
    get_filename_component(TESTNAME ${_case} NAME_WLE)
    message(STATUS "Generated test ${TESTNAME}")
    add_wmma_test(NAME ${TESTNAME} FILES common.art ${_case})
endforeach()

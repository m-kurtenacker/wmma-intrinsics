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
add_wmma_test(NAME medium FILES common.art medium.art)

file(GLOB GENERATED_TESTS generated/*.art)
if(NOT GENERATED_TESTS)
    message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
endif()
foreach(_case ${GENERATED_TESTS})
    get_filename_component(TESTNAME ${_case} NAME_WLE)
    message(STATUS "Generated test ${TESTNAME}")
    add_wmma_test(NAME ${TESTNAME} FILES common.art ${_case})
endforeach()


set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

add_wmma_test(NAME oneapi FILES oneapi.art)
add_wmma_test(NAME oneapi_half FILES oneapi_half.art)
add_wmma_test(NAME oneapi_double FILES oneapi_double.art)

target_link_libraries(oneapi ${BLAS_LIBRARIES})
target_link_libraries(oneapi_half ${BLAS_LIBRARIES})
target_link_libraries(oneapi_double ${BLAS_LIBRARIES})

add_wmma_test(NAME large  FILES common_blas.art large.art)
target_link_libraries(large ${BLAS_LIBRARIES})

add_wmma_test(NAME large_ref  FILES nvvm_ref_blas.art large.art)
target_link_libraries(large_ref ${BLAS_LIBRARIES})

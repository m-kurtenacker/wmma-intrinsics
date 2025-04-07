#set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

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

set(BACKEND "nvvm" CACHE STRING "select the backend from the following: NVVM, AMDGPU_HSA, ONEAPI")
set_property(CACHE BACKEND PROPERTY STRINGS nvvm amdgpu_hsa levelzero oneapi)
string(TOLOWER ${BACKEND} BACKEND)

if (BACKEND STREQUAL "nvvm")
    add_wmma_test(NAME simple FILES common.art operations/nvvm_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/nvvm_ref.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/nvvm_ref.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME} FILES common.art operations/nvvm_blocked.art ${_case})
    endforeach()

    add_wmma_test(NAME large  FILES common.art operations/nvvm_blas.art layouts/large.art)
    target_link_libraries(large ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_tiled  FILES common.art operations/nvvm_tiled.art layouts/large.art)
    target_link_libraries(large_tiled ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_ref  FILES common.art operations/nvvm_blocked.art layouts/large.art)
    target_link_libraries(large_ref ${BLAS_LIBRARIES})

    #add_wmma_test(NAME medium_prgmod  FILES common.art operations/prgmod.art layouts/medium.art)
    #target_link_libraries(medium_prgmod ${BLAS_LIBRARIES})
elseif (BACKEND STREQUAL "amdgpu_hsa")
    add_wmma_test(NAME simple FILES common.art operations/hsa_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/hsa_ref.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/hsa_ref.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME} FILES common.art operations/hsa_blocked.art ${_case})
    endforeach()

    add_wmma_test(NAME large  FILES common.art operations/hsa_blas.art layouts/large.art)
    target_link_libraries(large ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_tiled  FILES common.art operations/hsa_tiled.art layouts/large.art)
    target_link_libraries(large_tiled ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_ref  FILES common.art operations/hsa_blocked.art layouts/large.art)
    target_link_libraries(large_ref ${BLAS_LIBRARIES})
elseif (BACKEND STREQUAL "levelzero")
    add_wmma_test(NAME simple FILES common.art operations/lzero_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/lzero_ref.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/lzero_ref.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME} FILES common.art operations/hsa_blocked.art ${_case})
    endforeach()

    add_wmma_test(NAME large  FILES common.art operations/lzero_blas.art layouts/large.art)
    target_link_libraries(large ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_tiled  FILES common.art operations/lzero_tiled.art layouts/medium.art)
    target_link_libraries(large_tiled ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_ref  FILES common.art operations/lzero_blocked.art layouts/large.art)
    target_link_libraries(large_ref ${BLAS_LIBRARIES})
elseif (BACKEND STREQUAL "oneapi")
    add_subdirectory(oneapi)
endif()

#set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

function (add_wmma_test)
    cmake_parse_arguments(
        "PARGS"
        ""
        "NAME"
        "FILES;UTILS"
        ${ARGN})

    string(TOUPPER ${PARGS_NAME} _name)

    anydsl_runtime_wrap(${_name}_OBJ
        FRONTEND "artic"
        NAME ${PARGS_NAME}
        FILES ${BACKEND_FILES} ${PARGS_FILES}
    )

    add_executable(${PARGS_NAME} ${${_name}_OBJ} ${PARGS_UTILS})
    target_link_libraries(${PARGS_NAME} ${AnyDSL_runtime_LIBRARIES})

    set_property(TARGET ${PARGS_NAME}
        APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${PARGS_NAME}.nvvm ${PARGS_NAME}.nvvm.bc cache)
    add_test(NAME ${PARGS_NAME}_test COMMAND ${PARGS_NAME})
endfunction()

set(BACKEND "nvvm" CACHE STRING "select the backend from the following: NVVM, AMDGPU_HSA, LEVELZERO, ONEAPI, HALIDE, AVX")
set_property(CACHE BACKEND PROPERTY STRINGS nvvm cuda amdgpu_hsa levelzero oneapi halide avx)
string(TOLOWER ${BACKEND} BACKEND)

if (BACKEND STREQUAL "nvvm")
    add_wmma_test(NAME simple FILES common.art operations/nvvm_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/nvvm_ref.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/nvvm_ref.art layouts/medium.art)

    add_wmma_test(NAME nowmma FILES common.art operations/nvvm_nowmma.art layouts/simple.art)
    add_wmma_test(NAME medium_nowmma FILES common.art operations/nvvm_tiled_nowmma.art layouts/medium.art)

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

    add_wmma_test(NAME medium_prgmod  FILES common.art operations/prgmod.art layouts/large.art)
    target_link_libraries(medium_prgmod ${BLAS_LIBRARIES})

    add_wmma_test(NAME bench_large_nowmma FILES benchmark.art operations/nvvm_tiled_nowmma.art layouts/large.art UTILS benchmark_util.c)
    add_wmma_test(NAME bench_large_tiled FILES benchmark.art operations/nvvm_tiled.art layouts/large.art UTILS benchmark_util.c)
    add_wmma_test(NAME bench_large_simple FILES benchmark.art operations/nvvm_ref.art layouts/large.art UTILS benchmark_util.c)
    add_wmma_test(NAME bench_large_blocked FILES benchmark.art operations/nvvm_blocked.art layouts/large.art UTILS benchmark_util.c)
elseif (BACKEND STREQUAL "cuda")
    add_wmma_test(NAME simple FILES common.art operations/cuda_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/cuda_ref.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/cuda_ref.art layouts/medium.art)

    add_wmma_test(NAME large  FILES common.art operations/cuda_blas.art layouts/large.art)
    target_link_libraries(large ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_tiled  FILES common.art operations/cuda_tiled.art layouts/large.art)
    target_link_libraries(large_tiled ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_ref  FILES common.art operations/cuda_blocked.art layouts/large.art)
    target_link_libraries(large_ref ${BLAS_LIBRARIES})
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
    add_wmma_test(NAME simple FILES common.art operations/lzero_ref.art operations/lzero_tiling_alchemist.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/lzero_ref.art operations/lzero_tiling_alchemist.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/lzero_ref.art operations/lzero_tiling_alchemist.art layouts/medium.art)

    add_wmma_test(NAME nowmma FILES common.art operations/lzero_nowmma.art layouts/simple.art)
    add_wmma_test(NAME medium_nowmma FILES common.art operations/lzero_tiled_nowmma.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME} FILES common.art operations/lzero_blocked.art operations/lzero_tiling_alchemist.art ${_case})
    endforeach()

    add_wmma_test(NAME large  FILES common.art operations/lzero_blas.art operations/lzero_tiling_alchemist.art layouts/large.art)
    target_link_libraries(large ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_tiled  FILES common.art operations/lzero_tiled.art operations/lzero_tiling_alchemist.art layouts/large.art)
    target_link_libraries(large_tiled ${BLAS_LIBRARIES})

    add_wmma_test(NAME large_ref  FILES common.art operations/lzero_blocked.art operations/lzero_tiling_alchemist.art layouts/large.art)
    target_link_libraries(large_ref ${BLAS_LIBRARIES})

    add_wmma_test(NAME nowmma_large FILES common.art operations/lzero_nowmma_blas.art layouts/large.art)
    target_link_libraries(nowmma_large ${BLAS_LIBRARIES})

    add_wmma_test(NAME wmma_large FILES common.art operations/lzero_wmma_blas.art operations/lzero_tiling_alchemist.art layouts/large.art)
    target_link_libraries(wmma_large ${BLAS_LIBRARIES})
elseif (BACKEND STREQUAL "avx")
    add_wmma_test(NAME simple FILES common.art operations/avx_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/avx_ref.art layouts/layout.art)
    add_wmma_test(NAME layout_inv FILES common.art operations/avx_inv.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/avx_ref.art layouts/medium.art)

    add_wmma_test(NAME dlp FILES common.art operations/avx_dlp.art layouts/medium_avx.art)
    add_wmma_test(NAME ilp FILES common.art operations/avx_ilp.art layouts/medium_avx.art)
    add_wmma_test(NAME mho FILES common.art operations/avx_mho.art layouts/medium_avx.art)
    add_wmma_test(NAME tlp FILES common.art operations/avx_tlp_blas.art layouts/medium_avx.art)
    target_link_libraries(tlp ${BLAS_LIBRARIES})

    add_wmma_test(NAME strassen FILES common.art operations/avx_strassen_blas.art layouts/medium_avx.art)
    target_link_libraries(strassen ${BLAS_LIBRARIES})

    add_wmma_test(NAME prgmod FILES common.art operations/avx_prgmod.art layouts/medium_prgmod.art)
    target_link_libraries(prgmod ${BLAS_LIBRARIES})

    add_wmma_test(NAME large FILES common.art operations/avx_tlp_blas.art layouts/large_avx.art)
    target_link_libraries(large ${BLAS_LIBRARIES})

    file(GLOB GENERATED_TESTS generated/*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME}_ref FILES common.art operations/avx_ref.art ${_case})
        add_wmma_test(NAME ${TESTNAME}_inv FILES common.art operations/avx_inv.art ${_case})
    endforeach()
elseif (BACKEND STREQUAL "oneapi")
    add_subdirectory(oneapi)
elseif (BACKEND STREQUAL "halide")
    add_subdirectory(halide)
endif()

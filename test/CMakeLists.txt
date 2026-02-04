#option 1, standard blas lib, optionally with vendor selected
#set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

#option 2, cuda blas lib, requires a shim library to translate functions
#find_package(CUDAToolkit)
#add_library(cublas_util util/cublas_util.cpp)
#target_link_libraries(cublas_util CUDA::cublas)

#option 3, rocblas blas lib
#find_package(rocblas REQUIRED)
#add_library(rocblas_util util/rocblas_util.cpp)
#target_link_libraries(rocblas_util roc::rocblas)

function (add_blas_libraries target)
    target_link_libraries(${target} ${BLAS_LIBRARIES})
    #target_link_libraries(${target} CUDA::cublas cublas_util)
    #target_link_libraries(${target} roc::rocblas rocblas_util)
endfunction()

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
        APPEND PROPERTY ADDITIONAL_CLEAN_FILES ${PARGS_NAME}.nvvm ${PARGS_NAME}.amdgpu ${PARGS_NAME}.amdgpu ${PARGS_NAME}.amdgpu.obj ${PARGS_NAME}.amdgpu.gcn ${PARGS_NAME}.nvvm.bc ${PARGS_NAME}.l0.spv ${PARGS_NAME}.cl.spv ${PARGS_NAME}.vk.spv cache)
    add_test(NAME ${PARGS_NAME}_test COMMAND ${PARGS_NAME})
endfunction()

set(BACKEND "nvvm" CACHE STRING "select the backend from the following: NVVM, AMDGPU_HSA, LEVELZERO, ONEAPI, BLAS, HALIDE, AVX")
set_property(CACHE BACKEND PROPERTY STRINGS nvvm cuda amdgpu_hsa levelzero oneapi blas halide avx)
string(TOLOWER ${BACKEND} BACKEND)

option(BENCHMARK "Include benchmarks in test loop.")

if (BACKEND STREQUAL "nvvm")
    add_wmma_test(NAME simple FILES common.art operations/nvvm_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/nvvm_ref.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/nvvm_ref.art layouts/medium.art)

    add_wmma_test(NAME nowmma FILES common.art operations/nvvm_nowmma.art layouts/simple.art)
    add_wmma_test(NAME medium_nowmma FILES common.art operations/nvvm_tiled_nowmma.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/test_*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME} FILES common.art operations/nvvm_blocked.art ${_case})
    endforeach()

    add_wmma_test(NAME large  FILES common.art operations/nvvm_blas.art layouts/large.art)
    add_blas_libraries(large)

    add_wmma_test(NAME large_tiled  FILES common.art operations/nvvm_tiled.art layouts/large.art)
    add_blas_libraries(large_tiled)

    add_wmma_test(NAME large_ref  FILES common.art operations/nvvm_blocked.art layouts/large.art)
    add_blas_libraries(large_ref)

    add_wmma_test(NAME medium_prgmod  FILES common.art operations/prgmod.art layouts/large.art)
    add_blas_libraries(medium_prgmod)

    if (BENCHMARK)
        file(GLOB GENERATED_TESTS generated/bench_*.art)
        foreach(_case ${GENERATED_TESTS})
            get_filename_component(TESTNAME ${_case} NAME_WLE)
            message(STATUS "Generated test ${TESTNAME}")
            add_wmma_test(NAME ${TESTNAME} FILES benchmark.art operations/nvvm_blocked.art ${_case} UTILS benchmark_util.c)
        endforeach()

        add_wmma_test(NAME bench_large_nowmma FILES benchmark.art operations/nvvm_tiled_nowmma.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_tiled FILES benchmark.art operations/nvvm_tiled.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_simple FILES benchmark.art operations/nvvm_ref.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_blocked FILES benchmark.art operations/nvvm_blocked.art layouts/large.art UTILS benchmark_util.c)
    endif()
elseif (BACKEND STREQUAL "cuda")
    add_wmma_test(NAME simple FILES common.art operations/cuda_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/cuda_ref.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/cuda_ref.art layouts/medium.art)

    add_wmma_test(NAME large  FILES common.art operations/cuda_blas.art layouts/large.art)
    add_blas_libraries(large)

    add_wmma_test(NAME large_tiled  FILES common.art operations/cuda_tiled.art layouts/large.art)
    add_blas_libraries(large_tiled)

    add_wmma_test(NAME large_ref  FILES common.art operations/cuda_blocked.art layouts/large.art)
    add_blas_libraries(large_ref)
elseif (BACKEND STREQUAL "amdgpu_hsa")
    add_wmma_test(NAME simple FILES common.art operations/hsa_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/hsa_ref.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/hsa_ref.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/test_*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME} FILES common.art operations/hsa_blocked.art ${_case})
    endforeach()

    add_wmma_test(NAME large  FILES common.art operations/hsa_blas.art layouts/large.art)
    add_blas_libraries(large)

    add_wmma_test(NAME large_tiled  FILES common.art operations/hsa_tiled.art layouts/large.art)
    add_blas_libraries(large_tiled)

    add_wmma_test(NAME large_ref  FILES common.art operations/hsa_blocked.art layouts/large.art)
    add_blas_libraries(large_ref)

    if (BENCHMARK)
        file(GLOB GENERATED_TESTS generated/bench_*.art)
        foreach(_case ${GENERATED_TESTS})
            get_filename_component(TESTNAME ${_case} NAME_WLE)
            message(STATUS "Generated test ${TESTNAME}")
            add_wmma_test(NAME ${TESTNAME} FILES benchmark.art operations/hsa_blocked.art ${_case} UTILS benchmark_util.c)
        endforeach()

        add_wmma_test(NAME bench_large_nowmma FILES benchmark.art operations/hsa_tiled_nowmma.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_tiled FILES benchmark.art operations/hsa_tiled.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_simple FILES benchmark.art operations/hsa_ref.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_blocked FILES benchmark.art operations/hsa_blocked.art layouts/large.art UTILS benchmark_util.c)
    endif()
elseif (BACKEND STREQUAL "levelzero")
    set(LZ_TARGET "alchemist" CACHE STRING "select the target architecture for level zero, options are alchemist and battlemage")
    set_property(CACHE LZ_TARGET PROPERTY STRINGS alchemist battlemage)

    add_wmma_test(NAME simple FILES common.art operations/lzero_ref.art operations/lzero_tiling_${LZ_TARGET}.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/lzero_ref.art operations/lzero_tiling_${LZ_TARGET}.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/lzero_ref.art operations/lzero_tiling_${LZ_TARGET}.art layouts/medium.art)

    add_wmma_test(NAME nowmma FILES common.art operations/lzero_nowmma.art layouts/simple.art)
    add_wmma_test(NAME medium_nowmma FILES common.art operations/lzero_tiled_nowmma.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/test_*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME} FILES common.art operations/lzero_tiled.art operations/lzero_tiling_${LZ_TARGET}.art ${_case})
    endforeach()

    add_wmma_test(NAME large  FILES common.art operations/lzero_blas.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art)
    add_blas_libraries(large)

    add_wmma_test(NAME large_tiled  FILES common.art operations/lzero_tiled.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art)
    add_blas_libraries(large_tiled)

    add_wmma_test(NAME large_ref  FILES common.art operations/lzero_tiled_shmem.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art)
    add_blas_libraries(large_ref)

    add_wmma_test(NAME large_blocked  FILES common.art operations/lzero_blocked.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art)
    add_blas_libraries(large_blocked)

    add_wmma_test(NAME nowmma_large FILES common.art operations/lzero_nowmma_blas.art layouts/large.art)
    add_blas_libraries(nowmma_large)

    add_wmma_test(NAME wmma_large FILES common.art operations/lzero_wmma_blas.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art)
    add_blas_libraries(wmma_large)

    if (BENCHMARK)
        add_wmma_test(NAME bench_large_simple FILES benchmark.art operations/lzero_ref.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_tiled FILES benchmark.art operations/lzero_tiled.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_nowmma FILES benchmark.art operations/lzero_tiled_nowmma.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_blocked FILES benchmark.art operations/lzero_blocked.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art UTILS benchmark_util.c)
        add_wmma_test(NAME bench_large_blocked_noshm FILES benchmark.art operations/lzero_blocked_noshm.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art UTILS benchmark_util.c)

        file(GLOB GENERATED_TESTS generated/bench_*.art)
        if(NOT GENERATED_TESTS)
            message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
        endif()
        foreach(_case ${GENERATED_TESTS})
            get_filename_component(TESTNAME ${_case} NAME_WLE)
            message(STATUS "Generated test ${TESTNAME}_tiled")
            add_wmma_test(NAME ${TESTNAME}_tiled FILES benchmark.art operations/lzero_tiled.art operations/lzero_tiling_${LZ_TARGET}.art ${_case} UTILS benchmark_util.c)
        endforeach()
        foreach(_case ${GENERATED_TESTS})
            get_filename_component(TESTNAME ${_case} NAME_WLE)
            message(STATUS "Generated test ${TESTNAME}_blocked")
            add_wmma_test(NAME ${TESTNAME}_blocked FILES benchmark.art operations/lzero_blocked.art operations/lzero_tiling_${LZ_TARGET}.art ${_case} UTILS benchmark_util.c)
        endforeach()
        foreach(_case ${GENERATED_TESTS})
            get_filename_component(TESTNAME ${_case} NAME_WLE)
            message(STATUS "Generated test ${TESTNAME}_blocked_noshm")
            add_wmma_test(NAME ${TESTNAME}_blocked_noshm FILES benchmark.art operations/lzero_blocked_noshm.art operations/lzero_tiling_${LZ_TARGET}.art ${_case} UTILS benchmark_util.c)
        endforeach()
    endif()
elseif (BACKEND STREQUAL "avx")
    add_wmma_test(NAME simple FILES common.art operations/avx_ref.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/avx_ref.art layouts/layout.art)
    add_wmma_test(NAME layout_inv FILES common.art operations/avx_inv.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/avx_ref.art layouts/medium.art)

    add_wmma_test(NAME dlp FILES common.art operations/avx_dlp.art layouts/medium_avx.art)
    add_wmma_test(NAME ilp FILES common.art operations/avx_ilp.art layouts/medium_avx.art)
    add_wmma_test(NAME mho FILES common.art operations/avx_mho.art layouts/medium_avx.art)
    add_wmma_test(NAME tlp FILES common.art operations/avx_tlp_blas.art layouts/medium_avx.art)
    add_blas_libraries(tlp)

    add_wmma_test(NAME strassen FILES common.art operations/avx_strassen_blas.art layouts/medium_avx.art)
    add_blas_libraries(strassen)

    add_wmma_test(NAME prgmod FILES common.art operations/avx_prgmod.art layouts/medium_prgmod.art)
    add_blas_libraries(prgmod)

    add_wmma_test(NAME large FILES common.art operations/avx_tlp_blas.art layouts/large_avx.art)
    add_blas_libraries(large)

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
elseif (BACKEND STREQUAL "blas")
    add_wmma_test(NAME medium FILES common.art operations/blas.art layouts/medium.art)
    add_blas_libraries(medium)

    add_wmma_test(NAME large FILES common.art operations/blas.art layouts/large.art)
    add_blas_libraries(large)

    if (BENCHMARK)
        add_wmma_test(NAME bench_large FILES benchmark.art operations/blas.art layouts/large.art UTILS benchmark_util.c)
        add_blas_libraries(bench_large)
    endif()
elseif (BACKEND STREQUAL "oneapi")
    add_subdirectory(oneapi)
elseif (BACKEND STREQUAL "halide")
    add_subdirectory(halide)
endif()

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
    set_tests_properties(${PARGS_NAME}_test PROPERTIES TIMEOUT 30)
endfunction()

set(BACKEND "nvvm" CACHE STRING "select the backend from the following: NVVM, AMDGPU_HSA, LEVELZERO, ONEAPI, BLAS, HALIDE, AVX")
set_property(CACHE BACKEND PROPERTY STRINGS nvvm cuda amdgpu_hsa levelzero oneapi blas halide avx)
string(TOLOWER ${BACKEND} BACKEND)

option(BENCHMARK "Include benchmarks in test loop.")

if (BACKEND STREQUAL "nvvm")
    add_wmma_test(NAME simple FILES common.art operations/nvvm_ref.art operations/nvvm_geometry.art operations/ref_naive.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/nvvm_ref.art operations/nvvm_geometry.art operations/ref_naive.art layouts/layout.art)
    #add_wmma_test(NAME medium FILES common.art operations/nvvm_ref.art operations/nvvm_geometry.art operations/ref_naive.art layouts/medium.art)

    add_wmma_test(NAME nowmma FILES common.art operations/nvvm_nowmma.art operations/nvvm_geometry.art operations/ref_naive.art layouts/simple.art)
    add_wmma_test(NAME medium_nowmma FILES common.art operations/nvvm_tiled_nowmma.art operations/nvvm_geometry.art operations/ref_naive.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/test_*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}_rtx-2080_blocked")
        add_wmma_test(NAME ${TESTNAME}_rtx-2080_blocked FILES common.art operations/nvvm_blocked.art operations/ref_blas.art ${_case})
        add_blas_libraries(${TESTNAME}_rtx-2080_blocked)
    endforeach()

    #add_wmma_test(NAME large  FILES common.art operations/nvvm_ref.art operations/nvvm_geometry.art operations/ref_blas.art layouts/large.art)
    #add_blas_libraries(large)

    add_wmma_test(NAME large_tiled  FILES common.art operations/nvvm_tiled.art operations/nvvm_geometry.art operations/ref_blas.art layouts/large.art)
    add_blas_libraries(large_tiled)

    #add_wmma_test(NAME large_ref  FILES common.art operations/nvvm_blocked.art operations/nvvm_geometry.art operations/ref_blas.art layouts/large.art)
    #add_blas_libraries(large_ref)

    #add_wmma_test(NAME medium_prgmod  FILES common.art operations/prgmod.art operations/nvvm_geometry.art operations/ref_blas.art layouts/large.art)
    #add_blas_libraries(medium_prgmod)

    if (BENCHMARK)
        file(GLOB GENERATED_TESTS generated/bench_*.art)
        if(NOT GENERATED_TESTS)
            message(STATUS "No generated benchmarks found. Consider generating with `python generate-cases.py`")
        endif()
        foreach(_case ${GENERATED_TESTS})
            get_filename_component(TESTNAME ${_case} NAME_WLE)
            message(STATUS "Generated test ${TESTNAME}_rtx-2080_blocked")
            add_wmma_test(NAME ${TESTNAME}_rtx-2080_blocked FILES benchmark.art operations/nvvm_blocked.art ${_case} UTILS util/benchmark_util.c)
        endforeach()

        #add_wmma_test(NAME bench_large_nowmma FILES benchmark.art operations/nvvm_tiled_nowmma.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_tiled FILES benchmark.art operations/nvvm_tiled.art operations/nvvm_geometry.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_simple FILES benchmark.art operations/nvvm_ref.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_blocked FILES benchmark.art operations/nvvm_blocked.art layouts/large.art UTILS util/benchmark_util.c)
    endif()
elseif (BACKEND STREQUAL "cuda")
    add_wmma_test(NAME simple FILES common.art operations/cuda_ref.art operations/ref_naive.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/cuda_ref.art operations/ref_naive.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/cuda_ref.art operations/ref_naive.art layouts/medium.art)

    add_wmma_test(NAME large  FILES common.art operations/cuda_ref.art operations/ref_blas.art layouts/large.art)
    add_blas_libraries(large)

    add_wmma_test(NAME large_tiled  FILES common.art operations/cuda_tiled.art operations/ref_blas.art layouts/large.art)
    add_blas_libraries(large_tiled)

    add_wmma_test(NAME large_ref  FILES common.art operations/cuda_blocked.art operations/ref_blas.art layouts/large.art)
    add_blas_libraries(large_ref)
elseif (BACKEND STREQUAL "amdgpu_hsa")
    add_wmma_test(NAME simple FILES common.art operations/hsa_ref.art operations/hsa_geometry.art operations/ref_naive.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/hsa_ref.art operations/hsa_geometry.art operations/ref_naive.art layouts/layout.art)
    #add_wmma_test(NAME medium FILES common.art operations/hsa_ref.art operations/hsa_geometry.art operations/ref_naive.art layouts/medium.art)

    file(GLOB GENERATED_TESTS generated/test_*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}_7900-xtx_blocked")
        add_wmma_test(NAME ${TESTNAME}_7900-xtx_blocked FILES common.art operations/hsa_blocked.art operations/ref_blas.art ${_case})
        add_blas_libraries(${TESTNAME}_7900-xtx_blocked)
    endforeach()

    #add_wmma_test(NAME large  FILES common.art operations/hsa_ref.art operations/hsa_geometry.art operations/ref_blas.art layouts/large.art)
    #add_blas_libraries(large)

    add_wmma_test(NAME large_tiled  FILES common.art operations/hsa_tiled.art operations/hsa_geometry.art operations/ref_blas.art layouts/large.art)
    add_blas_libraries(large_tiled)

    add_wmma_test(NAME large_ref  FILES common.art operations/hsa_blocked.art operations/hsa_geometry.art operations/ref_blas.art layouts/large.art)
    add_blas_libraries(large_ref)

    if (BENCHMARK)
        file(GLOB GENERATED_TESTS generated/bench_*.art)
        if(NOT GENERATED_TESTS)
            message(STATUS "No generated benchmarks found. Consider generating with `python generate-cases.py`")
        endif()
        foreach(_case ${GENERATED_TESTS})
            get_filename_component(TESTNAME ${_case} NAME_WLE)
            message(STATUS "Generated test ${TESTNAME}_7900-xtx_blocked")
            add_wmma_test(NAME ${TESTNAME}_7900-xtx_blocked FILES benchmark.art operations/hsa_blocked.art ${_case} UTILS util/benchmark_util.c)
        endforeach()

        #add_wmma_test(NAME bench_large_nowmma FILES benchmark.art operations/hsa_tiled_nowmma.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_tiled FILES benchmark.art operations/hsa_tiled.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_simple FILES benchmark.art operations/hsa_ref.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_blocked FILES benchmark.art operations/hsa_blocked.art layouts/large.art UTILS util/benchmark_util.c)
    endif()
elseif (BACKEND STREQUAL "levelzero")
    set(LZ_TARGET "alchemist" CACHE STRING "select the target architecture for level zero, options are alchemist and battlemage")
    set_property(CACHE LZ_TARGET PROPERTY STRINGS alchemist battlemage)

    add_wmma_test(NAME simple FILES common.art operations/lzero_ref.art operations/ref_naive.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/simple.art)

    add_wmma_test(NAME large_tiled  FILES common.art operations/lzero_tiled.art operations/ref_blas.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art)
    add_blas_libraries(large_tiled)

    add_wmma_test(NAME large_dpas  FILES common.art operations/lzero_dpas_${LZ_TARGET}.art operations/ref_blas.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art)
    add_blas_libraries(large_dpas)

    add_wmma_test(NAME bench_large_tiled FILES benchmark.art operations/lzero_tiled.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art UTILS util/benchmark_util.c)
    add_wmma_test(NAME bench_large_dpas FILES benchmark.art operations/lzero_dpas_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art UTILS util/benchmark_util.c)

    if (FALSE)

    add_wmma_test(NAME simple FILES common.art operations/lzero_ref.art operations/ref_naive.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/lzero_ref.art operations/ref_naive.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/lzero_ref.art operations/ref_naive.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/medium.art)

    #add_wmma_test(NAME nowmma FILES common.art operations/lzero_nowmma.art operations/ref_naive.art layouts/simple.art)
    #add_wmma_test(NAME medium_nowmma FILES common.art operations/lzero_tiled_nowmma.art operations/ref_naive.art layouts/medium.art)

    add_wmma_test(NAME large  FILES common.art operations/lzero_ref.art operations/ref_blas.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art)
    add_blas_libraries(large)

    add_wmma_test(NAME large_tiled  FILES common.art operations/lzero_tiled.art operations/ref_blas.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art)
    add_blas_libraries(large_tiled)

    add_wmma_test(NAME large_ref  FILES common.art operations/lzero_tiled_shmem.art operations/ref_blas.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art)
    add_blas_libraries(large_ref)

    add_wmma_test(NAME large_blocked  FILES common.art operations/lzero_blocked.art operations/ref_blas.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art)
    add_blas_libraries(large_blocked)

    #add_wmma_test(NAME large_swizzle  FILES common.art operations/lzero_swizzle.art operations/ref_blas.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art)
    #add_blas_libraries(large_swizzle)

    #add_wmma_test(NAME nowmma_large FILES common.art operations/lzero_nowmma.art operations/ref_blas.art layouts/large.art)
    #add_blas_libraries(nowmma_large)

    add_wmma_test(NAME wmma_large FILES common.art operations/lzero_wmma.art operations/ref_blas.art operations/lzero_tiling_${LZ_TARGET}.art operations/lzero_tiling_${LZ_TARGET}_geometry.art layouts/large.art)
    add_blas_libraries(wmma_large)

    #add_wmma_test(NAME nowmma_large_tiled FILES common.art operations/lzero_nowmma_tiled.art operations/ref_blas.art layouts/large.art)
    #add_blas_libraries(nowmma_large_tiled)

    #add_wmma_test(NAME dpas_large FILES common.art operations/lzero_dpas_${LZ_TARGET}.art operations/ref_blas.art layouts/large.art)
    #add_blas_libraries(dpas_large)

    #add_wmma_test(NAME dpas_shmem_large FILES common.art operations/lzero_dpas_shmem_${LZ_TARGET}.art operations/ref_blas.art layouts/large.art)
    #add_blas_libraries(dpas_shmem_large)

    file(GLOB GENERATED_TESTS generated/test_*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)

        #message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_tiled")
        #add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_tiled FILES common.art operations/lzero_tiled.art operations/lzero_tiling_${LZ_TARGET}.art operations/ref_blas.art ${_case} UTILS util/benchmark_util.c)
        #add_blas_libraries(${TESTNAME}_${LZ_TARGET}_tiled)

        #message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_blocked")
        #add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_blocked FILES common.art operations/lzero_blocked.art operations/lzero_tiling_${LZ_TARGET}.art operations/ref_blas.art ${_case} UTILS util/benchmark_util.c)
        #add_blas_libraries(${TESTNAME}_${LZ_TARGET}_blocked)

        #message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_shmem")
        #add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_shmem FILES common.art operations/lzero_tiled_shmem.art operations/lzero_tiling_${LZ_TARGET}.art operations/ref_blas.art ${_case} UTILS util/benchmark_util.c)
        #add_blas_libraries(${TESTNAME}_${LZ_TARGET}_shmem)

        #message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_wmma")
        #add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_wmma FILES common.art operations/lzero_wmma.art operations/lzero_tiling_${LZ_TARGET}.art operations/ref_blas.art ${_case})
        #add_blas_libraries(${TESTNAME}_${LZ_TARGET}_wmma)

        #message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_dpas")
        #add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_dpas FILES common.art operations/lzero_dpas_${LZ_TARGET}.art operations/ref_blas.art ${_case})
        #add_blas_libraries(${TESTNAME}_${LZ_TARGET}_dpas)

        #message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_dpas-shmem")
        #add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_dpas-shmem FILES common.art operations/lzero_dpas_shmem_${LZ_TARGET}.art operations/ref_blas.art ${_case})
        #add_blas_libraries(${TESTNAME}_${LZ_TARGET}_dpas-shmem)

        message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_swizzle")
        add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_swizzle FILES common.art operations/lzero_swizzle.art operations/lzero_tiling_${LZ_TARGET}.art operations/ref_blas.art ${_case})
        add_blas_libraries(${TESTNAME}_${LZ_TARGET}_swizzle)
    endforeach()

    if (BENCHMARK)
        #add_wmma_test(NAME bench_large_simple FILES benchmark.art operations/lzero_ref.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_tiled FILES benchmark.art operations/lzero_tiled.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_nowmma FILES benchmark.art operations/lzero_tiled_nowmma.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_blocked FILES benchmark.art operations/lzero_blocked.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art UTILS util/benchmark_util.c)
        #add_wmma_test(NAME bench_large_blocked_noshm FILES benchmark.art operations/lzero_blocked_noshm.art operations/lzero_tiling_${LZ_TARGET}.art layouts/large.art UTILS util/benchmark_util.c)

        file(GLOB GENERATED_TESTS generated/bench_*.art)
        if(NOT GENERATED_TESTS)
            message(STATUS "No generated benchmarks found. Consider generating with `python generate-cases.py`")
        endif()

        #foreach(_case ${GENERATED_TESTS})
        #    get_filename_component(TESTNAME ${_case} NAME_WLE)
        #    message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_tiled")
        #    add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_tiled FILES benchmark.art operations/lzero_tiled.art operations/lzero_tiling_${LZ_TARGET}.art ${_case} UTILS util/benchmark_util.c)
        #endforeach()

        #foreach(_case ${GENERATED_TESTS})
        #    get_filename_component(TESTNAME ${_case} NAME_WLE)
        #    message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_blocked")
        #    add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_blocked FILES benchmark.art operations/lzero_blocked.art operations/lzero_tiling_${LZ_TARGET}.art ${_case} UTILS util/benchmark_util.c)
        #endforeach()

        #foreach(_case ${GENERATED_TESTS})
        #    get_filename_component(TESTNAME ${_case} NAME_WLE)
        #    message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_shmem")
        #    add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_shmem FILES benchmark.art operations/lzero_tiled_shmem.art operations/lzero_tiling_${LZ_TARGET}.art ${_case} UTILS util/benchmark_util.c)
        #endforeach()

        #foreach(_case ${GENERATED_TESTS})
        #    get_filename_component(TESTNAME ${_case} NAME_WLE)
        #    message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_wmma")
        #    add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_wmma FILES benchmark.art operations/lzero_wmma.art operations/lzero_tiling_${LZ_TARGET}.art ${_case} UTILS util/benchmark_util.c)
        #endforeach()

        #foreach(_case ${GENERATED_TESTS})
        #    get_filename_component(TESTNAME ${_case} NAME_WLE)
        #    message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_dpas")
        #    add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_dpas FILES benchmark.art operations/lzero_dpas_${LZ_TARGET}.art ${_case} UTILS util/benchmark_util.c)
        #endforeach()

        #foreach(_case ${GENERATED_TESTS})
        #    get_filename_component(TESTNAME ${_case} NAME_WLE)
        #    message(STATUS "Generated test ${TESTNAME}_${LZ_TARGET}_dpas-shmem")
        #    add_wmma_test(NAME ${TESTNAME}_${LZ_TARGET}_dpas-shmem FILES benchmark.art operations/lzero_dpas_shmem_${LZ_TARGET}.art ${_case} UTILS util/benchmark_util.c)
        #endforeach()
    endif()

    endif()
elseif (BACKEND STREQUAL "avx")
    add_wmma_test(NAME simple FILES common.art operations/avx_ref.art operations/ref_naive.art layouts/simple.art)
    add_wmma_test(NAME layout FILES common.art operations/avx_ref.art operations/ref_naive.art layouts/layout.art)
    add_wmma_test(NAME layout_inv FILES common.art operations/avx_inv.art operations/ref_naive.art layouts/layout.art)
    add_wmma_test(NAME medium FILES common.art operations/avx_ref.art operations/ref_naive.art layouts/medium.art)

    add_wmma_test(NAME dlp FILES common.art operations/avx_dlp.art operations/ref_naive.art layouts/medium_avx.art)
    add_wmma_test(NAME ilp FILES common.art operations/avx_ilp.art operations/ref_naive.art layouts/medium_avx.art)
    add_wmma_test(NAME mho FILES common.art operations/avx_mho.art operations/ref_naive.art layouts/medium_avx.art)
    add_wmma_test(NAME tlp FILES common.art operations/avx_tlp.art operations/ref_blas.art layouts/medium_avx.art)
    add_blas_libraries(tlp)

    add_wmma_test(NAME strassen FILES common.art operations/avx_strassen.art operations/ref_blas.art layouts/medium_avx.art)
    add_blas_libraries(strassen)

    add_wmma_test(NAME prgmod FILES common.art operations/avx_prgmod.art operations/ref_blas.art layouts/medium_prgmod.art)
    add_blas_libraries(prgmod)

    add_wmma_test(NAME large FILES common.art operations/avx_tlp.art operations/ref_blas.art layouts/large_avx.art)
    add_blas_libraries(large)

    file(GLOB GENERATED_TESTS generated/*.art)
    if(NOT GENERATED_TESTS)
        message(STATUS "No generated tests found. Consider generating with `python generate-cases.py`")
    endif()
    foreach(_case ${GENERATED_TESTS})
        get_filename_component(TESTNAME ${_case} NAME_WLE)
        message(STATUS "Generated test ${TESTNAME}")
        add_wmma_test(NAME ${TESTNAME}_ref FILES common.art operations/avx_ref.art operations/ref_naive.art ${_case})
        add_wmma_test(NAME ${TESTNAME}_inv FILES common.art operations/avx_inv.art operations/ref_naive.art ${_case})
    endforeach()
elseif (BACKEND STREQUAL "blas")
    add_wmma_test(NAME medium FILES common.art operations/blas.art layouts/medium.art)
    add_blas_libraries(medium)

    add_wmma_test(NAME large FILES common.art operations/blas.art layouts/large.art)
    add_blas_libraries(large)

    if (BENCHMARK)
        add_wmma_test(NAME bench_large FILES benchmark.art operations/blas.art layouts/large.art UTILS util/benchmark_util.c)
        add_blas_libraries(bench_large)
    endif()
elseif (BACKEND STREQUAL "oneapi")
    add_subdirectory(oneapi)
elseif (BACKEND STREQUAL "halide")
    add_subdirectory(halide)
endif()

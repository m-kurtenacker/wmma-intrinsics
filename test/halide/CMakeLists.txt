include(FetchContent)
FetchContent_Declare(
        Halide
        GIT_REPOSITORY https://github.com/halide/Halide
        GIT_TAG main
)
FetchContent_MakeAvailable(Halide)
FetchContent_GetProperties(Halide SOURCE_DIR Halide_SRC)

anydsl_runtime_wrap(HALIDE_OBJ
    FRONTEND "artic"
    NAME halide
    FILES ${BACKEND_FILES} halide.art
)

add_halide_generator(matmul_generator SOURCES halide_generator.cpp)
#add_halide_library(matmul FROM matmul_generator GENERATOR matmul_generator SCHEDULE matmul_generator_schedule)
#add_halide_library(matmul FROM matmul_generator GENERATOR matmul_generator AUTOSCHEDULER Halide::Adams2019 SCHEDULE matmul_generator_schedule)
add_halide_library(matmul FROM matmul_generator GENERATOR matmul_generator SCHEDULE matmul_generator_schedule FEATURES cuda)
#add_halide_library(matmul FROM matmul_generator GENERATOR matmul_generator AUTOSCHEDULER Halide::Li2018 SCHEDULE matmul_generator_schedule FEATURES cuda)

MESSAGE(STATUS "Schedule will be written to ${matmul_generator_schedule}")

add_executable(halide ${HALIDE_OBJ} halide.cpp matmul)
target_link_libraries(halide ${AnyDSL_runtime_LIBRARIES} Halide::Halide matmul)
target_include_directories(halide PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${Halide_SRC}/tutorial ${Halide_SRC}/tools)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(halide
        "-framework Metal"
        "-framework MetalKit"
        "-framework AppKit"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

add_test(NAME halide_test COMMAND halide)

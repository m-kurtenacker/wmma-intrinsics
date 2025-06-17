include(FetchContent)
FetchContent_Declare(
        Halide
        GIT_REPOSITORY https://github.com/halide/Halide
        GIT_TAG main
)
FetchContent_MakeAvailable(Halide)

anydsl_runtime_wrap(HALIDE_OBJ
    FRONTEND "artic"
    NAME halide
    FILES ${BACKEND_FILES} halide.art
)

add_executable(matmul_generator halide_generator.cpp)
target_link_libraries(matmul_generator Halide::Halide)
add_custom_command(OUTPUT matmul.a COMMAND matmul_generator DEPENDS halide_generator.cpp)

add_executable(halide ${HALIDE_OBJ} halide.cpp matmul.a)
target_link_libraries(halide ${AnyDSL_runtime_LIBRARIES} Halide::Halide ${CMAKE_CURRENT_BINARY_DIR}/matmul.a)
target_include_directories(halide PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME halide_test COMMAND halide)

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

add_executable(halide ${HALIDE_OBJ} halide.cpp)
target_link_libraries(halide ${AnyDSL_runtime_LIBRARIES} Halide::Halide)

add_test(NAME halide_test COMMAND halide)

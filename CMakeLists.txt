cmake_minimum_required(VERSION 3.25)

project (Test)

find_package (AnyDSL_runtime REQUIRED)
include_directories(${AnyDSL_runtime_INCLUDE_DIRS})

anydsl_runtime_wrap(TEST_OBJ
    FRONTEND "artic"
    FILES test.art intrinsics_wmma.art
)

add_executable(test ${TEST_OBJ})
target_link_libraries(test ${AnyDSL_runtime_LIBRARIES})

set_property(TARGET test
    APPEND PROPERTY ADDITIONAL_CLEAN_FILES test.nvvm test.nvvm.bc cache)

add_test(NAME test COMMAND test)

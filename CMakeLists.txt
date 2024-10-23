cmake_minimum_required(VERSION 3.25)

project (Test)

find_package (AnyDSL_runtime REQUIRED)
include_directories(${AnyDSL_runtime_INCLUDE_DIRS})

file(GLOB BACKEND_FILES
    backend/*.art
)

include(CTest)
add_subdirectory(test)

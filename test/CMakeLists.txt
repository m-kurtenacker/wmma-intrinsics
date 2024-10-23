anydsl_runtime_wrap(SIMPLE_OBJ
    FRONTEND "artic"
    FILES ${BACKEND_FILES} simple.art
)

add_executable(simple ${SIMPLE_OBJ})
target_link_libraries(simple ${AnyDSL_runtime_LIBRARIES})

set_property(TARGET simple
    APPEND PROPERTY ADDITIONAL_CLEAN_FILES simple.nvvm simple.nvvm.bc cache)

add_test(NAME simple_test COMMAND simple)

anydsl_runtime_wrap(MEDIUM_OBJ
    FRONTEND "artic"
    FILES ${BACKEND_FILES} medium.art
)

add_executable(medium ${MEDIUM_OBJ})
target_link_libraries(medium ${AnyDSL_runtime_LIBRARIES})

set_property(TARGET medium
    APPEND PROPERTY ADDITIONAL_CLEAN_FILES medium.nvvm medium.nvvm.bc cache)

add_test(NAME medium_test COMMAND medium)

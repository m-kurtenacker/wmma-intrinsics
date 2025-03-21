set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)

add_wmma_test(NAME oneapi FILES oneapi.art)
add_wmma_test(NAME oneapi_half FILES oneapi_half.art)
add_wmma_test(NAME oneapi_double FILES oneapi_double.art)

target_link_libraries(oneapi ${BLAS_LIBRARIES})
target_link_libraries(oneapi_half ${BLAS_LIBRARIES})
target_link_libraries(oneapi_double ${BLAS_LIBRARIES})

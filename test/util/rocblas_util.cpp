#include <assert.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include "rocblas/rocblas.h"

rocblas_handle handle;
hipStream_t stream;
bool created = false;

extern "C" {

//#[import(cc="C", name="cblas_sgemm")]
//fn cblas_sgemm (_layout : i32, _transpose_a : i32, _transpose_b: i32, _m : i64, _n : i64, _k : i64, _alpha : f32, _a : &mut [f32], _stride_a : i64, _b : &mut [f32], _stride_b : i64, _beta : f32, _c : &mut [f32], _stride_c : i64) -> ();
void cblas_sgemm(int layout, int transpose_a, int transpose_b, size_t m, size_t n, size_t k, float alpha, float * a, size_t stride_a, float * b, size_t stride_b, float beta, float * c, size_t stride_c) {
    if (!created) {
        rocblas_create_handle(&handle);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        rocblas_get_stream(handle, &stream);
        created = true;
    }

    //assert(layout == 102); //Important: cublas only supports col-major for accu matrix.

    rocblas_sgemm(
            handle,
            transpose_a == 111 ? rocblas_operation_none : rocblas_operation_transpose,
            transpose_b == 111 ? rocblas_operation_none : rocblas_operation_transpose,
            m,
            n,
            k,
            &alpha,
            a,
            stride_a,
            b,
            stride_b,
            &beta,
            c,
            stride_c);

    hipStreamSynchronize(stream);
}

//#[import(cc="C", name="cblas_hgemm")]
//fn cblas_hgemm_cuda (_layout : i32, _transpose_a : i32, _transpose_b: i32, _m : i64, _n : i64, _k : i64, _alpha : i16, _a : &mut [f16], _stride_a : i64, _b : &mut [f16], _stride_b : i64, _beta : i16, _c : &mut [f16], _stride_c : i64) -> ();
//Clang turns float-16 parameters into int-16! (The pointers are untyped anyways, so they don't change.)
void cblas_hgemm(int layout, int transpose_a, int transpose_b, size_t m, size_t n, size_t k, rocblas_half alpha, rocblas_half * a, size_t stride_a, rocblas_half * b, size_t stride_b, rocblas_half beta, rocblas_half * c, size_t stride_c) {
    if (!created) {
        rocblas_create_handle(&handle);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        rocblas_get_stream(handle, &stream);
        created = true;
    }

    assert(layout == 102); //Important: cublas only supports col-major for accu matrix.

    rocblas_hgemm(
            handle,
            transpose_a == 111 ? rocblas_operation_none : rocblas_operation_transpose,
            transpose_b == 111 ? rocblas_operation_none : rocblas_operation_transpose,
            m,
            n,
            k,
            &alpha,
            a,
            stride_a,
            b,
            stride_b,
            &beta,
            c,
            stride_c);

    hipStreamSynchronize(stream);
}

//#[import(cc="C", name="cblas_scopy")]
//fn cblas_scopy(_n : i32, _x : &[f32], incx : i32, _y : &mut [f32], incy : i32) -> ();
void cblas_scopy(int n, float * x, int incx, float * y, int incy) {
    if (!created) {
        rocblas_create_handle(&handle);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        rocblas_get_stream(handle, &stream);
        created = true;
    }

    rocblas_scopy(
            handle,
            n,
            x,
            incx,
            y,
            incy);

    hipStreamSynchronize(stream);
}

}

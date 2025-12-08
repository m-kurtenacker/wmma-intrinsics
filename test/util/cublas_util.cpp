#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

cublasHandle_t handle;
bool created = false;

extern "C" {

//#[import(cc="C", name="cblas_sgemm")]
//fn cblas_sgemm (_layout : i32, _transpose_a : i32, _transpose_b: i32, _m : i64, _n : i64, _k : i64, _alpha : f32, _a : &mut [f32], _stride_a : i64, _b : &mut [f32], _stride_b : i64, _beta : f32, _c : &mut [f32], _stride_c : i64) -> ();
void cblas_sgemm(int layout, int transpose_a, int transpose_b, size_t m, size_t n, size_t k, float alpha, float * a, size_t stride_a, float * b, size_t stride_b, float beta, float * c, size_t stride_c) {
    if (!created) {
        cublasCreate(&handle);
        created = true;
    }

    assert(layout == 102); //Important: cublas only supports col-major for accu matrix.

    cublasSgemm(
            handle,
            transpose_a == 111 ? CUBLAS_OP_N : CUBLAS_OP_T,
            transpose_b == 111 ? CUBLAS_OP_N : CUBLAS_OP_T,
            n,
            m,
            k,
            &alpha,
            a,
            stride_a,
            b,
            stride_b,
            &beta,
            c,
            stride_c);
}

//#[import(cc="C", name="cblas_hgemm")]
//fn cblas_hgemm_cuda (_layout : i32, _transpose_a : i32, _transpose_b: i32, _m : i64, _n : i64, _k : i64, _alpha : i16, _a : &mut [f16], _stride_a : i64, _b : &mut [f16], _stride_b : i64, _beta : i16, _c : &mut [f16], _stride_c : i64) -> ();
//Clang turns float-16 parameters into int-16! (The pointers are untyped anyways, so they don't change.)
void cblas_hgemm(int layout, int transpose_a, int transpose_b, size_t m, size_t n, size_t k, __half alpha, __half * a, size_t stride_a, __half * b, size_t stride_b, __half beta, __half * c, size_t stride_c) {
    if (!created) {
        cublasCreate(&handle);
        created = true;
    }

    assert(layout == 102); //Important: cublas only supports col-major for accu matrix.

    cublasHgemm(
            handle,
            transpose_a == 111 ? CUBLAS_OP_N : CUBLAS_OP_T,
            transpose_b == 111 ? CUBLAS_OP_N : CUBLAS_OP_T,
            n,
            m,
            k,
            &alpha,
            a,
            stride_a,
            b,
            stride_b,
            &beta,
            c,
            stride_c);
}

//#[import(cc="C", name="cblas_scopy")]
//fn cblas_scopy(_n : i32, _x : &[f32], incx : i32, _y : &mut [f32], incy : i32) -> ();
void cblas_scopy(int n, float * x, int incx, float * y, int incy) {
    if (!created) {
        cublasCreate(&handle);
        created = true;
    }

    cublasScopy(
            handle,
            n,
            x,
            incx,
            y,
            incy);
}

}

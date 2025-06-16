
#include "Halide.h"

using namespace Halide;

extern "C" {

//#import[(cc = "C", name = "halide_alloc")] fn halide_alloc(i32, i32) -> &mut [i8];
void * halide_alloc (int size_x, int size_y) {
    auto buffer = new Buffer<float>({size_x, size_y});
    return static_cast<void *>(buffer);
}

//#import[(cc = "C", name = "halide_copy_from_buffer")] halide_copy_from_buffer(_src : &[i8], _dst : &mut [i8]) -> ();
void halide_copy_from_buffer (void * src, void * dst) {
    Buffer<float> * src_buffer = static_cast<Buffer<float>*>(src);
    float * dst_buffer = static_cast<float*>(dst);

    for (int x = 0; x < src_buffer->dim(0).extent(); x++) {
        for (int y = 0; y < src_buffer->dim(1).extent(); y++) {
            dst_buffer[x * src_buffer->dim(1).extent() + y] = (*src_buffer)(x, y);
        }
    }
}


//#import[(cc = "C", name = "halide_copy_to_buffer")] halide_copy_to_buffer(_src : &[i8], _dst : &mut [i8]) -> ();
void halide_copy_to_buffer (void * src, void * dst) {
    float * src_buffer = static_cast<float*>(src);
    Buffer<float> * dst_buffer = static_cast<Buffer<float>*>(dst);

    for (int x = 0; x < dst_buffer->dim(0).extent(); x++) {
        for (int y = 0; y < dst_buffer->dim(1).extent(); y++) {
            (*dst_buffer)(x, y) = src_buffer[x * dst_buffer->dim(1).extent() + y];
        }
    }
}


//#import[(cc = "C", name = "halide_matmul")] fn halide_matmul(_a : &[i8], _b : &[i8], _c : &[i8], _d : &mut [i8]) -> ();
void halide_matmul(void * a, void * b, void * c, void * d) {
    ImageParam A(Float(32), 2);
    ImageParam B(Float(32), 2);
    ImageParam C(Float(32), 2);

    Var x, y;
    Func matmul;

    matmul(x, y) = C(x, y);

    RDom k(0, A.dim(1).extent());
    matmul(x, y) += A(x, k) * B(k, y);


    Buffer<float> * a_buffer = static_cast<Buffer<float>*>(a);
    Buffer<float> * b_buffer = static_cast<Buffer<float>*>(b);
    Buffer<float> * c_buffer = static_cast<Buffer<float>*>(c);
    Buffer<float> * d_buffer = static_cast<Buffer<float>*>(d);

    int N = a_buffer->dim(0).extent();
    int M = b_buffer->dim(1).extent();

    A.set(*a_buffer);
    B.set(*b_buffer);
    C.set(*c_buffer);

    Buffer<float> output = matmul.realize({N, M});

    for (int x = 0; x < d_buffer->dim(0).extent(); x++) {
        for (int y = 0; y < d_buffer->dim(1).extent(); y++) {
            (*d_buffer)(x, y) = output(x, y);
        }
    }
}

} /* extern "C" */

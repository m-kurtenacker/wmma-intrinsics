
#include "Halide.h"
#include "matmul.h"

#include "clock.h"

using namespace Halide;

extern "C" {

//#import[(cc = "C", name = "halide_alloc")] fn halide_alloc(i32, i32) -> &mut [i8];
void * halide_alloc (int size_x, int size_y) {
    auto buffer = new Runtime::Buffer<float>(size_x, size_y);

    buffer->set_host_dirty();

    return static_cast<void *>(buffer);
}

//#import[(cc = "C", name = "halide_copy_from_buffer")] halide_copy_from_buffer(_src : &[i8], _dst : &mut [i8]) -> ();
void halide_copy_from_buffer (void * src, void * dst) {
    Runtime::Buffer<float> * src_buffer = static_cast<Runtime::Buffer<float>*>(src);
    float * dst_buffer = static_cast<float*>(dst);

    if (src_buffer->device_dirty())
        src_buffer->copy_to_host();

    for (int x = 0; x < src_buffer->dim(0).extent(); x++) {
        for (int y = 0; y < src_buffer->dim(1).extent(); y++) {
            dst_buffer[x * src_buffer->dim(1).extent() + y] = (*src_buffer)(x, y);
        }
    }
}


//#import[(cc = "C", name = "halide_copy_to_buffer")] halide_copy_to_buffer(_src : &[i8], _dst : &mut [i8]) -> ();
void halide_copy_to_buffer (void * src, void * dst) {
    float * src_buffer = static_cast<float*>(src);
    Runtime::Buffer<float> * dst_buffer = static_cast<Runtime::Buffer<float>*>(dst);

    for (int x = 0; x < dst_buffer->dim(0).extent(); x++) {
        for (int y = 0; y < dst_buffer->dim(1).extent(); y++) {
            (*dst_buffer)(x, y) = src_buffer[x * dst_buffer->dim(1).extent() + y];
        }
    }
}


double halide_matmul(void * a, void * b, void * c, void * d) {
    Runtime::Buffer<float> * a_buffer = static_cast<Runtime::Buffer<float>*>(a);
    Runtime::Buffer<float> * b_buffer = static_cast<Runtime::Buffer<float>*>(b);
    Runtime::Buffer<float> * c_buffer = static_cast<Runtime::Buffer<float>*>(c);
    Runtime::Buffer<float> * d_buffer = static_cast<Runtime::Buffer<float>*>(d);

    double t1 = current_time();

    int error = matmul(*a_buffer, *b_buffer, *c_buffer, *d_buffer);
    if (d_buffer->device_dirty())
        halide_device_sync(nullptr, *d_buffer);

    double t2 = current_time();

    assert(error == 0);

    return (t2 - t1) * 1000; //Time in microseconds.
}

} /* extern "C" */

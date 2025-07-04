#include <mma.h>
using namespace nvcuda;

typedef wmma::fragment<wmma::accumulator, 16, 16, 16, half> cuda_acc_datatype;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> cuda_mat_a_datatype;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> cuda_mat_b_datatype;

typedef         __half_raw f16;

extern "C" {

__device__ void cuda_load_matrix_sync_c(cuda_acc_datatype* fragment, f16* data, int stride) {
    wmma::load_matrix_sync(*fragment, (half*) data, stride, wmma::mem_row_major);
}

__device__ void cuda_store_matrix_sync(f16* data, cuda_acc_datatype fragment, int stride) {
    wmma::store_matrix_sync((half*) data, fragment, stride, wmma::mem_row_major);
}

__device__ void cuda_load_matrix_sync_a(cuda_mat_a_datatype* fragment, f16* data, int stride) {
    wmma::load_matrix_sync(*fragment, (half*) data, stride);
}

__device__ void cuda_load_matrix_sync_b(cuda_mat_b_datatype* fragment, f16* data, int stride) {
    wmma::load_matrix_sync(*fragment, (half*) data, stride);
}

__device__ void cuda_mma_sync(cuda_acc_datatype* acc, cuda_mat_a_datatype a, cuda_mat_b_datatype b, cuda_acc_datatype c) {
    wmma::mma_sync(*acc, a, b, c);
}

}

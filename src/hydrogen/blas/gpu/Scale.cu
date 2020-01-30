#include <hydrogen/blas/gpu/Scale.hpp>

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>
#include <hydrogen/device/gpu/CUDA.hpp>

#include <cuda_runtime.h>

namespace
{

template <typename T, typename SizeT>
__global__ void scale_1d_kernel_naive(
    SizeT num_entries, T alpha, T* A, SizeT stride_A)
{
    SizeT const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_entries)
        A[idx*stride_A] *= alpha;
}

template <int TILE_DIM, int BLK_COLS, typename T, typename SizeT>
__global__ void scale_2d_kernel_naive(
    SizeT m, SizeT n, T alpha, T* A, SizeT lda)
{
    size_t const row_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t const col_idx = blockIdx.y * TILE_DIM + threadIdx.y;

    if (row_idx < m)
    {
        for (int ii = 0; ii < TILE_DIM && col_idx + ii < n; ii += BLK_COLS)
            A[row_idx + (col_idx+ii)*lda] *= alpha;
    }
}

}// namespace <anon>

namespace hydrogen
{

template <typename T, typename SizeT, typename>
void Scale_GPU_impl(
    SizeT num_entries,
    T const& alpha, T* A, SizeT lda,
    cudaStream_t stream)
{
    if (!num_entries)
        return;

    constexpr size_t threads_per_block = 128;
    auto blocks = (num_entries + threads_per_block - 1)/ threads_per_block;
    T arg_alpha = alpha;
    void* args[] = { &num_entries, &arg_alpha, &A, &lda};
    H_CHECK_CUDA(
        cudaLaunchKernel(
            (void const*)&scale_1d_kernel_naive<T,SizeT>,
            blocks, threads_per_block, args, 0, stream));
}

template <typename T, typename SizeT, typename>
void Scale_GPU_impl(
    SizeT num_rows, SizeT num_cols,
    T const& alpha, T* A, SizeT lda,
    cudaStream_t stream)
{
    if (num_rows == TypeTraits<SizeT>::Zero()
        || num_cols == TypeTraits<SizeT>::Zero())
    {
        return;
    }

    constexpr int TILE_DIM = 32;
    constexpr int BLK_COLS = 8;

    dim3 blks((num_rows + TILE_DIM - 1) / TILE_DIM,
              (num_cols + TILE_DIM - 1) / TILE_DIM,
              1);
    dim3 thds(TILE_DIM, BLK_COLS, 1);

    T arg_alpha = alpha;
    void* args[] = { &num_rows, &num_cols,
                     &arg_alpha, &A, &lda};
    H_CHECK_CUDA(
        cudaLaunchKernel(
            (void const*)&scale_2d_kernel_naive<TILE_DIM,BLK_COLS,T,SizeT>,
            blks, thds, args, 0, stream));
}

#define ETI(DataType, SizeType)                         \
    template void Scale_GPU_impl(                       \
        SizeType,                                       \
        DataType const&, DataType*, SizeType,           \
        cudaStream_t);                                  \
    template void Scale_GPU_impl(                       \
        SizeType, SizeType,                             \
        DataType const&, DataType*, SizeType,           \
        cudaStream_t)

ETI(float, int);
ETI(float, long);
ETI(float, long long);
ETI(float, unsigned);
ETI(float, size_t);

ETI(double, int);
ETI(double, long);
ETI(double, long long);
ETI(double, unsigned);
ETI(double, size_t);

#ifdef HYDROGEN_GPU_USE_FP16
ETI(gpu_half_type, int);
ETI(gpu_half_type, long);
ETI(gpu_half_type, long long);
ETI(gpu_half_type, unsigned);
ETI(gpu_half_type, size_t);
#endif

}// namespace hydrogen

#include <hydrogen/blas/gpu/Fill.hpp>

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>
#include <hydrogen/device/gpu/CUDA.hpp>

#include <cuda_runtime.h>

namespace hydrogen
{
namespace
{

template <typename T>
__global__ void Fill1D_kernel(size_t size, T value, T* buffer)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < size; pos += numThreads)
    {
        buffer[pos] = value;
    }
}

template <typename T>
__global__ void Fill2D_kernel(size_t height, size_t width, T value,
                              T* buffer, size_t ldim)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t numThreads = blockDim.x * gridDim.x;
    for (size_t pos = tid; pos < height * width; pos += numThreads)
    {
        const size_t i = pos % height;
        const size_t j = pos / height;
        buffer[i+j*ldim] = value;
    }
}

template <typename T>
bool CompareEqual(T const& a, T const& b)
{
    return a == b;
}

#ifdef HYDROGEN_GPU_USE_FP16
inline bool CompareEqual(gpu_half_type const& a, gpu_half_type const& b)
{
    return float(a) == float(b);
}
#endif // HYDROGEN_GPU_USE_FP16

}// namespace <anon>

template <typename T, typename>
void Fill_GPU_impl(
    size_t height, size_t width, T const& value,
    T* buffer, size_t ldim, cudaStream_t stream)
{
    if (height <= 0 || width <= 0)
        return;

    size_t size = height * width;
    constexpr size_t blockDim = 256;
    const size_t gridDim = (size + blockDim - 1) / blockDim;
    if (CompareEqual(value, TypeTraits<T>::Zero()))
    {
        if (width == 1 || ldim == height)
        {
            H_CHECK_CUDA(cudaMemsetAsync(buffer, 0x0, size*sizeof(T),
                                         stream));
        }
        else
        {
            H_CHECK_CUDA(
                cudaMemset2DAsync(
                    buffer, ldim*sizeof(T), 0x0,
                    height*sizeof(T), width,
                    stream));
        }
    }
    else
    {
        T arg_value = value;
        if (width == 1 || ldim == height)
        {
            void* args[] = {&size, &arg_value, &buffer};
            H_CHECK_CUDA(
                cudaLaunchKernel(
                    (void const*)&Fill1D_kernel<T>,
                    gridDim, blockDim, args, 0, stream));

        }
        else
        {
            void* args[] = {&height, &width, &arg_value, &buffer, &ldim};
            H_CHECK_CUDA(
                cudaLaunchKernel(
                    (void const*)&Fill2D_kernel<T>,
                    gridDim, blockDim, args, 0, stream));
        }
    }

}

#define ETI(T)                                                          \
    template void Fill_GPU_impl(                                        \
        size_t, size_t, T const&, T*, size_t, cudaStream_t)

#ifdef HYDROGEN_GPU_USE_FP16
ETI(gpu_half_type);
#endif
ETI(float);
ETI(double);

}// namespace hydrogen

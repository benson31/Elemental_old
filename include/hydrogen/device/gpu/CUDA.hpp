#ifndef HYDROGEN_IMPORTS_CUDA_HPP_
#define HYDROGEN_IMPORTS_CUDA_HPP_

#include <El/hydrogen_config.h>

#include <hydrogen/Device.hpp>
#include <hydrogen/utils/HalfPrecision.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hydrogen
{

/** @class CudaError
 *
 *  Exception class for CUDA errors.
 *
 *  \todo Clean up the error-handling macros
 */
struct CudaError : std::runtime_error
{
    std::string build_error_string_(
        cudaError_t cuda_error, char const* file, int line, bool async = false)
    {
        std::ostringstream oss;
        oss << ( async ? "Asynchronous CUDA error" : "CUDA error" )
            << " (error code=" << cuda_error << ") (" << file << ":" << line << "): "
            << cudaGetErrorString(cuda_error);
        return oss.str();
    }
    CudaError(cudaError_t cuda_error, char const* file, int line, bool async = false)
        : std::runtime_error{build_error_string_(cuda_error,file,line,async)}
    {}
}; // struct CudaError

#define H_CUDA_SYNC(async)                                             \
    do                                                                  \
    {                                                                   \
        /* Synchronize GPU and check for errors. */                     \
        cudaError_t status_CUDA_SYNC = cudaDeviceSynchronize();         \
        if (status_CUDA_SYNC == cudaSuccess)                            \
            status_CUDA_SYNC = cudaGetLastError();                      \
        if (status_CUDA_SYNC != cudaSuccess) {                          \
            cudaDeviceReset();                                          \
            throw hydrogen::CudaError(status_CUDA_SYNC,__FILE__,__LINE__,async);  \
        }                                                               \
    }                                                                   \
    while( 0 )
#define H_FORCE_CHECK_CUDA(cuda_call)                                  \
    do                                                                  \
    {                                                                   \
        /* Call CUDA API routine, synchronizing before and after to */  \
        /* check for errors. */                                         \
        H_CUDA_SYNC(true);                                             \
        cudaError_t status_CHECK_CUDA = cuda_call ;                     \
        if( status_CHECK_CUDA != cudaSuccess ) {                        \
            cudaDeviceReset();                                          \
            throw hydrogen::CudaError(status_CHECK_CUDA,__FILE__,__LINE__,false); \
        }                                                               \
        H_CUDA_SYNC(false);                                            \
    } while (0)
#define H_FORCE_CHECK_CUDA_NOSYNC(cuda_call)                           \
    do                                                                  \
    {                                                                   \
        /* Call CUDA API routine, and check for errors without */       \
        /* synchronizing. */                                            \
        cudaError_t status_CHECK_CUDA = cuda_call ;                     \
        if( status_CHECK_CUDA != cudaSuccess ) {                        \
            cudaDeviceReset();                                          \
            throw hydrogen::CudaError(status_CHECK_CUDA,__FILE__,__LINE__,false); \
        }                                                               \
    } while (0)
#define H_LAUNCH_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args)      \
    do                                                          \
    {                                                           \
        /* Dg is a dim3 specifying grid dimensions. */          \
        /* Db is a dim3 specifying block dimensions. */         \
        /* Ns is a size_t specifying dynamic memory. */         \
        /* S is a cudaStream_t specifying stream. */            \
        kernel <<< Dg, Db, Ns, S >>> args ;                     \
    }                                                           \
    while (0)
#define H_FORCE_CHECK_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args) \
    do                                                          \
    {                                                           \
        /* Launch CUDA kernel, synchronizing before */          \
        /* and after to check for errors. */                    \
        H_CUDA_SYNC(true);                                     \
        H_LAUNCH_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args);     \
        H_CUDA_SYNC(false);                                    \
    }                                                           \
    while (0)

#ifdef HYDROGEN_RELEASE_BUILD
#define H_CHECK_CUDA( cuda_call ) H_FORCE_CHECK_CUDA_NOSYNC(cuda_call)
#define H_CHECK_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args) \
  H_LAUNCH_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args)
#else
#define H_CHECK_CUDA( cuda_call ) H_FORCE_CHECK_CUDA( cuda_call )
#define H_CHECK_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args) \
  H_FORCE_CHECK_CUDA_KERNEL(kernel, Dg, Db, Ns, S, args)
#endif // HYDROGEN_RELEASE_BUILD

// Function to determine if a pointer is GPU memory
inline bool IsGPUMemory(const void* ptr)
{
    cudaPointerAttributes attrs;
    auto err = cudaPointerGetAttributes(&attrs, ptr);
    if (err == cudaErrorInvalidValue)
    {
        if ((err = cudaGetLastError()) == cudaErrorInvalidValue)
            return false;
        else
            H_FORCE_CHECK_CUDA(err);
    }
    else
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if ((err = cudaGetLastError()) == cudaSuccess)
            return (attrs.memoryType == cudaMemoryTypeDevice);
        else
            H_FORCE_CHECK_CUDA(err);
#pragma GCC diagnostic pop
    }
    return false;// silence compiler warning
}

/** Initialize CUDA environment.
 *  We assume that all MPI ranks within a compute node have access to
 *  exactly one unique GPU or to the same (possibly empty) list of
 *  GPUs. GPU assignments can be controled with the
 *  CUDA_VISIBLE_DEVICES environment variable.
 */
void InitializeCUDA(int,char*[]);
/** Finalize CUDA environment. */
void FinalizeCUDA();

/** Singleton class to manage CUDA objects.
 *  This class also manages cuBLAS objects. Note that the CUDA device
 *  is set whenever the singleton instance is requested, i.e. in most
 *  of the static functions.
 */
class GPUManager
{
public:

    GPUManager( const GPUManager& ) = delete;
    GPUManager& operator=( const GPUManager& ) = delete;
    ~GPUManager();

    /** Create new singleton instance of CUDA manager. */
    static void Create( int device = 0 );
    /** Initilize CUBLAS. */
    static void InitializeCUBLAS();
    /** Destroy singleton instance of CUDA manager. */
    static void Destroy();
    /** Get singleton instance of CUDA manager. */
    static GPUManager* Instance();
    /** Get number of visible CUDA devices. */
    static unsigned int NumDevices();
    /** Get currently active CUDA device. */
    static int Device();
    /** Set active CUDA device. */
    static void SetDevice( int device );
    /** Get CUDA stream. */
    static cudaStream_t Stream();
    /** Get CUDA event. */
    static cudaEvent_t Event();
    /** Synchronize CUDA stream. */
    static void SynchronizeStream();
    /** Synchronize CUDA device.
     *  If checkError is true, an exception will be thrown if an error
     *  from an asynchronous CUDA kernel is detected.
     */
    static void SynchronizeDevice( bool checkError = false );
    /** Get cuBLAS handle. */
    static cublasHandle_t cuBLASHandle();

private:

    /** Singleton instance. */
    static std::unique_ptr<GPUManager> instance_;

    /** Number of visible CUDA devices. */
    unsigned int numDevices_;
    /** Currently active CUDA device. */
    int device_;
    /** CUDA stream. */
    cudaStream_t stream_;
    /** CUDA event. */
    cudaEvent_t event_;
    /** cuBLAS handle */
    cublasHandle_t cublasHandle_;

    GPUManager( int device = 0 );

}; // class GPUManager

template <Device D1, Device D2>
constexpr cudaMemcpyKind CUDAMemcpyKind();

template <>
constexpr cudaMemcpyKind CUDAMemcpyKind<Device::CPU,Device::GPU>()
{
    return cudaMemcpyHostToDevice;
}

template <>
constexpr cudaMemcpyKind CUDAMemcpyKind<Device::GPU,Device::CPU>()
{
    return cudaMemcpyDeviceToHost;
}

template <>
constexpr cudaMemcpyKind CUDAMemcpyKind<Device::GPU,Device::GPU>()
{
    return cudaMemcpyDeviceToDevice;
}

template <>
struct InterDeviceCopy<Device::CPU,Device::GPU>
{
    template <typename T>
    static void MemCopy1DAsync(
        T * __restrict__ const dest,
        T const* __restrict__ const src,
        size_t const size,
        cudaStream_t stream = GPUManager::Stream())
    {
        H_CHECK_CUDA(
            cudaMemcpyAsync(
                dest, src, size*sizeof(T),
                CUDAMemcpyKind<Device::CPU,Device::GPU>(),
                stream));
    }

#if defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)
    // These two types are bitwise-compatible across the two devices.
    static void MemCopy1DAsync(gpu_half_type * __restrict__ const dest,
                               cpu_half_type const* __restrict__ const src,
                               size_t const size,
                               cudaStream_t stream = GPUManager::Stream())
    {
        H_CHECK_CUDA(
            cudaMemcpyAsync(
                dest, src, size*sizeof(gpu_half_type),
                CUDAMemcpyKind<Device::CPU,Device::GPU>(),
                stream));
    }

    static void MemCopy1DAsync(
        cpu_half_type * __restrict__ const dest,
        gpu_half_type const* __restrict__ const src,
        size_t const size,
        cudaStream_t stream = GPUManager::Stream())
    {
        H_CHECK_CUDA(
            cudaMemcpyAsync(
                dest, src, size*sizeof(gpu_half_type),
                CUDAMemcpyKind<Device::CPU,Device::GPU>(),
                stream));
    }
#endif // defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)

    template <typename T>
    static void MemCopy2DAsync(
        T * __restrict__ const dest, size_t const dest_ldim,
        T const* __restrict__ const src,
        size_t const src_ldim,
        size_t const height, size_t const width,
        cudaStream_t stream = GPUManager::Stream())
    {
        H_CHECK_CUDA(
            cudaMemcpy2DAsync(
                dest, dest_ldim*sizeof(T),
                src, src_ldim*sizeof(T),
                height*sizeof(T), width,
                CUDAMemcpyKind<Device::CPU,Device::GPU>(),
                stream));
    }

#if defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)
    // These two types are bitwise-compatible across the two devices.
    static void MemCopy2DAsync(
        gpu_half_type * __restrict__ const dest,
        size_t const dest_ldim,
        cpu_half_type const* __restrict__ const src,
        size_t const src_ldim,
        size_t const height, size_t const width,
        cudaStream_t stream = GPUManager::Stream())
    {
        H_CHECK_CUDA(
            cudaMemcpy2DAsync(
                dest, dest_ldim*sizeof(gpu_half_type),
                src, src_ldim*sizeof(cpu_half_type),
                height*sizeof(gpu_half_type), width,
                CUDAMemcpyKind<Device::CPU,Device::GPU>(),
                stream));
    }
    static void MemCopy2DAsync(
        cpu_half_type * __restrict__ const dest,
        size_t const dest_ldim,
        gpu_half_type const* __restrict__ const src,
        size_t const src_ldim,
        size_t const height, size_t const width,
        cudaStream_t stream = GPUManager::Stream())
    {
        H_CHECK_CUDA(cudaMemcpy2DAsync(
            dest, dest_ldim*sizeof(cpu_half_type),
            src, src_ldim*sizeof(gpu_half_type),
            height*sizeof(gpu_half_type), width,
            CUDAMemcpyKind<Device::CPU,Device::GPU>(),
            stream));
    }
#endif // defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)
};// InterDevice<CPU,GPU>

template <>
struct InterDeviceCopy<Device::GPU,Device::CPU>
{
    template <typename T>
    static void MemCopy1DAsync(
        T * __restrict__ const dest,
        T const* __restrict__ const src, size_t const size,
        cudaStream_t stream = GPUManager::Stream())
    {
        H_CHECK_CUDA(
            cudaMemcpyAsync(
                dest, src, size*sizeof(T),
                CUDAMemcpyKind<Device::GPU,Device::CPU>(),
                stream));
    }

    template <typename T>
    static void MemCopy2DAsync(
        T * __restrict__ const dest, size_t const dest_ldim,
        T const* __restrict__ const src, size_t const src_ldim,
        size_t const height, size_t const width,
        cudaStream_t stream = GPUManager::Stream())
    {
        H_CHECK_CUDA(
            cudaMemcpy2DAsync(
                dest, dest_ldim*sizeof(T),
                src, src_ldim*sizeof(T),
                height*sizeof(T), width,
                CUDAMemcpyKind<Device::GPU,Device::CPU>(),
                stream));
    }
};// InterDevice<CPU,GPU>

} // namespace hydrogen

#endif // HYDROGEN_IMPORTS_CUDA_HPP_

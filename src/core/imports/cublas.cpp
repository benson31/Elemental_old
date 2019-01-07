
#include "El-lite.hpp"
#include "El/core/imports/cuda.hpp"
#include "El/core/imports/cublas.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef USE_MAGMABLAS_GEMM
#include <magma.h>
#endif

namespace El
{

void InitializeCUBLAS()
{
    GPUManager::InitializeCUBLAS();
}
namespace cublas
{

namespace
{

inline cublasOperation_t CharTocuBLASOp(char c)
{
    switch (c)
    {
    case 'N':
        return CUBLAS_OP_N;
    case 'T':
        return CUBLAS_OP_T;
    case 'C':
        return CUBLAS_OP_C;
    default:
        RuntimeError("cuBLAS: Unknown operation type.");
        return CUBLAS_OP_N; // Compiler yells about not returning anything...
    }
}

} // namespace <anon>

//
// BLAS 1
//
#define ADD_AXPY_IMPL(ScalarType, TypeChar)           \
    void Axpy(int n, ScalarType const& alpha,         \
              ScalarType const* X, int incx,          \
              ScalarType* Y, int incy)                \
    {                                                 \
        EL_CHECK_CUBLAS(cublas ## TypeChar ## axpy(   \
            GPUManager::cuBLASHandle(),               \
            n, &alpha, X, incx, Y, incy));            \
    }

#define ADD_COPY_IMPL(ScalarType, TypeChar)           \
    void Copy(int n, ScalarType const* X, int incx,   \
              ScalarType* Y, int incy)                \
    {                                                 \
        EL_CHECK_CUBLAS(cublas ## TypeChar ## copy(   \
            GPUManager::cuBLASHandle(),               \
            n, X, incx, Y, incy));                    \
    }

//
// BLAS 2
//
#define ADD_GEMV_IMPL(ScalarType, TypeChar)                     \
    void Gemv(                                                  \
        char transA, int m, int n,                              \
        ScalarType const& alpha,                                \
        ScalarType const* A, int ALDim,                         \
        ScalarType const* B, int BLDim,                         \
        ScalarType const& beta,                                 \
        ScalarType* C, int CLDim )                              \
    {                                                           \
        EL_CHECK_CUBLAS(cublas ## TypeChar ## gemv(             \
            GPUManager::cuBLASHandle(),                         \
            CharTocuBLASOp(transA),                             \
            m, n, &alpha, A, ALDim, B, BLDim, &beta, C, CLDim));\
    }

//
// BLAS 3
//
#ifndef USE_MAGMABLAS_GEMM
#define ADD_GEMM_IMPL(ScalarType, TypeChar)                             \
    void Gemm(                                                          \
        char transA, char transB, int m, int n, int k,                  \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const* B, int BLDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
         EL_CHECK_CUBLAS(cublas ## TypeChar ## gemm(                    \
            GPUManager::cuBLASHandle(),                                 \
            CharTocuBLASOp(transA), CharTocuBLASOp(transB),             \
            m, n, k, &alpha, A, ALDim, B, BLDim, &beta, C, CLDim));     \
    }
#else

namespace
{

struct magma_queue_wrapper
{
    magma_queue_t queue_;
    operator magma_queue_t() const noexcept { return queue_; }

    magma_queue_wrapper() noexcept
    {
        magma_queue_create(GPUManager::Device(), &queue_);
    }

    ~magma_queue_wrapper()
    {
        magma_queue_destroy(queue_);
    }
};// struct magma_queue_wrapper

magma_trans_t CharToMAGMAOp(char a) noexcept
{
    switch (a)
    {
    case 'T':
    case 't':
        return MagmaTrans;
    case 'H':
    case 'h':
    case 'C':
    case 'c':
        return MagmaConjTrans;
    }

    return MagmaNoTrans;
}

} // namespace anon

magma_queue_wrapper magma_queue_;

#define ADD_GEMM_IMPL(ScalarType, TypeChar)                     \
    void Gemm(                                                  \
        char transA, char transB, int m, int n, int k,          \
        ScalarType const& alpha,                                \
        ScalarType const* A, int ALDim,                         \
        ScalarType const* B, int BLDim,                         \
        ScalarType const& beta,                                 \
        ScalarType* C, int CLDim)                               \
    {                                                           \
    cudaStream_t stream;                                        \
    EL_CHECK_CUBLAS(                                            \
        cublasGetStream(GPUManager::cuBLASHandle(), &stream));  \
    SyncInfo<Device::GPU> master{                               \
        magma_queue_get_cuda_stream(magma_queue_),              \
        GPUManager::Event()},                                   \
    other{stream, GPUManager::Event()};                         \
    auto multisync = MakeMultiSync(master, other);              \
                                                                \
    magmablas_ ## TypeChar ## gemm(                             \
        CharToMAGMAOp(transA), CharToMAGMAOp(transB),           \
        m, n, k, alpha, A, ALDim, B, BLDim, beta, C, CLDim,     \
        magma_queue_);                                          \
}
#endif // USE_MAGMABLAS_GEMM

//
// BLAS-like Extension
//
#define ADD_GEAM_IMPL(ScalarType, TypeChar)                             \
    void Geam(                                                          \
        char transA, char transB,                                       \
        int m, int n,                                                   \
        ScalarType const& alpha,                                        \
        ScalarType const* A, int ALDim,                                 \
        ScalarType const& beta,                                         \
        ScalarType const* B, int BLDim,                                 \
        ScalarType* C, int CLDim )                                      \
    {                                                                   \
       EL_CHECK_CUBLAS(cublas ## TypeChar ## geam(                      \
            GPUManager::cuBLASHandle(),                                 \
            CharTocuBLASOp(transA), CharTocuBLASOp(transB),             \
            m, n, &alpha, A, ALDim, &beta, B, BLDim, C, CLDim));        \
    }

// BLAS 1
ADD_AXPY_IMPL(float, S)
ADD_AXPY_IMPL(double, D)

ADD_COPY_IMPL(float, S)
ADD_COPY_IMPL(double, D)

// BLAS 2
ADD_GEMV_IMPL(float, S)
ADD_GEMV_IMPL(double, D)

// BLAS 3
#ifndef USE_MAGMABLAS_GEMM
ADD_GEMM_IMPL(float, S)
ADD_GEMM_IMPL(double, D)
#else
ADD_GEMM_IMPL(float, s)
ADD_GEMM_IMPL(double, d)
#endif

// BLAS-like extension
ADD_GEAM_IMPL(float, S)
ADD_GEAM_IMPL(double, D)

} // namespace cublas

} // namespace El

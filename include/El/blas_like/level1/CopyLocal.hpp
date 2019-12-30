#ifndef EL_BLAS_LIKE_LEVEL1_COPYLOCAL_HPP_
#define EL_BLAS_LIKE_LEVEL1_COPYLOCAL_HPP_

#ifdef _OPENMP
#include <omp.h>
#endif

#include "El/blas_like/level1/EntrywiseMap.hpp"

#include <hydrogen/Device.hpp>
#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/blas/GPU_BLAS.hpp>
#endif

namespace El
{

// There are the following cases:
//
// 0. The source or target has a type that is not self-compatible.
//    -> LogicError
//
// 1. The source and target have the same type on the same device.
//    -> Simple copy operation
//
// 2. The source and target have the same type but on different devices.
//    -> Simple interdevice copy operation (reduces to device-vendor API call)
//
// 3. The source and target have different types but the same binary
//    layout on different devices.
//    -> Reinterpret cast and simple interdevice copy operation.
//
// 4. The source and target have different types but on the same device.
//    -> Casting copy.
//
// 5. The source and target have different types on different devices.
//    -> Unclear if we can implement this directly by hand; may have
//       to use a temporary to convert to the appropriate device/type
//       and then to the appropriate type/device, resp.

// LEGACY: Handle the cast-less case on CPU.
// (Case 1, CPU)
template <typename T,
          EnableWhen<IsStorageType<T, Device::CPU>, int> = 0>
void CopyImpl(Matrix<T, Device::CPU> const& A, Matrix<T, Device::CPU>& B)
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    const Int size = height * width;
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
          T* EL_RESTRICT BBuf = B.Buffer();

    if (ldA == height && ldB == height)
    {
// #ifdef _OPENMP
// #if defined(HYDROGEN_HAVE_OMP_TASKLOOP)
//         const Int numThreads = omp_get_num_threads();
//         #pragma omp taskloop default(shared)
//         for (Int thread = 0; thread < numThreads; ++thread)
//         {
// #else
//         #pragma omp parallel
//         {
//             const Int numThreads = omp_get_num_threads();
//             const Int thread = omp_get_thread_num();
// #endif // defined(HYDROGEN_HAVE_OMP_TASKLOOP)
//             const Int chunk = (size + numThreads - 1) / numThreads;
//             const Int start = Min(chunk * thread, size);
//             const Int end = Min(chunk * (thread + 1), size);
//             MemCopy(&BBuf[start], &ABuf[start], end - start);
//         }
// #else
        MemCopy(BBuf, ABuf, size);
//#endif // _OPENMP
    }
    else
    {
        EL_PARALLEL_FOR
        for (Int j=0; j<width; ++j)
        {
            MemCopy(&BBuf[j*ldB], &ABuf[j*ldA], height);
        }
    }
}
//}// DELETE ME

// LEGACY: Generic implementation of casting on the CPU
// (Case 4, CPU)
template<typename S, typename T,
         EnableWhen<And<CanCast<S,T>,
                        IsStorageType<S, Device::CPU>,
                        IsStorageType<T, Device::CPU>>, int> = 0>
void CopyImpl(Matrix<S, Device::CPU> const& A, Matrix<T, Device::CPU>& B)
{
    EL_DEBUG_CSE;
    EntrywiseMap(A, B, MakeFunction(Caster<S,T>::Cast));
}

#ifdef HYDROGEN_HAVE_GPU
// Inter-type copy on the GPU. The gpu_blas can handle this via a
// custom kernel.
// (Case 4, GPU)
template <typename T, typename U,
          EnableWhen<And<IsStorageType<T, Device::GPU>,
                         IsStorageType<U, Device::GPU>>, int> = 0>
void CopyImpl(Matrix<T, Device::GPU> const& A, Matrix<U, Device::GPU>& B)
{
    EL_DEBUG_CSE;
    Int const height = A.Height();
    Int const width = A.Width();
    B.Resize(height, width);
    Int const ldA = A.LDim();
    Int const ldB = B.LDim();
    T const* ABuf = A.LockedBuffer();
    U* BBuf = B.Buffer();

    SyncInfo<Device::GPU> syncInfoA = SyncInfoFromMatrix(A),
        syncInfoB = SyncInfoFromMatrix(B);
    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    gpu_blas::Copy(TransposeMode::NORMAL,
                   height, width,
                   ABuf, ldA,
                   BBuf, ldB,
                   syncInfoB);
}

#ifdef HYDROGEN_HAVE_CUDA
// If using CUDA, prefer the cudaMemcpy2D implementation. This is
// ASYNCHRONOUS with respect to the host.
// (Case 1, GPU)
//
// TODO: Profile to verify this is, indeed, faster.
template <typename T,
          EnableWhen<IsStorageType<T, Device::CPU>, int> = 0>
void CopyImpl(Matrix<T, Device::GPU> const& A, Matrix<T, Device::GPU>& B)
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* ABuf = A.LockedBuffer();
    T* BBuf = B.Buffer();

    SyncInfo<Device::GPU> syncInfoA = SyncInfoFromMatrix(A),
        syncInfoB = SyncInfoFromMatrix(B);
    auto syncHelper = MakeMultiSync(syncInfoB, syncInfoA);

    // Launch the copy
    H_CHECK_CUDA(
        cudaMemcpy2DAsync(BBuf, ldB*sizeof(T),
                          ABuf, ldA*sizeof(T),
                          height*sizeof(T), width,
                          cudaMemcpyDeviceToDevice,
                          syncInfoB.stream_));
}
#endif // HYDROGEN_HAVE_CUDA

// These inter-device copy functions are SYNCHRONOUS with respect to
// the host.
// (Case 2, CPU->GPU)
template <typename T,
          EnableWhen<And<IsStorageType<T, Device::CPU>,
                         IsStorageType<T, Device::GPU>>, int> = 0>
void CopyImpl(Matrix<T, Device::CPU> const& A, Matrix<T, Device::GPU>& B)
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    T* EL_RESTRICT BBuf = B.Buffer();

    SyncInfo<Device::GPU> syncInfoB = SyncInfoFromMatrix(B);
    InterDeviceCopy<Device::CPU,Device::GPU>::MemCopy2DAsync(
        BBuf, ldB, ABuf, ldA, height, width, syncInfoB.stream_);
    Synchronize(syncInfoB); // Is this necessary??
}

// These inter-device copy functions are SYNCHRONOUS with respect to
// the host.
// (Case 2, CPU->GPU)
template <typename T,
          EnableWhen<And<IsStorageType<T, Device::GPU>,
                         IsStorageType<T, Device::CPU>>, int> = 0>
void CopyImpl(Matrix<T,Device::GPU> const& A, Matrix<T,Device::CPU>& B)
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    T* EL_RESTRICT BBuf = B.Buffer();

    SyncInfo<Device::GPU> syncInfoA = SyncInfoFromMatrix(A);
    InterDeviceCopy<Device::GPU, Device::CPU>::MemCopy2DAsync(
        BBuf, ldB, ABuf, ldA, height, width, syncInfoA.stream_);
    Synchronize(syncInfoA); // Is this necessary??
}

// These inter-device copy functions are SYNCHRONOUS with respect to
// the host.
// (Case 3, CPU->GPU)
template <typename T,
          EnableWhen<And<IsStorageType<T, Device::CPU>,
                         Not<IsSame<T, details::GPUStorageType<T>>>>, int> = 0>
void CopyImpl(Matrix<T, Device::CPU> const& A,
              Matrix<details::GPUStorageType<T>, Device::GPU>& B)
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    const T* EL_RESTRICT ABuf = A.LockedBuffer();
    // This is "safe" because T and GPUStoragetype<T> are bitwise-equivalent.
    T* EL_RESTRICT BBuf = reinterpret_cast<T*>(B.Buffer());

    SyncInfo<Device::GPU> syncInfoB = SyncInfoFromMatrix(B);
    InterDeviceCopy<Device::CPU, Device::GPU>::MemCopy2DAsync(
        BBuf, ldB, ABuf, ldA, height, width, syncInfoB.stream_);
    Synchronize(syncInfoB); // Is this necessary??
}

// These inter-device copy functions are SYNCHRONOUS with respect to
// the host.
// (Case 3, CPU->GPU)
template <typename T,
          EnableWhen<And<IsStorageType<T, Device::CPU>,
                         Not<IsSame<T, details::GPUStorageType<T>>>>, int> = 0>
void CopyImpl(Matrix<details::GPUStorageType<T>, Device::GPU> const& A,
              Matrix<T,Device::CPU>& B)
{
    EL_DEBUG_CSE;
    const Int height = A.Height();
    const Int width = A.Width();
    B.Resize(height, width);
    const Int ldA = A.LDim();
    const Int ldB = B.LDim();
    // This is "safe" because T and GPUStoragetype<T> are bitwise-equivalent.
    const T* EL_RESTRICT ABuf = reinterpret_cast<T const*>(A.LockedBuffer());
    T* EL_RESTRICT BBuf = B.Buffer();

    SyncInfo<Device::GPU> syncInfoA = SyncInfoFromMatrix(A);
    InterDeviceCopy<Device::GPU,Device::CPU>::MemCopy2DAsync(
        BBuf, ldB, ABuf, ldA, height, width, syncInfoA.stream_);
    Synchronize(syncInfoA); // Is this necessary??
}

#endif // HYDROGEN_HAVE_GPU

// Inter-device and inter-type Copy.
// (Case 5)
template <typename T, typename U, Device D1, Device D2,
          EnableWhen<And<BoolVT<D1!=D2>,
                         IsStorageType<T, D1>,
                         IsStorageType<U, D2>>, int> = 0>
void CopyImpl(Matrix<T, D1> const& src, Matrix<U, D2>& tgt)
{
    // FIXME: Note that there might be instances in which it could
    // feasibly be better to change type and then change device. In
    // any case, none of this is ideal; aiming for correctness now.
    using T_on_D2 = details::CompatibleStorageType<T, D2>;
    Matrix<T_on_D2, D2> tmp;
    CopyImpl(src, tmp); // Change device
    CopyImpl(tmp, tgt); // Change type
}

// (Case 0)
template <typename T, typename U, Device D1, Device D2,
          EnableWhen<Or<Not<IsStorageType<T,D1>>,
                        Not<IsStorageType<U,D2>>>, int> = 0>
void CopyImpl(Matrix<T, D1> const&, Matrix<U, D2>&)
{
    LogicError("Cannot dispatch Copy.");
}

// "Plain-ol' function"
template <typename T, typename U, Device D1, Device D2>
void Copy(Matrix<T, D1> const& src, Matrix<U, D2>& tgt)
{
    CopyImpl(src, tgt);
}

}// namespace El
#endif // EL_BLAS_LIKE_LEVEL1_COPYLOCAL_HPP_

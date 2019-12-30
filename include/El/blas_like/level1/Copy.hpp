/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_HPP
#define EL_BLAS_COPY_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include <El/blas_like/level1/Copy/internal_decl.hpp>
#include <El/blas_like/level1/Copy/GeneralPurpose.hpp>
#include <El/blas_like/level1/Copy/util.hpp>

#include <hydrogen/meta/MetaUtilities.hpp>

// Introduce some metaprogramming notions.
//
// TODO: Move elsewhere.
namespace El
{

template <bool B>
using BoolVT = std::integral_constant<bool, B>;

namespace details
{

/** @brief A simple metafunction for interoping bitwise-equivalent
 *         types across device interfaces.
 */
template <typename T, Device D>
struct CompatibleStorageTypeT
{
    using type = T;
};

template <typename T, Device D>
using CompatibleStorageType = typename CompatibleStorageTypeT<T, D>::type;

#if defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)

template <>
struct CompatibleStorageTypeT<cpu_half_type, El::Device::GPU>
{
    using type = gpu_half_type;
};

#endif // defined(HYDROGEN_HAVE_HALF) && defined(HYDROGEN_GPU_USE_FP16)

template <typename T>
using CPUStorageType = CompatibleStorageType<T, Device::CPU>;

#ifdef HYDROGEN_HAVE_GPU
template <typename T>
using GPUStorageType = CompatibleStorageType<T, Device::GPU>;
#endif
}// namespace details
}// namespace El

//
// Include all the definitions
//
#include "CopyLocal.hpp"
#include "CopyAsyncLocal.hpp"
#include "CopyDistMatrix.hpp"
#include "CopyAsyncDistMatrix.hpp"
#include "CopyFromRoot.hpp"

#if 0
namespace El
{

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T)                                                        \
    EL_EXTERN template void Copy(                                       \
        AbstractDistMatrix<T> const& A,                                 \
        AbstractDistMatrix<T>& B);                                      \
    EL_EXTERN template void CopyFromRoot(                               \
        Matrix<T> const& A,                                             \
        DistMatrix<T,CIRC,CIRC>& B,                                     \
        bool includingViewers);                                         \
    EL_EXTERN template void CopyFromNonRoot(                            \
        DistMatrix<T,CIRC,CIRC>& B,                                     \
        bool includingViewers);                                         \
    EL_EXTERN template void CopyFromRoot(                               \
        Matrix<T> const& A,                                             \
        DistMatrix<T,CIRC,CIRC,BLOCK>& B,                               \
        bool includingViewers);                                         \
    EL_EXTERN template void CopyFromNonRoot(                            \
        DistMatrix<T,CIRC,CIRC,BLOCK>& B,                               \
        bool includingViewers);                                         \
    EL_EXTERN template void CopyAsync(                                  \
        AbstractDistMatrix<T> const& A,                                 \
        AbstractDistMatrix<T>& B);

#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGINT
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

#undef EL_EXTERN

} // namespace El
#endif // 0

#endif // ifndef EL_BLAS_COPY_HPP

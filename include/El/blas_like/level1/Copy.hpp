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

#ifdef HYDROGEN_HAVE_GPU
#include <hydrogen/blas/GPU_BLAS.hpp>
#endif

#include "CopyFunctionsThatDoStuff.hpp"

namespace El {

//
// AbstractMatrix dispatch
//

// Moved elsewhere!

//
// DistMatrix stuff
//

template<typename T,Dist U,Dist V,Device D,
         typename = EnableIf<IsDeviceValidType<T,D>>>
void Copy(const ElementalMatrix<T>& A,
          DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE;
    B = A;
}

template <typename T,Dist U,Dist V,Device D,
          typename = DisableIf<IsDeviceValidType<T,D>>,
          typename = void>
void Copy(const ElementalMatrix<T>& A,
          DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE;
    LogicError("Copy: bad data/device combination.");
}

// Datatype conversions should not be very common, and so it is likely best to
// avoid explicitly instantiating every combination
template <typename S,typename T,Dist U,Dist V,Device D,
          typename = EnableIf<IsDeviceValidType<T,D>>>
void Copy(const ElementalMatrix<S>& A, DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE;
    if (A.Grid() == B.Grid() && A.ColDist() == U && A.RowDist() == V
        && A.GetLocalDevice() == D)
    {
        if (!B.RootConstrained())
            B.SetRoot(A.Root());
        if (!B.ColConstrained())
            B.AlignCols(A.ColAlign());
        if (!B.RowConstrained())
            B.AlignRows(A.RowAlign());
        if (A.Root() == B.Root() &&
            A.ColAlign() == B.ColAlign() && A.RowAlign() == B.RowAlign())
        {
            B.Resize(A.Height(), A.Width());
            Copy(A.LockedMatrix(), B.Matrix());
            return;
        }
    }
    DistMatrix<S,U,V,ELEMENT,D> BOrig(A.Grid());
    BOrig.AlignWith(B);
    BOrig = A;
    B.Resize(A.Height(), A.Width());
    Copy(BOrig.LockedMatrix(), B.Matrix());
}

template <typename S,typename T,Dist U,Dist V,Device D,
          typename=DisableIf<IsDeviceValidType<T,D>>,
          typename=void>
void Copy(const ElementalMatrix<S>& A, DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE;
    LogicError("Copy: bad data/device combination.");
}

template <typename T,Dist U,Dist V>
void Copy(const BlockMatrix<T>& A, DistMatrix<T,U,V,BLOCK>& B)
{
    EL_DEBUG_CSE;
    B = A;
}

// Datatype conversions should not be very common, and so it is likely best to
// avoid explicitly instantiating every combination
template <typename S,typename T,Dist U,Dist V>
void Copy(const BlockMatrix<S>& A, DistMatrix<T,U,V,BLOCK>& B)
{
    EL_DEBUG_CSE;
    if (A.Grid() == B.Grid() && A.ColDist() == U && A.RowDist() == V)
    {
        if (!B.RootConstrained())
            B.SetRoot(A.Root());
        if (!B.ColConstrained())
            B.AlignColsWith(A.DistData());
        if (!B.RowConstrained())
            B.AlignRowsWith(A.DistData());
        if (A.Root() == B.Root() &&
            A.ColAlign() == B.ColAlign() &&
            A.RowAlign() == B.RowAlign() &&
            A.ColCut() == B.ColCut() &&
            A.RowCut() == B.RowCut())
        {
            B.Resize(A.Height(), A.Width());
            Copy(A.LockedMatrix(), B.Matrix());
            return;
        }
    }
    DistMatrix<S,U,V,BLOCK> BOrig(A.Grid());
    BOrig.AlignWith(B);
    BOrig = A;
    B.Resize(A.Height(), A.Width());
    Copy(BOrig.LockedMatrix(), B.Matrix());
}

template <typename S,typename T,
          typename/*=EnableIf<CanCast<S,T>>*/>
void Copy(const ElementalMatrix<S>& A, ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE;
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                                        \
        (B.ColDist() == CDIST) && (B.RowDist() == RDIST)                \
            && (B.Wrap() == WRAP) && (B.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                                \
        auto& BCast =                                                   \
            static_cast<DistMatrix<T,CDIST,RDIST,ELEMENT,DEVICE>&>(B);  \
        Copy(A, BCast);
    #include <El/macros/DeviceGuardAndPayload.h>
}

template <typename T>
void Copy(const AbstractDistMatrix<T>& A, AbstractDistMatrix<T>& B)
{
    EL_DEBUG_CSE;
    const DistWrap wrapA=A.Wrap(), wrapB=B.Wrap();
    if (wrapA == ELEMENT && wrapB == ELEMENT)
    {
        auto& ACast = static_cast<const ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        Copy(ACast, BCast);
    }
    else if (wrapA == BLOCK && wrapB == BLOCK)
    {
        auto& ACast = static_cast<const BlockMatrix<T>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        Copy(ACast, BCast);
    }
    else
    {
        copy::GeneralPurpose(A, B);
    }
}

template <typename T, Dist U, Dist V, Device D1, Device D2>
void CopyAsync(DistMatrix<T,U,V,ELEMENT,D1> const& A,
               DistMatrix<T,U,V,ELEMENT,D2>& B)
{
    EL_DEBUG_CSE;
#ifndef EL_RELEASE
    auto const Adata = A.DistData(), Bdata = B.DistData();
    if (!((Adata.blockHeight == Bdata.blockHeight) &&
          (Adata.blockWidth == Bdata.blockWidth) &&
          (Adata.colAlign == Bdata.colAlign) &&
          (Adata.rowAlign == Bdata.rowAlign) &&
          (Adata.colCut == Bdata.colCut) &&
          (Adata.rowCut == Bdata.rowCut) &&
          (Adata.root == Bdata.root) &&
          (Adata.grid == Bdata.grid)))
    {
        LogicError("CopyAsync: "
                   "A and B must have the same DistData, except device.");
    }
#endif // !defined(EL_RELEASE)
    B.Resize(A.Height(), A.Width());
    CopyAsync(A.LockedMatrix(), B.Matrix());
}

template <typename T, Dist U, Dist V, Device D>
void CopyAsync(DistMatrix<T,U,V,ELEMENT,D> const& A,
               DistMatrix<T,U,V,ELEMENT,D>& B)
{
    LogicError("CopyAsync: Both matrices on same device (D=",
               DeviceName<D>(), ").");
}

template <typename T, Dist U, Dist V, Device D>
void CopyAsync(ElementalMatrix<T> const& A, DistMatrix<T,U,V,ELEMENT,D>& B)
{
    EL_DEBUG_CSE;
    if ((A.ColDist() == U) && (A.RowDist() == V))
    {
        switch (A.GetLocalDevice())
        {
        case Device::CPU:
            CopyAsync(
                static_cast<DistMatrix<T,U,V,ELEMENT,Device::CPU> const&>(A),
                B);
            break;
#ifdef HYDROGEN_HAVE_CUDA
        case Device::GPU:
            CopyAsync(
                static_cast<DistMatrix<T,U,V,ELEMENT,Device::GPU> const&>(A),
                B);
            break;
#endif // HYDROGEN_HAVE_CUDA
        default:
            LogicError("CopyAsync: Unknown device type.");
        }
    }
    else
        LogicError("CopyAsync requires A and B to have the same distribution.");
}

template <typename T>
void CopyAsync(ElementalMatrix<T> const& A, ElementalMatrix<T>& B)
{
    EL_DEBUG_CSE;
#define GUARD(CDIST,RDIST,WRAP,DEVICE)                              \
    (B.ColDist() == CDIST) && (B.RowDist() == RDIST)                \
        && (B.Wrap() == WRAP) && (B.GetLocalDevice() == DEVICE)
#define PAYLOAD(CDIST,RDIST,WRAP,DEVICE)                            \
    auto& BCast =                                                   \
        static_cast<DistMatrix<T,CDIST,RDIST,ELEMENT,DEVICE>&>(B);  \
    CopyAsync(A, BCast);
    #include <El/macros/DeviceGuardAndPayload.h>
}


template <typename T>
void CopyAsync(AbstractDistMatrix<T> const& A, AbstractDistMatrix<T>& B)
{
    EL_DEBUG_CSE;
    const DistWrap wrapA = A.Wrap(), wrapB = B.Wrap();
    if (wrapA == ELEMENT && wrapB == ELEMENT)
    {
        auto& ACast = static_cast<const ElementalMatrix<T>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        CopyAsync(ACast, BCast);
    }
    else
        LogicError("CopyAsync only implemented for ElementalMatrix.");
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy(const AbstractDistMatrix<S>& A, AbstractDistMatrix<T>& B)
{
    EL_DEBUG_CSE;
    const DistWrap wrapA=A.Wrap(), wrapB=B.Wrap();
    if (wrapA == ELEMENT && wrapB == ELEMENT)
    {
        auto& ACast = static_cast<const ElementalMatrix<S>&>(A);
        auto& BCast = static_cast<ElementalMatrix<T>&>(B);
        Copy(ACast, BCast);
    }
    else if (wrapA == BLOCK && wrapB == BLOCK)
    {
        auto& ACast = static_cast<const BlockMatrix<S>&>(A);
        auto& BCast = static_cast<BlockMatrix<T>&>(B);
        Copy(ACast, BCast);
    }
    else
    {
        LogicError("If you see this error, please tell Tom.");
        copy::GeneralPurpose(A, B);
    }
}

template<typename S,typename T,
         typename/*=EnableIf<CanCast<S,T>>*/>
void Copy(const BlockMatrix<S>& A, BlockMatrix<T>& B)
{
    EL_DEBUG_CSE;
    #define GUARD(CDIST,RDIST,WRAP) \
        B.ColDist() == CDIST && B.RowDist() == RDIST && B.Wrap() == WRAP \
            && B.GetLocalDevice() == Device::CPU
    #define PAYLOAD(CDIST,RDIST,WRAP) \
      auto& BCast = static_cast<DistMatrix<T,CDIST,RDIST,BLOCK>&>(B); \
      Copy(A, BCast);
    #include <El/macros/GuardAndPayload.h>
}

#include "CopyFromRoot.hpp"

#ifdef EL_INSTANTIATE_BLAS_LEVEL1
# define EL_EXTERN
#else
# define EL_EXTERN extern
#endif

#define PROTO(T)

#define TMP_PROTO_SDFLSKDJFLSDKJ(T)                                     \
    EL_EXTERN template void Copy(                                       \
        AbstractMatrix<T> const& A,                                     \
        AbstractMatrix<T>& B);                                          \
    EL_EXTERN template void Copy(                                       \
        Matrix<T> const& A,                                             \
        Matrix<T>& B);                                                  \
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

#endif // ifndef EL_BLAS_COPY_HPP

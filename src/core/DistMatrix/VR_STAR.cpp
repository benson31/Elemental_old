/*
   Copyright (c) 2009-2014, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

#define ColDist VR
#define RowDist STAR

#include "./setup.hpp"

namespace El {

// Public section
// ##############

// Assignment and reconfiguration
// ==============================

// Return a view
// -------------
template<typename T>
DM DM::operator()( Range<Int> indVert, Range<Int> indHorz )
{
    DEBUG_ONLY(CallStackEntry cse("DM[VR,STAR]( ind, ind )"))
    if( this->Locked() )
        return LockedView( *this, indVert, indHorz );
    else
        return View( *this, indVert, indHorz );
}

template<typename T>
const DM DM::operator()( Range<Int> indVert, Range<Int> indHorz ) const
{
    DEBUG_ONLY(CallStackEntry cse("DM[VR,STAR]( ind, ind )"))
    return LockedView( *this, indVert, indHorz );
}

// Make a copy
// -----------
template<typename T>
DM& DM::operator=( const DM& A )
{
    DEBUG_ONLY(CallStackEntry cse("DM[VR,STAR] = DM[VR,STAR]"))
    copy::Translate( A, *this );
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,MC,MR>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [MC,MR]"))
    DistMatrix<T,VC,STAR> A_VC_STAR( A );
    *this = A_VC_STAR;
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,MC,STAR>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [MC,STAR]"))
    DistMatrix<T,VC,STAR> A_VC_STAR( A );
    *this = A_VC_STAR;
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,STAR,MR>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [STAR,MR]"))
    auto A_MC_MR = MakeUnique<DistMatrix<T,MC,MR>>( A );
    auto A_VC_STAR = MakeUnique<DistMatrix<T,VC,STAR>>( *A_MC_MR );
    A_MC_MR.reset(); 
    *this = *A_VC_STAR;
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,MD,STAR>& A )
{
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [MD,STAR]"))
    // TODO: Optimize this later if important
    DistMatrix<T,STAR,STAR> A_STAR_STAR( A );
    *this = A_STAR_STAR;
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,STAR,MD>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [STAR,MD]"))
    // TODO: Optimize this later if important
    DistMatrix<T,STAR,STAR> A_STAR_STAR( A );
    *this = A_STAR_STAR;
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,MR,MC>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [MR,MC]"))
    copy::ColAllToAllDemote( A, *this );
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,MR,STAR>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [MR,STAR]"))
    copy::PartialColFilter( A, *this );
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,STAR,MC>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [STAR,MC]"))
    DistMatrix<T,MR,MC> A_MR_MC( A );
    *this = A_MR_MC;
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,VC,STAR>& A )
{ 
    DEBUG_ONLY(
        CallStackEntry cse("[VR,STAR] = [VC,STAR]");
        AssertSameGrids( *this, A );
        this->AssertNotLocked();
    )
    const El::Grid& g = this->Grid();
    this->Resize( A.Height(), A.Width() );
    if( !this->Participating() )
        return *this;

    const Int r = g.Height();
    const Int c = g.Width();
    const Int p = g.Size();
    const Int rankCM = g.VCRank();
    const Int rankRM = g.VRRank();

    const Int height = this->Height();
    const Int width = this->Width();
    const Int localHeight = this->LocalHeight();
    const Int localHeightOfA = A.LocalHeight();
    const Int maxLocalHeight = MaxLength(height,p);

    const Int portionSize = maxLocalHeight * width;

    const Int colShift = this->ColShift();
    const Int colShiftOfA = A.ColShift();

    // Compute which rowmajor rank has the colShift equal to our colShiftOfA
    const Int sendRankRM = (rankRM+(p+colShiftOfA-colShift)) % p;

    // Compute which rowmajor rank has the A colShift that we need
    const Int recvRankCM = (rankCM+(p+colShift-colShiftOfA)) % p;
    const Int recvRankRM = (recvRankCM/r)+c*(recvRankCM%r);

    T* buffer = this->auxMemory_.Require( 2*portionSize );
    T* sendBuf = &buffer[0];
    T* recvBuf = &buffer[portionSize];

    // Pack
    const Int ALDim = A.LDim();
    const T* ABuf = A.LockedBuffer();
    EL_PARALLEL_FOR
    for( Int j=0; j<width; ++j )
    {
        const T* ACol = &ABuf[j*ALDim];
        T* sendBufCol = &sendBuf[j*localHeightOfA];
        MemCopy( sendBufCol, ACol, localHeightOfA );
    }

    // Communicate
    mpi::SendRecv
    ( sendBuf, portionSize, sendRankRM, 
      recvBuf, portionSize, recvRankRM, g.VRComm() );

    // Unpack
    T* thisBuf = this->Buffer();
    const Int thisLDim = this->LDim();
    EL_PARALLEL_FOR
    for( Int j=0; j<width; ++j )
    {
        const T* recvBufCol = &recvBuf[j*localHeight];
        T* thisCol = &thisBuf[j*thisLDim];
        MemCopy( thisCol, recvBufCol, localHeight );
    }
    this->auxMemory_.Release();
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,STAR,VC>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [STAR,VC]"))
    DistMatrix<T,MR,MC> A_MR_MC( A );
    *this = A_MR_MC;
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,STAR,VR>& A )
{ 
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [STAR,VR]"))
    auto A_MC_MR = MakeUnique<DistMatrix<T,MC,MR>>( A );
    auto A_VC_STAR = MakeUnique<DistMatrix<T,VC,STAR>>( *A_MC_MR );
    A_MC_MR.reset(); 
    *this = *A_VC_STAR;
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,STAR,STAR>& A )
{
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [STAR,STAR]"))
    copy::ColFilter( A, *this );
    return *this;
}

template<typename T>
DM& DM::operator=( const DistMatrix<T,CIRC,CIRC>& A )
{
    DEBUG_ONLY(CallStackEntry cse("[VR,STAR] = [CIRC,CIRC]"))
    copy::Scatter( A, *this );
    return *this;
}

template<typename T>
DM& DM::operator=( const AbstractDistMatrix<T>& A )
{
    DEBUG_ONLY(CallStackEntry cse("DM = ADM"))
    #define GUARD(CDIST,RDIST) \
      A.DistData().colDist == CDIST && A.DistData().rowDist == RDIST
    #define PAYLOAD(CDIST,RDIST) \
      auto& ACast = dynamic_cast<const DistMatrix<T,CDIST,RDIST>&>(A); \
      *this = ACast;
    #include "El/macros/GuardAndPayload.h"
    return *this;
}

// Basic queries
// =============

template<typename T>
mpi::Comm DM::DistComm() const { return this->grid_->VRComm(); }
template<typename T>
mpi::Comm DM::CrossComm() const { return mpi::COMM_SELF; }
template<typename T>
mpi::Comm DM::RedundantComm() const { return mpi::COMM_SELF; }
template<typename T>
mpi::Comm DM::ColComm() const { return this->grid_->VRComm(); }
template<typename T>
mpi::Comm DM::RowComm() const { return mpi::COMM_SELF; }
template<typename T>
mpi::Comm DM::PartialColComm() const { return this->grid_->MRComm(); }
template<typename T>
mpi::Comm DM::PartialUnionColComm() const { return this->grid_->MCComm(); }

template<typename T>
Int DM::ColStride() const { return this->grid_->VRSize(); }
template<typename T>
Int DM::RowStride() const { return 1; }
template<typename T>
Int DM::PartialColStride() const { return this->grid_->MRSize(); }
template<typename T>
Int DM::PartialUnionColStride() const { return this->grid_->MCSize(); }
template<typename T>
Int DM::DistSize() const { return this->grid_->VRSize(); }
template<typename T>
Int DM::CrossSize() const { return 1; }
template<typename T>
Int DM::RedundantSize() const { return 1; }

// Instantiate {Int,Real,Complex<Real>} for each Real in {float,double}
// ####################################################################

#define SELF(T,U,V) \
  template DistMatrix<T,ColDist,RowDist>::DistMatrix \
  ( const DistMatrix<T,U,V>& A );
#define OTHER(T,U,V) \
  template DistMatrix<T,ColDist,RowDist>::DistMatrix \
  ( const BlockDistMatrix<T,U,V>& A ); \
  template DistMatrix<T,ColDist,RowDist>& \
           DistMatrix<T,ColDist,RowDist>::operator= \
           ( const BlockDistMatrix<T,U,V>& A )
#define BOTH(T,U,V) \
  SELF(T,U,V); \
  OTHER(T,U,V)
#define PROTO(T) \
  template class DistMatrix<T,ColDist,RowDist>; \
  BOTH( T,CIRC,CIRC); \
  BOTH( T,MC,  MR  ); \
  BOTH( T,MC,  STAR); \
  BOTH( T,MD,  STAR); \
  BOTH( T,MR,  MC  ); \
  BOTH( T,MR,  STAR); \
  BOTH( T,STAR,MC  ); \
  BOTH( T,STAR,MD  ); \
  BOTH( T,STAR,MR  ); \
  BOTH( T,STAR,STAR); \
  BOTH( T,STAR,VC  ); \
  BOTH( T,STAR,VR  ); \
  BOTH( T,VC,  STAR); \
  OTHER(T,VR,  STAR);

#include "El/macros/Instantiate.h"

} // namespace El

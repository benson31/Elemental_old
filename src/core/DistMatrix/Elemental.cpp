/*
   Copyright (c) 2009-2015, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include "El.hpp"

namespace El {

// Public section
// ##############

// Constructors and destructors
// ============================

template<typename T>
ElementalMatrix<T>::ElementalMatrix( const El::Grid& grid, int root )
: AbstractDistMatrix<T>(grid,root)
{ }

template<typename T>
ElementalMatrix<T>::ElementalMatrix( ElementalMatrix<T>&& A ) 
EL_NO_EXCEPT
: AbstractDistMatrix<T>(std::move(A))
{ }

template<typename T>
ElementalMatrix<T>::~ElementalMatrix() { }

// Assignment and reconfiguration
// ==============================
template<typename T>
void
ElementalMatrix<T>::Resize( Int height, Int width )
{
    DEBUG_ONLY(
      CSE cse("EM::Resize");
      this->AssertNotLocked();
      if( this->Viewing() && (height > this->height_ || width > this->width_) )
          LogicError("Tried to increase the size of a view");
    )
    this->height_ = height; 
    this->width_ = width;
    if( this->Participating() )
        this->matrix_.Resize_
        ( Length(height,this->ColShift(),this->ColStride()),
          Length(width,this->RowShift(),this->RowStride()) );
}

template<typename T>
void
ElementalMatrix<T>::Resize( Int height, Int width, Int ldim )
{
    DEBUG_ONLY(
      CSE cse("EM::Resize");
      this->AssertNotLocked();
      if( this->Viewing() && 
          (height > this->height_ || width > this->width_ || 
           ldim > this->matrix_.LDim()) )
          LogicError("Tried to increase the size of a view");
    )
    this->height_ = height; 
    this->width_ = width;
    if( this->Participating() )
        this->matrix_.Resize_
        ( Length(height,this->ColShift(),this->ColStride()),
          Length(width,this->RowShift(),this->RowStride()), ldim );
}

template<typename T>
void
ElementalMatrix<T>::MakeConsistent( bool includingViewers )
{
    DEBUG_ONLY(CSE cse("EM::MakeConsistent"))

    const Int msgLength = 9;
    Int message[msgLength];
    if( this->CrossRank() == this->Root() )
    {
        message[0] = this->viewType_;
        message[1] = this->height_;
        message[2] = this->width_;
        message[3] = this->colConstrained_;
        message[4] = this->rowConstrained_;
        message[5] = this->rootConstrained_;
        message[6] = this->colAlign_;
        message[7] = this->rowAlign_;
        message[8] = this->root_;
    }

    const auto& g = *this->grid_;
    if( !g.InGrid() && !includingViewers )
        LogicError("Non-participating process called MakeConsistent");
    if( g.InGrid() )
    {
        // TODO: Ensure roots are consistent within each cross communicator
        mpi::Broadcast( message, msgLength, this->Root(), this->CrossComm() );
    }
    if( includingViewers )
    {
        const Int vcRoot = g.VCToViewing(0);
        mpi::Broadcast( message, msgLength, vcRoot, g.ViewingComm() );
    }
    const ViewType newViewType    = static_cast<ViewType>(message[0]);
    const Int newHeight           = message[1]; 
    const Int newWidth            = message[2];
    const bool newConstrainedCol  = message[3];
    const bool newConstrainedRow  = message[4];
    const bool newConstrainedRoot = message[5];
    const Int newColAlign         = message[6];
    const Int newRowAlign         = message[7];
    const int root                = message[8];

    this->root_            = root;
    this->viewType_        = newViewType;
    this->colConstrained_  = newConstrainedCol;
    this->rowConstrained_  = newConstrainedRow;
    this->rootConstrained_ = newConstrainedRoot;
    this->colAlign_        = newColAlign;
    this->rowAlign_        = newRowAlign;

    this->SetShifts();
    this->Resize( newHeight, newWidth );
}

// Realignment
// -----------

template<typename T>
void
ElementalMatrix<T>::Align( int colAlign, int rowAlign, bool constrain )
{ 
    DEBUG_ONLY(CSE cse("EM::Align"))
    const bool requireChange =
      this->colAlign_ != colAlign || this->rowAlign_ != rowAlign;
    DEBUG_ONLY(
      if( this->Viewing() && requireChange )
          LogicError("Tried to realign a view");
    )
    if( requireChange )
        this->Empty();
    if( constrain )
    {
        this->colConstrained_ = true;
        this->rowConstrained_ = true;
    }
    this->colAlign_ = colAlign;
    this->rowAlign_ = rowAlign;
    this->SetShifts();
}

template<typename T>
void
ElementalMatrix<T>::AlignCols( int colAlign, bool constrain )
{ 
    DEBUG_ONLY(
      CSE cse("EM::AlignCols");
      if( this->Viewing() && this->colAlign_ != colAlign )
          LogicError("Tried to realign a view");
    )
    if( this->colAlign_ != colAlign )
        this->EmptyData();
    if( constrain )
        this->colConstrained_ = true;
    this->colAlign_ = colAlign;
    this->SetShifts();
}

template<typename T>
void
ElementalMatrix<T>::AlignRows( int rowAlign, bool constrain )
{ 
    DEBUG_ONLY(
      CSE cse("EM::AlignRows");
      if( this->Viewing() && this->rowAlign_ != rowAlign )
          LogicError("Tried to realign a view");
    )
    if( this->rowAlign_ != rowAlign )
        this->EmptyData();
    if( constrain )
        this->rowConstrained_ = true;
    this->rowAlign_ = rowAlign;
    this->SetShifts();
}

template<typename T>
void
ElementalMatrix<T>::FreeAlignments() 
{ 
    if( !this->Viewing() )
    {
        this->colConstrained_ = false;
        this->rowConstrained_ = false;
        this->rootConstrained_ = false;
    }
    else
        LogicError("Cannot free alignments of views");
}

template<typename T>
void
ElementalMatrix<T>::AlignWith
( const El::ElementalData& data, bool constrain, bool allowMismatch )
{ 
    DEBUG_ONLY(CSE cse("EM::AlignWith"))
    this->AlignColsWith( data, constrain, allowMismatch );
    this->AlignRowsWith( data, constrain, allowMismatch );
}

template<typename T>
void
ElementalMatrix<T>::AlignColsWith
( const El::ElementalData& data, bool constrain, bool allowMismatch )
{
    DEBUG_ONLY(CSE cse("EM::AlignColsWith"))
    this->SetGrid( *data.grid );
    this->SetRoot( data.root );
    if( data.colDist == this->ColDist() || 
        data.colDist == this->PartialColDist() )
        this->AlignCols( data.colAlign, constrain );
    else if( data.rowDist == this->ColDist() ||
             data.rowDist == this->PartialColDist() )
        this->AlignCols( data.rowAlign, constrain );
    else if( data.colDist == this->PartialUnionColDist() )
        this->AlignCols( data.colAlign % this->ColStride(), constrain );
    else if( data.rowDist == this->PartialUnionColDist() )
        this->AlignCols( data.rowAlign % this->ColStride(), constrain );
    else if( this->ColDist() != this->CollectedColDist() && 
             data.colDist    != this->CollectedColDist() && 
             data.rowDist    != this->CollectedColDist() && !allowMismatch )
        LogicError("Nonsensical alignment");
}

template<typename T>
void ElementalMatrix<T>::AlignRowsWith
( const El::ElementalData& data, bool constrain, bool allowMismatch )
{
    DEBUG_ONLY(CSE cse("EM::AlignRowsWith"))
    this->SetGrid( *data.grid );
    this->SetRoot( data.root );
    if( data.colDist == this->RowDist() || 
        data.colDist == this->PartialRowDist() )
        this->AlignRows( data.colAlign, constrain );
    else if( data.rowDist == this->RowDist() || 
             data.rowDist == this->PartialRowDist() )
        this->AlignRows( data.rowAlign, constrain );
    else if( data.colDist == this->PartialUnionRowDist() )
        this->AlignRows( data.colAlign % this->RowStride(), constrain );
    else if( data.rowDist == this->PartialUnionRowDist() )
        this->AlignRows( data.rowAlign % this->RowStride(), constrain );
    else if( this->RowDist() != this->CollectedRowDist() && 
             data.colDist    != this->CollectedRowDist() && 
             data.rowDist    != this->CollectedRowDist() && !allowMismatch )
        LogicError("Nonsensical alignment");
}

template<typename T>
void
ElementalMatrix<T>::AlignAndResize
( int colAlign, int rowAlign, Int height, Int width, 
  bool force, bool constrain )
{
    DEBUG_ONLY(CSE cse("EM::AlignAndResize"))
    if( !this->Viewing() )
    {
        if( force || !this->ColConstrained() )
        {
            this->colAlign_ = colAlign;
            this->SetColShift(); 
        }
        if( force || !this->RowConstrained() )
        {
            this->rowAlign_ = rowAlign;
            this->SetRowShift();
        }
    }
    if( constrain )
    {
        this->colConstrained_ = true;
        this->rowConstrained_ = true;
    }
    if( force && (this->colAlign_ != colAlign || this->rowAlign_ != rowAlign) )
        LogicError("Could not set alignments"); 
    this->Resize( height, width );
}

template<typename T>
void
ElementalMatrix<T>::AlignColsAndResize
( int colAlign, Int height, Int width, bool force, bool constrain )
{
    DEBUG_ONLY(CSE cse("EM::AlignColsAndResize"))
    if( !this->Viewing() && (force || !this->ColConstrained()) )
    {
        this->colAlign_ = colAlign;
        this->SetColShift(); 
    }
    if( constrain )
        this->colConstrained_ = true;
    if( force && this->colAlign_ != colAlign )
        LogicError("Could not set col alignment");
    this->Resize( height, width );
}

template<typename T>
void
ElementalMatrix<T>::AlignRowsAndResize
( int rowAlign, Int height, Int width, bool force, bool constrain )
{
    DEBUG_ONLY(CSE cse("EM::AlignRowsAndResize"))
    if( !this->Viewing() && (force || !this->RowConstrained()) )
    {
        this->rowAlign_ = rowAlign;
        this->SetRowShift(); 
    }
    if( constrain )
        this->rowConstrained_ = true;
    if( force && this->rowAlign_ != rowAlign )
        LogicError("Could not set row alignment");
    this->Resize( height, width );
}

// Buffer attachment
// -----------------

template<typename T>
void
ElementalMatrix<T>::Attach
( Int height, Int width, const El::Grid& g, 
  int colAlign, int rowAlign, T* buffer, Int ldim, int root )
{
    DEBUG_ONLY(CSE cse("EM::Attach"))
    this->Empty();

    this->grid_ = &g;
    this->root_ = root;
    this->height_ = height;
    this->width_ = width;
    this->colAlign_ = colAlign;
    this->rowAlign_ = rowAlign;
    this->colConstrained_ = true;
    this->rowConstrained_ = true;
    this->rootConstrained_ = true;
    this->viewType_ = VIEW;
    this->SetShifts();
    if( this->Participating() )
    {
        Int localHeight = Length(height,this->colShift_,this->ColStride());
        Int localWidth = Length(width,this->rowShift_,this->RowStride());
        this->matrix_.Attach_( localHeight, localWidth, buffer, ldim );
    }
}

template<typename T>
void
ElementalMatrix<T>::Attach
( Int height, Int width, const El::Grid& g,
  int colAlign, int rowAlign, El::Matrix<T>& A, int root )
{
    // TODO: Assert that the local dimensions are correct
    Attach( height, width, g, colAlign, rowAlign, A.Buffer(), A.LDim(), root );
}

template<typename T>
void
ElementalMatrix<T>::Attach( const El::Grid& g, El::Matrix<T>& A )
{
    DEBUG_ONLY(CSE cse("EM::Attach"))
    if( g.Size() != 1 )
        LogicError("Assumed a grid size of one");
    Attach( A.Height(), A.Width(), g, 0, 0, A.Buffer(), A.LDim() );
}

template<typename T>
void
ElementalMatrix<T>::LockedAttach
( Int height, Int width, const El::Grid& g, 
  int colAlign, int rowAlign, const T* buffer, Int ldim, int root )
{
    DEBUG_ONLY(CSE cse("EM::LockedAttach"))
    this->Empty();

    this->grid_ = &g;
    this->root_ = root;
    this->height_ = height;
    this->width_ = width;
    this->colAlign_ = colAlign;
    this->rowAlign_ = rowAlign;
    this->colConstrained_ = true;
    this->rowConstrained_ = true;
    this->rootConstrained_ = true;
    this->viewType_ = LOCKED_VIEW;
    this->SetShifts();
    if( this->Participating() )
    {
        Int localHeight = Length(height,this->colShift_,this->ColStride());
        Int localWidth = Length(width,this->rowShift_,this->RowStride());
        this->matrix_.LockedAttach_( localHeight, localWidth, buffer, ldim );
    }
}

template<typename T>
void
ElementalMatrix<T>::LockedAttach
( Int height, Int width, const El::Grid& g, 
  int colAlign, int rowAlign, const El::Matrix<T>& A, int root )
{
    // TODO: Assert that the local dimensions are correct
    LockedAttach
    ( height, width, g, colAlign, rowAlign, A.LockedBuffer(), A.LDim(), root );
}

template<typename T>
void
ElementalMatrix<T>::LockedAttach( const El::Grid& g, const El::Matrix<T>& A )
{
    DEBUG_ONLY(CSE cse("EM::LockedAttach"))
    if( g.Size() != 1 )
        LogicError("Assumed a grid size of one");
    LockedAttach( A.Height(), A.Width(), g, 0, 0, A.LockedBuffer(), A.LDim() );
}

// Operator overloading
// ====================

// Copy
// ----
template<typename T>
const ElementalMatrix<T>&
ElementalMatrix<T>::operator=( const ElementalMatrix<T>& A )
{
    DEBUG_ONLY(CSE cse("EM::operator=(EM&)"))
    Copy( A, *this );
    return *this;
}

template<typename T>
const ElementalMatrix<T>&
ElementalMatrix<T>::operator=( const DistMultiVec<T>& A )
{
    DEBUG_ONLY(CSE cse("EM::operator=(DMV&)"))
    Copy( A, *this );
    return *this;
}

// Addition/subtraction
// --------------------
template<typename T>
const ElementalMatrix<T>&
ElementalMatrix<T>::operator+=( const ElementalMatrix<T>& A )
{
    DEBUG_ONLY(CSE cse("EM::operator+="))
    Axpy( T(1), A, *this );
    return *this;
}

template<typename T>
const ElementalMatrix<T>&
ElementalMatrix<T>::operator-=( const ElementalMatrix<T>& A )
{
    DEBUG_ONLY(CSE cse("EM::operator-="))
    Axpy( T(-1), A, *this );
    return *this;
}

// Move assignment
// ---------------
template<typename T>
ElementalMatrix<T>& 
ElementalMatrix<T>::operator=( ElementalMatrix<T>&& A )
{
    DEBUG_ONLY(CSE cse("EM::operator=(EM&&)"))
    if( this->Viewing() || A.Viewing() )
    {
        Copy( A, *this );
    }
    else
    {
        this->matrix_.ShallowSwap( A.matrix_ );
        this->viewType_ = A.viewType_;
        this->height_ = A.height_;
        this->width_ = A.width_;
        this->colConstrained_ = A.colConstrained_;
        this->rowConstrained_ = A.rowConstrained_;
        this->rootConstrained_ = A.rootConstrained_;
        this->colAlign_ = A.colAlign_;
        this->rowAlign_ = A.rowAlign_;
        this->colShift_ = A.colShift_;
        this->rowShift_ = A.rowShift_;
        this->root_ = A.root_;
        this->grid_ = A.grid_;
    }
    return *this;
}

// Basic queries
// =============

// Distribution information
// ------------------------

template<typename T>
int ElementalMatrix<T>::RowOwner( Int i ) const EL_NO_EXCEPT
{
    if( i == END ) i = this->height_ - 1;
    return int((i+this->ColAlign()) % this->ColStride());
}

template<typename T>
int ElementalMatrix<T>::ColOwner( Int j ) const EL_NO_EXCEPT
{ 
    if( j == END ) j = this->width_ - 1;
    return int((j+this->RowAlign()) % this->RowStride()); 
}

template<typename T>
Int ElementalMatrix<T>::LocalRowOffset( Int i ) const EL_NO_EXCEPT
{ 
    if( i == END ) i = this->height_ - 1;
    return Length_(i,this->ColShift(),this->ColStride()); 
}

template<typename T>
Int ElementalMatrix<T>::LocalColOffset( Int j ) const EL_NO_EXCEPT
{ 
    if( j == END ) j = this->width_ - 1;
    return Length_(j,this->RowShift(),this->RowStride()); 
}

template<typename T>
Int ElementalMatrix<T>::GlobalRow( Int iLoc ) const EL_NO_EXCEPT
{ 
    if( iLoc == END ) iLoc = this->LocalHeight() - 1;
    return this->ColShift() + iLoc*this->ColStride(); 
}

template<typename T>
Int ElementalMatrix<T>::GlobalCol( Int jLoc ) const EL_NO_EXCEPT
{ 
    if( jLoc == END ) jLoc = this->LocalWidth() - 1;
    return this->RowShift() + jLoc*this->RowStride(); 
}

// Diagonal manipulation
// =====================
template<typename T>
bool ElementalMatrix<T>::DiagonalAlignedWith
( const El::ElementalData& d, Int offset ) const EL_NO_EXCEPT
{
    DEBUG_ONLY(CSE cse("EM::DiagonalAlignedWith"))
    if( this->Grid() != *d.grid )
        return false;

    const Int diagRoot = DiagonalRoot(offset);
    if( diagRoot != d.root )
        return false;

    const int diagAlign = DiagonalAlign(offset);
    const Dist UDiag = DiagCol( this->ColDist(), this->RowDist() ); 
    const Dist VDiag = DiagRow( this->ColDist(), this->RowDist() );
    if( d.colDist == UDiag && d.rowDist == VDiag )
        return d.colAlign == diagAlign;
    else if( d.colDist == VDiag && d.rowDist == UDiag )
        return d.rowAlign == diagAlign;
    else
        return false;
}

template<typename T>
int ElementalMatrix<T>::DiagonalRoot( Int offset ) const EL_NO_EXCEPT
{
    DEBUG_ONLY(CSE cse("EM::DiagonalRoot"))
    const auto& grid = this->Grid();

    if( this->ColDist() == MC && this->RowDist() == MR )
    {
        // Result is an [MD,* ] or [* ,MD]
        int owner;
        if( offset >= 0 )
        {
            const int procRow = this->ColAlign();
            const int procCol = (this->RowAlign()+offset) % this->RowStride();
            owner = procRow + this->ColStride()*procCol;
        }
        else
        {
            const int procRow = (this->ColAlign()-offset) % this->ColStride();
            const int procCol = this->RowAlign();
            owner = procRow + this->ColStride()*procCol;
        }
        return grid.Diag(owner);
    }
    else if( this->ColDist() == MR && this->RowDist() == MC )
    {
        // Result is an [MD,* ] or [* ,MD]
        int owner;
        if( offset >= 0 )
        {
            const int procCol = this->ColAlign();
            const int procRow = (this->RowAlign()+offset) % this->RowStride();
            owner = procRow + this->ColStride()*procCol;
        }
        else
        {
            const int procCol = (this->ColAlign()-offset) % this->ColStride();
            const int procRow = this->RowAlign();
            owner = procRow + this->ColStride()*procCol;
        }
        return grid.Diag(owner);
    }
    else
        return this->Root();
}

template<typename T>
int ElementalMatrix<T>::DiagonalAlign( Int offset ) const EL_NO_EXCEPT
{
    DEBUG_ONLY(CSE cse("EM::DiagonalAlign"))
    const auto& grid = this->Grid();

    if( this->ColDist() == MC && this->RowDist() == MR )
    {
        // Result is an [MD,* ] or [* ,MD]
        int owner;
        if( offset >= 0 )
        {
            const int procRow = this->ColAlign();
            const int procCol = (this->RowAlign()+offset) % this->RowStride();
            owner = procRow + this->ColStride()*procCol;
        }
        else
        {
            const int procRow = (this->ColAlign()-offset) % this->ColStride();
            const int procCol = this->RowAlign();
            owner = procRow + this->ColStride()*procCol;
        }
        return grid.DiagRank(owner);
    }
    else if( this->ColDist() == MR && this->RowDist() == MC )
    {
        // Result is an [MD,* ] or [* ,MD]
        int owner;
        if( offset >= 0 )
        {
            const int procCol = this->ColAlign();
            const int procRow = (this->RowAlign()+offset) % this->RowStride();
            owner = procRow + this->ColStride()*procCol;
        }
        else
        {
            const int procCol = (this->ColAlign()-offset) % this->ColStride();
            const int procRow = this->RowAlign();
            owner = procRow + this->ColStride()*procCol;
        }
        return grid.DiagRank(owner);
    }
    else if( this->ColDist() == STAR )
    {
        // Result is a [V,* ] or [* ,V]
        if( offset >= 0 )
            return (this->RowAlign()+offset) % this->RowStride();
        else
            return this->RowAlign();
    }
    else
    {
        // Result is [U,V] or [V,U], where V is either STAR or CIRC
        if( offset >= 0 )
            return this->ColAlign();
        else
            return (this->ColAlign()-offset) % this->ColStride();
    }
}

// Private section
// ###############

// Outside of class
// ----------------

template<typename T> 
void
AssertConforming1x2
( const ElementalMatrix<T>& AL, const ElementalMatrix<T>& AR )
{
    if( AL.Height() != AR.Height() )    
        LogicError
        ("1x2 not conformant:\n",
         DimsString(AL,"Left"),"\n",DimsString(AR,"Right"));
    if( AL.ColAlign() != AR.ColAlign() )
        LogicError("1x2 is misaligned");
}

template<typename T> 
void
AssertConforming2x1
( const ElementalMatrix<T>& AT, const ElementalMatrix<T>& AB )
{
    if( AT.Width() != AB.Width() )
        LogicError
        ("2x1 is not conformant:\n",
         DimsString(AT,"Top"),"\n",DimsString(AB,"Bottom"));
    if( AT.RowAlign() != AB.RowAlign() )
        LogicError("2x1 is not aligned");
}

template<typename T> 
void
AssertConforming2x2
( const ElementalMatrix<T>& ATL, const ElementalMatrix<T>& ATR, 
  const ElementalMatrix<T>& ABL, const ElementalMatrix<T>& ABR ) 
{
    if( ATL.Width() != ABL.Width() || ATR.Width() != ABR.Width() ||
        ATL.Height() != ATR.Height() || ABL.Height() != ABR.Height() )
        LogicError
        ("2x2 is not conformant:\n",
         DimsString(ATL,"TL"),"\n",DimsString(ATR,"TR"),"\n",
         DimsString(ABL,"BL"),"\n",DimsString(ABR,"BR"));
    if( ATL.ColAlign() != ATR.ColAlign() ||
        ABL.ColAlign() != ABR.ColAlign() ||
        ATL.RowAlign() != ABL.RowAlign() ||
        ATR.RowAlign() != ABR.RowAlign() )
        LogicError("2x2 set of matrices must aligned to combine");
}

template<typename T>
void
ElementalMatrix<T>::ShallowSwap( ElementalMatrix<T>& A )
{
    this->matrix_.ShallowSwap( A.matrix_ );
    std::swap( this->viewType_, A.viewType_ );
    std::swap( this->height_ , A.height_ );
    std::swap( this->width_, A.width_ );
    std::swap( this->colConstrained_, A.colConstrained_ );
    std::swap( this->rowConstrained_, A.rowConstrained_ );
    std::swap( this->rootConstrained_, A.rootConstrained_ );
    std::swap( this->colAlign_, A.colAlign_ );
    std::swap( this->rowAlign_, A.rowAlign_ );
    std::swap( this->colShift_, A.colShift_ );
    std::swap( this->rowShift_, A.rowShift_ );
    std::swap( this->root_, A.root_ );
    std::swap( this->grid_, A.grid_ );
}

// Instantiations for {Int,Real,Complex<Real>} for each Real in {float,double}
// ###########################################################################

#ifndef EL_RELEASE
 #define PROTO(T) \
  template class ElementalMatrix<T>;\
  template void AssertConforming1x2\
  ( const ElementalMatrix<T>& AL, const ElementalMatrix<T>& AR );\
  template void AssertConforming2x1\
  ( const ElementalMatrix<T>& AT, const ElementalMatrix<T>& AB );\
  template void AssertConforming2x2\
  ( const ElementalMatrix<T>& ATL, const ElementalMatrix<T>& ATR,\
    const ElementalMatrix<T>& ABL, const ElementalMatrix<T>& ABR );
#else
 #define PROTO(T) template class ElementalMatrix<T>;
#endif

#define EL_ENABLE_QUAD
#include "El/macros/Instantiate.h"

} // namespace El
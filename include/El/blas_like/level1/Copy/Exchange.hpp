/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_BLAS_COPY_EXCHANGE_HPP
#define EL_BLAS_COPY_EXCHANGE_HPP

namespace El {
namespace copy {

template<typename T,Device D,typename=EnableIf<IsDeviceValidType<T,D>>>
void Exchange_impl
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
  int sendRank, int recvRank, mpi::Comm comm )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids( A, B ))

    const int myRank = mpi::Rank( comm );
    EL_DEBUG_ONLY(
      if( myRank == sendRank && myRank != recvRank )
          LogicError("Sending to self but receiving from someone else");
      if( myRank != sendRank && myRank == recvRank )
          LogicError("Receiving from self but sending to someone else");
    )
    B.Resize( A.Height(), A.Width() );
    if( myRank == sendRank )
    {
        Copy( A.LockedMatrix(), B.Matrix() );
        return;
    }

    const Int localHeightA = A.LocalHeight();
    const Int localHeightB = B.LocalHeight();
    const Int localWidthA = A.LocalWidth();
    const Int localWidthB = B.LocalWidth();
    const Int sendSize = localHeightA*localWidthA;
    const Int recvSize = localHeightB*localWidthB;

    const bool contigA = ( A.LocalHeight() == A.LDim() );
    const bool contigB = ( B.LocalHeight() == B.LDim() );

    // DEBUG
    Timer clock;
    double time_elapsed;

    if( contigA && contigB )
    {
        OutputFromRoot(A.Grid().Comm(),
                       "Exchange (Contig A, Contig B)");
        clock.Start();
        mpi::SendRecv
        ( A.LockedBuffer(), sendSize, sendRank,
          B.Buffer(),       recvSize, recvRank, comm );

        time_elapsed = clock.Stop();
        OutputFromRoot(A.Grid().Comm(),
                       "  SendRecv: ", time_elapsed, "s");
    }
    else if( contigB )
    {
        OutputFromRoot(A.Grid().Comm(),
                       "Exchange (Non-Contig A, Contig B)");
        // Pack A's data
        simple_buffer<T,D> buf(sendSize);

        clock.Start();
        copy::util::InterleaveMatrix<T,D>
        ( localHeightA, localWidthA,
          A.LockedBuffer(), 1, A.LDim(),
          buf.data(),       1, localHeightA );

        time_elapsed = clock.Stop();
        OutputFromRoot(A.Grid().Comm(),
                       "  InterleaveMatrix: ", time_elapsed, "s");

        // Exchange with the partner
        clock.Reset();
        clock.Start();
        mpi::SendRecv
        ( buf.data(), sendSize, sendRank,
          B.Buffer(), recvSize, recvRank, comm );

        time_elapsed = clock.Stop();
        OutputFromRoot(A.Grid().Comm(),
                       "          SendRecv: ", time_elapsed, "s");
    }
    else if( contigA )
    {
        OutputFromRoot(A.Grid().Comm(),
                       "Exchange (Contig A, Non-Contig B)");
        // Exchange with the partner
        simple_buffer<T,D> buf(recvSize);

        clock.Start();
        mpi::SendRecv
        ( A.LockedBuffer(), sendSize, sendRank,
          buf.data(),       recvSize, recvRank, comm );

        time_elapsed = clock.Stop();
        OutputFromRoot(A.Grid().Comm(),
                       "          SendRecv: ", time_elapsed, "s");

        // Unpack
        clock.Reset();
        clock.Start();
        copy::util::InterleaveMatrix<T,D>
        ( localHeightB, localWidthB,
          buf.data(), 1, localHeightB,
          B.Buffer(), 1, B.LDim() );

        time_elapsed = clock.Stop();
        OutputFromRoot(A.Grid().Comm(),
                       "  InterleaveMatrix: ", time_elapsed, "s");
    }
    else
    {
        OutputFromRoot(A.Grid().Comm(),
                       "Exchange (Non-Contig A, Non-Contig B)");
        // Pack A's data
        simple_buffer<T,D> sendBuf(sendSize), recvBuf(recvSize);
        clock.Start();
        copy::util::InterleaveMatrix<T,D>
        ( localHeightA, localWidthA,
          A.LockedBuffer(), 1, A.LDim(),
          sendBuf.data(),   1, localHeightA );

        time_elapsed = clock.Stop();
        OutputFromRoot(A.Grid().Comm(),
                       "  InterleaveMatrix: ", time_elapsed, "s");

        // Exchange with the partner
        clock.Reset();
        clock.Start();
        mpi::SendRecv
        ( sendBuf.data(), sendSize, sendRank,
          recvBuf.data(), recvSize, recvRank, comm );

        time_elapsed = clock.Stop();
        OutputFromRoot(A.Grid().Comm(),
                       "          SendRecv: ", time_elapsed, "s");

        // Unpack
        clock.Reset();
        clock.Start();
        copy::util::InterleaveMatrix<T,D>
        ( localHeightB, localWidthB,
          recvBuf.data(), 1, localHeightB,
          B.Buffer(),     1, B.LDim() );

        time_elapsed = clock.Stop();
        OutputFromRoot(A.Grid().Comm(),
                       "  InterleaveMatrix: ", time_elapsed, "s");
    }
}

template<typename T,Device D,
         typename=DisableIf<IsDeviceValidType<T,D>>,typename=void>
void Exchange_impl
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
  int sendRank, int recvRank, mpi::Comm comm )
{
    LogicError("Exchange: Bad Device/type combo.");
}

template<typename T>
void Exchange
( const ElementalMatrix<T>& A,
        ElementalMatrix<T>& B,
  int sendRank, int recvRank, mpi::Comm comm )
{
    if (A.GetLocalDevice() != B.GetLocalDevice())
        LogicError("Exchange: Device error.");
    switch (A.GetLocalDevice())
    {
    case Device::CPU:
        Exchange_impl<T,Device::CPU>(A,B,sendRank,recvRank,comm);
        break;
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        Exchange_impl<T,Device::GPU>(A,B,sendRank,recvRank,comm);
        break;
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Exchange: Bad device.");
    }
}

template<typename T,Dist U,Dist V,Device D>
void ColwiseVectorExchange
( DistMatrix<T,ProductDist<U,V>(),STAR,ELEMENT,D> const& A,
  DistMatrix<T,ProductDist<V,U>(),STAR,ELEMENT,D>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );

    if( !B.Participating() )
        return;

    const Int distSize = A.DistSize();
    const Int colDiff = A.ColShift() - B.ColShift();
    const Int sendRankB = Mod( B.DistRank()+colDiff, distSize );
    const Int recvRankA = Mod( A.DistRank()-colDiff, distSize );
    const Int recvRankB =
      (recvRankA/A.PartialColStride())+
      (recvRankA%A.PartialColStride())*A.PartialUnionColStride();
    copy::Exchange_impl<T,D>( A, B, sendRankB, recvRankB, B.DistComm() );
}

template<typename T,Dist U,Dist V,Device D>
void RowwiseVectorExchange
( DistMatrix<T,STAR,ProductDist<U,V>(),ELEMENT,D> const& A,
  DistMatrix<T,STAR,ProductDist<V,U>(),ELEMENT,D>& B )
{
    EL_DEBUG_CSE
    AssertSameGrids( A, B );

    if( !B.Participating() )
        return;

    Timer clock;
    double time_elapsed;

    OutputFromRoot(A.Grid().Comm(), "RowwiseVectorExchange");

    const Int distSize = A.DistSize();
    const Int rowDiff = A.RowShift() - B.RowShift();
    const Int sendRankB = Mod( B.DistRank()+rowDiff, distSize );
    const Int recvRankA = Mod( A.DistRank()-rowDiff, distSize );
    const Int recvRankB =
      (recvRankA/A.PartialRowStride())+
      (recvRankA%A.PartialRowStride())*A.PartialUnionRowStride();

    clock.Start();
    copy::Exchange_impl<T,D>( A, B, sendRankB, recvRankB, B.DistComm() );
    time_elapsed = clock.Stop();
    MPI_Reduce(A.Grid().Comm().Rank() == 0 ? MPI_IN_PLACE : &time_elapsed,
               &time_elapsed, 1, mpi::TypeMap<T>(),
               MPI_MAX, 0, A.Grid().Comm().comm);

    OutputFromRoot(A.Grid().Comm(),
                   "  Exchange_impl: ", time_elapsed, "s");
}

} // namespace copy
} // namespace El

#endif // ifndef EL_BLAS_COPY_EXCHANGE_HPP

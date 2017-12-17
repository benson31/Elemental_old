/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

namespace El {
namespace syr2k {

template<typename T>
void UT_C
( T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  bool conjugate=false )
{
    EL_DEBUG_CSE
    const Int r = APre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const Orientation orientation = ( conjugate ? ADJOINT : TRANSPOSE );
    const T alphaSec = ( conjugate ? Conj(alpha) : alpha );

    DistMatrixReadProxy<T,T,Dist::MC,Dist::MR>
      AProx( APre ),
      BProx( BPre );
    DistMatrixReadWriteProxy<T,T,Dist::MC,Dist::MR>
      CProx( CPre );
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,Dist::MR,  Dist::STAR> A1Trans_MR_STAR(g), B1Trans_MR_STAR(g);
    DistMatrix<T,Dist::STAR,Dist::VR  > A1_STAR_VR(g), B1_STAR_VR(g);
    DistMatrix<T,Dist::STAR,Dist::MC  > A1_STAR_MC(g), B1_STAR_MC(g);

    A1Trans_MR_STAR.AlignWith( C );
    B1Trans_MR_STAR.AlignWith( C );
    A1_STAR_MC.AlignWith( C );
    B1_STAR_MC.AlignWith( C );

    for( Int k=0; k<r; k+=bsize )
    {
        const Int nb = Min(bsize,r-k);

        auto A1 = A( IR(k,k+nb), ALL );
        auto B1 = B( IR(k,k+nb), ALL );

        Transpose( A1, A1Trans_MR_STAR );
        Transpose( A1Trans_MR_STAR, A1_STAR_VR );
        A1_STAR_MC = A1_STAR_VR;

        Transpose( B1, B1Trans_MR_STAR );
        Transpose( B1Trans_MR_STAR, B1_STAR_VR );
        B1_STAR_MC = B1_STAR_VR;

        LocalTrr2k
        ( UPPER, orientation, TRANSPOSE, orientation, TRANSPOSE,
          alpha,    A1_STAR_MC, B1Trans_MR_STAR,
          alphaSec, B1_STAR_MC, A1Trans_MR_STAR, T(1), C );
    }
}

template<typename T>
void UT_Dot
( T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  const bool conjugate,
  Int blockSize=2000 )
{
    EL_DEBUG_CSE
    const Int n = CPre.Height();
    const Grid& g = APre.Grid();

    const Orientation orient = ( conjugate ? ADJOINT : TRANSPOSE );

    DistMatrixReadProxy<T,T,Dist::VC,Dist::STAR> AProx( APre );
    auto& A = AProx.GetLocked();

    ElementalProxyCtrl BCtrl;
    BCtrl.colConstrain = true;
    BCtrl.colAlign = A.ColAlign();
    DistMatrixReadProxy<T,T,Dist::VC,Dist::STAR> BProx( BPre, BCtrl );
    auto& B = BProx.GetLocked();

    DistMatrixReadWriteProxy<T,T,Dist::MC,Dist::MR> CProx( CPre );
    auto& C = CProx.Get();

    DistMatrix<T,Dist::STAR,Dist::STAR> Z( blockSize, blockSize, g );
    Zero( Z );
    for( Int kOuter=0; kOuter<n; kOuter+=blockSize )
    {
        const Int nbOuter = Min(blockSize,n-kOuter);
        const Range<Int> indOuter( kOuter, kOuter+nbOuter );

        auto A1 = A( ALL, indOuter );
        auto B1 = B( ALL, indOuter );
        auto C11 = C( indOuter, indOuter );

        Z.Resize( nbOuter, nbOuter );
        Syr2k
        ( UPPER, TRANSPOSE, alpha, A1.Matrix(), B1.Matrix(), Z.Matrix(),
          conjugate );
        AxpyContract( T(1), Z, C11 );

        for( Int kInner=0; kInner<kOuter; kInner+=blockSize )
        {
            const Int nbInner = Min(blockSize,kOuter-kInner);
            const Range<Int> indInner( kInner, kInner+nbInner );

            auto A2 = A( ALL, indInner );
            auto B2 = B( ALL, indInner );
            auto C21 = C( indInner, indOuter );

            LocalGemm( orient, NORMAL, alpha, A1, B2, Z );
            LocalGemm( orient, NORMAL, Conj(alpha), B1, A2, Z );
            AxpyContract( T(1), Z, C21 );
        }
    }
}

template<typename T>
void UT
( T alpha,
  const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C,
  bool conjugate=false )
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids( A, B, C );
      if( A.Width() != C.Height() || A.Width() != C.Width() ||
          B.Width() != C.Height() || B.Width() != C.Width() ||
          A.Height() != B.Height() )
          LogicError
          ("Nonconformal:\n",
           DimsString(A,"A"),"\n",
           DimsString(B,"B"),"\n",
           DimsString(C,"C"));
    )
    const Int r = A.Height();
    const Int n = A.Width();

    const double weightAwayFromDot = 10.;

    const Int blockSizeDot = 2000;

    if( r > weightAwayFromDot*n )
        UT_Dot( alpha, A, B, C, conjugate, blockSizeDot );
    else
        UT_C( alpha, A, B, C, conjugate );
}

} // namespace syr2k
} // namespace El

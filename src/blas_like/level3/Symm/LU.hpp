/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/

#include "El/core/Proxy.hpp"

namespace El
{
namespace symm
{

template<typename T>
void LocalAccumulateLU
(Orientation orientation, T alpha,
  const DistMatrix<T>& A,
  const DistMatrix<T,Dist::MC,  Dist::STAR>& B_MC_STAR,
  const DistMatrix<T,Dist::STAR,Dist::MR  >& BTrans_STAR_MR,
        DistMatrix<T,Dist::MC,  Dist::STAR>& Z_MC_STAR,
        DistMatrix<T,Dist::MR,  Dist::STAR>& Z_MR_STAR)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(
      AssertSameGrids(A, B_MC_STAR, BTrans_STAR_MR, Z_MC_STAR, Z_MR_STAR);
      if(A.Height() != A.Width() ||
          A.Height() != B_MC_STAR.Height() ||
          A.Height() != BTrans_STAR_MR.Width() ||
          A.Height() != Z_MC_STAR.Height() ||
          A.Height() != Z_MR_STAR.Height() ||
          B_MC_STAR.Width() != BTrans_STAR_MR.Height() ||
          BTrans_STAR_MR.Height() != Z_MC_STAR.Width() ||
          Z_MC_STAR.Width() != Z_MR_STAR.Width())
          LogicError
          ("Nonconformal:\n",
           DimsString(A,"A"),"\n",
           DimsString(B_MC_STAR,"B[MC,* ]"),"\n",
           DimsString(BTrans_STAR_MR,"B'[* ,MR]"),"\n",
           DimsString(Z_MC_STAR,"Z[MC,* ]"),"\n",
           DimsString(Z_MR_STAR,"Z[MR,* ]"));
      if(B_MC_STAR.ColAlign() != A.ColAlign() ||
          BTrans_STAR_MR.RowAlign() != A.RowAlign() ||
          Z_MC_STAR.ColAlign() != A.ColAlign() ||
          Z_MR_STAR.ColAlign() != A.RowAlign())
          LogicError("Partial matrix distributions are misaligned");
   )
    const Int m = B_MC_STAR.Height();
    const Grid& g = A.Grid();
    const Int ratio = Max(g.Height(), g.Width());
    const Int bsize = ratio*Blocksize();

    DistMatrix<T> D11(g);

    for(Int k=0; k<m; k+=bsize)
    {
        const Int nb = Min(bsize,m-k);

        const Range<Int> ind1(k,    k+nb),
                         ind2(k+nb, m   );

        auto A11 = A(ind1, ind1);
        auto A12 = A(ind1, ind2);

        auto B1_MC_STAR = B_MC_STAR(ind1, ALL);

        auto B1Trans_STAR_MR = BTrans_STAR_MR(ALL, ind1);
        auto B2Trans_STAR_MR = BTrans_STAR_MR(ALL, ind2);

        auto Z1_MC_STAR = Z_MC_STAR(ind1, ALL);

        auto Z1_MR_STAR = Z_MR_STAR(ind1, ALL);
        auto Z2_MR_STAR = Z_MR_STAR(ind2, ALL);

        D11.AlignWith(A11);
        D11 = A11;
        MakeTrapezoidal(UpperOrLower::UPPER, D11);
        LocalGemm
        (Orientation::NORMAL, orientation, alpha, D11, B1Trans_STAR_MR, T(1), Z1_MC_STAR);
        FillDiagonal(D11, T(0));

        LocalGemm
        (orientation, Orientation::NORMAL, alpha, D11, B1_MC_STAR, T(1), Z1_MR_STAR);

        LocalGemm
        (Orientation::NORMAL, orientation, alpha, A12, B2Trans_STAR_MR, T(1), Z1_MC_STAR);

        LocalGemm
        (orientation, Orientation::NORMAL, alpha, A12, B1_MC_STAR, T(1), Z2_MR_STAR);
    }
}

template<typename T>
void LUA
(T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  bool conjugate=false)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids(APre, BPre, CPre))
    const Int m = CPre.Height();
    const Int n = CPre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const Orientation orientation = (conjugate ? Orientation::ADJOINT : Orientation::TRANSPOSE);

    DistMatrixReadProxy<T,T,Dist::MC,Dist::MR> AProx(APre), BProx(BPre);
    DistMatrixReadWriteProxy<T,T,Dist::MC,Dist::MR> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    DistMatrix<T,Dist::MC,Dist::STAR> B1_MC_STAR(g);
    DistMatrix<T,Dist::VR,Dist::STAR> B1_VR_STAR(g);
    DistMatrix<T,Dist::STAR,Dist::MR> B1Trans_STAR_MR(g);
    DistMatrix<T> Z1(g);
    DistMatrix<T,Dist::MC,Dist::STAR> Z1_MC_STAR(g);
    DistMatrix<T,Dist::MR,Dist::STAR> Z1_MR_STAR(g);
    DistMatrix<T,Dist::MR,Dist::MC  > Z1_MR_MC(g);

    B1_MC_STAR.AlignWith(A);
    B1_VR_STAR.AlignWith(A);
    B1Trans_STAR_MR.AlignWith(A);
    Z1_MC_STAR.AlignWith(A);
    Z1_MR_STAR.AlignWith(A);

    for(Int k=0; k<n; k+=bsize)
    {
        const Int nb = Min(bsize,n-k);
        auto B1 = B(ALL, IR(k,k+nb));
        auto C1 = C(ALL, IR(k,k+nb));

        B1_MC_STAR = B1;
        B1_VR_STAR = B1_MC_STAR;
        Transpose(B1_VR_STAR, B1Trans_STAR_MR, conjugate);
        Z1_MC_STAR.Resize(m, nb);
        Z1_MR_STAR.Resize(m, nb);
        Zero(Z1_MC_STAR);
        Zero(Z1_MR_STAR);
        LocalAccumulateLU
        (orientation,
          alpha, A, B1_MC_STAR, B1Trans_STAR_MR, Z1_MC_STAR, Z1_MR_STAR);

        Contract(Z1_MR_STAR, Z1_MR_MC);
        Z1.AlignWith(C1);
        Z1 = Z1_MR_MC;
        AxpyContract(T(1), Z1_MC_STAR, Z1);
        C1 += Z1;
    }
}

template<typename T>
void LUC
(T alpha,
  const AbstractDistMatrix<T>& APre,
  const AbstractDistMatrix<T>& BPre,
        AbstractDistMatrix<T>& CPre,
  bool conjugate=false)
{
    EL_DEBUG_CSE
    EL_DEBUG_ONLY(AssertSameGrids(APre, BPre, CPre))
    const Int m = CPre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const Orientation orientation = (conjugate ? Orientation::ADJOINT : Orientation::TRANSPOSE);

    DistMatrixReadProxy<T,T,Dist::MC,Dist::MR> AProx(APre), BProx(BPre);
    DistMatrixReadWriteProxy<T,T,Dist::MC,Dist::MR> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    DistMatrix<T,Dist::MC,  Dist::STAR> AT1_MC_STAR(g);
    DistMatrix<T,Dist::STAR,Dist::MC  > A1R_STAR_MC(g);
    DistMatrix<T,Dist::MR,  Dist::STAR> B1Trans_MR_STAR(g);

    B1Trans_MR_STAR.AlignWith(C);

    for(Int k=0; k<m; k+=bsize)
    {
        const Int nb = Min(bsize,m-k);

        const Range<Int> indT(0, k+nb),
                         ind1(k, k+nb),
                         indB(k, m   ), indR(k, m);

        auto A1R = A(ind1, indR);
        auto AT1 = A(indT, ind1);

        auto B1 = B(ind1, ALL);

        auto CT = C(indT, ALL);
        auto CB = C(indB, ALL);

        AT1_MC_STAR.AlignWith(CT);
        A1R_STAR_MC.AlignWith(CB);
        AT1_MC_STAR = AT1;
        A1R_STAR_MC = A1R;
        MakeTrapezoidal(UpperOrLower::UPPER, AT1_MC_STAR, -k);
        MakeTrapezoidal(UpperOrLower::UPPER, A1R_STAR_MC, 1);

        Transpose(B1, B1Trans_MR_STAR);

        LocalGemm
        (Orientation::NORMAL, Orientation::TRANSPOSE,
          alpha, AT1_MC_STAR, B1Trans_MR_STAR, T(1), CT);

        LocalGemm
        (orientation, Orientation::TRANSPOSE,
          alpha, A1R_STAR_MC, B1Trans_MR_STAR, T(1), CB);
    }
}

template<typename T>
void LU
(T alpha,
  const AbstractDistMatrix<T>& A,
  const AbstractDistMatrix<T>& B,
        AbstractDistMatrix<T>& C,
  bool conjugate=false)
{
    EL_DEBUG_CSE
    // TODO: Come up with a better routing mechanism
    if(A.Height() > 5*B.Width())
        symm::LUA(alpha, A, B, C, conjugate);
    else
        symm::LUC(alpha, A, B, C, conjugate);
}

} // namespace symm
} // namespace El

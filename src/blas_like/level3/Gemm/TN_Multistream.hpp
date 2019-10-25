namespace El {
namespace gemm {

// Transpose Normal Gemm that avoids communicating the matrix A
template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_TNA_impl_multistream(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;
    constexpr Device D = Device::GPU;

    auto SyncInfo_C = SyncInfoFromMatrix(
        static_cast<Matrix<T,D> const&>(CPre.LockedMatrix()));

    AUTO_PROFILE_REGION(
        "SUMMA.TNA.multistream",
        SyncInfo_C);

    const Int n = CPre.Width();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    auto const num_blocks = (n + bsize - 1) / bsize;

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool();
    auto const num_stream_teams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size() / 2, size_t(num_blocks));

    // Temporary distributions
    std::vector<DistMatrix<T,MC,STAR,ELEMENT,D>> B1_MC_STAR;
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> D1_MR_STAR;
    std::vector<DistMatrix<T,MR,MC  ,ELEMENT,D>> D1_MR_MC;

    B1_MC_STAR.reserve(num_stream_teams);
    D1_MR_STAR.reserve(num_stream_teams);
    D1_MR_MC.reserve(num_stream_teams);

    // Setup temporaries
    for (auto id = 0UL; id < num_stream_teams; ++id)
    {
        auto B1 = B1_MC_STAR.emplace(B1_MC_STAR.end(), g);
        auto D1 = D1_MR_STAR.emplace(D1_MR_STAR.end(), g);
        auto D2 = D1_MR_MC.emplace(D1_MR_MC.end(), g);

        auto const& stream_one = stream_pool.Next();
        auto const& stream_two = stream_pool.Next();

        B1->AlignWith(A);
        D1->AlignWith(A);

        SetSyncInfo(B1->Matrix(), stream_one);
        SetSyncInfo(D1->Matrix(), stream_two);
        SetSyncInfo(D2->Matrix(), stream_two);
    }

    size_t team_id = 0UL;
    for(Int k=0; k<n; k+=bsize)
    {
        DistMatrix<T,MC,MR,ELEMENT,D> A1(g);
        LockedView(A1, A);

        const Int nb = Min(bsize,n-k);
        auto B1 = B(ALL, IR(k,k+nb));
        auto C1 = C(ALL, IR(k,k+nb));

        auto& BMCSTAR = B1_MC_STAR[team_id];
        auto& DMRSTAR = D1_MR_STAR[team_id];
        auto& DMRMC = D1_MR_MC[team_id];

        auto const& stream_one =
            SyncInfoFromMatrix(BMCSTAR.Matrix());
        auto const& stream_two =
            SyncInfoFromMatrix(DMRSTAR.Matrix());

        SetSyncInfo(A1.Matrix(), stream_one);
        SetSyncInfo(B1.Matrix(), stream_one);
        SetSyncInfo(C1.Matrix(), stream_two);

        // D1[MR,*] := alpha (A1[MC,MR])^T B1[MC,*]
        //           = alpha (A1^T)[MR,MC] B1[MC,*]
        BMCSTAR = B1;
        LocalGemm(orientA, NORMAL, alpha, A1, BMCSTAR, DMRSTAR);

        // C1[MC,MR] += scattered & transposed D1[MR,*] summed over grid cols
        Contract(DMRSTAR, DMRMC);
        Axpy(TypeTraits<T>::One(), DMRMC, C1);

        // Bookkeeping.
        team_id = (team_id + 1) % num_stream_teams;

        // Have C wait on all streams
        AddSynchronizationPoint(stream_two, SyncInfo_C);
    }
}

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>, typename=void>
void SUMMA_TNA_impl_multistream(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_TNA_impl_multistream type-device combo not supported.");
}

// Transpose Normal Gemm that avoids communicating the matrix B
template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_TNB_impl_multistream(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;
    constexpr Device D = Device::GPU;

    auto SyncInfo_C = SyncInfoFromMatrix(
        static_cast<Matrix<T,D> const&>(CPre.LockedMatrix()));

    AUTO_PROFILE_REGION(
        "SUMMA.TNB.multistream",
        SyncInfo_C);

    const Int m = CPre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    const bool conjugate = (orientA == ADJOINT);
    auto const num_blocks = (m + bsize - 1) / bsize;

    auto const& stream_pool = GetSyncInfoPool();
    auto const num_stream_teams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size() / 2, size_t(num_blocks));

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    std::vector<DistMatrix<T,MC,STAR,ELEMENT,D>> A1_MC_STAR;
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> D1Trans_MR_STAR;

    A1_MC_STAR.reserve(num_stream_teams);
    D1Trans_MR_STAR.reserve(num_stream_teams);

    // Setup temporaries
    for (auto id = 0UL; id < num_stream_teams; ++id)
    {
        auto A1 = A1_MC_STAR.emplace(A1_MC_STAR.end(), g);
        auto D1 = D1Trans_MR_STAR.emplace(D1Trans_MR_STAR.end(), g);

        auto const& stream_one = stream_pool.Next();
        auto const& stream_two = stream_pool.Next();

        A1->AlignWith(B);
        D1->AlignWith(B);

        SetSyncInfo(A1->Matrix(), stream_one);
        SetSyncInfo(D1->Matrix(), stream_two);
    }

    size_t team_id = 0UL;
    for(Int k=0; k<m; k+=bsize)
    {
        DistMatrix<T,MC,MR,ELEMENT,D> B1(g);
        LockedView(B1, B);

        const Int nb = Min(bsize,m-k);
        auto A1 = A(ALL,        IR(k,k+nb));
        auto C1 = C(IR(k,k+nb), ALL       );

        auto& AMCSTAR = A1_MC_STAR[team_id];
        auto& DTMRSTAR = D1Trans_MR_STAR[team_id];

        auto const& stream_one =
            SyncInfoFromMatrix(AMCSTAR.LockedMatrix());
        auto const& stream_two =
            SyncInfoFromMatrix(DTMRSTAR.LockedMatrix());

        SetSyncInfo(A1.Matrix(), stream_one);
        SetSyncInfo(B1.Matrix(), stream_one);
        SetSyncInfo(C1.Matrix(), stream_two);

        // D1[*,MR] := alpha (A1[MC,*])^[T/H] B[MC,MR]
        //           = alpha (A1^[T/H])[*,MC] B[MC,MR]
        AMCSTAR = A1; // A1[MC,*] <- A1[MC,MR]
        LocalGemm(orientA, NORMAL, TypeTraits<T>::One(), B1, AMCSTAR, DTMRSTAR);
        TransposeAxpyContract(alpha, DTMRSTAR, C1, conjugate);

        // Bookkeeping.
        team_id = (team_id + 1) % num_stream_teams;

        // Syncronize against C
        AddSynchronizationPoint(stream_two, SyncInfo_C);
    }
}

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>, typename=void>
void SUMMA_TNB_impl_multistream(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_TNB_impl_multistream type-device combo not supported.");
}

// Transpose Normal Gemm that avoids communicating the matrix C
template <typename T, typename=EnableIf<IsDeviceValidType<T,Device::GPU>>>
void SUMMA_TNC_impl_multistream(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    EL_DEBUG_CSE;
    constexpr Device D = Device::GPU;

    AUTO_PROFILE_REGION(
        "SUMMA.TNC.multistream",
        SyncInfoFromMatrix(
            static_cast<Matrix<T,D> const&>(CPre.LockedMatrix())));

    const Int sumDim = BPre.Height();
    const Int bsize = Blocksize();
    const Grid& g = APre.Grid();
    auto const num_blocks = (sumDim + bsize - 1) / bsize;

    // Get the sync pool.
    auto const& stream_pool = GetSyncInfoPool();
    auto const num_stream_teams =
        stream_pool.Size() == 1UL
        ? 1UL
        : std::min(stream_pool.Size() / 2, size_t(num_blocks));

    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> AProx(APre);
    DistMatrixReadProxy<T,T,MC,MR,ELEMENT,D> BProx(BPre);
    DistMatrixReadWriteProxy<T,T,MC,MR,ELEMENT,D> CProx(CPre);
    auto& A = AProx.GetLocked();
    auto& B = BProx.GetLocked();
    auto& C = CProx.Get();

    // Temporary distributions
    std::vector<DistMatrix<T,STAR,MC,ELEMENT,D>> A1_STAR_MC;
    std::vector<DistMatrix<T,MR,STAR,ELEMENT,D>> B1Trans_MR_STAR;
    std::vector<DistMatrix<T,MC,MR,ELEMENT,D>> C_TMP;

    A1_STAR_MC.reserve(num_stream_teams);
    B1Trans_MR_STAR.reserve(num_stream_teams);
    C_TMP.reserve(num_stream_teams);

    for (auto id = 0UL; id < num_stream_teams; ++id)
    {
        A1_STAR_MC.emplace_back(g);
        B1Trans_MR_STAR.emplace_back(g);
        C_TMP.emplace_back(g);

        auto& A1 = A1_STAR_MC.back();
        auto& B1 = B1Trans_MR_STAR.back();
        auto& C1 = C_TMP.back();

        auto const& stream_one = stream_pool.Next();
        auto const& stream_two = stream_pool.Next();

        A1.AlignWith(C);
        B1.AlignWith(C);

        SetSyncInfo(A1.Matrix(), stream_one);
        SetSyncInfo(B1.Matrix(), stream_two);

        if (id == 0UL)
        {
            View(C1, C);
            SetSyncInfo(C1.Matrix(), stream_two);
        }
        else
        {
            C1.AlignWith(C);
            SetSyncInfo(C1.Matrix(), stream_two);
            Zeros(C1, C.Height(), C.Width());
        }
    }

    size_t team_id = 0UL;
    for (Int k=0; k<sumDim; k+=bsize)
    {
        const Int nb = Min(bsize,sumDim-k);
        auto A1 = A(IR(k,k+nb), ALL);
        auto B1 = B(IR(k,k+nb), ALL);

        auto& ASTARMC = A1_STAR_MC[team_id];
        auto& BTMRSTAR = B1Trans_MR_STAR[team_id];
        auto& CView = C_TMP[team_id];

        auto const& stream_one =
            SyncInfoFromMatrix(ASTARMC.Matrix());
        auto const& stream_two =
            SyncInfoFromMatrix(CView.Matrix());

        SetSyncInfo(A1.Matrix(), stream_one);
        SetSyncInfo(B1.Matrix(), stream_two);

        // C[MC,MR] += alpha (A1[*,MC])^T B1[*,MR]
        //           = alpha (A1^T)[MC,*] B1[*,MR]
        ASTARMC = A1;
        Transpose(B1, BTMRSTAR);

        LocalGemm(
            orientA, TRANSPOSE,
            alpha, ASTARMC, BTMRSTAR,
            TypeTraits<T>::One(), CView);

        // Bookkeeping.
        team_id = (team_id + 1) % num_stream_teams;
    }

    AddSynchronizationPoint(SyncInfoFromMatrix(C_TMP.front().LockedMatrix()),
                            SyncInfoFromMatrix(C.LockedMatrix()));

    // Compute the reduction into the "real C". This work will
    // serialize on C's stream, so there is no race here.
    for (size_t ii = 1; ii < C_TMP.size(); ++ii)
    {
        Axpy(TypeTraits<T>::One(), C_TMP[ii], C);
    }
}

template <typename T,
          typename=DisableIf<IsDeviceValidType<T,Device::GPU>>, typename=void>
void SUMMA_TNC_impl_multistream(
    Orientation orientA,
    T alpha,
    AbstractDistMatrix<T> const& APre,
    AbstractDistMatrix<T> const& BPre,
    AbstractDistMatrix<T>& CPre)
{
    LogicError("SUMMA_TNC_impl_multistream type-device combo not supported.");
}

} // namespace gemm
} // namespace El

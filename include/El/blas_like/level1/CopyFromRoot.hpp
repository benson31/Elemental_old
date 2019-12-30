
template<typename T>
void CopyFromRoot(const Matrix<T>& A, DistMatrix<T,CIRC,CIRC>& B,
                  bool includingViewers)
{
    EL_DEBUG_CSE;
    if (B.CrossRank() != B.Root())
        LogicError("Called CopyFromRoot from non-root");
    B.Resize(A.Height(), A.Width());
    B.MakeSizeConsistent(includingViewers);
    B.Matrix() = A;
}

template<typename T>
void CopyFromNonRoot(DistMatrix<T,CIRC,CIRC>& B, bool includingViewers)
{
    EL_DEBUG_CSE;
    if (B.CrossRank() == B.Root())
        LogicError("Called CopyFromNonRoot from root");
    B.MakeSizeConsistent(includingViewers);
}

template<typename T>
void CopyFromRoot (const Matrix<T>& A, DistMatrix<T,CIRC,CIRC,BLOCK>& B,
                   bool includingViewers)
{
    EL_DEBUG_CSE;
    if (B.CrossRank() != B.Root())
        LogicError("Called CopyFromRoot from non-root");
    B.Resize(A.Height(), A.Width());
    B.MakeSizeConsistent(includingViewers);
    B.Matrix() = A;
}

template<typename T>
void CopyFromNonRoot (DistMatrix<T,CIRC,CIRC,BLOCK>& B, bool includingViewers)
{
    EL_DEBUG_CSE;
    if (B.CrossRank() == B.Root())
        LogicError("Called CopyFromNonRoot from root");
    B.MakeSizeConsistent(includingViewers);
}

#include "El/core.hpp"
#include "El/blas_like/level1/Copy.hpp"
#include "El/blas_like/level1/CopyAsync.hpp"

namespace El
{
namespace
{

// For now, I just want to generate the tensor-product of
// {MatrixTypes}^2 for Copy. This will cover most of the usecases in
// LBANN. AFAIK, the integer matrices are always converted to a
// floating-point type before they interact "virtually"; all
// operations on those matrices are dispatched statically.

using MatrixTypes = TypeList<
    float, double
#ifdef HYDROGEN_HAVE_HALF
    , cpu_half_type
#endif // HYDROGEN_HAVE_HALF
#ifdef HYDROGEN_GPU_USE_FP16
    , gpu_half_type
#endif // HYDROGEN_GPU_USE_FP16
    >;

template <template <typename> class X, typename... Ts>
using Expand = TypeList<X<Ts>...>;

template <template <typename> class X, typename List>
struct ExpandTLT {};

template <template <typename> class X, typename... Ts>
struct ExpandTLT<X, TypeList<Ts...>>
{
    using type = Expand<X, Ts...>;
};

template <template <typename> class X, typename List>
using ExpandTL = typename ExpandTLT<X, List>::type;

// This is replaced by a generic multiple dispatch engine in
// DiHydrogen; this is a one-off use-case for now, so there's no need
// to backport a robust implementation.
template <typename FunctorT, typename LHSList, typename RHSList>
struct CopyDispatcher
{
    static void Do(FunctorT f,
                   BaseDistMatrix const& src, BaseDistMatrix& tgt)
    {
        using LHead = Head<LHSList>;
        using LTail = Tail<LHSList>;
        if (auto const* ptr = dynamic_cast<LHead const*>(&src))
            return CopyDispatcher<FunctorT, LHSList, RHSList>::DoRHS(
                f, *ptr, tgt);
        else
            return CopyDispatcher<FunctorT, LTail, RHSList>::Do(f, src, tgt);
    }

    template <typename LHSType>
    static void DoRHS(FunctorT f, LHSType const& src, BaseDistMatrix& tgt)
    {
        using RHead = Head<RHSList>;
        using RTail = Tail<RHSList>;
        if (auto* ptr = dynamic_cast<RHead*>(&tgt))
            return f(src, *ptr);
        else
            return CopyDispatcher<FunctorT, LHSList, RTail>::DoRHS(f, src, tgt);
    }
};// struct CopyDispatcher

template <typename FunctorT, typename RHSList>
struct CopyDispatcher<FunctorT, TypeList<>, RHSList>
{
    static void Do(FunctorT const&,
                   BaseDistMatrix const&, BaseDistMatrix const&)
    {
        LogicError("Source matrix type not found.");
    }
};

template <typename FunctorT, typename LHSList>
struct CopyDispatcher<FunctorT, LHSList, TypeList<>>
{
    static void DoRHS(FunctorT const&,
                      BaseDistMatrix const&, BaseDistMatrix const&)
    {
        LogicError("Target matrix type not found.");
    }
};

// This layer of indirection checks the Tgt types and launches the
// copy if possible.
template <typename CopyFunctor,
          typename T, typename U, Device D1, Device D2,
          EnableWhen<IsStorageType<T, D1>, int> = 0>
void LaunchCopy(Matrix<T, D1> const& src, Matrix<U, D2>& tgt,
                CopyFunctor const& F)
{
   return F(src, tgt);
}

template <typename CopyFunctor,
          typename T, typename U, Device D1, Device D2,
          EnableUnless<IsStorageType<T, D1>, int> = 0>
void LaunchCopy(Matrix<T, D1> const&, Matrix<U, D2>&,
                CopyFunctor const&)
{
    LogicError("The combination U=", TypeTraits<U>::Name(), " "
               "and D=", DeviceName<D2>(), " is not supported.");
}

// This layer of indirection checks the Src types; this overload is
// also useful for some DistMatrix instantiations.
template <typename CopyFunctor,
          typename T, typename U, Device D2,
          EnableWhen<IsStorageType<U, D2>, int> = 0>
void LaunchCopy(AbstractMatrix<T> const& src, Matrix<U, D2>& tgt,
                CopyFunctor const& F)
{
    switch (src.GetDevice())
    {
    case Device::CPU:
        return LaunchCopy(
            static_cast<Matrix<T, Device::CPU> const&>(src), tgt, F);
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        return LaunchCopy(
            static_cast<Matrix<T, Device::GPU> const&>(src), tgt, F);
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Copy: Bad device.");
    }
}

template <typename CopyFunctor,
          typename T, typename U, Device D2,
          EnableUnless<IsStorageType<U, D2>, int> = 0>
void LaunchCopy(AbstractMatrix<T> const&, Matrix<U, D2>&,
                CopyFunctor const&)
{
    LogicError("The combination U=", TypeTraits<U>::Name(), " "
               "and D=", DeviceName<D2>(), " is not supported.");
}

// The variadic templates allow these functors to be recycled across
// sequential and distributed matrices.

struct CopyFunctor
{
    template <typename... Args>
    void operator()(Args&&... args) const
    {
        return Copy(std::forward<Args>(args)...);
    }
};// CopyFunctor

struct CopyAsyncFunctor
{
    template <typename... Args>
    void operator()(Args&&... args) const
    {
        return CopyAsync(std::forward<Args>(args)...);
    }
};// CopyAsyncFunctor

}// namespace <anon>

template <typename T, typename U>
void Copy(AbstractMatrix<T> const& Source, AbstractMatrix<U>& Target)
{
    switch (Target.GetDevice())
    {
    case Device::CPU:
        return LaunchCopy(
            Source, static_cast<Matrix<U, Device::CPU>&>(Target),
            CopyFunctor{});
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        return LaunchCopy(
            Source, static_cast<Matrix<U, Device::GPU>&>(Target),
            CopyFunctor{});
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Copy: Bad device.");
    }
}

template <typename T, typename U>
void CopyAsync(AbstractMatrix<T> const& Source, AbstractMatrix<U>& Target)
{
    switch (Target.GetDevice())
    {
    case Device::CPU:
        return LaunchCopy(
            Source, static_cast<Matrix<U, Device::CPU>&>(Target),
            CopyAsyncFunctor{});
#ifdef HYDROGEN_HAVE_GPU
    case Device::GPU:
        return LaunchCopy(
            Source, static_cast<Matrix<U, Device::GPU>&>(Target),
            CopyAsyncFunctor{});
#endif // HYDROGEN_HAVE_GPU
    default:
        LogicError("Copy: Bad device.");
    }
}
void Copy(BaseDistMatrix const& Source, BaseDistMatrix& Target)
{
    using FunctorT = CopyFunctor;
    using MatrixTs = ExpandTL<AbstractDistMatrix, MatrixTypes>;
    using Dispatcher = CopyDispatcher<FunctorT, MatrixTs, MatrixTs>;
    CopyFunctor f;
    return Dispatcher::Do(f, Source, Target);
}

void CopyAsync(BaseDistMatrix const& Source, BaseDistMatrix& Target)
{
    using FunctorT = CopyAsyncFunctor;
    using MatrixTs = ExpandTL<AbstractDistMatrix, MatrixTypes>;
    using Dispatcher = CopyDispatcher<FunctorT, MatrixTs, MatrixTs>;
    CopyAsyncFunctor f;
    return Dispatcher::Do(f, Source, Target);
}

#define PROTO(T, U)                                                     \
    template void Copy(AbstractMatrix<T> const&, AbstractMatrix<U>&);   \
    template void CopyAsync(AbstractMatrix<T> const&, AbstractMatrix<U>&)

#define PROTO_SAME(T) PROTO(T, T)

#define PROTO_BASIC(T)                          \
    PROTO(T, float);                            \
    PROTO(T, double)

#ifdef HYDROGEN_HAVE_HALF
#define PROTO_CPU_HALF(T)                       \
    PROTO(T, cpu_half_type)
#else
#define PROTO_CPU_HALF(T)
#endif // HYDROGEN_HAVE_HALF

#ifdef HYDROGEN_GPU_USE_FP16
#define PROTO_GPU_HALF(T)                       \
    PROTO(T, gpu_half_type)
#else
#define PROTO_GPU_HALF(T)
#endif // HYDROGEN_GPU_USE_FP16

#define PROTO_COMPLETE(T)                       \
    PROTO_BASIC(T);                             \
    PROTO_CPU_HALF(T);                          \
    PROTO_GPU_HALF(T)

#if 0
PROTO_COMPLETE(float);
PROTO_COMPLETE(double);

#ifdef HYDROGEN_HAVE_HALF
PROTO_COMPLETE(cpu_half_type);
#endif // HYDROGEN_HAVE_HALF

#ifdef HYDROGEN_GPU_USE_FP16
PROTO_COMPLETE(gpu_half_type);
#endif // HYDROGEN_GPU_USE_FP16

// Integer types
PROTO_SAME(uint8_t);
PROTO_SAME(int);
#endif // 0

// Complex types
PROTO_SAME(Complex<float>);
PROTO_SAME(Complex<double>);

}// namespace El

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#pragma once
#ifndef EL_IMPORTS_MPI_HPP
#define EL_IMPORTS_MPI_HPP

#include <memory>
#include <tuple>
#include <type_traits>

#include <El/config.h>
#include <El/core/Device.hpp>
#include <El/core/SyncInfo.hpp>
#include <El/core/types.hpp>

#if defined(HYDROGEN_HAVE_AL_MPI_CUDA) || defined(HYDROGEN_HAVE_NCCL2)
#include <El/core/imports/cuda.hpp>
#define HYDROGEN_ALUMINUM_USES_GPU
#endif // defined(HYDROGEN_HAVE_AL_MPI_CUDA) || defined(HYDROGEN_HAVE_NCCL2)

#include <El/core/imports/aluminum.hpp>
#include <El/core/Element/decl.hpp>

namespace El
{

namespace mpi
{

template <typename T, Device D>
struct IsMpiDeviceValidType : IsDeviceValidType<T,D> {};

#ifdef HYDROGEN_HAVE_CUDA
// Signed integer types
template <>
struct IsMpiDeviceValidType<char, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<signed char, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<long int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<short int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<long long int, Device::GPU> : std::true_type {};

// Unsigned types
template <>
struct IsMpiDeviceValidType<unsigned char, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<unsigned short, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<unsigned int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<unsigned long int, Device::GPU> : std::true_type {};
template <>
struct IsMpiDeviceValidType<unsigned long long int, Device::GPU>
    : std::true_type {};
#endif // HYDROGEN_HAVE_CUDA

// Yes, I realize there's some code duplication here, but it's SO MUCH
// EASIER for the human to read I really don't care... The
// preprocessor is the only thing that could care, but it doesn't get
// feelings. Bwahahaha.

#ifndef HYDROGEN_HAVE_ALUMINUM

struct Comm
{
    MPI_Comm comm;
    Comm(MPI_Comm mpiComm=MPI_COMM_WORLD) EL_NO_EXCEPT : comm(mpiComm) { }

    inline int Rank() const EL_NO_RELEASE_EXCEPT;
    inline int Size() const EL_NO_RELEASE_EXCEPT;

    inline void Reinit() const EL_NO_EXCEPT {}
    inline void Reset() const EL_NO_EXCEPT {}
};

#else
namespace internal
{
template <typename IntT, IntT... Is>
struct IntegerSequence {};

template <size_t... Is>
using IndexSequence = IntegerSequence<size_t, Is...>;

template <typename Seq1, typename Seq2>
struct MergeIndexSequences;

template <size_t... Is1, size_t... Is2>
struct MergeIndexSequences<IndexSequence<Is1...>, IndexSequence<Is2...>>
{
  using type = IndexSequence<Is1..., (Is2 + sizeof...(Is1))...>;
};

template <size_t N>
struct GenerateIndexSequence
    : MergeIndexSequences<typename GenerateIndexSequence<N/2>::type,
                          typename GenerateIndexSequence<N-N/2>::type>
{};

template <>
struct GenerateIndexSequence<1>
{
  using type = IndexSequence<0>;
};

template <size_t N>
using MakeIndexSequence = typename GenerateIndexSequence<N>::type;

template <size_t N, typename List>
struct GetIthTypeT
    : GetIthTypeT<N-1, Tail<List>>
{};

template <typename List>
struct GetIthTypeT<0,List>
    : HeadT<List>
{};

template <size_t I, typename List>
using GetIthType = typename GetIthTypeT<I,List>::type;

struct Zero
{
  constexpr static size_t value = 0;
};

template <typename T>
struct PlusOne
{
    constexpr static size_t value = 1 + T::value;
};

template <typename T, typename List>
struct IndexInTypeList
{
    constexpr static size_t value =
        std::conditional<std::is_same<T,Head<List>>::value,
                         Zero,
                         PlusOne<IndexInTypeList<T,Tail<List>>>>::type::value;
};

template <typename T>
struct SharedPtrCommTupleT;

template <typename... Ts>
struct SharedPtrCommTupleT<TypeList<Ts...>>
{
    using type = std::tuple<std::shared_ptr<typename Ts::comm_type>...>;
};

template <typename T>
using SharedPtrCommTuple = typename SharedPtrCommTupleT<T>::type;

template <typename... Ts, size_t I, size_t... Is>
void ConstructAllComms_impl(std::tuple<std::shared_ptr<Ts>...>& tup,
                            MPI_Comm comm, IndexSequence<I,Is...>)
{
    using comm_type_list = TypeList<Ts...>;
    using this_comm_type = GetIthType<I,comm_type_list>;

    std::get<I>(tup) = std::make_shared<this_comm_type>(
        comm
#ifdef HYDROGEN_ALUMINUM_USES_GPU
        , GPUManager::Stream()
#endif
        );

    ConstructAllComms_impl(tup, comm, IndexSequence<Is...>{});
}

template <typename... Ts>
void ConstructAllComms_impl(std::tuple<std::shared_ptr<Ts>...>&,
                            MPI_Comm, IndexSequence<>)
{}

template <typename... Ts>
void ConstructAllComms(std::tuple<std::shared_ptr<Ts>...>& tup, MPI_Comm comm)
{
    ConstructAllComms_impl(tup,comm,MakeIndexSequence<sizeof...(Ts)>{});
}

template <typename... Ts, size_t I, size_t... Is>
void ResetAllPtrs_impl(std::tuple<std::shared_ptr<Ts>...>& tup,
                       IndexSequence<I,Is...>)
{
    std::get<I>(tup).reset();
    ResetAllPtrs_impl(tup, IndexSequence<Is...>{});
}

template <typename... Ts>
void ResetAllPtrs_impl(std::tuple<std::shared_ptr<Ts>...>&,
                       IndexSequence<>)
{}

template <typename... Ts>
void ResetAllPtrs(std::tuple<std::shared_ptr<Ts>...>& tup)
{
    ResetAllPtrs_impl(tup,MakeIndexSequence<sizeof...(Ts)>{});
}

struct DelayCtorType {};
}// namespace internal

struct Comm
{
#ifdef HYDROGEN_ALUMINUM_USES_GPU
    using comm_ptr_tuple_type =
        internal::SharedPtrCommTuple<BackendsForDevice<Device::GPU>>;
#else
    using comm_ptr_tuple_type =
        internal::SharedPtrCommTuple<BackendsForDevice<Device::CPU>>;
#endif // HYDROGEN_ALUMINUM_USES_GPU

    // Hack to handle global objects, MPI_COMM could be int or void*...
    explicit Comm(internal::DelayCtorType const&,
                  MPI_Comm mpicomm) EL_NO_EXCEPT : comm(mpicomm) {}
#ifdef HYDROGEN_HAVE_NCCL2
    Comm(MPI_Comm mpiComm=MPI_COMM_WORLD);
#else
    Comm(MPI_Comm mpiComm=MPI_COMM_WORLD) EL_NO_EXCEPT;
#endif

    // These do soft copies
    Comm(Comm const& comm_cpy) = default;
    Comm& operator=(Comm const& comm_cpy) = default;
    Comm(Comm&& comm_cpy) EL_NO_EXCEPT = default;
    Comm& operator=(Comm&& comm_cpy) EL_NO_EXCEPT = default;

    inline int Rank() const EL_NO_RELEASE_EXCEPT;
    inline int Size() const EL_NO_RELEASE_EXCEPT;

    // Re-sync the Aluminum comms with the MPI comm
    inline void Reinit();
    inline void Reset() EL_NO_EXCEPT;

    template <typename BackendT>
    typename BackendT::comm_type& GetComm() EL_NO_EXCEPT
    {
#ifdef HYDROGEN_ALUMINUM_USES_GPU
        using BackendList = BackendsForDevice<Device::GPU>;
#else
        using BackendList = BackendsForDevice<Device::CPU>;
#endif // HYDROGEN_ALUMINUM_USES_GPU

        return *(std::get<internal::IndexInTypeList<BackendT,BackendList>::value>(al_comms));
    }

    MPI_Comm comm;
    comm_ptr_tuple_type al_comms;
};

    // FIXME (trb): This could be more elegant.
#ifdef HYDROGEN_ALUMINUM_USES_GPU
template <>
inline typename Al::MPIBackend::comm_type& Comm::GetComm<Al::MPIBackend>()
    EL_NO_EXCEPT
{
    // All GPU backend comm_types should work with CPU MPIBackend.
    return *(std::get<0>(al_comms));
}
#endif // HYDROGEN_ALUMINUM_USES_GPU


inline
#ifdef HYDROGEN_HAVE_NCCL2
Comm::Comm(MPI_Comm mpiComm)
#else
Comm::Comm(MPI_Comm mpiComm) EL_NO_EXCEPT
#endif
: comm(mpiComm)
{
    Reinit();
}

inline void Comm::Reinit()
{
    internal::ConstructAllComms(al_comms,comm);
}

inline void Comm::Reset() EL_NO_EXCEPT
{
    internal::ResetAllPtrs(al_comms);
}

// Some SyncInfo stuff
inline SyncInfo<Device::CPU> SyncInfoFromComm(
    Al::MPIBackend::comm_type const&, SyncInfo<Device::CPU> const&)
{
    return SyncInfo<Device::CPU>{};
}

#ifdef HYDROGEN_HAVE_CUDA
inline SyncInfo<Device::GPU> SyncInfoFromComm(
    Al::CUDACommunicator const& comm, SyncInfo<Device::GPU> const& other)
{
    return SyncInfo<Device::GPU>{comm.get_stream(), other.event_};
}
#endif // HYDROGEN_HAVE_CUDA
#endif // HYDROGEN_HAVE_ALUMINUM

inline bool operator==( const Comm& a, const Comm& b ) EL_NO_EXCEPT
{ return a.comm == b.comm; }
inline bool operator!=( const Comm& a, const Comm& b ) EL_NO_EXCEPT
{ return a.comm != b.comm; }
// Hopefully, despite the fact that MPI_Comm is opaque, the following will
// reliably hold (otherwise it must be extended). Typically, MPI_Comm is
// either 'int' or 'void*'.
inline bool operator<( const Comm& a, const Comm& b ) EL_NO_EXCEPT
{ return a.comm < b.comm; }

struct Group
{
    MPI_Group group;
    Group( MPI_Group mpiGroup=MPI_GROUP_NULL ) EL_NO_EXCEPT
    : group(mpiGroup) { }

    inline int Rank() const EL_NO_RELEASE_EXCEPT;
    inline int Size() const EL_NO_RELEASE_EXCEPT;
};
inline bool operator==( const Group& a, const Group& b ) EL_NO_EXCEPT
{ return a.group == b.group; }
inline bool operator!=( const Group& a, const Group& b ) EL_NO_EXCEPT
{ return a.group != b.group; }

struct Op
{
    MPI_Op op;
    Op( MPI_Op mpiOp=MPI_SUM ) EL_NO_EXCEPT : op(mpiOp) { }
};
inline bool operator==( const Op& a, const Op& b ) EL_NO_EXCEPT
{ return a.op == b.op; }
inline bool operator!=( const Op& a, const Op& b ) EL_NO_EXCEPT
{ return a.op != b.op; }

// Datatype definitions
// TODO(poulson): Convert these to structs/classes
typedef MPI_Aint Aint;
typedef MPI_Datatype Datatype;
typedef MPI_Errhandler ErrorHandler;
typedef MPI_Status Status;
typedef MPI_User_function UserFunction;

template<typename T>
struct Request
{
    Request() { }

    MPI_Request backend;

    std::vector<byte> buffer;
    bool receivingPacked=false;
    int recvCount;
    T* unpackedRecvBuf;
};

// Standard constants
extern const int ANY_SOURCE;
extern const int ANY_TAG;
extern const int THREAD_SINGLE;
extern const int THREAD_FUNNELED;
extern const int THREAD_SERIALIZED;
extern const int THREAD_MULTIPLE;

extern const int UNDEFINED;
extern const Group GROUP_NULL;
extern const Comm COMM_NULL;// = MPI_COMM_NULL;
extern const Comm COMM_SELF;// = MPI_COMM_SELF;
extern const Comm COMM_WORLD;// = MPI_COMM_WORLD;
extern const ErrorHandler ERRORS_RETURN;
extern const ErrorHandler ERRORS_ARE_FATAL;
extern const Group GROUP_EMPTY;
extern const Op MAX;
extern const Op MIN;
extern const Op MAXLOC;
extern const Op MINLOC;
extern const Op PROD;
extern const Op SUM;
extern const Op LOGICAL_AND;
extern const Op LOGICAL_OR;
extern const Op LOGICAL_XOR;
extern const Op BINARY_AND;
extern const Op BINARY_OR;
extern const Op BINARY_XOR;

template<typename T>
struct Types
{
    static bool createdTypeBeforeResize;
    static El::mpi::Datatype typeBeforeResize;

    static bool createdType;
    static El::mpi::Datatype type;

    static bool haveSumOp;
    static bool createdSumOp;
    static El::mpi::Op sumOp;

    static bool haveProdOp;
    static bool createdProdOp;
    static El::mpi::Op prodOp;

    static bool haveMinOp;
    static bool createdMinOp;
    static El::mpi::Op minOp;

    static bool haveMaxOp;
    static bool createdMaxOp;
    static El::mpi::Op maxOp;

    static bool haveUserOp;
    static bool createdUserOp;
    static El::mpi::Op userOp;

    static bool haveUserCommOp;
    static bool createdUserCommOp;
    static El::mpi::Op userCommOp;

    static std::function<T(const T&,const T&)> userFunc, userCommFunc;

    // Internally called once per type between MPI_Init and MPI_Finalize
    static void Destroy();
};

template<typename T>
struct MPIBaseHelper { typedef T value; };
template<typename T>
struct MPIBaseHelper<ValueInt<T>> { typedef T value; };
template<typename T>
struct MPIBaseHelper<Entry<T>> { typedef T value; };
template<typename T>
using MPIBase = typename MPIBaseHelper<T>::value;

template<typename T>
Datatype& TypeMap() EL_NO_EXCEPT
{ return Types<T>::type; }

template<typename T>
Op& UserOp() { return Types<T>::userOp; }
template<typename T>
Op& UserCommOp() { return Types<T>::userCommOp; }
template<typename T>
Op& SumOp() { return Types<T>::sumOp; }
template<typename T>
Op& ProdOp() { return Types<T>::prodOp; }
// The following are currently only defined for real datatypes but could
// potentially use lexicographic ordering for complex numbers
template<typename T>
Op& MaxOp() { return Types<T>::maxOp; }
template<typename T>
Op& MinOp() { return Types<T>::minOp; }
template<typename T>
Op& MaxLocOp() { return Types<ValueInt<T>>::maxOp; }
template<typename T>
Op& MinLocOp() { return Types<ValueInt<T>>::minOp; }
template<typename T>
Op& MaxLocPairOp() { return Types<Entry<T>>::maxOp; }
template<typename T>
Op& MinLocPairOp() { return Types<Entry<T>>::minOp; }

// Added constant(s)
const int MIN_COLL_MSG = 1; // minimum message size for collectives
inline int Pad( int count ) EL_NO_EXCEPT
{ return std::max(count,MIN_COLL_MSG); }

bool CommSameSizeAsInteger() EL_NO_EXCEPT;
bool GroupSameSizeAsInteger() EL_NO_EXCEPT;

// Environment routines
void Initialize( int& argc, char**& argv ) EL_NO_EXCEPT;
int InitializeThread( int& argc, char**& argv, int required ) EL_NO_EXCEPT;
void Finalize() EL_NO_EXCEPT;
bool Initialized() EL_NO_EXCEPT;
bool Finalized() EL_NO_EXCEPT;
int QueryThread() EL_NO_EXCEPT;
void Abort( Comm comm, int errCode ) EL_NO_EXCEPT;
double Time() EL_NO_EXCEPT;
void Create( UserFunction* func, bool commutes, Op& op ) EL_NO_RELEASE_EXCEPT;
void Free( Op& op ) EL_NO_RELEASE_EXCEPT;
void Free( Datatype& type ) EL_NO_RELEASE_EXCEPT;

// Communicator manipulation
int Rank( Comm comm=COMM_WORLD ) EL_NO_RELEASE_EXCEPT;
int Size( Comm comm=COMM_WORLD ) EL_NO_RELEASE_EXCEPT;
void Create
( Comm parentComm, Group subsetGroup, Comm& subsetComm ) EL_NO_RELEASE_EXCEPT;
void Dup( Comm original, Comm& duplicate ) EL_NO_RELEASE_EXCEPT;
void Split( Comm comm, int color, int key, Comm& newComm ) EL_NO_RELEASE_EXCEPT;
void Free( Comm& comm ) EL_NO_RELEASE_EXCEPT;
bool Congruent( Comm comm1, Comm comm2 ) EL_NO_RELEASE_EXCEPT;
void ErrorHandlerSet
( Comm comm, ErrorHandler errorHandler ) EL_NO_RELEASE_EXCEPT;

// Cartesian communicator routines
void CartCreate
( Comm comm, int numDims, const int* dimensions, const int* periods,
  bool reorder, Comm& cartComm ) EL_NO_RELEASE_EXCEPT;
void CartSub
( Comm comm, const int* remainingDims, Comm& subComm ) EL_NO_RELEASE_EXCEPT;

// Group manipulation
int Rank( Group group ) EL_NO_RELEASE_EXCEPT;
int Size( Group group ) EL_NO_RELEASE_EXCEPT;
void CommGroup( Comm comm, Group& group ) EL_NO_RELEASE_EXCEPT;
void Dup( Group group, Group& newGroup ) EL_NO_RELEASE_EXCEPT;
void Union( Group groupA, Group groupB, Group& newGroup ) EL_NO_RELEASE_EXCEPT;
void Incl
( Group group, int n, const int* ranks, Group& subGroup ) EL_NO_RELEASE_EXCEPT;
void Excl
( Group group, int n, const int* ranks, Group& subGroup ) EL_NO_RELEASE_EXCEPT;
void Difference
( Group parent, Group subset, Group& complement ) EL_NO_RELEASE_EXCEPT;
void Free( Group& group ) EL_NO_RELEASE_EXCEPT;
bool Congruent( Group group1, Group group2 ) EL_NO_RELEASE_EXCEPT;
int Translate
( Group origGroup, int origRank, Group newGroup ) EL_NO_RELEASE_EXCEPT;
int Translate
( Comm  origComm,  int origRank, Group newGroup ) EL_NO_RELEASE_EXCEPT;
int Translate
( Group origGroup, int origRank, Comm  newComm  ) EL_NO_RELEASE_EXCEPT;
int Translate
( Comm  origComm,  int origRank, Comm  newComm  ) EL_NO_RELEASE_EXCEPT;
void Translate
( Group origGroup, int size, const int* origRanks,
  Group newGroup,                  int* newRanks ) EL_NO_RELEASE_EXCEPT;
void Translate
( Comm origComm,  int size, const int* origRanks,
  Group newGroup,                 int* newRanks ) EL_NO_RELEASE_EXCEPT;
void Translate
( Group origGroup, int size, const int* origRanks,
  Comm newComm,                    int* newRanks ) EL_NO_RELEASE_EXCEPT;
void Translate
( Comm origComm, int size, const int* origRanks,
  Comm newComm,                  int* newRanks ) EL_NO_RELEASE_EXCEPT;

// Utilities
void Barrier( Comm comm=COMM_WORLD ) EL_NO_RELEASE_EXCEPT;

template<typename T>
void Wait( Request<T>& request ) EL_NO_RELEASE_EXCEPT;

template<typename T,
         typename=EnableIf<IsPacked<T>>>
void Wait( Request<T>& request, Status& status ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Wait( Request<T>& request, Status& status ) EL_NO_RELEASE_EXCEPT;

template<typename T>
void WaitAll( int numRequests, Request<T>* requests ) EL_NO_RELEASE_EXCEPT;

template<typename T,
         typename=EnableIf<IsPacked<T>>>
void WaitAll( int numRequests, Request<T>* requests, Status* statuses )
EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void WaitAll( int numRequests, Request<T>* requests, Status* statuses )
EL_NO_RELEASE_EXCEPT;

template<typename T>
bool Test( Request<T>& request ) EL_NO_RELEASE_EXCEPT;
bool IProbe
( int source, int tag, Comm comm, Status& status ) EL_NO_RELEASE_EXCEPT;

template<typename T>
int GetCount( Status& status ) EL_NO_RELEASE_EXCEPT;

template<typename T>
void SetUserReduceFunc
( std::function<T(const T&,const T&)> func, bool commutative=true )
{
    if( commutative )
        Types<T>::userCommFunc = func;
    else
        Types<T>::userFunc = func;
}

// Point-to-point communication
// ============================

// Send
// ----
template <typename Real, Device D,
          typename=EnableIf<IsPacked<Real>>>
void TaggedSend(
    const Real* buf, int count, int to, int tag, Comm comm,
    SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
          typename=EnableIf<IsPacked<Real>>>
void TaggedSend(
    const Complex<Real>* buf, int count, int to, int tag, Comm comm,
    SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedSend(
    const T* buf, int count, int to, int tag, Comm comm, SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;


// If the tag is irrelevant
template<typename T, Device D>
void Send( const T* buf, int count, int to, Comm comm, SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;

// If the send-count is one
template<typename T>
void TaggedSend( T b, int to, int tag, Comm comm )
    EL_NO_RELEASE_EXCEPT;

// If the send-count is one and the tag is irrelevant
template<typename T>
void Send( T b, int to, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Non-blocking send
// -----------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedISend
( const Real* buf, int count, int to, int tag, Comm comm,
  Request<Real>& request ) EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedISend
( const Complex<Real>* buf, int count, int to, int tag, Comm comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedISend
( const T* buf, int count, int to, int tag, Comm comm,
  Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void ISend( const T* buf, int count, int to, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one
template<typename T>
void TaggedISend( T b, int to, int tag, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one and the tag is irrelevant
template<typename T>
void ISend( T b, int to, Comm comm, Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// Non-blocking ready-mode send
// ----------------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedIRSend
( const Real* buf, int count, int to, int tag, Comm comm,
  Request<Real>& request ) EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedIRSend
( const Complex<Real>* buf, int count, int to, int tag, Comm comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedIRSend
( const T* buf, int count, int to, int tag, Comm comm,
  Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void IRSend( const T* buf, int count, int to, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one
template<typename T>
void TaggedIRSend( T b, int to, int tag, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one and the tag is irrelevant
template<typename T>
void IRSend( T b, int to, Comm comm, Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// Non-blocking synchronous Send
// -----------------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedISSend
( const Real* buf, int count, int to, int tag, Comm comm,
  Request<Real>& request )
EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedISSend
( const Complex<Real>* buf, int count, int to, int tag, Comm comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedISSend
( const T* buf, int count, int to, int tag, Comm comm,
  Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void ISSend( const T* buf, int count, int to, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one
template<typename T>
void TaggedISSend( T b, int to, int tag, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the send count is one and the tag is irrelevant
template<typename T>
void ISSend( T b, int to, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// Recv
// ----
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void TaggedRecv
( Real* buf, int count, int from, int tag, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void TaggedRecv
( Complex<Real>* buf, int count, int from, int tag, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedRecv
( T* buf, int count, int from, int tag, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template <typename T, Device D>
void Recv( T* buf, int count, int from, Comm comm, SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;

// If the recv count is one
template<typename T>
T TaggedRecv( int from, int tag, Comm comm ) EL_NO_RELEASE_EXCEPT;

// If the recv count is one and the tag is irrelevant
template<typename T>
T Recv( int from, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Non-blocking recv
// -----------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedIRecv
( Real* buf, int count, int from, int tag, Comm comm,
  Request<Real>& request ) EL_NO_RELEASE_EXCEPT;
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void TaggedIRecv
( Complex<Real>* buf, int count, int from, int tag, Comm comm,
  Request<Complex<Real>>& request ) EL_NO_RELEASE_EXCEPT;
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void TaggedIRecv
( T* buf, int count, int from, int tag, Comm comm,
  Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// If the tag is irrelevant
template<typename T>
void IRecv( T* buf, int count, int from, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the recv count is one
template<typename T>
T TaggedIRecv( int from, int tag, Comm comm, Request<T>& request )
EL_NO_RELEASE_EXCEPT;

// If the recv count is one and the tag is irrelevant
template<typename T>
T IRecv( int from, Comm comm, Request<T>& request ) EL_NO_RELEASE_EXCEPT;

// SendRecv
// --------
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void TaggedSendRecv(
    const Real* sbuf, int sc, int to,   int stag,
    Real* rbuf, int rc, int from, int rtag, Comm comm, SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
          typename=EnableIf<IsPacked<Real>>>
void TaggedSendRecv(
    const Complex<Real>* sbuf, int sc, int to,   int stag,
    Complex<Real>* rbuf, int rc, int from, int rtag, Comm comm,
    SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
          typename=DisableIf<IsPacked<T>>,
          typename=void>
void TaggedSendRecv(
    const T* sbuf, int sc, int to,   int stag,
    T* rbuf, int rc, int from, int rtag, Comm comm, SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;

// If the tags are irrelevant
#define COLL Collective::SENDRECV

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm comm,
              SyncInfo<D> const& syncInfo);

// Aluminum in-place
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
void SendRecv(T* buf, int count, int to, int from, Comm comm,
              SyncInfo<D> const& syncInfo);

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm comm,
              SyncInfo<D> const& syncInfo);

// Non-aluminum, device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=void>
void SendRecv(const T* sbuf, int sc, int to,
              T* rbuf, int rc, int from, Comm comm,
              SyncInfo<D> const& syncInfo);

// Non-aluminum, non-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
void SendRecv( T* buf, int count, int to, int from, Comm comm,
               SyncInfo<D> const& syncInfo);

// Non-aluminum, device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=void>
void SendRecv( T* buf, int count, int to, int from, Comm comm,
               SyncInfo<D> const& syncInfo);

#undef COLL // Collective::SENDRECV

// If the send and recv counts are one
template <typename T, Device D>
T TaggedSendRecv(
    T sb, int to, int stag, int from, int rtag, Comm comm, SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;

// If the send and recv counts are one and the tags don't matter
template <typename T, Device D>
T SendRecv( T sb, int to, int from, Comm comm, SyncInfo<D> const& );

// Single-buffer SendRecv
// ----------------------
template <typename Real, Device D,
          typename=EnableIf<IsPacked<Real>>>
void TaggedSendRecv(
    Real* buf, int count, int to, int stag, int from, int rtag, Comm comm,
    SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
          typename=EnableIf<IsPacked<Real>>>
void TaggedSendRecv(
    Complex<Real>* buf, int count, int to, int stag, int from, int rtag,
    Comm comm, SyncInfo<D> const& ) EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
          typename=DisableIf<IsPacked<T>>,
          typename=void>
void TaggedSendRecv(
    T* buf, int count, int to, int stag, int from, int rtag, Comm comm,
    SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;

// Collective communication
// ========================

// Broadcast
// ---------
#define COLL Collective::BROADCAST
#define COLLECTIVE_SIGNATURE                            \
    void Broadcast(                                     \
        T* buffer, int count, int root, Comm comm,      \
        SyncInfo<D> const&)
#define COLLECTIVE_SIGNATURE_COMPLEX                            \
    void Broadcast(                                             \
        Complex<T>* buffer, int count, int root, Comm comm,     \
        SyncInfo<D> const&)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

// If the message length is one
template<typename T, Device D>
void Broadcast( T& b, int root, Comm comm, SyncInfo<D> const& );

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE
#undef COLL // Collective::BROADCAST

// Non-blocking broadcast
// ----------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void IBroadcast
( Real* buf, int count, int root, Comm comm, Request<Real>& request );
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void IBroadcast
( Complex<Real>* buf, int count, int root, Comm comm,
  Request<Complex<Real>>& request );
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void IBroadcast
( T* buf, int count, int root, Comm comm, Request<T>& request );

// If the message length is one
template<typename T>
void IBroadcast( T& b, int root, Comm comm, Request<T>& request );

// Gather
// ------

#define COLL Collective::GATHER
#define COLLECTIVE_SIGNATURE                    \
    void Gather(                                \
        const T* sbuf, int sc,                  \
        T* rbuf, int rc, int root, Comm comm,   \
        SyncInfo<D> const& syncInfo)
#define COLLECTIVE_SIGNATURE_COMPLEX                     \
    void Gather(                                         \
        const Complex<T>* sbuf, int sc,                  \
        Complex<T>* rbuf, int rc, int root, Comm comm,   \
        SyncInfo<D> const& syncInfo)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Even though EL_AVOID_COMPLEX_MPI being defined implies that an std::vector
// copy of the input data will be created, and the memory allocation can clearly
// fail and throw an exception, said exception is not necessarily thrown on
// Linux platforms due to the "optimistic" allocation policy. Therefore we will
// go ahead and allow std::terminate to be called should such an std::bad_alloc
// exception occur in a Release build

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE
#undef COLL /* Collective::GATHER */

// Non-blocking gather
// -------------------
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void IGather
( const Real* sbuf, int sc,
        Real* rbuf, int rc, int root, Comm comm,
  Request<Real>& request );
template<typename Real,
         typename=EnableIf<IsPacked<Real>>>
void IGather
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, int rc,
  int root, Comm comm,
  Request<Complex<Real>>& request );
template<typename T,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void IGather
( const T* sbuf, int sc,
        T* rbuf, int rc, int root, Comm comm,
  Request<T>& request );

// Gather with variable recv sizes
// -------------------------------
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void Gather
( const Real* sbuf, int sc,
        Real* rbuf, const int* rcs, const int* rds,
  int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void Gather
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, const int* rcs, const int* rds,
  int root, Comm comm, SyncInfo<D> const& ) EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Gather
( const T* sbuf, int sc,
        T* rbuf, const int* rcs, const int* rds,
  int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

// AllGather
// ---------
// NOTE: See the corresponding note for Gather on std::bad_alloc exceptions

#define COLL Collective::ALLGATHER
#define COLLECTIVE_SIGNATURE                                    \
    void AllGather(                                             \
        T const* sbuf, int sc, T* rbuf, int rc, Comm comm,      \
        SyncInfo<D> const& syncInfo)
#define COLLECTIVE_SIGNATURE_COMPLEX                                    \
    void AllGather(                                                     \
        Complex<T> const* sbuf, int sc, Complex<T>* rbuf, int rc, Comm comm, \
        SyncInfo<D> const& syncInfo)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE
#undef COLL // Collective::ALLGATHER

// AllGather with variable recv sizes
// ----------------------------------
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void AllGather
( const Real* sbuf, int sc,
        Real* rbuf, const int* rcs, const int* rds, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void AllGather
( const Complex<Real>* sbuf, int sc,
        Complex<Real>* rbuf, const int* rcs, const int* rds,
  Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void AllGather
( const T* sbuf, int sc,
        T* rbuf, const int* rcs, const int* rds, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

// Scatter
// -------
#define COLL Collective::SCATTER
#define COLLECTIVE_SIGNATURE                    \
    void Scatter(                               \
        const T* sbuf, int sc,                  \
        T* rbuf, int rc, int root, Comm comm,   \
        SyncInfo<D> const& syncInfo)
#define COLLECTIVE_SIGNATURE_COMPLEX                    \
    void Scatter(                                       \
        const Complex<T>* sbuf, int sc,                 \
        Complex<T>* rbuf, int rc, int root, Comm comm,  \
        SyncInfo<D> const& syncInfo)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

// In-place option
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void Scatter( Real* buf, int sc, int rc, int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void Scatter( Complex<Real>* buf, int sc, int rc, int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Scatter( T* buf, int sc, int rc, int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE
#undef COLL // Collective::SCATTER

// TODO(poulson): MPI_Scatterv support

// AllToAll
// --------
// NOTE: See the corresponding note on std::bad_alloc for Gather
#define COLL Collective::ALLTOALL
#define COLLECTIVE_SIGNATURE                                    \
    void AllToAll(                                              \
        T const* sbuf, int sc, T* rbuf, int rc, Comm comm,      \
        SyncInfo<D> const&)
#define COLLECTIVE_SIGNATURE_COMPLEX                                    \
    void AllToAll(                                                      \
        Complex<T> const* sbuf, int sc, Complex<T>* rbuf, int rc,       \
        Comm comm, SyncInfo<D> const&)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE
#undef COLL // Collective::ALLTOALL

// AllToAll with non-uniform send/recv sizes
// -----------------------------------------
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void AllToAll
( const Real* sbuf, const int* scs, const int* sds,
        Real* rbuf, const int* rcs, const int* rds, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void AllToAll
( const Complex<Real>* sbuf, const int* scs, const int* sds,
        Complex<Real>* rbuf, const int* rcs, const int* rds,
  Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void AllToAll
( const T* sbuf, const int* scs, const int* sds,
        T* rbuf, const int* rcs, const int* rds, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

template<typename T>
std::vector<T> AllToAll
( const std::vector<T>& sendBuf,
  const std::vector<int>& sendCounts,
  const std::vector<int>& sendDispls,
  Comm comm ) EL_NO_RELEASE_EXCEPT;

// Reduce
// ------
#define COLL Collective::REDUCE
#define COLLECTIVE_SIGNATURE                                    \
    void Reduce(                                                \
        T const* sbuf, T* rbuf, int count, Op op,               \
        int root, Comm comm, SyncInfo<D> const& syncInfo)
#define COLLECTIVE_SIGNATURE_COMPLEX                                    \
    void Reduce(                                                        \
        Complex<T> const* sbuf, Complex<T>* rbuf, int count, Op op,     \
        int root, Comm comm, SyncInfo<D> const& syncInfo)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE

template <typename T, Device D,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void Reduce
( const T* sb, T* rb, int count, OpClass op, bool commutative,
  int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        Reduce( sb, rb, count, UserCommOp<T>(), root, comm );
    else
        Reduce( sb, rb, count, UserOp<T>(), root, comm );
}

// Default to SUM
template <typename T, Device D>
void Reduce(
    const T* sbuf, T* rbuf, int count, int root, Comm comm,
    SyncInfo<D> const& syncInfo);

// With a message-size of one
template <typename T, Device D>
T Reduce( T sb, Op op, int root, Comm comm,
          SyncInfo<D> const& syncInfo );

template <typename T, Device D,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
T Reduce
( T sb, OpClass op, bool commutative, int root, Comm comm )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        return Reduce( sb, UserCommOp<T>(), root, comm );
    else
        return Reduce( sb, UserOp<T>(), root, comm );
}

// With a message-size of one and default to SUM
template <typename T, Device D>
T Reduce( T sb, int root, Comm comm, SyncInfo<D> const& syncInfo );

// Single-buffer reduce
// --------------------
#define COLLECTIVE_SIGNATURE                                    \
    void Reduce(                                                \
        T* buf, int count, Op op,                               \
        int root, Comm comm, SyncInfo<D> const& syncInfo)
#define COLLECTIVE_SIGNATURE_COMPLEX                                    \
    void Reduce(                                                        \
        Complex<T>* buf, int count, Op op,                              \
        int root, Comm comm, SyncInfo<D> const& syncInfo)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

template <typename T, Device D,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void Reduce
( T* buf, int count, OpClass op, bool commutative, int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        Reduce( buf, count, UserCommOp<T>(), root, comm );
    else
        Reduce( buf, count, UserOp<T>(), root, comm );
}

// Default to SUM
template <typename T, Device D>
void Reduce( T* buf, int count, int root, Comm comm,
             SyncInfo<D> const& syncInfo );

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE
#undef COLL // Collective::REDUCE

// AllReduce
// ---------

#define COLL Collective::ALLREDUCE
#define COLLECTIVE_SIGNATURE                                    \
    void AllReduce(                                             \
        T const* sbuf, T* rbuf, int count, Op op, Comm comm,    \
        SyncInfo<D> const&)
#define COLLECTIVE_SIGNATURE_COMPLEX                                    \
    void AllReduce(                                                     \
        Complex<T> const* sbuf, Complex<T>* rbuf, int count, Op op,     \
        Comm comm, SyncInfo<D> const&)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE

//
// The "IN_PLACE" allreduce
//

#define COLLECTIVE_SIGNATURE                            \
    void AllReduce(T* buf, int count, Op op, Comm comm, \
                   SyncInfo<D> const& syncInfo)
#define COLLECTIVE_SIGNATURE_COMPLEX                             \
    void AllReduce(Complex<T>* buf, int count, Op op, Comm comm, \
                   SyncInfo<D> const& syncInfo)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

template <typename T, Device D>
void AllReduce(const T* sbuf, T* rbuf, int count, Comm comm,
               SyncInfo<D> const& syncInfo);

template <typename T, Device D>
T AllReduce(T sb, Op op, Comm comm, SyncInfo<D> const& syncInfo);

template <typename T, Device D>
T AllReduce(T sb, Comm comm, SyncInfo<D> const& syncInfo);

template <typename T, Device D>
void AllReduce(T* buf, int count, Comm comm, SyncInfo<D> const& syncInfo);

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE
#undef COLL // Collective::ALLREDUCE

// ReduceScatter
// -------------
#define COLL Collective::REDUCESCATTER
#define COLLECTIVE_SIGNATURE                                    \
    void ReduceScatter(                                         \
        T const* sbuf, T* rbuf, int rc, Op op, Comm comm,       \
        SyncInfo<D> const& syncInfo )
#define COLLECTIVE_SIGNATURE_COMPLEX                                    \
    void ReduceScatter(                                                 \
        Complex<T> const* sbuf, Complex<T>* rbuf, int rc, Op op,        \
            Comm comm, SyncInfo<D> const& syncInfo )

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE

// FIXME: WHAT TO DO HERE??
template<typename T, Device D, class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void ReduceScatter(
    const T* sb, T* rb, int count, OpClass op, bool commutative, Comm comm,
    SyncInfo<D> const& syncInfo)
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        ReduceScatter( sb, rb, count, UserCommOp<T>(), comm, syncInfo );
    else
        ReduceScatter( sb, rb, count, UserOp<T>(), comm, syncInfo );
}

// Default to SUM
template <typename T, Device D>
void ReduceScatter( T const* sbuf, T* rbuf, int rc, Comm comm,
                    SyncInfo<D> const& syncInfo);

// Single-buffer ReduceScatter
// ---------------------------

#define COLLECTIVE_SIGNATURE                    \
    void ReduceScatter(                         \
        T* buf, int count, Op op, Comm comm,    \
        SyncInfo<D> const& syncInfo)
#define COLLECTIVE_SIGNATURE_COMPLEX                     \
    void ReduceScatter(                                  \
        Complex<T>* buf, int count, Op op, Comm comm,    \
        SyncInfo<D> const& syncInfo)

// Aluminum
template <typename T, Device D,
          typename=EnableIf<IsAluminumSupported<T,D,COLL>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, not-device-ok
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=DisableIf<IsMpiDeviceValidType<T,D>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, not-packed
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=DisableIf<IsPacked<T>>>
COLLECTIVE_SIGNATURE;

// Non-aluminum, device-ok, packed, complex
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<Complex<T>,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<Complex<T>,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=void>
COLLECTIVE_SIGNATURE_COMPLEX;

// Non-aluminum, device-ok, packed, real
template <typename T, Device D,
          typename=DisableIf<IsAluminumSupported<T,D,COLL>>,
          typename=EnableIf<IsMpiDeviceValidType<T,D>>,
          typename=EnableIf<IsPacked<T>>,
          typename=DisableIf<IsComplex<T>>,
          typename=void>
COLLECTIVE_SIGNATURE;

// FIXME: WHAT TO DO HERE??
template<typename T, Device D, class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void ReduceScatter(
    T* buf, int count, OpClass op, bool commutative, Comm comm,
    SyncInfo<D> const& syncInfo)
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        ReduceScatter( buf, count, UserCommOp<T>(), comm, syncInfo );
    else
        ReduceScatter( buf, count, UserOp<T>(), comm, syncInfo );
}

// Default to SUM
template <typename T, Device D>
void ReduceScatter(T* buf, int rc, Comm comm, SyncInfo<D> const&);

#undef COLLECTIVE_SIGNATURE_COMPLEX
#undef COLLECTIVE_SIGNATURE
#undef COLL // Collective::REDUCESCATTER

// Variable-length ReduceScatter
// -----------------------------
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void ReduceScatter
( const Real* sbuf, Real* rbuf, const int* rcs, Op op, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void ReduceScatter
( const Complex<Real>* sbuf, Complex<Real>* rbuf, const int* rcs, Op op,
  Comm comm, SyncInfo<D> const& ) EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void ReduceScatter
( const T* sbuf, T* rbuf, const int* rcs, Op op, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

template <typename T, Device D,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void ReduceScatter
( const T* sb, T* rb, const int* rcs, OpClass op, bool commutative,
  Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        ReduceScatter( sb, rb, rcs, UserCommOp<T>(), comm );
    else
        ReduceScatter( sb, rb, rcs, UserOp<T>(), comm );
}

// Default to SUM
template <typename T, Device D>
void ReduceScatter(
    const T* sbuf, T* rbuf, const int* rcs, Comm comm, SyncInfo<D> const& )
    EL_NO_RELEASE_EXCEPT;

// Scan
// ----
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void Scan( const Real* sbuf, Real* rbuf, int count, Op op, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void Scan
( const Complex<Real>* sbuf, Complex<Real>* rbuf, int count, Op op, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Scan( const T* sbuf, T* rbuf, int count, Op op, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

template <typename T, Device D,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void Scan(
    const T* sb, T* rb, int count, OpClass op, bool commutative,
    int root, Comm comm, SyncInfo<D> const& syncInfo)
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        Scan(sb, rb, count, UserCommOp<T>(), root, comm, syncInfo);
    else
        Scan(sb, rb, count, UserOp<T>(), root, comm, syncInfo);
}

// Default to SUM
template <typename T, Device D>
void Scan( const T* sbuf, T* rbuf, int count, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

// With a message-size of one
template<typename T>
T Scan( T sb, Op op, Comm comm ) EL_NO_RELEASE_EXCEPT;

template <typename T, Device D,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
T Scan( T sb, OpClass op, bool commutative, int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        return Scan( sb, UserCommOp<T>(), root, comm );
    else
        return Scan( sb, UserOp<T>(), root, comm );
}

// With a message-size of one and default to SUM
template<typename T>
T Scan( T sb, Comm comm ) EL_NO_RELEASE_EXCEPT;

// Single-buffer scan
// ------------------
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void Scan( Real* buf, int count, Op op, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename Real, Device D,
         typename=EnableIf<IsPacked<Real>>>
void Scan( Complex<Real>* buf, int count, Op op, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;
template <typename T, Device D,
         typename=DisableIf<IsPacked<T>>,
         typename=void>
void Scan( T* buf, int count, Op op, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT;

template <typename T, Device D,class OpClass,
         typename=DisableIf<IsData<OpClass>>>
void Scan
( T* buf, int count, OpClass op, bool commutative, int root, Comm comm, SyncInfo<D> const& )
EL_NO_RELEASE_EXCEPT
{
    SetUserReduceFunc( std::function<T(const T&,const T&)>(op), commutative );
    if( commutative )
        Scan( buf, count, UserCommOp<T>(), root, comm );
    else
        Scan( buf, count, UserOp<T>(), root, comm );
}

// Default to SUM
template<typename T, Device D>
void Scan(T* buf, int count, Comm comm, SyncInfo<D> const&)
    EL_NO_RELEASE_EXCEPT;

template<typename T>
void SparseAllToAll
( const std::vector<T>& sendBuffer,
  const std::vector<int>& sendCounts,
  const std::vector<int>& sendOffs,
        std::vector<T>& recvBuffer,
  const std::vector<int>& recvCounts,
  const std::vector<int>& recvOffs,
        Comm comm ) EL_NO_RELEASE_EXCEPT;

void VerifySendsAndRecvs
( const std::vector<int>& sendCounts,
  const std::vector<int>& recvCounts, Comm comm );

void CreateCustom() EL_NO_RELEASE_EXCEPT;
void DestroyCustom() EL_NO_RELEASE_EXCEPT;

#ifdef HYDROGEN_HAVE_MPC
void CreateBigIntFamily();
void DestroyBigIntFamily();
void CreateBigFloatFamily();
void DestroyBigFloatFamily();
#endif

// Convenience functions which might not be very useful
int Comm::Rank() const EL_NO_RELEASE_EXCEPT { return mpi::Rank(*this); }
int Comm::Size() const EL_NO_RELEASE_EXCEPT { return mpi::Size(*this); }
int Group::Rank() const EL_NO_RELEASE_EXCEPT { return mpi::Rank(*this); }
int Group::Size() const EL_NO_RELEASE_EXCEPT { return mpi::Size(*this); }

} // mpi
} // elem

#endif // ifndef EL_IMPORTS_MPI_HPP

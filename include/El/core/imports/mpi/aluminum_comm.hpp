#pragma once
#ifndef EL_IMPORTS_MPI_ALUMINUM_COMM_HPP_
#define EL_IMPORTS_MPI_ALUMINUM_COMM_HPP_

#include <El/config.h>
#include <El/core/imports/aluminum.hpp>
#include <El/core/imports/mpi/comm_impl.hpp>
#include <El/core/imports/mpi/meta.hpp>
#include <El/core/Device.hpp>
#include <El/core/SyncInfo.hpp>

#include <forward_list>
#include <memory>
#include <type_traits>

namespace El
{
namespace mpi
{
namespace internal
{

/** @class SharedPtrCommTupleT
 *  @brief A tuple of shared_ptrs to comms
 *
 *  Used to extract the comm_type from each Aluminum backend.
 *
 *  @tparam T A typelist containing all relevant Aluminum backends.
 */
template <typename T>
struct SharedPtrCommTupleT;

template <typename... BackendTs>
struct SharedPtrCommTupleT<TypeList<BackendTs...>>
{
private:
    template <typename T>
    using container_type = std::forward_list<T>;

    template <typename T>
    using pointer_type = std::shared_ptr<T>;

    template <typename BackendT>
    using key_type = SyncInfo<DeviceForBackend<BackendT>()>;

    template <typename BackendT>
    using value_type = pointer_type<typename BackendT::comm_type>;

    template <typename BackendT>
    using pair_type = std::pair<key_type<BackendT>, value_type<BackendT>>;

    template <typename BackendT>
    using map_type = container_type<pair_type<BackendT>>;

public:
    /** @brief Map from SyncInfo objects to backends. */
    using type = std::tuple<map_type<BackendTs>...>;
};

/** @brief A convenience alias over @c SharedPtrCommTupleT. */
template <typename T>
using SharedPtrCommTuple = typename SharedPtrCommTupleT<T>::type;

/** @brief Shorthand for @c std::remove_reference<std::remove_cv<T>> */
template <typename T>
using PlainType = typename std::decay<T>::type;

template <typename... Ts>
void ResetAll_impl(std::tuple<Ts...>&, IndexSequence<>)
{}

template <typename... Ts, size_t I, size_t... Is>
void ResetAll_impl(std::tuple<Ts...>& tup, IndexSequence<I,Is...>)
{
    auto& container = std::get<I>(tup);

    // Delete all contents
    PlainType<decltype(container)>{}.swap(container);

    ResetAll_impl(tup, IndexSequence<Is...>{});
}

/** @brief Reset all @c shared_ptr objects in the tuple to nullptr.
 *
 *  Generates a compile-time loop to call @c reset() on each member of the tuple.
 *
 *  @param tup The tuple of pointers.
 *
 *  @todo Could be generalized to other pointer types. Currently no
 *        need and it's not without overhead.
 */
template <typename... Ts>
void ResetAll(std::tuple<Ts...>& tup)
{
    ResetAll_impl(tup, MakeIndexSequence<sizeof...(Ts)>{});
}

// Should go in SyncInfo stuff, but not 100% sure about this.
inline bool SyncInfoSame(SyncInfo<Device::CPU> const&,
                         SyncInfo<Device::CPU> const&) EL_NO_EXCEPT
{
    return true;
}

#ifdef HYDROGEN_HAVE_CUDA
inline bool SyncInfoSame(SyncInfo<Device::GPU> const& a,
                         SyncInfo<Device::GPU> const& b) EL_NO_EXCEPT
{
    return a.stream_ == b.stream_;
}
#endif

template <Device D1, Device D2, typename=DisableIf<SameDevice<D1,D2>>>
bool SyncInfoSame(SyncInfo<D1> const&, SyncInfo<D2> const&) EL_NO_EXCEPT
{
    return false;
}
}// namespace internal


/** @class AluminumComm
 *  @brief A communicator implementation wrapping Aluminum communicators.
 *
 *  This manages the the communicators used with Aluminum.
 */
class AluminumComm
    : public CommImpl<AluminumComm>
{
    using backend_list = AllAluminumBackends;
    using comm_tuple_type =
        internal::SharedPtrCommTuple<AllAluminumBackends>;

    /** @brief The collection of all communicators. */
    mutable comm_tuple_type al_comms_;

public:

    AluminumComm() = default;
    AluminumComm(MPI_Comm comm)
        : CommImpl<AluminumComm>{comm}
    {}

    /** @brief Reset all the internal state. */
    void DoReset() EL_NO_EXCEPT
    {
        internal::ResetAll(al_comms_);
    }

    template <typename BackendT>
    typename BackendT::comm_type& GetComm() const
    {
        static_assert(
            DeviceForBackend<BackendT>() == Device::CPU,
            "Requires CPU backend for this compatibility function.");
        return GetComm<BackendT>(SyncInfo<DeviceForBackend<BackendT>()>{});
    }

    template <typename BackendT, Device D>
    typename BackendT::comm_type&
    GetComm(SyncInfo<D> const& syncinfo) const
    {
        using comm_type = typename BackendT::comm_type;

        constexpr size_t idx
            = internal::IndexInTypeList<BackendT,backend_list>::value;

        auto& comm_map = std::get<idx>(al_comms_);

        using value_type =
            typename std::decay<decltype(comm_map)>::type::value_type;
        auto it = std::find_if(
            std::begin(comm_map), std::end(comm_map),
            [&syncinfo](value_type const& x)
            {
                return internal::SyncInfoSame(x.first, syncinfo);
            });

        // FIXME (trb): Exposes the forward_list detail
        if (it == std::end(comm_map))
        {
            comm_map.emplace_front(
                std::make_pair(syncinfo,
                               MakeWithSyncInfo<comm_type>(
                                   this->GetMPIComm(), syncinfo)));
            return *(comm_map.front().second);
        }

        return *it->second;
    }

private:
    template <typename CommT>
    std::shared_ptr<CommT> MakeWithSyncInfo(
        MPI_Comm comm, SyncInfo<Device::CPU> const&) const
    {
        return std::make_shared<CommT>(comm);
    }

#ifdef HYDROGEN_HAVE_CUDA
    template <typename CommT>
    std::shared_ptr<CommT> MakeWithSyncInfo(
        MPI_Comm comm, SyncInfo<Device::GPU> const& syncinfo) const
    {
        return std::make_shared<CommT>(comm, syncinfo.stream_);
    }
#endif // HYDROGEN_HAVE_CUDA
}; // class AluminumComm

}// namespace mpi
}// namespace El
#endif /* EL_IMPORTS_MPI_ALUMINUM_COMM_HPP_ */

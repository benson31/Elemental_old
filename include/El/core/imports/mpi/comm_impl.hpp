#pragma once
#ifndef EL_IMPORTS_MPI_COMM_IMPL_HPP_
#define EL_IMPORTS_MPI_COMM_IMPL_HPP_

#include <mpi.h>

namespace El
{
namespace mpi
{

template <typename SpecificCommImpl>
class CommImpl
{
public:
    CommImpl() = default;
    CommImpl(MPI_Comm mpi_comm)
        : comm_{mpi_comm}
    {}

    /** @brief Get the process rank in the communicator. */
    int Rank() const EL_NO_RELEASE_EXCEPT
    {
        if (comm_ == MPI_COMM_NULL)
            return MPI_UNDEFINED;

        int rank;
        MPI_Comm_rank(comm_, &rank);
        return rank;
    }

    /** @brief Get the number of processes in the communicator. */
    int Size() const EL_NO_RELEASE_EXCEPT
    {
        if (comm_ == MPI_COMM_NULL)
            return MPI_UNDEFINED;

        int size;
        MPI_Comm_size(comm_, &size);
        return size;
    }

    /** @brief Clear all the state of the communicator. */
    void Reset() EL_NO_EXCEPT
    {
        static_cast<SpecificCommImpl*>(this)->DoReset();
        comm_ = MPI_COMM_NULL;
    }

    void Free() EL_NO_RELEASE_EXCEPT
    {
        MPI_Comm_free(&comm_);
        Reset();
    }

    MPI_Comm GetMPIComm() const EL_NO_EXCEPT { return comm_; };

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
};// class CommImpl

// NOTE (trb): With how this class has evolved, I'm not really
// comfortable with this anymore. Nonetheless, I'll keep them around
// for back-compatiblity until I can dig further.
//
// Combined with Jack's comment below, I think this is actually a
// truly horrific abomination, but semantic perfection is a steep goal
// at the moment.
template <typename ImplT>
bool operator==(
    CommImpl<ImplT> const& a, CommImpl<ImplT> const& b) EL_NO_EXCEPT
{ return a.GetMPIComm() == b.GetMPIComm(); }

template <typename ImplT>
bool operator!=(
    CommImpl<ImplT> const& a, CommImpl<ImplT> const& b) EL_NO_EXCEPT
{ return !(a == b); }

// Hopefully, despite the fact that MPI_Comm is opaque, the following will
// reliably hold (otherwise it must be extended). Typically, MPI_Comm is
// either 'int' or 'void*'.
template <typename ImplT>
bool operator<(
    CommImpl<ImplT> const& a, CommImpl<ImplT> const& b ) EL_NO_EXCEPT
{ return a.GetMPIComm() < b.GetMPIComm(); }

}// namespace mpi
}// namespace El
#endif /* EL_IMPORTS_MPI_COMM_IMPL_HPP_ */

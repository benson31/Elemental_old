#include <El/config.h>
#include <El/core/Environment.hpp>
#include <El/core/imports/mpi.hpp>
#include <El/core/imports/qd.hpp>

namespace El
{

Environment::Environment()
    : Environment{0, nullptr}
{}

Environment::Environment(int argc, char** argv)
    : args_{argc, argv, mpi::COMM_WORLD, std::cerr}
{

    // NOTE (trb): CUDA must be initialized before MPI. CUDA-aware MPI
    // may make CUDA API calls and we must ensure the proper context
    // is setup first lest the default context be used.
#ifdef HYDROGEN_HAVE_CUDA
    InitializeCUDA(argc, argv);
    InitializeCUBLAS(); // FIXME (trb): Part of InitializeCUDA??
#endif // HYDROGEN_HAVE_CUDA

    // NOTE (trb): We may or may not be the ones who actually
    // "MPI_Init" MPI (e.g., if Aluminum does it or if a user does it.)
    if (!mpi::Initialized())
    {
        if (mpi::Finalized())
        {
            LogicError(
                "Environment::Environment(): "
                "Cannot initialize Hydrogen after finalizing MPI.");
        }

#ifdef EL_HAVE_OPENMP
        const Int provided =
            mpi::InitializeThread(
                argc, argv, mpi::THREAD_MULTIPLE);
        const int commRank = mpi::Rank(mpi::COMM_WORLD);
        if (provided != mpi::THREAD_MULTIPLE && commRank == 0)
        {
            std::cerr << "WARNING: Could not achieve THREAD_MULTIPLE support."
                      << std::endl;
        }
#else
        mpi::Initialize( argc, argv );
#endif
    }
    else
    {
#ifdef EL_HAVE_OPENMP
        const Int provided = mpi::QueryThread();
        if( provided != mpi::THREAD_MULTIPLE )
        {
            RuntimeError(
                "Environment::Environment(): "
                "MPI initialized with inadequate thread support for Hydrogen.");
        }
#endif
    }

    // Queue a default algorithmic blocksize
    EmptyBlocksizeStack();
    PushBlocksizeStack(128);
    // TODO (trb): What's the current use of this?? That is, what's
    // the value of having a "stack" vs. just a plain ol' setable
    // value?

    // FIXME (trb): I don't like that these exist.
    // Build the default grid
    Grid::InitializeDefault();
    Grid::InitializeTrivial();

#ifdef HYDROGEN_HAVE_QD
    InitializeQD();
#endif

    InitializeRandom();

    // Create the types and ops.
    // mpfr::SetPrecision within InitializeRandom created the BigFloat types
    mpi::CreateCustom(); // FIXME (trb): Why not part of mpi::Initialize??
}

Environment::~Environment()
{
    EL_DEBUG_CSE;

    if (mpi::Finalized())
    {
        std::cerr << "Environment::~Environment: "
                  << "Warning: MPI was finalized before Hydrogen."
                  << std::endl;
    }
    else
    {
        // Destroy the types and ops
        mpi::DestroyCustom();

        // FIXME (trb): I don't like that these exist.
        Grid::FinalizeDefault();
        Grid::FinalizeTrivial();

        mpi::Finalize();

        EmptyBlocksizeStack();

#ifdef HYDROGEN_HAVE_QD
        FinalizeQD();
#endif

        FinalizeRandom();
    }

#ifdef HYDROGEN_HAVE_CUDA
    FinalizeCUDA();
    // FIXME (trb): If InitializeCUBLAS is going to remain, we should
    // nominally add FinalizeCUBLAS, too, so that the resources are
    // logically freed.
#endif

#ifndef EL_RELEASE
    CloseLog()
#endif // !EL_RELEASE

#ifdef HYDROGEN_HAVE_MPC
    if (EL_RUNNING_ON_VALGRIND)
        mpfr_free_cache();// FIXME (trb): Should be hidden.
#endif
}

}// namespace El

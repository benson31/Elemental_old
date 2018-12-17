#include <exception>

#include <El/core/Exception.hpp>

namespace El
{

void break_on_me() noexcept {}

void ReportException(std::exception const& e, std::ostream& os )
{
    try
    {
        ArgException const& argExcept = dynamic_cast<const ArgException&>(e);
        if (std::string(argExcept.what()) != "")
            os << argExcept.what() << endl;
#ifndef EL_RELEASE
        DumpCallStack(os);
#endif // !EL_RELEASE
    }
    catch (UnrecoverableException const& recovExcept)
    {
        if (std::string(e.what()) != "")
        {
            os << "Process " << mpi::Rank()
               << " caught an unrecoverable exception with message:\n"
               << e.what() << endl;
        }
#ifndef EL_RELEASE
        DumpCallStack(os);
#endif // !EL_RELEASE
        mpi::Abort( mpi::COMM_WORLD, 1 );
    }
    catch (std::exception const& castExcept)
    {
        if (std::string(e.what()) != "")
        {
            os << "Process " << mpi::Rank() << " caught error message:\n"
               << e.what() << endl;
        }
#ifndef EL_RELEASE
        DumpCallStack(os);
#endif // !EL_RELEASE
    }
}

}// namespace El

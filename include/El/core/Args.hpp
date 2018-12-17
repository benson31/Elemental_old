#pragma once
#ifndef EL_CORE_ARGS_HPP_
#define EL_CORE_ARGS_HPP_

#include <iostream>

#include <El/core/imports/mpi.hpp>
#include <El/core/imports/mpi_choice.hpp>

namespace El
{

// For getting the MPI argument instance (for internal usage)
class Args : public choice::MpiArgs
{
public:
    Args(int argc, char** argv,
         mpi::Comm comm=mpi::COMM_WORLD,
         std::ostream& error=std::cerr)
        : choice::MpiArgs{argc, argv, comm, error}
    { }
    virtual ~Args() { }
protected:
    void HandleVersion(std::ostream& os=std::cout) const override;
    void HandleBuild(std::ostream& os=std::cout) const override;
};
Args& GetArgs();

}// namespace El
#endif /* EL_CORE_ARGS_HPP_ */

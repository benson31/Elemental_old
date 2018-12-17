#pragma once
#ifndef EL_CORE_ENVIRONMENT_HPP_
#define EL_CORE_ENVIRONMENT_HPP_

#include <El/core/Args.hpp>

namespace El
{

// For initializing/finalizing Elemental using RAII
class Environment
{
public:
    Environment();
    Environment(int argc, char** argv);
    ~Environment();
private:
    Args args_;
};

}// namespace El
#endif /* EL_CORE_ENVIRONMENT_HPP_ */

#pragma once
#ifndef EL_CORE_TYPEDEFS_HPP_
#define EL_CORE_TYPEDEFS_HPP_

#include <El/config.h>

namespace El
{

using byte = unsigned char;

// If these are changes, you must make sure that they have
// existing MPI datatypes. This is only sometimes true for 'long long'
#ifdef EL_USE_64BIT_INTS
using Int = long long int;
using Unsigned = long long unsigned int;
#else
using Int = int;
using Unsigned = unsigned;
#endif

}// namespace El
#endif /* EL_CORE_TYPEDEFS_HPP_ */

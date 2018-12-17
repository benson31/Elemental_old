#pragma once
#ifndef EL_MACROS_HPP_
#define EL_MACROS_HPP_

#include <El/config.h>

// NOTE (trb): Preprocessor macros have global scope. At time of
// writing, there is no namespace scope in this file. Do not add
// anything to it that should have namespace scope without adding such
// a scope, too.

#define EL_UNUSED(expr) (void)(expr)

#ifdef EL_RELEASE
# define EL_DEBUG_ONLY(cmd)
# define EL_RELEASE_ONLY(cmd) cmd;
#else
# define EL_DEBUG_ONLY(cmd) cmd;
# define EL_RELEASE_ONLY(cmd)
#endif

#define EL_NO_EXCEPT noexcept

#ifdef EL_RELEASE
# define EL_NO_RELEASE_EXCEPT EL_NO_EXCEPT
#else
# define EL_NO_RELEASE_EXCEPT
#endif

#define EL_CONCAT2(name1,name2) name1 ## name2
#define EL_CONCAT(name1,name2) EL_CONCAT2(name1,name2)

#endif /* EL_MACROS_HPP_ */

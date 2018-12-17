/*
   Copyright (c) 2009-2016, Jack Poulson
                      2013, Jeff Hammond
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>

#include <algorithm>
#include <set>

namespace El {

void PrintVersion( ostream& os )
{
    os << "Elemental version information:\n"
       << "  Git revision: " << EL_GIT_SHA1 << "\n"
       << "  Version:      " << EL_VERSION_MAJOR << "."
                             << EL_VERSION_MINOR << "\n"
       << "  Build type:   " << EL_CMAKE_BUILD_TYPE << "\n"
       << endl;
}

void PrintConfig( ostream& os )
{
    os <<
      "Elemental configuration:\n" <<
      "  Math libraries:               " << EL_MATH_LIBS << "\n"
#ifdef EL_HAVE_FLA_BSVD
      "  Have FLAME bidiagonal SVD:    YES\n"
#else
      "  Have FLAME bidiagonal SVD:    NO\n"
#endif
#ifdef EL_HYBRID
      "  Hybrid mode:                  YES\n"
#else
      "  Hybrid mode:                  NO\n"
#endif
#ifdef EL_HAVE_QT5
      "  Have Qt5:                     YES\n"
#else
      "  Have Qt5:                     NO\n"
#endif
#ifdef EL_AVOID_COMPLEX_MPI
      "  Avoiding complex MPI:         YES\n"
#else
      "  Avoiding complex MPI:         NO\n"
#endif
#ifdef EL_USE_BYTE_ALLGATHERS
      "  Use byte AllGathers:          YES\n"
#else
      "  Use byte AllGathers:          NO\n"
#endif
       << endl;
}

void PrintCCompilerInfo( ostream& os )
{
    os << "Elemental's C compiler info:\n"
       << "  EL_CMAKE_C_COMPILER:    " << EL_CMAKE_C_COMPILER << "\n"
       << "  EL_MPI_C_COMPILER:      " << EL_MPI_C_COMPILER << "\n"
       << "  EL_MPI_C_INCLUDE_PATH:  " << EL_MPI_C_INCLUDE_PATH << "\n"
       << "  EL_MPI_C_COMPILE_FLAGS: " << EL_MPI_C_COMPILE_FLAGS << "\n"
       << "  EL_MPI_C_LINK_FLAGS:    " << EL_MPI_C_LINK_FLAGS << "\n"
       << "  EL_MPI_C_LIBRARIES:     " << EL_MPI_C_LIBRARIES << "\n"
       << endl;
}

void PrintCxxCompilerInfo( ostream& os )
{
    os << "Elemental's C++ compiler info:\n"
       << "  EL_CMAKE_CXX_COMPILER:    " << EL_CMAKE_CXX_COMPILER << "\n"
       << "  EL_CXX_FLAGS:             " << EL_CXX_FLAGS << "\n"
       << "  EL_MPI_CXX_COMPILER:      " << EL_MPI_CXX_COMPILER << "\n"
       << "  EL_MPI_CXX_INCLUDE_PATH:  " << EL_MPI_CXX_INCLUDE_PATH << "\n"
       << "  EL_MPI_CXX_COMPILE_FLAGS: " << EL_MPI_CXX_COMPILE_FLAGS << "\n"
       << "  EL_MPI_CXX_LINK_FLAGS:    " << EL_MPI_CXX_LINK_FLAGS << "\n"
       << "  EL_MPI_CXX_LIBRARIES:     " << EL_MPI_CXX_LIBRARIES << "\n"
       << endl;
}

bool Using64BitInt()
{
#ifdef EL_USE_64BIT_INTS
    return true;
#else
    return false;
#endif
}

bool Using64BitBlasInt()
{
#ifdef EL_USE_64BIT_BLAS_INTS
    return true;
#else
    return false;
#endif
}


Args& GetArgs()
{
    if( args == 0 )
        throw std::runtime_error("No available instance of Args");
    return *::args;
}

void Args::HandleVersion( ostream& os ) const
{
    string version = "--version";
    char** arg = std::find( argv_, argv_+argc_, version );
    const bool foundVersion = ( arg != argv_+argc_ );
    if( foundVersion )
    {
        if( mpi::Rank() == 0 )
            PrintVersion();
        throw ArgException();
    }
}

void Args::HandleBuild( ostream& os ) const
{
    string build = "--build";
    char** arg = std::find( argv_, argv_+argc_, build );
    const bool foundBuild = ( arg != argv_+argc_ );
    if( foundBuild )
    {
        if( mpi::Rank() == 0 )
        {
            PrintVersion();
            PrintConfig();
            PrintCCompilerInfo();
            PrintCxxCompilerInfo();
        }
        throw ArgException();
    }
}

void ComplainIfDebug()
{
#ifndef EL_RELEASE
    if (mpi::Rank() == 0 )
    {
        Output("=======================================================");
        Output(" In debug mode! Do not expect competitive performance! ");
        Output("=======================================================");
    }
#endif // !EL_RELEASE
}

template<typename T>
bool IsSorted( const vector<T>& x )
{
    const Int vecLength = x.size();
    for( Int i=1; i<vecLength; ++i )
    {
        if( x[i] < x[i-1] )
            return false;
    }
    return true;
}

// While is_strictly_sorted exists in Boost, it does not exist in the STL (yet)
template<typename T>
bool IsStrictlySorted( const vector<T>& x )
{
    const Int vecLength = x.size();
    for( Int i=1; i<vecLength; ++i )
    {
        if( x[i] <= x[i-1] )
            return false;
    }
    return true;
}

void Union
( vector<Int>& both, const vector<Int>& first, const vector<Int>& second )
{
    both.resize( first.size()+second.size() );
    auto it = std::set_union
      ( first.cbegin(),  first.cend(),
        second.cbegin(), second.cend(),
        both.begin() );
    both.resize( Int(it-both.begin()) );
}

vector<Int>
Union( const vector<Int>& first, const vector<Int>& second )
{
    vector<Int> both;
    Union( both, first, second );
    return both;
}

void RelativeIndices
( vector<Int>& relInds, const vector<Int>& sub, const vector<Int>& full )
{
    const Int numSub = sub.size();
    relInds.resize( numSub );
    auto it = full.cbegin();
    for( Int i=0; i<numSub; ++i )
    {
        const Int index = sub[i];
        it = std::lower_bound( it, full.cend(), index );
        EL_DEBUG_ONLY(
          if( it == full.cend() )
              LogicError("Index was not found");
        )
        relInds[i] = Int(it-full.cbegin());
    }
}

vector<Int> RelativeIndices( const vector<Int>& sub, const vector<Int>& full )
{
    vector<Int> relInds;
    RelativeIndices( relInds, sub, full );
    return relInds;
}

Int Find( const vector<Int>& sortedInds, Int index )
{
    EL_DEBUG_CSE
    auto it = std::lower_bound( sortedInds.cbegin(), sortedInds.cend(), index );
    EL_DEBUG_ONLY(
      if( it == sortedInds.cend() )
          LogicError("All indices were smaller");
      if( *it != index )
          LogicError("Could not find index");
    )
    return it - sortedInds.cbegin();
}

#define EL_NO_COMPLEX_PROTO
#define PROTO(T) \
  template bool IsSorted( const vector<T>& x ); \
  template bool IsStrictlySorted( const vector<T>& x );
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include <El/macros/Instantiate.h>

} // namespace El

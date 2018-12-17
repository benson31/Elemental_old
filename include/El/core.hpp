/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#ifndef EL_CORE_HPP
#define EL_CORE_HPP

// This would ideally be included within core/imports/mpi.hpp, but it is
// well-known that this must often be included first.
#include <mpi.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <random>
#include <type_traits> // std::enable_if
#include <vector>

#include <El/Macros.hpp>
#include <El/core/Typedefs.hpp>

#include <El/core/Meta.hpp>

#ifdef HYDROGEN_HAVE_QUADMATH
#include <quadmath.h>
#endif

namespace El {

#ifdef HYDROGEN_HAVE_QUADMATH
typedef __float128 Quad;
#endif

// Forward declarations
// --------------------
#ifdef HYDROGEN_HAVE_QD
struct DoubleDouble;
struct QuadDouble;
#endif
#ifdef HYDROGEN_HAVE_MPC
class BigInt;
class BigFloat;
#endif
template<typename Real>
class Complex;

#ifdef HYDROGEN_HAVE_MPC
template<>
struct IsIntegral<BigInt> { static const bool value = true; };
#endif

// For querying whether an element's type is a scalar
// --------------------------------------------------
#ifdef HYDROGEN_HAVE_QD
template<> struct IsScalar<DoubleDouble> : std::true_type {};
template<> struct IsScalar<QuadDouble> : std::true_type {};
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
template<> struct IsScalar<Quad> : std::true_type {};
#endif
#ifdef HYDROGEN_HAVE_MPC
template<> struct IsScalar<BigInt> : std::true_type {};
template<> struct IsScalar<BigFloat> : std::true_type {};
#endif
template<typename T> struct IsScalar<Complex<T>> : IsScalar<T> {};

// For querying whether an element's type is a field
// -------------------------------------------------
#ifdef HYDROGEN_HAVE_QD
template<> struct IsField<DoubleDouble> : std::true_type {};
template<> struct IsField<QuadDouble> : std::true_type {};
#endif
#ifdef HYDROGEN_HAVE_QUADMATH
template<> struct IsField<Quad> : std::true_type {};
#endif
#ifdef HYDROGEN_HAVE_MPC
template<> struct IsField<BigFloat> : std::true_type {};
#endif
template<typename T> struct IsField<Complex<T>> : IsField<T> {};

// For querying whether an element's type is supported by the STL's math
// ---------------------------------------------------------------------
#ifdef HYDROGEN_HAVE_QUADMATH
template<> struct IsStdScalar<Quad> : std::true_type {};
#endif
template<typename T> struct IsStdScalar<Complex<T>> : IsStdScalar<T> {};

// For querying whether an element's type is a field supported by STL
// ------------------------------------------------------------------
#ifdef HYDROGEN_HAVE_QUADMATH
template<> struct IsStdField<Quad> : std::true_type {};
#endif
template<typename T> struct IsStdField<Complex<T>> : IsStdField<T> {};

} // namespace El

// Declare the intertwined core parts of our library
#include <El/core/Environment.hpp>
#include <El/core/Exception.hpp>

#include <El/core/imports/valgrind.hpp>
#include <El/core/imports/omp.hpp>
#include <El/core/imports/qd.hpp>
#include <El/core/imports/mpfr.hpp>
#include <El/core/imports/qt5.hpp>

#include <El/core/Element/decl.hpp>
#include <El/core/Serialize.hpp>

#include <El/core/imports/blas.hpp>
#ifdef HYDROGEN_HAVE_CUDA
#include <El/core/imports/cuda.hpp>
#include <El/core/imports/cublas.hpp>
#endif // HYDROGEN_HAVE_CUDA
#ifdef HYDROGEN_HAVE_CUB
#include <El/core/imports/cub.hpp>
#endif // HYDROGEN_HAVE_CUB

#include <El/core/Device.hpp>
#include <El/core/SyncInfo.hpp>

#include <El/core/imports/mpi.hpp>
#include <El/core/imports/choice.hpp>
#include <El/core/imports/mpi_choice.hpp>
#include <El/core/environment/decl.hpp>

#include <El/core/Timer.hpp>
#include <El/core/indexing/decl.hpp>
#include <El/core/imports/lapack.hpp>
#include <El/core/imports/flame.hpp>
#include <El/core/imports/mkl.hpp>
#include <El/core/imports/openblas.hpp>
#include <El/core/imports/scalapack.hpp>

#include <El/core/limits.hpp>

namespace El
{

template <typename T=double> class AbstractMatrix;
template<typename T=double, Device D=Device::CPU> class Matrix;

template<typename T=double> class AbstractDistMatrix;

template<typename T=double> class ElementalMatrix;
template<typename T=double> class BlockMatrix;

template<typename T=double, Dist U=MC, Dist V=MR,
         DistWrap wrap=ELEMENT, Device=Device::CPU>
class DistMatrix;

} // namespace El


#include <El/core/Memory.hpp>
#include <El/core/SimpleBuffer.hpp>
#include <El/core/AbstractMatrix.hpp>
#include <El/core/Matrix/decl.hpp>
#include <El/core/DistMap/decl.hpp>
#include <El/core/View/decl.hpp>
#include <El/blas_like/level1/decl.hpp>

#include <El/core/Matrix/impl.hpp>
#include <El/core/Grid.hpp>
#include <El/core/DistMatrix.hpp>
#include <El/core/Proxy.hpp>
#include <El/core/ProxyDevice.hpp>

// Implement the intertwined parts of the library
#include <El/core/Element/impl.hpp>
#include <El/core/environment/impl.hpp>
#include <El/core/indexing/impl.hpp>

// Declare and implement the decoupled parts of the core of the library
// (perhaps these should be moved into their own directory?)
#include <El/core/View/impl.hpp>
#include <El/core/FlamePart.hpp>
#include <El/core/random/decl.hpp>
#include <El/core/random/impl.hpp>

// TODO: Sequential map
//#include <El/core/Map.hpp>

#include <El/core/DistMap.hpp>

#include <El/core/Permutation.hpp>
#include <El/core/DistPermutation.hpp>

#endif // ifndef EL_CORE_HPP

#ifndef HYDROGEN_UTILS_HALFPRECISION_HPP_
#define HYDROGEN_UTILS_HALFPRECISION_HPP_

/** @file
 *
 *  Here we handle a few high-level metaprogramming tasks for 16-bit
 *  floating point support. This currently just includes IEEE-754
 *  support but will likely soon expand to include "bfloat16" as well.
 *
 *  When compiling using v2.1.0 of the Half library on OSX (10.14 with
 *  Apple LLVM version 10.0.1 (clang-1001.0.46.4)), there were many
 *  interesting issues that showed up. Hence the strange layout of
 *  this file.
 */

#ifdef HYDROGEN_HAVE_HALF

#include <El/hydrogen_config.h>
#include <hydrogen/meta/TypeTraits.hpp>

// Forward-declare things so I can start specializing templates.
namespace half_float
{
class half;
}// namespace half_float

// Specialize some STL-y things
#include <type_traits>
namespace std
{
template <>
struct is_floating_point<half_float::half> : true_type {};

template <>
struct is_integral<half_float::half> : false_type {};

template <>
struct is_arithmetic<half_float::half> : true_type {};
}// namespace std

// Now include the actual Half library.
#include <half.hpp>

// Declare the hydrogen typedef
namespace hydrogen
{
using namespace half_float::literal;

using cpu_half_type = half_float::half;

template <>
struct TypeTraits<cpu_half_type>
{
    static cpu_half_type One() noexcept { return 1._h; }
    static cpu_half_type Zero() noexcept { return 0._h; }
    static std::string Name() { return typeid(cpu_half_type).name(); }
};// struct TypeTraits<cpu_half_type>

}// namespace hydrogen

#endif // HYDROGEN_HAVE_HALF

// Finally, do the GPU stuff
#ifdef HYDROGEN_GPU_USE_FP16

// Grab the right header
#if defined(HYDROGEN_HAVE_CUDA)
#include <cuda_fp16.h>
#elif defined(HYDROGEN_HAVE_AMDGPU)
#include <rocblas-types.h>
#endif // HYDROGEN_HAVE_CUDA

namespace hydrogen
{

#if defined(HYDROGEN_HAVE_CUDA)
/** @brief Unified name for the FP16 type on GPU */
using gpu_half_type = __half;

template <>
struct TypeTraits<gpu_half_type>
{
    static gpu_half_type One() noexcept { return 1.f; }
    static gpu_half_type Zero() noexcept { return 0.f; }
    static std::string Name() { return typeid(gpu_half_type).name(); }
};// struct TypeTraits<gpu_half_type>

#elif defined(HYDROGEN_HAVE_AMDGPU)
/** @brief Unified name for the FP16 type on GPU */
using gpu_half_type = rocblas_half;
#endif // HYDROGEN_HAVE_CUDA

}// namespace hydrogen

#if defined(HYDROGEN_HAVE_CUDA) && !defined(__CUDACC__)

/** @brief Enable "update" functionality for __half. */
inline hydrogen::gpu_half_type& operator+=(
    hydrogen::gpu_half_type& val, hydrogen::gpu_half_type const& rhs)
{
    val = float(val) + float(rhs);
    return val;
}

/** @brief Enable subtract-equal functionality for __half. */
inline hydrogen::gpu_half_type& operator-=(
    hydrogen::gpu_half_type& val, hydrogen::gpu_half_type const& rhs)
{
    val = float(val) - float(rhs);
    return val;
}

/** @brief Enable "scale" functionality for __half. */
inline hydrogen::gpu_half_type& operator*=(
    hydrogen::gpu_half_type& val, hydrogen::gpu_half_type const& rhs)
{
    val = float(val) * float(rhs);
    return val;
}

/** @brief Enable divide-equal functionality for __half. */
inline hydrogen::gpu_half_type& operator/=(
    hydrogen::gpu_half_type& val, hydrogen::gpu_half_type const& rhs)
{
    val = float(val) / float(rhs);
    return val;
}
#endif // defined(HYDROGEN_HAVE_CUDA) && !defined(__CUDACC__)
#endif // HYDROGEN_GPU_USE_FP16
#endif // HYDROGEN_UTILS_HALFPRECISION_HPP_

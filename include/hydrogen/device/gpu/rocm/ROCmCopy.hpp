#ifndef HYDROGEN_DEVICE_GPU_ROCM_ROCMCOPY_HPP_
#define HYDROGEN_DEVICE_GPU_ROCM_ROCMCOPY_HPP_

#include "ROCmError.hpp"

#include <hydrogen/Device.hpp>
#include <hydrogen/SyncInfo.hpp>

#include <hydrogen/device/GPU.hpp>
#include <hydrogen/blas/gpu/Fill.hpp>

#include <hip/hip_runtime.h>

#include <stdio.h>

#define TOM_PTR_OK(src, dst, kind)                                      \
    do                                                                  \
    {                                                                   \
    if (!ok(src, dst, kind))                                            \
    {                                                                   \
        printf("%s:%d ERROR Bad ptrs in %s\n\n  src: %p (%s)\n\n  dst: %p (%s)\n\n", \
               __FILE__, __LINE__, #kind, src, device_string(src), dst, device_string(dst)); \
        throw std::string("BAD POINTERS");                              \
    }                                                                   \
        } while (0)

namespace hydrogen
{
namespace gpu
{

/** @todo Flesh out documentation
 *  @todo these are actually only valid for "packed" types
 */

// These functions are synchronous with respect to their SyncInfo
// objects (that is, they require explicit synchronization to the
// host).

template <typename T>
void Fill1DBuffer(T* buffer, size_t num_elements, T value,
                  SyncInfo<Device::GPU> const& si)
{
    if (num_elements == 0UL)
        return;

    Fill_GPU_1D_impl(buffer, num_elements, value, si);
}

inline hipMemoryType memtype(void const* ptr)
{
    hipPointerAttribute_t attr;
    auto const status = hipPointerGetAttributes(&attr, ptr);
    if (status == hipErrorInvalidValue)
        return hipMemoryTypeHost;
    H_CHECK_HIP(status);
    return attr.memoryType;
}

inline bool is_host_ptr(void const* ptr)
{
    return memtype(ptr) == hipMemoryTypeHost;
}

inline bool is_device_ptr(void const* ptr)
{
    return memtype(ptr) == hipMemoryTypeDevice;
}

inline bool ok(void const* src, void const* tgt, hipMemcpyKind kind)
{
    switch (kind)
    {
    case hipMemcpyHostToDevice:
        return is_host_ptr(src) && is_device_ptr(tgt);
    case hipMemcpyHostToHost:
        return is_host_ptr(src) && is_host_ptr(tgt);
    case hipMemcpyDeviceToDevice:
        return is_device_ptr(src) && is_device_ptr(tgt);
    case hipMemcpyDeviceToHost:
        return is_device_ptr(src) && is_host_ptr(tgt);
    case hipMemcpyDefault:
        return true;
    default:
        return false;
    }
}

inline char const* device_string(void const* ptr)
{
    if (is_host_ptr(ptr)) return "host_ptr";
    if (is_device_ptr(ptr)) return "device_ptr";
    return "unknown_ptr";
}

template <typename T>
void Copy1DIntraDevice(T const* H_RESTRICT src, T* H_RESTRICT dest,
                       size_t num_elements,
                       SyncInfo<Device::GPU> const& si)
{
    if (num_elements == 0UL)
        return;

    TOM_PTR_OK(src, dest, hipMemcpyDeviceToDevice);
    H_CHECK_HIP(
        hipMemcpyAsync(
            dest, src, num_elements*sizeof(T),
            hipMemcpyDefault, si.Stream()));
}

template <typename T>
void Copy1DToHost(T const* H_RESTRICT src, T* H_RESTRICT dest,
                  size_t num_elements,
                  SyncInfo<Device::GPU> const& src_si)
{
    if (num_elements == 0UL)
        return;

    TOM_PTR_OK(src, dest, hipMemcpyDeviceToHost);
    H_CHECK_HIP(
        hipMemcpyAsync(
            dest, src, num_elements*sizeof(T),
            hipMemcpyDefault, src_si.Stream()));
}

template <typename T>
void Copy1DToDevice(T const* H_RESTRICT src, T* H_RESTRICT dest,
                    size_t num_elements,
                    SyncInfo<Device::GPU> const& dest_si)
{
    if (num_elements == 0UL)
        return;

    TOM_PTR_OK(src, dest, hipMemcpyHostToDevice);
    H_CHECK_HIP(
        hipMemcpyAsync(
            dest, src, num_elements*sizeof(T),
            hipMemcpyDefault, dest_si.Stream()));
}

template <typename T>
void Copy2DIntraDevice(T const* src, size_t src_ldim,
                       T* dest, size_t dest_ldim,
                       size_t height, size_t width,
                       SyncInfo<Device::GPU> const& si)
{
    if (height == 0UL || width == 0UL)
        return;

    TOM_PTR_OK(src, dest, hipMemcpyDeviceToDevice);
    H_CHECK_HIP(
        hipMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            hipMemcpyDefault, si.Stream()));
}

template <typename T>
void Copy2DToHost(T const* src, size_t src_ldim,
                  T* dest, size_t dest_ldim,
                  size_t height, size_t width,
                  SyncInfo<Device::GPU> const& src_si)
{
    if (height == 0UL || width == 0UL)
        return;

    TOM_PTR_OK(src, dest, hipMemcpyDeviceToHost);
    H_CHECK_HIP(
        hipMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            hipMemcpyDefault, src_si.Stream()));
}

template <typename T>
void Copy2DToDevice(T const* src, size_t src_ldim,
                    T* dest, size_t dest_ldim,
                    size_t height, size_t width,
                    SyncInfo<Device::GPU> const& dest_si)
{
    if (height == 0UL || width == 0UL)
        return;

    TOM_PTR_OK(src, dest, hipMemcpyHostToDevice);
    H_CHECK_HIP(
        hipMemcpy2DAsync(
            dest, dest_ldim*sizeof(T),
            src, src_ldim*sizeof(T),
            height*sizeof(T), width,
            hipMemcpyDefault, dest_si.Stream()));
}

}// namespace gpu
}// namespace hydrogen
#endif // HYDROGEN_DEVICE_GPU_ROCM_ROCMCOPY_HPP_

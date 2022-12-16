#include <memory>
#include "El-lite.hpp"
#include "El/core/MemoryPool.hpp"

namespace El
{
bool details::debug_mempool() noexcept
{
    char const* const env = std::getenv("H_MEMPOOL_DEBUG");
    return (env && std::strlen(env) && env[0] != '0');
}

namespace
{
#ifdef HYDROGEN_HAVE_GPU
std::unique_ptr<MemoryPool<true>> pinnedHostMemoryPool_;
#endif  // HYDROGEN_HAVE_GPU
std::unique_ptr<MemoryPool<false>> hostMemoryPool_;
}  // namespace <anon>

#ifdef HYDROGEN_HAVE_GPU

MemoryPool<true>& PinnedHostMemoryPool()
{
    if (!pinnedHostMemoryPool_)
        pinnedHostMemoryPool_.reset(new MemoryPool<true>());
    return *pinnedHostMemoryPool_;
}

void DestroyPinnedHostMemoryPool()
{ pinnedHostMemoryPool_.reset(); }

#endif  // HYDROGEN_HAVE_GPU

MemoryPool<false>& HostMemoryPool()
{
    if (!hostMemoryPool_)
        hostMemoryPool_.reset(new MemoryPool<false>());
    return *hostMemoryPool_;
}

void DestroyHostMemoryPool()
{ hostMemoryPool_.reset(); }

}  // namespace El

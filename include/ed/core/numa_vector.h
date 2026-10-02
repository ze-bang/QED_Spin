#pragma once
// =============================================================================
// include/ed/core/numa_vector.h -- a std::vector whose resize() does not touch the memory.
//
// Linux places a page on the NUMA node of the thread that first writes it, so a serial zero
// fill would put a whole large array (a reduced CSR, a rank table: tens to hundreds of GB) on
// ONE node, and every later parallel pass would stream it across the interconnect (measured on
// Fir: ~80 GB/s from one domain vs ~8x that interleaved). A NumaVector is resized without
// initialisation and then first-touched in parallel by its builder.
// =============================================================================

#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace ed::core {

/// An allocator whose value-less construct() default-initialises: resize() leaves trivial
/// elements untouched instead of zero-filling them on the calling thread.
template <class T, class A = std::allocator<T>>
struct DefaultInitAllocator : A {
    using A::A;
    template <class U>
    struct rebind { using other = DefaultInitAllocator<U, typename std::allocator_traits<A>::template rebind_alloc<U>>; };
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) { ::new (static_cast<void*>(p)) U; }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        std::allocator_traits<A>::construct(static_cast<A&>(*this), p, std::forward<Args>(args)...);
    }
};

template <class T>
using NumaVector = std::vector<T, DefaultInitAllocator<T>>;

}  // namespace ed::core

#pragma once
// =============================================================================
// include/ed/core/numa_vector.h -- allocators for large arrays: NumaVector (resize() does not touch
// the memory) and ReleasingAllocator (a large block is unmapped on release).
//
// Linux places a page on the NUMA node of the thread that first writes it, so a serial zero
// fill would put a whole large array (a reduced CSR, a rank table: tens to hundreds of GB) on
// ONE node, and every later parallel pass would stream it across the interconnect (measured on
// Fir: ~80 GB/s from one domain vs ~8x that interleaved). A NumaVector is resized without
// initialisation and then first-touched in parallel by its builder.
// =============================================================================

#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__unix__)
#include <sys/mman.h>
#endif

namespace ed::core {

/// An allocator whose value-less construct() default-initialises: resize() leaves trivial
/// elements untouched instead of zero-filling them on the calling thread.
template <class T, class A = std::allocator<T>> struct DefaultInitAllocator : A {
    using A::A;
    template <class U> struct rebind {
        using other = DefaultInitAllocator<U, typename std::allocator_traits<A>::template rebind_alloc<U>>;
    };
    template <class U> void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <class U, class... Args> void construct(U* p, Args&&... args) {
        std::allocator_traits<A>::construct(static_cast<A&>(*this), p, std::forward<Args>(args)...);
    }
};

template <class T> using NumaVector = std::vector<T, DefaultInitAllocator<T>>;

/// An allocator that maps blocks of at least 1 MiB straight from the kernel and unmaps them on
/// release, so freed memory leaves the RSS at once: malloc keeps a released chunk of a few MB
/// resident (its mmap threshold grows with use), which held a reduced CSR's build slabs beside the
/// finished arrays (tri36 Gamma A1: 14.5 GiB at the copy for a 7.2 GB CSR). Smaller blocks come
/// from operator new.
template <class T> struct ReleasingAllocator {
    using value_type = T;
    static constexpr std::size_t kMapBytes = std::size_t{1} << 20;
    ReleasingAllocator() noexcept = default;
    template <class U> ReleasingAllocator(const ReleasingAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        const std::size_t bytes = n * sizeof(T);
#if defined(__unix__)
        if (bytes >= kMapBytes) {
            void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) throw std::bad_alloc();
            return static_cast<T*>(p);
        }
#endif
        return static_cast<T*>(::operator new(bytes));
    }
    void deallocate(T* p, std::size_t n) noexcept {
        const std::size_t bytes = n * sizeof(T);
#if defined(__unix__)
        if (bytes >= kMapBytes) {
            ::munmap(p, bytes);
            return;
        }
#endif
        ::operator delete(p);
    }
    template <class U> bool operator==(const ReleasingAllocator<U>&) const noexcept { return true; }
    template <class U> bool operator!=(const ReleasingAllocator<U>&) const noexcept { return false; }
};

}  // namespace ed::core

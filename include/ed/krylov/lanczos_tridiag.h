#pragma once
// =============================================================================
// include/ed/krylov/lanczos_tridiag.h
//
// Direct kernel-facade helper for the canonical CPU Lanczos build that
// feeds FTLM / LTLM / Hybrid / KPM_DOS / TPQ_DYNAMICAL.
//
// Phase 5 of the Krylov-unification gap-fill (May 2026 day 12+):
//
// Today the four CPU thermal kernels (FTLM, LTLM, Hybrid, KPM_DOS) build
// the per-sample Krylov subspace through
// `build_lanczos_tridiagonal_with_basis(...)` in `src/solvers/cpu/lanczos.cpp`,
// which since the Phase-A roll-up already delegates to
// `ed::krylov::lanczos_kernel<CpuBackend>` on the fast path
// (`full_reorth == true && basis_vectors != nullptr`). The kernel call,
// however, sits behind a `std::function<void(const Complex*, Complex*, int)>`
// matvec adapter and a basis-translation loop (`UniqueVec` -> `ComplexVector`
// copy). The two add ~5 ns / matvec and one O(M * N) heap-resident copy per
// run respectively — small in absolute terms but pure overhead.
//
// `ed::krylov::lanczos_tridiag` collapses both indirections:
//
//   * MatvecFn is taken by template forwarding reference, so the call
//     compiles down to a direct virtual `op.apply(...)` (a lambda over
//     `Operator::apply`) with NO `std::function` indirection.
//   * The kernel's `vector<UniqueVec>` basis is returned in-place inside
//     `LanczosKernelResult`; callers that want a `vector<ComplexVector>`
//     translation can do it once at the call site (the historical FTLM
//     pattern), but the helper itself avoids the copy.
//
// The CPU backend is fetched via `ed::matvec::default_cpu_backend()` (a
// process-wide thread-safe singleton); no allocation, no destruction.
//
// Public API:
//
//   namespace ed::krylov {
//     template <typename MatvecFn>
//     LanczosKernelResult lanczos_tridiag(MatvecFn&& matvec,
//                                          std::size_t N,
//                                          const Complex* v0,
//                                          const LanczosKernelOptions& opts);
//
//     // MatVecOperator convenience overload (no std::function adapter).
//     LanczosKernelResult lanczos_tridiag(const ed::matvec::MatVecOperator& op,
//                                          std::size_t N,
//                                          const Complex* v0,
//                                          const LanczosKernelOptions& opts);
//   }
//
// Both share the same `LanczosKernelResult` ABI: alpha, beta, basis,
// iters_done. The matvec signature for the templated path is
// `void(const Complex*, Complex*, std::size_t)` -- matching the kernel
// directly. For the `MatVecOperator` overload the helper synthesises the
// lambda internally; no `std::function` heap allocation occurs.
//
// This header is host-only (no CUDA-only types).
// =============================================================================

#include <complex>
#include <cstddef>
#include <utility>

#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/matvec/matvec.h>

namespace ed::krylov {

/// Templated direct call into `lanczos_kernel<CpuBackend>` (the kernel
/// is backend-agnostic; this helper hard-codes the CPU backend for
/// callers in `cpu/*.cpp`). No `std::function` indirection: the matvec
/// callable is forwarded as a universal reference and dispatched at
/// compile time.
template <typename MatvecFn>
LanczosKernelResult lanczos_tridiag(MatvecFn&& matvec,
                                     std::size_t N,
                                     const Complex* v0,
                                     const LanczosKernelOptions& opts)
{
    return lanczos_kernel(ed::matvec::default_cpu_backend(),
                          std::forward<MatvecFn>(matvec),
                          N, v0, opts);
}

/// Convenience overload for the MatVecOperator hierarchy. Forwards
/// directly to the templated helper above with a lambda that closes
/// over `op` and invokes `op.apply(...)`. No std::function adapter is
/// involved; the lambda is captured by reference at the call site so
/// the per-matvec cost reduces to one virtual call (the existing cost
/// of `MatVecOperator::apply`).
inline LanczosKernelResult lanczos_tridiag(
    const ed::matvec::MatVecOperator& op,
    std::size_t N,
    const Complex* v0,
    const LanczosKernelOptions& opts)
{
    auto matvec = [&op](const Complex* in, Complex* out, std::size_t n) {
        op.apply(in, out, n);
    };
    return lanczos_tridiag(matvec, N, v0, opts);
}

}  // namespace ed::krylov

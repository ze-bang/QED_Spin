// =============================================================================
// src/core/operator_gpu.cpp
//
// WEAK ed_core fallbacks for the non-virtual GPU mirror hooks declared on
// ``Operator`` / ``FixedSzOperator``. ed_core is always on the link line, so
// these guarantee the inline ``bind_cuda()`` / ``geometry()`` delegators
// resolve in EVERY binary -- including the CPU-only ones (benchmarks,
// ed_distributed_main, the bfg drivers) that never link ed_solvers_gpu.
//
// The fallbacks report "no device mirror": ``cuda_mirror_available_`` returns
// false (so ``geometry().supports_device_matvec`` stays false and
// ``select_backend`` never picks the CudaBackend lane) and the *_impl_ hooks
// degrade to ``bind_cpu()`` defensively.
//
// On a WITH_CUDA build the STRONG definitions in ``src/core/operator_gpu.cu``
// (ed_solvers_gpu, which precedes ed_core in the link order) override these
// for any binary that links the GPU archive. Keeping ``bind_cuda()`` inline in
// the headers means Operator's vtable has no out-of-line key function, so it
// stays weak/COMDAT and the split here never turns into an undefined-vtable
// link error.
//
// Under !WITH_CUDA the hooks are never referenced (the headers' bind_cuda()
// returns bind_cpu() directly and geometry() never sets the device flag), so
// this TU is empty.
//
// Phase 2a of the "Lean operator architecture collapse" plan (Jun 2026).
// =============================================================================

#ifdef WITH_CUDA

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator.h>

// GCC/Clang honour __attribute__((weak)) on member-function definitions: the
// strong same-mangled-name definitions in ed_solvers_gpu win whenever that
// archive is linked.
__attribute__((weak)) bool Operator::cuda_mirror_available_() noexcept {
    return false;
}

__attribute__((weak)) ed::LinearOperator::MatvecFn
Operator::bind_cuda_full_impl_() const {
    return bind_cpu();
}

// Operator-collapse Phase 4: weak fallback for the FixedSz alias
// (SubspaceOperator<FixedSzBasisPolicy, Host>). Replaces the legacy
// FixedSzOperator::bind_cuda_fixed_sz_impl_ weak fallback.
template <>
__attribute__((weak)) ed::LinearOperator::MatvecFn
ed::SubspaceOperator<ed::matvec::basis::FixedSzBasisPolicy,
                     ed::matvec::MemorySpace::Host>::bind_cuda_impl_() const {
    return bind_cpu();
}

#endif  // WITH_CUDA

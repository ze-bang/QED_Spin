// =============================================================================
// src/core/operator_gpu.cu
//
// STRONG (ed_solvers_gpu) definitions of the non-virtual GPU mirror hooks
// ``Operator::cuda_mirror_available_`` / ``Operator::bind_cuda_full_impl_`` /
// ``FixedSzOperator::bind_cuda_fixed_sz_impl_``. The matching WEAK fallbacks
// (mirror unavailable -> bind_cpu()) live in ``operator_gpu.cpp`` (ed_core,
// always linked). ``bind_cuda()`` / ``geometry()`` themselves stay inline in
// the headers so Operator's vtable has no key function pinned to this archive
// and every CPU-only binary still links. Because ed_solvers_gpu precedes
// ed_core in the link order, GPU binaries resolve these strong symbols and the
// weak ed_core fallbacks are never pulled.
//
// This is the operator-collapse GPU-parity lane for the plain full-Hilbert /
// fixed-Sz operators: it routes their production GPU matvec through the SOTA
// no-atomic-gather ``CudaMatVecBackend`` instead of the bespoke atomic-scatter
// ``GPUFixedSzOperator::matVecGPU``. Mirrors the
// ``ed::symmetry::SectorOperator::bind_cuda()`` split (sector_operator_gpu.cu)
// so a host-resident Operator is backend-complete (CPU + GPU) with zero caller
// changes -- ``select_backend`` picks ``CudaBackend`` off the
// ``geometry().supports_device_matvec`` flag and ``bind_cuda()`` hands it a
// device-pointer matvec.
//
// Contract: the orchestrator's CUDA lane keeps every Krylov vector resident on
// the device, so the returned ``MatvecFn`` consumes DEVICE pointers. We upload
// the term SoA once (``upload_terms``) and the bound lambda runs the kernel on
// the device buffers via ``apply_complex_device`` (no per-apply H2D/D2H).
//
// Phase 2a of the "Lean operator architecture collapse" plan (Jun 2026).
// =============================================================================

#ifdef WITH_CUDA

#include <cstdlib>

#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/matvec/cuda_matvec_backend.cuh>

// STRONG definitions of the non-virtual GPU mirror hooks. They override the
// weak ed_core fallbacks (operator_gpu.cpp) for any binary that links
// ed_solvers_gpu, which precedes ed_core in the link order. See the note on
// Operator::bind_cuda_full_impl_ in operator.h.

bool Operator::cuda_mirror_available_() noexcept {
    // GPU code IS linked here (this TU lives in ed_solvers_gpu), so the only
    // gate is the env override used for CPU/GPU bisection. Mirrors the
    // ed::symmetry::SectorOperator gate (ED_GPU_OPERATOR_MIRROR=0 -> off).
    static const bool kEnabled = [] {
        const char* e = std::getenv("ED_GPU_OPERATOR_MIRROR");
        if (e == nullptr || e[0] == '\0') return true;   // default ON
        if (e[0] == '0' && e[1] == '\0')  return false;  // "0" -> OFF
        return true;                                     // else -> ON
    }();
    return kEnabled;
}

ed::LinearOperator::MatvecFn Operator::bind_cuda_full_impl_() const {
    if (!cuda_backend_) {
        cuda_backend_ = ed::matvec::make_cuda_full_backend<
            DiagonalOneBody, OffDiagonalOneBody,
            DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
            ThreeBodyTransformData>(n_bits_, static_cast<double>(spin_l_));
    }
    // Snapshot the (possibly mutated) host term SoA to the device. Force a
    // re-upload so a re-bind after term edits picks up the new couplings.
    const auto tv = term_view_();  // commits pending transforms + builds view
    cuda_backend_->invalidate_caches();
    cuda_backend_->upload_terms(&tv);

    auto backend = cuda_backend_;  // shared_ptr copy keeps the mirror alive
    return [backend](const Complex* in, Complex* out, std::size_t n) {
        backend->apply_complex_device(in, out, n);
    };
}

// fp32 device matvec (memory-halving mTPQ lane). Reuses the same lazily-built
// CudaMatVecBackend + uploaded (cuDoubleComplex) term SoA as bind_cuda_full;
// only the in/out vectors are cuFloatComplex (passed as void*).
ed::LinearOperator::Fp32DeviceMatvecFn Operator::bind_cuda_f32_impl_() const {
    if (!cuda_backend_) {
        cuda_backend_ = ed::matvec::make_cuda_full_backend<
            DiagonalOneBody, OffDiagonalOneBody,
            DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
            ThreeBodyTransformData>(n_bits_, static_cast<double>(spin_l_));
    }
    const auto tv = term_view_();
    cuda_backend_->invalidate_caches();
    cuda_backend_->upload_terms(&tv);

    auto backend = cuda_backend_;
    return [backend](const void* in, void* out, std::size_t n) {
        backend->apply_complex_device_f32(in, out, n);
    };
}

// Operator-collapse Phase 4: FixedSzOperator is now the alias
// SubspaceOperator<FixedSzBasisPolicy, Host>; this strong definition of its
// bind_cuda_impl_ member specialization replaces the legacy
// FixedSzOperator::bind_cuda_fixed_sz_impl_. The sector basis states come from
// the owned FixedSzSubspace producer.
template <>
ed::LinearOperator::MatvecFn
ed::SubspaceOperator<ed::matvec::basis::FixedSzBasisPolicy,
                     ed::matvec::MemorySpace::Host>::bind_cuda_impl_() const {
    if (!cuda_backend_) {
        cuda_backend_ = ed::matvec::make_cuda_fixed_sz_backend<
            DiagonalOneBody, OffDiagonalOneBody,
            DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
            ThreeBodyTransformData>(producer_.basis_states(),
                                    static_cast<double>(spin_l_));
    }
    const auto tv = term_view_();
    cuda_backend_->invalidate_caches();
    cuda_backend_->upload_terms(&tv);

    auto backend = cuda_backend_;
    return [backend](const Complex* in, Complex* out, std::size_t n) {
        backend->apply_complex_device(in, out, n);
    };
}

// Strong definition of ed::thermal::mtpq_f32 (co-located here so it rides the
// same reliable archive-pull as the Operator GPU-mirror hooks above). See the
// header banner for why a standalone .cu would not be pulled.
#include "mtpq_f32_impl.cuh"

#endif  // WITH_CUDA

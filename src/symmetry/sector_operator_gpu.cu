// =============================================================================
// src/symmetry/sector_operator_gpu.cu
//
// WITH_CUDA definition of ``ed::symmetry::SectorOperator::bind_cuda()``.
// Compiled into ``ed_solvers_gpu``. The matching non-CUDA throwing stub
// lives in ``sector_operator_gpu.cpp`` (ed_core).
//
// This is the GPU lane of the operator-collapse work: it makes a
// standalone ``SectorOperator`` backend-complete (CPU + GPU), matching
// the legacy ``StreamingSymmetryOperator::SectorView`` so the production
// sector loop can be cut over with zero caller changes. The heavy lifting
// (device mirror build + unified kernel launch) is shared with the legacy
// path via ``ed::symmetry::make_sector_matvec_gpu``.
//
// Phase A of the operator-collapse GPU-parity work (Jun 2026).
// =============================================================================

#ifdef WITH_CUDA

#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_gpu_mirror.h>

#include <cstdlib>

namespace {
// Gate for the on-the-fly representative GPU matvec ("On-the-fly
// representative SpMV" plan, Jun 2026). DEFAULT ON after the validation cycle
// (test_rep_symmetry_gpu pins it bit-for-bit vs the CPU applySymmetrized
// reference across n_up/k sectors incl. the |G|=8 fixed-Sz fixture). It is the
// resident N=32 Sz+Symm path: no orbit CSR, no O(full-Sz-dim) projection
// table. Set ``ED_GPU_SYMMETRY_REP=0`` to opt back into the legacy orbit-CSR
// mirror (diagnostics / bisection). Only fixed-Sz sectors take this path
// (``rep_data_.usable()``); sym-only full-Hilbert sectors always fall back.
inline bool rep_path_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("ED_GPU_SYMMETRY_REP");
        if (e == nullptr || e[0] == '\0') return true;   // default ON
        if (e[0] == '0' && e[1] == '\0')  return false;  // "0" -> OFF
        return true;                                     // anything else -> ON
    }();
    return enabled;
}
}  // namespace

// Operator-collapse Phase 4: SectorOperator is now the alias
// SubspaceOperator<SymmetryBasisPolicy, Host>; this strong definition of its
// bind_cuda_impl_ member specialization replaces the legacy out-of-line
// ed::symmetry::SectorOperator::bind_cuda. The orbit data + lazy-rep state come
// from the owned SectorBasis producer.
template <>
ed::LinearOperator::MatvecFn
ed::SubspaceOperator<ed::matvec::basis::SymmetryBasisPolicy,
                     ed::matvec::MemorySpace::Host>::bind_cuda_impl_() const {
    // Bake any pending in-place transforms into ``terms_`` before the
    // device mirror snapshots the term SoA (mirrors the legacy
    // ``bind_cuda_for_sector`` ordering).
    commitPendingTransforms();

    // On-the-fly representative path: resident, no orbit CSR / no
    // O(full-Sz-dim) projection table. Engaged when the env gate is on AND
    // this is a fixed-Sz sector with a usable RepSectorData. The factory
    // either populates ``rep_data_`` eagerly or supplies a CSR-free provider
    // (``configureRepLazy``); ``ensure_rep_data_`` builds it on first use,
    // WITHOUT ever materialising the host orbit CSR. Sym-only sectors leave
    // it unusable and fall through to the orbit-CSR mirror.
    //
    // Backend-specific policy: unlike the CPU (where the precomputed orbit-CSR
    // walk is ~100x faster per matvec than regenerating the group action), the
    // GPU rep kernel is FASTER than the orbit-CSR mirror -- it skips the Pass-2
    // CSR build + the multi-GiB device upload, and the per-element min-image /
    // projection recompute is hidden by the thousands of resident threads
    // (measured N=24: rep 4.8 s vs CSR-mirror 18.2 s end-to-end). So the GPU
    // takes the rep path in BOTH regimes whenever it is usable; only the CPU
    // reserves it for the lazy regime (see ``make_backend_``).
    // Unified decision: the planner can force the orbit-CSR mirror by setting the
    // strategy SLOT to OrbitMaterialized; otherwise the GPU prefers the rep path
    // in both regimes (faster than the mirror). We read the RAW slot (not the
    // env-folded resolver) on purpose: the CPU env knobs (ED_SYM_REP /
    // ED_SYM_REDUCED_CSR) are CPU-lane overrides; the GPU's own override is
    // ED_GPU_SYMMETRY_REP (rep_path_enabled). (RepStream / RepReducedCsr both ->
    // GPU rep; the CPU-only reduced-CSR sub-choice does not apply on device.)
    const bool planner_forces_orbit =
        ed::planner::sym_matvec_repr()
        == static_cast<int>(ed::planner::SymMatvecRepr::OrbitMaterialized);
    if (rep_path_enabled() && !planner_forces_orbit) {
        const ed::symmetry::RepSectorData& rd = producer_.ensureRepData();
        if (rd.usable()) {
            return ed::symmetry::make_sector_matvec_gpu_rep(
                rd,
                static_cast<double>(spin_l_),
                terms_);
        }
    }

    // Fallback orbit-CSR mirror: needs the host orbit CSR. In CSR-free lazy
    // mode it has not been built yet -- materialise it now (only reached for
    // sym-only sectors or when ED_GPU_SYMMETRY_REP=0).
    producer_.ensureHostCsr();

    // One sector == one mirror, built once and captured by the returned
    // callable. ``sector_n_up_()`` returns the shared magnetization of a
    // fixed-Sz sector (every orbit representative has the same popcount),
    // which selects the O(1) dense combinadic rank-table device lookup;
    // it returns -1 for a full-Hilbert (sym-only) sector, falling back to
    // the open-addressing hash that handles mixed-popcount states.
    return ed::symmetry::make_sector_matvec_gpu(
        producer_.sector(),
        static_cast<double>(producer_.group_size()),
        static_cast<double>(spin_l_),
        terms_,
        static_cast<int>(n_bits_),
        /*n_up=*/sector_n_up_());
}

#endif  // WITH_CUDA

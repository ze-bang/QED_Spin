#pragma once
// =============================================================================
// include/ed/symmetry/sector_operator.h
//
// ed::symmetry::SectorOperator: the standalone per-sector symmetry operator.
//
// Operator-collapse Phase 4 (Jun 2026): SectorOperator is now a using-alias
// over the collapsed ``ed::SubspaceOperator<SymmetryBasisPolicy>`` template.
// It OWNS the orbit data for exactly ONE symmetry sector as an
// ``ed::symmetry::SectorBasis`` producer (which also carries the CSR-free
// lazy-rep state), and inherits the canonical term storage + management from
// ``ed::Operator``. The matvec runs through the unified
// ``CpuMatVecBackend<SymmetryBasisPolicy>`` (orbit-CSR walk) or
// ``CpuMatVecBackend<RepSymmetryBasisPolicy>`` (CSR-free on-the-fly rep walk).
//
// All construction sites (``SectorOperator(n_bits, spin_l, SectorBasis)``) and
// the public surface (basis / materialized_basis / configureRepLazy /
// setRepSectorData / repSectorData / rep_lazy / host_csr_materialized) are
// unchanged from the legacy standalone class.
// =============================================================================

#include <cstdlib>
#include <memory>

#include <ed/core/subspace_operator.h>
#include <ed/matvec/symmetry_basis_policy.h>
#include <ed/planner/sym_matvec_policy_hook.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_gpu_mirror.h>
#include <ed/symmetry/rep_sector_data.h>

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// CPU on-the-fly representative SpMV gate ("Optimized symmetry ED + NLCE"
// plan, Jun 2026). When enabled (default ON), a fixed-Sz symmetry sector's CPU
// matvec runs the CSR-free representative kernel
// (``make_cpu_rep_symmetry_backend``) instead of materialising the per-sector
// orbit CSR (~24 GiB/sector at N=32). Set ``ED_SYM_REP=0`` to restore the
// legacy orbit-CSR path (A/B + bisection).
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool cpu_rep_symmetry_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("ED_SYM_REP");
        if (e == nullptr || e[0] == '\0') return true;   // default ON
        if (e[0] == '0' && e[1] == '\0')  return false;  // "0" -> OFF
        return true;                                      // anything else -> ON
    }();
    return enabled;
}

}  // namespace ed::symmetry

namespace ed {

// ---------------------------------------------------------------------------
// SubspaceProducerTraits specialization for the Symmetry lane: the owning
// producer is ed::symmetry::SectorBasis (orbit data + lazy-rep state). Defined
// here, where SectorBasis is a complete type, keeping the symmetry headers out
// of the leaf subspace_operator.h.
// ---------------------------------------------------------------------------
template <>
struct SubspaceProducerTraits<ed::matvec::basis::SymmetryBasisPolicy> {
    using type = ed::symmetry::SectorBasis;
};

// ---------------------------------------------------------------------------
// make_backend_ specialization (Symmetry lane).
//
// CPU on-the-fly representative path: for a fixed-Sz symmetry sector the
// CSR-free RepSectorData is all the matvec needs (NEVER materialises the
// per-sector orbit CSR). Taken only in the lazy regime (``rep_lazy()``);
// otherwise the eager orbit-CSR backend is faster when the CSR fits. Falls
// back to the orbit-CSR path when the rep path is disabled (ED_SYM_REP=0) or
// the RepSectorData is unusable (sym-only sectors with varying popcount).
// ---------------------------------------------------------------------------
template <>
inline std::unique_ptr<ed::matvec::MatVecBackendBase>
SubspaceOperator<ed::matvec::basis::SymmetryBasisPolicy,
                 ed::matvec::MemorySpace::Host>::make_backend_() const {
    // Planner decides rep walk vs orbit-CSR (sym_matvec_policy_hook). When unset
    // (Auto) fall back to the producer's rep_lazy() heuristic, so behaviour is
    // unchanged unless the planner ran. OrbitCsr forces the materialized path.
    const int  sym_repr = ed::planner::sym_matvec_repr();
    const bool want_rep =
        (sym_repr == static_cast<int>(ed::planner::SymMatvecRepr::Rep)) ||
        (sym_repr == static_cast<int>(ed::planner::SymMatvecRepr::Auto)
         && producer_.rep_lazy());
    if (want_rep && ed::symmetry::cpu_rep_symmetry_enabled()) {
        const ed::symmetry::RepSectorData& rd = producer_.ensureRepData();
        if (rd.usable()) {
            return ed::matvec::make_cpu_rep_symmetry_backend<
                DiagonalOneBody, OffDiagonalOneBody,
                DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
                ThreeBodyTransformData>(rd);
        }
    }
    // Legacy orbit-CSR path. In CSR-free lazy mode the host CSR has not been
    // built yet -- materialise it now (once).
    producer_.ensureHostCsr();
    return ed::matvec::make_cpu_symmetry_backend<
        DiagonalOneBody, OffDiagonalOneBody,
        DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
        ThreeBodyTransformData>(producer_.policy());
}

#ifdef WITH_CUDA
// Device matvec build for the Symmetry lane. Strong definition in
// src/symmetry/sector_operator_gpu.cu (ed_solvers_gpu). No weak fallback: a
// CPU-only WITH_CUDA binary that constructs a SectorOperator must link
// ed_solvers_gpu (matches the legacy out-of-line bind_cuda key function).
template <>
ed::LinearOperator::MatvecFn
SubspaceOperator<ed::matvec::basis::SymmetryBasisPolicy,
                 ed::matvec::MemorySpace::Host>::bind_cuda_impl_() const;
#endif

}  // namespace ed

namespace ed::symmetry {

// Back-compat alias: ed::symmetry::SectorOperator ==
// ed::SubspaceOperator<SymmetryBasisPolicy>.
using SectorOperator =
    ed::SubspaceOperator<ed::matvec::basis::SymmetryBasisPolicy>;

}  // namespace ed::symmetry

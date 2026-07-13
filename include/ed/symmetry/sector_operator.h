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
// ``CpuMatVecBackend<RepSymmetryBasisPolicy>`` (CSR-free on-the-fly rep walk,
// with the budget-gated reduced-CSR sub-mode).
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
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_gpu_mirror.h>
#include <ed/symmetry/rep_sector_data.h>

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// Stage 11c-2b (Jul 2026): the CSR-free representative kernel is THE CPU
// symmetry matvec. The legacy orbit-CSR backend and its ``ED_SYM_REP=0``
// escape were retired -- every production SectorBasis carries a rep
// provider (11c-1/2a), so the escape was the only route in.
// ---------------------------------------------------------------------------

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
// CPU on-the-fly representative path: the CSR-free RepSectorData is all the
// matvec needs (the per-sector orbit CSR is never materialised for SpMV).
// ---------------------------------------------------------------------------
template <>
inline std::unique_ptr<ed::matvec::MatVecBackendBase>
SubspaceOperator<ed::matvec::basis::SymmetryBasisPolicy,
                 ed::matvec::MemorySpace::Host>::make_backend_() const {
    // ONE representation (Stage 11c-2b): the CSR-free RepSectorData drives
    // both sub-modes -- RepStream walks representatives on the fly;
    // RepReducedCsr (default) assembles the reduced sector matrix straight
    // from ``index_and_projection`` (build_reduced_symmetry_csr_rep) inside
    // the rep backend, budget-gated. The per-sector orbit CSR is never a
    // matvec representation any more.
    const ed::symmetry::RepSectorData& rd = producer_.ensureRepData();
    if (!rd.usable()) {
        throw std::runtime_error(
            "SectorOperator::make_backend_: RepSectorData unusable -- every "
            "production SectorBasis carries a rep provider since Stage "
            "11c-2a, so this indicates a construction bug, not a "
            "configuration problem.");
    }
    return ed::matvec::make_cpu_rep_symmetry_backend<
        DiagonalOneBody, OffDiagonalOneBody,
        DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
        ThreeBodyTransformData>(rd);
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

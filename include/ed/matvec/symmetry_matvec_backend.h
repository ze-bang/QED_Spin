#pragma once
// =============================================================================
// include/ed/matvec/symmetry_matvec_backend.h
//
// CPU symmetry backend factories (rep + non-abelian oracle).
//
// ``matvec_backend.h`` ships ``make_cpu_full_basis_backend`` (FullBasisPolicy)
// and ``make_cpu_fixed_sz_backend`` (FixedSzBasisPolicy) but deliberately
// stays free of any symmetry dependency so it remains a light, host-only
// leaf header. The symmetry policy lives in ``symmetry_basis_policy.h``,
// which transitively pulls in the heavyweight ``streaming_symmetry.h`` /
// ``operator.h`` chain -- including that from ``matvec_backend.h`` would
// create an include cycle (operator.h already includes matvec_backend.h).
//
// This header is the leaf that ties the two together: it constructs a
// ``CpuMatVecBackend<SymmetryBasisPolicy, ...>`` over a non-owning
// ``SymmetryBasisPolicy`` view (typically obtained from a live
// ``ed::symmetry::SectorBasis::policy()``). The backend then drives the
// unified matrix-free kernel (``apply_terms<SymmetryBasisPolicy, Scalar>``),
// whose ``if constexpr (needs_orbit_walk)`` branch performs the orbit walk
// and applies the per-emit symmetry weighting.
//
// CSR / real-fast-path safety: ``CpuMatVecBackend`` compiles out the
// assembled-CSR and real-input fast paths for any policy with
// ``needs_orbit_walk == true`` (see the ``if constexpr`` guards in
// apply_complex / apply_real). A symmetry sector therefore always runs the
// complex matrix-free kernel from apply_complex, which is the only path
// that preserves the imaginary part of complex momentum phases. The
// tunables passed here are consequently inert for the symmetry lane; they
// are forwarded only so the backend's ABI matches the other two factories.
//
// Lifetime: the returned backend stores the ``SymmetryBasisPolicy`` BY
// VALUE, but that POD view holds non-owning pointers into the
// ``SymmetrySector`` and lookup index owned by the originating
// ``SectorBasis``. The SectorBasis (or whatever owns the sector) MUST
// outlive the backend.
// =============================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <ed/matvec/matvec_backend.h>
#include <ed/matvec/symmetry_basis_policy.h>
#include <ed/matvec/nonabelian_symmetry_basis_policy.h>
#include <ed/matvec/rep_symmetry_basis_policy.h>
#include <ed/symmetry/rep_sector_data.h>

namespace ed::matvec {

// (Stage 11c-2b: ``make_cpu_symmetry_backend`` -- the orbit-CSR walk over a
// ``SymmetryBasisPolicy`` view -- was deleted with the legacy orbit matvec
// lane; the rep factory below is THE CPU symmetry backend.)

// ---------------------------------------------------------------------------
// make_cpu_nonabelian_symmetry_backend: same engine, the d≥2 (non-abelian)
// policy. The SAB sector (norm = 1) + multi-target lookup are viewed by the
// policy; the matvec forces the SCATTER kernel (multiplicity emits to several
// targets). Identical construction to the abelian factory above — non-abelian
// is just another BasisPolicy on CpuMatVecBackend.
// ---------------------------------------------------------------------------
template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
[[nodiscard]] inline std::unique_ptr<MatVecBackendBase>
make_cpu_nonabelian_symmetry_backend(basis::NonAbelianSymmetryBasisPolicy policy)
{
    using Backend = CpuMatVecBackend<basis::NonAbelianSymmetryBasisPolicy,
                                     DiagOne, OffDiagOne, DiagTwo, MixedTwo,
                                     OffDiagTwo, ThreeBody>;
    auto tunables = detail::read_symmetry_tunables(1ULL << 13);
    const std::uint64_t dim = policy.dim();
    return std::make_unique<Backend>(
        std::move(policy), tunables,
        "CpuNonAbelian(dim=" + std::to_string(dim) + ")");
}

// ---------------------------------------------------------------------------
// make_cpu_rep_symmetry_backend: the CPU on-the-fly representative SpMV
// backend ("Optimized symmetry ED + NLCE" plan, Jun 2026). Builds a
// ``CpuMatVecBackend<RepSymmetryBasisPolicy, ...>`` over a non-owning view
// into a ``RepSectorData`` (reps + 1/norm + group perms + per-sector
// characters). NO orbit CSR is materialised; the group action + projection
// phase are regenerated arithmetically in the matvec.
//
// Lifetime: the ``RepSectorData`` (typically ``SectorOperator::rep_data_``)
// MUST outlive the returned backend (the policy holds raw pointers into its
// vectors).
// ---------------------------------------------------------------------------
[[nodiscard]] inline basis::RepSymmetryBasisPolicy
rep_policy_from(const ed::symmetry::RepSectorData& rd) noexcept
{
    // Single source of the mapping now lives on RepSectorData (so the dense
    // assembly lane and the matvec factory can never drift). Kept as a thin
    // forwarder for the existing call sites.
    return rd.make_policy();
}

template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
[[nodiscard]] inline std::unique_ptr<MatVecBackendBase>
make_cpu_rep_symmetry_backend(const ed::symmetry::RepSectorData& rd)
{
    using Backend = CpuMatVecBackend<basis::RepSymmetryBasisPolicy,
                                     DiagOne, OffDiagOne, DiagTwo, MixedTwo,
                                     OffDiagTwo, ThreeBody>;
    // The rep policy forces the complex matrix-free path; tunables are inert
    // (no CSR branch) but forwarded for ABI parity with the other factories.
    auto tunables = detail::read_symmetry_tunables();
    const std::uint64_t dim = rd.reps.size();
    return std::make_unique<Backend>(
        rep_policy_from(rd),
        tunables,
        "CpuRepSymmetry(dim=" + std::to_string(dim) + ")");
}

// On-the-fly representative host cell (Jun 2026). Definition in
// src/matvec/cpu_backend_instantiations.cpp.
extern template class CpuMatVecBackend<basis::RepSymmetryBasisPolicy,
                                       DiagOneBody, OffDiagOneBody, DiagTwoBody,
                                       MixedTwoBody, OffDiagTwoBody, ThreeBodyTerm>;

} // namespace ed::matvec

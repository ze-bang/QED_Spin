#pragma once
// =============================================================================
// include/ed/matvec/symmetry_matvec_backend.h
//
// CPU symmetry backend factory (on-the-fly representative SpMV).
//
// ``matvec_backend.h`` ships ``make_cpu_full_basis_backend`` (FullBasisPolicy)
// and ``make_cpu_fixed_sz_backend`` (FixedSzBasisPolicy) but deliberately
// stays free of any symmetry dependency so it remains a light, host-only
// leaf header. This header ties the representative symmetry policy to the
// backend: ``CpuMatVecBackend`` compiles out the assembled-CSR and real-input
// fast paths for the rep policy (see the ``if constexpr`` guards in
// apply_complex / apply_real), so a symmetry sector always runs a matrix-free
// or reduced-CSR rep kernel that preserves complex momentum phases.
//
// Lifetime: the returned backend stores the policy BY VALUE, but that POD
// view holds non-owning pointers into the ``RepSectorData`` it was built
// from, which MUST outlive the backend.
// =============================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <ed/matvec/matvec_backend.h>
#include <ed/matvec/symmetry_basis_policy.h>
#include <ed/matvec/rep_symmetry_basis_policy.h>
#include <ed/symmetry/rep_sector_data.h>

namespace ed::matvec {

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

#pragma once
// =============================================================================
// include/ed/matvec/symmetry_matvec_backend.h
//
// make_cpu_symmetry_backend: the missing third CPU backend factory.
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

namespace ed::matvec {

template <class DiagOne, class OffDiagOne, class DiagTwo, class MixedTwo,
          class OffDiagTwo, class ThreeBody>
[[nodiscard]] inline std::unique_ptr<MatVecBackendBase>
make_cpu_symmetry_backend(basis::SymmetryBasisPolicy policy,
                          std::uint64_t default_csr_cutoff = (1ULL << 13))
{
    using Backend = CpuMatVecBackend<basis::SymmetryBasisPolicy,
                                     DiagOne, OffDiagOne, DiagTwo, MixedTwo,
                                     OffDiagTwo, ThreeBody>;
    // read_symmetry_tunables honours ED_SYM_CSR_DIM_MAX / ED_CSR_DIM_MAX /
    // ED_CSR_FORCE for ABI parity, but the symmetry lane never actually
    // takes the CSR branch (compiled out by needs_orbit_walk).
    auto tunables = detail::read_symmetry_tunables(default_csr_cutoff);
    const std::uint64_t dim = policy.dim();
    return std::make_unique<Backend>(
        std::move(policy),
        tunables,
        "CpuSymmetry(dim=" + std::to_string(dim) + ")");
}

// ---------------------------------------------------------------------------
// P6 (operator-collapse): extern-template declaration for the Symmetry host
// cell over the canonical term-view shape. The explicit instantiation
// DEFINITION lives in src/matvec/cpu_backend_instantiations.cpp (ed_matvec);
// every consumer links ed_matvec transitively. Kept here -- rather than in
// matvec_backend.h -- because SymmetryBasisPolicy drags in the heavyweight
// streaming-symmetry chain that the matvec_backend.h leaf header avoids.
// ``DiagOneBody`` ... ``ThreeBodyTerm`` are visible via term_storage.h, which
// matvec_backend.h now includes.
// ---------------------------------------------------------------------------
extern template class CpuMatVecBackend<basis::SymmetryBasisPolicy,
                                       DiagOneBody, OffDiagOneBody, DiagTwoBody,
                                       MixedTwoBody, OffDiagTwoBody, ThreeBodyTerm>;

} // namespace ed::matvec

#pragma once
// =============================================================================
// tests/common/symmetry_reference.h
//
// Carrier-free reference for the symmetry-adapted (orbit-basis) matvec.
//
// The legacy parity tests compared the unified
// ``CpuMatVecBackend<SymmetryBasisPolicy>`` / ``SectorOperator`` path against
// ``StreamingSymmetryOperator::applySymmetrized``. After the operator-collapse
// carrier retirement (Phase 3, Jun 2026) that golden reference is gone, so the
// tests instead pin against an INDEPENDENT dense reference built from:
//
//   * the sector's orbit data (``::SymmetrySector`` -- orbit elements,
//     coefficients, per-state norm), which the kept ``SectorBasis`` produces,
//     and
//   * a full-Hilbert apply ``H |psi>`` over the 2^N computational basis (the
//     thoroughly-tested ``Operator`` / ``FullBasisPolicy`` matvec).
//
// Convention (matches ``SymmetryBasisPolicy`` exactly, see
// ``include/ed/matvec/symmetry_basis_policy.h``):
//
//   out[k] = group_norm * (1/norm_k)
//            * sum_{s'} conj(coeff_{k,s'}) * (H psi)[s'],
//   psi[s] = sum_j in[j] * coeff_{j,s} / norm_j,
//   group_norm = 1 / |G|.
//
// i.e. the backend matvec equals ``group_norm * P^dagger H P`` where ``P`` has
// columns ``coeff_{j,.} / norm_j``. Building ``psi`` / projecting back uses the
// same orbit coefficients the backend reads, so the only independent ingredient
// is the full-Hilbert ``H`` apply -- no symmetry carrier required.
//
// O(2^N) per call; intended for the small-N unit tests only.
// =============================================================================

#include <ed/core/linear_operator.h>            // Complex
#include <ed/symmetry/symmetry_sector_data.h>   // ::SymmetrySector

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

namespace ed_tests {

/// Independent symmetrized matvec reference.
///
/// @param sec        Materialised sector orbit data (e.g. ``sb.sector()``).
/// @param n_bits     Lattice site count (full Hilbert dim = 2^n_bits).
/// @param group_size |G| (== ``info.max_clique.size()``).
/// @param full_apply Full-Hilbert ``H`` apply over the 2^n_bits basis.
/// @param in         Orbit-basis input vector, length ``dim``.
/// @param out        Orbit-basis output vector, length ``dim`` (overwritten).
/// @param dim        Sector dimension (== ``sec.basis_states.size()``).
inline void apply_symmetrized_reference(
    const ::SymmetrySector&                                            sec,
    std::uint64_t                                                      n_bits,
    double                                                             group_size,
    const std::function<void(const Complex*, Complex*, std::size_t)>&  full_apply,
    const Complex*                                                     in,
    Complex*                                                           out,
    std::size_t                                                        dim)
{
    const std::uint64_t full_dim = (1ULL << n_bits);

    // psi = P x  (full-space spread of the orbit-basis input).
    std::vector<Complex> psi(full_dim, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < dim; ++j) {
        const auto& bs = sec.basis_states[j];
        const double inv_norm_j = (bs.norm > 0.0) ? (1.0 / bs.norm) : 0.0;
        const Complex xj = in[j];
        const std::size_t M = bs.orbit_elements.size();
        for (std::size_t t = 0; t < M; ++t) {
            psi[bs.orbit_elements[t]] +=
                xj * bs.orbit_coefficients[t] * inv_norm_j;
        }
    }

    // phi = H psi  (full-Hilbert matvec; the independent ingredient).
    std::vector<Complex> phi(full_dim, Complex(0.0, 0.0));
    full_apply(psi.data(), phi.data(), static_cast<std::size_t>(full_dim));

    // out = group_norm * P^dagger phi.
    const double group_norm = (group_size > 0.0) ? (1.0 / group_size) : 0.0;
    for (std::size_t k = 0; k < dim; ++k) {
        const auto& bs = sec.basis_states[k];
        const double inv_norm_k = (bs.norm > 0.0) ? (1.0 / bs.norm) : 0.0;
        Complex acc(0.0, 0.0);
        const std::size_t M = bs.orbit_elements.size();
        for (std::size_t t = 0; t < M; ++t) {
            acc += std::conj(bs.orbit_coefficients[t])
                 * phi[bs.orbit_elements[t]];
        }
        out[k] = acc * (group_norm * inv_norm_k);
    }
}

}  // namespace ed_tests

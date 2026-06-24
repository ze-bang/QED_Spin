#pragma once
// =============================================================================
// include/ed/symmetry/symmetry_adapted.h
//
// Symmetry-adapted basis (SAB) construction for a FULL (possibly non-abelian)
// finite group, using the numerically-decomposed irreps from irreps.h.
//
// For an irrep Γ and a fixed partner n0=0, the SAB partner functions over an
// orbit are the column space of the projector
//     P^Γ_{00} = (d_Γ/|G|) Σ_g  D^Γ_{00}(g)*  U(g),     U(g)|x> = |perm_g(x)>
// applied to the orbit. An orbit ≅ G/stab carries Γ with multiplicity
//     m_Γ(O) = (1/|stab|) Σ_{h∈stab} χ_Γ(h)
// (= d_Γ for a free orbit), so the orbit yields m_Γ(O) orthonormal partner-0
// SAB vectors -- obtained by projecting the orbit and taking an SVD column
// basis. For 1-D irreps this collapses to the usual one-vector-per-rep abelian
// construction (m ∈ {0,1}).
//
// Completeness: Σ_Γ d_Γ · (#partner-0 SAB vectors of Γ) == 2^n_sites. The
// per-Γ block Hamiltonian H_Γ = Φ_Γ† H Φ_Γ has the Γ-sector eigenvalues, each
// with physical degeneracy d_Γ; recombining over Γ reproduces the full
// spectrum. This is the reference construction (correct for any group); the
// production rep-path matvec is an optimisation of the d_Γ=1 / multiplicity=1
// rows of it.
// =============================================================================

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

#include <ed/symmetry/irreps.h>

namespace ed::symmetry {

/// One symmetry-adapted basis vector |φ> = Σ_k coeffs[k] |states[k]>,
/// orthonormal, supported on one orbit.
struct SABVector {
    std::vector<std::uint64_t>        states;
    std::vector<std::complex<double>> coeffs;
};

/// All partner-0 SAB vectors for irrep `irrep_index` over the full 2^n_sites
/// Hilbert space (handles non-abelian multiplicity). `max_clique` is the closed
/// permutation group (as passed to `decompose_irreps`, same indexing as `gi`).
[[nodiscard]] std::vector<SABVector>
build_sab_partition0(const GroupIrreps&                    gi,
                     const std::vector<std::vector<int>>&  max_clique,
                     int                                   irrep_index,
                     int                                   n_sites);

/// Full symmetry-adapted spectrum of a Hamiltonian `H_full` (a 2^n_sites
/// matvec, `H_full(in, out, dim)`) under the group `gi` / `max_clique`,
/// correct for ANY (abelian or non-abelian) point group. For each irrep Γ it
/// builds the partner-0 SAB, materialises the small block H_Γ = Φ_Γ† H Φ_Γ via
/// the matvec, diagonalises it, and emits each eigenvalue with its physical
/// degeneracy d_Γ. Returns the sorted full spectrum (length 2^n_sites). H must
/// commute with the group (else throws on a non-Hermitian block).
///
/// This is the correct reference solver; it embeds SAB vectors in the full 2^n
/// space (so it is O(2^n) memory — a moderate-N convenience, not the at-scale
/// rep-path matvec).
struct SymAdaptedSpectrum {
    std::vector<double> eigenvalues;   ///< sorted, with d_Γ multiplicities
    std::vector<int>    block_irrep_dim;  ///< d_Γ of each non-empty block
    std::vector<int>    block_size;       ///< n_Γ (# SAB partner-0 vectors) per block
};

[[nodiscard]] SymAdaptedSpectrum
symmetry_adapted_spectrum(
    const std::function<void(const std::complex<double>*,
                             std::complex<double>*, std::uint64_t)>& H_full,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites);

}  // namespace ed::symmetry

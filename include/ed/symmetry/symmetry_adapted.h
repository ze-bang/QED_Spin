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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ed/core/thermal_types.h>   // ThermodynamicData
#include <ed/symmetry/irreps.h>
#include <ed/symmetry/symmetry_sector_data.h>  // ::SymmetrySector, ::SymBasisState

class Operator;  // global; the GPU consumer decls take a const ::Operator&

namespace ed::symmetry {

/// One symmetry-adapted basis vector |φ> = Σ_k coeffs[k] |states[k]>,
/// orthonormal, supported on one orbit.
struct SABVector {
    std::vector<std::uint64_t>        states;
    std::vector<std::complex<double>> coeffs;
};

/// All partner-0 SAB vectors for irrep `irrep_index` (handles non-abelian
/// multiplicity). `max_clique` is the closed permutation group (as passed to
/// `decompose_irreps`, same indexing as `gi`).
///
/// `n_up`: if < 0, enumerate over the full 2^n_sites Hilbert space. If ≥ 0,
/// restrict to the fixed-Sz subspace (states of popcount n_up) — the combined
/// U(1)×G reduction. Since site permutations preserve popcount, every orbit of
/// a fixed-Sz state stays in that Sz sector, so the projector / multiplicity
/// math is unchanged; only the enumeration source differs.
[[nodiscard]] std::vector<SABVector>
build_sab_partition0(const GroupIrreps&                    gi,
                     const std::vector<std::vector<int>>&  max_clique,
                     int                                   irrep_index,
                     int                                   n_sites,
                     int                                   n_up   = -1,
                     int                                   partner = 0,
                     int                                   sz_parity = -1);

// ---------------------------------------------------------------------------
// Pack the partner-0 SAB of one irrep into the PRODUCTION sector data structure
// (::SymmetrySector / ::SymBasisState), so non-abelian reduction reuses the same
// representation as abelian/Sz instead of a parallel one. Each SAB vector becomes
// one SymBasisState (orbit_elements = states, orbit_coefficients = coeffs); the
// SAB is already orthonormal, so norm = 1 (and the matvec uses group_norm = 1),
// which makes SymmetryBasisPolicy's coeff_modifier/iter_orbit reproduce the SAB
// matvec verbatim. The ONE difference from abelian is multiplicity: several SAB
// vectors share an orbit, so a state maps to MANY basis indices (handled by the
// multi-target lookup, not here). `n_up >= 0` restricts to the fixed-Sz sector.
[[nodiscard]] ::SymmetrySector
build_symmetry_adapted_sector(const GroupIrreps&                   gi,
                              const std::vector<std::vector<int>>& max_clique,
                              int                                  irrep_index,
                              int                                  n_sites,
                              int                                  n_up    = -1,
                              int                                  partner = 0,
                              int                                  sz_parity = -1);

/// Result of a symmetry-adapted spectrum: sorted eigenvalues with d_Γ
/// multiplicities + per-block (n_Γ, d_Γ). Filled by the ed_solvers engine
/// consumers (ed::solvers::symmetry_adapted_full_spectrum etc.).
struct SymAdaptedSpectrum {
    std::vector<double> eigenvalues;
    std::vector<int>    block_irrep_dim;
    std::vector<int>    block_size;
};

/// `connect(s, emit)` invokes `emit(s', <s'|H|s>)` for every connected state —
/// the sparse row-enumerator contract (e.g. Operator::for_each_connected_state).
using ConnectFn = std::function<void(
    std::uint64_t,
    const std::function<void(std::uint64_t, std::complex<double>)>&)>;

/// Exact canonical Z/E/C/S(β) from a full eigenvalue list (multiplicities folded
/// in). Shared by the CPU and GPU finite-T consumers.
[[nodiscard]] ThermodynamicData
canonical_thermo_from_eigs(const std::vector<double>& eigenvalues,
                           const std::vector<double>& temperatures);

// Packed per-irrep blocks H_Γ (column-major, concatenated) — the GPU hand-off
// boundary. Built by ed::solvers::symmetry_adapted_blocks_packed (engine-
// materialised); consumed by the batched GPU eigensolver.
struct SymBlocksPacked {
    std::vector<int>                  block_dim;        ///< n_Γ per block
    std::vector<int>                  block_irrep_dim;  ///< d_Γ per block
    std::vector<std::size_t>          offset;           ///< start of each block in `data`
    std::vector<std::complex<double>> data;             ///< concatenated column-major H_Γ
};

// ---------------------------------------------------------------------------
// GPU consumers (defined in src/solvers/gpu/symmetry_adapted_gpu.cu; only linked
// in WITH_CUDA builds). The host materialises the blocks via the production
// engine (ed::solvers::symmetry_adapted_blocks_packed over the operator's terms);
// the GPU does the batched cuSOLVER eigensolve. One upload, one download.
// ---------------------------------------------------------------------------
[[nodiscard]] SymAdaptedSpectrum
symmetry_adapted_spectrum_gpu(
    const ::Operator&                     op,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   n_up = -1);

[[nodiscard]] ThermodynamicData
symmetry_adapted_thermodynamics_gpu(
    const ::Operator&                     op,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    const std::vector<double>&            temperatures,
    int                                   n_up = -1);

// ---------------------------------------------------------------------------
// Ground-state DSSF result S(ω) = Σ_n |<n|O|0>|² · Lorentzian(ω-(E_n-E0)).
// Produced by ed::solvers::symmetry_adapted_ground_state_dssf.
struct SymDSSFResult {
    std::vector<double> omega;
    std::vector<double> spectral;     ///< S(ω)
    double              ground_energy = 0.0;
    double              total_weight  = 0.0;  ///< Σ_n |<n|O|0>|² = <0|O†O|0> (sum rule)
};

}  // namespace ed::symmetry

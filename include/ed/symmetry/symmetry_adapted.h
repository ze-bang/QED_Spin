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
                     int                                   partner = 0);

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

/// `n_up < 0` → full Hilbert space; `n_up ≥ 0` → the combined fixed-Sz × group
/// reduction (only states of popcount n_up; the block spectrum is that of H
/// restricted to the Sz sector, by symmetry sub-blocks Γ).
[[nodiscard]] SymAdaptedSpectrum
symmetry_adapted_spectrum(
    const std::function<void(const std::complex<double>*,
                             std::complex<double>*, std::uint64_t)>& H_full,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   n_up = -1);

/// `connect(s, emit)` must invoke `emit(s', h)` for every state s' connected to
/// s by a Hamiltonian term (h = <s'|H|s>). Such a sparse single-state row
/// enumerator (e.g. `Operator::for_each_connected_state`) lets this build each
/// block H_Γ by applying H over the orbit support only -- NO 2^n_sites vector
/// and no full-space matvec. The at-scale non-abelian path.
using ConnectFn = std::function<void(
    std::uint64_t,
    const std::function<void(std::uint64_t, std::complex<double>)>&)>;

[[nodiscard]] SymAdaptedSpectrum
symmetry_adapted_spectrum_terms(
    const ConnectFn&                      connect,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   n_up = -1);

// ---------------------------------------------------------------------------
// The reduced (per-irrep) block operator H_Γ = Φ_Γ† H Φ_Γ as a LINEAR OPERATOR,
// decoupled from how it is solved. This is the seam that makes the symmetry
// reduction orthogonal to the eigensolver: `apply` is the symmetry-reduced
// matvec (consumed by any iterative method — Lanczos / FTLM / cf-DSSF), and
// `materialize_colmajor` is the dense form (the same action sampled on unit
// columns; consumed by dense / batched eigensolvers). Both share the orbit
// support and the SAB amplitudes, so they cannot diverge.
//
// Construction is one-time (build the state→(row,amplitude) map); `apply` is
// O(orbit support · terms) per call. Lives in ed_symmetry (depends only on the
// Hamiltonian's `connect` row enumerator) — NOT on any solver, so iterative
// consumers live one layer up in ed_solvers and feed `apply` to the kernel.
// ---------------------------------------------------------------------------
class SymAdaptedBlockOp {
public:
    SymAdaptedBlockOp(ConnectFn connect, std::vector<SABVector> sab);

    [[nodiscard]] int dim() const noexcept { return nb_; }

    /// y_out[0..dim) = H_Γ · x_in[0..dim)  (the symmetry-reduced matvec).
    void apply(const std::complex<double>* x_in, std::complex<double>* y_out) const;

    /// Dense H_Γ, column-major dim×dim (Eigen-free; dense / batched eigensolvers
    /// map this directly). Equivalent to `apply` on the unit columns.
    [[nodiscard]] std::vector<std::complex<double>> materialize_colmajor() const;

private:
    ConnectFn              connect_;
    std::vector<SABVector> sab_;
    int                    nb_ = 0;
    // state s' -> [(SAB row k, coeff c^k_{s'})]  over this block's SAB.
    std::unordered_map<std::uint64_t,
                       std::vector<std::pair<int, std::complex<double>>>> bystate_;
};

// ---------------------------------------------------------------------------
// Consumer 2 — FINITE TEMPERATURE. Exact canonical thermodynamics of the
// symmetry-reduced spectrum: build the (small) per-irrep blocks H_Γ, take ALL
// their eigenvalues with multiplicity d_Γ, and evaluate Z/E/C/S(β) exactly on
// `temperatures`. (Optimal for symmetry blocks, which are dim/|G| — a dense
// per-block eigensolve beats stochastic FTLM here and is exact.) `n_up ≥ 0`
// restricts to the fixed-Sz sector (combined reduction).
// ---------------------------------------------------------------------------
[[nodiscard]] ThermodynamicData
symmetry_adapted_thermodynamics(
    const ConnectFn&                      connect,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    const std::vector<double>&            temperatures,
    int                                   n_up = -1);

// ---------------------------------------------------------------------------
// Packed symmetry blocks — the GPU hand-off boundary. Builds every non-empty
// per-irrep block H_Γ on the HOST (the irregular orbit/SAB work is cheap,
// O(reduced·|G|)), packed COLUMN-MAJOR and concatenated, so the GPU eigensolver
// uploads the whole batch in ONE transfer and runs a multi-stream batched
// Hermitian eigensolve. (`n_up ≥ 0` → fixed-Sz; partner = 0, i.e. eigenvalues —
// pass partner blocks separately for the DSSF eigenvector path.)
// ---------------------------------------------------------------------------
/// Exact canonical Z/E/C/S(β) from a full eigenvalue list (multiplicities folded
/// in). Shared by the CPU and GPU finite-T consumers.
[[nodiscard]] ThermodynamicData
canonical_thermo_from_eigs(const std::vector<double>& eigenvalues,
                           const std::vector<double>& temperatures);

struct SymBlocksPacked {
    std::vector<int>                  block_dim;        ///< n_Γ per block
    std::vector<int>                  block_irrep_dim;  ///< d_Γ per block
    std::vector<std::size_t>          offset;           ///< start of each block in `data`
    std::vector<std::complex<double>> data;             ///< concatenated column-major H_Γ
};

[[nodiscard]] SymBlocksPacked
build_symmetry_blocks_packed(
    const ConnectFn&                      connect,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   n_up = -1);

// ---------------------------------------------------------------------------
// GPU consumers (defined in src/solvers/gpu/symmetry_adapted_gpu.cu; only
// linked in WITH_CUDA builds). Same results as the CPU functions; the batched
// dense eigensolve of the symmetry blocks runs on the device. Host builds the
// blocks; one upload, multi-stream cuSOLVER eigensolve, one download.
// ---------------------------------------------------------------------------
[[nodiscard]] SymAdaptedSpectrum
symmetry_adapted_spectrum_gpu(
    const ConnectFn&                      connect,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   n_up = -1);

[[nodiscard]] ThermodynamicData
symmetry_adapted_thermodynamics_gpu(
    const ConnectFn&                      connect,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    const std::vector<double>&            temperatures,
    int                                   n_up = -1);

// ---------------------------------------------------------------------------
// Consumer 3 — GROUND-STATE DSSF. S(ω) = Σ_n |<n|O|0>|² · Lorentzian(ω-(E_n-E0)).
// Builds the symmetry blocks over the FULL Hilbert space (so every final state
// |n> is captured even when O changes the irrep / Sz), takes the global ground
// state |0>, applies O via `o_connect` (same row-enumerator contract as H's
// `connect`: o_connect(s, emit) → emit(s', <s'|O|s>)), and Lehmann-sums over
// all symmetry-block eigenstates. Correct for any (Sz-conserving or -changing)
// observable.
// ---------------------------------------------------------------------------
struct SymDSSFResult {
    std::vector<double> omega;
    std::vector<double> spectral;     ///< S(ω)
    double              ground_energy = 0.0;
    double              total_weight  = 0.0;  ///< Σ_n |<n|O|0>|² = <0|O†O|0> (sum rule)
};

[[nodiscard]] SymDSSFResult
symmetry_adapted_ground_state_dssf(
    const ConnectFn&                      h_connect,
    const ConnectFn&                      o_connect,
    const GroupIrreps&                    gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    double                                omega_min,
    double                                omega_max,
    int                                   n_omega,
    double                                broadening);

}  // namespace ed::symmetry

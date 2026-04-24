// =============================================================================
// include/ed/bfg/structure_factor.h
//
// Bond-bilinear structure factors and Fourier-transformed dimer operators on
// a BFG cluster (P2.1 structure-factor slice). Pulled out of
// `compute_bfg_order_parameters.cpp` so:
//   * the GPU driver and Python bindings can call the same kernels,
//   * the algorithms are unit-testable in isolation against analytic
//     small-cluster cases (Catch2),
//   * the CPU driver shrinks to a thin physics orchestrator.
//
// Two operating modes are supported, controlled by a process-global flag set
// via `set_memory_efficient_mode(n_states)` (call once near the top of the
// pipeline, after the wavefunction has been loaded, so all subsequent
// kernels share the same decision):
//
//   * Default (fast) mode  -- thread-local accumulators, then a single
//     critical-section reduction. Wins when each thread's working set fits
//     comfortably in RAM (~16 bytes per Hilbert-space basis state).
//
//   * Memory-efficient mode -- atomic updates into a shared real/imag buffer
//     pair. Used when total `n_threads * n_states * sizeof(Complex)` would
//     blow the host's RAM budget (the heuristic threshold is 4 GB).
//
// Both modes return numerically identical results up to floating-point
// reduction order; the equivalence is locked down in
// `tests/unit/test_bfg_structure_factor.cpp`.
//
// Bit conventions follow the rest of the engine: bit=0 -> spin UP
// (Sz=+1/2), bit=1 -> spin DOWN (Sz=-1/2). Bonds are ordered pairs (i,j)
// with `bond_centers[b]` the 2D real-space midpoint of bond b (used by the
// Fourier phase exp(i q . r_b)).
//
// Audit ref: P2.1.
// =============================================================================

#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace ed::bfg {

using Complex = std::complex<double>;

/**
 * Bundle of dimer-structure-factor pieces returned by the *_sf_direct
 * kernels.
 *   `overlap`    = ||D(q)|psi>||^2 = <psi| D^dag(q) D(q) |psi>
 *   `expect_q1`  = <psi| D(q) |psi>            (the single-Q expectation)
 *   `expect_q2`  = <psi| D(q) |psi>  (alias of expect_q1; kept for the
 *                  cross-Q future use the original code anticipated)
 *
 * The dimer structure factor consumed by callers is then
 *   S_D(q) = overlap - |expect_q1|^2.
 */
struct DimerSFResult {
    Complex overlap;
    Complex expect_q1;
    Complex expect_q2;
};

/**
 * Decide once whether subsequent dimer / Heisenberg structure-factor
 * kernels should use the atomic-update (memory-efficient) implementation
 * or the thread-local-buffer (fast) implementation.
 *
 * Heuristic: enabled when `n_threads * n_states * sizeof(Complex)` would
 * exceed 4 GB. Emits a one-line note to stdout when memory-efficient mode
 * is enabled, mirroring the previous CPU driver behaviour.
 *
 * Thread-safe to call once at startup; the global is read (without any
 * synchronization) by every kernel below, so do *not* flip the mode while
 * a kernel is running.
 */
void set_memory_efficient_mode(uint64_t n_states);

/// Read-only accessor for the global memory-efficient flag.
bool memory_efficient_mode_enabled();

/**
 * Direct evaluation of S_D(q) = <D^dag(q) D(q)> for the XY dimer operator
 *   D_b = S^+_i S^-_j + S^-_i S^+_j
 * via norm of D(q)|psi>. Avoids materialising the per-thread ket explicitly
 * twice (once for ket, once for bra) by computing the expectation in the
 * same parallel loop. O(N_bonds * Hilbert) time, O(Hilbert) memory.
 */
DimerSFResult compute_dimer_sf_direct(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
);

/**
 * Same as above for the Heisenberg dimer operator
 *   D_b = S_i . S_j = SzSz + (1/2)(S+S- + S-S+).
 */
DimerSFResult compute_heisenberg_sf_direct(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
);

/**
 * Apply the Fourier-transformed XY dimer operator
 *   D(q) = sum_b exp(i q . r_b) D_b,    D_b = S+S- + S-S+
 * to `psi` and return the resulting ket.
 */
std::vector<Complex> apply_dimer_fourier(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
);

/**
 * Apply the Fourier-transformed Heisenberg dimer operator
 *   D(q) = sum_b exp(i q . r_b) (S_i . S_j)
 * to `psi`. Returns {D(q)|psi>, <D(q)>}, where the second element is
 * computed in-line for free during the SzSz pass.
 */
std::pair<std::vector<Complex>, Complex> apply_heisenberg_dimer_fourier(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
);

/**
 * Real-space dimer-dimer correlation
 *   <psi| D_{b1} D_{b2} |psi>
 * for bond b1 = (i1, j1), b2 = (i2, j2) using the XY dimer
 *   D = S+_i S-_j + S-_i S+_j.
 * Pure 4-spin off-diagonal expectation (the bonds may share sites; the
 * caller is responsible for any double-counting cancellation that produces).
 */
Complex compute_dimer_dimer_correlation(
    const std::vector<Complex>& psi,
    int i1, int j1, int i2, int j2
);

/**
 * Real-space Heisenberg dimer-dimer correlation
 *   <psi| (S_i1 . S_j1) (S_i2 . S_j2) |psi>
 * combining the SzSz, mixed SzSz x XY, and 4-spin XY x XY terms.
 * Returns a real number because the symmetrized expectation is real on a
 * Hermitian state up to floating-point noise.
 */
double compute_heisenberg_dimer_dimer_correlation(
    const std::vector<Complex>& psi,
    int i1, int j1, int i2, int j2
);

}  // namespace ed::bfg

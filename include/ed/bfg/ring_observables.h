// =============================================================================
// include/ed/bfg/ring_observables.h
//
// Multi-spin ring-flip observables on a BFG cluster: bowtie (4-corner ring
// flip), triangle chirality (3-spin ring exchange), and the Fourier-applied
// bowtie operator used for the plaquette structure factor (P2.1
// ring-observables slice).
//
// Pulled out of `compute_bfg_order_parameters.cpp` so:
//   * the GPU driver and Python bindings can share the same kernels,
//   * each operator is unit-testable on hand-checkable spin-1/2 states,
//   * the CPU driver shrinks toward a thin physics orchestrator.
//
// `apply_bowtie_fourier` consults the global memory-efficient flag set by
// `ed::bfg::set_memory_efficient_mode` (see structure_factor.h) so it
// shares the same atomic / thread-local switching policy as the dimer
// kernels.
//
// Bit conventions: bit=0 -> spin UP, bit=1 -> spin DOWN. Sites are
// integer indices into the basis-state bit pattern.
//
// Audit ref: P2.1.
// =============================================================================

#pragma once

#include <array>
#include <complex>
#include <vector>

#include "ed/bfg/topology.h"

namespace ed::bfg {

using Complex = std::complex<double>;

/**
 * Apply the Fourier-transformed bowtie ring-flip operator
 *   P(q) = sum_bt exp(i q . r_bt) (S+_1 S-_2 S+_3 S-_4 + h.c.)
 * to `psi`, restricted to the bowties supplied (typically one
 * orientation, or all of them at once depending on what the caller wants
 * to plot). Only the outer-corner site indices `s1..s4` and the
 * `center` field of each `Bowtie` are read; `s0` and `orientation` are
 * ignored.
 *
 * Returns the resulting ket. Length matches `psi.size()`.
 */
std::vector<Complex> apply_bowtie_fourier(
    const std::vector<Bowtie>& bowties,
    const std::vector<Complex>& psi,
    const std::array<double, 2>& q
);

/**
 * Real-space bowtie ring-flip expectation
 *   <psi | S+_s1 S-_s2 S+_s3 S-_s4 + h.c. | psi>
 * for a single bowtie. Expensive in absolute terms (O(Hilbert)) but
 * embarrassingly parallel.
 */
Complex compute_bowtie_resonance(
    const std::vector<Complex>& psi,
    int s1, int s2, int s3, int s4
);

/**
 * Real-space triangle ring-exchange expectation
 *   <psi | S+_s1 S-_s2 S+_s3 + S-_s1 S+_s2 S-_s3 | psi>
 * for a single triangle. Note this is the symmetrized (S+S-S+ + S-S+S-)
 * combination, not the antisymmetric scalar chirality
 * S_1 . (S_2 x S_3) -- see `compute_triangle_chirality_imag` (future
 * slice) if you need the latter.
 */
Complex compute_triangle_chiral(
    const std::vector<Complex>& psi,
    int s1, int s2, int s3
);

}  // namespace ed::bfg

// =============================================================================
// include/ed/dssf/operator_spec.h
//
// `ed::dssf::OperatorSpec` and `build_observable_pairs()` -- the canonical,
// library-level entry point for assembling the (O1, O2, name) triplets that
// every DSSF/SSSF/static workflow needs to evaluate.
//
// Before this refactor (P1.10), the assembly logic was duplicated:
//   * src/apps/ed_main.cpp::construct_operators_from_config(...)
//   * src/apps/TPQ_DSSF.cpp::main() inline switch (lines ~3000-3270)
//
// Both copies handled the same {operator_type x basis x momentum x spin-combo
// x fixed-Sz} cross-product but drifted independently. By moving the
// authoritative implementation here, future bug fixes (e.g. for `transverse`
// SF/NSF naming) only need to be made once.
//
// Audit ref: P1.10 (DSSF PR-A) / "modern python interface" + "DSSF cleanup".
// =============================================================================

#pragma once

#include <ed/core/construct_ham.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ed::dssf {

/**
 * Options describing which DSSF observables to assemble.
 *
 * Mirrors the CLI surface of TPQ_DSSF / `ED dssf` so the parameter blob
 * can flow straight from argv into a single library call.
 */
struct OperatorSpec {
    /// "sum" | "transverse" | "sublattice" | "experimental"
    /// | "transverse_experimental"
    std::string operator_type{"sum"};

    /// "ladder" (S+/S-/Sz) or "xyz" (Sx/Sy/Sz). For `experimental*` the
    /// builder always uses xyz internally regardless of this flag.
    std::string basis{"ladder"};

    /// Pairs (op1, op2) where each op is 0/1/2 in the chosen basis. For
    /// the `single_expectation` method, set both entries to the same op.
    std::vector<std::pair<int, int>> spin_combinations;

    /// Momentum grid in units of 2π / a (each Q is a 3-vector). The
    /// builder iterates the outer product `momentum_points x spin_combinations`.
    std::vector<std::vector<double>> momentum_points;

    /// Real-space polarization vector for `transverse*` operators
    /// (must already be unit-norm).
    std::vector<double> polarization{1.0, 0.0, 0.0};

    /// Tilt angle for `experimental*` operators (Sz·cosθ + Sx·sinθ).
    double theta{0.0};

    /// Number of sublattices for `sublattice` operator type.
    std::uint64_t unit_cell_size{4};

    /// Total number of sites in the system.
    std::uint64_t num_sites{0};

    /// Local spin quantum number (typically 0.5).
    float spin_length{0.5f};

    /// If true, build operators restricted to a fixed-Sz sector with
    /// `n_up` up-spins. Otherwise build full Hilbert-space operators.
    bool use_fixed_sz{false};

    /// Number of up-spins for the fixed-Sz sector (ignored when
    /// `use_fixed_sz` is false).
    std::int64_t n_up{0};

    /// Path to the lattice positions file (passed through to every
    /// `*Operator(...)` constructor).
    std::string positions_file;
};

/**
 * Output of `build_observable_pairs`: parallel vectors of equal length.
 * Element `i` of `obs_1[i] / obs_2[i] / names[i]` describes the i-th
 * observable pair to evaluate.
 *
 * For SF/NSF-decomposed operator types (`transverse`,
 * `transverse_experimental`) we emit two consecutive entries per momentum
 * point -- the SF projection first, then the NSF projection -- matching the
 * legacy ordering so existing HDF5 file layouts stay bit-identical.
 */
struct ObservablePairs {
    std::vector<Operator>    obs_1;
    std::vector<Operator>    obs_2;
    std::vector<std::string> names;
};

/**
 * Build the set of observable pairs requested by `spec`.
 *
 * @throws std::invalid_argument if `spec.operator_type` is unrecognized,
 *         `spec.spin_combinations` is empty, `spec.momentum_points` is
 *         empty, or `spec.polarization` is not a 3-vector.
 */
ObservablePairs build_observable_pairs(const OperatorSpec& spec);

} // namespace ed::dssf

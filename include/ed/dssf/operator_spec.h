// =============================================================================
// include/ed/dssf/operator_spec.h
//
// `ed::dssf::OperatorSpec` and `build_observables()` -- the library-level entry
// point for assembling the momentum-resolved spin operators a DSSF/SSSF/static
// evaluation needs, one operator per (Q, component), each with a stable name.
//
// Single source of truth for the {operator_type x basis x momentum x component}
// cross-product: the Python bindings (`qed.dssf`) call this, and it is the only
// place observable assembly happens.
// =============================================================================

#pragma once

#include <ed/ops/construct_ham.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ed::dssf {

/**
 * Options describing which DSSF observables to assemble.
 *
 * The Python binding (`qed.dssf.OperatorSpec`) exposes the same fields
 * verbatim.
 */
struct OperatorSpec {
    /// "sum" | "transverse" | "sublattice" | "experimental"
    /// | "transverse_experimental"
    std::string operator_type{"sum"};

    /// "ladder" (S+/S-/Sz) or "xyz" (Sx/Sy/Sz). For `experimental*` the
    /// builder always uses xyz internally regardless of this flag.
    std::string basis{"ladder"};

    /// Spin components, each 0/1/2 in the chosen basis (ladder: S+, S-, Sz;
    /// xyz: Sx, Sy, Sz). One operator per (momentum point, component). The
    /// `experimental*` types take their component from `theta` and ignore this.
    std::vector<int> components;

    /// Momentum grid in units of 2π / a (each Q is a 3-vector). The
    /// builder iterates the outer product `momentum_points x components`.
    std::vector<std::vector<double>> momentum_points;

    /// Real-space polarization vector for `transverse*` operators
    /// (must already be unit-norm).
    std::vector<double> polarization{1.0, 0.0, 0.0};

    /// Tilt angle for `experimental*` operators (Sz·cosθ + Sx·sinθ).
    double theta{0.0};

    /// Number of sublattices for `sublattice` operator type.
    std::uint64_t unit_cell_size{4};

    /// Total number of (spin-1/2) sites in the system.
    std::uint64_t num_sites{0};

    /// Path to the lattice positions file (passed through to every
    /// `*Operator(...)` constructor).
    std::string positions_file;

    /// If set, the `sublattice` builder emits only this sublattice instead of
    /// every one in `0 .. unit_cell_size - 1`. Ignored for the other
    /// `operator_type`s.
    std::optional<std::uint64_t> sublattice;
};

/**
 * Output of `build_observables`: parallel vectors of equal length; element `i`
 * of `operators` / `names` is the i-th observable.
 *
 * The `transverse*` types emit two consecutive entries per momentum point (and
 * component): the projection on e1 = polarization (named `..._NSF`) first, then
 * the one on e2 (named `..._SF`); see `compute_transverse_bases`.
 */
struct Observables {
    std::vector<Operator>    operators;
    std::vector<std::string> names;
};

/**
 * Build the observables requested by `spec`.
 *
 * @throws std::invalid_argument if `spec.operator_type` is unrecognized,
 *         `spec.components` is empty, `spec.momentum_points` is empty, or
 *         `spec.polarization` is not a 3-vector.
 */
Observables build_observables(const OperatorSpec& spec);

/**
 * Compute the (e1, e2) orthonormal basis used by `transverse*` operators
 * at a single momentum point.
 *
 * - `e1` is always the polarization vector itself.
 * - `e2` = normalize(Q × polarization). When Q ∥ polarization the cross
 *   product vanishes and we fall back to {y, polarization} or
 *   {x, polarization} depending on which component of `polarization`
 *   dominates (pinned bit-for-bit by `test_dssf_operator_spec.cpp`).
 *
 * Exposed publicly so Python callers can introspect the bases that
 * `build_observables` will use internally (e.g. for logging).
 *
 * @throws std::invalid_argument if `Q` or `polarization` is not a 3-vector.
 */
std::pair<std::array<double, 3>, std::array<double, 3>>
compute_transverse_bases(const std::vector<double>& Q,
                         const std::vector<double>& polarization);

} // namespace ed::dssf

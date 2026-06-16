// =============================================================================
// include/ed/dssf/operator_spec.h
//
// `ed::dssf::OperatorSpec` and `build_observable_pairs()` -- the canonical,
// library-level entry point for assembling the (O1, O2, name) triplets that
// every DSSF/SSSF/static workflow needs to evaluate.
//
// Single source of truth for the {operator_type x basis x momentum x
// spin-combo x fixed-Sz} cross-product. Both the C++ CLI (`ED dssf`) and
// the Python bindings (`qed.dssf`) call this; bug fixes (e.g. for
// `transverse` SF/NSF naming) only need to be made once.
//
// Historical context: before P1.10 the assembly logic was duplicated
// across `ed_main.cpp::construct_operators_from_config(...)` and the now-
// deleted `src/apps/TPQ_DSSF.cpp::main()` inline switch (~270 LOC), and
// the two copies drifted. P1.10 introduced this seam; P2.14 deleted the
// `TPQ_DSSF` binary so this header is now the *only* place observable
// assembly happens.
//
// Audit ref: P1.10 (DSSF PR-A); P2.14 (DSSF PR-H).
// =============================================================================

#pragma once

#include <ed/core/construct_ham.h>
#include <ed/core/fixed_sz_operator.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ed::dssf {

/**
 * Options describing which DSSF observables to assemble.
 *
 * Mirrors the `ED dssf <method>` CLI surface so the parameter blob can
 * flow straight from argv into a single library call. The Python binding
 * (`qed.dssf.OperatorSpec`) exposes the same fields verbatim.
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

    /// If true, build only the left observable (`obs_1`) and leave
    /// `obs_2` empty. Used by the `single_expectation` workflow that
    /// evaluates ⟨ψ|O|ψ⟩ rather than ⟨ψ|O₁†O₂|ψ⟩.
    ///
    /// When set, the ladder-basis swap of the first operator index
    /// (`first = 1 - first` for op != 2) is also skipped, and the
    /// observable name uses just the first operator label (e.g. "Sz")
    /// instead of the concatenation ("SzSz"). HDF5 group names are
    /// pinned bit-for-bit by `tests/unit/test_dssf_legacy_schema.cpp`.
    bool single_obs_only{false};

    /// If set, restrict the `sublattice` builder to exactly one
    /// (sub_i, sub_j) pair instead of iterating over the full
    /// `i <= j < unit_cell_size` triangle. Ignored for non-sublattice
    /// `operator_type`s.
    ///
    /// In `single_obs_only` mode (single_expectation workflow) the
    /// emitted name uses just `_sub<sub_i>` rather than the full
    /// `_sub<sub_i>_sub<sub_j>` (pinned by `test_dssf_legacy_schema.cpp`).
    std::optional<std::pair<std::uint64_t, std::uint64_t>> sublattice_filter;
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

    /// Audit #2 (FixedSz->Operator path): parallel
    /// `shared_ptr<FixedSzOperator>` arrays of equal length to obs_1/obs_2,
    /// populated only when `spec.use_fixed_sz` is true. Needed because
    /// slicing a fixed-Sz operator into a value-type `Operator`
    /// destroys the dimension semantics: the base `Operator::apply` checks
    /// `size != (1ULL << n_bits_)` and throws on the smaller fixed-Sz
    /// dimension. CPU dispatch in workflows that consume these vectors
    /// must call `obs_1_fs[i]->apply(...)` instead of `obs_1[i].apply(...)`
    /// when `use_fixed_sz` is true. The GPU path is unaffected because
    /// `convertOperatorToGPU` only reads the basis-independent
    /// `transform_data_` member, which lives on the base and is preserved
    /// through the slice.
    std::vector<std::shared_ptr<FixedSzOperator>> obs_1_fs;
    std::vector<std::shared_ptr<FixedSzOperator>> obs_2_fs;
};

/**
 * Build the set of observable pairs requested by `spec`.
 *
 * @throws std::invalid_argument if `spec.operator_type` is unrecognized,
 *         `spec.spin_combinations` is empty, `spec.momentum_points` is
 *         empty, or `spec.polarization` is not a 3-vector.
 */
ObservablePairs build_observable_pairs(const OperatorSpec& spec);

/**
 * Compute the (e1, e2) orthonormal basis used by `transverse*` operators
 * at a single momentum point.
 *
 * - `e1` is always the polarization vector itself (the SF/longitudinal
 *   projection).
 * - `e2` = normalize(Q × polarization). When Q ∥ polarization the cross
 *   product vanishes and we fall back to {y, polarization} or
 *   {x, polarization} depending on which component of `polarization`
 *   dominates (pinned bit-for-bit by `test_dssf_operator_spec.cpp`).
 *
 * Exposed publicly so CLI / Python callers can introspect the bases that
 * `build_observable_pairs` will use internally (e.g. for logging).
 *
 * @throws std::invalid_argument if `Q` or `polarization` is not a 3-vector.
 */
std::pair<std::array<double, 3>, std::array<double, 3>>
compute_transverse_bases(const std::vector<double>& Q,
                         const std::vector<double>& polarization);

} // namespace ed::dssf

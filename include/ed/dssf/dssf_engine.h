// =============================================================================
// include/ed/dssf/dssf_engine.h
//
// `ed::dssf::run(...)` -- the canonical, library-level entry point that wraps
// the entire DSSF / SSSF / single-expectation pipeline behind a single
// `enum class DSSFMethod` dispatch table.
//
// Before this refactor (P2.2 / DSSF PR-C), every caller into the DSSF
// pipeline had to pick the correct workflow by hand:
//
//   * `compute_dynamical_response_workflow`  -- ω-resolved response
//   * `compute_static_response_workflow`     -- thermal expectation values
//   * `compute_ground_state_dssf_workflow`   -- T=0 continued fraction
//   * `TPQ_DSSF::main()`                     -- TPQ-based S(q,ω) / S(q)
//
// And the matching CLI flags duplicated the dispatch logic across
// `ed_main.cpp` (`--mode=dynamical|static|gs-dssf`), `TPQ_DSSF.cpp`'s
// 14-positional-arg form, and the future `ED dssf` subcommand (P2.4).
//
// `ed::dssf::run(...)` collapses all of those into one function. It takes a
// `DSSFRequest` (composed of an `OperatorSpec` + a `DSSFMethod` + the
// per-method parameters) and dispatches to the correct workflow under the
// hood. The body of each method is currently still hosted by the existing
// `compute_*_workflow` functions in `src/cli/workflows.cpp` -- this PR
// introduces the *seam*; later PRs (P2.3, P2.4, P2.5) move the actual
// computation bodies onto this seam one at a time so `ctest` stays green
// between commits.
//
// Audit ref: P2.2 (DSSF PR-C) of `MODERNIZATION_AUDIT.md` (§3.10 + §6).
// =============================================================================

#pragma once

#include <ed/core/ed_config.h>
#include <ed/dssf/operator_spec.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::dssf {

/**
 * Which DSSF / SSSF / static computation kernel to run.
 *
 * The numeric values are stable and may be persisted (e.g. to HDF5
 * metadata). New entries must be appended at the end so existing on-disk
 * files keep round-tripping.
 */
enum class DSSFMethod : std::uint32_t {
    /// ω-resolved dynamical response S(Q, ω) at one or more temperatures
    /// via FTLM continued fraction (Lanczos coefficients sampled over
    /// `num_random_states` random vectors).
    DYNAMICAL_THERMAL = 0,

    /// Thermal expectation values ⟨O(Q)⟩(T) -- no ω axis. Used for static
    /// structure factor S(Q, T) and other zero-frequency observables.
    STATIC_THERMAL = 1,

    /// T = 0 dynamical structure factor S(Q, ω) via continued fraction
    /// applied to the Lanczos-converged ground state. The optimal path for
    /// fixed-Sz 32-site ED on a single node.
    GROUND_STATE_DSSF = 2,

    /// Single expectation value ⟨ψ|O|ψ⟩ (no Hermitian conjugate / no
    /// product). Used by the legacy single-observable mode of TPQ_DSSF
    /// where only one operator is written instead of an O₁†O₂ pair.
    SINGLE_EXPECTATION = 3,
};

/// Convert a DSSFMethod to its lowercase string token (the same token that
/// the `ED dssf` CLI accepts via `--method=`).
std::string to_string(DSSFMethod method);

/// Inverse of `to_string`. Accepts canonical lowercase tokens
/// ("dynamical_thermal", "static_thermal", "ground_state_dssf",
/// "single_expectation") and throws `std::invalid_argument` otherwise.
DSSFMethod method_from_string(const std::string& token);

/**
 * One "run this DSSF computation" payload. Composed of an OperatorSpec
 * (which defines *what* observable to evaluate) plus a DSSFMethod (which
 * defines *how* to evaluate it), plus a reference to the canonical
 * `EDConfig` blob that already carries every method-specific knob the
 * `compute_*_workflow` functions consume.
 *
 * The `EDConfig` reference is intentionally kept around for one release:
 * it lets `ed::dssf::run(...)` delegate to the existing CLI workflows
 * verbatim while we incrementally move per-method parameters onto fields
 * of `DSSFRequest`. Once the migration is complete (post-P2.5),
 * `EDConfig` will be dropped from this struct in favour of typed
 * `DSSFRequest` fields.
 */
struct DSSFRequest {
    /// What observables to evaluate (already split into O₁ / O₂ pairs by
    /// `build_observable_pairs`).
    OperatorSpec operators;

    /// Which kernel to run.
    DSSFMethod method{DSSFMethod::DYNAMICAL_THERMAL};

    /// Where to persist results. Forwarded to the corresponding
    /// `compute_*_workflow(...)` call so the on-disk layout stays
    /// bit-identical with the legacy CLI for now.
    std::string output_dir;

    /// Backstop pointer to the legacy `EDConfig` blob. The transitional
    /// `ed::dssf::run(...)` implementation copies the per-method knobs
    /// (frequency window, broadening, temperature grid, krylov dim,
    /// random seed, ...) out of this config into the corresponding
    /// `compute_*_workflow` argument. Pass nullptr and we'll fall back
    /// to the defaults baked into `OperatorSpec` + this struct.
    const EDConfig* config{nullptr};
};

/**
 * Lightweight result envelope. Workflows persist their full numerical
 * output to HDF5 inside `run(...)` itself; this struct just reports back
 * a small, typed status for callers that want to verify success without
 * round-tripping through the file system.
 *
 * The detailed numerical fields (frequencies, S(Q, ω), ⟨O⟩(T), error
 * bars, ...) will be added in P2.3 when the unified `/dssf/...` HDF5
 * schema lands and the engine starts populating in-memory result tensors
 * directly (instead of only writing them to disk).
 */
struct DSSFResult {
    /// Method that was actually executed.
    DSSFMethod method{DSSFMethod::DYNAMICAL_THERMAL};

    /// Number of (operator pair × temperature × momentum) tasks the
    /// engine attempted. Useful as a quick "did the dispatch table even
    /// fire" smoke check.
    std::uint64_t num_tasks_attempted{0};

    /// Where the results were written (mirrors `request.output_dir`).
    std::string output_dir;
};

/**
 * Dispatch into the correct DSSF pipeline.
 *
 * For now (P2.2 / DSSF PR-C transitional cut) this function simply
 * delegates to the matching `compute_*_workflow(*request.config)` call:
 *
 *   DYNAMICAL_THERMAL  -> compute_dynamical_response_workflow
 *   STATIC_THERMAL     -> compute_static_response_workflow
 *   GROUND_STATE_DSSF  -> compute_ground_state_dssf_workflow
 *   SINGLE_EXPECTATION -> compute_static_response_workflow
 *                         (legacy TPQ_DSSF parity; will get its own
 *                          dedicated workflow in P2.3)
 *
 * Future PRs (P2.3 / P2.4) will move the workflow bodies onto this seam
 * so `EDConfig` no longer has to be passed in.
 *
 * @throws std::invalid_argument if `request.method` is unrecognised or
 *         if the matching workflow requires `request.config` and it is
 *         null.
 */
DSSFResult run(const DSSFRequest& request);

} // namespace ed::dssf

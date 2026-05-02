// =============================================================================
// include/ed/dssf/dssf_engine.h
//
// `ed::dssf::run(...)` -- the canonical, library-level entry point that
// wraps the entire DSSF / SSSF / single-expectation pipeline behind a
// single `enum class DSSFMethod` dispatch table.
//
// One seam, four kernels, three callers:
//
//   Kernels (live in `src/cli/workflows.cpp`):
//     * `compute_dynamical_response_workflow`  -- ω-resolved response
//     * `compute_static_response_workflow`     -- thermal expectation values
//     * `compute_ground_state_dssf_workflow`   -- T=0 continued fraction
//     * SINGLE_EXPECTATION                     -- single ⟨ψ|O|ψ⟩
//
//   Callers all funnel through `run(DSSFRequest)`:
//     * The C++ CLI: `ED dssf <method> ...` (src/apps/ed_main.cpp).
//     * The C++ CLI: legacy `--dynamical-response` / `--static-response` /
//       `--ground-state-dssf` flags (src/apps/ed_main.cpp).
//     * The Python bindings: `qed.dssf.run(...)` (P2.8).
//
// Historical context (archived; do not regress):
//   Before P2.14 the codebase carried *two* DSSF entry points: this seam
//   plus a 4 174-LOC standalone `TPQ_DSSF` binary with its own
//   14-positional-arg CLI and a duplicated `/dssf_results/...` HDF5 schema.
//   `TPQ_DSSF` was deleted in P2.14; `ED dssf <method>` is now the only
//   supported front end and the kernels are the only authoritative
//   implementation.
//
// Audit ref: P2.2 (DSSF PR-C) -- seam introduction; P2.4 (DSSF PR-E) --
// `ED dssf` subcommand; P2.14 (DSSF PR-H) -- `TPQ_DSSF` deletion.
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

    /// Single expectation value ⟨ψ|O|ψ⟩ (no Hermitian conjugate, no
    /// O₁†O₂ product): writes one operator per group rather than a pair.
    /// Useful for diagnostic observables (energy, magnetisation, etc.)
    /// where the second operator would be the identity.
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
 * The DSSFMethod -> kernel mapping:
 *
 *   DYNAMICAL_THERMAL  -> compute_dynamical_response_workflow
 *   STATIC_THERMAL     -> compute_static_response_workflow
 *   GROUND_STATE_DSSF  -> compute_ground_state_dssf_workflow
 *   SINGLE_EXPECTATION -> compute_static_response_workflow
 *                         (single-operator diagnostic mode; flagged via
 *                          OperatorSpec::single_obs_only on the request)
 *
 * The `compute_*_workflow` bodies live in `src/cli/workflows.cpp` and
 * read per-method parameters off `request.config`. New methods just need
 * an entry in this dispatch table plus a kernel.
 *
 * @throws std::invalid_argument if `request.method` is unrecognised or
 *         if the matching workflow requires `request.config` and it is
 *         null.
 */
DSSFResult run(const DSSFRequest& request);

} // namespace ed::dssf

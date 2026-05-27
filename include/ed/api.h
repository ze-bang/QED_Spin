#pragma once
// =============================================================================
// include/ed/api.h
//
// `ed::api::*` -- Python-named, kwargs-style facade over the existing
// `ed::workflows::solve / thermal / spectral` orchestrator. Lets C++
// callers write code that reads line-for-line like the Python equivalents:
//
//     auto res = ed::api::solve(spec, {
//         .num_eigenvalues = 5,
//         .solver          = "LANCZOS",
//         .device          = "auto",
//     });
//
// matching `qed.solve(H, num_eigenvalues=5, solver="LANCZOS", device="auto")`.
//
// The facade is a *pure* translation layer:
//
//   * Field names match Python's `qed.solve` / `qed.thermal` / `qed.spectral`
//     kwargs.
//   * Defaults match Python's (T_min=0.1, num_T=24, num_samples=40,
//     taylor_order=8, kpm_num_moments=200, ...).
//   * Method enums are accepted as case-insensitive strings
//     ("LANCZOS" / "lanczos" / "Lanczos" / "FTLM" / "ftlm" / ...).
//   * `device=` accepts "auto" / "cpu" / "gpu" / "mpi" / "mpi_gpu" and
//     converts to `BackendConstraints` via the same threshold heuristic
//     Python uses (`global_dim >= 1 << 14` for "auto" => GPU).
//
// The actual kernel work is delegated to `ed::workflows::solve / thermal /
// spectral`, which keeps its C++-style option structs untouched (no ABI
// risk to legacy callers). The byte-equality pinning test
// `tests/unit/test_api_mirror.cpp` verifies that a representative cell
// from each method returns identical numbers via both surfaces.
//
// Author: ed-collapse, Phase A of the "mirror examples" plan (May 2026).
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <ed/core/make_operator.h>
#include <ed/core/results.h>
#include <ed/core/select_backend.h>
#include <ed/orchestrator.h>  // legacy workflow option structs we translate to

namespace ed::api {

// ---------------------------------------------------------------------------
// String parsers for the three method enums.
//
// Accepts both Python tokens (`"LANCZOS"`, `"FTLM"`, `"KPM_DOS"`,
// `"ground_state_cf"`, `"ftlm_dynamical"`) and C++ enum spellings
// (`"Lanczos"`, `"KpmDos"`, `"GroundStateCF"`). Returns `std::nullopt`
// when the token is unrecognised.
// ---------------------------------------------------------------------------

[[nodiscard]] std::optional<ed::workflows::SolveMethod>
parse_solve_method(std::string_view name);

[[nodiscard]] std::optional<ed::workflows::ThermalOptions::Method>
parse_thermal_method(std::string_view name);

[[nodiscard]] std::optional<ed::workflows::SpectralOptions::Method>
parse_spectral_method(std::string_view name);

// ---------------------------------------------------------------------------
// `device_constraints(device_str, dim_hint)` -- map a Python-style
// `device=` string to a `BackendConstraints`. Threshold for "auto"
// matches the Python facade (`global_dim >= 1 << 14` => GPU preferred).
// ---------------------------------------------------------------------------
[[nodiscard]] ed::BackendConstraints
device_constraints(std::string_view device, std::uint64_t dim_hint = 0);

// ---------------------------------------------------------------------------
// Build-time introspection. Mirrors Python's `qed.has_cuda_build()`,
// `qed.has_mpi_build()`, `qed.has_nccl_build()` -- which let example
// scripts and tests gate device-specific paths without hard-coding the
// build flags. Implementations live in src/api/build_introspection.cpp.
// ---------------------------------------------------------------------------

}  // namespace ed::api

namespace ed {
[[nodiscard]] bool has_cuda_build() noexcept;
[[nodiscard]] bool has_mpi_build()  noexcept;
[[nodiscard]] bool has_nccl_build() noexcept;
}  // namespace ed

namespace ed::api {

// ===========================================================================
// SolveOptions -- Python-named mirror of qed.solve()'s kwargs.
//
// Designated-initializer aggregate: callers write
//
//     ed::api::SolveOptions{ .num_eigenvalues = 5, .device = "auto" }
//
// every field optional. Defaults match `python/qed/workflow.py::solve`.
// ===========================================================================
struct SolveOptions {
    std::size_t num_eigenvalues       = 1;
    double      tolerance             = 1e-10;
    bool        compute_eigenvectors  = false;
    /// Method name. Empty / "auto" => let the orchestrator pick. Accepts
    /// "LANCZOS", "BLOCK_LANCZOS", "KRYLOV_SCHUR", "FULL"
    /// (case-insensitive). For thermal methods ("FTLM", "mTPQ", ...) use
    /// `ed::api::thermal` instead.
    std::string solver                = "";
    /// Device hint. "" / "auto" / "cpu" / "gpu" / "mpi" / "mpi_gpu".
    std::string device                = "";
    /// If set, project to a single Sz sector (n_up = number of up spins).
    std::optional<int> sz             = std::nullopt;
    /// HDF5 output directory (empty disables disk writes).
    std::string output_dir            = "";
    /// `std::nullopt` => auto-tuned by the orchestrator.
    std::optional<std::size_t> max_iterations = std::nullopt;
    /// Block size for `BLOCK_LANCZOS`. `std::nullopt` => 4.
    std::optional<std::size_t> block_size     = std::nullopt;
    /// Streaming-symmetry sector filter (empty = walk every non-empty).
    std::vector<std::size_t> selected_sectors;
};

// ===========================================================================
// ThermalOptions -- Python-named mirror of qed.thermal()'s kwargs.
// Defaults match `python/qed/thermal.py::thermal`.
// ===========================================================================
struct ThermalOptions {
    std::string method                = "FTLM";
    double      T_min                 = 0.1;
    double      T_max                 = 10.0;
    std::size_t num_T                 = 24;
    std::optional<int> sz_min         = std::nullopt;
    std::optional<int> sz_max         = std::nullopt;
    std::size_t num_samples           = 40;
    std::optional<std::size_t> krylov_dim = std::nullopt;
    double      tolerance             = 1e-10;
    std::size_t max_iterations        = 1000;
    std::uint64_t random_seed         = 0;
    bool        use_symmetry_if_available = false;
    bool        use_sz_if_conserved   = true;
    std::string output_dir            = "";
    std::string device                = "";
    /// KPM_DOS knobs (Python defaults).
    int         kpm_num_moments       = 200;
    int         kpm_num_random_vectors = 16;
    /// mTPQ / cTPQ knobs (Python defaults).
    std::size_t tpq_num_measure_points = 100;
    std::optional<double> tpq_measure_beta_min = std::nullopt;
    std::optional<double> tpq_measure_beta_max = std::nullopt;
    double      tpq_delta_beta        = 0.05;
    std::size_t tpq_taylor_order      = 8;
    std::size_t tpq_measurement_interval = 1;
    double      tpq_energy_shift      = 0.0;
    /// Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
    /// probe-beta list for mTPQ/cTPQ state-vector snapshots. Empty =
    /// trajectory-only persistence. Plumbed straight through to
    /// ``ed::workflows::ThermalOptions::probe_betas``.
    std::vector<double> probe_betas;
    /// Streaming-symmetry sector filter (empty = walk every non-empty).
    std::vector<std::size_t> selected_sectors;
};

// ===========================================================================
// SpectralOptions -- Python-named mirror of qed.spectral()'s kwargs.
// Defaults match `python/qed/spectral.py::spectral` where Python has
// explicit defaults; otherwise picks the legacy C++ defaults.
// ===========================================================================
struct SpectralOptions {
    /// "" / "auto" => pick from (T, omega): if T is set => FtlmDynamical,
    /// else GroundStateCF.
    std::string method                = "";
    /// Optional single temperature (mirrors Python's float-or-iterable).
    std::optional<double> T           = std::nullopt;
    /// Explicit temperature list (overrides `T` when non-empty).
    std::vector<double> temperatures;
    /// Explicit omega grid (overrides `omega_min`/`omega_max`/`num_omega`
    /// when non-empty).
    std::vector<double> omega;
    double      omega_min             = -10.0;
    double      omega_max             =  10.0;
    std::size_t num_omega             = 200;
    /// Lorentzian broadening (Python: `eta`).
    double      eta                   = 0.05;
    std::optional<std::size_t> krylov_dim = std::nullopt;
    /// FTLM random-vector count. `std::nullopt` => 30 (legacy C++ default).
    std::optional<std::size_t> num_random_vectors = std::nullopt;
    double      energy_shift          = 0.0;
    std::string output_dir            = "";
    std::string observable_type       = "";
    std::string device                = "";
    /// If set, project to a single Sz sector.
    std::optional<int> sz             = std::nullopt;
    /// SOTA streaming-symmetry knobs.
    std::vector<double> momentum_transfer;
    double      momentum_tolerance    = 1e-6;
    std::vector<std::size_t> selected_sectors;

    /// Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026):
    /// caller-supplied seed state for the GroundStateCF lane. Empty
    /// (default) -> the orchestrator computes the GS via an inner
    /// Lanczos solve. Length MUST match ``H.geometry().local_dim``.
    /// Mirror of ``ed::workflows::SpectralOptions::initial_state``.
    std::vector<std::complex<double>> initial_state;

    /// Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026):
    /// KPM-dynamical knobs (used only by ``method == "kpm_dynamical"``;
    /// ignored by the GroundStateCF / FtlmDynamical lanes).
    std::size_t kpm_moments           = 200;
    /// Chebyshev kernel for KPM-dynamical: "Jackson" (default) or
    /// "Lorentz".
    std::string kpm_kernel            = "Jackson";
    /// Lorentz kernel decay parameter (only used when
    /// ``kpm_kernel == "Lorentz"``).
    double      kpm_lorentz_lambda    = 4.0;
    /// Optional override for the (E_min, E_max) Chebyshev rescaling.
    /// Unset (default) -> let the kernel discover them via Lanczos.
    std::optional<std::pair<double, double>> kpm_spectral_bounds;
};

// ---------------------------------------------------------------------------
// Translation helpers. Public so callers (CLI, tests) can drive the
// legacy `ed::workflows::*` orchestrator directly with a Python-named
// options struct.
// ---------------------------------------------------------------------------

[[nodiscard]] ed::workflows::SolveOptions
to_legacy(const SolveOptions& opts, std::uint64_t dim_hint = 0);

[[nodiscard]] ed::workflows::ThermalOptions
to_legacy(const ThermalOptions& opts, std::uint64_t dim_hint = 0);

[[nodiscard]] ed::workflows::SpectralOptions
to_legacy(const SpectralOptions& opts, std::uint64_t dim_hint = 0);

// ===========================================================================
// Verb facades.
//
// Overload set for each verb:
//   * Spec form: takes an OperatorSpec, builds the operator via
//     `ed::make_operator`, and routes to `ed::workflows::solve / thermal /
//     spectral`. Mirrors Python's `qed.solve(H, ...)` when `H` is built
//     via `qed.input.HamiltonianBuilder` or `qed.Operator`.
//   * Operator form: takes a pre-built `LinearOperator&`. For callers who
//     hold an `Operator` / `FixedSzOperator` / `StreamingSymmetryOperator`
//     directly and don't want to round-trip through `OperatorSpec`.
// ===========================================================================

// Note: the OperatorSpec form takes `spec` by value because OperatorSpec
// is move-only (its `InMemoryOperator` alternative carries a
// `std::unique_ptr<Operator>`). Callers should `std::move(spec)` when
// they no longer need the spec object, or construct it inline at the
// call site (which is the common Python-mirror idiom). Specs whose
// active source is `FilePaths` / `DirectoryPath` are cheap to construct.
//
// LinearOperator-form overloads live in the .cpp (one translation unit
// is enough: no `make_operator` instantiation, so they don't drag in
// the `WITH_MPI` distributed constructors). The OperatorSpec-form
// overloads are `inline` in the header to keep the back-edge link
// dependency at the consumer (every call site that wants
// `ed::api::solve(spec, ...)` already needs to link `ed_distributed`
// when MPI is on, so this just moves the demand from the api library
// to the consumer, which `ed_add_test` and the examples register
// already wire up).

[[nodiscard]] ed::GroundStateResult
solve(const ed::LinearOperator& H, SolveOptions opts = {});

[[nodiscard]] ed::ThermalResult
thermal(const ed::LinearOperator& H, ThermalOptions opts = {});

[[nodiscard]] ed::SpectralResult
spectral(const ed::LinearOperator& H,
         const std::vector<const ed::LinearOperator*>& observables,
         SpectralOptions opts = {});

[[nodiscard]] inline ed::GroundStateResult
solve(ed::OperatorSpec spec, SolveOptions opts = {}) {
    if (opts.sz.has_value() && !spec.fixed_sz.has_value()) {
        spec.fixed_sz = *opts.sz;
    }
    auto op = ed::make_operator(std::move(spec));
    auto wf_opts = to_legacy(opts, op ? op->geometry().global_dim : 0);
    return ed::workflows::solve(*op, std::move(wf_opts));
}

[[nodiscard]] inline ed::ThermalResult
thermal(ed::OperatorSpec spec, ThermalOptions opts = {}) {
    auto op = ed::make_operator(std::move(spec));
    auto wf_opts = to_legacy(opts, op ? op->geometry().global_dim : 0);
    return ed::workflows::thermal(*op, std::move(wf_opts));
}

[[nodiscard]] inline ed::SpectralResult
spectral(ed::OperatorSpec spec,
         const std::vector<const ed::LinearOperator*>& observables,
         SpectralOptions opts = {}) {
    auto op = ed::make_operator(std::move(spec));
    auto wf_opts = to_legacy(opts, op ? op->geometry().global_dim : 0);
    return ed::workflows::spectral(*op, observables, std::move(wf_opts));
}

}  // namespace ed::api

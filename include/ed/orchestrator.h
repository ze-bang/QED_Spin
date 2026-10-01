#pragma once
// =============================================================================
// include/ed/orchestrator.h
//
// Two top-level entry points over any `ed::LinearOperator`:
//
//     ed::solve     -- ground-state eigenproblem (Lanczos / Krylov-Schur /
//                       full diag) on the Backend lane select_backend
//                       picks (CPU or CUDA).
//     ed::workflows::thermal -- finite-temperature workflows (FTLM /
//                       OFTLM / mTPQ).
//
// Both:
//
//   * accept any `ed::LinearOperator` (no separate CPU / GPU entry point);
//   * dispatch under `std::visit(select_backend(...))` so the kernel
//     family runs on the right Backend;
//   * return `GroundStateResult` / `ThermalResult` with
//     backend metadata + Krylov diagnostics carried uniformly across lanes.
//
// The ed::sectors verbs (include/ed/sectors/) run these on each symmetry
// block. Implementation: src/orchestrator/ (see orchestrator_internal.h).
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ed/core/linear_operator.h>
#include <ed/core/results.h>
#include <ed/core/select_backend.h>

// `ed::thermal` and `ed::observables` are kernel namespaces. To avoid a
// function/namespace collision, the orchestrator entry points live under
// `ed::workflows::`. The public-facing convention is:
//
//     auto gs = ed::workflows::solve(H, opts);
//     auto th = ed::workflows::thermal(H, opts);
//
// `ed::solve` is also exported as a top-level alias (it does not clash
// with any namespace) for ergonomics.
namespace ed::workflows {

enum class SolveMethod : std::uint8_t {
    Auto = 0,          ///< pick from dim + num_eigs
    Lanczos = 1,
    KrylovSchur = 3,
    FullDiag = 4,
};

struct SolveOptions {
    std::size_t num_eigs       = 1;
    std::size_t max_iter       = 0;        ///< 0 => auto
    double      tolerance      = 1e-10;
    bool        compute_vectors = false;
    SolveMethod method         = SolveMethod::Auto;
    BackendConstraints backend;
};

struct ThermalOptions {
    /// Host-side transform applied to every sample seed the kernels
    /// draw (see FtlmOptions/MtpqOptions::seed_transform), e.g. the
    /// Lowdin projection onto one spin tower installed by the sector
    /// loops; the stochastic trace then runs over that subspace.
    std::function<void(Complex*, std::size_t)> seed_transform;

    /// Static observables for FTLM (Method::FTLM only): <O>(T) is returned in
    /// ThermalResult::observables. Each is bound to the backend the run selects.
    std::vector<std::shared_ptr<const LinearOperator>> observables;

    /// Method discriminator.
    enum class Method : std::uint8_t {
        FTLM = 0, mTPQ = 2, OFTLM = 5,
    } method = Method::FTLM;

    // ---------------------------------------------------------------
    // Sampling defaults.
    // ---------------------------------------------------------------
    std::size_t num_samples    = 40;
    std::size_t krylov_dim     = 100;
    std::size_t num_exact      = 8;    ///< OFTLM: # low-lying states treated exactly (N_V).
    /// Blocks up to this dimension are diagonalised exactly instead of sampled (a
    /// sampled trace needs D >> samples); 0 always samples.
    std::uint64_t dense_max_dim = 512;
    std::vector<double> betas;
    std::uint64_t random_seed  = 0;
    BackendConstraints backend;

    /// Temperature grid used only when `betas` is empty: num_temp_bins
    /// points linear in T on [temp_min, temp_max].
    double      temp_min       = 0.1;
    double      temp_max       = 10.0;
    std::size_t num_temp_bins  = 24;

    // -----------------------------------------------------------------
    // Caller-supplied spectral bounds for the mTPQ auto-tune. When BOTH
    // ``e_min_override`` and ``e_max_override`` are finite (and ordered),
    // the mTPQ lane skips its own Lanczos-based spectral estimation and
    // uses the supplied window to place the shift L. NaN means
    // "estimate per call".
    // -----------------------------------------------------------------
    double      e_min_override = std::numeric_limits<double>::quiet_NaN();
    double      e_max_override = std::numeric_limits<double>::quiet_NaN();

    // -----------------------------------------------------------------
    // mTPQ expert override. The microcanonical iteration
    // |psi_{k+1}> = (L*I - H)|psi_k> uses a "large value" L that the
    // orchestrator auto-tunes from a short Lanczos spectral-bound
    // estimate (see the mTPQ branch in src/orchestrator/orch_thermal.cpp).
    // A finite,
    // POSITIVE value here pins L directly and skips the auto-tune,
    // mirroring the HPhi ``LargeValue`` knob. ``0.0`` (default) means
    // "auto". Ignored by every non-mTPQ lane.
    // -----------------------------------------------------------------
    double      energy_shift   = 0.0;
};

// ---------------------------------------------------------------------------
// Public entry points.
//
// `ed::solve`     -- ground-state eigenproblem. Picks Lanczos /
//                    Krylov-Schur / full-diag based on
//                    (num_eigs, geometry().global_dim, opts.method).
// `thermal`       -- finite-T workflow. Switches on `opts.method`.
//
// Both orchestrators construct the appropriate Backend internally
// via `select_backend(H.geometry(), opts.backend)` and return a uniform
// Result struct carrying the BackendMetadata that fired.
// ---------------------------------------------------------------------------

GroundStateResult solve(const LinearOperator&  H,
                         SolveOptions           opts = {});

ThermalResult     thermal(const LinearOperator& H,
                           ThermalOptions        opts = {});

}  // namespace ed::workflows

namespace ed {
// Top-level alias for the most-used entry point. `thermal` does not get an
// alias (it would shadow the kernel namespace).
using ed::workflows::solve;

// Option structs and result enums get top-level aliases so callers can
// write `ed::SolveOptions` / `ed::ThermalOptions`
// without reaching into `ed::workflows::`.
using ed::workflows::SolveOptions;
using ed::workflows::ThermalOptions;
using ed::workflows::SolveMethod;
}  // namespace ed

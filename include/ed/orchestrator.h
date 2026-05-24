#pragma once
// =============================================================================
// include/ed/orchestrator.h
//
// Three top-level entry points the public-facing API now collapses to:
//
//     ed::solve     -- ground-state eigenproblem (Lanczos / Krylov-Schur /
//                       Block-Lanczos / full diag), one of the four
//                       Backend lanes auto-selected via select_backend.
//     ed::thermal   -- finite-temperature workflows (FTLM / LTLM / mTPQ /
//                       cTPQ / KPM-DOS).
//     ed::spectral  -- dynamical correlators (DSSF ground state /
//                       finite-T) via continued-fraction Lanczos.
//
// Replaces the legacy `auto/solve.cpp`, `auto/thermal.cpp`,
// `auto/dssf.cpp` orchestrators and the per-deployment `ed_wrapper*`
// stack. The new entries:
//
//   * accept any `ed::LinearOperator` (no separate CPU / GPU / MPI / etc
//     entry point);
//   * dispatch under `std::visit(select_backend(...))` so the kernel
//     family runs on the right Backend;
//   * return `GroundStateResult` / `ThermalResult` / `SpectralResult`
//     (Phase 3.3) with backend metadata + Krylov diagnostics carried
//     uniformly across lanes.
//
// Phase 4.2 of the Minimalist ED Collapse (May 2026).
//
// Implementation note: the full kernel/method-selection logic in
// `ed::solve` is non-trivial (Krylov-Schur for many-eigs, single-vector
// Lanczos for one-eig, full diag for small dims). The minimal first
// landing keeps that decision tree centralised in `src/orchestrator.cpp`
// and uses ONLY the unified `lanczos_kernel<Backend>` (single-vector)
// + `block_lanczos_kernel<Backend>` (multi-eig) + `krylov_schur_kernel<Backend>`
// codepaths from Phase 2 --- which is everything every consumer in the
// project actually exercises today.
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <ed/core/linear_operator.h>
#include <ed/core/results.h>
#include <ed/core/select_backend.h>

// `ed::thermal` and `ed::observables` already exist as namespaces (the
// per-kernel headers from Phase 2). To avoid a function/namespace
// collision, the three orchestrator entry points live under
// `ed::workflows::`. The public-facing convention is:
//
//     auto gs = ed::workflows::solve(H, opts);
//     auto th = ed::workflows::thermal(H, opts);
//     auto sp = ed::workflows::spectral(H, observables, opts);
//
// `ed::solve` is also exported as a top-level alias (since `ed::solve`
// does not clash with any existing namespace) for ergonomics.
namespace ed::workflows {

enum class SolveMethod : std::uint8_t {
    Auto = 0,          ///< pick from dim + num_eigs
    Lanczos = 1,
    BlockLanczos = 2,
    KrylovSchur = 3,
    FullDiag = 4,
};

struct SolveOptions {
    std::size_t num_eigs       = 1;
    std::size_t max_iter       = 0;        ///< 0 => auto
    std::size_t block_size     = 4;
    double      tolerance      = 1e-10;
    bool        compute_vectors = false;
    /// Output directory for HDF5 results (legacy behaviour). Empty
    /// means "do not write".
    std::string output_dir;
    SolveMethod method         = SolveMethod::Auto;
    BackendConstraints backend;

    // -----------------------------------------------------------------
    // CLI parity knobs (Wave A5 -- Full unified-interface collapse,
    // May 2026). These extend SolveOptions so the CLI migration
    // (Wave C) can pure-refactor without losing any of the orthogonal
    // axes the legacy `EDParameters` carried.
    // -----------------------------------------------------------------

    /// If true, the operator construction should project to a single
    /// Sz sector. The actual sector is picked by `n_up` (or, when
    /// `n_up < 0`, the half-filled sector). This duplicates the
    /// `OperatorSpec::fixed_sz` axis but is carried on the solve
    /// options too so the CLI can flip the axis on without touching
    /// the operator factory call site.
    bool        use_fixed_sz   = false;

    /// If true, the operator construction should use the
    /// streaming-symmetry path. Same redundancy story as
    /// `use_fixed_sz` above.
    bool        use_symmetry   = false;

    /// Number of "up" spins for the fixed-Sz sector. -1 means
    /// "half-filled" (= num_sites / 2). Only consulted when
    /// `use_fixed_sz` is true.
    int         n_up           = -1;

    /// Directory for caching the streaming-symmetry basis between
    /// runs. Empty means "no caching" (recompute every run). The
    /// streaming-symmetry kernel writes `automorphism_results/` and
    /// `basis_cache/` HDF5 files here.
    std::string basis_cache_dir;

    /// If true, generate the symmetry basis + the sector-block
    /// Hamiltonians and exit without diagonalising. Used by the CLI's
    /// `precompute_basis_only` mode to pre-warm the cache on a single
    /// node before launching the diagonalisation step on a cluster.
    bool        precompute_basis_only = false;
};

struct ThermalOptions {
    /// Method discriminator (matches the legacy auto/thermal lane tags).
    enum class Method : std::uint8_t {
        FTLM = 0, LTLM, mTPQ, cTPQ, KpmDos,
    } method = Method::FTLM;
    std::size_t num_samples    = 30;
    std::size_t krylov_dim     = 100;
    std::size_t taylor_order   = 50;
    std::vector<double> betas;
    double      delta_beta     = 0.1;
    double      beta_max       = 1000.0;
    std::uint64_t random_seed  = 0;
    std::string output_dir;
    BackendConstraints backend;

    // -----------------------------------------------------------------
    // CLI parity knobs (Wave A5). Match the temperature-scan
    // controls the legacy `EDParameters` / CLI thermo section carry.
    // -----------------------------------------------------------------

    /// Temperature scan range (used by both the thermodynamics
    /// post-processing and the KPM-DOS lane). Linear in T by
    /// default; the CLI may convert to inverse-temperature betas if
    /// needed.
    double      temp_min       = 0.01;
    double      temp_max       = 10.0;
    std::size_t num_temp_bins  = 100;

    /// Lorentzian broadening eta for the KPM-DOS density lane.
    double      broadening     = 0.05;
};

struct SpectralOptions {
    enum class Method : std::uint8_t {
        GroundStateCF = 0,   ///< compute_ground_state_dssf via cf_spectral_kernel
        FtlmDynamical = 1,
    } method = Method::GroundStateCF;
    std::size_t krylov_dim    = 200;
    double      broadening    = 0.05;
    double      omega_min     = -10.0;
    double      omega_max     =  10.0;
    std::size_t num_omega     = 200;
    /// Optional ground-state energy shift (used to align the CF
    /// resolvent). 0 means "auto-detect from the tridiag".
    double      energy_shift  = 0.0;
    std::string output_dir;
    BackendConstraints backend;

    // -----------------------------------------------------------------
    // CLI parity knobs (Wave A5). FTLM-dynamical needs sample-count +
    // temperature scan controls; the static-response lane carries an
    // observable-type discriminator used by the legacy CLI.
    // -----------------------------------------------------------------

    /// Number of random samples for FtlmDynamical averaging. Ignored
    /// by GroundStateCF (which uses the ground-state vector).
    std::size_t num_samples   = 30;

    /// Temperature scan for FtlmDynamical; empty means "T = 0 only".
    std::vector<double> temperatures;

    /// Observable-type label carried for HDF5 output / Python
    /// roundtripping (e.g., "Sz", "Sx_Sx", "Sz_Sz"). Optional --
    /// orchestrator does not consume it; the CLI uses it for naming.
    std::string observable_type;
};

// ---------------------------------------------------------------------------
// Public entry points.
//
// `ed::solve`     -- ground-state eigenproblem. Picks Lanczos /
//                    Block-Lanczos / Krylov-Schur / full-diag based on
//                    (num_eigs, geometry().global_dim, opts.method).
// `ed::thermal`   -- finite-T workflow. Switches on `opts.method`.
// `ed::spectral`  -- dynamical correlators. `observables` is a span of
//                    operator pointers (typically size 1: the
//                    spin/charge operator).
//
// All three orchestrators construct the appropriate Backend internally
// via `select_backend(H.geometry(), opts.backend)` and return a uniform
// Result struct (Phase 3.3) carrying the BackendMetadata that fired.
// ---------------------------------------------------------------------------

GroundStateResult solve(const LinearOperator&  H,
                         SolveOptions           opts = {});

ThermalResult     thermal(const LinearOperator& H,
                           ThermalOptions        opts = {});

SpectralResult    spectral(const LinearOperator&                          H,
                            const std::vector<const LinearOperator*>&     observables,
                            SpectralOptions                               opts = {});

}  // namespace ed::workflows

namespace ed {
// Top-level alias for the most-used entry point. `thermal` and `spectral`
// don't get aliases (they would shadow the kernel namespaces).
using ed::workflows::solve;

// Option structs and result enums get top-level aliases so callers can
// write `ed::SolveOptions` / `ed::ThermalOptions` / `ed::SpectralOptions`
// without reaching into `ed::workflows::`.
using ed::workflows::SolveOptions;
using ed::workflows::ThermalOptions;
using ed::workflows::SpectralOptions;
using ed::workflows::SolveMethod;
}  // namespace ed

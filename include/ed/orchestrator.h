#pragma once
// =============================================================================
// include/ed/orchestrator.h
//
// Two top-level entry points over any `ed::LinearOperator`:
//
//     ed::solve     -- ground-state eigenproblem (Lanczos / Krylov-Schur /
//                       Block-Lanczos / full diag), one of the four
//                       Backend lanes auto-selected via select_backend.
//     ed::thermal   -- finite-temperature workflows (FTLM / OFTLM /
//                       mTPQ).
//
// Both:
//
//   * accept any `ed::LinearOperator` (no separate CPU / GPU / MPI / etc
//     entry point);
//   * dispatch under `std::visit(select_backend(...))` so the kernel
//     family runs on the right Backend;
//   * return `GroundStateResult` / `ThermalResult` (Phase 3.3) with
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

// `ed::thermal` and `ed::observables` already exist as namespaces (the
// per-kernel headers from Phase 2). To avoid a function/namespace
// collision, the orchestrator entry points live under
// `ed::workflows::`. The public-facing convention is:
//
//     auto gs = ed::workflows::solve(H, opts);
//     auto th = ed::workflows::thermal(H, opts);
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
    BlockKrylovSchur = 5,   ///< thick-restart block Lanczos (degenerate/clustered)
};

struct SolveOptions {
    std::size_t num_eigs       = 1;
    std::size_t max_iter       = 0;        ///< 0 => auto
    std::size_t block_size     = 4;
    double      tolerance      = 1e-10;
    bool        compute_vectors = false;
    /// Block Lanczos: keep the full block-Krylov basis (full reorth + eigenvectors)
    /// vs lean local-reorth (eigenvalues-only, bounded memory). The planner sets
    /// this false for large-N eigenvalue runs whose full basis exceeds the budget;
    /// env ED_BLOCK_LANCZOS_LEAN=1 forces lean. Ignored when compute_vectors=true.
    bool        block_lanczos_keep_basis = true;
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

    /// Stage 8 (SymmetryEngine v2) composition toggles, per call:
    /// -1 = auto (commutation check + env gate), 0 = off,
    ///  1 = require (throw when H does not carry the symmetry).
    /// Consumed by the streaming-symmetry sector loops (spin-flip
    /// transport/projection across Sz blocks; time-reversal k <-> -k
    /// sector pairing).
    int spin_flip     = -1;
    int time_reversal = -1;

    /// Stage 7a: raw-irrep image maps under the non-abelian residue of
    /// the spatial group (one vector per coset representative;
    /// star_maps[c][k] = image irrep of k under p_c, -1 = unresolved).
    /// Isospectral orbit members are solved once and copied. Computed
    /// by the Python automorphism pipeline; empty = no star reduction.
    std::vector<std::vector<int>> star_maps;

    /// Sz-parity sectors (diagonal Z2 remnant): -1 = off/auto-detect
    /// upstream, 0 = even half, 1 = odd half, 2 = both halves.
    int sz_parity = -1;

    /// If true, generate the symmetry basis + the sector-block
    /// Hamiltonians and exit without diagonalising. Used by the CLI's
    /// `precompute_basis_only` mode to pre-warm the cache on a single
    /// node before launching the diagonalisation step on a cluster.
    bool        precompute_basis_only = false;

    /// Filter for the streaming-symmetry sector loop. When non-empty,
    /// only sectors whose linear index appears in this list are
    /// solved. Empty means "iterate over every non-empty sector"
    /// (the legacy / default behaviour). Used to probe a single
    /// irrep without paying for the rest of the spectrum.
    std::vector<std::size_t> selected_sectors;

    // -----------------------------------------------------------------
    // Stage 12 (SU(2) rollout): total-spin axis.
    // -----------------------------------------------------------------

    /// Target the spin-S tower (two_total_spin = 2S; -1 = off). Requires
    /// an SU(2)-invariant Hamiltonian (term-level check throws otherwise).
    /// The sector loops wrap each block in the Lowdin
    /// ``CasimirProjectedOperator`` and project the Krylov seed; blocks
    /// with no weight in the tower are skipped.
    int two_total_spin   = -1;

    /// Post-hoc <S^2> labeling toggle: -1 = auto (label whenever H is
    /// SU(2)-invariant AND eigenvectors are available in-sector), 0 =
    /// off, 1 = require (throw when H is not SU(2)-invariant). Fills
    /// ``GroundStateResult::{s2_of_eigenvalue, two_S_of_eigenvalue}``.
    int label_total_spin = -1;

    /// Host-side transform applied to the randomly drawn Krylov seed
    /// before it is staged into the backend (e.g. the Lowdin total-spin
    /// projection). A non-null transform disables the ``lanczos_real``
    /// fast lane, which draws its own seed internally.
    std::function<void(Complex*, std::size_t)> seed_transform;
};

struct ThermalOptions {
    /// Stage 8 (SymmetryEngine v2) composition toggles, per call:
    /// -1 = auto, 0 = off, 1 = require. See SolveOptions for semantics.
    int spin_flip     = -1;
    int time_reversal = -1;

    /// Stage 7a: raw-irrep image maps under the non-abelian residue of
    /// the spatial group (one vector per coset representative;
    /// star_maps[c][k] = image irrep of k under p_c, -1 = unresolved).
    /// Isospectral orbit members are solved once and copied. Computed
    /// by the Python automorphism pipeline; empty = no star reduction.
    std::vector<std::vector<int>> star_maps;

    /// Sz-parity sectors (diagonal Z2 remnant): -1 = off/auto-detect
    /// upstream, 0 = even half, 1 = odd half, 2 = both halves.
    int sz_parity = -1;

    // -----------------------------------------------------------------
    // Stage 12f (SU(2) rollout): per-tower thermal sampling.
    // -----------------------------------------------------------------

    /// Restrict the stochastic trace to the spin-S tower (2S; -1 = off).
    /// The sector loops wrap each block matvec in the Lowdin projector
    /// and install `seed_transform` per sector; the caller recombines
    /// the per-tower results with (2S+1) degeneracy weights via the
    /// degeneracy overload of ``combine_sector_thermodynamics``.
    /// Consumed by the FTLM and mTPQ kernels (their seeds carry the
    /// projection); other methods ignore it.
    int two_total_spin = -1;

    /// Host-side transform applied to every sample seed the kernels
    /// draw (see FtlmOptions/MtpqOptions::seed_transform). Installed by
    /// the sector loops when `two_total_spin >= 0`; direct callers may
    /// also set it for custom subspace-restricted sampling.
    std::function<void(Complex*, std::size_t)> seed_transform;

    /// Method discriminator (matches the legacy auto/thermal lane tags).
    enum class Method : std::uint8_t {
        FTLM = 0, mTPQ = 2, OFTLM = 5,
    } method = Method::FTLM;

    // ---------------------------------------------------------------
    // Sampling defaults.
    // ---------------------------------------------------------------
    std::size_t num_samples    = 40;
    std::size_t krylov_dim     = 100;
    std::size_t num_exact      = 8;    ///< OFTLM: # low-lying states treated exactly (N_V).
    std::vector<double> betas;
    std::uint64_t random_seed  = 0;
    BackendConstraints backend;

    // -----------------------------------------------------------------
    // CLI parity knobs (Wave A5). Match the temperature-scan
    // controls the legacy `EDParameters` / CLI thermo section carry.
    // -----------------------------------------------------------------

    /// Temperature scan range (used by the thermodynamics
    /// post-processing). Linear in T by default; the CLI may convert
    /// to inverse-temperature betas if needed.
    double      temp_min       = 0.1;  ///< Python `T_min` default (was 0.01).
    double      temp_max       = 10.0;
    std::size_t num_temp_bins  = 24;   ///< Python `num_T` default (was 100).

    /// Filter for the streaming-symmetry sector loop. See
    /// ``SolveOptions::selected_sectors``. Empty => walk every
    /// non-empty sector.
    std::vector<std::size_t> selected_sectors;

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
    // mTPQ expert override (June 2026). The microcanonical iteration
    // |psi_{k+1}> = (L*I - H)|psi_k> uses a "large value" L that the
    // orchestrator auto-tunes from a short Lanczos spectral-bound
    // estimate (see the mTPQ branch in orchestrator.cpp). A finite,
    // POSITIVE value here pins L directly and skips the auto-tune,
    // mirroring the HPhi ``LargeValue`` knob. ``0.0`` (default) means
    // "auto". Ignored by every non-mTPQ lane.
    // -----------------------------------------------------------------
    double      energy_shift   = 0.0;

    // -----------------------------------------------------------------
    // fp32 single-GPU mTPQ (memory-halving lane, July 2026). When true AND
    // the operator advertises ``supports_cuda_f32()`` (full-Hilbert Operator
    // on a WITH_CUDA build) AND the method is mTPQ, the orchestrator routes
    // to ``ed::thermal::mtpq_f32``: state vectors + matvec in complex<float>
    // (half the footprint, so the full 2^32 Hilbert space fits two vectors on
    // one 80 GB H100), reductions accumulated in double. Ignored otherwise.
    // -----------------------------------------------------------------
    bool        mtpq_fp32      = false;

    // -----------------------------------------------------------------
    // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
    // user-supplied probe-betas for TPQ state-vector snapshots. The
    // orchestrator passes this through to ``MtpqOptions::probe_betas``
    // Empty (default) -> no snapshots are taken. Ignored by FTLM / OFTLM
    // (which never have a meaningful TPQ state to snapshot).
    // -----------------------------------------------------------------
    std::vector<double> probe_betas;
};

// ---------------------------------------------------------------------------
// Public entry points.
//
// `ed::solve`     -- ground-state eigenproblem. Picks Lanczos /
//                    Block-Lanczos / Krylov-Schur / full-diag based on
//                    (num_eigs, geometry().global_dim, opts.method).
// `ed::thermal`   -- finite-T workflow. Switches on `opts.method`.
//
// Both orchestrators construct the appropriate Backend internally
// via `select_backend(H.geometry(), opts.backend)` and return a uniform
// Result struct (Phase 3.3) carrying the BackendMetadata that fired.
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

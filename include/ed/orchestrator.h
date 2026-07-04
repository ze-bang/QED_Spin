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
//                       KPM-DOS).
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
#include <limits>
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
    /// Output directory for HDF5 results (legacy behaviour). Empty
    /// means "do not write".
    std::string output_dir;
    SolveMethod method         = SolveMethod::Auto;
    BackendConstraints backend;

    /// Completion guarantee (planner-as-dictator). When false (default), a solve
    /// whose ExecutionPlan is infeasible (working set / basis construction would
    /// exceed the memory budget) is REFUSED cleanly -- the orchestrator throws
    /// with the bottleneck + ranked suggestions BEFORE any large allocation, so
    /// the run can never OOM/crash mid-flight. Set true to dispatch anyway (the
    /// manual escape hatch; the Python lane maps `force=True` onto this). The
    /// Python `qed.solve` pre-flight enforces the same verdict before this point;
    /// this gate is the safety net for direct C++ / CLI / api_facade callers.
    bool        allow_infeasible = false;

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

    /// Method discriminator (matches the legacy auto/thermal lane tags).
    enum class Method : std::uint8_t {
        FTLM = 0, LTLM, mTPQ, KpmDos = 4, OFTLM,
    } method = Method::FTLM;

    // ---------------------------------------------------------------
    // Defaults aligned to `python/qed/thermal.py` (May 2026,
    // PR-1 step 2 of the "mirror examples" plan). Previous C++ defaults
    // are noted inline for callers consulting the diff. Most existing
    // tests / call sites override these fields explicitly; the legacy
    // defaults were also rather arbitrary (num_samples=30, taylor_order=50,
    // num_temp_bins=100). Pinning the C++ defaults to the Python
    // canonical values makes the two surfaces line up byte-for-byte
    // when `ed::api::ThermalOptions` is passed through `to_legacy()`.
    // ---------------------------------------------------------------
    std::size_t num_samples    = 40;   ///< Python default (was 30).
    std::size_t krylov_dim     = 100;
    std::size_t num_exact      = 8;    ///< OFTLM: # low-lying states treated exactly (N_V).
    std::size_t taylor_order   = 8;    ///< Python default (was 50). mTPQ Taylor truncation.
    std::vector<double> betas;
    double      delta_beta     = 0.05; ///< Python default (was 0.1). mTPQ imag-time step.
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
    double      temp_min       = 0.1;  ///< Python `T_min` default (was 0.01).
    double      temp_max       = 10.0;
    std::size_t num_temp_bins  = 24;   ///< Python `num_T` default (was 100).

    /// Lorentzian broadening eta for the KPM-DOS density lane.
    double      broadening     = 0.05;

    /// Filter for the streaming-symmetry sector loop. See
    /// ``SolveOptions::selected_sectors``. Empty => walk every
    /// non-empty sector.
    std::vector<std::size_t> selected_sectors;

    // -----------------------------------------------------------------
    // Wave B3 (May 2026): caller-supplied spectral bounds for the
    // KPM-DOS lane. When BOTH ``e_min_override`` and ``e_max_override``
    // are finite, the kernel skips its own Lanczos-based spectral
    // estimation and uses the supplied window directly. Used by the
    // streaming-symmetry binding to estimate the band edges once and
    // reuse them across N sector calls. NaN means "estimate per
    // sector" (the legacy behaviour).
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
    // KPM-DOS knobs (closing-the-symmetry-gap follow-up, May 2026).
    //
    // Prior to this revision ``ThermalOptions`` carried no
    // moment-count or Hutchinson-sample knobs for the KpmDos lane,
    // so the orchestrator silently fell back to ``KpmDosOptions``'
    // defaults (M=2048, R=20). The Python facade
    // (qed.thermal(..., kpm_num_moments=..., kpm_num_random_vectors=...))
    // had no way to reach the streaming-symmetry binding -- a ~250x
    // amplifier on FT-KPM_DOS Symm at production sizes.
    //
    // ``0`` means "use the kernel default" so existing call sites
    // (CLI, single-operator orchestrator entry) keep their behaviour.
    // -----------------------------------------------------------------
    // Aligned to `python/qed/thermal.py` defaults (May 2026): 200
    // Chebyshev moments and 16 Hutchinson random vectors. Previous
    // value of 0 meant "use kernel default" (which was 2048/20 for
    // production use); the new defaults are tighter and match what the
    // Python facade has been shipping for the streaming-symmetry binding.
    int kpm_num_moments        = 200; ///< Python default (was 0 -> kernel 2048).
    int kpm_num_random_vectors = 16;  ///< Python default (was 0 -> kernel 20).

    // -----------------------------------------------------------------
    // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
    // user-supplied probe-betas for TPQ state-vector snapshots. The
    // orchestrator passes this through to ``MtpqOptions::probe_betas``
    // Empty (default) -> no snapshots
    // are taken. Ignored by FTLM / LTLM / KPM-DOS (which never have
    // a meaningful TPQ state to snapshot). Combine with
    // ``output_dir`` to land the saved states on disk under
    // ``ed_results.h5`` (``/tpq/samples/sample_<s>/state_beta_<b>``).
    // -----------------------------------------------------------------
    std::vector<double> probe_betas;

    /// Completion guarantee: when false (default), a thermal run whose plan would
    /// not fit the memory budget (kernel working set) is REFUSED cleanly before
    /// any allocation. Set true (Python force=True) to dispatch anyway.
    bool        allow_infeasible = false;
};

struct SpectralOptions {
    enum class Method : std::uint8_t {
        GroundStateCF = 0,   ///< compute_ground_state_dssf via cf_spectral_kernel
        FtlmDynamical = 1,
        /// Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026):
        /// Chebyshev expansion of `delta(omega - H)` against a single
        /// seed (GS by default, ``initial_state`` when provided).
        /// Backed by `ed::observables::kpm_dynamical_correlator` ->
        /// `ed::kpm::compute_kpm_ltlm_from_states` at beta = 0.
        KpmDynamical  = 2,
    } method = Method::GroundStateCF;

    /// KPM Chebyshev moments for ``Method::KpmDynamical``. Ignored by
    /// the other lanes. Bigger M = sharper features but more matvecs.
    std::size_t kpm_moments = 200;

    /// Window function applied to the Chebyshev moments when
    /// reconstructing S(omega). ``Jackson`` is the standard optimal
    /// Gibbs-suppressing window; ``Lorentz`` is the alternative used by
    /// the legacy `ed::kpm` driver and parametrised by
    /// ``kpm_lorentz_lambda``.
    enum class KpmKernel : std::uint8_t {
        Jackson = 0,
        Lorentz = 1,
    } kpm_kernel = KpmKernel::Jackson;

    /// Lorentz kernel decay parameter (only used when
    /// ``kpm_kernel == Lorentz``).
    double kpm_lorentz_lambda = 4.0;

    /// Override the auto-detected ``[E_min, E_max]`` spectral bounds
    /// used to rescale H into ``[-1, 1]`` before the Chebyshev
    /// recursion. When unset (default), the orchestrator runs a small
    /// Lanczos in `ed::kpm_dos::estimate_spectral_bounds` to discover
    /// them. NB: the kernel currently uses the legacy driver's own
    /// bound-estimation path; this field is reserved for the next
    /// landing wave once we plumb caller-supplied bounds through
    /// ``compute_kpm_ltlm_from_states``.
    std::optional<std::pair<double, double>> kpm_spectral_bounds;
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

    // -----------------------------------------------------------------
    // SOTA streaming-symmetry spectral controls (May 2026).
    // -----------------------------------------------------------------

    /// Momentum transfer Q of the probe operator (in fractional
    /// reciprocal-lattice units, e.g. ``Q = (1/3, 0)`` for a three-site
    /// chain). Used by the streaming-symmetry spectral workflow to
    /// pick the (initial, final) sector pairs that survive the
    /// selection rule ``k_final = k_initial + Q``. Empty => no
    /// momentum filter (every sector pair is walked).
    std::vector<double> momentum_transfer;

    /// Tolerance for the momentum-transfer match (in units of the
    /// fractional reciprocal-lattice coordinate). Pairs whose
    /// ``|k_final - k_initial - Q| mod 1`` exceeds this threshold are
    /// skipped. Default chosen so cubic-symmetry lattice integer
    /// quanta are not accidentally rejected.
    double      momentum_tolerance = 1e-6;

    /// Filter for the streaming-symmetry sector loop. See
    /// ``SolveOptions::selected_sectors``. Empty => walk every
    /// (initial, final) pair surviving the selection rule.
    std::vector<std::size_t> selected_sectors;

    // -----------------------------------------------------------------
    // Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026):
    // user-supplied seed state for the GroundStateCF lane. When
    // non-empty the orchestrator skips the inner Lanczos GS solve and
    // feeds this vector (after L2-renormalisation) straight into
    // ``cf_spectral_kernel`` -- this is the TPQ-to-CF pipeline (use
    // ``ThermalOptions::probe_betas`` to persist a TPQ state at
    // beta = 1/T, reload it via h5py, pass it here).
    //
    // Length MUST equal ``H.geometry().local_dim``. Empty (default)
    // keeps the legacy behaviour: compute the GS via Lanczos and use
    // it as the seed.
    // -----------------------------------------------------------------
    std::vector<std::complex<double>> initial_state;

    /// Completion guarantee: when false (default), a spectral run whose plan would
    /// not fit the memory budget is REFUSED cleanly before any allocation. Set
    /// true (Python force=True) to dispatch anyway.
    bool        allow_infeasible = false;
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

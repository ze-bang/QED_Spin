#pragma once

// =============================================================================
// EDParameters — extracted from ed_wrapper.h so that translation units which
// only need the parameter struct (notably ed_config_adapter.h and any code
// that converts between EDConfig and the legacy EDParameters bag) don't have
// to drag in TPQ.h, CG.h, lanczos.h, ftlm.h, ltlm.h, hybrid_thermal.h,
// arpack.h, hdf5_io.h, observables.h, system_utils.h, the GPU wrapper, etc.
// (D-2 in the modernization audit.)
//
// The struct still owns std::vector<Operator>, so this header includes
// ed/core/construct_ham.h for the Operator definition. That is the ONLY
// large header it brings in — everything else is std::* and ed/core/ed_types.h.
// =============================================================================

#include <ed/core/construct_ham.h>  // For Operator (member of EDParameters)
#include <ed/core/ed_types.h>       // For DiagonalizationMethod (used by callers)

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief Structure for exact diagonalization parameters
 *
 * Contains all parameters needed to configure various diagonalization methods,
 * including convergence criteria, method-specific options, and observable
 * calculations. New code is encouraged to use the structured EDConfig
 * (see ed/core/ed_config.h); EDParameters is retained for the legacy
 * dispatcher path.
 */
struct EDParameters {
    // ========== General Parameters ==========
    uint64_t max_iterations = 10000;
    uint64_t num_eigenvalues = 1;
    double tolerance = 1e-10;
    bool compute_eigenvectors = false;
    std::string output_dir = "";

    // ========== Method-Specific Parameters ==========
    uint64_t block_size = 4;

    // ========== Thermal Calculation Parameters ==========
    uint64_t num_samples = 1;
    double temp_min = 1e-3;
    double temp_max = 20;
    uint64_t num_temp_bins = 100;

    // ========== TPQ-Specific Parameters ==========
    uint64_t tpq_max_steps = 10000;
    uint64_t tpq_measurement_interval = 100;
    // mTPQ LargeValue L in |psi_{k+1}> = (L*I - H)|psi_k>. 0.0 = AUTO: the
    // orchestrator derives a sensible L from the spectral bounds/bandwidth
    // (L_auto). A finite POSITIVE value is the HPhi-style expert override that
    // pins L. (Was 1e5, which always pinned L huge -> the step is ~L*I, so
    // mTPQ barely cooled -- it stayed near infinite temperature regardless of
    // step count. Default to AUTO so mTPQ actually reaches low T.)
    double tpq_energy_shift = 0.0;

    double tpq_beta_max = 20.0;
    double tpq_delta_beta = 1e-2;
    uint64_t tpq_taylor_order = 100;

    bool tpq_continue = false;
    uint64_t tpq_continue_sample = 0;
    double tpq_continue_beta = 0.0;
    double tpq_target_beta = 1000.0;

    uint64_t tpq_num_measure_points = 20;
    double tpq_measure_beta_min = 1.0;
    double tpq_measure_beta_max = 1000.0;

    // ------------------------------------------------------------------
    // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
    // user-supplied probe-beta list for mTPQ/cTPQ state-vector
    // snapshots. The Python facade (``qed.thermal(probe_betas=[...])``)
    // routes here, then ``_ed_params_to_thermal_options`` forwards it
    // to ``_core.ThermalOptions::probe_betas``. Empty (default) means
    // "trajectory only, no state vectors persisted".
    // ------------------------------------------------------------------
    std::vector<double> tpq_probe_betas;

    // ------------------------------------------------------------------
    // Removed in matvec-unification Phase 7.5:
    //   - num_order()             -> tpq_taylor_order
    //   - num_measure_freq()      -> tpq_measurement_interval
    //   - delta_tau()             -> tpq_delta_beta
    //   - large_value()           -> tpq_energy_shift
    //   - continue_quenching()    -> tpq_continue
    //   - continue_sample()       -> tpq_continue_sample
    //   - continue_beta()         -> tpq_continue_beta
    //   - target_beta()           -> tpq_target_beta
    //
    // These were [[deprecated]] accessor shims for the canonical
    // tpq_<name> data members above. All in-tree callers have been
    // migrated; out-of-tree callers should rename the call sites
    // (search-and-replace).
    // ------------------------------------------------------------------

    // ========== FTLM-Specific Parameters ==========
    uint64_t ftlm_krylov_dim = 100;
    bool ftlm_full_reorth = true;
    uint64_t ftlm_reorth_freq = 10;
    uint64_t ftlm_seed = 0;
    bool ftlm_store_samples = false;
    bool ftlm_error_bars = true;

    // ========== LTLM-Specific Parameters ==========
    uint64_t ltlm_krylov_dim = 200;
    uint64_t ltlm_ground_krylov = 100;
    bool ltlm_full_reorth = true;
    uint64_t ltlm_reorth_freq = 10;
    uint64_t ltlm_seed = 0;
    bool ltlm_store_data = false;
    // ========== KPM-DOS-Specific Parameters ==========
    // Kernel Polynomial Method density-of-states + thermodynamics.
    // See include/ed/solvers/kpm_dos.h for full algorithmic specification.
    uint64_t kpm_num_moments = 2048;          // M, Chebyshev moments
    uint64_t kpm_num_random_vectors = 20;     // R, Hutchinson samples
    uint64_t kpm_num_quadrature_nodes = 0;    // 0 = auto = 2*M
    uint64_t kpm_spectral_bounds_krylov = 150;
    double kpm_spectral_bound_buffer = 0.05;
    bool kpm_use_jackson_kernel = true;       // false = Lorentz kernel
    double kpm_lorentz_lambda = 4.0;
    bool kpm_full_reorth = true;
    uint64_t kpm_reorth_freq = 10;
    uint64_t kpm_seed = 0;

    // ========== GPU Lanczos / Krylov-Schur Determinism ==========
    // Starting-vector RNG seed for the GPU Lanczos / GPU Krylov-Schur
    // family. 0 keeps the legacy deterministic seed (42); a nonzero
    // value passes through verbatim so a GPU run can be made to
    // reproduce a CPU run that uses the same seed (audit S1 #21).
    uint64_t lanczos_seed = 0;

    // ========== Observable Calculations ==========
    mutable std::vector<Operator> observables = {};
    mutable std::vector<std::string> observable_names = {};
    double omega_min = -10.0;
    double omega_max = 10.0;
    uint64_t num_points = 1000;
    double t_end = 50.0;
    double dt = 0.01;

    // ========== Lattice Parameters ==========
    uint64_t num_sites = 0;
    float spin_length = 0.5;
    uint64_t sublattice_size = 1;
    std::vector<int> selected_sectors;

    // ========== TPQ Observable Parameters ==========
    bool save_thermal_states = false;
    bool compute_spin_correlations = false;

    // ------------------------------------------------------------------
    // Removed in matvec-unification Phase 7.5:
    //   - calc_observables()      -> save_thermal_states
    //   - measure_spin()          -> compute_spin_correlations
    //
    // These were [[deprecated]] accessor shims. All in-tree callers
    // have been migrated.
    // ------------------------------------------------------------------

    // ========== Fixed-Sz Parameters ==========
    bool use_fixed_sz = false;
    int64_t n_up = -1;
    mutable FixedSzOperator* fixed_sz_op = nullptr;
    bool full_sz_split = false;

    // ========== Phase 7: orthogonal device / parallelism axes ==========
    //
    // The DiagonalizationMethod enum used to encode the device backend
    // ("LANCZOS" vs "LANCZOS_GPU") and the parallelism backend ("mTPQ" vs
    // "mTPQ_MPI") in the enum value itself, which (a) caused a quadratic
    // blow-up of the enum surface, (b) duplicated the same axis already
    // factored out for fixed-Sz, and (c) led to inconsistent naming
    // (mTPQ_GPU vs mTPQ_CUDA, both for the same code path).
    //
    // Phase 7 promotes these to flags on EDParameters, mirroring the
    // existing use_fixed_sz convention. The dispatcher canonicalizes
    // legacy `_GPU` / `_CUDA` / `_MPI` enum values onto base method +
    // flag at the entry point, so old call sites keep working without
    // change while new code can stay on the orthogonal axes:
    //
    //     SOLVER_type  ×  use_fixed_sz  ×  use_gpu  ×  use_mpi
    //
    // ScaLAPACK is *not* a "FULL + use_mpi" alias -- it's a distinct
    // distributed dense kernel. Retired in May 2026 (minimalist refactor).
    bool use_gpu = false;
    bool use_mpi = false;

    // -----------------------------------------------------------------------
    // GPU fallback policy (matvec-unification, May 2026).
    //
    // ``allow_gpu_cpu_fallback`` controls what happens when
    // ``use_gpu=true`` is requested but the build / runtime can't
    // service the GPU lane:
    //   * true  (default) -- silently re-canonicalise to the CPU base
    //                        method and emit one stderr line. The
    //                        Python facade depends on this: it surfaces
    //                        device='gpu' as a hint and lets the
    //                        orchestrator pick CPU when WITH_CUDA=OFF.
    //   * false           -- throw with an actionable message instead of
    //                        falling back. Set this when the caller has
    //                        explicitly asked for a GPU run and would
    //                        rather fail than silently get CPU output.
    //                        The orchestrator (``ed::workflows::solve``
    //                        with ``BackendConstraints::allow_gpu =
    //                        true`` but no other lane allowed) sets this
    //                        to false.
    bool allow_gpu_cpu_fallback = true;

    // ========== Symmetry Options (5th orthogonal axis) ==========
    //
    // Symmetry projection used to be encoded in the entry-point name
    // (`exact_diagonalization_from_directory_symmetrized`,
    // `exact_diagonalization_streaming_symmetry`,
    // `exact_diagonalization_chunked_symmetry`, etc., plus the same
    // four for fixed-Sz, giving 8 distinct symmetry-aware entry
    // points). The matvec / surface-unification refactors collapsed
    // them all onto a single flag plus the orthogonal axes:
    //
    //     SOLVER_type × use_fixed_sz × use_gpu × use_mpi × use_symmetry
    //
    // When ``use_symmetry == true``, the orchestrator
    // (``ed::workflows::solve``) routes through the streaming-symmetry
    // operator built by ``ed::make_streaming_symmetry_operator``, which
    // is the only path that:
    //   * keeps orbit data in memory (no disk basis materialisation),
    //   * supports ``use_gpu`` (per-sector GPU kernels),
    //   * supports ``use_fixed_sz`` orthogonally,
    //   * scales to the largest tractable systems (32-site spin-1/2 etc.),
    //   * works in pure Python (no ``automorphism_results/`` on disk
    //     required beyond the one-shot generator the streaming path runs).
    //
    // `translation_only` is an *orthogonal* sub-flag: when true, the
    // automorphism generator restricts to the translation subgroup. It
    // controls *which* symmetries are used, not whether symmetries are
    // exploited at all.
    bool use_symmetry = false;
    bool translation_only = false;

    // Streaming-symmetry-specific options. They are only consulted when
    // ``ed::make_operator``/``ed::workflows::solve`` route through the
    // streaming-symmetry kernel (``use_symmetry == true``). Carried on
    // ``EDParameters`` so that there is *one* parameter bag carrying
    // every solve option.
    //
    //   * basis_cache_dir       -- if non-empty, cache the per-sector
    //                              orbit basis here so reruns at fixed
    //                              symmetry sector skip the basis build.
    //                              Empty (default) disables caching.
    //   * precompute_basis_only -- when true, return immediately after
    //                              the basis cache is written; eigen-
    //                              values are not computed. Useful for
    //                              prebuilding caches in batch jobs.
    std::string basis_cache_dir{};
    bool precompute_basis_only = false;

    // ScaLAPACK / ARPACK / hybrid / Davidson / LOBPCG parameter blocks
    // were retired with their solvers in the minimalist-architecture rev
    // (May 2026). EDParameters now carries only the knobs that are still
    // used by the kept solvers (LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR /
    // FULL / FTLM / LTLM / mTPQ / cTPQ / KPM_DOS).
};

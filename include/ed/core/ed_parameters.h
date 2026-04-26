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

class FixedSzOperator;  // forward decl — used as bare pointer member

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
    double shift = 0.0;
    uint64_t block_size = 4;
    uint64_t max_subspace = 100;
    double target_lower = 0.0;
    double target_upper = 0.0;

    // ========== Thermal Calculation Parameters ==========
    uint64_t num_samples = 1;
    double temp_min = 1e-3;
    double temp_max = 20;
    uint64_t num_temp_bins = 100;

    // ========== TPQ-Specific Parameters ==========
    uint64_t tpq_max_steps = 10000;
    uint64_t tpq_measurement_interval = 100;
    double tpq_energy_shift = 1e5;

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

    // ========== DEPRECATED PARAMETER ACCESSORS ==========
    [[deprecated("Use tpq_taylor_order instead")]]
    uint64_t& num_order() { return tpq_taylor_order; }
    [[deprecated("Use tpq_taylor_order instead")]]
    uint64_t num_order() const { return tpq_taylor_order; }

    [[deprecated("Use tpq_measurement_interval instead")]]
    uint64_t& num_measure_freq() { return tpq_measurement_interval; }
    [[deprecated("Use tpq_measurement_interval instead")]]
    uint64_t num_measure_freq() const { return tpq_measurement_interval; }

    [[deprecated("Use tpq_delta_beta instead")]]
    double& delta_tau() { return tpq_delta_beta; }
    [[deprecated("Use tpq_delta_beta instead")]]
    double delta_tau() const { return tpq_delta_beta; }

    [[deprecated("Use tpq_energy_shift instead")]]
    double& large_value() { return tpq_energy_shift; }
    [[deprecated("Use tpq_energy_shift instead")]]
    double large_value() const { return tpq_energy_shift; }

    [[deprecated("Use tpq_continue instead")]]
    bool& continue_quenching() { return tpq_continue; }
    [[deprecated("Use tpq_continue instead")]]
    bool continue_quenching() const { return tpq_continue; }

    [[deprecated("Use tpq_continue_sample instead")]]
    uint64_t& continue_sample() { return tpq_continue_sample; }
    [[deprecated("Use tpq_continue_sample instead")]]
    uint64_t continue_sample() const { return tpq_continue_sample; }

    [[deprecated("Use tpq_continue_beta instead")]]
    double& continue_beta() { return tpq_continue_beta; }
    [[deprecated("Use tpq_continue_beta instead")]]
    double continue_beta() const { return tpq_continue_beta; }

    [[deprecated("Use tpq_target_beta instead")]]
    double& target_beta() { return tpq_target_beta; }
    [[deprecated("Use tpq_target_beta instead")]]
    double target_beta() const { return tpq_target_beta; }

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
    [[deprecated("Use method=HYBRID instead")]]
    bool use_hybrid_method = false;
    double hybrid_crossover = 1.0;
    bool hybrid_auto_crossover = false;

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

    [[deprecated("Use save_thermal_states instead")]]
    bool& calc_observables() { return save_thermal_states; }
    [[deprecated("Use save_thermal_states instead")]]
    bool calc_observables() const { return save_thermal_states; }

    [[deprecated("Use compute_spin_correlations instead")]]
    bool& measure_spin() { return compute_spin_correlations; }
    [[deprecated("Use compute_spin_correlations instead")]]
    bool measure_spin() const { return compute_spin_correlations; }

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
    // distributed dense kernel (different LAPACK call, different
    // block-cyclic data layout, mixed-precision refinement). So
    // ScaLAPACK / SCALAPACK_MIXED stay as their own DiagonalizationMethod
    // values; they implicitly require MPI.
    bool use_gpu = false;
    bool use_mpi = false;

    // ========== Symmetry Options (Phase 7.1: 5th orthogonal axis) ==========
    //
    // Symmetry projection used to be encoded in the entry-point name:
    //   exact_diagonalization_from_directory_symmetrized(...)   // disk-block
    //   exact_diagonalization_streaming_symmetry(...)           // streaming
    //   exact_diagonalization_chunked_symmetry(...)             // chunked
    //   exact_diagonalization_disk_chunked_symmetry(...)        // disk-chunked
    // plus the same four for fixed-Sz, giving 8 distinct symmetry-aware
    // entry points. Phase 7.1 collapses all of them onto a single flag:
    //
    //     SOLVER_type  ×  use_fixed_sz  ×  use_gpu  ×  use_mpi  ×  use_symmetry
    //
    // When `use_symmetry == true`, exact_diagonalization_from_files /
    // exact_diagonalization_from_directory route through the streaming
    // symmetry kernel (ed_wrapper_streaming.h), which is the only path that:
    //   * keeps orbit data in memory (no disk basis materialisation),
    //   * supports use_gpu (per-sector GPU kernels),
    //   * supports use_fixed_sz orthogonally,
    //   * scales to the largest tractable systems (32-site spin-1/2 etc.),
    //   * works in pure Python (no /automorphism_results/ on disk required
    //     beyond the one-shot generator the streaming path runs).
    //
    // The deprecated explicit-block path (`*_symmetrized`) and the
    // chunked / disk-chunked variants are kept as CLI-only escape hatches
    // for very-large-N memory-budget edge cases, but they are no longer
    // selectable via the EDParameters flag axis. Setting use_symmetry=true
    // is the only way new code should request symmetry projection.
    //
    // `translation_only` is an *orthogonal* sub-flag: when true, the
    // automorphism generator restricts to the translation subgroup. It
    // controls *which* symmetries are used, not whether symmetries are
    // exploited at all.
    bool use_symmetry = false;
    bool translation_only = false;

    // ========== ScaLAPACK Distributed Diagonalization Options ==========
    int scalapack_nprow = 0;
    int scalapack_npcol = 0;
    int scalapack_block_size = 64;
    bool scalapack_mixed_precision = true;
    double scalapack_refinement_tol = 1e-12;
    int scalapack_max_refinement_iter = 5;
    bool scalapack_verbose = true;

    // Phase 8 #5: when true (the default), ``scalapack_block_size`` is
    // ignored at solve time and replaced by ``get_optimal_block_size(N,
    // nprow, npcol)`` -- a heuristic that balances the local-tile size
    // against the BLACS process grid for the actual matrix dimension.
    // The legacy default of 64 tends to be too small for the larger
    // matrices we now run (N >> 64*sqrt(P)), and over-blocks the smaller
    // ones; the auto path consistently beats the fixed default in the
    // benchmarks against MKL ScaLAPACK 2024 on 4-32 MPI ranks.
    //
    // CLI ``--scalapack-block-size N`` overrides ``scalapack_block_size``
    // *and* sets this flag to false, preserving backward compatibility
    // for users who tune the block size manually.
    bool scalapack_block_size_auto = true;

    // ========== ARPACK Advanced Options ==========
    bool arpack_advanced_verbose = false;
    std::string arpack_which = "SR";
    int64_t arpack_ncv = -1;
    uint64_t arpack_max_restarts = 2;
    double arpack_ncv_growth = 1.5;
    bool arpack_auto_enlarge_ncv = true;
    bool arpack_two_phase_refine = true;
    double arpack_relaxed_tol = 1e-6;
    bool arpack_shift_invert = false;
    double arpack_sigma = 0.0;
    bool arpack_auto_switch_shift_invert = true;
    double arpack_switch_sigma = 0.0;
    bool arpack_adaptive_inner_tol = true;
    double arpack_inner_tol_factor = 1e-2;
    double arpack_inner_tol_min = 1e-14;
    uint64_t arpack_inner_max_iter = 300;
};

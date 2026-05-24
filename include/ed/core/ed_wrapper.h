#pragma once

// ============================================================================
// DEPRECATION NOTICE (Phase 5 — Minimalist ED Collapse, May 2026)
// ============================================================================
// This entire header is part of the LEGACY public surface that has been
// SUPERSEDED by `ed/orchestrator.h` (`ed::workflows::solve / thermal /
// spectral`). Every public entry point declared here — including
// `exact_diagonalization_core()`, `exact_diagonalization_from_files()`,
// `exact_diagonalization_fixed_sz()`, and the per-method routers — is
// scheduled for removal once all in-tree callers (CLI + Python binding +
// unit tests) have been migrated.
//
// MIGRATION PATH:
//   * C++ callers:    `ed::workflows::solve(op, opts)`           (ground state)
//                     `ed::workflows::thermal(op, opts)`         (mTPQ/cTPQ/FTLM)
//                     `ed::workflows::spectral(op, obs, opts)`   (DSSF/CF)
//   * Python:         `qed.solve(...)`, `qed.thermal(...)`, `qed.spectral(...)`
//                     The old `qed.diag` / `qed.dssf.compute` symbols remain
//                     as thin aliases (see `python/qed/_aliases.py`).
//
// During the transition, the entry points below continue to work and are
// implemented in terms of the new kernels via `src/orchestrator.cpp`.
// New code MUST use `ed::workflows::*`; legacy code is grandfathered.
// ============================================================================

// ============================================================================
// INCLUDES
// ============================================================================
#include <ed/solvers/TPQ.h>
#include <ed/solvers/lanczos.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/kpm_dos.h>
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/solvers/observables.h>
#include <ed/core/ed_config.h>
#include <ed/core/system_utils.h>
#include <ed/core/ed_logging.h>  // for ed_log::debug() — D-5 routes [DEBUG] prints through here
#include <sys/stat.h>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <memory>  // For std::unique_ptr

// GPU support
#ifdef WITH_CUDA
#include <ed/gpu/gpu_ed_wrapper.h>
#endif

// ============================================================================
// Legacy CPU/GPU dispatcher header. The `_GPU` / `_CUDA` / `_MPI` /
// `_FIXED_SZ` enum aliases were retired in the minimalist-architecture
// rev (May 2026); device and parallelism are now plain
// EDParameters::use_gpu / use_mpi / use_fixed_sz flags. See
// docs/architecture/STRUCTURAL_AUDIT.md Part IV.
// ============================================================================


// The five `std::vector<Complex>` operator overloads (`operator+`, `operator-`,
// `operator+=`, `operator-=`, `operator*` with a scalar) that used to live
// here were retired in the minimalist-architecture rev (May 2026): every
// vector-arithmetic hotspot already goes through cblas_z* / cublasZ* / the
// `Backend::axpy` family, so the inline `for`-loop versions had no callers
// and only existed to keep this header monolithic.

// ============================================================================
// ENUMS AND STRUCTURES
// ============================================================================

// DiagonalizationMethod has been moved to <ed/core/ed_types.h> so that the
// implementation in src/core/ed_config.cpp does not need to include this
// 4500-line header just to mention the enum. (P0.14 / audit Q6.)
#include <ed/core/ed_types.h>
// Method-classification predicates (is_tpq_method, is_gpu_method, ...) live
// in their own header so other TUs don't have to include this monolith
// just to call them. (D-4 in the audit.)
#include <ed/core/ed_method_traits.h>

// ============================================================================
// FEATURE AVAILABILITY CHECKS
// ============================================================================

/**
 * @brief Check if ScaLAPACK support was compiled in. The dense ScaLAPACK
 * kernel was retired in the minimalist-architecture rev; the helper is
 * kept as `false` for back-compat with feature-check callers.
 */
inline bool is_scalapack_compiled() { return false; }

/**
 * @brief Check if CUDA/GPU support was compiled in
 */
inline bool is_cuda_compiled() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}

// `get_fallback_method` was retired in the minimalist-architecture rev
// (May 2026): after the ScaLAPACK / ARPACK / Davidson / LOBPCG / Hybrid
// cleanup it was an identity pass-through with zero callers. The use_gpu /
// use_mpi fallback axes are handled directly in `ed::auto_pilot::solve`
// (see include/ed/auto/solve.h).

/**
 * @brief Structure to hold exact diagonalization results
 */
struct EDResults {
    std::vector<double> eigenvalues;
    bool eigenvectors_computed;
    std::string eigenvectors_path;
    ThermodynamicData thermo_data;  // For thermal calculations
    FTLMResults ftlm_results;       // For FTLM calculations (includes per-sector data)
};

// EDParameters lives in <ed/core/ed_parameters.h> so that the legacy-config
// adapter (and any other code that only needs the parameter bag) doesn't have
// to drag in TPQ.h / CG.h / lanczos.h / ftlm.h / ltlm.h / arpack.h / hdf5_io.h /
// observables.h / system_utils.h / GPU wrappers via this header. (D-2 in the
// modernization audit.)
#include <ed/core/ed_parameters.h>


/**
 * @brief Enum for Hamiltonian file formats
 */
enum class HamiltonianFileFormat {
    STANDARD,       // InterAll.dat and Trans.dat format
    SPARSE_MATRIX,  // Sparse matrix format
    CUSTOM          // Custom format requiring a parser function
};

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// Core diagonalization function
EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    uint64_t hilbert_space_dim,
    DiagonalizationMethod method,
    const EDParameters& params
);

// Helper functions
namespace ed_internal {

    // ========== Method Classification Helpers ==========
    //
    // The actual predicates (is_tpq_method, is_ftlm_method, is_gpu_method,
    // is_deprecated_fixed_sz_method, normalize_method,
    // requires_ground_state_sector, requires_sector_combination) live in
    // <ed/core/ed_method_traits.h>. They are constexpr-friendly and
    // depend only on <ed/core/ed_types.h>, so other translation units
    // (workflows.cpp, ed_config.cpp, future Python bindings) can include
    // *just* the traits header instead of dragging in this whole monolith
    // every time they want to ask "is this method GPU?". (D-4 in the
    // modernization audit.)
    //
    // The `using` declarations inside ed_method_traits.h re-export the
    // ed:: predicates into ed_internal::, so existing call sites keep
    // working unchanged.
    
    // `normalize_method_and_fixed_sz` (no-op shim kept for back-compat with
    // legacy `_FIXED_SZ` enum variants) was retired in the May 2026
    // cleanup sweep: zero call sites anywhere in the tree.

    // `supports_fixed_sz(method)` was retired in the minimalist-
    // architecture rev (May 2026): after Phase 1 every remaining
    // method (LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR / FULL / FTLM /
    // LTLM / TPQ_MICROCANONICAL / TPQ_CANONICAL / KPM_DOS) supports
    // fixed-Sz via `FixedSzOperator` (or its GPU counterpart), so the
    // helper unconditionally returned true. The downstream guard
    // (line ~1640) that consulted it is now redundant and was
    // removed alongside.

    // ========== Forward Declarations ==========
    //
    // `process_thermal_correlations` and `diagonalize_matrix_free` were retired
    // in the minimalist-architecture rev (May 2026): the former was wired only
    // to a commented-out call site below and its body was already non-functional
    // (TODO marker), the latter had zero callers anywhere in the tree.

    Operator load_hamiltonian_from_files(
        const std::string& interaction_file,
        const std::string& single_site_file,
        const std::string& counterterm_file,
        const std::string& three_body_file,
        uint64_t num_sites,
        float spin_length,
        DiagonalizationMethod method,
        HamiltonianFileFormat format
    );

    std::function<void(const Complex*, Complex*, int)> create_hamiltonian_apply_function(
        Operator& hamiltonian
    );

    // Phase 9 cleanup: the explicit-block symmetry helpers
    // (``diagonalize_symmetry_block``, ``setup_*_symmetry_basis``,
    // ``find_ground_state_sector*``, ``diagonalize_fixed_sz_sector``,
    // ``transform_sector_to_*``, ``transform_and_save_*``, plus the
    // ``GroundStateSectorInfo``/``SectorInfo``/``SectorResult`` PODs)
    // were removed together with the deprecated public
    // ``exact_diagonalization_*_symmetrized`` entry points. The
    // canonical symmetry path is the streaming kernel in
    // ``ed/core/ed_wrapper_streaming.h`` -- it is matrix-free
    // (no per-sector sparse blocks on disk), GPU-capable per
    // sector, and reachable via
    // ``exact_diagonalization_from_directory(...)`` with
    // ``EDParameters::use_symmetry = true``.
}

// ============================================================================
// MAIN EXACT DIAGONALIZATION FUNCTION
// ============================================================================

/**
 * @brief Main wrapper function for exact diagonalization
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param hilbert_space_dim Dimension of the Hilbert space
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    uint64_t hilbert_space_dim,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params_in = EDParameters()
) {
    EDResults results;

    // Phase 6.1: empty params.output_dir historically meant "current working
    // directory" -- underlying solvers (e.g. thick_restart_lanczos,
    // shift_invert_lanczos) would dump eigenvalues.dat / eigenvalues.txt
    // into wherever the process happened to be running, polluting cwd
    // every time the Python dispatcher was used. Mirror the
    // ``output_dir_or_devnull`` convention used by the standalone Python
    // wrappers: empty -> "/dev/null", which the C++ HDF5 layer
    // (``HDF5IO::isDisabledOutputPath``) and the ``ofstream`` writes (via
    // path concatenation that produces a non-openable path) both treat
    // as "skip all I/O". Pass ``"."`` explicitly to restore legacy
    // cwd-dump behaviour.
    EDParameters params = params_in;
    if (params.output_dir.empty()) {
        params.output_dir = "/dev/null";
    }

    // Phase 7: collapse legacy `_GPU` / `_CUDA` / `_MPI` / `_FIXED_SZ`
    // enum variants onto the canonical (base, use_fixed_sz, use_gpu,
    // use_mpi) tuple, then rebuild the legacy "_GPU" enum value from
    // the flag for the existing CPU-vs-GPU dispatcher branches below.
    // This is the single normalization point: every code path beneath
    // this one sees a fully orthogonal ``(method, params)``.
    {
        const auto canon = ed::canonicalize_method_and_flags(
            method, params.use_fixed_sz, params.use_gpu, params.use_mpi);
        params.use_fixed_sz = canon.use_fixed_sz;
        params.use_gpu      = canon.use_gpu;
        params.use_mpi      = canon.use_mpi;
        method = ed::legacy_method_for_dispatch(canon.method, canon.use_gpu);
    }

    // Initialize output directory if needed (skip the disabled sentinel).
    if (!HDF5IO::isDisabledOutputPath(params.output_dir)) {
        std::string cmd = "mkdir -p " + params.output_dir;
        safe_system_call(cmd);
    }

    // Set eigenvectors flag in results
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors &&
        !HDF5IO::isDisabledOutputPath(params.output_dir)) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // Call the appropriate diagonalization method
    switch (method) {
        case DiagonalizationMethod::FULL:
            full_diagonalization(H, hilbert_space_dim, params.num_eigenvalues, results.eigenvalues, 
                                 params.output_dir, params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::LANCZOS:
            lanczos(H, hilbert_space_dim, params.max_iterations, params.num_eigenvalues,
                    params.tolerance, results.eigenvalues, params.output_dir,
                    params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::KRYLOV_SCHUR:
            krylov_schur(H, hilbert_space_dim, params.max_iterations,
                       params.num_eigenvalues, params.tolerance,
                       results.eigenvalues, params.output_dir,
                       params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::mTPQ: {
            // Trivial-sector short-circuit: dim=1 yields ``D_S = log2(1) = 0``
            // which makes the mTPQ β formula divide by zero (β goes
            // negative and the trajectory is unusable). For a 1-D
            // sector the thermodynamics are exact: <H>(β) = e_0,
            // Cv(β) = 0, F(β) = e_0, S(β) = 0.
            if (hilbert_space_dim == 1) {
                Complex v0(1.0, 0.0), Hv0(0.0, 0.0);
                H(&v0, &Hv0, 1);
                double e0 = Hv0.real();
                results.eigenvalues.push_back(e0);
                // Match the LOG-spaced T grid used by
                // ``compute_tpq_unified_thermo`` so the recombiner
                // sees consistent temperature axes across sectors.
                ThermodynamicData td;
                td.temperatures.resize(params.num_temp_bins);
                td.energy.assign(params.num_temp_bins, e0);
                td.specific_heat.assign(params.num_temp_bins, 0.0);
                td.entropy.assign(params.num_temp_bins, 0.0);
                td.free_energy.assign(params.num_temp_bins, e0);
                if (params.num_temp_bins <= 1) {
                    td.temperatures[0] = params.temp_min;
                } else {
                    double log_min = std::log(std::max(params.temp_min, 1e-300));
                    double log_max = std::log(std::max(params.temp_max, 1e-300));
                    double step = (log_max - log_min) /
                                  static_cast<double>(params.num_temp_bins - 1);
                    for (std::uint64_t i = 0; i < params.num_temp_bins; ++i) {
                        td.temperatures[i] =
                            std::exp(log_min + step * static_cast<double>(i));
                    }
                }
                results.thermo_data = std::move(td);
                break;
            }

            // Auto-pick ``tpq_energy_shift`` (a.k.a. ``LargeValue``)
            // when the caller passes the unified-pipeline sentinel
            // ``tpq_energy_shift = 0``. mTPQ recurs via
            // ``|v_{n+1}> = (L - H/D_S)|v_n>`` and needs ``L * D_S``
            // greater than the largest eigenvalue of H. The
            // historical default 1e5 is too large for small/medium
            // systems: each step advances β by only ``2 / (L * D_S)``,
            // so the chain only reaches ``β << 1`` in the budgeted
            // ``max_iterations`` and the recombiner sees no usable
            // data. A 24-step Lanczos sweep fixes this for ~all
            // single-sector workloads.
            double large_value = params.tpq_energy_shift;
            if (large_value == 0.0) {
                std::vector<double> alpha_lv, beta_lv;
                std::mt19937 gen_lv(0xC0FFEEu);
                ComplexVector v0_lv = generateGaussianRandomVector(
                    static_cast<int>(hilbert_space_dim), gen_lv);
                const std::uint64_t kdim = std::min<std::uint64_t>(
                    24, hilbert_space_dim);
                build_lanczos_tridiagonal_with_basis(
                    H, v0_lv, hilbert_space_dim, kdim, /*tol=*/1e-12,
                    /*full_reorth=*/true, /*reorth_freq=*/0,
                    alpha_lv, beta_lv, /*basis_vectors=*/nullptr);
                std::vector<double> ritz_lv, w_lv;
                diagonalize_tridiagonal_ritz(
                    alpha_lv, beta_lv, ritz_lv, w_lv, /*evecs=*/nullptr);
                if (!ritz_lv.empty()) {
                    double e_min = ritz_lv.front();
                    double e_max = ritz_lv.back();
                    double bw = std::max(e_max - e_min, 1.0);
                    large_value = e_max + 0.05 * bw;
                } else {
                    large_value = 1.0;
                }
            }
            // Honour ``tpq_max_steps`` as a hard cap on the chain length,
            // independent of the more generic ``max_iterations``. The CLI
            // / EDConfig path lets users say "stop after 5000 TPQ steps
            // regardless of solver-iteration heuristics"; the field had
            // previously been parsed but silently dropped before reaching
            // the solver.
            const std::uint64_t mtpq_max_steps =
                (params.tpq_max_steps > 0)
                    ? std::min<std::uint64_t>(params.max_iterations,
                                              params.tpq_max_steps)
                    : params.max_iterations;
            microcanonical_tpq(H, hilbert_space_dim,
                            mtpq_max_steps, params.num_samples,
                            params.tpq_measurement_interval,
                            results.eigenvalues,
                            params.output_dir,
                            params.compute_eigenvectors,
                            large_value,
                            params.save_thermal_states, params.observables, params.observable_names,
                            params.omega_min, params.omega_max,
                            params.num_points, params.t_end, params.dt, params.spin_length, 
                            params.compute_spin_correlations, params.sublattice_size, params.num_sites,
                            params.fixed_sz_op,
                            params.tpq_continue,
                            params.tpq_continue_sample,
                            params.tpq_continue_beta,
                            params.tpq_target_beta,
                            params.tpq_num_measure_points,
                            params.tpq_measure_beta_min,
                            params.tpq_measure_beta_max);
            // Audit follow-up: populate ``results.thermo_data`` so the
            // unified ``ed::auto_pilot::thermal`` entry point (and any
            // other caller of ``exact_diagonalization_core``) can read
            // the binned thermodynamics in-memory.
            if (!params.output_dir.empty()) {
                results.thermo_data = compute_tpq_unified_thermo(
                    params.output_dir,
                    params.temp_min,
                    params.temp_max,
                    params.num_temp_bins
                );
            }
            break;
        }

        case DiagonalizationMethod::cTPQ:
            // Trivial-sector short-circuit (same rationale as mTPQ).
            // cTPQ's Taylor expansion of ``exp(-Δβ H)`` is degenerate
            // for a 1-D Hilbert space, so we fill in the exact thermo
            // directly instead of running the iteration.
            if (hilbert_space_dim == 1) {
                Complex v0(1.0, 0.0), Hv0(0.0, 0.0);
                H(&v0, &Hv0, 1);
                double e0 = Hv0.real();
                results.eigenvalues.push_back(e0);
                ThermodynamicData td;
                td.temperatures.resize(params.num_temp_bins);
                td.energy.assign(params.num_temp_bins, e0);
                td.specific_heat.assign(params.num_temp_bins, 0.0);
                td.entropy.assign(params.num_temp_bins, 0.0);
                td.free_energy.assign(params.num_temp_bins, e0);
                if (params.num_temp_bins <= 1) {
                    td.temperatures[0] = params.temp_min;
                } else {
                    double log_min = std::log(std::max(params.temp_min, 1e-300));
                    double log_max = std::log(std::max(params.temp_max, 1e-300));
                    double step = (log_max - log_min) /
                                  static_cast<double>(params.num_temp_bins - 1);
                    for (std::uint64_t i = 0; i < params.num_temp_bins; ++i) {
                        td.temperatures[i] =
                            std::exp(log_min + step * static_cast<double>(i));
                    }
                }
                results.thermo_data = std::move(td);
                break;
            }
            // beta_max: prefer the dedicated ``tpq_beta_max`` knob when the
            // user has set it (config-file or auto-pilot path). Fall back
            // to ``temp_max`` for back-compat with callers that only set
            // the generic thermal grid bounds.
            {
                const double ctpq_beta_max =
                    (params.tpq_beta_max > 0.0) ? params.tpq_beta_max
                                                : params.temp_max;
                canonical_tpq(
                H,                      // Hamiltonian matvec
                hilbert_space_dim,      // N
                ctpq_beta_max,          // beta_max (tpq_beta_max if set, else temp_max)
                params.num_samples,     // num_samples
                params.tpq_measurement_interval, // temp_interval / measurement frequency
                results.eigenvalues,    // energies output vector
                params.output_dir,      // output dir
                params.tpq_delta_beta,  // delta_beta (imaginary-time step)
                params.tpq_taylor_order, // taylor_order
                params.save_thermal_states, // compute_observables
                params.observables,     // observables
                params.observable_names,// observable names
                params.omega_min,       // omega_min
                params.omega_max,       // omega_max
                params.num_points,      // num_points
                params.t_end,           // t_end
                params.dt,              // dt
                params.spin_length,     // spin length
                params.compute_spin_correlations, // measure Sz and fluctuations
                params.sublattice_size, // sublattice size
                params.num_sites,       // number of sites
                params.fixed_sz_op,      // Fixed-Sz operator for embedding
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max
            );
            }  // close ctpq_beta_max scope
            if (!params.output_dir.empty()) {
                results.thermo_data = compute_tpq_unified_thermo(
                    params.output_dir,
                    params.temp_min,
                    params.temp_max,
                    params.num_temp_bins
                );
            }
            break;


        // Note: DiagonalizationMethod::mTPQ_CUDA used to fall through here
        // as a no-op (silently returning empty EDResults). Phase 7 collapses
        // mTPQ_CUDA -> mTPQ + use_gpu=true via canonicalize_method_and_flags,
        // so it now reaches the GPU-error branch below (`case mTPQ_GPU:`)
        // and produces a clear "GPU methods must be called via
        // exact_diagonalization_from_files" diagnostic instead.

        case DiagonalizationMethod::BLOCK_LANCZOS:
            block_lanczos(H, hilbert_space_dim,
                        params.max_iterations, params.num_eigenvalues, params.block_size,
                        params.tolerance, results.eigenvalues,
                        params.output_dir, params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::FTLM:
            {
                // Setup FTLM parameters
                FTLMParameters ftlm_params;
                ftlm_params.krylov_dim = params.ftlm_krylov_dim;
                ftlm_params.num_samples = params.num_samples;
                ftlm_params.max_iterations = params.max_iterations;
                ftlm_params.tolerance = params.tolerance;
                ftlm_params.full_reorthogonalization = params.ftlm_full_reorth;
                ftlm_params.reorth_frequency = params.ftlm_reorth_freq;
                ftlm_params.random_seed = params.ftlm_seed;
                ftlm_params.store_intermediate = params.ftlm_store_samples;
                ftlm_params.compute_error_bars = params.ftlm_error_bars;
                
                // Run FTLM
                FTLMResults ftlm_results = finite_temperature_lanczos(
                    H, hilbert_space_dim, ftlm_params,
                    params.temp_min, params.temp_max, params.num_temp_bins,
                    params.output_dir
                );
                
                // Store results (both thermodynamics and full FTLM data)
                results.thermo_data = ftlm_results.thermo_data;
                results.ftlm_results = ftlm_results;  // Store for sector combination
                
                // Store ground state estimate as eigenvalue
                if (ftlm_results.ground_state_estimate != 0.0) {
                    results.eigenvalues.push_back(ftlm_results.ground_state_estimate);
                }
                
                // Save FTLM results to file (HDF5 goes to main output dir)
                if (!params.output_dir.empty()) {
                    safe_system_call("mkdir -p " + params.output_dir);
                    save_ftlm_results(ftlm_results, params.output_dir + "/ftlm_thermo.txt");
                }
            }
            break;
        
        case DiagonalizationMethod::LTLM:
            {
                // Standard LTLM (the legacy `use_hybrid_method` flag and
                // the HYBRID enum were retired in the minimalist refactor;
                // workflow-level glue picks LTLM vs FTLM by temperature.)
                // Setup LTLM parameters
                LTLMParameters ltlm_params;
                ltlm_params.krylov_dim = params.ltlm_krylov_dim;
                ltlm_params.ground_state_krylov = params.ltlm_ground_krylov;
                ltlm_params.num_samples = 1;  // LTLM typically uses 1 sample
                ltlm_params.max_iterations = params.max_iterations;
                ltlm_params.tolerance = params.tolerance;
                ltlm_params.full_reorthogonalization = params.ltlm_full_reorth;
                ltlm_params.reorth_frequency = params.ltlm_reorth_freq;
                ltlm_params.random_seed = params.ltlm_seed;
                ltlm_params.store_intermediate = params.ltlm_store_data;
                ltlm_params.compute_error_bars = false;  // Not needed for single sample
                
                // Run LTLM
                LTLMResults ltlm_results = low_temperature_lanczos(
                    H, hilbert_space_dim, ltlm_params,
                    params.temp_min, params.temp_max, params.num_temp_bins,
                    nullptr, params.output_dir
                );
                
                // Store results
                results.thermo_data = ltlm_results.thermo_data;
                results.eigenvalues.push_back(ltlm_results.ground_state_energy);
                
                // Save LTLM results to file (HDF5 goes to main output dir)
                if (!params.output_dir.empty()) {
                    safe_system_call("mkdir -p " + params.output_dir);
                    save_ltlm_results(ltlm_results, params.output_dir + "/ltlm_thermo.txt");
                }
            }
            break;
        
        case DiagonalizationMethod::KPM_DOS:
            {
                // Build temperature grid (linear in T, matching what FTLM/LTLM do)
                std::vector<double> betas;
                std::vector<double> temps;
                betas.reserve(params.num_temp_bins);
                temps.reserve(params.num_temp_bins);
                if (params.num_temp_bins == 1) {
                    temps.push_back(params.temp_min);
                    betas.push_back(1.0 / std::max(params.temp_min, 1e-300));
                } else {
                    double dT = (params.temp_max - params.temp_min) /
                                static_cast<double>(params.num_temp_bins - 1);
                    for (uint64_t i = 0; i < params.num_temp_bins; ++i) {
                        double T = params.temp_min + dT * static_cast<double>(i);
                        temps.push_back(T);
                        betas.push_back(1.0 / std::max(T, 1e-300));
                    }
                }

                ed::kpm_dos::KPMDOSParameters kpm_params;
                kpm_params.num_moments              = params.kpm_num_moments;
                kpm_params.num_random_vectors       = params.kpm_num_random_vectors;
                kpm_params.num_quadrature_nodes     = params.kpm_num_quadrature_nodes;
                kpm_params.spectral_bounds_krylov   = params.kpm_spectral_bounds_krylov;
                kpm_params.spectral_bound_buffer    = params.kpm_spectral_bound_buffer;
                kpm_params.use_jackson_kernel       = params.kpm_use_jackson_kernel;
                kpm_params.lorentz_lambda           = params.kpm_lorentz_lambda;
                kpm_params.tolerance                = params.tolerance;
                kpm_params.full_reorthogonalization = params.kpm_full_reorth;
                kpm_params.reorth_frequency         = params.kpm_reorth_freq;
                kpm_params.random_seed              = params.kpm_seed;

                ed::kpm_dos::KPMDOSResult kpm_res =
                    ed::kpm_dos::compute_kpm_dos(H, hilbert_space_dim, betas,
                                                 /*dos_energies=*/{}, kpm_params);

                results.thermo_data.temperatures   = temps;
                results.thermo_data.energy         = kpm_res.energy;
                results.thermo_data.specific_heat  = kpm_res.specific_heat;
                results.thermo_data.entropy        = kpm_res.entropy;
                results.thermo_data.free_energy    = kpm_res.free_energy;
                // Provide e_min so the resummation pipeline has the same
                // ground-state reference convention as FTLM/LTLM.
                results.thermo_data.e_min          = kpm_res.e_min_estimate;
                results.eigenvalues.push_back(kpm_res.e_min_estimate);

                // Persist /thermodynamics/{temperatures,energy,specific_heat,
                // entropy,free_energy} so the NLCE summation kernel
                // (NLC_sum_ftlm.py) finds the same dataset layout it reads
                // for FTLM.
                if (!params.output_dir.empty()) {
                    safe_system_call("mkdir -p " + params.output_dir);
                    try {
                        std::string h5_file = HDF5IO::createOrOpenFile(params.output_dir);
                        HDF5IO::saveThermodynamics(h5_file, temps, "energy",        kpm_res.energy);
                        HDF5IO::saveThermodynamics(h5_file, temps, "specific_heat", kpm_res.specific_heat);
                        HDF5IO::saveThermodynamics(h5_file, temps, "entropy",       kpm_res.entropy);
                        HDF5IO::saveThermodynamics(h5_file, temps, "free_energy",   kpm_res.free_energy);
                    } catch (const std::exception& e) {
                        std::cerr << "  KPM-DOS: failed to write thermodynamics HDF5: "
                                  << e.what() << std::endl;
                    }
                }
            }
            break;

        default:
            // Hard-fail on an unrecognised method instead of silently
            // returning empty results: in non-interactive runs the cerr
            // line was getting buried and downstream code would happily
            // operate on an empty EDResults.
            throw std::runtime_error(
                "exact_diagonalization_core: unknown DiagonalizationMethod "
                "(enum value " + std::to_string(static_cast<int>(method)) +
                "). Either the enum was extended without updating the "
                "dispatcher or the caller is passing an uninitialised method.");
    }

    if (params.compute_eigenvectors) {
        std::cout << "Eigenvectors computed and saved to " << params.output_dir << std::endl;
    }

    return results;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

namespace ed_internal {


// `create_operator` was retired in the minimalist-architecture rev
// (May 2026): no callers. It also leaked raw `new`-allocated
// `Operator*`/`FixedSzOperator*` pointers, which is the wrong owner-
// ship convention for the rest of this layer (the live workflows hold
// `std::shared_ptr<Operator>` / `std::shared_ptr<FixedSzOperator>`
// pairs and dispatch via a lambda -- see
// `build_workflow_hamiltonian` in src/cli/workflows.cpp).

/**
 * @brief Load Hamiltonian from files based on format
 */
inline Operator load_hamiltonian_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file,
    const std::string& counterterm_file,
    const std::string& three_body_file,
    uint64_t num_sites,
    float spin_length,
    DiagonalizationMethod method,
    HamiltonianFileFormat format
) {
    Operator hamiltonian(num_sites, spin_length);
    
    switch (format) {
        case HamiltonianFileFormat::STANDARD:
            // Load terms from files
            if (!single_site_file.empty()) {
                hamiltonian.loadFromFile(single_site_file);
            }
            if (!interaction_file.empty()) {
                hamiltonian.loadFromInterAllFile(interaction_file);
            }
            // Load three-body terms if provided
            if (!three_body_file.empty() && std::filesystem::exists(three_body_file)) {
                std::cout << "Loading three-body terms from: " << three_body_file << std::endl;
                hamiltonian.loadThreeBodyTerm(three_body_file);
            }
            break;
            
        case HamiltonianFileFormat::SPARSE_MATRIX:
            throw std::runtime_error("Sparse matrix format not yet implemented");
            
        case HamiltonianFileFormat::CUSTOM:
            throw std::runtime_error("Custom format requires a parser function");
            
        default:
            throw std::runtime_error("Unknown Hamiltonian file format");
    }
    
    return hamiltonian;
}

/**
 * @brief Create a lambda function to apply the Hamiltonian
 */
inline std::function<void(const Complex*, Complex*, int)> create_hamiltonian_apply_function(
    Operator& hamiltonian
) {
    return [&hamiltonian](const Complex* in, Complex* out, uint64_t n) {
        // Directly use pointer-based apply to avoid temporary vector allocations
        hamiltonian.apply(in, out, n);
    };
}

// `diagonalize_matrix_free` was retired in the minimalist-architecture rev
// (May 2026); call `exact_diagonalization_core(apply_wrapper, dim, method, params)`
// directly — the wrapper added no useful logic beyond a method-validity check
// that lives in `exact_diagonalization_core` itself now.

} // namespace ed_internal

// ============================================================================
// Sz CONSERVATION CHECKER
// ============================================================================

/**
 * @brief Check if a Hamiltonian conserves total Sz by scanning its term files.
 *
 * For spin-1/2 in the ladder basis (0=S+, 1=S-, 2=Sz):
 *   - Two-body (InterAll.dat): Sz is conserved iff every (Op_i, Op_j) pair is
 *     one of (0,1), (1,0), or (2,2)  — i.e. S+S-, S-S+, SzSz.
 *   - Single-site (Trans.dat):  Sz is conserved iff every Op is 2 (Sz).
 *     An on-site S+ or S- (transverse field) breaks conservation.
 *
 * @return true if the Hamiltonian conserves total Sz
 */
inline bool hamiltonian_conserves_sz(const std::string& interaction_file,
                                     const std::string& single_site_file) {
    // --- Check two-body terms (InterAll.dat) ---
    if (!interaction_file.empty()) {
        std::ifstream file(interaction_file);
        if (file.is_open()) {
            std::string line;
            // Skip header: 3 separator/num lines + 3 more lines (matches loadFromInterAllFile)
            for (int i = 0; i < 5; ++i) std::getline(file, line);

            while (std::getline(file, line)) {
                std::istringstream iss(line);
                uint64_t op_i, site_i, op_j, site_j;
                double re, im;
                if (!(iss >> op_i >> site_i >> op_j >> site_j >> re >> im)) continue;
                if (std::abs(re) < 1e-15 && std::abs(im) < 1e-15) continue;

                // Allowed Sz-conserving pairs: (S+,S-)=(0,1), (S-,S+)=(1,0), (Sz,Sz)=(2,2)
                bool ok = (op_i == 0 && op_j == 1) ||
                          (op_i == 1 && op_j == 0) ||
                          (op_i == 2 && op_j == 2);
                if (!ok) {
                    std::cout << "[Sz check] Non-conserving two-body term: Op("
                              << op_i << "," << op_j << ") on sites ("
                              << site_i << "," << site_j << ") with coeff ("
                              << re << "," << im << ")" << std::endl;
                    return false;
                }
            }
        }
    }

    // --- Check single-site terms (Trans.dat) ---
    if (!single_site_file.empty()) {
        std::ifstream file(single_site_file);
        if (file.is_open()) {
            std::string line;
            // Skip header (matches loadFromFile): 1 separator + "num N" + 3 separator lines
            for (int i = 0; i < 5; ++i) std::getline(file, line);

            while (std::getline(file, line)) {
                std::istringstream iss(line);
                uint64_t op, site;
                double re, im;
                if (!(iss >> op >> site >> re >> im)) continue;
                if (std::abs(re) < 1e-15 && std::abs(im) < 1e-15) continue;

                // Only Sz (op=2) preserves total Sz
                if (op != 2) {
                    std::cout << "[Sz check] Non-conserving single-site term: Op("
                              << op << ") on site " << site << " with coeff ("
                              << re << "," << im << ")" << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

// ============================================================================
// FULL DIAGONALIZATION VIA Sz SECTOR SPLITTING
// ============================================================================

/**
 * @brief Full diagonalization by splitting into Sz sectors
 * 
 * Instead of diagonalizing the full 2^N Hilbert space at once, this function:
 * 1. Loops over all Sz sectors (n_up = 0, 1, ..., num_sites)
 * 2. Builds a FixedSzOperator per sector
 * 3. Constructs the dense Hamiltonian using the sparse-to-dense conversion
 *    (via buildFixedSzMatrix() + toDense()), which is far more efficient than
 *    N column-by-column SpMV applications
 * 4. Diagonalizes each sector independently with LAPACKE_zheevd
 * 5. Collects and sorts all eigenvalues
 * 6. Saves the merged result as a single HDF5 file (transparent to downstream code)
 * 
 * Memory advantage: The largest sector has dimension C(N, N/2), e.g.:
 *   N=15: 2^15 = 32768 -> C(15,7) = 6435  (5.1x reduction, 26x less memory)
 *   N=17: 2^17 = 131072 -> C(17,8) = 24310 (5.4x reduction, 29x less memory)
 *   N=19: 2^19 = 524288 -> C(19,9) = 92378 (5.7x reduction, 32x less memory)
 * 
 * IMPORTANT: This assumes Sz is a good quantum number (total Sz commutes with H).
 * Models that break Sz conservation (e.g., anisotropic J±±, Jz±, in-plane field)
 * will produce INCORRECT results. The caller must verify Sz conservation.
 * 
 * @param interaction_file Path to interaction file (InterAll.dat)
 * @param single_site_file Path to single-site file (Trans.dat)
 * @param num_sites Number of spin sites
 * @param spin_length Spin length (usually 0.5)
 * @param params Parameters for diagonalization
 * @return EDResults containing ALL eigenvalues from ALL sectors, sorted
 */
inline EDResults exact_diagonalization_all_sz_sectors(
    const std::string& interaction_file,
    const std::string& single_site_file,
    uint64_t num_sites,
    float spin_length,
    const EDParameters& params
) {
    EDResults results;
    
    uint64_t full_dim = 1ULL << num_sites;
    uint64_t num_sectors = num_sites + 1;  // n_up = 0, 1, ..., num_sites
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "FULL DIAGONALIZATION VIA Sz SECTOR SPLITTING" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Full Hilbert space dimension: " << full_dim << std::endl;
    std::cout << "Number of Sz sectors: " << num_sectors << std::endl;
    
    // Pre-compute sector dimensions for progress reporting
    std::vector<uint64_t> sector_dims(num_sectors);
    uint64_t largest_sector_dim = 0;
    uint64_t total_sector_dim = 0;
    for (uint64_t n_up = 0; n_up <= num_sites; ++n_up) {
        auto basis = generateFixedSzBasis(num_sites, n_up);
        sector_dims[n_up] = basis.size();
        total_sector_dim += sector_dims[n_up];
        if (sector_dims[n_up] > largest_sector_dim) {
            largest_sector_dim = sector_dims[n_up];
        }
    }
    
    std::cout << "Largest sector dimension: " << largest_sector_dim 
              << " (reduction: " << std::fixed << std::setprecision(1) 
              << (double)full_dim / largest_sector_dim << "x)" 
              << std::defaultfloat << std::endl;
    std::cout << "Memory for full matrix:    " 
              << (double)full_dim * full_dim * sizeof(Complex) / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    std::cout << "Memory for largest sector: " 
              << (double)largest_sector_dim * largest_sector_dim * sizeof(Complex) / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    
    // Collect all eigenvalues from all sectors
    std::vector<double> all_eigenvalues;
    all_eigenvalues.reserve(full_dim);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (uint64_t n_up = 0; n_up <= num_sites; ++n_up) {
        uint64_t dim = sector_dims[n_up];
        double sz_value = spin_length * num_sites - n_up;
        
        std::cout << "\n--- Sector " << n_up + 1 << "/" << num_sectors 
                  << ": n_up=" << n_up << ", Sz=" << sz_value
                  << ", dim=" << dim << " ---" << std::endl;
        
        if (dim == 0) {
            std::cout << "  Empty sector, skipping." << std::endl;
            continue;
        }
        
        auto sector_start = std::chrono::high_resolution_clock::now();
        
        // Create FixedSzOperator for this sector
        FixedSzOperator op(num_sites, spin_length, n_up);
        
        // Load Hamiltonian terms
        if (!single_site_file.empty()) {
            std::ifstream file(single_site_file);
            if (file.is_open()) {
                op.loadFromFile(single_site_file);
            }
        }
        if (!interaction_file.empty()) {
            std::ifstream file(interaction_file);
            if (file.is_open()) {
                op.loadFromInterAllFile(interaction_file);
            }
        }
        
        // Build the sparse matrix, then convert to dense for LAPACK
        // This is O(nnz) — much faster than dim SpMV calls which is O(dim * nnz)
        op.buildFixedSzMatrix();
        auto sparse_matrix = op.getFixedSzMatrix();
        
        // Convert sparse -> dense (column-major for LAPACK)
        Eigen::MatrixXcd dense = Eigen::MatrixXcd(sparse_matrix);
        
        // Eigenvalue array
        std::vector<double> evals(dim);
        
        // Call LAPACKE_zheevd (divide-and-conquer) — eigenvalues only, no eigenvectors needed
        lapack_int info = LAPACKE_zheevd(
            LAPACK_COL_MAJOR,
            params.compute_eigenvectors ? 'V' : 'N',   // Eigenvalues only for NLCE thermodynamics
            'U',                         // Upper triangle
            dim,                         // Matrix dimension
            reinterpret_cast<lapack_complex_double*>(dense.data()),
            dim,                         // Leading dimension
            evals.data()                 // Output eigenvalues
        );
        
        if (info != 0) {
            std::cerr << "  LAPACKE_zheevd failed for sector n_up=" << n_up 
                      << " with error code " << info << std::endl;
            continue;
        }
        
        // Append eigenvalues to the combined list
        all_eigenvalues.insert(all_eigenvalues.end(), evals.begin(), evals.end());
        
        auto sector_end = std::chrono::high_resolution_clock::now();
        double sector_time = std::chrono::duration<double>(sector_end - sector_start).count();
        
        std::cout << "  Diagonalized in " << std::fixed << std::setprecision(2) 
                  << sector_time << " s, found " << dim << " eigenvalues"
                  << " [" << evals.front() << ", " << evals.back() << "]" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    // Sort all eigenvalues globally
    std::sort(all_eigenvalues.begin(), all_eigenvalues.end());
    
    // Store results
    if (params.num_eigenvalues > 0 && params.num_eigenvalues < all_eigenvalues.size()) {
        results.eigenvalues.assign(all_eigenvalues.begin(), 
                                   all_eigenvalues.begin() + params.num_eigenvalues);
    } else {
        results.eigenvalues = std::move(all_eigenvalues);
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Sz-SPLIT FULL DIAG COMPLETE" << std::endl;
    std::cout << "Total eigenvalues: " << results.eigenvalues.size() << std::endl;
    std::cout << "Total wall time: " << std::fixed << std::setprecision(2) << total_time << " s" << std::endl;
    if (!results.eigenvalues.empty()) {
        std::cout << "Ground state energy: " << results.eigenvalues[0] << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
    
    // Save results to HDF5
    if (!params.output_dir.empty()) {
        safe_system_call("mkdir -p " + params.output_dir);
        
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
            std::cout << "Saved " << results.eigenvalues.size() 
                      << " eigenvalues to " << hdf5_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save to HDF5: " << e.what() << std::endl;
            // Fallback to text file
            std::string txt_file = params.output_dir + "/eigenvalues.txt";
            std::ofstream ofs(txt_file);
            if (ofs.is_open()) {
                ofs << std::setprecision(15);
                for (double ev : results.eigenvalues) {
                    ofs << ev << "\n";
                }
                std::cout << "Saved eigenvalues to " << txt_file << std::endl;
            }
        }
    }
    
    return results;
}

// ============================================================================
// GPU FULL DIAGONALIZATION VIA Sz SECTOR SPLITTING
// ============================================================================

/**
 * @brief GPU full diagonalization by splitting into Sz sectors
 * 
 * GPU equivalent of exact_diagonalization_all_sz_sectors(). Instead of building
 * a CPU sparse matrix per sector, this function:
 * 1. Parses interaction files once into vectors
 * 2. Loops over all Sz sectors (n_up = 0, 1, ..., num_sites)
 * 3. Creates a GPUFixedSzOperator per sector
 * 4. Calls gpuFullDiagonalization (builds dense H via matVec, cuSOLVER zheevd)
 * 5. Collects and sorts all eigenvalues
 * 6. Saves the merged result to HDF5
 * 
 * Same memory advantage as CPU sector splitting - the largest sector is C(N,N/2).
 * Each sector's GPU operator is created and destroyed independently, so only one
 * sector's dense matrix is in GPU memory at a time.
 * 
 * @param interaction_file Path to interaction file (InterAll.dat)
 * @param single_site_file Path to single-site file (Trans.dat)
 * @param num_sites Number of spin sites
 * @param spin_length Spin length (usually 0.5)
 * @param params Parameters for diagonalization
 * @return EDResults containing ALL eigenvalues from ALL sectors, sorted
 */
#ifdef WITH_CUDA
inline EDResults exact_diagonalization_all_sz_sectors_gpu(
    const std::string& interaction_file,
    const std::string& single_site_file,
    uint64_t num_sites,
    float spin_length,
    const EDParameters& params
) {
    EDResults results;
    
    uint64_t full_dim = 1ULL << num_sites;
    uint64_t num_sectors = num_sites + 1;  // n_up = 0, 1, ..., num_sites
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "GPU FULL DIAGONALIZATION VIA Sz SECTOR SPLITTING" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Full Hilbert space dimension: " << full_dim << std::endl;
    std::cout << "Number of Sz sectors: " << num_sectors << std::endl;
    
    // Pre-compute sector dimensions using binomial coefficients
    auto binomial = [](uint64_t n, uint64_t k) -> uint64_t {
        if (k > n - k) k = n - k;
        uint64_t result = 1;
        for (uint64_t i = 0; i < k; ++i) {
            result *= (n - i);
            result /= (i + 1);
        }
        return result;
    };
    
    std::vector<uint64_t> sector_dims(num_sectors);
    uint64_t largest_sector_dim = 0;
    for (uint64_t n_up = 0; n_up <= num_sites; ++n_up) {
        sector_dims[n_up] = binomial(num_sites, n_up);
        if (sector_dims[n_up] > largest_sector_dim) {
            largest_sector_dim = sector_dims[n_up];
        }
    }
    
    std::cout << "Largest sector dimension: " << largest_sector_dim 
              << " (reduction: " << std::fixed << std::setprecision(1) 
              << (double)full_dim / largest_sector_dim << "x)" 
              << std::defaultfloat << std::endl;
    std::cout << "Memory for full dense matrix:    " 
              << (double)full_dim * full_dim * 16.0 / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    std::cout << "Memory for largest sector:       " 
              << (double)largest_sector_dim * largest_sector_dim * 16.0 / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    
    // Parse interaction files ONCE into vectors for reuse across sectors
    std::vector<std::tuple<int, int, char, char, double>> gpu_interactions;
    std::vector<std::tuple<int, char, double>> gpu_single_site_ops;
    
    {
        std::ifstream inter_file(interaction_file);
        if (inter_file.is_open()) {
            std::string line;
            std::getline(inter_file, line);
            std::getline(inter_file, line);
            std::istringstream iss(line);
            uint64_t numLines;
            std::string m;
            iss >> m >> numLines;
            for (uint64_t i = 0; i < 3; ++i) std::getline(inter_file, line);
            
            auto mapOp = [](uint64_t op) -> char {
                if (op == 0) return '+';
                if (op == 1) return '-';
                return 'z';
            };
            
            uint64_t lineCount = 0;
            while (std::getline(inter_file, line) && lineCount < numLines) {
                std::istringstream lineStream(line);
                uint64_t Op_i, indx_i, Op_j, indx_j;
                double E, F;
                if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) { lineCount++; continue; }
                gpu_interactions.push_back(std::make_tuple(indx_i, indx_j, mapOp(Op_i), mapOp(Op_j), E));
                lineCount++;
            }
        }
        
        if (!single_site_file.empty()) {
            std::ifstream ss_file(single_site_file);
            if (ss_file.is_open()) {
                std::string line;
                std::getline(ss_file, line);
                std::getline(ss_file, line);
                std::istringstream iss(line);
                uint64_t numLines;
                std::string m;
                iss >> m >> numLines;
                for (uint64_t i = 0; i < 3; ++i) std::getline(ss_file, line);
                
                auto mapOp = [](uint64_t op) -> char {
                    if (op == 0) return '+';
                    if (op == 1) return '-';
                    return 'z';
                };
                
                uint64_t lineCount = 0;
                while (std::getline(ss_file, line) && lineCount < numLines) {
                    std::istringstream lineStream(line);
                    uint64_t Op_i, indx_i;
                    double E, F;
                    if (!(lineStream >> Op_i >> indx_i >> E >> F)) { lineCount++; continue; }
                    gpu_single_site_ops.push_back(std::make_tuple(indx_i, mapOp(Op_i), E));
                    lineCount++;
                }
            }
        }
    }
    
    std::cout << "Loaded " << gpu_interactions.size() << " interactions, "
              << gpu_single_site_ops.size() << " single-site terms" << std::endl;
    
    // Collect all eigenvalues from all sectors
    std::vector<double> all_eigenvalues;
    all_eigenvalues.reserve(full_dim);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (uint64_t n_up = 0; n_up <= num_sites; ++n_up) {
        uint64_t dim = sector_dims[n_up];
        double sz_value = spin_length * num_sites - n_up;
        
        std::cout << "\n--- Sector " << n_up + 1 << "/" << num_sectors 
                  << ": n_up=" << n_up << ", Sz=" << sz_value
                  << ", dim=" << dim << " ---" << std::endl;
        
        if (dim == 0) {
            std::cout << "  Empty sector, skipping." << std::endl;
            continue;
        }
        
        auto sector_start = std::chrono::high_resolution_clock::now();
        
        // Create GPU Fixed Sz operator for this sector
        void* gpu_op_handle = GPUEDWrapper::createGPUFixedSzOperatorDirect(
            num_sites, n_up, spin_length,
            gpu_interactions, gpu_single_site_ops);
        
        if (!gpu_op_handle) {
            std::cerr << "  Error: Failed to create GPU operator for sector n_up=" << n_up << std::endl;
            continue;
        }
        
        // Run GPU full diag on this sector (no HDF5 save per sector)
        std::vector<double> sector_eigenvalues;
        GPUEDWrapper::runGPUFullDiag(
            gpu_op_handle,
            static_cast<int>(dim),
            static_cast<int>(dim),  // Get ALL eigenvalues in each sector
            sector_eigenvalues,
            "",   // Empty dir = don't save per-sector HDF5
            false // No eigenvectors needed for thermodynamics
        );
        
        // Destroy GPU operator for this sector (frees GPU memory)
        GPUEDWrapper::destroyGPUOperator(gpu_op_handle);
        
        // Append eigenvalues
        all_eigenvalues.insert(all_eigenvalues.end(), 
                                sector_eigenvalues.begin(), sector_eigenvalues.end());
        
        auto sector_end = std::chrono::high_resolution_clock::now();
        double sector_time = std::chrono::duration<double>(sector_end - sector_start).count();
        
        if (!sector_eigenvalues.empty()) {
            std::cout << "  GPU diag: " << std::fixed << std::setprecision(2) 
                      << sector_time << " s, " << sector_eigenvalues.size() << " eigenvalues"
                      << " [" << sector_eigenvalues.front() << ", " << sector_eigenvalues.back() << "]" 
                      << std::defaultfloat << std::endl;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    // Sort all eigenvalues globally
    std::sort(all_eigenvalues.begin(), all_eigenvalues.end());
    
    // Store results
    if (params.num_eigenvalues > 0 && static_cast<size_t>(params.num_eigenvalues) < all_eigenvalues.size()) {
        results.eigenvalues.assign(all_eigenvalues.begin(), 
                                   all_eigenvalues.begin() + params.num_eigenvalues);
    } else {
        results.eigenvalues = std::move(all_eigenvalues);
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "GPU Sz-SPLIT FULL DIAG COMPLETE" << std::endl;
    std::cout << "Total eigenvalues: " << results.eigenvalues.size() << std::endl;
    std::cout << "Total wall time: " << std::fixed << std::setprecision(2) << total_time << " s" << std::endl;
    if (!results.eigenvalues.empty()) {
        std::cout << "Ground state energy: " << results.eigenvalues[0] << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
    
    // Save results to HDF5
    if (!params.output_dir.empty()) {
        safe_system_call("mkdir -p " + params.output_dir);
        
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
            std::cout << "Saved " << results.eigenvalues.size() 
                      << " eigenvalues to " << hdf5_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save to HDF5: " << e.what() << std::endl;
        }
    }
    
    return results;
}
#endif // WITH_CUDA


// ============================================================================
// FIXED SZ EXACT DIAGONALIZATION
// ============================================================================

/**
 * @brief Exact diagonalization in fixed Sz sector
 * 
 * Performs diagonalization on a restricted Hilbert space with fixed
 * total Sz quantum number, significantly reducing memory and computational cost.
 * 
 * @param interaction_file Path to interaction file
 * @param single_site_file Path to single-site file
 * @param num_sites Number of sites
 * @param spin_length Spin length (usually 0.5)
 * @param n_up Number of up spins (determines Sz sector)
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_fixed_sz(
    const std::string& interaction_file,
    const std::string& single_site_file,
    uint64_t num_sites,
    float spin_length,
    int64_t n_up,
    DiagonalizationMethod method,
    const EDParameters& params_in
) {
    // Phase 7: idempotent canonicalization (also covers callers that
    // skip exact_diagonalization_from_files and jump straight to here,
    // e.g. workflows.cpp::run_standard_workflow when --fixed-sz is set
    // in the config).
    EDParameters params = params_in;
    {
        const auto canon = ed::canonicalize_method_and_flags(
            method, params.use_fixed_sz, params.use_gpu, params.use_mpi);
        params.use_fixed_sz = canon.use_fixed_sz;
        params.use_gpu      = canon.use_gpu;
        params.use_mpi      = canon.use_mpi;
        method = ed::legacy_method_for_dispatch(canon.method, canon.use_gpu);
    }

    FixedSzOperator hamiltonian(num_sites, spin_length, n_up);
    
    // Load Hamiltonian terms
    if (!single_site_file.empty()) {
        std::ifstream file(single_site_file);
        if (file.is_open()) {
            hamiltonian.loadFromFile(single_site_file);
        }
    }
    if (!interaction_file.empty()) {
        std::ifstream file(interaction_file);
        if (file.is_open()) {
            hamiltonian.loadFromInterAllFile(interaction_file);
        }
    }
    
    // Get dimension of fixed Sz sector
    uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
    uint64_t full_dim = 1ULL << num_sites;
    
    std::cout << "Fixed Sz basis: dim=" << fixed_sz_dim 
              << " (reduction: " << std::fixed << std::setprecision(1) 
              << (double)full_dim / fixed_sz_dim << "x)" << std::defaultfloat << std::endl;
    
    // GPU dispatch is gated solely on EDParameters::use_gpu now.
    const bool is_gpu_method = params.use_gpu;
    
    EDResults results;
    
    if (is_gpu_method) {
#ifdef WITH_CUDA
        // Prepare interactions and single-site operators
        std::vector<std::tuple<int, int, char, char, double>> gpu_interactions;
        std::vector<std::tuple<int, char, double>> gpu_single_site_ops;
        
        // Load from files
        std::ifstream inter_file(interaction_file);
        if (inter_file.is_open()) {
            std::string line;
            std::getline(inter_file, line);
            std::getline(inter_file, line);
            std::istringstream iss(line);
            uint64_t numLines;
            std::string m;
            iss >> m >> numLines;
            
            for (uint64_t i = 0; i < 3; ++i) std::getline(inter_file, line);
            
            uint64_t lineCount = 0;
            while (std::getline(inter_file, line) && lineCount < numLines) {
                std::istringstream lineStream(line);
                uint64_t Op_i, indx_i, Op_j, indx_j;
                double E, F;
                
                if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) continue;
                
                // File operator codes: 0=S+, 1=S-, 2=Sz
                // Map to chars: '+'=S+, '-'=S-, 'z'=Sz
                auto mapOp = [](uint64_t op) -> char {
                    if (op == 0) return '+';  // S+
                    if (op == 1) return '-';  // S-
                    return 'z';  // Sz
                };
                
                gpu_interactions.push_back(std::make_tuple(indx_i, indx_j, mapOp(Op_i), mapOp(Op_j), E));
                lineCount++;
            }
        }
        
        // Load single-site terms if present
        if (!single_site_file.empty()) {
            std::ifstream ss_file(single_site_file);
            if (ss_file.is_open()) {
                std::string line;
                std::getline(ss_file, line);
                std::getline(ss_file, line);
                std::istringstream iss(line);
                uint64_t numLines;
                std::string m;
                iss >> m >> numLines;
                
                for (uint64_t i = 0; i < 3; ++i) std::getline(ss_file, line);
                
                uint64_t lineCount = 0;
                while (std::getline(ss_file, line) && lineCount < numLines) {
                    std::istringstream lineStream(line);
                    uint64_t Op_i, indx_i;
                    double E, F;
                    
                    if (!(lineStream >> Op_i >> indx_i >> E >> F)) continue;
                    
                    // File operator codes: 0=S+, 1=S-, 2=Sz
                    // Map to chars: '+'=S+, '-'=S-, 'z'=Sz
                    auto mapOp = [](uint64_t op) -> char {
                        if (op == 0) return '+';  // S+
                        if (op == 1) return '-';  // S-
                        return 'z';  // Sz
                    };
                    
                    gpu_single_site_ops.push_back(std::make_tuple(indx_i, mapOp(Op_i), E));
                    lineCount++;
                }
            }
        }
        
        // Create GPU operator
        void* gpu_op_handle = GPUEDWrapper::createGPUFixedSzOperatorDirect(
            num_sites, n_up, spin_length,
            gpu_interactions, gpu_single_site_ops);
        
        std::cout << "Loaded " << gpu_interactions.size() << " interactions and " 
                  << gpu_single_site_ops.size() << " single-site terms\n";
        
        // Run appropriate GPU method
        std::vector<double> eigenvalues;
        
        if (method == DiagonalizationMethod::LANCZOS) {
            GPUEDWrapper::runGPULanczosFixedSz(
                gpu_op_handle, n_up,
                params.max_iterations,
                params.num_eigenvalues,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors,
                params.lanczos_seed);
        } else if (method == DiagonalizationMethod::mTPQ) {
            GPUEDWrapper::runGPUMicrocanonicalTPQFixedSz(
                gpu_op_handle, n_up,
                params.max_iterations,
                params.num_samples,
                params.tpq_measurement_interval,
                eigenvalues,
                params.output_dir,
                params.tpq_energy_shift,
                params.tpq_continue,
                params.tpq_continue_sample,
                params.tpq_continue_beta,
                params.save_thermal_states,
                params.tpq_target_beta,
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max);
        } else if (method == DiagonalizationMethod::cTPQ) {
            GPUEDWrapper::runGPUCanonicalTPQFixedSz(
                gpu_op_handle, n_up,
                params.temp_max,  // beta_max
                params.num_samples,
                params.tpq_measurement_interval,
                eigenvalues,
                params.output_dir,
                params.tpq_delta_beta,  // delta_beta
                params.tpq_taylor_order,  // taylor_order
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max);
        } else if (method == DiagonalizationMethod::BLOCK_LANCZOS) {
            GPUEDWrapper::runGPUBlockLanczosFixedSz(
                gpu_op_handle, n_up,
                params.max_iterations,
                params.num_eigenvalues,
                params.block_size,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::KRYLOV_SCHUR) {
            GPUEDWrapper::runGPUKrylovSchurFixedSz(
                gpu_op_handle, n_up,
                params.num_eigenvalues,
                params.max_iterations,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::FULL) {
            GPUEDWrapper::runGPUFullDiag(
                gpu_op_handle,
                fixed_sz_dim,
                params.num_eigenvalues,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else {
            throw std::runtime_error(
                "exact_diagonalization_fixed_sz: GPU dispatch only "
                "supports LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR / "
                "FULL / mTPQ / cTPQ at this revision.");
        }

        results.eigenvalues = eigenvalues;

        if (method == DiagonalizationMethod::mTPQ || method == DiagonalizationMethod::cTPQ) {
            std::cout << "\nGPU TPQ states were automatically transformed to full Hilbert space during save." << std::endl;
            
            // MPI-safe HDF5 merge: merge per-rank files on rank 0
#ifdef WITH_MPI
            int mpi_size, mpi_rank;
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            
            if (mpi_size > 1) {
                MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks have finished writing
            }
            if (mpi_rank == 0) {
                std::cout << "\nMerging per-rank HDF5 files..." << std::endl;
                HDF5IO::mergePerRankTPQFiles(params.output_dir, mpi_size, "ed_results.h5", true);
                // Convert TPQ results to unified thermodynamic format
                convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
            }
#else
            // Non-MPI: convert TPQ results to unified thermodynamic format
            convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
#endif
        }
        
        // Cleanup
        GPUEDWrapper::destroyGPUOperator(gpu_op_handle);
        
        std::cout << "GPU diagonalization complete\n";
#else
        throw std::runtime_error("GPU methods require CUDA support (compile with -DWITH_CUDA=ON)");
#endif
    } else {

        
        // Create apply function
        auto apply_hamiltonian = [&hamiltonian, fixed_sz_dim](const Complex* in, Complex* out, uint64_t n) {
            if (n != fixed_sz_dim) {
                throw std::runtime_error("Dimension mismatch in fixed Sz apply");
            }
            hamiltonian.apply(in, out, n);
        };
        
        // Set the fixed_sz_op in params so TPQ can transform states before saving
        params.fixed_sz_op = &hamiltonian;
        
        // Perform diagonalization
        std::cout << "\nDiagonalizing..." << std::endl;
        results = exact_diagonalization_core(apply_hamiltonian, fixed_sz_dim, method, params);
    }

    // Check if this is a TPQ method
    bool is_tpq_method = (method == DiagonalizationMethod::mTPQ ||
                          method == DiagonalizationMethod::cTPQ);

    // Transform eigenvectors from fixed-Sz basis to full basis
    size_t n_eigs = results.eigenvalues.size();
    if (!params.output_dir.empty() && params.compute_eigenvectors && n_eigs > 0) {
        // Transform eigenvectors if computed - load from HDF5, transform, save back
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            
            for (size_t i = 0; i < n_eigs; ++i) {
                // Load eigenvector from HDF5 (in fixed-Sz basis)
                std::vector<Complex> fixed_sz_vec = HDF5IO::loadEigenvector(hdf5_file, i);
                
                if (fixed_sz_vec.size() == fixed_sz_dim) {
                    // Transform to full basis
                    std::vector<Complex> full_vec = hamiltonian.embedToFull(fixed_sz_vec);
                    
                    // Save back to HDF5 with full-space eigenvector
                    HDF5IO::saveEigenvector(hdf5_file, i, full_vec);
                }
            }
            std::cout << "Transformed " << n_eigs << " eigenvectors to full space" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to transform eigenvectors: " << e.what() << std::endl;
        }
    }

    return results;
}

// ============================================================================
// FILE-BASED WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Wrapper function to perform exact diagonalization from Hamiltonian files
 * 
 * Loads Hamiltonian from input files and performs diagonalization.
 * 
 * @param interaction_file Path to interaction file (e.g., InterAll.dat)
 * @param single_site_file Path to single-site file (e.g., Trans.dat)
 * @param counterterm_file Path to counter term file (optional)
 * @param three_body_file Path to three-body interaction file (e.g., ThreeBodyG.dat, optional)
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @param format File format for Hamiltonian
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    const std::string& counterterm_file = "",
    const std::string& three_body_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params_in = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD
) {
    if (ed_log::isVerbose()) {
        std::ostringstream _dbg;
        _dbg << "exact_diagonalization_from_files: num_sites=" << params_in.num_sites
             << ", method=" << static_cast<int>(method);
        ed_log::debug(_dbg.str());
    }

    // ========== Phase 7: Single canonicalization point ==========
    // Collapse legacy _FIXED_SZ / _GPU / _CUDA / _MPI enum variants onto
    // the canonical (base_method, use_fixed_sz, use_gpu, use_mpi) tuple.
    // This is the only place where the deprecated variants are observed;
    // every dispatch branch below sees flags only.
    EDParameters params = params_in;
    {
        const auto canon = ed::canonicalize_method_and_flags(
            method, params.use_fixed_sz, params.use_gpu, params.use_mpi);
        params.use_fixed_sz = canon.use_fixed_sz;
        params.use_gpu      = canon.use_gpu;
        params.use_mpi      = canon.use_mpi;
        method = ed::legacy_method_for_dispatch(canon.method, canon.use_gpu);
    }

    bool use_fixed_sz = params.use_fixed_sz;

    // ========== Phase 7.1: Symmetry routing (5th orthogonal axis) ==========
    // When `use_symmetry == true` the canonical dispatch is to route to
    // the streaming symmetry kernel in ed_wrapper_streaming.h. We CANNOT
    // call it from this header because ed_wrapper_streaming.h includes
    // ed_wrapper.h (circular). Instead, the routing happens in the
    // higher-level dispatcher seam:
    //   * Python:  ed_dispatch::exact_diagonalization_from_directory(...)
    //              in ed_dispatch_symmetry.h (included by the Python
    //              binding, which sees both headers).
    //   * CLI:     ed_main.cpp routes via run_streaming_symmetry_workflow
    //              when WorkflowConfig::run_symm_auto is set.
    //   * C++:     direct callers of from_files() that set use_symmetry=true
    //              get a hard error here so the bug is loud, not silent.
    if (params.use_symmetry) {
        throw std::runtime_error(
            "EDParameters::use_symmetry=true is not supported by "
            "exact_diagonalization_from_files() directly. Call "
            "ed_dispatch::exact_diagonalization_from_directory() "
            "(declared in ed/core/ed_dispatch_symmetry.h) or invoke "
            "exact_diagonalization_streaming_symmetry[_fixed_sz]() "
            "explicitly. See docs/history/PHASE_7_SYMMETRY_AXIS.md.");
    }

    // (Method/fixed-Sz compatibility check retired May 2026 -- every
    // remaining method supports fixed-Sz; see the note next to the
    // former `supports_fixed_sz` helper.)

    // Route to fixed-Sz function if use_fixed_sz is true
    // This ensures all fixed-Sz GPU logic is handled in one place
    if (use_fixed_sz) {
        int64_t n_up = (params.n_up >= 0) ? params.n_up : params.num_sites / 2;
        return exact_diagonalization_fixed_sz(
            interaction_file,
            single_site_file,
            params.num_sites,
            params.spin_length,
            n_up,
            method,
            params
        );
    }
    
    // ========== Full Sz-Sector Split ==========
    // Auto-enable Sz sector splitting for FULL diag when the Hamiltonian
    // conserves total Sz.  This is always beneficial: it reduces the largest
    // dense block from 2^N to C(N,N/2) (e.g. 29x less memory at N=17).
    // The flag --full-sz-split can force it on (caller takes responsibility)
    // or it is auto-detected by scanning InterAll.dat / Trans.dat.
    if (method == DiagonalizationMethod::FULL) {
        
        bool use_sz_split = params.full_sz_split;  // Explicitly requested?
        
        if (!use_sz_split) {
            // Auto-detect: check if Hamiltonian conserves Sz
            bool sz_conserved = hamiltonian_conserves_sz(interaction_file, single_site_file);
            if (sz_conserved) {
                std::cout << "[Auto] Hamiltonian conserves Sz — enabling sector splitting for full diag" << std::endl;
                use_sz_split = true;
            }
        }
        
        if (use_sz_split) {
            std::cout << "Using Sz-sector splitting for full diagonalization" << std::endl;
            return exact_diagonalization_all_sz_sectors(
                interaction_file,
                single_site_file,
                params.num_sites,
                params.spin_length,
                params
            );
        }
    }
    
    // ========== GPU FULL Sz-Sector Split ==========
    // Same auto-detection for GPU full diag: split into Sz sectors on GPU
#ifdef WITH_CUDA
    if (params.use_gpu && method == DiagonalizationMethod::FULL) {
        bool use_sz_split = params.full_sz_split;

        if (!use_sz_split) {
            bool sz_conserved = hamiltonian_conserves_sz(interaction_file, single_site_file);
            if (sz_conserved) {
                std::cout << "[Auto] Hamiltonian conserves Sz — enabling GPU sector splitting for full diag" << std::endl;
                use_sz_split = true;
            }
        }

        if (use_sz_split) {
            std::cout << "Using GPU Sz-sector splitting for full diagonalization" << std::endl;
            return exact_diagonalization_all_sz_sectors_gpu(
                interaction_file,
                single_site_file,
                params.num_sites,
                params.spin_length,
                params
            );
        }
    }
#endif

    // Handle GPU methods separately (they don't need CPU Operator)
#ifdef WITH_CUDA
    if (params.use_gpu) {
        
        std::cout << "Running GPU-accelerated algorithm (full Hilbert space)..." << std::endl;
        
        // Check if GPU is available
        if (!GPUEDWrapper::isGPUAvailable()) {
            std::cerr << "Error: No CUDA-capable GPU found!" << std::endl;
            throw std::runtime_error("GPU not available");
        }
        
        GPUEDWrapper::printGPUInfo();
        
        EDResults results;
        uint64_t hilbert_space_dim = static_cast<int>(1ULL << params.num_sites);
        
        if (method == DiagonalizationMethod::LANCZOS) {
            if (!std::filesystem::exists(interaction_file)) {
                std::cerr << "Error: " << interaction_file << " not found!" << std::endl;
                throw std::runtime_error("InterAll.dat file not found");
            }

            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);

            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }

            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPULanczos(
                gpu_op,
                hilbert_space_dim,
                params.max_iterations,
                params.num_eigenvalues,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors,
                params.lanczos_seed
            );

            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            std::cout << "GPU Lanczos completed successfully!" << std::endl;

        } else if (method == DiagonalizationMethod::mTPQ) {
            std::cout << "Running GPU microcanonical TPQ..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUMicrocanonicalTPQ(
                gpu_op,
                hilbert_space_dim,
                params.max_iterations,
                params.num_samples,
                params.tpq_measurement_interval,
                eigenvalues,
                params.output_dir,
                params.tpq_energy_shift,
                params.tpq_continue,
                params.tpq_continue_sample,
                params.tpq_continue_beta,
                params.save_thermal_states,
                params.tpq_target_beta,
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            // MPI-safe HDF5 merge: merge per-rank files on rank 0
#ifdef WITH_MPI
            int mpi_size, mpi_rank;
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            
            if (mpi_size > 1) {
                MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks have finished writing
            }
            if (mpi_rank == 0) {
                std::cout << "\nMerging per-rank HDF5 files..." << std::endl;
                HDF5IO::mergePerRankTPQFiles(params.output_dir, mpi_size, "ed_results.h5", true);
                // Convert TPQ results to unified thermodynamic format
                convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
            }
#else
            // Non-MPI: convert TPQ results to unified thermodynamic format
            convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
#endif
            
            std::cout << "GPU mTPQ completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::cTPQ) {
            std::cout << "Running GPU canonical TPQ..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUCanonicalTPQ(
                gpu_op,
                hilbert_space_dim,
                params.temp_max,
                params.num_samples,
                params.tpq_measurement_interval,
                eigenvalues,
                params.output_dir,
                params.tpq_delta_beta,
                params.tpq_taylor_order,
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            // MPI-safe HDF5 merge: merge per-rank files on rank 0
#ifdef WITH_MPI
            {
                int mpi_size, mpi_rank;
                MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                
                if (mpi_size > 1) {
                    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks have finished writing
                }
                if (mpi_rank == 0) {
                    std::cout << "\nMerging per-rank HDF5 files..." << std::endl;
                    HDF5IO::mergePerRankTPQFiles(params.output_dir, mpi_size, "ed_results.h5", true);
                    // Convert TPQ results to unified thermodynamic format
                    convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
                }
            }
#else
            // Non-MPI: convert TPQ results to unified thermodynamic format
            convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
#endif
            
            std::cout << "GPU cTPQ completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::FTLM) {
            std::cout << "Running GPU Finite Temperature Lanczos Method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            GPUEDWrapper::runGPUFTLM(
                gpu_op,
                hilbert_space_dim,
                params.ftlm_krylov_dim,
                params.num_samples,
                params.temp_min,
                params.temp_max,
                params.num_temp_bins,
                params.tolerance,
                params.output_dir,
                params.ftlm_full_reorth,
                params.ftlm_reorth_freq,
                params.ftlm_seed
            );
            
            // FTLM doesn't return eigenvalues in the traditional sense
            // Results are thermodynamic quantities written to files
            
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU FTLM completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::BLOCK_LANCZOS) {
            std::cout << "Running GPU Block Lanczos method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUBlockLanczos(
                gpu_op,
                hilbert_space_dim,
                params.max_iterations,
                params.num_eigenvalues,
                params.block_size,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Block Lanczos completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::KRYLOV_SCHUR) {
            std::cout << "Running GPU Krylov-Schur method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUKrylovSchur(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                params.max_iterations,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Krylov-Schur completed successfully!" << std::endl;
        } else if (method == DiagonalizationMethod::FULL) {
            std::cout << "Running GPU full diagonalization (cuSOLVER zheevd)..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUFullDiag(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU full diagonalization completed successfully!" << std::endl;
        } else {
            throw std::runtime_error(
                "exact_diagonalization_from_files: GPU dispatch only "
                "supports LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR / "
                "FULL / mTPQ / cTPQ / FTLM at this revision.");
        }

        return results;
    }
#endif
    
    // Load Hamiltonian (for CPU methods)
    Operator hamiltonian = ed_internal::load_hamiltonian_from_files(
        interaction_file, single_site_file, counterterm_file, three_body_file, 
        params.num_sites, params.spin_length, method, format
    );
    
    // Calculate Hilbert space dimension
    uint64_t hilbert_space_dim = static_cast<int>(1ULL << params.num_sites);
    if (ed_log::isVerbose()) {
        ed_log::debug("hilbert_space_dim=" + std::to_string(hilbert_space_dim));
    }
    
    // Create Hamiltonian apply function
    auto apply_hamiltonian = ed_internal::create_hamiltonian_apply_function(hamiltonian);
    
    // Perform diagonalization
    return exact_diagonalization_core(apply_hamiltonian, hilbert_space_dim, method, params);
}

/**
 * @brief Wrapper function to perform exact diagonalization from a directory
 * 
 * Convenience function that constructs file paths from a directory and
 * calls exact_diagonalization_from_files.
 * 
 * @param directory Directory containing Hamiltonian files
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @param format File format for Hamiltonian
 * @param interaction_filename Name of interaction file (default: "InterAll.dat")
 * @param single_site_filename Name of single-site file (default: "Trans.dat")
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_from_directory(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& counterterm_filename = "CounterTerm.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat"
) {
    // Construct full file paths
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    std::string counterterm_file = directory + "/" + counterterm_filename;
    std::string three_body_file = directory + "/" + three_body_filename;
    
    // Check if counter term file exists
    struct stat buffer;
    if (stat(counterterm_file.c_str(), &buffer) != 0) {
        counterterm_file = "";  // File doesn't exist, pass empty string
    }
    
    // Check if three-body file exists
    if (stat(three_body_file.c_str(), &buffer) != 0) {
        three_body_file = "";  // File doesn't exist, pass empty string
    }
    
    // Call the file-based wrapper
    return exact_diagonalization_from_files(
        interaction_file, single_site_file, counterterm_file, three_body_file, method, params, format
    );
}


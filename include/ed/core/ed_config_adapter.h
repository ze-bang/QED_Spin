#pragma once

#include <ed/core/ed_config.h>
#include <ed/core/ed_parameters.h>  // EDParameters
#include <ed/core/ed_types.h>       // DiagonalizationMethod
#include <ed/orchestrator.h>        // ed::workflows::SolveOptions / SolveMethod

/**
 * @brief Adapter to convert between new EDConfig and legacy EDParameters
 * 
 * This allows gradual migration from the old parameter structure
 * to the new configuration system.
 */
namespace ed_adapter {

/**
 * @brief Convert new EDConfig to legacy EDParameters
 */
inline EDParameters toEDParameters(const EDConfig& config) {
    EDParameters params;

    // Diagonalization
    params.max_iterations = config.diag.max_iterations;
    params.num_eigenvalues = config.diag.num_eigenvalues;
    params.tolerance = config.diag.tolerance;
    params.compute_eigenvectors = config.diag.compute_eigenvectors;
    params.block_size = config.diag.block_size;

    // Thermal
    params.num_samples = config.thermal.num_samples;
    params.temp_min = config.thermal.temp_min;
    params.temp_max = config.thermal.temp_max;
    params.num_temp_bins = config.thermal.num_temp_bins;
    // TPQ-specific parameters (using new names with tpq_ prefix)
    params.tpq_taylor_order = config.thermal.tpq_taylor_order;
    params.tpq_measurement_interval = config.thermal.tpq_measurement_interval;
    params.tpq_delta_beta = config.thermal.tpq_delta_beta;
    params.tpq_energy_shift = config.thermal.tpq_energy_shift;
    params.tpq_continue = config.thermal.tpq_continue;
    params.tpq_continue_sample = config.thermal.tpq_continue_sample;
    params.tpq_continue_beta = config.thermal.tpq_continue_beta;
    params.tpq_target_beta = config.thermal.tpq_target_beta;
    params.tpq_num_measure_points = config.thermal.tpq_num_measure_points;
    params.tpq_measure_beta_min = config.thermal.tpq_measure_beta_min;
    params.tpq_measure_beta_max = config.thermal.tpq_measure_beta_max;
    params.tpq_max_steps = config.thermal.tpq_max_steps;
    params.tpq_beta_max = config.thermal.tpq_beta_max;

    // FTLM (via thermal config)
    params.ftlm_krylov_dim = config.thermal.ftlm_krylov_dim;
    params.ftlm_full_reorth = config.thermal.ftlm_full_reorth;
    params.ftlm_reorth_freq = config.thermal.ftlm_reorth_freq;
    params.ftlm_seed = config.thermal.ftlm_seed;
    params.ftlm_store_samples = config.thermal.ftlm_store_samples;
    params.ftlm_error_bars = config.thermal.ftlm_error_bars;

    // LTLM (via thermal config)
    params.ltlm_krylov_dim = config.thermal.ltlm_krylov_dim;
    params.ltlm_ground_krylov = config.thermal.ltlm_ground_krylov;
    params.ltlm_full_reorth = config.thermal.ltlm_full_reorth;
    params.ltlm_reorth_freq = config.thermal.ltlm_reorth_freq;
    params.ltlm_seed = config.thermal.ltlm_seed;
    params.ltlm_store_data = config.thermal.ltlm_store_data;

    // Observable (TPQ thermal state and spin correlation options)
    params.save_thermal_states = config.observable.save_thermal_states;
    params.compute_spin_correlations = config.observable.compute_spin_correlations;
    params.observables = config.observable.operators;
    params.observable_names = config.observable.names;
    params.omega_min = config.observable.omega_min;
    params.omega_max = config.observable.omega_max;
    params.num_points = config.observable.num_points;
    params.t_end = config.observable.t_end;
    params.dt = config.observable.dt;

    // System
    params.num_sites = config.system.num_sites;
    params.spin_length = config.system.spin_length;
    params.sublattice_size = config.system.sublattice_size;
    params.use_fixed_sz = config.system.use_fixed_sz;
    params.n_up = config.system.n_up;
    params.full_sz_split = config.system.full_sz_split;

    // Orthogonal device / parallelism axes
    params.use_gpu = config.system.use_gpu;
    params.use_mpi = config.system.use_mpi;
    // Symmetry projection. Honour either the canonical SystemConfig flag
    // or the legacy WorkflowConfig flag (the CLI parser sets both, but
    // external callers may set only one).
    params.use_symmetry = config.system.use_symmetry || config.workflow.run_symm_auto;

    // Output
    params.output_dir = config.workflow.output_dir;

    // Sector selection
    params.selected_sectors = config.workflow.selected_sectors;

    // Symmetry options
    params.translation_only = config.workflow.translation_only;

    return params;
}

/**
 * @brief Convert legacy EDParameters to new EDConfig
 */
inline EDConfig fromEDParameters(const EDParameters& params, DiagonalizationMethod method) {
    EDConfig config(method);

    // Diagonalization
    config.diag.max_iterations = params.max_iterations;
    config.diag.num_eigenvalues = params.num_eigenvalues;
    config.diag.tolerance = params.tolerance;
    config.diag.compute_eigenvectors = params.compute_eigenvectors;
    config.diag.block_size = params.block_size;

    // Thermal
    config.thermal.num_samples = params.num_samples;
    config.thermal.temp_min = params.temp_min;
    config.thermal.temp_max = params.temp_max;
    config.thermal.num_temp_bins = params.num_temp_bins;
    // TPQ-specific parameters (using new names with tpq_ prefix)
    config.thermal.tpq_taylor_order = params.tpq_taylor_order;
    config.thermal.tpq_measurement_interval = params.tpq_measurement_interval;
    config.thermal.tpq_delta_beta = params.tpq_delta_beta;
    config.thermal.tpq_energy_shift = params.tpq_energy_shift;
    config.thermal.tpq_continue = params.tpq_continue;
    config.thermal.tpq_continue_sample = params.tpq_continue_sample;
    config.thermal.tpq_continue_beta = params.tpq_continue_beta;
    config.thermal.tpq_target_beta = params.tpq_target_beta;
    config.thermal.tpq_num_measure_points = params.tpq_num_measure_points;
    config.thermal.tpq_measure_beta_min = params.tpq_measure_beta_min;
    config.thermal.tpq_measure_beta_max = params.tpq_measure_beta_max;
    config.thermal.tpq_max_steps = params.tpq_max_steps;
    config.thermal.tpq_beta_max = params.tpq_beta_max;

    // FTLM (via thermal config)
    config.thermal.ftlm_krylov_dim = params.ftlm_krylov_dim;
    config.thermal.ftlm_full_reorth = params.ftlm_full_reorth;
    config.thermal.ftlm_reorth_freq = params.ftlm_reorth_freq;
    config.thermal.ftlm_seed = params.ftlm_seed;
    config.thermal.ftlm_store_samples = params.ftlm_store_samples;
    config.thermal.ftlm_error_bars = params.ftlm_error_bars;

    // LTLM (via thermal config)
    config.thermal.ltlm_krylov_dim = params.ltlm_krylov_dim;
    config.thermal.ltlm_ground_krylov = params.ltlm_ground_krylov;
    config.thermal.ltlm_full_reorth = params.ltlm_full_reorth;
    config.thermal.ltlm_reorth_freq = params.ltlm_reorth_freq;
    config.thermal.ltlm_seed = params.ltlm_seed;
    config.thermal.ltlm_store_data = params.ltlm_store_data;

    // Observable
    config.observable.save_thermal_states = params.save_thermal_states;
    config.observable.compute_spin_correlations = params.compute_spin_correlations;
    config.observable.operators = params.observables;
    config.observable.names = params.observable_names;
    config.observable.omega_min = params.omega_min;
    config.observable.omega_max = params.omega_max;
    config.observable.num_points = params.num_points;
    config.observable.t_end = params.t_end;
    config.observable.dt = params.dt;

    // System
    config.system.num_sites = params.num_sites;
    config.system.spin_length = params.spin_length;
    config.system.sublattice_size = params.sublattice_size;
    config.system.use_fixed_sz = params.use_fixed_sz;
    config.system.n_up = params.n_up;
    config.system.full_sz_split = params.full_sz_split;

    // Orthogonal device / parallelism axes
    config.system.use_gpu = params.use_gpu;
    config.system.use_mpi = params.use_mpi;
    // Symmetry projection. Mirror the flag onto both the canonical
    // SystemConfig field and the legacy WorkflowConfig flag so the
    // existing ed_main.cpp dispatch keeps firing
    // run_streaming_symmetry_workflow.
    config.system.use_symmetry = params.use_symmetry;
    if (params.use_symmetry) {
        config.workflow.run_symm_auto = true;
    }

    // Output
    config.workflow.output_dir = params.output_dir;

    // Sector selection + symmetry shape options
    config.workflow.selected_sectors = params.selected_sectors;
    config.workflow.translation_only = params.translation_only;

    return config;
}

/**
 * @brief Map `EDParameters` + `DiagonalizationMethod` onto the new
 *        `ed::workflows::SolveOptions`.
 *
 * Full Unified-Interface Collapse, Wave C2 (May 2026): used by the CLI
 * workflows in src/cli/workflows.cpp to feed the orchestrator surface
 * without losing any of the orthogonal axes the legacy EDParameters
 * carries. Symmetric to `toEDParameters`: every field that survives
 * the collapse is preserved; the few that have no orchestrator
 * counterpart (sector_index, save_thermal_states, observable name
 * lists, etc.) are left for the CLI layer to consume directly.
 *
 * Stage 11a-tail (Jul 2026): this is THE one EDParameters -> SolveOptions
 * converter. The Python surface (`qed/_params.py`) had grown a twin that
 * silently drifted from this one in three fields (backend wiring,
 * allow_infeasible, selected_sectors) -- the same cross-copy drift class
 * that produced the thermal-converter fork. The union now lives here and
 * the deliberate semantic differences between the two callers are the
 * explicit flags:
 *
 *   - `auto_method`: defer the eigensolver choice to the orchestrator
 *     (Python `method=None`); overrides the mapped `method`.
 *   - `wire_backend`: map `params.use_gpu/use_mpi` onto the backend
 *     CONSTRAINTS. Python passes true (`device='cpu'` must pin CPU);
 *     the CLI passes false (its historical semantics: `select_backend`
 *     auto-promotes regardless of the flag, which only picks the
 *     GPU-specific workflow variants).
 *   - `allow_infeasible`: skip the orchestrator's up-front feasibility
 *     refusal (Python expert escape hatch).
 */
inline ed::workflows::SolveOptions
toSolveOptions(const EDParameters& params,
               ::DiagonalizationMethod method = ::DiagonalizationMethod::LANCZOS,
               bool auto_method = false,
               bool wire_backend = false,
               bool allow_infeasible = false) {
    ed::workflows::SolveOptions opts;

    opts.num_eigs       = static_cast<std::size_t>(params.num_eigenvalues);
    opts.max_iter       = static_cast<std::size_t>(params.max_iterations);
    opts.block_size     = static_cast<std::size_t>(params.block_size);
    opts.tolerance      = params.tolerance;
    opts.compute_vectors = params.compute_eigenvectors;
    opts.output_dir     = params.output_dir;

    // Map `DiagonalizationMethod` onto `SolveMethod`. The orchestrator
    // already knows how to auto-select between Lanczos / Krylov-Schur /
    // FullDiag based on `num_eigs` + `geometry().global_dim` when
    // `SolveMethod::Auto` is set; we honour the caller's explicit
    // choice when it maps cleanly and fall back to `Auto` otherwise.
    using SM = ed::workflows::SolveMethod;
    switch (method) {
        case ::DiagonalizationMethod::LANCZOS:        opts.method = SM::Lanczos; break;
        case ::DiagonalizationMethod::BLOCK_LANCZOS:  opts.method = SM::BlockLanczos; break;
        case ::DiagonalizationMethod::KRYLOV_SCHUR:   opts.method = SM::KrylovSchur; break;
        case ::DiagonalizationMethod::BLOCK_KRYLOV_SCHUR: opts.method = SM::BlockKrylovSchur; break;
        case ::DiagonalizationMethod::FULL:           opts.method = SM::FullDiag; break;
        default:                                      opts.method = SM::Auto; break;
    }
    if (auto_method) {
        opts.method = SM::Auto;
    }

    // Device axes -> backend CONSTRAINTS (Python surface only; see the
    // flag documentation above).
    if (wire_backend) {
        opts.backend.allow_gpu = params.use_gpu;
        opts.backend.allow_mpi = params.use_mpi;
    }
    opts.allow_infeasible = allow_infeasible;

    // Orthogonal axes mirrored onto SolveOptions (Wave A5 CLI parity).
    opts.use_fixed_sz          = params.use_fixed_sz;
    opts.use_symmetry          = params.use_symmetry;
    opts.n_up                  = static_cast<int>(params.n_up);
    opts.basis_cache_dir       = params.basis_cache_dir;
    opts.precompute_basis_only = params.precompute_basis_only;

    // Streaming-symmetry sector filter (CLI ``--sectors=`` -> the
    // per-sector solve loop). ``EDParameters`` stores it as ``int``;
    // ``SolveOptions`` as ``std::size_t``. Previously this never made it
    // onto SolveOptions, so the CLI flag was silently ignored.
    opts.selected_sectors.reserve(params.selected_sectors.size());
    for (int s : params.selected_sectors) {
        if (s >= 0) {
            opts.selected_sectors.push_back(static_cast<std::size_t>(s));
        }
    }

    return opts;
}

} // namespace ed_adapter

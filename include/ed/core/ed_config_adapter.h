#ifndef ED_CONFIG_ADAPTER_H
#define ED_CONFIG_ADAPTER_H

#include <ed/core/ed_config.h>
#include <ed/core/ed_wrapper.h>

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
    params.shift = config.diag.shift;
    params.block_size = config.diag.block_size;
    params.max_subspace = config.diag.max_subspace;
    params.target_lower = config.diag.target_lower;
    params.target_upper = config.diag.target_upper;
    
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
    // Note: Legacy alias methods (num_order(), delta_tau(), etc.) are deprecated
    // and should not be used. The above tpq_* fields are the canonical ones.
    
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
    params.use_hybrid_method = config.thermal.use_hybrid_method;
    params.hybrid_crossover = config.thermal.hybrid_crossover;
    params.hybrid_auto_crossover = config.thermal.hybrid_auto_crossover;
    
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
    
    // Output
    params.output_dir = config.workflow.output_dir;
    
    // ARPACK
    params.arpack_advanced_verbose = config.arpack.verbose;
    params.arpack_which = config.arpack.which;
    params.arpack_ncv = config.arpack.ncv;
    params.arpack_max_restarts = config.arpack.max_restarts;
    params.arpack_ncv_growth = config.arpack.ncv_growth;
    params.arpack_auto_enlarge_ncv = config.arpack.auto_enlarge_ncv;
    params.arpack_two_phase_refine = config.arpack.two_phase_refine;
    params.arpack_relaxed_tol = config.arpack.relaxed_tol;
    params.arpack_shift_invert = config.arpack.shift_invert;
    params.arpack_sigma = config.arpack.sigma;
    params.arpack_auto_switch_shift_invert = config.arpack.auto_switch_shift_invert;
    params.arpack_switch_sigma = config.arpack.switch_sigma;
    params.arpack_adaptive_inner_tol = config.arpack.adaptive_inner_tol;
    params.arpack_inner_tol_factor = config.arpack.inner_tol_factor;
    params.arpack_inner_tol_min = config.arpack.inner_tol_min;
    params.arpack_inner_max_iter = config.arpack.inner_max_iter;
    
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
    config.diag.shift = params.shift;
    config.diag.block_size = params.block_size;
    config.diag.max_subspace = params.max_subspace;
    config.diag.target_lower = params.target_lower;
    config.diag.target_upper = params.target_upper;
    
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
    config.thermal.use_hybrid_method = params.use_hybrid_method;
    config.thermal.hybrid_crossover = params.hybrid_crossover;
    
    // Observable (TPQ thermal state and spin correlation options)
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
    
    // Output
    config.workflow.output_dir = params.output_dir;
    
    // ARPACK
    config.arpack.verbose = params.arpack_advanced_verbose;
    config.arpack.which = params.arpack_which;
    config.arpack.ncv = params.arpack_ncv;
    config.arpack.max_restarts = params.arpack_max_restarts;
    config.arpack.ncv_growth = params.arpack_ncv_growth;
    config.arpack.auto_enlarge_ncv = params.arpack_auto_enlarge_ncv;
    config.arpack.two_phase_refine = params.arpack_two_phase_refine;
    config.arpack.relaxed_tol = params.arpack_relaxed_tol;
    config.arpack.shift_invert = params.arpack_shift_invert;
    config.arpack.sigma = params.arpack_sigma;
    config.arpack.auto_switch_shift_invert = params.arpack_auto_switch_shift_invert;
    config.arpack.switch_sigma = params.arpack_switch_sigma;
    config.arpack.adaptive_inner_tol = params.arpack_adaptive_inner_tol;
    config.arpack.inner_tol_factor = params.arpack_inner_tol_factor;
    config.arpack.inner_tol_min = params.arpack_inner_tol_min;
    config.arpack.inner_max_iter = params.arpack_inner_max_iter;
    
    return config;
}

} // namespace ed_adapter

#endif // ED_CONFIG_ADAPTER_H

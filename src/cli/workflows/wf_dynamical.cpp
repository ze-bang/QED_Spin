// =============================================================================
// src/cli/workflows/wf_dynamical.cpp
//
// `compute_dynamical_response_workflow`: the spectral-function driver behind
// the dynamical-response CLI lane (FTLM / LTLM / continued fraction, CPU, GPU
// and MPI master-worker shapes). One function, ~1 kLOC -- it is the largest
// single body in the former workflows.cpp and cannot be split further without
// changing behaviour.
//
// Moved verbatim out of src/cli/workflows.cpp (WP14). See
// workflows_internal.h for the file map.
// =============================================================================

#include "workflows_internal.h"

/**
 * @brief Compute dynamical response (spectral functions)
 */
void compute_dynamical_response_workflow(const EDConfig& config) {
    // Plain locals (not a structured binding): C++17 forbids capturing
    // structured bindings in lambdas (clang enforces; gcc extension).
    int rank = 0, size = 1;
    std::tie(rank, size) = get_mpi_rank_size_safe();

    if (!config.dynamical.thermal_average) {
        if (rank == 0) {
            std::cerr << "Note: Only thermal mode is supported. Setting thermal_average mode.\n";
        }
    }

    if (rank == 0) {
        std::cout << "\nDynamical Response Calculation\n";

#ifdef WITH_CUDA
        if (config.dynamical.use_gpu) {
            if (config.system.use_fixed_sz) {
                std::cout << "  GPU: enabled (fixed-Sz; transverse channels "
                             "with operators that change Sz still fall back "
                             "to CPU until cross-sector wiring lands)\n";
            } else {
                std::cout << "  GPU: enabled (multi-temperature path; single-T "
                             "and 1-sample tasks fall back to CPU)\n";
            }
        }
#else
        if (config.dynamical.use_gpu) {
            std::cout << "  GPU: requested but unavailable (build has no CUDA "
                         "support; using CPU)\n";
        }
#endif
    }

    bool use_config_operators = config.dynamical.operator_file.empty() ||
                                config.dynamical.operator_type != "sum";

    auto wh = build_workflow_hamiltonian(
        config, rank,
        config.system.use_fixed_sz ? "Fixed-Sz dynamical response" : nullptr);
    const bool use_fixed_sz   = wh.use_fixed_sz;
    const uint64_t N          = wh.N;
    auto& ham_full            = wh.ham_full;
    auto& ham_fs              = wh.ham_fs;
    auto& H_func              = wh.H_func;
    Operator& ham             = wh.ham_ref();
    (void)ham_full; (void)ham_fs;  // alive via wh; captured by lambdas elsewhere

    // Setup parameters
    DynamicalResponseParameters params;
    params.num_samples = config.dynamical.num_random_states;
    params.krylov_dim = config.dynamical.krylov_dim;
    params.broadening = config.dynamical.broadening;
    params.random_seed = config.dynamical.random_seed;
    
    // Ensure output directory exists
    create_directory_mpi_safe(config.workflow.output_dir);
    
    if (rank == 0) {
        std::cout << "Random states: " << params.num_samples << "\n";
        std::cout << "Krylov dimension: " << params.krylov_dim << "\n";
        std::cout << "Temperature range: [" << config.dynamical.temp_min << ", " << config.dynamical.temp_max << "]\n";
        std::cout << "Temperature bins: " << config.dynamical.num_temp_bins << "\n";
    }
    
    // Find ground state energy for proper energy shifting
    double ground_state_energy = 0.0;
    bool found_ground_state = false;
    
    if (rank == 0) {
        std::string h5_file = config.workflow.output_dir + "/ed_results.h5";
        
        // Method 1: Try to load eigenvalues from HDF5
        if (HDF5IO::fileExists(h5_file)) {
            try {
                auto eigenvalues = HDF5IO::loadEigenvalues(h5_file);
                if (!eigenvalues.empty()) {
                    ground_state_energy = eigenvalues[0];
                    found_ground_state = true;
                    std::cout << "  Loaded ground state energy from HDF5 eigenvalues\n";
                }
            } catch (const std::exception& e) {
                // Continue to next method
            }
            
            // Method 2: Try TPQ thermodynamics from HDF5
            if (!found_ground_state) {
                try {
                    auto points = HDF5IO::loadTPQThermodynamics(h5_file, 0);
                    if (!points.empty()) {
                        double min_energy = std::numeric_limits<double>::max();
                        for (size_t i = 1; i < points.size(); ++i) {  // Skip first entry
                            if (points[i].energy < min_energy) {
                                min_energy = points[i].energy;
                            }
                        }
                        if (min_energy < std::numeric_limits<double>::max()) {
                            ground_state_energy = min_energy;
                            found_ground_state = true;
                            std::cout << "  Loaded ground state energy from HDF5 TPQ data\n";
                        }
                    }
                } catch (const std::exception& e) {
                    // Continue to fallback
                }
            }
        }
        
        // Method 3 (fallback): Compute using Lanczos
        if (!found_ground_state) {
            std::cout << "  Computing ground state energy using Lanczos...\n";
            ComplexVector ground_state(N);
            ground_state_energy = find_ground_state_lanczos(
                H_func, N, params.krylov_dim, params.tolerance,
                params.full_reorthogonalization, params.reorth_frequency,
                ground_state
            );
            found_ground_state = true;
            
            // Save computed ground state energy to HDF5
            try {
                std::string h5_path = HDF5IO::createOrOpenFile(config.workflow.output_dir);
                HDF5IO::saveEigenvalues(h5_path, {ground_state_energy});
            } catch (...) {
                // Ignore save errors
            }
        }
        
        std::cout << "  Ground state energy: " << std::fixed << std::setprecision(10) 
                  << ground_state_energy << "\n";
    }
    
    #ifdef WITH_MPI
    // Broadcast ground state energy to all ranks
    MPI_Bcast(&ground_state_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&found_ground_state, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    #endif
    
    if (!found_ground_state) {
        if (rank == 0) {
            std::cerr << "Error: Failed to obtain ground state energy\n";
        }
        return;
    }
    
    // Generate temperature grid
    std::vector<double> temperatures(config.dynamical.num_temp_bins);
    if (config.dynamical.num_temp_bins == 1) {
        temperatures[0] = config.dynamical.temp_min;
    } else {
        double log_tmin = std::log(config.dynamical.temp_min);
        double log_tmax = std::log(config.dynamical.temp_max);
        double log_step = (log_tmax - log_tmin) / (config.dynamical.num_temp_bins - 1);
        for (uint64_t i = 0; i < config.dynamical.num_temp_bins; i++) {
            temperatures[i] = std::exp(log_tmin + i * log_step);
        }
    }
    
    if (use_config_operators) {
        // Configuration-based operator construction
        if (rank == 0) {
            std::cout << "  Operator type: " << config.dynamical.operator_type 
                      << " (" << config.dynamical.basis << " basis)\n";
        }
        
        // Parse configuration
        auto spin_combinations = parse_spin_combinations(config.dynamical.spin_combinations);
        spin_combinations = filter_fixed_sz_transverse_channels(
            spin_combinations,
            config.system.use_fixed_sz,
            (config.dynamical.basis == "xyz"),
            rank,
            "compute_dynamical_response_workflow");
        auto momentum_points = parse_momentum_points(config.dynamical.momentum_points);
        auto polarization = parse_polarization(config.dynamical.polarization);
        
        // Get positions file
        std::string positions_file = config.system.hamiltonian_dir + "/positions.dat";
        
        // Determine fixed-Sz parameters
        bool use_fixed_sz = config.system.use_fixed_sz;
        int64_t n_up = (use_fixed_sz && config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        
        // Construct operators (audit #2: also obtain shared_ptr<FixedSzOperator>
        // arrays so the CPU apply path correctly dispatches at the fixed-Sz
        // dimension instead of slicing into Operator::apply which throws).
        ed::dssf::OperatorSpec _spec;
        _spec.operator_type     = config.dynamical.operator_type;
        _spec.basis             = config.dynamical.basis;
        _spec.spin_combinations = spin_combinations;
        _spec.momentum_points   = momentum_points;
        _spec.polarization      = polarization;
        _spec.theta             = config.dynamical.theta;
        _spec.unit_cell_size    = config.dynamical.unit_cell_size;
        _spec.num_sites         = config.system.num_sites;
        _spec.spin_length       = config.system.spin_length;
        _spec.use_fixed_sz      = use_fixed_sz;
        _spec.n_up              = n_up;
        _spec.positions_file    = positions_file;
        auto _pairs = ed::dssf::build_observable_pairs(_spec);
        std::vector<Operator>&    obs_1 = _pairs.obs_1;
        std::vector<Operator>&    obs_2 = _pairs.obs_2;
        std::vector<std::string>& names = _pairs.names;
        std::vector<std::shared_ptr<FixedSzOperator>>& obs_1_fs = _pairs.obs_1_fs;
        std::vector<std::shared_ptr<FixedSzOperator>>& obs_2_fs = _pairs.obs_2_fs;
        // CPU dispatcher used by every O1/O2 lambda below.
        auto apply_obs1 = [&obs_1, &obs_1_fs, use_fixed_sz](
            int op_idx, const Complex* in, Complex* out, uint64_t dim) {
            if (use_fixed_sz) obs_1_fs[op_idx]->apply(in, out, dim);
            else              obs_1[op_idx].apply(in, out, dim);
        };
        auto apply_obs2 = [&obs_2, &obs_2_fs, use_fixed_sz](
            int op_idx, const Complex* in, Complex* out, uint64_t dim) {
            if (use_fixed_sz) obs_2_fs[op_idx]->apply(in, out, dim);
            else              obs_2[op_idx].apply(in, out, dim);
        };
        
        if (rank == 0) {
            std::cout << "  Operators: " << obs_1.size() << " pair(s)\n";
        }
        
        // ============================================================
        // MPI Task Distribution
        // ============================================================
        
        // Decide whether to use optimized multi-temperature workflow
        int num_operators = obs_1.size();
        int num_temps = config.dynamical.num_temp_bins;
        bool use_optimized_multi_temp = (num_temps > 1);
        
        if (rank == 0 && use_optimized_multi_temp) {
            std::cout << "  Multi-temperature optimization enabled (" << num_temps << " temps)\n";
        }
        
        struct DynTask {
            int temp_idx;
            int op_idx;
            size_t weight;  // estimated cost (number of operators * samples)
            bool is_multi_temp;  // True if this task handles all temperatures for one operator
        };
        
        std::vector<DynTask> all_tasks;
        
        if (rank == 0) {
            if (use_optimized_multi_temp) {
                // OPTIMIZED: Create one task per operator (handles all temperatures)
                for (int o = 0; o < num_operators; o++) {
                    size_t weight = params.num_samples * params.krylov_dim * num_temps;
                    all_tasks.push_back({0, o, weight, true});
                }
            } else {
                // Standard: Create one task per (temperature, operator) pair
                for (int t = 0; t < num_temps; t++) {
                    for (int o = 0; o < num_operators; o++) {
                        size_t weight = params.num_samples * params.krylov_dim;
                        all_tasks.push_back({t, o, weight, false});
                    }
                }
                std::cout << "\nStandard Mode: " << all_tasks.size() << " tasks = "
                          << num_temps << " temperatures × " << num_operators << " operators\n";
            }
            
            // Sort by weight (descending) for better load balance
            std::sort(all_tasks.begin(), all_tasks.end(), 
                      [](const DynTask& a, const DynTask& b) { return a.weight > b.weight; });
            
            std::cout << "Running on " << size << " MPI rank(s)\n";
        }
        
        // Broadcast optimization flag and task count
        int num_tasks = all_tasks.size();
        #ifdef WITH_MPI
        MPI_Bcast(&use_optimized_multi_temp, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            all_tasks.resize(num_tasks);
        }
        
        // Broadcast all tasks
        for (int i = 0; i < num_tasks; i++) {
            int buf[3] = {all_tasks[i].temp_idx, all_tasks[i].op_idx, all_tasks[i].is_multi_temp ? 1 : 0};
            size_t w = all_tasks[i].weight;
            MPI_Bcast(buf, 3, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&w, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            if (rank != 0) {
                all_tasks[i] = {buf[0], buf[1], w, buf[2] != 0};
            }
        }
        #endif
        
        // Lambda to process a single task (single temperature, single operator).
        //
        // GPU is only supported in the multi-temperature path
        // (`process_operator_all_temps` below, via GPUFTLMSolver). Single-T
        // tasks silently use the CPU kernel; the relevant heads-up is printed
        // once at the workflow banner above (`config.dynamical.use_gpu`
        // summary), so we do not repeat it per task here.
        // Audit 2026-07-31 (H2): split into compute / write halves so the
        // MPI master-worker path can keep computes distributed while rank
        // 0 stays the ONLY HDF5 writer (concurrent writers raced on the
        // shared ed_results.h5 file lock). The sequential lane composes
        // the two halves and is byte-identical to the old body.
        auto compute_task_single =
            [&](const DynTask& task) -> DynamicalResponseResults {
            int t_idx = task.temp_idx;
            int op_idx = task.op_idx;
            double temperature = temperatures[t_idx];

            // Family-3 consolidation (audit 2026-07-31): single-T tasks
            // route through the SAME backend-generic multitemp kernel as
            // the multi-T path (temperatures = {T}), retiring the
            // per-task legacy ::compute_dynamical_correlation -- and
            // incidentally giving single-T tasks the GPU lane the old
            // CPU-only body never had. Random stream changes vs the
            // legacy driver (same estimator, different draws).
            const uint64_t n_omega = config.dynamical.num_omega_points;
            std::vector<double> omega_grid(n_omega);
            const double omega_step = (config.dynamical.omega_max -
                config.dynamical.omega_min) /
                static_cast<double>(std::max<uint64_t>(1, n_omega - 1));
            for (uint64_t i = 0; i < n_omega; ++i)
                omega_grid[i] = config.dynamical.omega_min +
                                static_cast<double>(i) * omega_step;
            const std::vector<double> one_T{temperature};

            ed::LinearOperator& H_op = ham;
            ed::LinearOperator& O1_op = config.system.use_fixed_sz
                ? static_cast<ed::LinearOperator&>(*obs_1_fs[op_idx])
                : static_cast<ed::LinearOperator&>(obs_1[op_idx]);
            ed::LinearOperator& O2_op = config.system.use_fixed_sz
                ? static_cast<ed::LinearOperator&>(*obs_2_fs[op_idx])
                : static_cast<ed::LinearOperator&>(obs_2[op_idx]);

            ed::BackendConstraints bc;
            bc.allow_gpu     = config.dynamical.use_gpu;
            bc.allow_mpi     = false;   // rank-local task; results are
            bc.allow_mpi_gpu = false;   // shipped by the master-worker
            auto variant = ed::select_backend(H_op.geometry(), bc);

            DynamicalResponseResults out;
            std::visit([&](auto& backend_uptr) {
                using BPtr = std::decay_t<decltype(backend_uptr)>;
                using B    = typename BPtr::element_type;
                constexpr bool is_cpu =
                    std::is_same_v<B, ed::matvec::CpuBackend>;
#ifdef WITH_CUDA
                constexpr bool is_cuda =
                    std::is_same_v<B, ed::matvec::CudaBackend>;
#else
                constexpr bool is_cuda = false;
#endif
                if constexpr (!(is_cpu || is_cuda)) {
                    throw std::runtime_error(
                        "dynamical FTLM: requires a CpuBackend or "
                        "CudaBackend; distributed backends are not wired.");
                } else {
                    ed::observables::FtlmDynamicalOptions kopts;
                    kopts.krylov_dim   = params.krylov_dim;
                    kopts.num_samples  = params.num_samples;
                    kopts.broadening   = params.broadening;
                    kopts.energy_shift = ground_state_energy;
                    kopts.tolerance    = params.tolerance;
                    kopts.random_seed  = params.random_seed;
                    kopts.global_n     = H_op.geometry().global_dim;

                    auto mv_h  = H_op.template bind<B>();
                    auto mv_o1 = O1_op.template bind<B>();
                    auto mv_o2 = O2_op.template bind<B>();
                    auto res = ed::observables::detail::
                        ftlm_dynamical_kernel_via_backend_multitemp(
                            *backend_uptr, mv_h, mv_o1, mv_o2,
                            H_op.geometry().local_dim, omega_grid,
                            one_T, kopts);
                    out.frequencies            = res[0].omega;
                    out.spectral_function      = res[0].spectral_real;
                    out.spectral_function_imag = res[0].spectral_imag;
                    out.spectral_error         = res[0].spectral_error_real;
                    out.spectral_error_imag    = res[0].spectral_error_imag;
                    out.total_samples          = res[0].total_samples;
                }
            }, variant);
            return out;
        };

        auto write_task_single = [&](const DynTask& task,
                                     const DynamicalResponseResults& results) {
            double temperature = temperatures[task.temp_idx];
            std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
            std::string op_name = names[task.op_idx];
            if (config.dynamical.num_temp_bins > 1) {
                op_name += "_T" + std::to_string(temperature);
            }
            HDF5IO::saveDynamicalResponseFull(
                h5_file, op_name,
                results.frequencies, results.spectral_function, results.spectral_function_imag,
                results.spectral_error, results.spectral_error_imag,
                results.total_samples, temperature
            );
        };

        auto process_task_single = [&](const DynTask& task) -> bool {
            write_task_single(task, compute_task_single(task));
            return true;
        };
        
        // Lambda to process all temperatures for one operator (OPTIMIZED!)
        auto process_operator_all_temps = [&](int op_idx) -> bool {
            if (rank == 0) {
                std::cout << "\n=== OPTIMIZED: Processing operator " << names[op_idx] 
                          << " for ALL " << temperatures.size() << " temperatures with SINGLE Lanczos run ===\n";
            }
            
            // Use optimized multi-temperature function
            // This runs Lanczos once per sample, then computes all temperatures efficiently
            std::map<double, DynamicalResponseResults> results_map;
            
            // Consolidation Family 3: one backend-generic dynamical-FTLM kernel
            // (ftlm_dynamical_kernel_via_backend_multitemp) replaces BOTH the
            // legacy GPUFTLMSolver multi-temp GPU path and the CPU
            // compute_dynamical_correlation_*_multi_temperature functions.
            // select_backend picks CudaBackend when --use-gpu is set and the
            // device matvec is available, else CpuBackend; the Krylov basis is
            // built once per random sample and reweighted across all T.
            {
                const uint64_t n_omega = config.dynamical.num_omega_points;
                std::vector<double> omega_grid(n_omega);
                const double omega_step = (config.dynamical.omega_max -
                    config.dynamical.omega_min) /
                    static_cast<double>(std::max<uint64_t>(1, n_omega - 1));
                for (uint64_t i = 0; i < n_omega; ++i)
                    omega_grid[i] = config.dynamical.omega_min +
                                    static_cast<double>(i) * omega_step;

                ed::LinearOperator& H_op = ham;
                ed::LinearOperator& O1_op = config.system.use_fixed_sz
                    ? static_cast<ed::LinearOperator&>(*obs_1_fs[op_idx])
                    : static_cast<ed::LinearOperator&>(obs_1[op_idx]);
                ed::LinearOperator& O2_op = config.system.use_fixed_sz
                    ? static_cast<ed::LinearOperator&>(*obs_2_fs[op_idx])
                    : static_cast<ed::LinearOperator&>(obs_2[op_idx]);

                ed::BackendConstraints bc;
                bc.allow_gpu     = config.dynamical.use_gpu;
                bc.allow_mpi     = false;   // DSSF FTLM lane is single-node
                bc.allow_mpi_gpu = false;
                auto variant = ed::select_backend(H_op.geometry(), bc);

                std::visit([&](auto& backend_uptr) {
                    using BPtr = std::decay_t<decltype(backend_uptr)>;
                    using B    = typename BPtr::element_type;
                    constexpr bool is_cpu =
                        std::is_same_v<B, ed::matvec::CpuBackend>;
#ifdef WITH_CUDA
                    constexpr bool is_cuda =
                        std::is_same_v<B, ed::matvec::CudaBackend>;
#else
                    constexpr bool is_cuda = false;
#endif
                    if constexpr (!(is_cpu || is_cuda)) {
                        throw std::runtime_error(
                            "dynamical FTLM: requires a CpuBackend or "
                            "CudaBackend; distributed backends are not wired.");
                    } else {
                        ed::observables::FtlmDynamicalOptions kopts;
                        kopts.krylov_dim   = params.krylov_dim;
                        kopts.num_samples  = params.num_samples;
                        kopts.broadening   = params.broadening;
                        kopts.energy_shift = ground_state_energy;
                        kopts.tolerance    = 1e-10;
                        kopts.random_seed  = params.random_seed;
                        kopts.global_n     = H_op.geometry().global_dim;

                        auto mv_h  = H_op.template bind<B>();
                        auto mv_o1 = O1_op.template bind<B>();
                        auto mv_o2 = O2_op.template bind<B>();
                        auto res = ed::observables::detail::
                            ftlm_dynamical_kernel_via_backend_multitemp(
                                *backend_uptr, mv_h, mv_o1, mv_o2,
                                H_op.geometry().local_dim, omega_grid,
                                temperatures, kopts);

                        for (std::size_t t = 0; t < temperatures.size(); ++t) {
                            DynamicalResponseResults r;
                            r.frequencies            = res[t].omega;
                            r.spectral_function      = res[t].spectral_real;
                            r.spectral_function_imag = res[t].spectral_imag;
                            r.spectral_error         = res[t].spectral_error_real;
                            r.spectral_error_imag    = res[t].spectral_error_imag;
                            r.total_samples          = res[t].total_samples;
                            results_map[temperatures[t]] = std::move(r);
                        }
                    }
                }, variant);
            }
            
            // Save results for all temperatures to HDF5
            std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
            for (const auto& [temperature, results] : results_map) {
                std::string op_name = names[op_idx];
                if (temperatures.size() > 1) {
                    op_name += "_T" + std::to_string(temperature);
                }
                HDF5IO::saveDynamicalResponseFull(
                    h5_file, op_name,
                    results.frequencies, results.spectral_function, results.spectral_function_imag,
                    results.spectral_error, results.spectral_error_imag,
                    results.total_samples, temperature
                );
            }
            
            return true;
        };

        // ============================================================
        // Audit #4: comm-aware variant of process_operator_all_temps.
        // Used by the MPI_Comm_split orchestration path below so that
        // each subgroup runs FTLM independently on its own communicator
        // (sample reductions stay local to the subgroup) and only the
        // subgroup's rank-0 returns the per-T results map. CPU only;
        // GPU and single-sample paths fall back to per-world-rank
        // execution and are not split.
        // ============================================================
#ifdef WITH_MPI
        auto process_operator_all_temps_on_comm =
            [&](int op_idx, MPI_Comm comm)
                -> std::map<double, DynamicalResponseResults> {
            int local_rank = 0;
            MPI_Comm_rank(comm, &local_rank);
            if (local_rank == 0) {
                std::cout << "  [op-group leader, world rank " << rank
                          << "] " << names[op_idx]
                          << " (" << temperatures.size() << " temps) on "
                          << "subgroup\n";
            }
            auto O1_func = [apply_obs1, op_idx]
                (const Complex* in, Complex* out, uint64_t dim) {
                    apply_obs1(op_idx, in, out, dim);
                };
            auto O2_func = [apply_obs2, op_idx]
                (const Complex* in, Complex* out, uint64_t dim) {
                    apply_obs2(op_idx, in, out, dim);
                };
            // Single-sample path also goes through the standard CPU
            // multi-sample multi-T kernel here (with num_samples=1) so
            // the comm parameter is honored. The state-based optimization
            // is skipped in this branch -- it doesn't move the needle
            // when each subgroup already has reduced sample count.
            return ed::dssf::
                compute_dynamical_correlation_multi_sample_multi_temperature_comm(
                    H_func, O1_func, O2_func, N, params,
                    config.dynamical.omega_min,
                    config.dynamical.omega_max,
                    config.dynamical.num_omega_points,
                    temperatures,
                    ground_state_energy,
                    config.workflow.output_dir,
                    comm);
        };
#endif

        // ============================================================
        // Multi-operator dispatcher (item #7): processes ALL operator
        // pairs in a single call so that the per-sample H-Lanczos chain
        // (and the cached Ritz eigenstates) are reused across pairs
        // instead of being recomputed P times.
        //
        // CPU multi-sample only. If GPU is requested (and available,
        // and not fixed-Sz) we keep the per-operator GPU path -- the
        // GPU multi-T kernel already amortizes Lanczos across
        // temperatures and is generally faster than the CPU shared-
        // Lanczos path for the same operator. Single-sample mode still
        // uses the state-based optimization in the per-op path.
        // ============================================================
        auto process_all_operators_at_once_cpu = [&]() -> int {
            const int P = static_cast<int>(obs_1.size());
            if (P == 0) return 0;

            std::vector<std::function<void(const Complex*, Complex*, int)>>
                O1_funcs, O2_funcs;
            O1_funcs.reserve(P);
            O2_funcs.reserve(P);
            for (int op_idx = 0; op_idx < P; ++op_idx) {
                O1_funcs.emplace_back([apply_obs1, op_idx]
                    (const Complex* in, Complex* out, uint64_t dim) {
                        apply_obs1(op_idx, in, out, dim);
                    });
                O2_funcs.emplace_back([apply_obs2, op_idx]
                    (const Complex* in, Complex* out, uint64_t dim) {
                        apply_obs2(op_idx, in, out, dim);
                    });
            }

            if (rank == 0) {
                std::cout << "\n=== SHARED-LANCZOS: " << P
                          << " operator pairs, all temperatures, "
                          << "single per-sample H-Lanczos chain ===\n";
            }

            auto results_list =
                compute_dynamical_correlation_multi_operator_multi_temperature(
                    H_func, O1_funcs, O2_funcs, N, params,
                    config.dynamical.omega_min,
                    config.dynamical.omega_max,
                    config.dynamical.num_omega_points,
                    temperatures,
                    ground_state_energy,
                    config.workflow.output_dir);

            if (rank == 0) {
                std::string h5_file =
                    HDF5IO::createOrOpenFile(config.workflow.output_dir);
                for (size_t op_idx = 0; op_idx < results_list.size(); ++op_idx) {
                    for (const auto& [temperature, results]
                         : results_list[op_idx]) {
                        std::string op_name = names[op_idx];
                        if (temperatures.size() > 1) {
                            op_name += "_T" + std::to_string(temperature);
                        }
                        HDF5IO::saveDynamicalResponseFull(
                            h5_file, op_name,
                            results.frequencies,
                            results.spectral_function,
                            results.spectral_function_imag,
                            results.spectral_error,
                            results.spectral_error_imag,
                            results.total_samples, temperature);
                    }
                }
            }
            // Count once globally: the kernel is collective (every rank
            // participates), but the MPI_Reduce'd "Processed X/N" total
            // must not multiply by the rank count.
            return (rank == 0) ? P : 0;
        };

        // Decide whether the multi-operator shared-Lanczos path applies.
        // Engages on the sequential lane AND (since the 2026-07-31 port)
        // on both MPI modes: the multi-op kernel is comm-collective with
        // global-index per-sample seeds, so serial and mpirun agree to
        // Reduce-reassociation roundoff at a fixed seed. The per-task
        // master-worker remains the lane for single-sample or
        // single-operator task sets (and the GPU path).
        bool use_shared_lanczos_multi_op = false;
        {
            const bool cpu_only =
#ifdef WITH_CUDA
                (!config.dynamical.use_gpu);
#else
                true;
#endif
            use_shared_lanczos_multi_op =
                cpu_only &&
                (params.num_samples > 1) &&
                (num_operators > 1);
        }

        // Execute tasks with dynamic work distribution
        int local_processed_count = 0;
        
        #ifdef WITH_MPI
        if (size > 1 && use_optimized_multi_temp) {
            // ============================================================
            // SYNCHRONIZED MODE: All ranks process the same operator at once
            // Required because compute_dynamical_correlation_multi_sample_multi_temperature_comm
            // uses MPI collectives (Barrier, Reduce) internally for sample distribution.
            // The master-worker pattern would cause collective mismatches since
            // different ranks would be processing different operators.
            //
            // Audit #4: when num_operators > 1 and CPU multi-sample (no
            // GPU, no shared-Lanczos), split MPI_COMM_WORLD into
            // op_groups so subgroups handle distinct operators in
            // parallel. Each subgroup runs FTLM on its own communicator;
            // HDF5 writes are serialized across subgroup leaders to
            // avoid concurrent writers on the same file.
            // ============================================================
            const bool gpu_path_active =
#ifdef WITH_CUDA
                config.dynamical.use_gpu;
#else
                false;
#endif
            const int num_op_groups =
                std::min<int>(num_operators, size);
            const bool can_split =
                !use_shared_lanczos_multi_op &&
                !gpu_path_active &&
                (num_op_groups > 1) &&
                (params.num_samples >= static_cast<uint64_t>(num_op_groups));

            if (use_shared_lanczos_multi_op) {
                // One synchronized call across all ranks; the new
                // multi-operator FTLM uses MPI collectives internally.
                local_processed_count += process_all_operators_at_once_cpu();
            } else if (can_split) {
                if (rank == 0) {
                    std::cout << "\n=== Audit #4: MPI_Comm_split into "
                              << num_op_groups << " op-groups ("
                              << (size / num_op_groups) << "-"
                              << ((size + num_op_groups - 1) / num_op_groups)
                              << " ranks per group, "
                              << num_operators << " operators, "
                              << params.num_samples << " samples) ===\n";
                }
                const int color = (rank * num_op_groups) / size;
                MPI_Comm op_comm;
                MPI_Comm_split(MPI_COMM_WORLD, color, rank, &op_comm);
                int op_rank = 0;
                MPI_Comm_rank(op_comm, &op_rank);

                // Round-robin assignment of operators (by all_tasks order
                // for load-balance reasons -- tasks are sorted heaviest
                // first by weight above).
                std::vector<int> my_ops;
                for (int t = 0; t < num_tasks; t++) {
                    if (t % num_op_groups == color) {
                        my_ops.push_back(all_tasks[t].op_idx);
                    }
                }

                // Run each assigned op on op_comm; cache (op_idx, results)
                // on subgroup leader for serialized HDF5 write below.
                std::vector<std::pair<int,
                    std::map<double, DynamicalResponseResults>>> cached;
                cached.reserve(my_ops.size());
                for (int op_idx : my_ops) {
                    auto results_map =
                        process_operator_all_temps_on_comm(op_idx, op_comm);
                    if (op_rank == 0) {
                        cached.emplace_back(op_idx, std::move(results_map));
                    }
                    local_processed_count++;
                }

                // Serialized HDF5 writes: each subgroup leader writes its
                // cached operators in order; non-leaders just barrier.
                for (int g = 0; g < num_op_groups; g++) {
                    if (color == g && op_rank == 0) {
                        std::string h5_file =
                            HDF5IO::createOrOpenFile(config.workflow.output_dir);
                        for (auto& kv : cached) {
                            int op_idx = kv.first;
                            auto& rm = kv.second;
                            for (auto& tv : rm) {
                                double temperature = tv.first;
                                auto& results = tv.second;
                                std::string op_name = names[op_idx];
                                if (temperatures.size() > 1) {
                                    op_name += "_T" + std::to_string(temperature);
                                }
                                HDF5IO::saveDynamicalResponseFull(
                                    h5_file, op_name,
                                    results.frequencies,
                                    results.spectral_function,
                                    results.spectral_function_imag,
                                    results.spectral_error,
                                    results.spectral_error_imag,
                                    results.total_samples, temperature);
                            }
                        }
                        std::cout << "  [Group " << g
                                  << " leader, world rank " << rank
                                  << "] wrote " << cached.size()
                                  << " operator(s) to HDF5\n";
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                MPI_Comm_free(&op_comm);
            } else {
                // Audit 2026-07-31 (H3): this fallback used to run
                // process_operator_all_temps on EVERY rank behind a stale
                // comment claiming internal collectives -- the current
                // body pins allow_mpi=false and runs the single-node
                // kernel, so mpirun -n P computed the identical FTLM P
                // times and raced P concurrent writers on the shared
                // HDF5 file. Rank 0 now computes and writes alone (same
                // wall time as before -- the other ranks were doing
                // redundant copies of the same work); everyone else
                // waits at the barrier.
                if (rank == 0) {
                    for (int task_idx = 0; task_idx < num_tasks; task_idx++) {
                        const auto& task = all_tasks[task_idx];
                        std::cout << "\n--- Task " << (task_idx + 1) << " / " << num_tasks
                                  << ": Operator " << names[task.op_idx]
                                  << " (ALL temperatures, rank 0 of "
                                  << size << " -- unsplittable task set) ---\n";
                        if (process_operator_all_temps(task.op_idx)) {
                            local_processed_count++;
                        }
                    }
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else if (size > 1 && !use_optimized_multi_temp
                   && use_shared_lanczos_multi_op) {
            // Shared-Lanczos port to the MPI shape (audit 2026-07-31,
            // the final consolidation-plan residue): the multi-operator
            // kernel is comm-collective over MPI_COMM_WORLD (samples
            // split by GLOBAL index across ranks, spectra Reduced), so
            // the single-T multi-op case now amortizes ONE H-Krylov
            // chain per sample across every operator pair under mpirun
            // exactly like the sequential lane -- same estimator, and
            // bit-identical to serial up to Reduce reassociation
            // (per-sample seeds key on the global sample index). All
            // ranks enter synchronously (the gate is config-derived).
            local_processed_count += process_all_operators_at_once_cpu();
        } else if (size > 1 && !use_optimized_multi_temp) {
            // Audit 2026-07-31 (H2): single-writer master-worker. Workers
            // compute and SHIP their packed spectra; rank 0 is the only
            // rank that opens ed_results.h5 (the old protocol had every
            // worker write into the shared file -- HDF5 file locking made
            // that a race -- and a worker dying on any exception never
            // sent DONE, spinning the master forever). See
            // run_mpi_master_worker_single_writer for the protocol.
            auto pack_dyn = [](const DynamicalResponseResults& r) {
                std::vector<double> p;
                pack_scalar(p, static_cast<double>(r.total_samples));
                pack_array(p, r.frequencies);
                pack_array(p, r.spectral_function);
                pack_array(p, r.spectral_function_imag);
                pack_array(p, r.spectral_error);
                pack_array(p, r.spectral_error_imag);
                return p;
            };
            auto unpack_dyn = [](const std::vector<double>& p) {
                DynamicalResponseResults r;
                std::size_t pos = 0;
                r.total_samples =
                    static_cast<uint64_t>(unpack_scalar(p, pos));
                r.frequencies            = unpack_array(p, pos);
                r.spectral_function      = unpack_array(p, pos);
                r.spectral_function_imag = unpack_array(p, pos);
                r.spectral_error         = unpack_array(p, pos);
                r.spectral_error_imag    = unpack_array(p, pos);
                return r;
            };
            local_processed_count += run_mpi_master_worker_single_writer(
                rank, size, num_tasks,
                [&](int t) {
                    return pack_dyn(compute_task_single(all_tasks[t]));
                },
                [&](int t, const std::vector<double>& p) {
                    write_task_single(all_tasks[t], unpack_dyn(p));
                },
                "dynamical-response master-worker");
        } else
        #endif
        {
            // Sequential execution (no MPI or single rank)
            if (use_shared_lanczos_multi_op) {
                local_processed_count += process_all_operators_at_once_cpu();
            } else {
            for (int task_idx = 0; task_idx < num_tasks; task_idx++) {
                const auto& task = all_tasks[task_idx];
                
                if (task.is_multi_temp) {
                    if (rank == 0) {
                        std::cout << "\n--- Task " << (task_idx + 1) << " / " << num_tasks
                                  << ": Operator " << names[task.op_idx] << " (ALL temperatures) ---\n";
                    }
                    if (process_operator_all_temps(task.op_idx)) {
                        local_processed_count++;
                    }
                } else {
                    if (rank == 0) {
                        std::cout << "\n--- Task " << (task_idx + 1) << " / " << num_tasks
                                  << ": T = " << temperatures[task.temp_idx]
                                  << ", operator: " << names[task.op_idx] << " ---\n";
                    }
                    if (process_task_single(task)) {
                        local_processed_count++;
                    }
                }
            }
            }  // end else (use_shared_lanczos_multi_op)
        }
        
        #ifdef WITH_MPI
        // Gather statistics
        int total_processed_count = local_processed_count;
        {
            int mpi_inited_red = 0;
            MPI_Initialized(&mpi_inited_red);
            if (mpi_inited_red) {
                MPI_Reduce(&local_processed_count, &total_processed_count, 1,
                           MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }

        if (rank == 0) {
            std::cout << "\nProcessed " << total_processed_count << "/" << num_tasks << " tasks successfully.\n";
        }
        #else
        if (rank == 0) {
            std::cout << "\nProcessed " << local_processed_count << "/" << num_tasks << " tasks successfully.\n";
        }
        #endif
        
    } else {
        // ============================================================
        // Legacy file-based operator loading
        // ============================================================
        if (rank == 0) std::cout << "\nUsing legacy file-based operator loading\n";
        
        if (config.dynamical.operator_file.empty()) {
            std::cerr << "Error: --dyn-operator=<file> is required for dynamical response\n";
            return;
        }
        
        std::string op_path = config.system.hamiltonian_dir + "/" + config.dynamical.operator_file;
        Operator op(config.system.num_sites, config.system.spin_length);
        op.loadFromInterAllFile(op_path);
        // Also load three-body terms if a companion file exists
        {
            std::string op_3body = op_path + ".3body";
            std::ifstream test_3b(op_3body);
            if (test_3b.good()) {
                op.loadThreeBodyTerm(op_3body);
            }
        }
        
        auto O_func = [&op](const Complex* in, Complex* out, uint64_t dim) {
            op.apply(in, out, dim);
        };

        // Hoist the optional second operator (and its 3-body sidecar) OUT of
        // the per-temperature loop. The previous version reloaded op2 from
        // disk, parsed InterAll, and reconstructed CSR for every temperature;
        // for a temperature scan with N_T points that's O(N_T) redundant disk
        // reads + sparse rebuilds.  The matrix elements don't depend on T.
        const bool have_op2 = !config.dynamical.operator2_file.empty();
        Operator op2(config.system.num_sites, config.system.spin_length);
        if (have_op2) {
            std::string op2_path = config.system.hamiltonian_dir + "/" + config.dynamical.operator2_file;
            op2.loadFromInterAllFile(op2_path);
            std::string op2_3body = op2_path + ".3body";
            std::ifstream test_3b2(op2_3body);
            if (test_3b2.good()) {
                op2.loadThreeBodyTerm(op2_3body);
            }
        }
        auto O2_func = [&op2](const Complex* in, Complex* out, uint64_t dim) {
            op2.apply(in, out, dim);
        };

        // Compute for each temperature. Only rank 0 narrates so multi-rank
        // logs stay readable; ranks > 0 still execute the loop body if the
        // legacy path was reached (typically size==1, but do not assume).
        for (uint64_t t_idx = 0; t_idx < config.dynamical.num_temp_bins; t_idx++) {
            double temperature = temperatures[t_idx];

            if (rank == 0) {
                std::cout << "\n--- Temperature " << (t_idx + 1) << " / " << config.dynamical.num_temp_bins
                          << ": T = " << temperature << " ---\n";
            }

            DynamicalResponseResults results;

            // Family-3 consolidation (audit 2026-07-31): both the
            // two-operator and the single-operator legacy calls route
            // through the backend-generic multitemp kernel (one-element
            // T grid; O1 = O2 = O reproduces <O†(t)O>). This branch is
            // host-functor based (legacy file-loaded operators), so the
            // CPU backend is pinned. Random stream changes vs the legacy
            // driver (same estimator, different draws).
            {
                if (rank == 0) {
                    std::cout << (have_op2
                        ? "Computing two-operator dynamical correlation "
                          "⟨O₁†(t)O₂⟩...\n"
                        : "Computing dynamical response ⟨O†(t)O⟩...\n");
                }
                const uint64_t n_omega = config.dynamical.num_omega_points;
                std::vector<double> omega_grid(n_omega);
                const double omega_step = (config.dynamical.omega_max -
                    config.dynamical.omega_min) /
                    static_cast<double>(std::max<uint64_t>(1, n_omega - 1));
                for (uint64_t i = 0; i < n_omega; ++i)
                    omega_grid[i] = config.dynamical.omega_min +
                                    static_cast<double>(i) * omega_step;
                const std::vector<double> one_T{temperature};
                ed::observables::FtlmDynamicalOptions kopts;
                kopts.krylov_dim   = params.krylov_dim;
                kopts.num_samples  = params.num_samples;
                kopts.broadening   = params.broadening;
                kopts.energy_shift = ground_state_energy;
                kopts.tolerance    = params.tolerance;
                kopts.random_seed  = params.random_seed;
                kopts.global_n     = N;
                ed::matvec::CpuBackend be;
                // Both observable slots must deduce ONE ApplyO type:
                // wrap the lambdas uniformly.
                std::function<void(const Complex*, Complex*, uint64_t)>
                    O1_sel = O_func;
                std::function<void(const Complex*, Complex*, uint64_t)>
                    O2_sel = O_func;
                if (have_op2) O2_sel = O2_func;
                auto res = ed::observables::detail::
                    ftlm_dynamical_kernel_via_backend_multitemp(
                        be, H_func, O1_sel, O2_sel,
                        N, omega_grid, one_T, kopts);
                results.frequencies            = res[0].omega;
                results.spectral_function      = res[0].spectral_real;
                results.spectral_function_imag = res[0].spectral_imag;
                results.spectral_error         = res[0].spectral_error_real;
                results.spectral_error_imag    = res[0].spectral_error_imag;
                results.total_samples          = res[0].total_samples;
            }

            // Save results for this temperature to HDF5 (rank 0 only -- the
            // shared HDF5 file is not concurrently writable from multiple
            // ranks without HDF5-MPI parallel I/O, which we don't link here).
            if (rank == 0) {
                std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
                std::string op_name = config.dynamical.output_prefix;
                if (config.dynamical.num_temp_bins > 1) {
                    op_name += "_T" + std::to_string(temperature);
                }
                HDF5IO::saveDynamicalResponseFull(
                    h5_file, op_name,
                    results.frequencies, results.spectral_function, results.spectral_function_imag,
                    results.spectral_error, results.spectral_error_imag,
                    results.total_samples, temperature
                );
                std::cout << "Results saved to HDF5: " << h5_file << " (" << op_name << ")\n";
            }
        }
    }
    
    if (rank == 0) {
        std::cout << "\nDynamical response complete.\n";
        std::cout << "Frequency range: [" << config.dynamical.omega_min << ", " << config.dynamical.omega_max << "]\n";
        std::cout << "Number of points: " << config.dynamical.num_omega_points << "\n";
    }
}

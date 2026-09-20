// =============================================================================
// src/cli/workflows/wf_static.cpp
//
// `compute_static_response_workflow`: thermal expectation values, either from
// the configured observable pairs (FTLM static correlations, with the MPI
// master-worker shape) or from the legacy `--static-operator` file path.
//
// Moved verbatim out of src/cli/workflows.cpp (WP14). See
// workflows_internal.h for the file map.
// =============================================================================

#include "workflows_internal.h"

/**
 * @brief Compute static response (thermal expectation values)
 */
void compute_static_response_workflow(const EDConfig& config) {
    // Plain locals (not a structured binding): C++17 forbids capturing
    // structured bindings in lambdas (clang enforces; gcc extension).
    int rank = 0, size = 1;
    std::tie(rank, size) = get_mpi_rank_size_safe();

    if (rank == 0) {
        std::cout << "\nStatic Response Calculation\n";

#ifdef WITH_CUDA
        if (config.static_resp.use_gpu) {
            if (config.system.use_fixed_sz) {
                std::cout << "  GPU: enabled (fixed-Sz; transverse channels "
                             "with operators that change Sz still fall back "
                             "to CPU until cross-sector wiring lands)\n";
            } else {
                std::cout << "  GPU: enabled (config-based operator path; "
                             "legacy --static-operator path is CPU-only)\n";
            }
        }
#else
        if (config.static_resp.use_gpu) {
            std::cout << "  GPU: requested but unavailable (build has no CUDA "
                         "support; using CPU)\n";
        }
#endif
    }

    bool use_config_operators = config.static_resp.operator_file.empty() ||
                                config.static_resp.operator_type != "sum";

    auto wh = build_workflow_hamiltonian(
        config, rank,
        config.system.use_fixed_sz ? "Fixed-Sz static response" : nullptr);
    const bool use_fixed_sz   = wh.use_fixed_sz;
    const uint64_t N          = wh.N;
    auto& ham_full            = wh.ham_full;
    auto& ham_fs              = wh.ham_fs;
    auto& H_func              = wh.H_func;
    Operator& ham             = wh.ham_ref();
    (void)ham_full; (void)ham_fs;

    // Setup parameters
    StaticResponseParameters params;
    params.num_samples = config.static_resp.num_random_states;
    params.krylov_dim = config.static_resp.krylov_dim;
    params.random_seed = config.static_resp.random_seed;
    
    // Ensure output directory exists
    create_directory_mpi_safe(config.workflow.output_dir);
    
    if (rank == 0) {
        std::cout << "Random states: " << params.num_samples << "\n";
        std::cout << "Krylov dimension: " << params.krylov_dim << "\n";
        std::cout << "Temperature range: [" << config.static_resp.temp_min << ", " << config.static_resp.temp_max << "]\n";
    }
    
    if (use_config_operators) {
        // ============================================================
        // Configuration-based operator construction (canonical `ED dssf` knobs)
        // ============================================================
        if (rank == 0) {
            std::cout << "\nUsing configuration-based operator construction\n";
            std::cout << "  Operator type: " << config.static_resp.operator_type << "\n";
            std::cout << "  Basis: " << config.static_resp.basis << "\n";
            std::cout << "  Spin combinations: " << config.static_resp.spin_combinations << "\n";
        }
        
        // Parse configuration
        auto spin_combinations = parse_spin_combinations(config.static_resp.spin_combinations);
        spin_combinations = filter_fixed_sz_transverse_channels(
            spin_combinations,
            config.system.use_fixed_sz,
            (config.static_resp.basis == "xyz"),
            rank,
            "compute_static_response_workflow");
        auto momentum_points = parse_momentum_points(config.static_resp.momentum_points);
        auto polarization = parse_polarization(config.static_resp.polarization);
        
        // Get positions file
        std::string positions_file = config.system.hamiltonian_dir + "/positions.dat";
        
        // Determine fixed-Sz parameters (shadows the outer use_fixed_sz; same value)
        bool use_fixed_sz = config.system.use_fixed_sz;
        int64_t n_up = (use_fixed_sz && config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        
        // Construct operators (audit #2: also obtain shared_ptr<FixedSzOperator>
        // arrays so the CPU apply path correctly dispatches at the fixed-Sz dim).
        ed::dssf::OperatorSpec _spec;
        _spec.operator_type     = config.static_resp.operator_type;
        _spec.basis             = config.static_resp.basis;
        _spec.spin_combinations = spin_combinations;
        _spec.momentum_points   = momentum_points;
        _spec.polarization      = polarization;
        _spec.theta             = config.static_resp.theta;
        _spec.unit_cell_size    = config.static_resp.unit_cell_size;
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
            std::cout << "Constructed " << obs_1.size() << " operator pair(s)\n";
        }
        
        // ============================================================
        // MPI Task Distribution (per-operator-pair sharding)
        // ============================================================
        
        // Build task list: each task is an operator pair
        struct StaticTask {
            int op_idx;
            size_t weight;  // estimated cost (number of samples * krylov dimension)
        };
        
        std::vector<StaticTask> all_tasks;
        int num_operators = obs_1.size();
        
        if (rank == 0) {
            // Create tasks
            for (int o = 0; o < num_operators; o++) {
                // Weight is proportional to samples, krylov dimension, and temperature points
                size_t weight = params.num_samples * params.krylov_dim * config.static_resp.num_temp_points;
                all_tasks.push_back({o, weight});
            }
            
            // Sort by weight (descending) for better load balance
            std::sort(all_tasks.begin(), all_tasks.end(), 
                      [](const StaticTask& a, const StaticTask& b) { return a.weight > b.weight; });
            
            std::cout << "\nMPI Parallelization: " << all_tasks.size() << " tasks = "
                      << num_operators << " operators\n";
            std::cout << "Running on " << size << " MPI rank(s)\n";
        }
        
        // Broadcast task count
        int num_tasks = all_tasks.size();
        #ifdef WITH_MPI
        // Audit fix: guard collective calls when MPI is not initialised
        // (workflow gets exercised from Catch2 unit tests).
        int mpi_inited_bcast = 0;
        MPI_Initialized(&mpi_inited_bcast);
        if (mpi_inited_bcast) {
            MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank != 0) {
                all_tasks.resize(num_tasks);
            }

            // Broadcast all tasks
            for (int i = 0; i < num_tasks; i++) {
                int op = all_tasks[i].op_idx;
                size_t w = all_tasks[i].weight;
                MPI_Bcast(&op, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&w, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
                if (rank != 0) {
                    all_tasks[i] = {op, w};
                }
            }
        }
        #endif
        
        // Lambda to process a single task
        // Audit 2026-07-31 (H2): compute/write halves -- see the
        // dynamical lane's compute_task_single for the rationale (single
        // HDF5 writer under MPI).
        auto compute_static_task =
            [&](const StaticTask& task) -> StaticResponseResults {
            int op_idx = task.op_idx;

            StaticResponseResults results;
            
            // Consolidation Family 3: one backend-generic FTLM static kernel
            // (ftlm_static_correlation_via_backend_multitemp) replaces BOTH the
            // GPUFTLMSolver::computeStaticCorrelation GPU path and the host
            // compute_static_response. select_backend picks CudaBackend when
            // --use-gpu is set and the device matvec is available, else
            // CpuBackend.
            {
                const int nTp = std::max<int>(
                    1, static_cast<int>(config.static_resp.num_temp_points));
                std::vector<double> temps(nTp);
                const double tstep = (config.static_resp.temp_max -
                    config.static_resp.temp_min) / std::max(1, nTp - 1);
                for (int i = 0; i < nTp; ++i)
                    temps[i] = config.static_resp.temp_min + i * tstep;

                ed::LinearOperator& H_op = ham;
                ed::LinearOperator& O1_op = config.system.use_fixed_sz
                    ? static_cast<ed::LinearOperator&>(*obs_1_fs[op_idx])
                    : static_cast<ed::LinearOperator&>(obs_1[op_idx]);
                ed::LinearOperator& O2_op = config.system.use_fixed_sz
                    ? static_cast<ed::LinearOperator&>(*obs_2_fs[op_idx])
                    : static_cast<ed::LinearOperator&>(obs_2[op_idx]);

                ed::BackendConstraints bc;
                bc.allow_gpu     = config.static_resp.use_gpu;
                bc.allow_mpi     = false;
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
                            "static FTLM: requires a CpuBackend or CudaBackend; "
                            "distributed backends are not wired.");
                    } else {
                        ed::observables::detail::FtlmStaticOptions kopts;
                        kopts.krylov_dim  = params.krylov_dim;
                        kopts.num_samples = params.num_samples;
                        kopts.tolerance   = 1e-10;
                        kopts.random_seed = params.random_seed;
                        kopts.global_n    = H_op.geometry().global_dim;

                        auto mv_h  = H_op.template bind<B>();
                        auto mv_o1 = O1_op.template bind<B>();
                        auto mv_o2 = O2_op.template bind<B>();
                        auto sr = ed::observables::detail::
                            ftlm_static_correlation_via_backend_multitemp(
                                *backend_uptr, mv_h, mv_o1, mv_o2,
                                H_op.geometry().local_dim, temps, kopts);
                        results.temperatures         = std::move(sr.temperatures);
                        results.expectation          = std::move(sr.expectation);
                        results.expectation_error    = std::move(sr.expectation_error);
                        results.variance             = std::move(sr.variance);
                        results.variance_error       = std::move(sr.variance_error);
                        results.susceptibility       = std::move(sr.susceptibility);
                        results.susceptibility_error = std::move(sr.susceptibility_error);
                        results.total_samples        = sr.total_samples;
                    }
                }, variant);
            }
            
            return results;
        };

        auto write_static_task = [&](const StaticTask& task,
                                     const StaticResponseResults& results) {
            std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
            HDF5IO::saveStaticResponse(
                h5_file, names[task.op_idx],
                results.temperatures, results.expectation, results.expectation_error,
                results.variance, results.variance_error,
                results.susceptibility, results.susceptibility_error,
                results.total_samples
            );
        };

        auto process_task = [&](const StaticTask& task) -> bool {
            write_static_task(task, compute_static_task(task));
            return true;
        };
        
        // Execute tasks with dynamic work distribution
        int local_processed_count = 0;
        
        #ifdef WITH_MPI
        if (size > 1) {
            // Audit 2026-07-31 (H2): single-writer master-worker -- see
            // run_mpi_master_worker_single_writer. Workers compute and
            // ship packed results; only rank 0 touches ed_results.h5;
            // failed tasks complete the protocol (recorded + reported)
            // instead of hanging the master.
            auto pack_static = [](const StaticResponseResults& r) {
                std::vector<double> p;
                pack_scalar(p, static_cast<double>(r.total_samples));
                pack_array(p, r.temperatures);
                pack_array(p, r.expectation);
                pack_array(p, r.expectation_error);
                pack_array(p, r.variance);
                pack_array(p, r.variance_error);
                pack_array(p, r.susceptibility);
                pack_array(p, r.susceptibility_error);
                return p;
            };
            auto unpack_static = [](const std::vector<double>& p) {
                StaticResponseResults r;
                std::size_t pos = 0;
                r.total_samples =
                    static_cast<uint64_t>(unpack_scalar(p, pos));
                r.temperatures         = unpack_array(p, pos);
                r.expectation          = unpack_array(p, pos);
                r.expectation_error    = unpack_array(p, pos);
                r.variance             = unpack_array(p, pos);
                r.variance_error       = unpack_array(p, pos);
                r.susceptibility       = unpack_array(p, pos);
                r.susceptibility_error = unpack_array(p, pos);
                return r;
            };
            local_processed_count += run_mpi_master_worker_single_writer(
                rank, size, num_tasks,
                [&](int t) {
                    return pack_static(compute_static_task(all_tasks[t]));
                },
                [&](int t, const std::vector<double>& p) {
                    write_static_task(all_tasks[t], unpack_static(p));
                },
                "static-response master-worker");
        } else
        #endif
        {
            // Sequential execution (no MPI or single rank)
            for (int task_idx = 0; task_idx < num_tasks; task_idx++) {
                if (rank == 0) {
                    std::cout << "  Processing operator: " << names[all_tasks[task_idx].op_idx] << "\n";
                }

                if (process_task(all_tasks[task_idx])) {
                    local_processed_count++;
                }
            }
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
        
        if (config.static_resp.operator_file.empty()) {
            std::cerr << "Error: --static-operator=<file> is required for static response\n";
            return;
        }
        
        std::string op_path = config.system.hamiltonian_dir + "/" + config.static_resp.operator_file;
        Operator op(config.system.num_sites, config.system.spin_length);
        op.loadFromInterAllFile(op_path);
        {
            // Optional legacy companion ``*.Trans.dat`` with the same basename as
            // ``*.InterAll.dat`` (num two-body lines may be zero).  One-body terms
            // from ``Trans.dat`` are needed for quadrupole / magnetostriction
            // observables built solely from on-site ``S^{±}``.
            std::string trans_path = op_path;
            const std::string suf = ".InterAll.dat";
            if (trans_path.size() >= suf.size() &&
                trans_path.compare(trans_path.size() - suf.size(), suf.size(), suf) == 0) {
                trans_path.replace(trans_path.size() - suf.size(), suf.size(), ".Trans.dat");
                if (std::filesystem::exists(trans_path)) {
                    op.loadFromFile(trans_path);
                    if (rank == 0) {
                        std::cout << "  Loaded one-body companion operator file: "
                                  << trans_path << "\n";
                    }
                }
            }
        }
        
        auto O_func = [&op](const Complex* in, Complex* out, uint64_t dim) {
            op.apply(in, out, dim);
        };
        
        // Compute response
        StaticResponseResults results;
        
        if (config.static_resp.single_operator_mode) {
            // Single operator expectation value: ⟨O⟩
            if (rank == 0) std::cout << "Computing thermal expectation value ⟨O⟩...\n";
            results = compute_thermal_expectation_value(
                H_func, O_func, N, params,
                config.static_resp.temp_min,
                config.static_resp.temp_max,
                config.static_resp.num_temp_points,
                config.workflow.output_dir
            );
        } else if (config.static_resp.operator2_file == "__connected_hamiltonian__") {
            // Magnetostriction / thermal expansion: ∂T⟨O⟩ =
            // (⟨OH⟩ - ⟨O⟩⟨H⟩) / T², evaluated with the selected thermal method.
            if (rank == 0) {
                std::cout << "Computing connected O-H thermal expansion "
                          << "(⟨OH⟩ - ⟨O⟩⟨H⟩) / T²...\n";
            }
            if (config.method == DiagonalizationMethod::LTLM) {
                LTLMParameters ltlm_params;
                ltlm_params.krylov_dim = config.thermal.ltlm_krylov_dim;
                ltlm_params.ground_state_krylov = config.thermal.ltlm_ground_krylov;
                ltlm_params.max_iterations = config.diag.max_iterations;
                ltlm_params.tolerance = config.diag.tolerance;
                ltlm_params.full_reorthogonalization = config.thermal.ltlm_full_reorth;
                ltlm_params.reorth_frequency = config.thermal.ltlm_reorth_freq;
                ltlm_params.random_seed = config.thermal.ltlm_seed;
                ltlm_params.store_intermediate = config.thermal.ltlm_store_data;
                ltlm_params.compute_error_bars = false;
                ltlm_params.num_samples = 1;

                results = compute_connected_qh_response_ltlm(
                    H_func, O_func, N, ltlm_params,
                    config.static_resp.temp_min,
                    config.static_resp.temp_max,
                    config.static_resp.num_temp_points,
                    config.workflow.output_dir
                );
            } else {
                results = compute_connected_qh_response(
                    H_func, O_func, N, params,
                    config.static_resp.temp_min,
                    config.static_resp.temp_max,
                    config.static_resp.num_temp_points,
                    config.workflow.output_dir
                );
            }
        } else if (!config.static_resp.operator2_file.empty()) {
            // Two different operators: ⟨O₁†O₂⟩
            if (rank == 0) std::cout << "Computing two-operator static response ⟨O₁†O₂⟩...\n";
            std::string op2_path = config.system.hamiltonian_dir + "/" + config.static_resp.operator2_file;
            Operator op2(config.system.num_sites, config.system.spin_length);
            op2.loadFromInterAllFile(op2_path);
            {
                std::string trans_path = op2_path;
                const std::string suf = ".InterAll.dat";
                if (trans_path.size() >= suf.size() &&
                    trans_path.compare(trans_path.size() - suf.size(), suf.size(), suf) == 0) {
                    trans_path.replace(trans_path.size() - suf.size(), suf.size(), ".Trans.dat");
                    if (std::filesystem::exists(trans_path)) {
                        op2.loadFromFile(trans_path);
                        if (rank == 0) {
                            std::cout << "  Loaded one-body companion for operator2: "
                                      << trans_path << "\n";
                        }
                    }
                }
            }

            auto O2_func = [&op2](const Complex* in, Complex* out, uint64_t dim) {
                op2.apply(in, out, dim);
            };

            results = compute_static_response(
                H_func, O_func, O2_func, N, params,
                config.static_resp.temp_min,
                config.static_resp.temp_max,
                config.static_resp.num_temp_points,
                config.workflow.output_dir
            );
        } else {
            // Same operator: ⟨O†O⟩ (default two-point correlation)
            if (rank == 0) std::cout << "Computing static response ⟨O†O⟩...\n";
            results = compute_static_response(
                H_func, O_func, O_func, N, params,
                config.static_resp.temp_min,
                config.static_resp.temp_max,
                config.static_resp.num_temp_points,
                config.workflow.output_dir
            );
        }

        // Save results to HDF5 -- only rank 0 writes to avoid concurrent
        // overwrites of the shared ed_results.h5 in the legacy single-task
        // path. (The MPI-sharded `use_config_operators` branch above writes
        // per-task and serialises through the master.)
        if (rank == 0) {
            std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
            HDF5IO::saveStaticResponse(
                h5_file, config.static_resp.output_prefix,
                results.temperatures, results.expectation, results.expectation_error,
                results.variance, results.variance_error,
                results.susceptibility, results.susceptibility_error,
                results.total_samples
            );
            std::cout << "Static response saved to HDF5: " << h5_file << "\n";
        }
    }
}

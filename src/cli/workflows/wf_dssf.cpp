// =============================================================================
// src/cli/workflows/wf_dssf.cpp
//
// `compute_ground_state_dssf_workflow`: the T=0 dynamical structure factor via
// continued fraction, with the same-sector lane (OpenMP over observable pairs)
// and the cross-sector lane (`ed::dssf::CrossSectorObservable`).
//
// Moved verbatim out of src/cli/workflows.cpp (WP14). See
// workflows_internal.h for the file map.
// =============================================================================

#include "workflows_internal.h"

/**
 * @brief Compute ground state dynamical spin structure factor (T=0 DSSF)
 * 
 * Uses the continued fraction method for efficient ground state dynamics:
 * S(q,ω) = -1/π Im⟨GS| O†(-q) 1/(ω + E₀ - H + iη) O(q) |GS⟩
 * 
 * This is optimal for 32-site ED where:
 * - Fixed-Sz sector has 601M states (~9GB per vector)
 * - Only need to store 2-3 Lanczos vectors (not full spectrum)
 * - Continued fraction avoids explicit eigendecomposition
 */
void compute_ground_state_dssf_workflow(const EDConfig& config) {
    // Plain locals (not a structured binding): C++17 forbids capturing
    // structured bindings in lambdas (clang enforces; gcc extension).
    int rank = 0, size = 1;
    std::tie(rank, size) = get_mpi_rank_size_safe();

    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "Computing Ground State DSSF (T=0)\n";
        std::cout << "==========================================\n";
        std::cout << "Using continued fraction method for optimal efficiency\n";
        if (config.dynamical.use_gpu || config.static_resp.use_gpu) {
            std::cout << "  Note: --use-gpu / --dyn-use-gpu is not implemented "
                         "for --ground-state-dssf; using CPU\n";
        }
    }

    auto wh = build_workflow_hamiltonian(
        config, rank,
        config.system.use_fixed_sz ? "Fixed-Sz sector"
                                   : "Full Hilbert space");
    const bool use_fixed_sz = wh.use_fixed_sz;
    const int64_t n_up      = wh.n_up;
    const uint64_t N        = wh.N;
    auto& ham_full          = wh.ham_full;
    auto& ham_fs            = wh.ham_fs;
    (void)ham_full; (void)ham_fs; (void)size;

    // Adapter: the GS-DSSF kernel below expects a `void(const*, *, int)`
    // signature instead of the `uint64_t` one `wh.H_func` carries.
    auto H_apply_int = [&H = wh.H_func](
        const Complex* in, Complex* out, int dim) {
        H(in, out, static_cast<uint64_t>(dim));
    };

    create_directory_mpi_safe(config.workflow.output_dir);

    // Setup ground state DSSF parameters
    GroundStateDSSFParameters gs_params;
    gs_params.krylov_dim = config.dynamical.krylov_dim > 0 ? config.dynamical.krylov_dim : 300;
    gs_params.omega_min = config.dynamical.omega_min;
    gs_params.omega_max = config.dynamical.omega_max;
    gs_params.num_omega_points = config.dynamical.num_omega_points;
    gs_params.broadening = config.dynamical.broadening;
    gs_params.tolerance = config.diag.tolerance;
    gs_params.full_reorthogonalization = true;

    if (rank == 0) {
        std::cout << "Krylov dimension: " << gs_params.krylov_dim << "\n";
        std::cout << "Frequency range: [" << gs_params.omega_min
                  << ", " << gs_params.omega_max << "]\n";
        std::cout << "Frequency points: " << gs_params.num_omega_points << "\n";
        std::cout << "Broadening (eta): " << gs_params.broadening << "\n";
    }

    // Parse configuration for operators.
    auto spin_combinations = parse_spin_combinations(config.dynamical.spin_combinations);
    spin_combinations = filter_fixed_sz_transverse_channels(
        spin_combinations,
        use_fixed_sz,
        (config.dynamical.basis == "xyz"),
        rank,
        "compute_ground_state_dssf_workflow");
    auto momentum_points = parse_momentum_points(config.dynamical.momentum_points);
    auto polarization = parse_polarization(config.dynamical.polarization);
    const std::string positions_file = config.system.hamiltonian_dir + "/positions.dat";

    if (rank == 0) {
        std::cout << "Operator type: " << config.dynamical.operator_type << "\n";
        std::cout << "Basis: " << config.dynamical.basis << "\n";
        std::cout << "Momentum points: " << momentum_points.size() << "\n";
        std::cout << "Spin combinations: " << spin_combinations.size() << "\n";
    }

    // ------------------------------------------------------------------
    // Audit #1 (full): partition spin pairs into same-sector (delta=0,
    // both Sz) vs cross-sector (delta != 0 but matched). Cross-sector
    // pairs are dispatched to the new kernel below.
    //
    // For now this dispatcher only handles operator_type == "sum" with
    // ladder basis (Sp/Sm/Sz). Other operator types fall through to the
    // legacy path; mixed-delta pairs were already filtered out above.
    // ------------------------------------------------------------------
    auto delta_of = [](int op) -> int {
        switch (op) {
            case 0: return -1;  // S+ (physics raising; bit 1->0)
            case 1: return +1;  // S- (physics lowering; bit 0->1)
            default: return 0;  // Sz
        }
    };

    std::vector<std::pair<int, int>> same_sector_pairs;
    std::vector<std::pair<int, int>> cross_sector_pairs;
    const bool use_xyz_basis = (config.dynamical.basis == "xyz");
    const bool cross_dispatch_supported =
        use_fixed_sz && !use_xyz_basis &&
        config.dynamical.operator_type == "sum";

    for (const auto& pr : spin_combinations) {
        const int d1 = delta_of(pr.first);
        const int d2 = delta_of(pr.second);
        if (cross_dispatch_supported && d1 == d2 && d1 != 0) {
            cross_sector_pairs.push_back(pr);
        } else {
            same_sector_pairs.push_back(pr);
        }
    }

    if (rank == 0 && !cross_sector_pairs.empty()) {
        std::cout << "\n  Audit #1 (full): " << cross_sector_pairs.size()
                  << " cross-sector pair(s) will be dispatched to "
                  << "compute_ground_state_dssf_cross_sector.\n";
    }

    // Construct same-sector operator pairs (audit #2: build via
    // build_observable_pairs to also obtain the typed shared_ptr<FixedSzOperator>
    // arrays needed for correct CPU dispatch under fixed-Sz).
    std::vector<Operator> obs_1, obs_2;
    std::vector<std::string> names;
    std::vector<std::shared_ptr<FixedSzOperator>> obs_1_fs, obs_2_fs;
    if (!same_sector_pairs.empty()) {
        ed::dssf::OperatorSpec _spec;
        _spec.operator_type     = config.dynamical.operator_type;
        _spec.basis             = config.dynamical.basis;
        _spec.spin_combinations = same_sector_pairs;
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
        obs_1    = std::move(_pairs.obs_1);
        obs_2    = std::move(_pairs.obs_2);
        names    = std::move(_pairs.names);
        obs_1_fs = std::move(_pairs.obs_1_fs);
        obs_2_fs = std::move(_pairs.obs_2_fs);
    }

    if (rank == 0) {
        std::cout << "Constructed " << names.size()
                  << " same-sector operator pair(s)\n";
    }

    if (rank == 0) {
        std::cout << "\n--- Finding ground state ---\n";
    }

    if (N > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::cerr << "Error: Hilbert space dimension " << N
                      << " exceeds INT_MAX. Use fixed-Sz or symmetry reduction.\n";
        }
        return;
    }

    ComplexVector ground_state(N);
    double ground_state_energy = 0.0;
    const std::string h5_file = config.workflow.output_dir + "/ed_results.h5";
    bool gs_loaded = false;

    if (HDF5IO::fileExists(h5_file)) {
        try {
            auto eigenvalues = HDF5IO::loadEigenvalues(h5_file);
            if (!eigenvalues.empty()) {
                ground_state_energy = eigenvalues[0];
                auto gs_vec = HDF5IO::loadEigenvector(h5_file, 0);
                if (gs_vec.size() == N) {
                    std::copy(gs_vec.begin(), gs_vec.end(), ground_state.begin());
                    gs_loaded = true;
                    if (rank == 0) {
                        std::cout << "Loaded ground state from HDF5: E0 = "
                                  << ground_state_energy << "\n";
                    }
                }
            }
        } catch (const std::exception&) {
            if (rank == 0) {
                std::cout << "Could not load ground state from HDF5, will compute...\n";
            }
        }
    }

    if (!gs_loaded) {
        ground_state_energy = find_ground_state_lanczos(
            H_apply_int, N, gs_params.krylov_dim, gs_params.tolerance,
            gs_params.full_reorthogonalization, gs_params.reorth_frequency,
            ground_state);
        if (rank == 0) {
            std::cout << "Computed ground state: E0 = " << ground_state_energy << "\n";
            try {
                std::string h5_path = HDF5IO::createOrOpenFile(config.workflow.output_dir);
                HDF5IO::saveEigenvalues(h5_path, {ground_state_energy});
                std::vector<Complex> gs_vec(ground_state.begin(), ground_state.end());
                HDF5IO::saveEigenvector(h5_path, 0, gs_vec);
                std::cout << "Saved ground state to HDF5: " << h5_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save ground state to HDF5: "
                          << e.what() << "\n";
            }
        }
    }

    // ------------------------------------------------------------------
    // Same-sector dispatch (existing code path).
    // ------------------------------------------------------------------
    if (!names.empty() && rank == 0) {
        std::cout << "\n--- Computing S(q,ω) for " << names.size()
                  << " same-sector operator pair(s) ---\n";
    }

    std::vector<int> my_tasks;
    for (int i = rank; i < (int)names.size(); i += size) {
        my_tasks.push_back(i);
    }

    // Wave 3.4 of the SOTA Performance rollout (May 2026): when running
    // single-rank (or when each rank still owns multiple pairs in the
    // round-robin partition) the (alpha, beta) pairs are completely
    // independent -- each launches its own Lanczos build against the
    // same ground state. OpenMP-parallelise across pairs so 8 pairs
    // on a single workstation hit all cores at once, instead of
    // chaining serial inner Lanczos runs. The HDF5 layer is wrapped
    // in a critical section -- HDF5 itself is not (always) thread
    // safe and the per-pair writes are tiny vs the Lanczos compute.
    //
    // Cap the outer team at ``ED_DSSF_PAIR_THREADS`` (default =
    // min(num_pairs, omp_max_threads / 2)); nested OMP keeps the
    // inner SpMV / BLAS-1 multi-threaded too.
    const int n_my_pairs = static_cast<int>(my_tasks.size());
    int pair_threads = 1;
    if (n_my_pairs > 1) {
#ifdef _OPENMP
        const int max_threads = omp_get_max_threads();
#else
        const int max_threads = 1;
#endif
        pair_threads = std::min(n_my_pairs, std::max(1, max_threads / 2));
        if (const char* env = ed::env::raw("ED_DSSF_PAIR_THREADS")) {
            try {
                const long t = std::stol(env);
                if (t >= 1 && t <= max_threads) {
                    pair_threads = std::min(
                        n_my_pairs, static_cast<int>(t));
                }
            } catch (...) {
                // malformed env: keep default.
            }
        }
#ifdef _OPENMP
        if (pair_threads > 1) omp_set_max_active_levels(2);
#endif
    }

    // Audit 2026-07-31 (H2): buffer per-pair results and write AFTER the
    // loop through a rank token ring. The old in-loop write was thread-
    // serialized (omp critical) but rank-CONCURRENT on the shared
    // ed_results.h5 -- HDF5 file locking races across processes.
    std::vector<DynamicalResponseResults> pair_results(
        static_cast<std::size_t>(n_my_pairs));
    #pragma omp parallel for num_threads(pair_threads) \
        if (pair_threads > 1) schedule(dynamic, 1)
    for (int t = 0; t < n_my_pairs; ++t) {
        const int op_idx = my_tasks[t];
        if (rank == 0) {
            #pragma omp critical(stdout_lock)
            std::cout << "[Rank " << rank << "] Processing: "
                      << names[op_idx] << "\n";
        }
        // Audit #2: dispatch via FixedSzOperator under fixed-Sz so the
        // CPU apply path uses the typed override instead of the sliced
        // Operator::apply (which would throw on the smaller dim).
        auto O1_func = [&obs_1, &obs_1_fs, op_idx, use_fixed_sz]
            (const Complex* in, Complex* out, int dim) {
            if (use_fixed_sz) obs_1_fs[op_idx]->apply(in, out, static_cast<uint64_t>(dim));
            else              obs_1[op_idx].apply(in, out, static_cast<uint64_t>(dim));
        };
        auto O2_func = [&obs_2, &obs_2_fs, op_idx, use_fixed_sz]
            (const Complex* in, Complex* out, int dim) {
            if (use_fixed_sz) obs_2_fs[op_idx]->apply(in, out, static_cast<uint64_t>(dim));
            else              obs_2[op_idx].apply(in, out, static_cast<uint64_t>(dim));
        };
        pair_results[static_cast<std::size_t>(t)] =
            compute_ground_state_cross_correlation(
                H_apply_int, O1_func, O2_func, ground_state,
                ground_state_energy, N, gs_params);
    }

    // Token-ring write: one rank in the file at a time (uniform barrier
    // count on every rank -- my_tasks lengths differ per rank, so the
    // ring must sit OUTSIDE the task loop).
    {
        const int ring = std::max(size, 1);
        for (int r = 0; r < ring; ++r) {
            if (rank == r) {
                for (int t = 0; t < n_my_pairs; ++t) {
                    const int op_idx = my_tasks[t];
                    const auto& results =
                        pair_results[static_cast<std::size_t>(t)];
                    std::string h5_path =
                        HDF5IO::createOrOpenFile(config.workflow.output_dir);
                    std::string op_name = "ground_state_dssf/" + names[op_idx];
                    HDF5IO::saveDynamicalResponseFull(
                        h5_path, op_name,
                        results.frequencies, results.spectral_function,
                        results.spectral_function_imag,
                        results.spectral_error, results.spectral_error_imag,
                        1, 0.0);
                    std::cout << "[Rank " << rank << "] Saved to HDF5: "
                              << op_name << "\n";
                }
            }
            #ifdef WITH_MPI
            if (size > 1) MPI_Barrier(MPI_COMM_WORLD);
            #endif
        }
    }

    // ------------------------------------------------------------------
    // Cross-sector dispatch (audit #1 full). Each (Q, op_pair) builds:
    //   * dst sector at n_up + delta_n_up
    //   * a FixedSzOperator inner Hamiltonian at the dst sector
    //   * two CrossSectorObservable instances for O1 and O2
    //   * routes through compute_ground_state_dssf_cross_sector.
    // ------------------------------------------------------------------
    if (!cross_sector_pairs.empty()) {
        if (rank == 0) {
            std::cout << "\n--- Computing cross-sector S(q,ω) for "
                      << cross_sector_pairs.size() << " pair(s) x "
                      << momentum_points.size() << " momentum point(s) ---\n";
        }
        // Cache one inner Hamiltonian per dst_n_up across pairs at the
        // same delta. With ladder basis the only deltas are +-1.
        const std::string interaction_file =
            config.system.hamiltonian_dir + "/" + config.system.interaction_file;
        const std::string single_site_file =
            config.system.hamiltonian_dir + "/" + config.system.single_site_file;
        std::map<int64_t, std::shared_ptr<FixedSzOperator>> ham_dst_cache;
        auto get_ham_dst = [&](int64_t dst_n_up)
            -> std::shared_ptr<FixedSzOperator> {
            auto it = ham_dst_cache.find(dst_n_up);
            if (it != ham_dst_cache.end()) return it->second;
            auto h = std::make_shared<FixedSzOperator>(
                config.system.num_sites, config.system.spin_length, dst_n_up);
            h->loadFromInterAllFile(interaction_file);
            h->loadFromFile(single_site_file);
            if (!config.system.three_body_file.empty()) {
                const std::string tb_file =
                    config.system.hamiltonian_dir + "/" + config.system.three_body_file;
                if (std::filesystem::exists(tb_file)) {
                    h->loadThreeBodyTerm(tb_file);
                }
            }
            ham_dst_cache[dst_n_up] = h;
            return h;
        };

        // Flatten (Q, pair) into a global task list for round-robin MPI.
        struct CrossTask {
            std::vector<double> Q;
            int op_type_1;
            int op_type_2;
            std::string name;
        };
        std::vector<CrossTask> cross_tasks;
        for (const auto& Q : momentum_points) {
            for (const auto& pr : cross_sector_pairs) {
                CrossTask t;
                t.Q = Q;
                t.op_type_1 = pr.first;
                t.op_type_2 = pr.second;
                // Mirror legacy naming: non-XYZ ladder basis swaps first
                // slot (Sp <-> Sm) so the spectral label matches the
                // physics convention <0|O1†|n><n|O2|0>. We keep raw
                // op_type_1/2 for kernel construction.
                int first_label = (pr.first == 2) ? 2 : (1 - pr.first);
                auto opname = [](int op) -> const char* {
                    switch (op) { case 2: return "Sz"; case 0: return "Sp";
                                  case 1: return "Sm"; default: return "?"; }
                };
                std::stringstream name_ss;
                name_ss << opname(first_label) << opname(pr.second)
                        << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2];
                t.name = name_ss.str();
                cross_tasks.push_back(t);
            }
        }

        std::vector<int> my_cross_tasks;
        for (int i = rank; i < (int)cross_tasks.size(); i += size) {
            my_cross_tasks.push_back(i);
        }
        std::vector<std::pair<std::string, DynamicalResponseResults>>
            cross_results;
        cross_results.reserve(my_cross_tasks.size());

        for (int idx : my_cross_tasks) {
            const auto& task = cross_tasks[idx];
            const int delta = delta_of(task.op_type_1);
            const int64_t dst_n_up = n_up + delta;
            if (dst_n_up < 0 ||
                dst_n_up > static_cast<int64_t>(config.system.num_sites)) {
                if (rank == 0) {
                    std::cerr << "  Skipping " << task.name
                              << ": dst sector n_up=" << dst_n_up
                              << " is out of range; spectrum identically zero.\n";
                }
                continue;
            }

            if (rank == 0) {
                std::cout << "[Rank " << rank << "] Cross-sector ["
                          << task.name << "], dst n_up=" << dst_n_up << "\n";
            }

            auto ham_dst = get_ham_dst(dst_n_up);
            const uint64_t dst_dim = ham_dst->getFixedSzDim();

            // Build src and dst fixed-Sz Sum operators (basis-aligned op).
            // We use src's transform_data_ (independent of n_up by
            // construction) as the operator definition; dst supplies the
            // basis + Lin lookup table.
            auto make_sum_op = [&](std::int64_t nup, int op_type) {
                auto p = std::make_shared<FixedSzOperator>(
                    config.system.num_sites, config.system.spin_length, nup);
                ed::ops::add_sum(*p, static_cast<uint64_t>(op_type), task.Q,
                                 positions_file, /*use_xyz=*/false);
                return p;
            };
            auto src_op1 = make_sum_op(n_up, task.op_type_1);
            auto dst_op1 = make_sum_op(dst_n_up, task.op_type_1);
            ed::dssf::CrossSectorObservable O1_cross(
                src_op1, dst_op1,
                src_op1->transform_data_, config.system.spin_length);

            auto src_op2 = make_sum_op(n_up, task.op_type_2);
            auto dst_op2 = make_sum_op(dst_n_up, task.op_type_2);
            ed::dssf::CrossSectorObservable O2_cross(
                src_op2, dst_op2,
                src_op2->transform_data_, config.system.spin_length);

            auto H_inner_apply = [ham_dst](const Complex* in, Complex* out, int dim) {
                ham_dst->apply(in, out, static_cast<uint64_t>(dim));
            };
            auto O1_apply = O1_cross.as_apply_function();
            auto O2_apply = O2_cross.as_apply_function();
            auto O1_apply_int = [O1_apply](const Complex* in, Complex* out, int dim) {
                O1_apply(in, out, static_cast<std::size_t>(dim));
            };
            auto O2_apply_int = [O2_apply](const Complex* in, Complex* out, int dim) {
                O2_apply(in, out, static_cast<std::size_t>(dim));
            };

            // Audit 2026-07-31 (H2): buffer, write in the token ring
            // below -- the in-loop write was rank-concurrent on the
            // shared HDF5 file.
            cross_results.emplace_back(
                "ground_state_dssf/" + task.name,
                compute_ground_state_dssf_cross_sector(
                    H_inner_apply, O1_apply_int, O2_apply_int,
                    ground_state, ground_state_energy, N, dst_dim,
                    gs_params));
        }

        // Token-ring write (H2): one rank in the shared file at a time.
        // The `cross_sector_pairs.empty()` gate above is config-derived
        // and therefore uniform across ranks, so the barrier counts
        // match on every rank.
        for (int r = 0; r < std::max(size, 1); ++r) {
            if (rank == r) {
                for (const auto& [op_name, results] : cross_results) {
                    std::string h5_path = HDF5IO::createOrOpenFile(
                        config.workflow.output_dir);
                    HDF5IO::saveDynamicalResponseFull(
                        h5_path, op_name,
                        results.frequencies, results.spectral_function,
                        results.spectral_function_imag,
                        results.spectral_error, results.spectral_error_imag,
                        1, 0.0);
                    std::cout << "[Rank " << rank << "] Saved to HDF5: "
                              << op_name << "\n";
                }
            }
            #ifdef WITH_MPI
            if (size > 1) MPI_Barrier(MPI_COMM_WORLD);
            #endif
        }
    }

    #ifdef WITH_MPI
    // size > 1, like the two barriers above: rank/size come from
    // get_mpi_rank_size_safe(), which reports (0, 1) when MPI was never
    // initialised. Unguarded, this collective aborts the process the moment
    // this workflow body is called from somewhere that is not the ED binary
    // -- MPI_Init lives in main(), so an in-process caller (the WP9.8
    // dssf_run binding) reaches here with MPI inactive and Open MPI kills
    // the process with "MPI_Barrier before MPI_INIT" AFTER the results have
    // been written.
    if (size > 1) MPI_Barrier(MPI_COMM_WORLD);
    #endif

    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "Ground State DSSF Complete\n";
        std::cout << "Results saved to: " << config.workflow.output_dir
                  << "/ed_results.h5\n";
        std::cout << "==========================================\n";
    }
}

// =============================================================================
// src/cli/workflows/wf_diagonalize.cpp
//
// The two eigensolve entry points -- `run_standard_workflow` (single sector or
// the ALL_SZ_SECTORS sweep) and `run_streaming_symmetry_workflow` -- plus
// `compute_thermodynamics`, which turns a finite spectrum into the
// /thermodynamics HDF5 group.
//
// Moved verbatim out of src/cli/workflows.cpp (WP14). See
// workflows_internal.h for the file map.
// =============================================================================

#include "workflows_internal.h"

// ============================================================================
// WORKFLOW FUNCTIONS
// ============================================================================

/**
 * @brief Run standard diagonalization workflow
 *
 * Full Unified-Interface Collapse, Wave C2 (May 2026): collapses what used
 * to be a single `ed::exact_diagonalization(directory, method, params, ...)`
 * call into the canonical three-step unified shape:
 *
 *     OperatorSpec -> ed::make_operator(spec) -> ed::workflows::solve(...)
 *
 * Two behavioural lanes are preserved:
 *   1. Standard: single sector (full Hilbert space, or fixed-Sz when the
 *      caller sets `use_fixed_sz`).
 *   2. ALL_SZ_SECTORS: when `params.full_sz_split && params.method == FULL`,
 *      loop over every Sz sector (n_up = 0..num_sites) via the same
 *      factory + orchestrator path, merge the eigenvalues, and sort.
 *      This replaces the dispatcher's internal `exact_diagonalization_all_sz_sectors`
 *      branch with an explicit, auditable loop in the CLI.
 *
 * HDF5 output is now emitted explicitly here (the orchestrator does not
 * auto-save eigenvalues from the Lanczos / Krylov-Schur lane today), so
 * the CLI's output contract is unchanged.
 */
EDResults run_standard_workflow(const EDConfig& config) {
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    create_directory_mpi_safe(params.output_dir);

    // Force off the symmetry axis (this is the non-symmetry lane).
    params.use_symmetry = false;
    params.use_fixed_sz = config.system.use_fixed_sz;
    if (config.system.use_fixed_sz && params.n_up < 0) {
        params.n_up = (config.system.n_up >= 0)
            ? config.system.n_up
            : static_cast<int64_t>(config.system.num_sites / 2);
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto build_spec = [&](std::optional<int> sector_n_up) {
        ed::OperatorSpec spec;
        spec.source    = ed::DirectoryPath{config.system.hamiltonian_dir};
        spec.num_sites = config.system.num_sites;
        spec.spin_l    = config.system.spin_length;
        if (sector_n_up.has_value()) {
            spec.fixed_sz = sector_n_up.value();
        }
        return spec;
    };

    ed::workflows::SolveOptions opts =
        ed_adapter::toSolveOptions(params, config.method);

    EDResults results;

    const bool all_sz_split =
        params.full_sz_split && config.method == DiagonalizationMethod::FULL;

    if (all_sz_split) {
        // ALL_SZ_SECTORS lane: loop over n_up = 0..num_sites, projecting
        // to each Sz sector via FixedSzOperator + workflows::solve, then
        // merge and globally sort. Matches the legacy
        // `exact_diagonalization_all_sz_sectors` behaviour now driven
        // explicitly from the CLI rather than buried in the dispatcher.
        std::vector<double> all_evals;
        for (uint64_t n_up = 0; n_up <= config.system.num_sites; ++n_up) {
            ed::OperatorSpec spec = build_spec(static_cast<int>(n_up));
            auto sector_op  = ed::make_operator(std::move(spec));
            ed::workflows::SolveOptions sopts = opts;
            sopts.use_fixed_sz = true;
            sopts.n_up         = static_cast<int>(n_up);
            auto r = ed::workflows::solve(*sector_op, sopts);
            all_evals.insert(all_evals.end(),
                             r.eigenvalues.begin(), r.eigenvalues.end());
        }
        std::sort(all_evals.begin(), all_evals.end());
        if (params.num_eigenvalues > 0 &&
            all_evals.size() > params.num_eigenvalues) {
            all_evals.resize(params.num_eigenvalues);
        }
        results.eigenvalues = std::move(all_evals);
    } else {
        ed::OperatorSpec spec = build_spec(
            params.use_fixed_sz ? std::optional<int>(static_cast<int>(params.n_up))
                                : std::nullopt);
        auto op = ed::make_operator(std::move(spec));
        auto r  = ed::workflows::solve(*op, opts);
        results.eigenvalues = std::move(r.eigenvalues);
    }

    // Explicit HDF5 save for CLI parity. The orchestrator currently only
    // auto-saves eigenvectors when `compute_vectors=true`; the CLI's
    // contract has always been that eigenvalues land on disk regardless.
    if (!params.output_dir.empty() && !results.eigenvalues.empty()) {
        try {
            std::string h5_path = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(h5_path, results.eigenvalues);
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to save eigenvalues to HDF5: "
                      << e.what() << "\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Print results summary
    if (!results.eigenvalues.empty()) {
        std::cout << "\n  Lowest eigenvalues:\n";
        size_t show = std::min(results.eigenvalues.size(), (size_t)5);
        for (size_t i = 0; i < show; i++) {
            std::cout << "    E[" << i << "] = " << std::fixed << std::setprecision(10) 
                      << results.eigenvalues[i] << "\n";
        }
        if (results.eigenvalues.size() > 5) {
            std::cout << "    ... (" << (results.eigenvalues.size() - 5) << " more)\n";
        }
    }
    
    std::cout << "\n  Time: " << std::fixed << std::setprecision(2) << duration / 1000.0 << " s\n";
    
    return results;
}

/**
 * @brief Run symmetry-exploiting diagonalization workflow
 *
 * Uses streaming symmetry path which handles both CPU and GPU
 * uniformly.  The streaming approach never materialises explicit block
 * matrices — it keeps the per-sector orbit data in memory so the GPU
 * symmetrized matvec kernel can use it directly.  On the CPU side the
 * same matrix-free operator is wrapped in a lambda and forwarded to the
 * standard solver dispatch (Lanczos, Block Lanczos, Davidson, etc.).
 */
EDResults run_streaming_symmetry_workflow(const EDConfig& config) {
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    create_directory_mpi_safe(params.output_dir);

    // Full Unified-Interface Collapse, Wave C2 (May 2026): same factory +
    // orchestrator pattern as run_standard_workflow, but with the
    // `streaming_symmetry` axis flipped on (and `fixed_sz` honoured when
    // the caller requests it).  The streaming-symmetry kernel walks every
    // symmetry sector internally; the orchestrator iterates them via
    // `StreamingSymmetryOperator::SectorView`.
    params.use_symmetry         = true;
    params.use_fixed_sz         = config.system.use_fixed_sz;
    if (config.system.use_fixed_sz && params.n_up < 0) {
        params.n_up = (config.system.n_up >= 0)
            ? config.system.n_up
            : static_cast<int64_t>(config.system.num_sites / 2);
    }
    params.basis_cache_dir      = config.workflow.basis_cache_dir;
    params.precompute_basis_only = config.workflow.precompute_basis_only;

    auto start = std::chrono::high_resolution_clock::now();

    ed::OperatorSpec spec;
    spec.source             = ed::DirectoryPath{config.system.hamiltonian_dir};
    spec.num_sites          = config.system.num_sites;
    spec.spin_l             = config.system.spin_length;
    spec.streaming_symmetry = true;
    if (params.use_fixed_sz) {
        spec.fixed_sz = static_cast<int>(params.n_up);
    }

    ed::workflows::SolveOptions opts =
        ed_adapter::toSolveOptions(params, config.method);

    // Operator-collapse Phase 3 (Jun 2026): the per-sector loop now builds the
    // symmetry sector set directly through ``ed::make_sector_operators_tagged``
    // -- a flat, compacted vector of standalone ``SectorOperator``s plus their
    // ``SectorTag``s -- instead of materialising a monolithic
    // ``FixedSzStreamingSymmetryOperator`` and walking it through
    // ``StreamingSymmetryHandle``. The CSR-free lazy-rep memory optimisation is
    // preserved end-to-end (the tagged factory hands out the same lazy
    // operators the handle used to, and the tag dim is read from Pass 1.5
    // without materialising any orbit CSR). The returned operators are
    // self-contained (no external carrier to keep alive).
    EDResults results;
    // Across-sector MPI (Level 1, SectorDistributor): the factory dim-balances
    // the |G| irrep sectors across ranks (Burnside dims + greedy packing), so
    // each rank BUILDS (orbit walk + per-sector RepSectorData) and SOLVES only
    // its own sectors -- both construction time AND per-rank memory distribute.
    // The merged spectrum is Allgatherv'd after the loop. The sectors are
    // independent eigenproblems reusing the existing single-node solve, so the
    // distributed result is bit-identical to the single-rank run.
    const auto [mpi_rank, mpi_size] = get_mpi_rank_size_safe();
    ed::SectorOperatorSet sector_set =
        ed::make_sector_operators_tagged(spec, mpi_rank, mpi_size);

    if (mpi_size == 1 && sector_set.operators.empty()) {
        throw std::runtime_error(
            "run_streaming_symmetry_workflow: make_sector_operators_tagged "
            "returned no symmetry sectors. Check the automorphism_results/ "
            "directory and the InterAll.dat deck.");
    }
    // Under across-sector MPI a rank may legitimately own zero sectors (e.g.
    // more ranks than surviving irreps); it simply contributes nothing to the
    // Allgatherv below. The empty-deck error is only meaningful single-rank.

    // ``selected_sectors`` filters by RAW irrep index (the tag's
    // ``sector_index``), matching the legacy ``filter_sectors`` semantics:
    // empty => keep all; out-of-range indices silently dropped.
    const bool keep_all_sectors = opts.selected_sectors.empty();
    const std::set<std::size_t> selected_set(opts.selected_sectors.begin(),
                                             opts.selected_sectors.end());

    // The factory already handed this rank only its assigned sectors (raw-index
    // partition), so the loop simply solves every sector in the local set. The
    // inner per-sector path must stay free of cross-rank MPI collectives: ranks
    // own DIFFERENT sectors, so any collective on MPI_COMM_WORLD (the per-sector
    // ``create_directory_mpi_safe`` / orchestrator HDF5 setup, or an MpiBackend
    // dot/nrm2) would mismatch and deadlock. We therefore force the inner solve
    // rank-local (allow_mpi=false) and suppress per-sector on-disk output under
    // multi-rank, emitting only the merged result from rank 0.
    std::vector<double>                      all_eigs;
    std::vector<ed::SectorTag>               touched_tags;
    std::vector<std::vector<double>>         eigs_per_sector;
    for (std::size_t i = 0; i < sector_set.operators.size(); ++i) {
        const ed::SectorTag& tag = sector_set.tags[i];
        if (!keep_all_sectors && selected_set.count(tag.sector_index) == 0) {
            continue;
        }
        auto& sec = sector_set.operators[i];
        if (!sec || sec->dim() == 0) continue;
        ed::workflows::SolveOptions sopts = opts;
        sopts.num_eigs = std::min<std::size_t>(opts.num_eigs, sec->dim());
        // Strip the per-sector filter / use_symmetry flag from the
        // inner call so the orchestrator does not try to re-enter the
        // streaming loop on a single sector operator.
        sopts.selected_sectors.clear();
        sopts.use_symmetry = false;
        // Per-sector HDF5 save: `sector_<idx>/ed_results.h5`. This is
        // the canonical layout that the symmetrized CLI workflow emits.
        // Under across-sector MPI it is suppressed (collective I/O would
        // deadlock when ranks own different sectors -- see note above);
        // the inner solve's output_dir is cleared so it does no I/O.
        if (mpi_size > 1) {
            // Force the inner per-sector solve RANK-LOCAL: select_backend would
            // otherwise pick MpiBackend under MPI_COMM_WORLD (its ctor does
            // MPI_Comm_dup, its dot/nrm2 do MPI_Allreduce -- all COMM_WORLD
            // collectives). With ranks owning DIFFERENT sectors those collectives
            // mismatch and deadlock. Pinning to the Cpu (or single-rank Cuda)
            // backend keeps each sector solve independent; the only cross-rank
            // step is the final Allgatherv of the spectrum.
            sopts.backend.allow_mpi     = false;
            sopts.backend.allow_mpi_gpu = false;
            sopts.output_dir.clear();
        } else if (!opts.output_dir.empty()) {
            sopts.output_dir =
                opts.output_dir + "/sector_" + std::to_string(tag.sector_index);
        }
        auto sr = ed::workflows::solve(*sec, sopts);
        touched_tags.push_back(tag);
        eigs_per_sector.push_back(sr.eigenvalues);
        all_eigs.insert(all_eigs.end(),
                        sr.eigenvalues.begin(), sr.eigenvalues.end());
        if (!sopts.output_dir.empty() && !sr.eigenvalues.empty()) {
            try {
                create_directory_mpi_safe(sopts.output_dir);
                std::string h5 = HDF5IO::createOrOpenFile(sopts.output_dir);
                HDF5IO::saveEigenvalues(h5, sr.eigenvalues);
                // SOTA: persist the irrep quantum-number tag alongside
                // the eigenvalues so downstream consumers (Python
                // loaders, postproc scripts) can identify the irrep
                // block on disk. HDF5IO does not expose a dedicated
                // helper for the int vector; the sector directory name
                // (``sector_<k>``) plus the per-sector listing in the
                // CLI summary covers identification at the directory
                // level. (A future HDF5IO::saveQuantumNumbers would
                // move that metadata inside the file too.)
                (void) tag;
            } catch (const std::exception& e) {
                std::cerr << "  Warning: sector " << tag.sector_index
                          << " HDF5 save failed: " << e.what() << "\n";
            }
        }
    }
    // Across-sector MPI: recombine the per-rank sub-spectra into the full merged
    // spectrum on every rank (exact union of the independent sector solves).
    all_eigs = mpi_allgatherv_doubles(all_eigs);
    std::sort(all_eigs.begin(), all_eigs.end());
    // Note: the merged eigenvalue list is NOT truncated at
    // `params.num_eigenvalues`; each sector contributes its own
    // ``min(num_eigs, sector_dim)`` and the global vector is the
    // union, sorted. The per-sector truncation is already enforced
    // above via ``sopts.num_eigs = std::min(opts.num_eigs, sec->dim())``.
    results.eigenvalues = std::move(all_eigs);

    // SOTA upgrade (May 2026): print per-sector breakdown so the CLI
    // user can see which irrep each low-lying eigenvalue came from
    // (matches the SOTA-level output of HPhi / EDLib / QuSpin).
    if (!touched_tags.empty()) {
        std::cout << "\n  Per-sector eigenvalues:\n";
        for (std::size_t s = 0; s < touched_tags.size(); ++s) {
            const auto& t = touched_tags[s];
            std::cout << "    sector " << t.sector_index
                      << "  dim=" << t.sector_dim;
            if (!t.quantum_numbers.empty()) {
                std::cout << "  QN=[";
                for (std::size_t q = 0; q < t.quantum_numbers.size(); ++q) {
                    std::cout << (q == 0 ? "" : ",") << t.quantum_numbers[q];
                }
                std::cout << "]";
            }
            if (t.n_up >= 0) std::cout << "  n_up=" << t.n_up;
            std::cout << "\n";
            const auto& evs = eigs_per_sector[s];
            std::size_t show = std::min<std::size_t>(evs.size(), 3);
            for (std::size_t i = 0; i < show; ++i) {
                std::cout << "      E[" << i << "] = "
                          << std::fixed << std::setprecision(10)
                          << evs[i] << "\n";
            }
        }
    }

    // Top-level merged HDF5 save (matches the legacy global summary).
    // Under across-sector MPI only rank 0 writes -- every rank holds the same
    // Allgatherv'd spectrum, so a single writer avoids a file-write race.
    if (mpi_rank == 0 && !params.output_dir.empty() && !results.eigenvalues.empty()) {
        try {
            std::string h5_path = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(h5_path, results.eigenvalues);
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to save eigenvalues to HDF5: "
                      << e.what() << "\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Print results summary
    if (!results.eigenvalues.empty()) {
        std::cout << "\n  Lowest eigenvalues:\n";
        size_t show = std::min(results.eigenvalues.size(), (size_t)5);
        for (size_t i = 0; i < show; i++) {
            std::cout << "    E[" << i << "] = " << std::fixed << std::setprecision(10) 
                      << results.eigenvalues[i] << "\n";
        }
        if (results.eigenvalues.size() > 5) {
            std::cout << "    ... (" << (results.eigenvalues.size() - 5) << " more)\n";
        }
    }
    
    std::cout << "\n  Time: " << std::fixed << std::setprecision(2) << duration / 1000.0 << " s\n";
    
    return results;
}

// ---------------------------------------------------------------------------
// run_disk_streaming_workflow() and run_chunked_symmetry_workflow() were
// retired in the matvec-unification cleanup (Phase 7.2). They were ultra-low-
// memory single-node fallbacks (-> std::FILE-backed sector cache; two-pass
// orbit discovery) intended for >64M-state Hilbert spaces on RAM-starved
// machines. They never had a unit test, never had a Python binding, did not
// support GPU, and were quietly slower than the streaming path even when
// they fit in RAM. The right answer for those sizes is MPI/distributed
// (which is now first-class in matvec-unification) -- there's no point in
// shipping the disk/chunked CPU-only specialisations alongside it.
// ---------------------------------------------------------------------------

/**
 * @brief Compute thermodynamics from eigenvalue spectrum
 */
void compute_thermodynamics(const std::vector<double>& eigenvalues, const EDConfig& config) {
    if (eigenvalues.empty()) return;
    
    auto thermo_data = calculate_thermodynamics_from_spectrum(
        eigenvalues,
        config.thermal.temp_min,
        config.thermal.temp_max,
        config.thermal.num_temp_bins
    );
    
    // Save results to HDF5
    try {
        std::string hdf5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "energy", thermo_data.energy);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "specific_heat", thermo_data.specific_heat);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "entropy", thermo_data.entropy);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "free_energy", thermo_data.free_energy);
        std::cout << "  Saved thermodynamic data to HDF5\n";
        
    } catch (const std::exception& e) {
        std::cerr << "  Error: Failed to save thermodynamics to HDF5: " << e.what() << std::endl;
    }
}

// =============================================================================
// src/cli/workflows/wf_common.cpp
//
// Shared plumbing for the CLI workflows: the argv string parsers, the fixed-Sz
// transverse-channel guard, the `construct_operators_from_config` wrapper, the
// MPI rank/size + Allgatherv helpers, the `build_workflow_hamiltonian`
// preamble every compute_*_workflow starts from, the master-worker payload
// pack/unpack helpers, and `print_eigenvalue_summary`.
//
// Moved verbatim out of src/cli/workflows.cpp (WP14). See
// workflows_internal.h for the file map.
// =============================================================================

#include "workflows_internal.h"

/**
 * @brief Parse spin combinations from string format
 * Format: "op1,op2;op3,op4;..." where op is 0=Sp/Sx, 1=Sm/Sy, 2=Sz
 */
std::vector<std::pair<int, int>> parse_spin_combinations(const std::string& spin_combinations_str) {
    std::vector<std::pair<int, int>> spin_combinations;
    std::stringstream ss(spin_combinations_str);
    std::string pair_str;
    
    while (std::getline(ss, pair_str, ';')) {
        std::stringstream pair_ss(pair_str);
        std::string op1_str, op2_str;
        
        if (std::getline(pair_ss, op1_str, ',') && std::getline(pair_ss, op2_str)) {
            try {
                int op1 = std::stoi(op1_str);
                int op2 = std::stoi(op2_str);
                
                if (op1 >= 0 && op1 <= 2 && op2 >= 0 && op2 <= 2) {
                    spin_combinations.push_back({op1, op2});
                } else {
                    std::cerr << "Warning: Invalid spin operator " << op1 << "," << op2 
                              << ". Operators must be 0, 1, or 2." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse spin combination: " << pair_str << std::endl;
            }
        }
    }
    
    if (spin_combinations.empty()) {
        std::cerr << "Warning: No valid spin combinations provided. Using default SzSz." << std::endl;
        spin_combinations = {{2, 2}};
    }
    
    return spin_combinations;
}

// ----------------------------------------------------------------------------
// Audit item #1 (partial): cross-sector DSSF correctness guard.
//
// In fixed-Sz mode, single-site transverse channels (S+/S-/Sx/Sy, op != 2)
// change the magnetisation by ±1 and therefore land in a *different*
// Sz sector than the one we have built a basis for. The legacy
// `FixedSz*Operator::apply` silently drops those contributions because
// they fail the `popcount(new_basis) == n_up_` filter, which means the
// computed spectrum is *zero* rather than the physical ⟨S^α(-Q,t) S^α(Q)⟩.
//
// Audit item #1 (full) -- this filter has been narrowed:
//   * Pairs that are *legitimately zero by sector orthogonality* (i.e.
//     delta(op1) != delta(op2), e.g. (S+, S-) or (Sz, S+)) are dropped
//     here with a clear warning.
//   * Pairs that are *cross-sector but legitimate* (delta(op1) == delta(op2)
//     and not (Sz, Sz)), like (S+, S+) or (S-, S-), are KEPT and routed
//     by the workflow (`compute_ground_state_dssf_workflow`) to the new
//     `compute_ground_state_dssf_cross_sector` kernel via
//     `ed::dssf::CrossSectorObservable`.
//   * (Sz, Sz) is the same-sector path and is unchanged.
//
// XYZ-basis (Sx, Sy) decomposition is NOT yet handled by the cross-sector
// dispatcher and is dropped here with a follow-up TODO.
//
// Returns the filtered list and writes a one-shot diagnostic to stdout
// (rank 0 only) describing what was removed and why.
// ----------------------------------------------------------------------------
std::vector<std::pair<int, int>>
filter_fixed_sz_transverse_channels(
    const std::vector<std::pair<int, int>>& spin_combinations,
    bool use_fixed_sz,
    bool use_xyz_basis,
    int rank,
    const char* workflow_label)
{
    if (!use_fixed_sz) return spin_combinations;

    // delta_n_up shift induced by each op_type in this codebase's
    // convention (bit=1 carries the popcount):
    //   op_type=0 ("S+", physics raising)  -> bit 1->0  -> delta = -1
    //   op_type=1 ("S-", physics lowering) -> bit 0->1  -> delta = +1
    //   op_type=2 ("Sz")                    -> diagonal -> delta =  0
    auto delta_of = [](int op) -> int {
        switch (op) {
            case 0: return -1;
            case 1: return +1;
            case 2: return  0;
            default: return  0;
        }
    };

    std::vector<std::pair<int, int>> kept;
    std::vector<std::pair<int, int>> dropped_zero;
    std::vector<std::pair<int, int>> dropped_xyz;
    kept.reserve(spin_combinations.size());

    for (const auto& pr : spin_combinations) {
        const int d1 = delta_of(pr.first);
        const int d2 = delta_of(pr.second);
        // XYZ basis: Sx and Sy are linear combos of S+ and S-, which
        // requires the cross-sector dispatcher to issue *two* sub-calls
        // and combine; not yet implemented (audit #1 follow-up).
        if (use_xyz_basis && (pr.first != 2 || pr.second != 2)) {
            dropped_xyz.push_back(pr);
        } else if (d1 != d2) {
            // Sector orthogonality: spectrum is identically zero.
            dropped_zero.push_back(pr);
        } else {
            kept.push_back(pr);
        }
    }

    if (rank == 0 && (!dropped_zero.empty() || !dropped_xyz.empty())) {
        const char* op0 = use_xyz_basis ? "Sx" : "Sp";
        const char* op1 = use_xyz_basis ? "Sy" : "Sm";
        const char* op2 = "Sz";
        const char* op_names[3] = {op0, op1, op2};
        std::cerr << "\n";
        std::cerr << "  ============================================================\n";
        std::cerr << "  NOTE (" << workflow_label << ", audit #1):\n";
        if (!dropped_zero.empty()) {
            std::cerr << "    Dropping " << dropped_zero.size()
                      << " spin pair(s) with delta_n_up(op1) != delta_n_up(op2);\n"
                      << "    these are identically zero by sector orthogonality:\n";
            for (const auto& pr : dropped_zero) {
                std::cerr << "      - " << op_names[pr.first]
                          << op_names[pr.second] << "\n";
            }
        }
        if (!dropped_xyz.empty()) {
            std::cerr << "    Dropping " << dropped_xyz.size()
                      << " XYZ-basis pair(s); cross-sector dispatch for Sx/Sy is\n"
                      << "    not yet wired (audit #1 follow-up). Workaround:\n"
                      << "    use the ladder basis (Sp/Sm/Sz) instead of xyz, or\n"
                      << "    re-run without --fixed-sz.\n";
            for (const auto& pr : dropped_xyz) {
                std::cerr << "      - " << op_names[pr.first]
                          << op_names[pr.second] << "\n";
            }
        }
        std::cerr << "  ============================================================\n\n";
    }

    if (kept.empty()) {
        if (rank == 0) {
            std::cerr << "  ERROR: all requested spin channels were filtered out "
                      << "(see warning above). Aborting " << workflow_label << ".\n";
        }
        throw std::invalid_argument(
            std::string("ed::") + workflow_label +
            ": no valid spin channels remain after fixed-Sz cross-sector "
            "filtering (audit item #1).");
    }

    return kept;
}

/**
 * @brief Parse momentum points from string format
 * Format: "Qx1,Qy1,Qz1;Qx2,Qy2,Qz2;..." (values are multiplied by π)
 */
std::vector<std::vector<double>> parse_momentum_points(const std::string& momentum_str) {
    std::vector<std::vector<double>> momentum_points;
    std::stringstream mom_ss(momentum_str);
    std::string point_str;
    
    while (std::getline(mom_ss, point_str, ';')) {
        std::stringstream point_ss(point_str);
        std::string coord_str;
        std::vector<double> point;
        
        while (std::getline(point_ss, coord_str, ',')) {
            try {
                double coord = std::stod(coord_str);
                coord *= M_PI;  // Scale to π
                point.push_back(coord);
            } catch (...) {
                std::cerr << "Warning: Failed to parse momentum coordinate: " << coord_str << std::endl;
            }
        }
        
        if (point.size() == 3) {
            momentum_points.push_back(point);
        } else {
            std::cerr << "Warning: Momentum point must have 3 coordinates, got " << point.size() << std::endl;
        }
    }
    
    // Use default momentum points if none provided or parsing failed
    if (momentum_points.empty()) {
        momentum_points = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 2.0 * M_PI}
        };
        std::cout << "Using default momentum points: (0,0,0) and (0,0,2π)" << std::endl;
    }
    
    return momentum_points;
}

/**
 * @brief Parse polarization vector from string format
 * Format: "px,py,pz" (will be normalized)
 */
std::vector<double> parse_polarization(const std::string& pol_str) {
    std::vector<double> polarization = {1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0};  // default
    
    std::stringstream pol_ss(pol_str);
    std::string coord_str;
    std::vector<double> pol_temp;
    
    while (std::getline(pol_ss, coord_str, ',')) {
        try {
            double coord = std::stod(coord_str);
            pol_temp.push_back(coord);
        } catch (...) {
            std::cerr << "Warning: Failed to parse polarization coordinate: " << coord_str << std::endl;
        }
    }
    
    if (pol_temp.size() == 3) {
        // Normalize the polarization vector
        double norm = std::sqrt(pol_temp[0]*pol_temp[0] + pol_temp[1]*pol_temp[1] + pol_temp[2]*pol_temp[2]);
        if (norm > 1e-10) {
            polarization = {pol_temp[0]/norm, pol_temp[1]/norm, pol_temp[2]/norm};
            std::cout << "Using custom polarization: (" << polarization[0] << "," 
                      << polarization[1] << "," << polarization[2] << ")" << std::endl;
        } else {
            std::cerr << "Warning: Polarization vector has zero norm, using default" << std::endl;
        }
    } else {
        std::cerr << "Warning: Polarization must have 3 coordinates, got " << pol_temp.size() << std::endl;
    }
    
    return polarization;
}

/**
 * @brief Construct operators based on configuration.
 *
 * Thin wrapper around ed::dssf::build_observable_pairs (P1.10). The
 * implementation moved to src/dssf/operator_spec.cpp; this function exists
 * purely so the four legacy call sites in ed_main.cpp keep compiling.
 * New code should call ed::dssf::build_observable_pairs directly.
 */
void construct_operators_from_config(
    const std::string& operator_type,
    const std::string& basis,
    const std::vector<std::pair<int, int>>& spin_combinations,
    const std::vector<std::vector<double>>& momentum_points,
    const std::vector<double>& polarization,
    double theta,
    uint64_t unit_cell_size,
    uint64_t num_sites,
    float spin_length,
    bool use_fixed_sz,
    int64_t n_up,
    const std::string& positions_file,
    std::vector<Operator>& obs_1_out,
    std::vector<Operator>& obs_2_out,
    std::vector<std::string>& names_out
) {
    ed::dssf::OperatorSpec spec;
    spec.operator_type    = operator_type;
    spec.basis            = basis;
    spec.spin_combinations = spin_combinations;
    spec.momentum_points  = momentum_points;
    spec.polarization     = polarization;
    spec.theta            = theta;
    spec.unit_cell_size   = unit_cell_size;
    spec.num_sites        = num_sites;
    spec.spin_length      = spin_length;
    spec.use_fixed_sz     = use_fixed_sz;
    spec.n_up             = n_up;
    spec.positions_file   = positions_file;
    auto pairs = ed::dssf::build_observable_pairs(spec);
    obs_1_out = std::move(pairs.obs_1);
    obs_2_out = std::move(pairs.obs_2);
    names_out = std::move(pairs.names);
}

// ============================================================================
// Internal workflow helpers (May 2026): factored out of the otherwise
// near-identical preambles in compute_*_workflow. Each function used to
// inline a ~50-line MPI rank + Hamiltonian + H_func + Hilbert-dim block;
// the helpers below collapse that to four lines per workflow without any
// behavioural change. Audit #2 (FixedSz->Operator path) handling is
// preserved by dispatching the apply() lambda on the concrete operator.
// ============================================================================

/// Returns (rank, size) from `MPI_COMM_WORLD`, falling back to (0, 1)
/// when MPI is unavailable or `MPI_Init` has not been called. Mirrors
/// the guard in `create_directory_mpi_safe`.
std::pair<int, int> get_mpi_rank_size_safe() {
    int rank = 0, size = 1;
#ifdef WITH_MPI
    int mpi_inited = 0;
    MPI_Initialized(&mpi_inited);
    if (mpi_inited) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
#endif
    return {rank, size};
}

/// Across-sector MPI (Level 1, SectorDistributor): gather variable-length
/// per-rank double vectors into the full vector on EVERY rank. No-op when
/// single-rank. The |G| irrep sectors are independent eigenproblems, so each
/// rank solves a disjoint subset and this Allgatherv is the exact (bit-identical)
/// recombination of the single-node sector loop.
std::vector<double>
mpi_allgatherv_doubles(const std::vector<double>& local) {
#ifdef WITH_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) {
        int size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1) {
            const int n = static_cast<int>(local.size());
            std::vector<int> counts(static_cast<std::size_t>(size));
            MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            std::vector<int> displs(static_cast<std::size_t>(size));
            int total = 0;
            for (int i = 0; i < size; ++i) { displs[i] = total; total += counts[i]; }
            std::vector<double> global(static_cast<std::size_t>(total));
            MPI_Allgatherv(local.data(), n, MPI_DOUBLE,
                           global.data(), counts.data(), displs.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            return global;
        }
    }
#endif
    return local;
}

/// Construct the Hamiltonian, three-body terms, sector dimension, and
/// `H_func` apply lambda from `config`. When `verbose_label` is non-null
/// and `rank == 0`, prints the three-body load and the sector-dim
/// summary the workflows used to print inline.
WorkflowHamiltonian
build_workflow_hamiltonian(const EDConfig& config,
                           int rank,
                           const char* verbose_label)
{
    WorkflowHamiltonian out;
    out.use_fixed_sz = config.system.use_fixed_sz;
    out.n_up = (out.use_fixed_sz && config.system.n_up >= 0)
                   ? config.system.n_up
                   : static_cast<int64_t>(config.system.num_sites) / 2;

    const std::string interaction_file =
        config.system.hamiltonian_dir + "/" + config.system.interaction_file;
    const std::string single_site_file =
        config.system.hamiltonian_dir + "/" + config.system.single_site_file;

    if (out.use_fixed_sz) {
        out.ham_fs = std::make_shared<FixedSzOperator>(
            config.system.num_sites, config.system.spin_length, out.n_up);
        out.ham_fs->loadFromInterAllFile(interaction_file);
        out.ham_fs->loadFromFile(single_site_file);
    } else {
        out.ham_full = std::make_shared<Operator>(
            config.system.num_sites, config.system.spin_length);
        out.ham_full->loadFromInterAllFile(interaction_file);
        out.ham_full->loadFromFile(single_site_file);
    }

    if (!config.system.three_body_file.empty()) {
        const std::string three_body_file =
            config.system.hamiltonian_dir + "/" + config.system.three_body_file;
        if (std::filesystem::exists(three_body_file)) {
            if (rank == 0 && verbose_label) {
                std::cout << "Loading three-body terms from: "
                          << three_body_file << "\n";
            }
            if (out.use_fixed_sz) out.ham_fs->loadThreeBodyTerm(three_body_file);
            else                  out.ham_full->loadThreeBodyTerm(three_body_file);
        }
    }

    if (out.use_fixed_sz) {
        out.N = 1;
        for (int64_t i = 0; i < out.n_up; i++) {
            out.N = out.N * (config.system.num_sites - i) / (i + 1);
        }
        if (rank == 0 && verbose_label) {
            std::cout << verbose_label << ": dim=" << out.N
                      << " (n_up=" << out.n_up << ")\n";
        }
    } else {
        out.N = 1ULL << config.system.num_sites;
        if (rank == 0 && verbose_label) {
            std::cout << verbose_label << ": full Hilbert space dim="
                      << out.N << "\n";
        }
    }

    // The lambda captures shared_ptrs by value so it survives any local
    // lifetime questions in the caller.
    auto ham_full_cap   = out.ham_full;
    auto ham_fs_cap     = out.ham_fs;
    const bool fz_cap   = out.use_fixed_sz;
    out.H_func = [ham_full_cap, ham_fs_cap, fz_cap](
        const Complex* in, Complex* outp, uint64_t dim) {
        if (fz_cap) ham_fs_cap->apply(in, outp, dim);
        else        ham_full_cap->apply(in, outp, dim);
    };

    return out;
}

#ifdef WITH_MPI
// Payload packing for the master-worker driver in workflows_internal.h:
// [n_arrays fields...] as flat doubles. Each array is emitted as
// (len, data...); scalars first.
void pack_scalar(std::vector<double>& p, double v) { p.push_back(v); }
void pack_array(std::vector<double>& p, const std::vector<double>& v) {
    p.push_back(static_cast<double>(v.size()));
    p.insert(p.end(), v.begin(), v.end());
}
double unpack_scalar(const std::vector<double>& p, std::size_t& pos) {
    return p.at(pos++);
}
std::vector<double> unpack_array(const std::vector<double>& p,
                                 std::size_t& pos) {
    const std::size_t n = static_cast<std::size_t>(p.at(pos++));
    std::vector<double> v(p.begin() + static_cast<std::ptrdiff_t>(pos),
                          p.begin() + static_cast<std::ptrdiff_t>(pos + n));
    pos += n;
    return v;
}
#endif  // WITH_MPI

/**
 * @brief Print eigenvalue summary
 */
void print_eigenvalue_summary(const std::vector<double>& eigenvalues, uint64_t max_show) {
    std::cout << "\nEigenvalues:\n";
    for (size_t i = 0; i < eigenvalues.size() && i < max_show; i++) {
        std::cout << "  " << i << ": " << std::setprecision(12) << eigenvalues[i] << "\n";
    }
    if (eigenvalues.size() > max_show) {
        std::cout << "  ... (" << eigenvalues.size() - max_show << " more)\n";
    }
}


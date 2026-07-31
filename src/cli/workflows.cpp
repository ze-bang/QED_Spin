// =============================================================================
// src/cli/workflows.cpp
//
// Implementations of the CLI-driven workflow entry points declared in
// `include/ed/cli/workflows.h`. Extracted verbatim from `src/apps/ed_main.cpp`
// in P1.11 (DSSF PR-B / audit §3.10): pure structural lift-and-shift, no
// behaviour change. Audit dashboards and unit tests should be bit-identical
// across this commit.
//
// What lives here today:
//   * String parsing helpers (parse_spin_combinations / parse_momentum_points
//     / parse_polarization).
//   * `construct_operators_from_config` — thin wrapper over the canonical
//     `ed::dssf::build_observable_pairs` (P1.10), preserved so the four
//     historical call sites in this file (and the `run_dssf_mode` shim
//     still living in ed_main.cpp) keep compiling.
//   * The two `run_*_workflow` (standard / streaming-symmetry) entry points.
//     (`run_disk_streaming_workflow` / `run_chunked_symmetry_workflow` were
//      retired in matvec-unification Phase 7.2; see comment in this file.)
//   * `compute_thermodynamics`.
//   * The three `compute_*_workflow` entry points (dynamical response,
//     static response, ground-state DSSF) — these are the principal
//     reason this TU exists; they account for ~1.5 kLOC of nearly
//     identical operator-construction + Lanczos/FTLM dispatch + HDF5 save
//     logic that future PRs (P2.2 dssf_engine, P2.3 unified /dssf/ HDF5
//     schema, P2.4 `ED dssf` subcommand) will collapse onto a single
//     `ed::dssf::run(...)` call.
//   * `print_eigenvalue_summary`.
//
// The include list mirrors what `ed_main.cpp` used to require for these
// functions (no more, no less). When the future DSSF refactor pulls the
// guts of the `compute_*_workflow` functions into `ed::dssf`, we can
// trim this list aggressively.
// =============================================================================

#include <ed/cli/workflows.h>
#include <tuple>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <filesystem>
#include <memory>
#include <random>
#include <algorithm>
#include <cstdlib>          // getenv (Wave 3.4 ED_DSSF_PAIR_THREADS)
#ifdef _OPENMP
#include <omp.h>            // Wave 3.4 OpenMP-over-pairs
#endif
#include <map>
#include <set>
#include <fstream>

#include <ed/core/ed_config.h>
#include <ed/core/ed_config_adapter.h>
#include <ed/core/ed_wrapper.h>            // residue: EDResults envelope (legacy types only)
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/core/system_utils.h>          // create_directory_mpi_safe (was via ed_wrapper_streaming.h)

// Full Unified-Interface Collapse, Wave C2 (May 2026): run_standard_workflow
// and run_streaming_symmetry_workflow now build their operator via the
// unified factory and dispatch through the orchestrator, replacing the
// legacy `ed::exact_diagonalization(directory, method, params, ...)` entry
// from the now-deleted <ed/core/dispatch.h>.
#include <ed/core/make_operator.h>
#include <ed/core/sector_loop.h>          // filter_sectors / resolve_target_sector
#include <ed/orchestrator.h>
#include <ed/dssf/operator_spec.h>
#include <ed/dssf/cross_sector_observable.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator_builders.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/ftlm_dist.h>
#include <ed/solvers/kpm_dos.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/observables.h>
#include <ed/observables/cf_dynamical.h>   // ftlm_dynamical_kernel_via_backend_multitemp
#include <ed/core/select_backend.h>        // select_backend + BackendConstraints

#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef WITH_CUDA
#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/kpm_dos_gpu.cuh>
#include <cuda_runtime.h>
#include <functional>

namespace {
// Operator-collapse Phase 2b (Jun 2026): wrap a unified host operator's device
// matvec (CudaMatVecBackend via Operator/FixedSzOperator::bind_cuda) as the
// cuDoubleComplex-typed callable the GPU FTLM / KPM drivers expect.
// std::complex<double> and cuDoubleComplex are layout-compatible, so the
// reinterpret_cast is well-defined. Replaces the legacy GPUOperator mirror +
// convertOperatorToGPU round-trip for the CLI DSSF / KPM GPU lanes.
inline std::function<void(const cuDoubleComplex*, cuDoubleComplex*, int)>
device_matvec_from(ed::LinearOperator& op) {
    auto fn = op.bind_cuda();
    return [fn = std::move(fn)](const cuDoubleComplex* in,
                                cuDoubleComplex* out, int n) {
        fn(reinterpret_cast<const ed::Complex*>(in),
           reinterpret_cast<ed::Complex*>(out),
           static_cast<std::size_t>(n));
    };
}
}  // namespace
#endif

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
inline std::vector<std::pair<int, int>>
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
static inline std::pair<int, int> get_mpi_rank_size_safe() {
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
static inline std::vector<double>
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

/// Bundles every piece of Hamiltonian state the compute_*_workflow
/// drivers need: the concrete operator (full or fixed-Sz), the sector
/// dimension, and an apply lambda. The lambda dispatches via the
/// concrete shared_ptr so the audit #2 fixed-Sz CPU path doesn't slice
/// back to `Operator::apply` at the wrong dimension.
struct WorkflowHamiltonian {
    bool                              use_fixed_sz = false;
    int64_t                           n_up         = -1;
    uint64_t                          N            = 0;
    std::shared_ptr<Operator>         ham_full;
    std::shared_ptr<FixedSzOperator>  ham_fs;
    std::function<void(const Complex*, Complex*, uint64_t)> H_func;

    /// Slice-as-base for legacy callers that need `Operator&`.
    Operator& ham_ref() {
        return use_fixed_sz ? static_cast<Operator&>(*ham_fs) : *ham_full;
    }
};

/// Construct the Hamiltonian, three-body terms, sector dimension, and
/// `H_func` apply lambda from `config`. When `verbose_label` is non-null
/// and `rank == 0`, prints the three-body load and the sector-dim
/// summary the workflows used to print inline.
static inline WorkflowHamiltonian
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

/**
 * @brief Compute dynamical response (spectral functions)
 */
#ifdef WITH_MPI
namespace {
// ============================================================================
// Audit 2026-07-31 (H2): shared MPI master-worker driver with a SINGLE
// HDF5 writer and worker-failure containment.
//
// The previous protocol had every worker write its own results into the
// shared ed_results.h5: HDF5's file locking made concurrent writers race
// (the loser threw "unable to lock file"), and a worker dying on ANY
// exception never sent DONE_TAG -- the master then spun forever in
// `while (completed < num_tasks)`. This driver fixes both:
//
//   * workers COMPUTE only and ship the packed result (vector<double>)
//     to the master; the master is the only rank that touches the file;
//   * every worker task is wrapped in try/catch and the DONE message
//     carries {task_id, ok} -- a failed task completes the protocol,
//     is recorded, and is reported as PARTIAL RESULTS at the end
//     instead of hanging the job.
//
// `compute(task_id)` -> packed payload (throws on failure; runs on the
// owning rank). `write(task_id, payload)` -> HDF5 (master only; a write
// failure is recorded like a compute failure). Returns this rank's
// successfully processed count under the legacy semantics (each rank
// counts the tasks IT computed), so the existing MPI_Reduce total and
// "Processed X/N" print stay meaningful.
// ============================================================================
template <class ComputeFn, class WriteFn>
int run_mpi_master_worker_single_writer(int rank, int size, int num_tasks,
                                        ComputeFn&& compute,
                                        WriteFn&&   write,
                                        const char* what)
{
    constexpr int TASK_TAG = 1, DONE_TAG = 2, STOP_TAG = 3, RESULT_TAG = 4;
    int ok_count = 0;
    if (rank == 0) {
        std::vector<int> failed;
        int next_task = 0;
        int first_idle_worker = size;
        for (int r = 1; r < size && next_task < num_tasks; r++) {
            MPI_Send(&next_task, 1, MPI_INT, r, TASK_TAG, MPI_COMM_WORLD);
            next_task++;
            first_idle_worker = r + 1;
        }
        for (int r = first_idle_worker; r < size; r++) {
            int dummy = -1;
            MPI_Send(&dummy, 1, MPI_INT, r, STOP_TAG, MPI_COMM_WORLD);
        }
        int completed = 0;
        while (completed < num_tasks) {
            if (next_task < num_tasks) {
                const int my_task = next_task++;
                std::cout << "Rank 0 processing task " << (my_task + 1)
                          << "/" << num_tasks << "\n";
                try {
                    auto payload = compute(my_task);
                    write(my_task, payload);
                    ++ok_count;
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[%s] rank 0: task %d FAILED: %s\n",
                                 what, my_task, e.what());
                    failed.push_back(my_task);
                }
                ++completed;
            }
            int flag = 0;
            MPI_Status status;
            MPI_Iprobe(MPI_ANY_SOURCE, DONE_TAG, MPI_COMM_WORLD, &flag,
                       &status);
            if (flag) {
                int hdr[2];
                MPI_Recv(hdr, 2, MPI_INT, status.MPI_SOURCE, DONE_TAG,
                         MPI_COMM_WORLD, &status);
                ++completed;
                if (hdr[1]) {
                    MPI_Status rs;
                    MPI_Probe(status.MPI_SOURCE, RESULT_TAG, MPI_COMM_WORLD,
                              &rs);
                    int count = 0;
                    MPI_Get_count(&rs, MPI_DOUBLE, &count);
                    std::vector<double> payload(
                        static_cast<std::size_t>(count));
                    MPI_Recv(payload.data(), count, MPI_DOUBLE,
                             status.MPI_SOURCE, RESULT_TAG, MPI_COMM_WORLD,
                             MPI_STATUS_IGNORE);
                    try {
                        write(hdr[0], payload);
                    } catch (const std::exception& e) {
                        std::fprintf(stderr,
                                     "[%s] rank 0: HDF5 write of task %d "
                                     "(from rank %d) FAILED: %s\n",
                                     what, hdr[0], status.MPI_SOURCE,
                                     e.what());
                        failed.push_back(hdr[0]);
                    }
                } else {
                    failed.push_back(hdr[0]);
                }
                if (next_task < num_tasks) {
                    MPI_Send(&next_task, 1, MPI_INT, status.MPI_SOURCE,
                             TASK_TAG, MPI_COMM_WORLD);
                    next_task++;
                } else {
                    int dummy = -1;
                    MPI_Send(&dummy, 1, MPI_INT, status.MPI_SOURCE,
                             STOP_TAG, MPI_COMM_WORLD);
                }
            }
        }
        if (!failed.empty()) {
            std::fprintf(stderr,
                         "[%s] PARTIAL RESULTS: %zu of %d task(s) failed"
                         " (ids:", what, failed.size(), num_tasks);
            for (int t : failed) std::fprintf(stderr, " %d", t);
            std::fprintf(stderr,
                         "); their datasets are absent from the HDF5 "
                         "output.\n");
        }
    } else {
        while (true) {
            int task_id;
            MPI_Status status;
            MPI_Recv(&task_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD,
                     &status);
            if (status.MPI_TAG == STOP_TAG) break;
            std::vector<double> payload;
            int ok = 1;
            try {
                payload = compute(task_id);
            } catch (const std::exception& e) {
                ok = 0;
                std::fprintf(stderr, "[%s] rank %d: task %d FAILED: %s\n",
                             what, rank, task_id, e.what());
            } catch (...) {
                ok = 0;
                std::fprintf(stderr,
                             "[%s] rank %d: task %d FAILED "
                             "(non-std exception)\n",
                             what, rank, task_id);
            }
            int hdr[2] = {task_id, ok};
            MPI_Send(hdr, 2, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
            if (ok) {
                MPI_Send(payload.data(), static_cast<int>(payload.size()),
                         MPI_DOUBLE, 0, RESULT_TAG, MPI_COMM_WORLD);
                ++ok_count;
            }
        }
    }
    return ok_count;
}

// Payload packing for the driver above: [n_arrays fields...] as flat
// doubles. Each array is emitted as (len, data...); scalars first.
inline void pack_scalar(std::vector<double>& p, double v) { p.push_back(v); }
inline void pack_array(std::vector<double>& p, const std::vector<double>& v) {
    p.push_back(static_cast<double>(v.size()));
    p.insert(p.end(), v.begin(), v.end());
}
inline double unpack_scalar(const std::vector<double>& p, std::size_t& pos) {
    return p.at(pos++);
}
inline std::vector<double> unpack_array(const std::vector<double>& p,
                                        std::size_t& pos) {
    const std::size_t n = static_cast<std::size_t>(p.at(pos++));
    std::vector<double> v(p.begin() + static_cast<std::ptrdiff_t>(pos),
                          p.begin() + static_cast<std::ptrdiff_t>(pos + n));
    pos += n;
    return v;
}
}  // namespace
#endif  // WITH_MPI

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

            // Create function wrappers for this operator pair (audit #2:
            // dispatch via apply_obs* so fixed-Sz uses the typed override).
            auto O1_func = [apply_obs1, op_idx](const Complex* in, Complex* out, uint64_t dim) {
                apply_obs1(op_idx, in, out, dim);
            };

            auto O2_func = [apply_obs2, op_idx](const Complex* in, Complex* out, uint64_t dim) {
                apply_obs2(op_idx, in, out, dim);
            };

            // Compute response on CPU (the only supported path for
            // single-T tasks).
            return compute_dynamical_correlation(
                H_func, O1_func, O2_func, N, params,
                config.dynamical.omega_min,
                config.dynamical.omega_max,
                config.dynamical.num_omega_points,
                temperature,
                config.workflow.output_dir,
                ground_state_energy
            );
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
            return P;
        };

        // Decide whether the multi-operator shared-Lanczos path applies.
        // NOTE (audit 2026-07-31): this optimization engages on the
        // SEQUENTIAL lane only -- the single-T MPI master-worker always
        // runs the per-task kernel. Both are valid FTLM estimators, but
        // their random-sample streams differ, so serial and mpirun
        // spectra of the same config agree only statistically (exactly
        // at num_samples == 1, which is what the mpi integration test
        // pins). Porting the shared-chain optimization into the MPI
        // task shape is possible future work.
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

            if (have_op2) {
                if (rank == 0) std::cout << "Computing two-operator dynamical correlation ⟨O₁†(t)O₂⟩...\n";
                results = compute_dynamical_correlation(
                    H_func, O_func, O2_func, N, params,
                    config.dynamical.omega_min,
                    config.dynamical.omega_max,
                    config.dynamical.num_omega_points,
                    temperature,
                    config.workflow.output_dir,
                    ground_state_energy
                );
            } else {
                if (rank == 0) std::cout << "Computing dynamical response ⟨O†(t)O⟩...\n";
                results = compute_dynamical_response_thermal(
                    H_func, O_func, N, params,
                    config.dynamical.omega_min,
                    config.dynamical.omega_max,
                    config.dynamical.num_omega_points,
                    temperature,
                    config.workflow.output_dir
                );
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
        if (const char* env = std::getenv("ED_DSSF_PAIR_THREADS")) {
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
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "Ground State DSSF Complete\n";
        std::cout << "Results saved to: " << config.workflow.output_dir
                  << "/ed_results.h5\n";
        std::cout << "==========================================\n";
    }
}

// ============================================================================
// KPM thermodynamics workflow (audit item #3)
//
// Operator-free thermodynamics from the Chebyshev-expanded density of
// states: Z(beta), E(beta), C(beta), S(beta), F(beta) (and an optional
// reconstructed DOS) for the configured Hamiltonian. Persists results
// under `/kpm_thermodynamics/...` in the run's HDF5 file.
//
// Driven by the existing `ed::kpm_dos::compute_kpm_dos` solver. The
// dispatch wiring lives in `src/cli/dssf_engine.cpp`; this is the body.
// ============================================================================
void compute_kpm_thermodynamics_workflow(const EDConfig& config) {
    // Plain locals (not a structured binding): C++17 forbids capturing
    // structured bindings in lambdas (clang enforces; gcc extension).
    int rank = 0, size = 1;
    std::tie(rank, size) = get_mpi_rank_size_safe();
    (void)size;

    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "KPM Thermodynamics (Chebyshev DOS)\n";
        std::cout << "==========================================\n";
    }

    auto wh = build_workflow_hamiltonian(config, rank, /*verbose_label=*/nullptr);
    const bool use_fixed_sz = wh.use_fixed_sz;
    const int64_t n_up      = wh.n_up;
    const uint64_t N        = wh.N;
    auto& ham_full          = wh.ham_full;
    auto& ham_fs            = wh.ham_fs;
    Operator& ham           = wh.ham_ref();
    auto& H_func            = wh.H_func;
    (void)ham_full; (void)ham_fs;

    // Temperature grid: prefer the dynamical-block grid when a sweep is
    // configured (num_temp_bins > 1); else fall back to the thermal block
    // (which the standard thermodynamics workflow uses) so the user gets
    // something sensible without having to repopulate dssf-specific knobs.
    std::vector<double> temperatures;
    {
        double tmin = config.dynamical.temp_min;
        double tmax = config.dynamical.temp_max;
        uint64_t nT = config.dynamical.num_temp_bins;
        if (nT <= 1) {
            tmin = config.thermal.temp_min;
            tmax = config.thermal.temp_max;
            nT   = std::max<uint64_t>(1, config.thermal.num_temp_bins);
        }
        temperatures.resize(nT);
        if (nT == 1) {
            temperatures[0] = tmin;
        } else {
            const double log_tmin = std::log(tmin);
            const double log_tmax = std::log(tmax);
            const double dl = (log_tmax - log_tmin) / (nT - 1);
            for (uint64_t i = 0; i < nT; ++i) {
                temperatures[i] = std::exp(log_tmin + i * dl);
            }
        }
    }

    std::vector<double> betas(temperatures.size());
    for (size_t i = 0; i < temperatures.size(); ++i) {
        if (!(temperatures[i] > 0.0)) {
            throw std::invalid_argument(
                "compute_kpm_thermodynamics_workflow: temperature must be > 0");
        }
        betas[i] = 1.0 / temperatures[i];
    }

    ed::kpm_dos::KPMDOSParameters kpm_params;
    if (config.dynamical.krylov_dim > 0) {
        // Reuse the user's Krylov budget for the spectral-bound Lanczos.
        kpm_params.spectral_bounds_krylov =
            static_cast<int>(std::min<uint64_t>(
                config.dynamical.krylov_dim, 500));
    }
    if (config.dynamical.num_random_states > 0) {
        kpm_params.num_random_vectors =
            static_cast<int>(config.dynamical.num_random_states);
    }
    if (config.dynamical.random_seed != 0) {
        kpm_params.random_seed =
            static_cast<std::uint64_t>(config.dynamical.random_seed);
    }
    // Tunables not yet exposed through ed_config.cpp — pick them up from env
    // so production scripts can sweep without rebuilding.  Defaults are
    // documented next to KPMDOSParameters in include/ed/solvers/kpm_dos.h.
    if (const char* env_M = std::getenv("ED_KPM_NUM_MOMENTS")) {
        const int v = std::atoi(env_M);
        if (v >= 4) kpm_params.num_moments = v;
    }
    if (const char* env_Nq = std::getenv("ED_KPM_NUM_QUAD")) {
        const int v = std::atoi(env_Nq);
        if (v > 0) kpm_params.num_quadrature_nodes = v;
    }
    if (const char* env_buf = std::getenv("ED_KPM_BOUND_BUFFER")) {
        const double v = std::atof(env_buf);
        if (v > 0.0) kpm_params.spectral_bound_buffer = v;
    }
    if (const char* env_kern = std::getenv("ED_KPM_KERNEL")) {
        const std::string s(env_kern);
        if (s == "lorentz" || s == "Lorentz" || s == "LORENTZ") {
            kpm_params.use_jackson_kernel = false;
        }
    }
    if (const char* env_lambda = std::getenv("ED_KPM_LORENTZ_LAMBDA")) {
        const double v = std::atof(env_lambda);
        if (v > 0.0) kpm_params.lorentz_lambda = v;
    }

    if (rank == 0) {
        std::cout << "  dim          = " << N << "\n";
        std::cout << "  num_moments  = " << kpm_params.num_moments << "\n";
        std::cout << "  num_samples  = " << kpm_params.num_random_vectors << "\n";
        std::cout << "  temperatures = " << temperatures.size() << "\n";
        std::cout << "  jackson      = "
                  << (kpm_params.use_jackson_kernel ? "yes" : "no") << "\n";
    }

    create_directory_mpi_safe(config.workflow.output_dir);

    // Single-process driver: no MPI distribution at this point.  GPU dispatch
    // is per rank when --use-gpu is enabled (and CUDA is built); otherwise we
    // fall back to the CPU operator-free implementation.
    if (rank != 0) {
        return;
    }

    ed::kpm_dos::KPMDOSResult kpm;
#ifdef WITH_CUDA
    const bool kpm_use_gpu =
        config.dynamical.use_gpu || config.system.use_gpu;
    if (kpm_use_gpu) {
        // Operator-collapse Phase 2b (Jun 2026): drive the GPU KPM
        // Chebyshev/Hutchinson loop straight off the unified host operator's
        // device matvec (CudaMatVecBackend via Operator/FixedSzOperator::
        // bind_cuda) -- no bespoke GPUOperator mirror / convertOperatorToGPU
        // round-trip. `ham` is a FixedSzOperator in the fixed-Sz sector and a
        // full-Hilbert Operator otherwise; bind_cuda() dispatches virtually.
        std::cout << "  backend      = GPU (SOTA gather matvec"
                  << (use_fixed_sz ? ", fixed-Sz" : "") << ")\n";
        kpm = ed::kpm_dos::compute_kpm_dos_gpu_with_matvec(
            device_matvec_from(ham), N, betas, /*dos_grid=*/{}, kpm_params);
    } else {
        std::cout << "  backend      = CPU (operator-free)\n";
        kpm = ed::kpm_dos::compute_kpm_dos(
            H_func, N, betas, /*dos_grid=*/{}, kpm_params);
    }
#else
    std::cout << "  backend      = CPU (operator-free; no CUDA build)\n";
    kpm = ed::kpm_dos::compute_kpm_dos(
        H_func, N, betas, /*dos_grid=*/{}, kpm_params);
#endif

    // Persist under the standard thermodynamics group so existing readers
    // (Python `qed.workflow`, the analysis scripts) just work, and stamp
    // a small KPM_THERMODYNAMICS provenance under /kpm_thermodynamics.
    const std::string h5_file =
        HDF5IO::createOrOpenFile(config.workflow.output_dir);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "energy",        kpm.energy);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "specific_heat", kpm.specific_heat);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "entropy",       kpm.entropy);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "free_energy",   kpm.free_energy);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "partition_function",
                               kpm.partition_function);

    try {
        H5::H5File file(h5_file, H5F_ACC_RDWR);
        if (!file.nameExists("/kpm_thermodynamics")) {
            file.createGroup("/kpm_thermodynamics");
        }
        H5::Group g = file.openGroup("/kpm_thermodynamics");
        const auto write_dbl = [&](const char* name, double v) {
            H5::DataSpace s(H5S_SCALAR);
            if (g.attrExists(name)) g.removeAttr(name);
            auto a = g.createAttribute(name, H5::PredType::NATIVE_DOUBLE, s);
            a.write(H5::PredType::NATIVE_DOUBLE, &v);
        };
        const auto write_int = [&](const char* name, int v) {
            H5::DataSpace s(H5S_SCALAR);
            if (g.attrExists(name)) g.removeAttr(name);
            auto a = g.createAttribute(name, H5::PredType::NATIVE_INT, s);
            a.write(H5::PredType::NATIVE_INT, &v);
        };
        write_int("num_moments_used",        kpm.num_moments_used);
        write_int("num_random_vectors_used", kpm.num_random_vectors_used);
        write_dbl("kpm_a",                   kpm.kpm_a);
        write_dbl("kpm_b",                   kpm.kpm_b);
        write_dbl("e_min_estimate",          kpm.e_min_estimate);
        write_dbl("e_max_estimate",          kpm.e_max_estimate);
        write_dbl("energy_shift_used",       kpm.energy_shift_used);

        if (!kpm.moments_weighted.empty()) {
            const std::string ds = "/kpm_thermodynamics/moments_weighted";
            if (file.nameExists(ds)) file.unlink(ds);
            hsize_t dims[1] = { kpm.moments_weighted.size() };
            H5::DataSpace ms(1, dims);
            auto d = file.createDataSet(
                ds, H5::PredType::NATIVE_DOUBLE, ms);
            d.write(kpm.moments_weighted.data(), H5::PredType::NATIVE_DOUBLE);
        }
    } catch (const H5::Exception& e) {
        std::cerr << "  Warning: KPM provenance write failed: "
                  << e.getDetailMsg() << "\n";
    }

    std::cout << "\nKPM thermodynamics complete.\n";
    std::cout << "Results: " << config.workflow.output_dir
              << "/ed_results.h5  (groups /thermodynamics, /kpm_thermodynamics)\n";
}

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


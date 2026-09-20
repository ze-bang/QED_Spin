#pragma once
// =============================================================================
// src/cli/workflows/workflows_internal.h -- PRIVATE to the CLI workflows.
//
// The workflow entry points declared in `include/ed/cli/workflows.h` are the
// argv-driven side of the library: `src/apps/ed_main.cpp` (and the subcommand
// table in `src/cli/dssf_engine.cpp`) parse an `EDConfig` and call exactly one
// of them. They were lifted verbatim out of ed_main.cpp in P1.11 and grew into
// a single ~3.4 kLOC translation unit; WP14 splits that file by workflow, as a
// pure structural move -- same argument parsing, same printed lines, same HDF5
// groups, same exit codes, same MPI rank/size plumbing.
//
// This header carries the include block every workflow translation unit needs
// (it is the former file's include list verbatim), the `WorkflowHamiltonian`
// bundle the compute_*_workflow drivers share, the MPI master-worker driver
// template, and the declarations of the shared helpers whose definitions live
// in wf_common.cpp. Nothing outside src/cli/workflows/ includes it: the public
// surface is include/ed/cli/workflows.h.
//
// The workflow functions live in the global namespace, exactly as they did in
// src/cli/workflows.cpp -- see the note in the public header about why.
//
// File map
//   wf_common.cpp        shared plumbing: the CLI string parsers
//                        (parse_spin_combinations / parse_momentum_points /
//                        parse_polarization), the fixed-Sz transverse-channel
//                        filter, `construct_operators_from_config`, the MPI
//                        rank/size probe and Allgatherv, the
//                        `build_workflow_hamiltonian` preamble, the
//                        master-worker payload pack/unpack helpers, and
//                        `print_eigenvalue_summary`
//   wf_diagonalize.cpp   run_standard_workflow / run_streaming_symmetry_workflow
//                        and `compute_thermodynamics`
//   wf_dynamical.cpp     compute_dynamical_response_workflow: the FTLM / LTLM /
//                        continued-fraction spectral-function driver. One
//                        function, ~1 kLOC -- it is the file's largest single
//                        body and cannot be shrunk without changing behaviour
//   wf_static.cpp        compute_static_response_workflow: thermal expectation
//                        values (FTLM static correlations + the legacy
//                        --static-operator file path)
//   wf_dssf.cpp          compute_ground_state_dssf_workflow: T=0 continued
//                        fraction DSSF, same-sector and cross-sector lanes
//   wf_kpm.cpp           compute_kpm_thermodynamics_workflow (audit item #3)
//                        plus the WITH_CUDA `device_matvec_from` adapter, whose
//                        only call site is the KPM GPU lane
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

#include <ed/config/env_registry.h>
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
#include <ed/solvers/lanczos.h>            // find_ground_state_lanczos
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
#endif  // WITH_CUDA

// -----------------------------------------------------------------------------
// Helpers shared by more than one CLI workflow translation unit. They were
// file-scope `static inline` functions (or anonymous-namespace functions)
// inside the single workflows.cpp; they now carry external linkage with their
// one definition in wf_common.cpp. None of them owns mutable state, so the
// change of linkage is observationally inert.
// -----------------------------------------------------------------------------

/// Cross-sector DSSF correctness guard: drop the spin-combination channels
/// that are zero by Sz-sector orthogonality (and the unsupported XYZ ones),
/// keeping the legitimately cross-sector pairs. Full contract: see the
/// definition in wf_common.cpp.
std::vector<std::pair<int, int>>
filter_fixed_sz_transverse_channels(
    const std::vector<std::pair<int, int>>& spin_combinations,
    bool use_fixed_sz,
    bool use_xyz_basis,
    int rank,
    const char* workflow_label);

/// Returns (rank, size) from `MPI_COMM_WORLD`, falling back to (0, 1) when
/// MPI is unavailable or `MPI_Init` has not been called.
std::pair<int, int> get_mpi_rank_size_safe();

/// Across-sector MPI (Level 1, SectorDistributor): gather variable-length
/// per-rank double vectors into the full vector on EVERY rank. No-op when
/// single-rank.
std::vector<double> mpi_allgatherv_doubles(const std::vector<double>& local);

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
/// `H_func` apply lambda from `config`. When `verbose_label` is non-null and
/// `rank == 0`, prints the three-body load and the sector-dim summary the
/// workflows used to print inline. Full contract: see the definition in
/// wf_common.cpp.
WorkflowHamiltonian
build_workflow_hamiltonian(const EDConfig& config,
                           int rank,
                           const char* verbose_label);

#ifdef WITH_MPI
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
// Definitions in wf_common.cpp.
void pack_scalar(std::vector<double>& p, double v);
void pack_array(std::vector<double>& p, const std::vector<double>& v);
double unpack_scalar(const std::vector<double>& p, std::size_t& pos);
std::vector<double> unpack_array(const std::vector<double>& p,
                                 std::size_t& pos);
#endif  // WITH_MPI

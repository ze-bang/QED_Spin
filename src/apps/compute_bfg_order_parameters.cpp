/**
 * @file compute_bfg_order_parameters.cpp
 * @brief Thin argv shell for the BFG order-parameter computation pipeline.
 *
 * Every physics kernel, HDF5 schema writer, and orchestration helper
 * (single-file pipeline, scan-mode dispatch, MPI work distribution) lives
 * in the `ed_bfg` static library:
 *
 *   include/ed/bfg/*.h  +  src/bfg/*.cpp
 *
 * In particular `include/ed/bfg/cli.h` exposes:
 *
 *   * `ed::bfg::cli::print_usage(prog)`                  -- usage banner
 *   * `ed::bfg::cli::SingleFileOptions` /
 *     `ed::bfg::cli::ScanOptions`                        -- argv-derived
 *                                                          option PODs
 *   * `ed::bfg::cli::run_single_file(opts)`              -- one wavefunction
 *                                                          + one cluster
 *                                                          -> one HDF5
 *   * `ed::bfg::cli::run_scan(opts, mpi_rank, mpi_size)` -- sweep over
 *                                                          Jpm=* dirs +
 *                                                          combined HDF5
 *                                                          summary
 *
 * The driver below is therefore restricted to:
 *   1. MPI_Init / MPI_Finalize wrapper (only the top-level main() owns
 *      these; library code stays MPI-agnostic).
 *   2. argv parsing into `SingleFileOptions` / `ScanOptions`.
 *   3. Dispatch to `run_single_file` or `run_scan`.
 *   4. `try { ... } catch` boundary that turns exceptions into a non-zero
 *      exit code with a friendly stderr message on rank 0.
 *   5. Wallclock timing summary on rank 0.
 *
 * Audit ref: P2.1 (ninth slice).
 *
 * Usage:
 *   ./compute_bfg_order_parameters <wavefunction.h5> <cluster_dir> [output.h5]
 *   ./compute_bfg_order_parameters --scan-dir <dir> [options]
 *
 * Compile via the project CMake (target: `compute_bfg_order_parameters`).
 */

#include <chrono>
#include <exception>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "ed/bfg/cli.h"

int main(int argc, char* argv[]) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
    int mpi_rank = 0;
    int mpi_size = 1;
#endif

    auto cleanup_mpi = []() {
#ifdef USE_MPI
        MPI_Finalize();
#endif
    };

    if (argc < 2) {
        if (mpi_rank == 0) {
            ed::bfg::cli::print_usage(argv[0]);
        }
        cleanup_mpi();
        return 1;
    }

    ed::bfg::cli::SingleFileOptions single_opts;
    ed::bfg::cli::ScanOptions scan_opts;
    bool scan_mode = false;

    // argv parsing: the named flags target either both option structs (so
    // the CLI surface stays uniform) or the scan-only POD when they only
    // make sense in scan mode. Positional arguments fill the single-file
    // POD.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scan-dir" && i + 1 < argc) {
            scan_opts.scan_dir = argv[++i];
            scan_mode = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            scan_opts.output_dir = argv[++i];
        } else if (arg == "--n-workers" && i + 1 < argc) {
            scan_opts.n_workers = std::stoi(argv[++i]);
        } else if (arg == "--n-q-grid" && i + 1 < argc) {
            int n = std::stoi(argv[++i]);
            scan_opts.n_q_grid = n;
            single_opts.n_q_grid = n;
        } else if (arg == "--save-full") {
            scan_opts.save_full = true;
        } else if (arg == "--tpq") {
            scan_opts.use_tpq = true;
            single_opts.use_tpq = true;
        } else if (arg == "--tpq-all-temps") {
            scan_opts.use_tpq = true;
            scan_opts.tpq_all_temps = true;
            single_opts.use_tpq = true;
        } else if (arg == "-h" || arg == "--help") {
            if (mpi_rank == 0) {
                ed::bfg::cli::print_usage(argv[0]);
            }
            cleanup_mpi();
            return 0;
        } else if (single_opts.wf_file.empty()) {
            single_opts.wf_file = arg;
        } else if (single_opts.cluster_dir.empty()) {
            single_opts.cluster_dir = arg;
        } else {
            single_opts.output_file = arg;
        }
    }

#ifdef _OPENMP
    if (mpi_rank == 0) {
        std::cout << "OpenMP enabled with " << omp_get_max_threads()
                  << " max threads" << std::endl;
    }
#endif

    auto total_start = std::chrono::high_resolution_clock::now();

    try {
        if (scan_mode) {
            if (scan_opts.scan_dir.empty()) {
                if (mpi_rank == 0) {
                    std::cerr << "Error: --scan-dir required in scan mode" << std::endl;
                }
                cleanup_mpi();
                return 1;
            }
            ed::bfg::cli::run_scan(scan_opts, mpi_rank, mpi_size);
        } else {
            // Single-file pipeline runs on rank 0 only; other ranks exit
            // immediately so we don't try to open the same HDF5 file
            // concurrently from N ranks.
#ifdef USE_MPI
            if (mpi_rank != 0) {
                cleanup_mpi();
                return 0;
            }
#endif

            if (single_opts.wf_file.empty() || single_opts.cluster_dir.empty()) {
                ed::bfg::cli::print_usage(argv[0]);
                cleanup_mpi();
                return 1;
            }

            ed::bfg::cli::run_single_file(single_opts);
        }
    } catch (const std::exception& e) {
        if (mpi_rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        cleanup_mpi();
        return 1;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    if (mpi_rank == 0) {
        std::cout << "\nTotal runtime: " << total_duration.count() << " seconds" << std::endl;
    }

    cleanup_mpi();
    return 0;
}

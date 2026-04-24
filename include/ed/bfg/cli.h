// =============================================================================
// include/ed/bfg/cli.h
//
// Orchestration helpers backing the `compute_bfg_order_parameters` CLI
// (P2.1 ninth slice).
//
// The argv parser in `src/apps/compute_bfg_order_parameters.cpp` builds a
// `SingleFileOptions` or `ScanOptions` and hands it to one of the two
// dispatchers below. All directory walking, MPI distribution, per-Jpm
// progress prints, and HDF5-results plumbing live here so the driver
// itself collapses to a thin argv shell.
//
// Why this lives in `ed_bfg`: the orchestration depends on every other
// `ed_bfg` slice (cluster loader, correlations, structure factors, ring
// observables, spin S(q), order parameters, results-IO writers) and
// nothing outside it, and the dispatchers are reusable from any future
// front-end (e.g. a Python `quantum_ed.cli` mirror, a benchmarking
// harness, or a unified `ED bfg-order-parameters` subcommand).
//
// Audit ref: P2.1.
// =============================================================================

#pragma once

#include <string>
#include <vector>

#include "ed/bfg/cluster.h"
#include "ed/bfg/results_io.h"

namespace ed::bfg::cli {

// -----------------------------------------------------------------------------
// Single-file mode (one wavefunction + one cluster -> one HDF5 results file)
// -----------------------------------------------------------------------------
struct SingleFileOptions {
    std::string wf_file;
    std::string cluster_dir;
    std::string output_file = "bfg_order_parameters.h5";
    int n_q_grid = 50;
    bool use_tpq = false;
};

// -----------------------------------------------------------------------------
// Scan mode (sweep over Jpm=* sub-directories under one parent directory)
// -----------------------------------------------------------------------------
struct ScanOptions {
    std::string scan_dir;
    std::string output_dir;  // empty -> scan_dir + "/order_parameter_results"
    int n_workers = 4;
    int n_q_grid = 50;
    bool save_full = false;
    bool use_tpq = false;
    bool tpq_all_temps = false;
};

// Print the canonical CLI usage help to stderr.
void print_usage(const char* prog);

// Run the scalar order-parameter pipeline on every TPQ snapshot stored in
// `wf_file`. Returns one `OrderParameterResults` per temperature.
//
// Note: `n_q_grid` and `save_full` are accepted but currently unused; the
// per-temperature path always runs the quick scalar pipeline. The
// arguments are kept for forward compatibility with future per-T full-grid
// support.
std::vector<OrderParameterResults> process_all_temperatures(
    const std::string& wf_file,
    const Cluster& cluster,
    double jpm,
    int n_q_grid,
    bool save_full
);

// Walk every `Jpm=*` sub-directory under `opts.scan_dir`, run the
// order-parameter pipeline on each, write per-Jpm HDF5 files, and return
// the per-Jpm scalar summaries. `mpi_rank` / `mpi_size` partition the
// work; pass `(0, 1)` for the non-MPI case.
std::vector<OrderParameterResults> scan_jpm_directories(
    const ScanOptions& opts,
    int mpi_rank = 0,
    int mpi_size = 1
);

// Single-file pipeline: load the cluster + wavefunction, run every
// order-parameter kernel, and write `opts.output_file`. Throws on error.
void run_single_file(const SingleFileOptions& opts);

// Scan pipeline: print the scan-mode banner (rank 0 only), make sure the
// output directory exists, run `scan_jpm_directories(opts, ...)`, and (on
// rank 0) write the combined scan results HDF5. Throws on error.
void run_scan(const ScanOptions& opts, int mpi_rank, int mpi_size);

}  // namespace ed::bfg::cli

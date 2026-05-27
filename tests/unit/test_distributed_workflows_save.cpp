// =============================================================================
// test_distributed_workflows_save.cpp
//
// MPI rank-0 unified-file emission lockdown (May 2026, "Universal save
// contract" follow-up).
//
// The orchestrator's HDF5 finalisers used to skip distributed lanes
// entirely on the assumption that every rank owns only a slab of the
// result. That is true for slab-distributed eigenvectors / TPQ state
// vectors -- the per-rank ``rank_<r>.h5`` files written by
// ``ed_distributed_main`` remain the canonical location for those --
// but the *aggregate* fields (eigenvalues, S(omega), aggregated thermo
// curves) are broadcast / reduced onto every rank, so rank 0 can write
// a single unified ``ed_results.h5`` alongside the slab files.
//
// This test pins:
//   * rank 0 produces a non-empty ``R.hdf5_path``.
//   * the corresponding file exists on disk under
//     ``output_dir/ed_results.h5``.
//   * the file carries ``/eigendata/eigenvalues`` (the rank-0
//     replicated aggregate).
//   * non-root ranks see ``R.hdf5_path.empty()`` and do not write to
//     the rank-0 path (no double-write / no collisions).
//
// Runs at np ∈ {1, 2, 4} via ``ed_add_mpi_test``.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_operator.h>
#include <ed/orchestrator.h>
#include <ed/core/hdf5_io.h>
#include "common/test_harness.h"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <H5Cpp.h>
#include <mpi.h>

using ed::distributed::DistributedOperator;

namespace {

// Hand the test a fresh subdir per MPI test invocation, picked up from
// ``ED_TEST_TMP_DIR`` (set by CTest) or a /tmp fallback. Always rooted
// per-rank-0 so MPI ranks share the same prefix and the rank-0 writer
// lands in a predictable place.
std::string fresh_outdir(const std::string& tag) {
    const char* env = std::getenv("ED_TEST_TMP_DIR");
    std::string base = env ? std::string(env)
                            : std::string("/tmp/ed_dist_workflows_save");
    return base + "/" + tag;
}

bool hdf5_has_dataset(const std::string& filepath,
                      const std::string& dataset_path) {
    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);
        return file.nameExists(dataset_path);
    } catch (const H5::Exception&) {
        return false;
    }
}

}  // namespace

TEST_CASE("MPI unified-file finalizer: solve rank-0 writes "
          "/eigendata/eigenvalues; other ranks stay silent",
          "[distributed_workflows_save][solve][rank0]") {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    auto dop = std::make_shared<DistributedOperator>(op, MPI_COMM_WORLD);

    const std::string outdir =
        fresh_outdir("solve_unified_np" + std::to_string(size));

    // Rank 0 prepares the directory (clean slate); barrier so all ranks
    // see the empty directory before the workflow runs.
    if (rank == 0) {
        std::error_code ec;
        std::filesystem::remove_all(outdir, ec);
        std::filesystem::create_directories(outdir, ec);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    ed::workflows::SolveOptions opts;
    opts.num_eigs        = 1;
    opts.method          = ed::workflows::SolveMethod::Lanczos;
    opts.max_iter        = 60;
    opts.tolerance       = 1e-12;
    opts.compute_vectors = false;   // eigenvalues only -- safe rank-0 emit
    opts.output_dir      = outdir;
    // Pin to the MPI lane; even on np=1 this exercises the same
    // rank-0 code path (since np=1 falls through the serial finalizer).
    opts.backend.allow_gpu = false;
    opts.backend.allow_mpi = true;

    auto R = ed::workflows::solve(*dop, opts);

    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 contract: hdf5_path filled, file exists, dataset present.
    const std::string h5_path = outdir + "/ed_results.h5";
    if (rank == 0) {
        INFO("solve rank=0 hdf5_path = " << R.hdf5_path
             << "  expected = " << h5_path);
        REQUIRE(!R.eigenvalues.empty());
        REQUIRE(R.hdf5_path == h5_path);
        REQUIRE(std::filesystem::exists(h5_path));
        REQUIRE(hdf5_has_dataset(h5_path, "/eigendata/eigenvalues"));
    } else {
        // Non-root ranks must not claim authorship of the unified
        // file; they still hold the replicated eigenvalues in memory.
        INFO("solve rank=" << rank << " hdf5_path = " << R.hdf5_path);
        REQUIRE(!R.eigenvalues.empty());
        REQUIRE(R.hdf5_path.empty());
    }
}

TEST_CASE("MPI unified-file finalizer: serial lane (np=1) "
          "still emits /eigendata/eigenvalues at output_dir",
          "[distributed_workflows_save][solve][serial]") {
    // Sanity / regression: the rank-0 emit must continue to fire when
    // running np=1 (where the geometry is technically "distributed
    // with 1 rank"). This protects against an accidental
    // ``geom.is_distributed() && size > 1`` regression in the
    // finalizer guard.
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        SUCCEED("only meaningful on np=1");
        return;
    }

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    auto dop = std::make_shared<DistributedOperator>(op, MPI_COMM_WORLD);

    const std::string outdir = fresh_outdir("solve_unified_serial");
    std::error_code ec;
    std::filesystem::remove_all(outdir, ec);
    std::filesystem::create_directories(outdir, ec);

    ed::workflows::SolveOptions opts;
    opts.num_eigs        = 1;
    opts.method          = ed::workflows::SolveMethod::Lanczos;
    opts.max_iter        = 60;
    opts.tolerance       = 1e-12;
    opts.compute_vectors = false;
    opts.output_dir      = outdir;
    opts.backend.allow_gpu = false;
    opts.backend.allow_mpi = true;

    auto R = ed::workflows::solve(*dop, opts);

    const std::string h5_path = outdir + "/ed_results.h5";
    REQUIRE(!R.eigenvalues.empty());
    REQUIRE(R.hdf5_path == h5_path);
    REQUIRE(std::filesystem::exists(h5_path));
    REQUIRE(hdf5_has_dataset(h5_path, "/eigendata/eigenvalues"));
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        // Silence non-root Catch output so the test summary is readable.
        FILE* dummy = std::freopen("/dev/null", "w", stdout);
        (void)dummy;
    }

    int result = Catch::Session().run(argc, argv);

    int global_result = 0;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);

    MPI_Finalize();
    return global_result;
}

// =============================================================================
// src/cli/ed_distributed_main.cpp  (Phase 3b #4)
//
// Standalone MPI driver that exercises the Phase 3b distributed solvers
// from `ed::distributed`. It is intentionally tiny (no JSON, no HDF5 I/O,
// no observable construction) -- the goal is a self-contained launcher
// that cluster sysadmins can use to verify a build / queue / interconnect
// triple end-to-end before scientists try the full DSSF/FTLM workflows on
// top of it.
//
// Build target:  ed_distributed_main  (only when -DWITH_MPI=ON)
//
// Usage (single node, four ranks):
//
//     mpiexec -np 4 ./ed_distributed_main \
//             --mode lanczos \
//             --N 24 \
//             --J 1.0 \
//             --periodic 1 \
//             --max-iter 100 \
//             --reorth 1 \
//             --seed 42
//
//     mpiexec -np 4 ./ed_distributed_main \
//             --mode ftlm \
//             --N 20 \
//             --J 1.0 \
//             --periodic 1 \
//             --max-iter 80 \
//             --samples 32 \
//             --groups 2 \
//             --betas "0.1,0.5,1.0,2.0"
//
// Output: one line per rank (rank 0 prints summary), parseable
// `key=value` style for grep-based regression suites.
//
// HONEST SCOPE: this driver builds an *unsymmetrised* Heisenberg chain
// in the full 2^N Hilbert space (no Sz / momentum sectors). For
// production runs at "honest 36+", a future Phase 3b #5 shim must
// project into a fixed-Sz sector and pass the projected operator to
// DistributedOperator -- this CLI is the launcher hook that workflow
// will eventually plug into.
// =============================================================================

#ifndef WITH_MPI
#  error "ed_distributed_main requires WITH_MPI=ON"
#endif

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_ftlm.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/multi_gpu.h>  // Phase 3c: NCCL status banner

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

namespace {

struct CliArgs {
    std::string mode = "lanczos";       // "lanczos" | "ftlm"
    std::uint64_t N = 12;
    double J = 1.0;
    bool periodic = false;
    std::uint64_t max_iter = 100;
    std::uint64_t exct = 1;
    bool reorth = true;
    unsigned long seed = 12345UL;
    int n_samples = 32;
    int n_groups = 1;
    std::vector<double> betas = {0.1, 0.5, 1.0};
    bool verbose = false;
};

[[noreturn]] void fail(const std::string& msg) {
    std::cerr << "ed_distributed_main: " << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

std::vector<double> parse_csv_doubles(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        try { out.push_back(std::stod(tok)); }
        catch (...) { fail("could not parse double from `" + tok + "`"); }
    }
    return out;
}

CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](int n) {
            if (i + n >= argc) fail("missing value for " + k);
        };
        if (k == "--mode")          { need(1); a.mode = argv[++i]; }
        else if (k == "--N")        { need(1); a.N = std::stoull(argv[++i]); }
        else if (k == "--J")        { need(1); a.J = std::stod(argv[++i]); }
        else if (k == "--periodic") { need(1); a.periodic = std::atoi(argv[++i]) != 0; }
        else if (k == "--max-iter") { need(1); a.max_iter = std::stoull(argv[++i]); }
        else if (k == "--exct")     { need(1); a.exct = std::stoull(argv[++i]); }
        else if (k == "--reorth")   { need(1); a.reorth = std::atoi(argv[++i]) != 0; }
        else if (k == "--seed")     { need(1); a.seed = std::stoul(argv[++i]); }
        else if (k == "--samples")  { need(1); a.n_samples = std::atoi(argv[++i]); }
        else if (k == "--groups")   { need(1); a.n_groups = std::atoi(argv[++i]); }
        else if (k == "--betas")    { need(1); a.betas = parse_csv_doubles(argv[++i]); }
        else if (k == "--verbose")  { a.verbose = true; }
        else if (k == "--help" || k == "-h") {
            std::cout <<
              "ed_distributed_main: Phase 3b distributed solver driver.\n"
              "Options:\n"
              "  --mode {lanczos|ftlm}     solver to run (default: lanczos)\n"
              "  --N <int>                 number of spin-1/2 sites (default: 12)\n"
              "  --J <double>              Heisenberg coupling (default: 1.0)\n"
              "  --periodic {0|1}          periodic BC (default: 0)\n"
              "  --max-iter <int>          Lanczos iterations (default: 100)\n"
              "  --exct <int>              eigenvalues to keep (lanczos mode)\n"
              "  --reorth {0|1}            full re-orthogonalization (default: 1)\n"
              "  --seed <ulong>            RNG seed (default: 12345)\n"
              "  --samples <int>           FTLM samples (ftlm mode, default: 32)\n"
              "  --groups <int>            outer parallelism over samples\n"
              "                            (must divide world size; default: 1)\n"
              "  --betas \"b1,b2,...\"     comma-separated betas (ftlm mode)\n"
              "  --verbose                 verbose progress to stdout\n";
            std::exit(0);
        }
        else fail("unknown arg `" + k + "`");
    }
    return a;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

std::unique_ptr<Operator> build_chain(std::uint64_t N, double J, bool periodic) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    using Complex = std::complex<double>;
    const Complex Jr(J, 0.0);
    const Complex Jh(0.5 * J, 0.0);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) {
        std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i;
        t.op_type_2 = 2; t.site_index_2 = j;
        t.coefficient = Jr; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 0; t.site_index = i;
        t.op_type_2 = 1; t.site_index_2 = j;
        t.coefficient = Jh; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 1; t.site_index = i;
        t.op_type_2 = 0; t.site_index_2 = j;
        t.coefficient = Jh; t.is_two_body = true;
        op->transform_data_.push_back(t);
    }
    return op;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        CliArgs a = parse_args(argc, argv);

        if (world_rank == 0) {
            std::cout << "ed_distributed_main: mode=" << a.mode
                      << " N=" << a.N
                      << " J=" << a.J
                      << " periodic=" << (a.periodic ? 1 : 0)
                      << " world_size=" << world_size
                      << " hilbert_dim=" << (1ULL << a.N)
                      << std::endl;
            // Phase 3c banner: print NCCL availability so cluster smoke
            // logs record whether the distributed Lanczos dot/norm path
            // can route through GPU collectives. The launcher itself
            // still uses the CPU MPI Allreduce path; this is purely a
            // diagnostic for now.
            std::cout << "ed_distributed_main: "
                      << ed::distributed::multi_gpu::nccl_status_string()
                      << std::endl;
        }

        auto t0 = std::chrono::steady_clock::now();
        auto op_unique = build_chain(a.N, a.J, a.periodic);
        std::shared_ptr<Operator> op(std::move(op_unique));

        if (a.mode == "lanczos") {
            ed::distributed::DistributedOperator dop(op, MPI_COMM_WORLD);
            ed::distributed::DistributedLanczosOptions lopts;
            lopts.max_iter    = a.max_iter;
            lopts.exct        = a.exct;
            lopts.full_reorth = a.reorth;
            lopts.verbose     = a.verbose;
            lopts.seed        = a.seed;

            auto res = ed::distributed::distributed_lanczos(dop, lopts);

            if (world_rank == 0) {
                std::cout << "elapsed_s=" << seconds_since(t0)
                          << " iterations=" << res.iterations
                          << " plan_bytes_rank0=" << dop.plan_bytes()
                          << " local_n_rank0=" << dop.local_size()
                          << std::endl;
                for (std::size_t k = 0; k < res.eigenvalues.size(); ++k) {
                    std::cout << "  eig[" << k << "]=" << res.eigenvalues[k]
                              << std::endl;
                }
            }
        }
        else if (a.mode == "ftlm") {
            ed::distributed::DistributedFtlmOptions fopts;
            fopts.n_samples         = a.n_samples;
            fopts.n_groups          = a.n_groups;
            fopts.lanczos_max_iter  = a.max_iter;
            fopts.betas             = a.betas;
            fopts.seed_offset       = a.seed;
            fopts.verbose           = a.verbose;

            auto res = ed::distributed::distributed_ftlm(
                op, fopts, MPI_COMM_WORLD);

            if (world_rank == 0) {
                std::cout << "elapsed_s=" << seconds_since(t0)
                          << " samples_used=" << res.samples_used
                          << std::endl;
                for (std::size_t b = 0; b < a.betas.size(); ++b) {
                    std::cout << "  beta=" << a.betas[b]
                              << " Z=" << res.Z[b] << std::endl;
                }
            }
        }
        else {
            fail("unknown --mode `" + a.mode + "` (use lanczos or ftlm)");
        }
    } catch (const std::exception& e) {
        std::cerr << "[rank " << world_rank << "] exception: " << e.what()
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}

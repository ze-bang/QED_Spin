// =============================================================================
// src/cli/ed_distributed_main.cpp
//
// Standalone MPI driver that exercises the ed::distributed solvers on either:
//
//   1. A built-in Heisenberg chain (legacy --N/--J/--periodic flags), useful
//      for cluster smoke-tests without any data files.
//
//   2. An arbitrary operator loaded from disk via --directory <path>, which
//      reads ``InterAll.dat`` / ``Trans.dat`` (and optionally
//      ``automorphism_results/`` for symmetry-projected runs).
//
// Build target:  ed_distributed_main  (only when -DWITH_MPI=ON)
//
// Usage examples:
//
//     # Built-in Heisenberg chain on 4 ranks:
//     mpiexec -np 4 ./ed_distributed_main \
//             --mode lanczos --N 24 --J 1.0 --periodic 1 --max-iter 100
//
//     # Arbitrary operator from a directory of dat files:
//     mpiexec -np 8 ./ed_distributed_main \
//             --mode lanczos --directory ed_runs/24site --num-sites 24 \
//             --max-iter 200 --exct 4 \
//             --result-file ed_runs/24site/result.h5
//
//     # Symmetry-projected distributed Lanczos (sector 0):
//     mpiexec -np 8 ./ed_distributed_main \
//             --mode lanczos --directory ed_runs/24site --num-sites 24 \
//             --use-symmetry --sector-index 0 --max-iter 200
//
//     # Distributed canonical TPQ:
//     mpiexec -np 4 ./ed_distributed_main \
//             --mode tpq --directory ed_runs/24site --num-sites 24 \
//             --betas "0.1,0.5,1.0,2.0" --samples 32 --groups 4 \
//             --result-file ed_runs/24site/tpq.h5
//
// Output:
//   * stdout: parseable `key=value` style for grep-based regression suites.
//   * --result-file (rank 0 only): HDF5 with /eigenvalues, /betas,
//     /energy, /variance, /Z, /elapsed_s as appropriate. The Python
//     workflow (qed.diag(H, device='mpi', ...)) reads this back so
//     callers get a real EDResults instead of a CompletedProcess.
//   * --eigenvector-dir (rank 0 only): one ``rank_<r>.h5`` slab per
//     rank, plus a ``manifest.json``. The Python workflow stitches
//     these into a global eigenvector on demand.
// =============================================================================

#ifndef WITH_MPI
#  error "ed_distributed_main requires WITH_MPI=ON"
#endif

#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/distributed/distributed_ftlm.h>
#ifdef ED_HAVE_NCCL
#  include <ed/distributed/distributed_ftlm_gpu.h>
#endif
#include <ed/distributed/distributed_krylov_schur.h>
#ifdef ED_HAVE_NCCL
#  include <ed/distributed/distributed_krylov_schur_gpu.h>
#endif
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/distributed/distributed_tpq.h>
#include <ed/distributed/multi_gpu.h>

// distributed_lanczos_gpu lives in ed_distributed_gpu, which is built
// only when (WITH_MPI && WITH_CUDA && NCCL_FOUND). The library
// propagates ED_HAVE_NCCL=1 as a PUBLIC compile_definition, so this
// guard is the right test for whether `distributed_lanczos_gpu` will
// actually link. A WITH_CUDA-only build (no NCCL) has the right
// headers but no link target.
#ifdef ED_HAVE_NCCL
#  include <ed/distributed/distributed_lanczos_gpu.h>
#  include <ed/distributed/distributed_tpq_gpu.h>
#endif

#include <H5Cpp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

namespace {

struct CliArgs {
    // -------- Solver mode -------------------------------------------------
    std::string mode = "lanczos";       // "lanczos" | "krylov_schur" | "ftlm" | "tpq"

    // -------- Hamiltonian source -----------------------------------------
    // Either: built-in Heisenberg chain (legacy default)
    std::uint64_t N = 12;
    double J = 1.0;
    bool periodic = false;
    // Or: load from disk (--directory wins over the legacy flags)
    std::string directory;
    std::string interaction_basename = "InterAll.dat";
    std::string single_site_basename = "Trans.dat";
    std::int64_t num_sites = -1;        // required when --directory is set
    double spin_length = 0.5;

    // -------- Solver knobs (Lanczos) -------------------------------------
    std::uint64_t max_iter = 100;
    std::uint64_t exct = 1;
    bool reorth = true;
    unsigned long seed = 12345UL;
    bool gpu = false;
    bool gpu_resident_spmv = false;
    bool compute_eigenvectors = false;

    // -------- Solver knobs (FTLM / TPQ) ----------------------------------
    int n_samples = 32;
    int n_groups = 1;
    std::vector<double> betas = {0.1, 0.5, 1.0};
    double delta_beta = 0.05;
    std::uint64_t taylor_order = 30;
    bool compute_variance = false;

    // -------- Symmetry projection (Layer 5a) -----------------------------
    bool use_symmetry = false;
    std::size_t sector_index = 0;

    // -------- Result IO (Layer 4 / 5b) -----------------------------------
    std::string result_file;
    std::string eigenvector_dir;

    // -------- Diagnostics ------------------------------------------------
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
        if      (k == "--mode")          { need(1); a.mode = argv[++i]; }
        else if (k == "--N")             { need(1); a.N = std::stoull(argv[++i]); }
        else if (k == "--J")             { need(1); a.J = std::stod(argv[++i]); }
        else if (k == "--periodic")      { need(1); a.periodic = std::atoi(argv[++i]) != 0; }
        else if (k == "--directory")     { need(1); a.directory = argv[++i]; }
        else if (k == "--interaction")   { need(1); a.interaction_basename = argv[++i]; }
        else if (k == "--single-site")   { need(1); a.single_site_basename = argv[++i]; }
        else if (k == "--num-sites")     { need(1); a.num_sites = std::stoll(argv[++i]); }
        else if (k == "--spin-length")   { need(1); a.spin_length = std::stod(argv[++i]); }
        else if (k == "--max-iter")      { need(1); a.max_iter = std::stoull(argv[++i]); }
        else if (k == "--exct")          { need(1); a.exct = std::stoull(argv[++i]); }
        else if (k == "--reorth")        { need(1); a.reorth = std::atoi(argv[++i]) != 0; }
        else if (k == "--seed")          { need(1); a.seed = std::stoul(argv[++i]); }
        else if (k == "--samples")       { need(1); a.n_samples = std::atoi(argv[++i]); }
        else if (k == "--groups")        { need(1); a.n_groups = std::atoi(argv[++i]); }
        else if (k == "--betas")         { need(1); a.betas = parse_csv_doubles(argv[++i]); }
        else if (k == "--verbose")       { a.verbose = true; }
        else if (k == "--gpu")           { a.gpu = true; }
        else if (k == "--delta-beta")    { need(1); a.delta_beta = std::stod(argv[++i]); }
        else if (k == "--taylor-order")  { need(1); a.taylor_order = std::stoull(argv[++i]); }
        else if (k == "--compute-variance")    { a.compute_variance = true; }
        else if (k == "--gpu-resident-spmv")   { a.gpu_resident_spmv = true; }
        else if (k == "--compute-eigenvectors"){ a.compute_eigenvectors = true; }
        else if (k == "--use-symmetry")        { a.use_symmetry = true; }
        else if (k == "--sector-index")  { need(1); a.sector_index = std::stoull(argv[++i]); }
        else if (k == "--result-file")   { need(1); a.result_file = argv[++i]; }
        else if (k == "--eigenvector-dir") { need(1); a.eigenvector_dir = argv[++i]; }
        else if (k == "--help" || k == "-h") {
            std::cout <<
              "ed_distributed_main: distributed solver driver.\n"
              "Hamiltonian source (pick one):\n"
              "  --N --J --periodic        built-in Heisenberg chain (default)\n"
              "  --directory <path>        load InterAll.dat / Trans.dat from path\n"
              "  --num-sites <int>         (REQUIRED with --directory) site count\n"
              "  --interaction <name>      override basename (default InterAll.dat)\n"
              "  --single-site <name>      override basename (default Trans.dat)\n"
              "  --spin-length <float>     spin (default 0.5)\n"
              "Solver mode:\n"
              "  --mode {lanczos|krylov_schur|ftlm|tpq}\n"
              "                            solver to run (default: lanczos)\n"
              "                            krylov_schur is thick-restart\n"
              "                            Lanczos with locking, useful when\n"
              "                            the requested num eigenvalues\n"
              "                            doesn't fit in --max-iter / 4\n"
              "                            of one Krylov sweep.\n"
              "  --max-iter <int>          Lanczos iterations (default: 100)\n"
              "  --exct <int>              eigenvalues to keep (lanczos mode)\n"
              "  --reorth {0|1}            full re-orthogonalization (default: 1)\n"
              "  --seed <ulong>            RNG seed (default: 12345)\n"
              "  --samples <int>           FTLM/TPQ samples (default: 32 / 8)\n"
              "  --groups <int>            outer parallelism over samples\n"
              "  --betas \"b1,b2,...\"      comma-separated betas (ftlm/tpq)\n"
              "  --delta-beta <double>     (tpq) imag-time substep (default: 0.05)\n"
              "  --taylor-order <int>      (tpq) Taylor truncation (default: 30)\n"
              "  --compute-variance        (tpq) also report <H^2>-<H>^2\n"
              "Symmetry projection (uses DistributedSymmetryOperator):\n"
              "  --use-symmetry            project onto a symmetry sector\n"
              "                            (requires --directory with\n"
              "                             automorphism_results/ inside)\n"
              "  --sector-index <int>      which sector (default: 0)\n"
              "Multi-GPU (requires WITH_CUDA=ON + NCCL_FOUND):\n"
              "  --gpu                     (lanczos only) distributed_lanczos_gpu\n"
              "  --gpu-resident-spmv       NCCL halo SpMV (no PCIe round-trip)\n"
              "Eigenvectors:\n"
              "  --compute-eigenvectors    return rank-local Ritz slabs\n"
              "  --eigenvector-dir <path>  directory for per-rank slab HDF5\n"
              "                            (one rank_<r>.h5 per rank +\n"
              "                             manifest.json on rank 0)\n"
              "Result IO:\n"
              "  --result-file <path>      rank 0 dumps HDF5 result here\n"
              "                            (datasets: /eigenvalues, /betas,\n"
              "                             /energy, /variance, /Z, /elapsed_s,\n"
              "                             /iterations, /samples_used)\n"
              "Diagnostics:\n"
              "  --verbose                 verbose progress to stdout\n";
            std::exit(0);
        }
        else fail("unknown arg `" + k + "`");
    }

    if (!a.directory.empty() && a.num_sites < 0) {
        fail("--directory requires --num-sites <int> (the site count is "
             "not stored in InterAll.dat / Trans.dat).");
    }
    if (a.use_symmetry && a.directory.empty()) {
        fail("--use-symmetry requires --directory <path> "
             "(the symmetry kernel reads automorphism_results/ from disk).");
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

/// Expand per-generator phase factors into per-group-element phases.
///
/// The on-disk JSON convention used by qed.workflow._write_symmetry_directory
/// (and the legacy automorphism_finder) stores ``phase_factors[k] = exp(2πi
/// q_k / o_k)`` (one entry per generator), and the in-memory streaming
/// kernel reconstructs the per-group-element character on the fly via
/// ``power_representation``. ``DistributedSymmetryOperator`` instead expects
/// per-group-element phases pre-multiplied (length = |max_clique|), so we
/// do the expansion here once after load. No-op if the stored phases are
/// already per-group-element.
void expand_per_generator_phases(SymmetryGroupInfo& info) {
    using Complex = std::complex<double>;
    const std::size_t n_group = info.max_clique.size();
    const std::size_t n_gen   = info.generators.size();
    if (n_group == 0 || n_gen == 0) return;
    if (info.power_representation.size() != n_group) return;
    for (auto& sector : info.sectors) {
        if (sector.phase_factors.size() == n_group) continue;
        if (sector.phase_factors.size() != n_gen) {
            throw std::runtime_error(
                "ed_distributed_main: sector " +
                std::to_string(sector.sector_id) +
                " has phase_factors size " +
                std::to_string(sector.phase_factors.size()) +
                "; expected " + std::to_string(n_gen) +
                " (per generator) or " + std::to_string(n_group) +
                " (per group element).");
        }
        const std::vector<Complex> per_gen = sector.phase_factors;
        std::vector<Complex> per_elem(n_group, Complex(1.0, 0.0));
        for (std::size_t g = 0; g < n_group; ++g) {
            const auto& powers = info.power_representation[g];
            Complex chi(1.0, 0.0);
            for (std::size_t k = 0; k < powers.size() && k < per_gen.size(); ++k) {
                if (powers[k] == 0) continue;
                if (powers[k] == 1) {
                    chi *= per_gen[k];
                } else {
                    chi *= std::pow(per_gen[k], static_cast<double>(powers[k]));
                }
            }
            per_elem[g] = chi;
        }
        sector.phase_factors = std::move(per_elem);
    }
}

std::unique_ptr<Operator> load_from_directory(const CliArgs& a) {
    auto op = std::make_unique<Operator>(static_cast<std::uint64_t>(a.num_sites),
                                         static_cast<float>(a.spin_length));
    namespace fs = std::filesystem;
    const fs::path dir(a.directory);
    const fs::path interall = dir / a.interaction_basename;
    const fs::path trans    = dir / a.single_site_basename;
    if (fs::exists(interall)) {
        op->loadFromInterAllFile(interall.string());
    }
    if (fs::exists(trans)) {
        op->loadFromFile(trans.string());
    }
    if (op->transform_data_.empty()) {
        fail("--directory `" + a.directory + "` contained no Hamiltonian "
             "terms (looked for " + a.interaction_basename + " / " +
             a.single_site_basename + ").");
    }
    return op;
}

// ---------------------------------------------------------------------------
// HDF5 result-file helpers (rank 0 only). Skip silently when --result-file
// is empty so the legacy stdout-only path keeps working.
// ---------------------------------------------------------------------------

void create_result_file(const std::string& filepath, const std::string& mode) {
    if (filepath.empty()) return;
    namespace fs = std::filesystem;
    const fs::path p(filepath);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    H5::H5File file(filepath, H5F_ACC_TRUNC);
    H5::StrType str_type(H5::PredType::C_S1, mode.size() + 1);
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attr = file.createAttribute("mode", str_type, scalar);
    attr.write(str_type, mode.c_str());
    file.close();
}

void dump_doubles(const std::string& filepath, const std::string& dataset,
                  const std::vector<double>& v) {
    if (filepath.empty()) return;
    HDF5IO::saveArray(filepath, dataset, v);
}

void dump_int_attr(const std::string& filepath, const std::string& key,
                   long long value) {
    if (filepath.empty()) return;
    H5::H5File file(filepath, H5F_ACC_RDWR);
    H5::DataSpace scalar(H5S_SCALAR);
    if (file.attrExists(key)) file.removeAttr(key);
    H5::Attribute attr = file.createAttribute(
        key, H5::PredType::NATIVE_LLONG, scalar);
    attr.write(H5::PredType::NATIVE_LLONG, &value);
    file.close();
}

void dump_double_attr(const std::string& filepath, const std::string& key,
                      double value) {
    if (filepath.empty()) return;
    H5::H5File file(filepath, H5F_ACC_RDWR);
    H5::DataSpace scalar(H5S_SCALAR);
    if (file.attrExists(key)) file.removeAttr(key);
    H5::Attribute attr = file.createAttribute(
        key, H5::PredType::NATIVE_DOUBLE, scalar);
    attr.write(H5::PredType::NATIVE_DOUBLE, &value);
    file.close();
}

// ---------------------------------------------------------------------------
// Per-rank eigenvector slab dump (Layer 5b). Each rank writes its own
// rank_<r>.h5 with /slab[k] = local_n complex doubles and an attr listing
// the global offset + total dim. Rank 0 also writes manifest.json so the
// Python loader can reconstruct without re-running MPI.
// ---------------------------------------------------------------------------

void dump_eigenvector_slabs(
    const std::string& outdir,
    int rank,
    int world_size,
    std::uint64_t local_n,
    std::uint64_t local_offset,
    std::uint64_t global_dim,
    const std::vector<std::vector<std::complex<double>>>& eigvecs_local,
    const std::vector<double>& eigenvalues
) {
    if (outdir.empty()) return;
    namespace fs = std::filesystem;
    if (rank == 0) fs::create_directories(outdir);
    MPI_Barrier(MPI_COMM_WORLD);

    const std::string path = outdir + "/rank_" + std::to_string(rank) + ".h5";
    H5::H5File file(path, H5F_ACC_TRUNC);

    // Per-rank metadata as scalar attributes.
    H5::DataSpace scalar(H5S_SCALAR);
    auto write_attr_ll = [&](const std::string& k, long long v) {
        H5::Attribute a = file.createAttribute(k, H5::PredType::NATIVE_LLONG, scalar);
        a.write(H5::PredType::NATIVE_LLONG, &v);
    };
    write_attr_ll("rank", rank);
    write_attr_ll("world_size", world_size);
    write_attr_ll("local_n", static_cast<long long>(local_n));
    write_attr_ll("local_offset", static_cast<long long>(local_offset));
    write_attr_ll("global_dim", static_cast<long long>(global_dim));
    write_attr_ll("n_eigenvectors", static_cast<long long>(eigvecs_local.size()));

    // Eigenvalues (replicated on every rank, but cheap to dump).
    if (!eigenvalues.empty()) {
        hsize_t e_dims[1] = {eigenvalues.size()};
        H5::DataSpace e_space(1, e_dims);
        H5::DataSet e_ds = file.createDataSet("/eigenvalues",
                                              H5::PredType::NATIVE_DOUBLE,
                                              e_space);
        e_ds.write(eigenvalues.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Per-eigenvector slab as 2*local_n doubles (interleaved real/imag).
    H5::Group g = file.createGroup("/slab");
    for (std::size_t k = 0; k < eigvecs_local.size(); ++k) {
        const auto& v = eigvecs_local[k];
        std::vector<double> interleaved(2 * v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            interleaved[2 * i + 0] = v[i].real();
            interleaved[2 * i + 1] = v[i].imag();
        }
        hsize_t dims[1] = {interleaved.size()};
        H5::DataSpace space(1, dims);
        H5::DataSet ds = g.createDataSet(std::to_string(k),
                                         H5::PredType::NATIVE_DOUBLE,
                                         space);
        ds.write(interleaved.data(), H5::PredType::NATIVE_DOUBLE);
    }
    g.close();
    file.close();

    // Rank 0 writes a tiny manifest so the Python loader knows how many
    // ranks contributed and in what order.
    if (rank == 0) {
        std::ofstream m(outdir + "/manifest.json");
        m << "{\n"
          << "  \"world_size\": " << world_size << ",\n"
          << "  \"global_dim\": " << global_dim << ",\n"
          << "  \"n_eigenvectors\": " << eigvecs_local.size() << ",\n"
          << "  \"layout\": \"slab\",\n"
          << "  \"complex_storage\": \"interleaved_real_imag_double\"\n"
          << "}\n";
    }
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

        // Determine effective N now that we know whether we're loading
        // from disk or building a chain.
        const std::uint64_t effective_N =
            a.directory.empty() ? a.N : static_cast<std::uint64_t>(a.num_sites);

        if (world_rank == 0) {
            std::cout << "ed_distributed_main: mode=" << a.mode
                      << " N=" << effective_N
                      << " source=" << (a.directory.empty()
                                            ? "chain"
                                            : a.directory)
                      << " use_symmetry=" << (a.use_symmetry ? 1 : 0)
                      << " world_size=" << world_size
                      << " hilbert_dim=" << (1ULL << effective_N)
                      << std::endl;
#ifdef ED_HAVE_NCCL
            std::cout << "ed_distributed_main: "
                      << ed::distributed::multi_gpu::nccl_status_string()
                      << std::endl;
#else
            std::cout << "ed_distributed_main: "
                         "NCCL/GPU collectives unavailable in this build "
                         "(WITH_CUDA=OFF or NCCL not found)."
                      << std::endl;
#endif
        }

        auto t0 = std::chrono::steady_clock::now();
        std::shared_ptr<Operator> op;
        if (a.directory.empty()) {
            op = std::shared_ptr<Operator>(
                build_chain(a.N, a.J, a.periodic).release());
        } else {
            op = std::shared_ptr<Operator>(load_from_directory(a).release());
            if (a.use_symmetry) {
                op->symmetry_info.loadFromDirectory(a.directory);
                expand_per_generator_phases(op->symmetry_info);
                if (a.sector_index >= op->symmetry_info.sectors.size()) {
                    fail("--sector-index " + std::to_string(a.sector_index) +
                         " >= number of sectors (" +
                         std::to_string(op->symmetry_info.sectors.size()) +
                         ") in " + a.directory + "/automorphism_results/");
                }
            }
        }

        // Initialise the result file (rank 0 only) so per-mode dumps below
        // can append datasets without TRUNCATE collisions.
        if (world_rank == 0 && !a.result_file.empty()) {
            create_result_file(a.result_file, a.mode);
        }
        // All ranks must wait until the file is created before any
        // collective dump (in practice only rank 0 dumps, but the
        // barrier prevents races with downstream subprocess parsing).
        MPI_Barrier(MPI_COMM_WORLD);

        if (a.mode == "lanczos") {
            // Build either DistributedOperator (full Hilbert space) or
            // DistributedSymmetryOperator (symmetry-projected sector).
            // Both implement the same SpMV interface, so the lanczos
            // call below is identical modulo the operator type.
            std::unique_ptr<ed::distributed::DistributedOperator> dop_full;
            std::unique_ptr<ed::distributed::DistributedSymmetryOperator> dop_sym;
            std::uint64_t local_n = 0, local_offset = 0, global_dim = 0;
            std::size_t plan_bytes = 0;

            if (a.use_symmetry) {
                dop_sym = std::make_unique<ed::distributed::DistributedSymmetryOperator>(
                    op, a.sector_index, MPI_COMM_WORLD);
                local_n      = dop_sym->local_size();
                local_offset = dop_sym->local_offset();
                global_dim   = dop_sym->global_dim();
                plan_bytes   = dop_sym->halo_plan()
                                   ? sizeof(*dop_sym->halo_plan())
                                   : 0;
            } else {
                dop_full = std::make_unique<ed::distributed::DistributedOperator>(
                    op, MPI_COMM_WORLD);
                local_n      = dop_full->local_size();
                local_offset = dop_full->local_offset();
                global_dim   = (1ULL << effective_N);
                plan_bytes   = dop_full->plan_bytes();
            }

            if (a.gpu) {
#ifdef ED_HAVE_NCCL
                ed::distributed::DistributedLanczosGPUOptions gopts;
                gopts.max_iter         = a.max_iter;
                gopts.exct             = a.exct;
                gopts.seed             = a.seed;
                gopts.verbose          = a.verbose;
                gopts.gpu_resident_spmv = a.gpu_resident_spmv;

                ed::distributed::DistributedLanczosGPUResult res;
                if (a.use_symmetry) {
                    // Phase D step 1: --mode lanczos --gpu --use-symmetry
                    // routes through DistributedSymmetryOperatorGPU
                    // (NCCL pairwise SendRecv halo + on-device CSR SpMV
                    // on the orbit row slab). The CPU `dop_sym` carries
                    // the orbit basis + LPT partition + halo plan; the
                    // GPU wrapper is built inside the function.
                    res = ed::distributed::distributed_lanczos_gpu_symmetry(
                        *dop_sym, gopts);
                } else {
                    res = ed::distributed::distributed_lanczos_gpu(
                        *dop_full, gopts);
                }

                if (world_rank == 0) {
                    std::cout << "elapsed_s=" << seconds_since(t0)
                              << " iterations=" << res.iterations
                              << " backend=gpu"
                              << " gpu_resident_spmv="
                              << (a.gpu_resident_spmv ? 1 : 0)
                              << " plan_bytes_rank0=" << plan_bytes
                              << " local_n_rank0=" << local_n
                              << std::endl;
                    for (std::size_t k = 0; k < res.eigenvalues.size(); ++k) {
                        std::cout << "  eig[" << k << "]="
                                  << res.eigenvalues[k] << std::endl;
                    }
                    dump_doubles(a.result_file, "/eigenvalues",
                                 res.eigenvalues);
                    dump_int_attr(a.result_file, "iterations", res.iterations);
                    dump_double_attr(a.result_file, "elapsed_s",
                                     seconds_since(t0));
                }
                // GPU path doesn't yet return rank-local Krylov slabs,
                // so --eigenvector-dir is a no-op here. We still emit
                // the manifest so the Python loader can detect the gap.
                if (!a.eigenvector_dir.empty()) {
                    if (world_rank == 0) {
                        std::cerr << "ed_distributed_main: warning -- "
                                     "--eigenvector-dir requested with "
                                     "--mode lanczos --gpu, which doesn't "
                                     "yet store rank-local Krylov slabs. "
                                     "Skipping slab dump."
                                  << std::endl;
                    }
                }
#else
                fail("--mode lanczos --gpu requires the GPU collective "
                     "backend (WITH_CUDA=ON + NCCL_FOUND).");
#endif
            } else {
                ed::distributed::DistributedLanczosOptions lopts;
                lopts.max_iter            = a.max_iter;
                lopts.exct                = a.exct;
                lopts.full_reorth         = a.reorth;
                lopts.verbose             = a.verbose;
                lopts.seed                = a.seed;
                lopts.compute_eigenvectors =
                    a.compute_eigenvectors || !a.eigenvector_dir.empty();

                ed::distributed::DistributedLanczosResult res;
                if (a.use_symmetry) {
                    res = ed::distributed::distributed_lanczos_symmetry(
                        *dop_sym, lopts);
                } else {
                    res = ed::distributed::distributed_lanczos(
                        *dop_full, lopts);
                }

                if (world_rank == 0) {
                    std::cout << "elapsed_s=" << seconds_since(t0)
                              << " iterations=" << res.iterations
                              << " backend=cpu"
                              << " plan_bytes_rank0=" << plan_bytes
                              << " local_n_rank0=" << local_n
                              << " global_dim=" << global_dim
                              << std::endl;
                    for (std::size_t k = 0; k < res.eigenvalues.size(); ++k) {
                        std::cout << "  eig[" << k << "]="
                                  << res.eigenvalues[k] << std::endl;
                    }
                    dump_doubles(a.result_file, "/eigenvalues",
                                 res.eigenvalues);
                    dump_int_attr(a.result_file, "iterations", res.iterations);
                    dump_int_attr(a.result_file, "global_dim", global_dim);
                    dump_double_attr(a.result_file, "elapsed_s",
                                     seconds_since(t0));
                }

                // Eigenvector slab round-trip. Only emitted when the
                // user explicitly asked for slabs (--eigenvector-dir);
                // otherwise we skip the per-rank reconstruction +
                // disk write entirely.
                if (!a.eigenvector_dir.empty() && lopts.compute_eigenvectors) {
                    std::vector<std::vector<std::complex<double>>> vecs;
                    vecs.reserve(res.eigenvalues.size());
                    for (std::size_t k = 0; k < res.eigenvalues.size(); ++k) {
                        std::vector<std::complex<double>> psi_k;
                        ed::distributed::reconstruct_local_eigenvector(
                            res, k, psi_k);
                        vecs.push_back(std::move(psi_k));
                    }
                    dump_eigenvector_slabs(
                        a.eigenvector_dir, world_rank, world_size,
                        local_n, local_offset, global_dim,
                        vecs, res.eigenvalues);
                }
            }
        }
        else if (a.mode == "krylov_schur" || a.mode == "krylov-schur" ||
                 a.mode == "ks") {
            // Layer 3 (Phase 9): thick-restart Lanczos with locking.
            // Phase B (device matrix MPI+GPU): on-device variant via
            // `distributed_krylov_schur_gpu`. The symmetry variant
            // would need a templated kernel like the plain Lanczos one
            // (TODO -- Phase D).
            if (a.use_symmetry) {
                fail("--mode krylov_schur --use-symmetry is not yet "
                     "supported. Run --mode lanczos --use-symmetry "
                     "instead, or pre-project to a fixed-Sz block.");
            }
            auto dop = std::make_unique<ed::distributed::DistributedOperator>(
                op, MPI_COMM_WORLD);
            const std::uint64_t local_n      = dop->local_size();
            const std::uint64_t local_offset = dop->local_offset();
            const std::uint64_t global_dim   = (1ULL << effective_N);
            const std::size_t plan_bytes     = dop->plan_bytes();

            ed::distributed::DistributedLanczosOptions lopts;
            lopts.max_iter             = a.max_iter;
            lopts.exct                 = a.exct;
            lopts.full_reorth          = a.reorth;
            lopts.verbose              = a.verbose;
            lopts.seed                 = a.seed;
            lopts.compute_eigenvectors =
                a.compute_eigenvectors || !a.eigenvector_dir.empty();

            ed::distributed::DistributedLanczosResult res;
            const char* backend_label = "cpu_krylov_schur";
            if (a.gpu) {
#ifdef ED_HAVE_NCCL
                if (lopts.compute_eigenvectors) {
                    fail("--mode krylov_schur --gpu does not yet support "
                         "--compute-eigenvectors / --eigenvector-dir. "
                         "The locked Ritz vectors live on device and are "
                         "not staged back to host. Run without --gpu, or "
                         "drop the eigenvector output, for now.");
                }
                res = ed::distributed::distributed_krylov_schur_gpu(
                    op, lopts, MPI_COMM_WORLD, /*device_index=*/-1);
                backend_label = "gpu_krylov_schur";
#else
                fail("--mode krylov_schur --gpu requires WITH_CUDA=ON + "
                     "NCCL_FOUND. This binary was built without GPU "
                     "support; rebuild with -DWITH_CUDA=ON or run "
                     "without --gpu.");
#endif
            } else {
                res = ed::distributed::distributed_krylov_schur(*dop, lopts);
            }

            if (world_rank == 0) {
                std::cout << "elapsed_s=" << seconds_since(t0)
                          << " iterations=" << res.iterations
                          << " backend=" << backend_label
                          << " plan_bytes_rank0=" << plan_bytes
                          << " local_n_rank0=" << local_n
                          << " global_dim=" << global_dim
                          << std::endl;
                for (std::size_t k = 0; k < res.eigenvalues.size(); ++k) {
                    std::cout << "  eig[" << k << "]="
                              << res.eigenvalues[k] << std::endl;
                }
                dump_doubles(a.result_file, "/eigenvalues", res.eigenvalues);
                dump_int_attr(a.result_file, "iterations", res.iterations);
                dump_int_attr(a.result_file, "global_dim", global_dim);
                dump_double_attr(a.result_file, "elapsed_s",
                                 seconds_since(t0));
            }

            if (!a.eigenvector_dir.empty() && lopts.compute_eigenvectors) {
                // distributed_krylov_schur with compute_eigenvectors=true
                // surfaces the locked Ritz slabs as krylov_basis_local
                // with an identity tridiag eigenvector matrix, so the
                // standard reconstruct_local_eigenvector call works.
                std::vector<std::vector<std::complex<double>>> vecs;
                vecs.reserve(res.eigenvalues.size());
                for (std::size_t k = 0; k < res.eigenvalues.size(); ++k) {
                    std::vector<std::complex<double>> psi_k;
                    ed::distributed::reconstruct_local_eigenvector(
                        res, k, psi_k);
                    vecs.push_back(std::move(psi_k));
                }
                dump_eigenvector_slabs(
                    a.eigenvector_dir, world_rank, world_size,
                    local_n, local_offset, global_dim,
                    vecs, res.eigenvalues);
            }
        }
        else if (a.mode == "tpq") {
            if (a.use_symmetry) {
                fail("--mode tpq --use-symmetry is not supported. "
                     "Canonical TPQ acts on a single random state in the "
                     "full sector; projecting onto a symmetry irrep "
                     "destroys the Z normalisation. Pre-project to a "
                     "fixed-Sz block instead, or use --mode ftlm which "
                     "DOES combine across symmetry blocks.");
            }

            ed::distributed::DistributedTpqResult res;
            const char* backend_label = "cpu_mpi";
            if (a.gpu) {
#ifdef ED_HAVE_NCCL
                // Phase 9 / Layer 2: multi-GPU canonical TPQ.
                ed::distributed::DistributedTpqGPUOptions gtopts;
                gtopts.n_samples        = a.n_samples;
                gtopts.n_groups         = a.n_groups;
                gtopts.delta_beta       = a.delta_beta;
                gtopts.taylor_order     = a.taylor_order;
                gtopts.betas            = a.betas;
                gtopts.seed_offset      = a.seed;
                gtopts.compute_variance = a.compute_variance;
                gtopts.verbose          = a.verbose;

                res = ed::distributed::distributed_tpq_gpu(
                    op, gtopts, MPI_COMM_WORLD);
                backend_label = "gpu_mpi";
#else
                fail("--mode tpq --gpu requires WITH_CUDA=ON + NCCL_FOUND.");
#endif
            } else {
                ed::distributed::DistributedTpqOptions topts;
                topts.n_samples        = a.n_samples;
                topts.n_groups         = a.n_groups;
                topts.delta_beta       = a.delta_beta;
                topts.taylor_order     = a.taylor_order;
                topts.betas            = a.betas;
                topts.seed_offset      = a.seed;
                topts.compute_variance = a.compute_variance;
                topts.verbose          = a.verbose;

                res = ed::distributed::distributed_tpq(
                    op, topts, MPI_COMM_WORLD);
            }

            if (world_rank == 0) {
                std::cout << "elapsed_s=" << seconds_since(t0)
                          << " samples_used=" << res.samples_used
                          << " backend=" << backend_label
                          << std::endl;
                for (std::size_t b = 0; b < a.betas.size(); ++b) {
                    std::cout << "  beta=" << a.betas[b]
                              << " E=" << res.energy[b];
                    if (a.compute_variance && b < res.variance.size()) {
                        std::cout << " variance=" << res.variance[b];
                    }
                    std::cout << std::endl;
                }
                dump_doubles(a.result_file, "/betas", a.betas);
                dump_doubles(a.result_file, "/energy", res.energy);
                if (a.compute_variance) {
                    dump_doubles(a.result_file, "/variance", res.variance);
                }
                dump_int_attr(a.result_file, "samples_used", res.samples_used);
                dump_double_attr(a.result_file, "elapsed_s",
                                 seconds_since(t0));
            }
        }
        else if (a.mode == "ftlm") {
            if (a.use_symmetry) {
                fail("--mode ftlm --use-symmetry: distributed FTLM with "
                     "the per-sector symmetry kernel is not yet wired. "
                     "Run per sector by setting --sector-index in a loop "
                     "and aggregating Z by hand for now.");
            }

            ed::distributed::DistributedFtlmResult res;
            const char* backend_label = "cpu_mpi";
            if (a.gpu) {
#ifdef ED_HAVE_NCCL
                // Phase A (device matrix MPI+GPU): on-device FTLM.
                ed::distributed::DistributedFtlmGPUOptions gfopts;
                gfopts.n_samples         = a.n_samples;
                gfopts.n_groups          = a.n_groups;
                gfopts.lanczos_max_iter  = a.max_iter;
                gfopts.betas             = a.betas;
                gfopts.seed_offset       = a.seed;
                gfopts.verbose           = a.verbose;

                res = ed::distributed::distributed_ftlm_gpu(
                    op, gfopts, MPI_COMM_WORLD);
                backend_label = "gpu_mpi";
#else
                fail("--mode ftlm --gpu requires WITH_CUDA=ON + NCCL_FOUND.");
#endif
            } else {
                ed::distributed::DistributedFtlmOptions fopts;
                fopts.n_samples         = a.n_samples;
                fopts.n_groups          = a.n_groups;
                fopts.lanczos_max_iter  = a.max_iter;
                fopts.betas             = a.betas;
                fopts.seed_offset       = a.seed;
                fopts.verbose           = a.verbose;

                res = ed::distributed::distributed_ftlm(
                    op, fopts, MPI_COMM_WORLD);
            }

            if (world_rank == 0) {
                std::cout << "elapsed_s=" << seconds_since(t0)
                          << " samples_used=" << res.samples_used
                          << " backend=" << backend_label
                          << std::endl;
                for (std::size_t b = 0; b < a.betas.size(); ++b) {
                    std::cout << "  beta=" << a.betas[b]
                              << " Z=" << res.Z[b] << std::endl;
                }
                dump_doubles(a.result_file, "/betas", a.betas);
                dump_doubles(a.result_file, "/Z", res.Z);
                dump_int_attr(a.result_file, "samples_used", res.samples_used);
                dump_double_attr(a.result_file, "elapsed_s",
                                 seconds_since(t0));
            }
        }
        else {
            fail("unknown --mode `" + a.mode + "` (use lanczos, ftlm, or tpq)");
        }
    } catch (const std::exception& e) {
        std::cerr << "[rank " << world_rank << "] exception: " << e.what()
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Finalize();
    return 0;
}

// =============================================================================
// benchmarks/bench_symmetry_full_spectrum.cpp
//
// Phase 4 benchmark of the "Optimized symmetry ED + NLCE" plan (Jun 2026):
// compare the cost of computing the COMPLETE eigenvalue spectrum of a 1D
// Heisenberg ring two ways:
//
//   DENSE  : a single LAPACK ZHEEV over the full 2^N Hilbert space
//            (`full_diagonalization(Hv, 2^N, ...)`), O(D^3) time, O(D^2) mem.
//   SYM    : decomposed by all (Sz x translation) blocks via the CPU
//            on-the-fly representative SpMV
//            (`make_cpu_rep_symmetry_backend` + per-block dense ZHEEV),
//            never materialising the 2^N matrix nor the orbit CSR.
//
// Both paths return the same multiset of eigenvalues (verified at the
// unit / Python level); this harness reports WALL TIME and PEAK RSS so the
// memory/throughput win of the symmetry path is visible as N grows.
//
// Unlike the Google-Benchmark micro-benchmarks, this is a standalone
// executable: each (path, N) phase runs in a forked child so the parent
// can read that phase's isolated peak RSS via wait4()/ru_maxrss -- a single
// process would only see a monotone high-water mark dominated by the dense
// path.
//
// Usage:
//   ./bench_symmetry_full_spectrum [Nmin] [Nmax]
//   (defaults to N = 8..14)
// =============================================================================

#include <ed/core/operator.h>
#include <ed/solvers/lanczos.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Complex = std::complex<double>;
using Clock   = std::chrono::steady_clock;

// ---- Heisenberg ring term list (shared by both paths) ----------------------

void push_heisenberg_ring(std::vector<Operator::TransformData>& terms,
                          std::uint64_t N, double J) {
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        terms.push_back(t);
        t.op_type = 0; t.site_index = i; t.op_type_2 = 1;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        terms.push_back(t);
        t.op_type = 1; t.site_index = i; t.op_type_2 = 0;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        terms.push_back(t);
    }
}

// ---- ZN translation symmetry fixtures (same shape the loader expects) ------

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

void write_zN_translation_fixtures(const std::string& dir, int N) {
    const std::string root = dir + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    {
        std::ofstream f(root + "/max_clique.json");
        f << "[";
        for (int g = 0; g < N; ++g) {
            const auto p = translation_perm(N, g);
            f << "[";
            for (std::size_t i = 0; i < p.size(); ++i)
                f << p[i] << (i + 1 < p.size() ? "," : "");
            f << "]" << (g + 1 < N ? "," : "");
        }
        f << "]";
    }
    {
        std::ofstream f(root + "/minimal_generators.json");
        const auto p = translation_perm(N, 1);
        f << "{\"generators\":[{\"permutation\":[";
        for (std::size_t i = 0; i < p.size(); ++i)
            f << p[i] << (i + 1 < p.size() ? "," : "");
        f << "],\"order\":" << N << "}]}";
    }
    {
        std::ofstream f(root + "/sector_metadata.json");
        f.precision(17);
        f << "{\"sectors\":[";
        for (int k = 0; k < N; ++k) {
            const double angle = -2.0 * M_PI * double(k) / double(N);
            f << "{\"sector_id\":" << k
              << ",\"quantum_numbers\":[" << k << "]"
              << ",\"phase_factors\":[{\"real\":" << std::cos(angle)
              << ",\"imag\":" << std::sin(angle) << "}]}";
            if (k + 1 < N) f << ",";
        }
        f << "]}";
    }
}

// ---- The two full-spectrum paths -------------------------------------------

std::size_t run_dense(std::uint64_t N) {
    const std::uint64_t dim = (1ULL << N);
    auto op = std::make_unique<Operator>(N, 0.5f);
    push_heisenberg_ring(op->transform_data_, N, 1.0);

    auto Hv = [&](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<std::size_t>(n));
    };
    std::vector<double> eigs;
    full_diagonalization(Hv, dim, /*num_eigs=*/dim, eigs, /*dir=*/"",
                         /*compute_eigenvectors=*/false);
    return eigs.size();
}

std::size_t run_symmetry(std::uint64_t N, const std::string& dir) {
    write_zN_translation_fixtures(dir, static_cast<int>(N));
    SymmetryGroupInfo info;
    info.loadFromDirectory(dir);

    std::size_t total = 0;
    for (std::int64_t n_up = 0; n_up <= static_cast<std::int64_t>(N); ++n_up) {
        // CSR-free lazy-rep SectorOperators: the on-the-fly representative
        // SpMV path (never materialises the 2^N matrix nor the orbit CSR).
        auto ops = ed::symmetry::build_fixed_sz_sector_operators_lazy(
            N, 0.5f, n_up, info,
            [&](ed::symmetry::SectorOperator& op) {
                push_heisenberg_ring(op.transform_data_, N, 1.0);
            });

        for (auto& op : ops) {
            const std::size_t sd = op->dim();
            if (sd == 0) continue;
            auto Hv = [&](const Complex* in, Complex* out, int n) {
                op->apply(in, out, static_cast<std::size_t>(n));
            };
            std::vector<double> eigs;
            full_diagonalization(Hv, sd, /*num_eigs=*/sd, eigs, "",
                                 /*compute_eigenvectors=*/false);
            total += eigs.size();
        }
    }
    return total;
}

// ---- fork-per-phase driver so peak RSS is isolated -------------------------

struct PhaseResult {
    double      wall_s   = 0.0;
    long        rss_kb   = 0;   // ru_maxrss of the child (KiB on Linux)
    std::size_t n_eigs   = 0;
    bool        ok       = false;
};

PhaseResult run_phase_forked(const char* label, std::uint64_t N,
                             bool symmetry, const std::string& dir) {
    PhaseResult r;
    const pid_t pid = fork();
    if (pid < 0) { std::perror("fork"); return r; }

    if (pid == 0) {  // ---- child ----
        // Silence the solver's progress chatter so only the parent's
        // formatted table reaches the terminal.
        std::freopen("/dev/null", "w", stdout);
        const auto t0 = Clock::now();
        std::size_t n = symmetry ? run_symmetry(N, dir) : run_dense(N);
        const double secs =
            std::chrono::duration<double>(Clock::now() - t0).count();
        // Communicate (wall, n_eigs) to the parent via a tiny temp file;
        // the exit status only carries 8 bits. NOTE: _Exit() skips
        // destructors, so the stream MUST be flushed/closed explicitly
        // before we leave -- otherwise the parent reads an empty file.
        {
            std::ofstream f(dir + "/phase_" + label + ".txt");
            f.precision(9);
            f << secs << " " << n << "\n";
            f.flush();
            f.close();
        }
        std::_Exit(0);
    }

    int status = 0;
    struct rusage ru{};
    wait4(pid, &status, 0, &ru);
    r.rss_kb = ru.ru_maxrss;
    r.ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    std::ifstream f(dir + "/phase_" + std::string(label) + ".txt");
    if (f) { f >> r.wall_s >> r.n_eigs; }
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    std::uint64_t Nmin = 8, Nmax = 12;
    if (argc > 1) Nmin = std::strtoull(argv[1], nullptr, 10);
    if (argc > 2) Nmax = std::strtoull(argv[2], nullptr, 10);

    std::printf("# Heisenberg ring full-spectrum: DENSE (2^N ZHEEV) vs "
                "SYM (Sz x translation rep blocks)\n");
    std::printf("# %-3s %-10s %12s %12s %10s  %10s %12s %12s %10s  %8s\n",
                "N", "dim", "dense_s", "dense_rssMB", "dense_eig",
                "sym_s", "sym_rssMB", "sym_eig", "speedup", "rss_x");
    // Flush BEFORE the first fork: a buffered (pipe-redirected) stdout would
    // otherwise be inherited by every child and re-emitted on its freopen.
    std::fflush(stdout);

    for (std::uint64_t N = Nmin; N <= Nmax; ++N) {
        const std::uint64_t dim = (1ULL << N);
        std::error_code ec;
        const std::string dir =
            (std::filesystem::temp_directory_path() /
             ("bench_symfs_N" + std::to_string(N))).string();
        std::filesystem::create_directories(dir, ec);

        // Run SYM first (low memory) then DENSE: isolated via fork either way.
        PhaseResult sym   = run_phase_forked("sym",   N, true,  dir);
        PhaseResult dense = run_phase_forked("dense", N, false, dir);

        const double dense_mb = dense.rss_kb / 1024.0;
        const double sym_mb   = sym.rss_kb / 1024.0;
        const double speedup  = (sym.wall_s > 0.0)
                                    ? dense.wall_s / sym.wall_s : 0.0;
        const double rss_x    = (sym_mb > 0.0) ? dense_mb / sym_mb : 0.0;

        std::printf("  %-3llu %-10llu %12.4f %12.1f %10zu  "
                    "%10.4f %12.1f %12zu %10.2f  %8.2f\n",
                    (unsigned long long)N, (unsigned long long)dim,
                    dense.wall_s, dense_mb, dense.n_eigs,
                    sym.wall_s, sym_mb, sym.n_eigs, speedup, rss_x);
        std::fflush(stdout);

        if (dense.n_eigs != dim || sym.n_eigs != dim) {
            std::printf("  ! WARNING N=%llu eigenvalue count mismatch "
                        "(dense=%zu sym=%zu expected=%llu)\n",
                        (unsigned long long)N, dense.n_eigs, sym.n_eigs,
                        (unsigned long long)dim);
        }
        std::filesystem::remove_all(dir, ec);
    }
    return 0;
}

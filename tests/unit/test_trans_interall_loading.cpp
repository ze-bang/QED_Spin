// =============================================================================
// test_trans_interall_loading
//
// Writes synthetic Trans.dat and InterAll.dat files that encode the same
// Heisenberg chain the in-memory fixture builds, loads them via
// Operator::loadFromFile / loadFromInterAllFile, and verifies both paths
// produce identical spectra.
// =============================================================================

#include "common/test_harness.h"

#include <ed/core/construct_ham.h>

#include <cstdio>
#include <filesystem>  // P0.12
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>  // P0.12

using namespace ed_tests;

// Write a Heisenberg-chain InterAll.dat whose contents reproduce the
// build_heisenberg_chain() programmatic Hamiltonian (J=1, OBC).
// Trans.dat is written but empty of one-body terms (no magnetic field).
static void write_files(uint64_t N, const std::string& dir) {
    // P0.12: was system("mkdir -p '...'") (shell-quoted).
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);

    {
        std::ofstream f(dir + "/Trans.dat");
        f << "===================\n";
        f << "num       0\n";
        f << "===================\n";
        f << "===================\n";
        f << "===================\n";
    }
    {
        std::ofstream f(dir + "/InterAll.dat");
        // 3 lines per bond: Sz_i Sz_j, 0.5 S+ S-, 0.5 S- S+
        uint64_t nbonds = (N - 1);
        uint64_t nlines = 3 * nbonds;
        f << "===================\n";
        f << "num       " << nlines << "\n";
        f << "===================\n";
        f << "===================\n";
        f << "===================\n";
        for (uint64_t i = 0; i < N - 1; ++i) {
            uint64_t j = i + 1;
            // Op codes follow construct_ham: 0=S+, 1=S-, 2=Sz
            f << "        2         " << i
              << "           2         " << j
              << "    1.000000    0.000000\n";
            f << "        0         " << i
              << "           1         " << j
              << "    0.500000    0.000000\n";
            f << "        1         " << i
              << "           0         " << j
              << "    0.500000    0.000000\n";
        }
    }
}

int main() {
    TestContext ctx("test_trans_interall_loading");
    const uint64_t N = 4;
    const uint64_t dim = 1ULL << N;

    std::string dir = make_scratch_dir("trans_interall");
    write_files(N, dir);

    // Reference: programmatic in-memory construction
    auto op_ref = build_heisenberg_chain(N, 1.0);
    auto ref = reference_from_operator(*op_ref, dim);

    // Loaded: via the public file-parsing API
    auto op_loaded = std::make_unique<Operator>(N, 0.5f);
    op_loaded->loadFromFile(dir + "/Trans.dat");
    op_loaded->loadFromInterAllFile(dir + "/InterAll.dat");
    auto loaded = reference_from_operator(*op_loaded, dim);

    check_eigs_close(ctx, loaded.eigs, ref.eigs, ref.eigs.size(), 1e-10,
                     "spectra from Trans+InterAll match programmatic H");

    // Extra sanity: the two dense matrices must agree element-wise.
    double err = (loaded.H - ref.H).norm() /
                 std::max(ref.H.norm(), 1e-30);
    check(ctx, err < 1e-12,
          "loaded H matches programmatic H element-wise",
          "rel err = " + std::to_string(err));

    return ctx.summary_exit_code();
}

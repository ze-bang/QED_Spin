// =============================================================================
// test_sector_loop_gate
//
// Validation pin for the operator-collapse refactor (Phase 3, Jun 2026):
// the production per-sector loop now keys off ``ed::core::SectorSetView``
// (the carrier-free replacement for the retired
// ``StreamingSymmetryHandle``).
//
// ``SectorSetView`` owns a (compacted) ``SectorOperatorSet`` -- one
// standalone ``ed::symmetry::SectorOperator`` per non-empty irrep, driven by
// the unified ``CpuMatVecBackend<SymmetryBasisPolicy>`` -- and exposes the
// same random-access surface (``num_sectors()`` over RAW irrep indices,
// ``sector(raw_k)`` returning the persistent operator or ``nullptr`` for a
// dropped/empty irrep). This test builds the set for the Heisenberg Z_N ring
// (N=6, J=1) and verifies that each surviving sector operator IS a
// ``SectorOperator`` and that the minimum sector ground state equals the
// Bethe-ansatz GS.
// =============================================================================

#include "common/catch2_harness.h"

#include <cstdlib>

#include <ed/core/make_operator.h>      // SectorOperatorSet + SectorSetView
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>     // build_full_sector_operators

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

using namespace ed_tests;

namespace {

constexpr double kE0_N6 = -2.802775637731995;  // Bethe-ansatz GS, J=1, N=6

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
            for (size_t i = 0; i < p.size(); ++i) {
                f << p[i] << (i + 1 < p.size() ? "," : "");
            }
            f << "]" << (g + 1 < N ? "," : "");
        }
        f << "]";
    }
    {
        std::ofstream f(root + "/minimal_generators.json");
        const auto p = translation_perm(N, 1);
        f << "{\"generators\":[{\"permutation\":[";
        for (size_t i = 0; i < p.size(); ++i) {
            f << p[i] << (i + 1 < p.size() ? "," : "");
        }
        f << "],\"order\":" << N << "}]}";
    }
    {
        std::ofstream f(root + "/sector_metadata.json");
        f << std::setprecision(17);
        f << "{\"sectors\":[";
        for (int k = 0; k < N; ++k) {
            const double angle = -2.0 * M_PI * static_cast<double>(k) /
                                 static_cast<double>(N);
            const double re = std::cos(angle);
            const double im = std::sin(angle);
            f << "{\"sector_id\":" << k
              << ",\"quantum_numbers\":[" << k << "]"
              << ",\"phase_factors\":[{\"real\":" << re
              << ",\"imag\":" << im << "}]}";
            if (k + 1 < N) f << ",";
        }
        f << "]}";
    }
}

void add_heisenberg_pbc_terms(ed::symmetry::SectorOperator& op, int N) {
    const Complex J_real(1.0, 0.0), J_half(0.5, 0.0);
    for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(N); ++i) {
        const std::uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, J_real);
        op.addTwoBodyTerm(0, i, 1, j, J_half);
        op.addTwoBodyTerm(1, i, 0, j, J_half);
    }
}

// Assemble a SectorOperatorSet (the payload SectorSetView wraps) from the
// carrier-free full-Hilbert sector builder, mirroring exactly what
// make_sector_operators_tagged() populates.
ed::SectorOperatorSet build_full_sector_set(int N, const SymmetryGroupInfo& info) {
    ed::SectorOperatorSet set;
    std::vector<std::size_t> ids;
    set.operators = ed::symmetry::build_full_sector_operators_lazy(
        static_cast<std::uint64_t>(N), 0.5f, info,
        [&](ed::symmetry::SectorOperator& op) { add_heisenberg_pbc_terms(op, N); },
        &ids);
    set.num_raw_sectors = info.sectors.size();
    for (const auto& sec : info.sectors) {
        set.all_quantum_numbers.push_back(sec.quantum_numbers);
    }
    for (std::size_t i = 0; i < set.operators.size(); ++i) {
        ed::SectorTag tag;
        tag.sector_index    = ids[i];
        tag.sector_dim      = static_cast<std::uint64_t>(set.operators[i]->dim());
        tag.quantum_numbers = info.sectors[ids[i]].quantum_numbers;
        tag.n_up            = -1;
        set.tags.push_back(std::move(tag));
    }
    return set;
}

double lowest_eigenvalue(ed::LinearOperator& op) {
    const std::size_t d = op.dim();
    Eigen::MatrixXcd H(d, d);
    std::vector<Complex> e(d, Complex(0.0, 0.0));
    std::vector<Complex> col(d, Complex(0.0, 0.0));
    for (std::size_t c = 0; c < d; ++c) {
        std::fill(e.begin(), e.end(), Complex(0.0, 0.0));
        e[c] = Complex(1.0, 0.0);
        op.apply(e.data(), col.data(), d);
        for (std::size_t r = 0; r < d; ++r) H(r, c) = col[r];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(H);
    return solver.eigenvalues()(0);
}

}  // namespace

TEST_CASE("sector_loop_gate: SectorSetView routes through SectorOperator (N=6)",
          "[symmetry][sector_loop][gate][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("sector_loop_gate", "N6");
    write_zN_translation_fixtures(dir, N);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));

    ed::core::SectorSetView view(build_full_sector_set(N, info));
    REQUIRE(view.num_sectors() == info.sectors.size());

    double e0_min = std::numeric_limits<double>::infinity();
    std::size_t surviving = 0;

    for (std::size_t k = 0; k < view.num_sectors(); ++k) {
        // SectorSetView path -> persistent standalone SectorOperator
        // (nullptr for an empty / dropped irrep).
        ed::symmetry::SectorOperator* gated = view.sector(k);
        if (gated == nullptr) continue;  // empty irrep
        ++surviving;
        const std::size_t sd = gated->dim();
        REQUIRE(sd > 0);
        REQUIRE(sd == view.sector_tag(k).sector_dim);

        // The per-sector operator must be Hermitian (probe symmetry of the
        // dense matvec) before we trust its spectrum.
        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) {
                std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            } else {
                x = random_unit_vector(sd, (k + 5) * 49157ULL + 3);
            }
            std::vector<Complex> y(sd, Complex(0.0, 0.0));
            gated->apply(x.data(), y.data(), sd);
            // <x|H x> must be real for a Hermitian operator.
            Complex quad(0.0, 0.0);
            for (std::size_t i = 0; i < sd; ++i) quad += std::conj(x[i]) * y[i];
            REQUIRE(std::abs(quad.imag()) < 1e-10);
        }

        e0_min = std::min(e0_min, lowest_eigenvalue(*gated));
    }

    REQUIRE(surviving > 0);
    REQUIRE(std::abs(e0_min - kE0_N6) < 1e-10);
}

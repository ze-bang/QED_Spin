// =============================================================================
// test_sector_loop_gate
//
// Validation pin for the operator-collapse refactor (Phase C, Jun 2026):
// the production cutover in
// ``ed::core::StreamingSymmetryHandle::sector(k)``.
//
// The handle returns a standalone ``ed::symmetry::SectorOperator`` (the
// collapse-target path, driven by the unified
// ``CpuMatVecBackend<SymmetryBasisPolicy>``). The legacy back-referencing
// ``SectorView`` and the ``ED_SYMMETRY_SECTOR_OPERATOR`` gate were removed in
// Phase C. This test wraps a ``StreamingSymmetryOperator`` in a handle and
// verifies -- for the Heisenberg Z_N ring (N=6, J=1) -- that the handle's
// per-sector operator IS a ``SectorOperator`` and that the minimum sector
// ground state equals the Bethe-ansatz GS.
// =============================================================================

#include "common/catch2_harness.h"

#include <cstdlib>  // setenv -- MUST precede the first gate read

#include <ed/core/sector_loop.h>
#include <ed/core/streaming_symmetry.h>
#include <ed/symmetry/sector_operator.h>

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

void add_heisenberg_pbc_terms(StreamingSymmetryOperator& op, int N) {
    const Complex J_real(1.0, 0.0), J_half(0.5, 0.0);
    for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(N); ++i) {
        std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        op.transform_data_.push_back(t);
        t.op_type = 0; t.site_index = i; t.op_type_2 = 1;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op.transform_data_.push_back(t);
        t.op_type = 1; t.site_index = i; t.op_type_2 = 0;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op.transform_data_.push_back(t);
    }
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

TEST_CASE("sector_loop_gate: handle routes through SectorOperator (N=6)",
          "[symmetry][sector_loop][gate][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("sector_loop_gate", "N6");
    write_zN_translation_fixtures(dir, N);

    auto sym_op = std::make_unique<StreamingSymmetryOperator>(
        static_cast<std::uint64_t>(N), 0.5f);
    add_heisenberg_pbc_terms(*sym_op, N);
    REQUIRE_NOTHROW(sym_op->generateSymmetrySectorsStreaming(dir));

    ed::core::StreamingSymmetryHandle handle(sym_op.get());
    REQUIRE(handle.num_sectors() == sym_op->getNumSectors());

    double e0_min = std::numeric_limits<double>::infinity();

    for (std::size_t k = 0; k < handle.num_sectors(); ++k) {
        const std::size_t sd = sym_op->getSectorDimension(k);
        if (sd == 0) continue;

        // Handle path -> standalone SectorOperator (collapse-target).
        auto gated = handle.sector(k);
        REQUIRE(gated != nullptr);
        REQUIRE(gated->dim() == sd);
        // The handle operator must be a SectorOperator.
        REQUIRE(dynamic_cast<ed::symmetry::SectorOperator*>(gated.get()) != nullptr);

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

    REQUIRE(std::abs(e0_min - kE0_N6) < 1e-10);
}

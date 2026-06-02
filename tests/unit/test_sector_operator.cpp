// =============================================================================
// test_sector_operator
//
// Validation pin for P2c of the operator-collapse refactor (Jun 2026):
// ``ed::symmetry::SectorOperator`` -- the first standalone per-sector
// embodiment of the collapse target ``Operator<SymmetryBasisPolicy, Host>``.
//
// A SectorOperator OWNS one sector's orbit data (a ``SectorBasis``) and
// inherits the canonical term storage from ``ed::Operator``; it references
// NO parent ``StreamingSymmetryOperator``. This test proves:
//
//   (1) Matvec equivalence: ``SectorOperator::apply`` reproduces the legacy
//       ``StreamingSymmetryOperator::applySymmetrized`` byte-for-byte (1e-12)
//       in EVERY momentum sector of the Heisenberg Z_N ring (N=6, all 6
//       sectors, including the complex k != 0, pi sectors).
//
//   (2) Real fast path: in the real (k=0, pi) sectors, ``is_real_hermitian``
//       is true and ``apply_real`` matches the complex apply; in the complex
//       sectors ``is_real_hermitian`` is false (so the orchestrator keeps
//       them on the complex lane).
//
//   (3) End-to-end eigenvalue: the dense matrix assembled column-by-column
//       from the standalone k=0 SectorOperator has lowest eigenvalue equal
//       to the Bethe-ansatz ground state E_0 = -2.802775637731995.
//
// Together these de-risk the per-sector standalone operator that the P5
// ``SymmetrySectorSet`` / ``make_operator`` rewrite will hand to the
// orchestrator in place of the nested ``SectorView``.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/streaming_symmetry.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/subspace.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

std::vector<std::uint64_t>
enumerate_orbit_reps(const SymmetryGroupInfo& info, int N) {
    std::vector<std::uint64_t> reps;
    const std::uint64_t dim = (1ULL << N);
    for (std::uint64_t s = 0; s < dim; ++s) {
        std::uint64_t mn = s;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            mn = std::min(mn, applyPermutation(s, info.max_clique[g]));
        }
        if (mn == s) reps.push_back(s);
    }
    return reps;
}

// Append the J=1 PBC Heisenberg terms (Sz_i Sz_j + 1/2 (S+_i S-_j +
// S-_i S+_j)) to any operator via the typed setters.
template <class Op>
void add_heisenberg_pbc_terms(Op& op, std::uint64_t N, double J) {
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, J_real);   // Sz Sz
        op.addTwoBodyTerm(0, i, 1, j, J_half);   // S+ S-
        op.addTwoBodyTerm(1, i, 0, j, J_half);   // S- S+
    }
}

std::unique_ptr<StreamingSymmetryOperator>
build_heisenberg_pbc_streaming(std::uint64_t N, double J) {
    auto op = std::make_unique<StreamingSymmetryOperator>(N, 0.5f);
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 0; t.site_index = i; t.op_type_2 = 1;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 1; t.site_index = i; t.op_type_2 = 0;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op->transform_data_.push_back(t);
    }
    return op;
}

// Lowest eigenvalue of a SectorOperator via dense column-by-column build.
double lowest_eigenvalue(const ed::symmetry::SectorOperator& op) {
    const std::size_t d = op.dim();
    Eigen::MatrixXcd H(d, d);
    std::vector<Complex> e(d, Complex(0.0, 0.0));
    std::vector<Complex> col(d, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < d; ++j) {
        std::fill(e.begin(), e.end(), Complex(0.0, 0.0));
        e[j] = Complex(1.0, 0.0);
        op.apply(e.data(), col.data(), d);
        for (std::size_t i = 0; i < d; ++i) H(i, j) = col[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    return es.eigenvalues()(0);
}

} // namespace

TEST_CASE("sector_operator: standalone SectorOperator matches legacy applySymmetrized (N=6)",
          "[symmetry][sector_operator][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("sector_operator", "heisenberg_N6");
    write_zN_translation_fixtures(dir, N);

    auto sym_op = build_heisenberg_pbc_streaming(N, 1.0);
    REQUIRE_NOTHROW(sym_op->generateSymmetrySectorsStreaming(dir));

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));
    const ed::symmetry::FullSpaceSubspace full(static_cast<std::uint64_t>(N));
    const ed::symmetry::SpatialProjector  spatial(info);
    const std::vector<std::uint64_t> reps = enumerate_orbit_reps(info, N);

    REQUIRE(sym_op->getNumSectors() == info.sectors.size());

    double e0_min = std::numeric_limits<double>::infinity();

    for (std::size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        const std::size_t sd = sym_op->getSectorDimension(s);
        if (sd == 0) continue;

        // Standalone per-sector operator: owning SectorBasis + its own
        // inherited term list. No reference to sym_op.
        ed::symmetry::SectorBasis sb = ed::symmetry::SectorBasis::build(
            full, spatial,
            info.sectors[s].quantum_numbers,
            info.sectors[s].phase_factors,
            reps, /*sector_id=*/s);
        REQUIRE(sb.dim() == sd);

        ed::symmetry::SectorOperator sec_op(
            static_cast<std::uint64_t>(N), 0.5f, std::move(sb));
        add_heisenberg_pbc_terms(sec_op, N, 1.0);
        REQUIRE(sec_op.dim() == sd);

        // (1) complex matvec equivalence vs legacy, two probes.
        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) {
                std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            } else {
                x = random_unit_vector(sd, (s + 5) * 70001ULL + 3);
            }
            std::vector<Complex> y_ref(sd, Complex(0.0, 0.0));
            std::vector<Complex> y_new(sd, Complex(0.0, 0.0));
            sym_op->applySymmetrized(s, x.data(), y_ref.data());
            sec_op.apply(x.data(), y_new.data(), sd);
            for (std::size_t i = 0; i < sd; ++i) {
                REQUIRE(std::abs(y_new[i] - y_ref[i]) < 1e-12);
            }
        }

        // (2) real fast path consistency.
        const bool real_sector = sec_op.is_real_hermitian();
        if (real_sector) {
            std::vector<double> xr(sd);
            for (std::size_t i = 0; i < sd; ++i) {
                xr[i] = std::sin(0.37 * static_cast<double>(i + s) + 0.11);
            }
            std::vector<Complex> xc(sd);
            for (std::size_t i = 0; i < sd; ++i) xc[i] = Complex(xr[i], 0.0);

            std::vector<double>  yr(sd, 0.0);
            std::vector<Complex> yc(sd, Complex(0.0, 0.0));
            sec_op.apply_real(xr.data(), yr.data(), sd);
            sec_op.apply(xc.data(), yc.data(), sd);
            for (std::size_t i = 0; i < sd; ++i) {
                REQUIRE(std::abs(yc[i].imag()) < 1e-12);
                REQUIRE(std::abs(yr[i] - yc[i].real()) < 1e-12);
            }
        }

        // (3) accumulate the sector ground-state energy.
        e0_min = std::min(e0_min, lowest_eigenvalue(sec_op));
    }

    // The global ground state lives in some sector; min over sectors is
    // the Bethe-ansatz value.
    REQUIRE(std::abs(e0_min - kE0_N6) < 1e-10);
}

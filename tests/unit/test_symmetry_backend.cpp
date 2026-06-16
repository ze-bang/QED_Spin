// =============================================================================
// test_symmetry_backend
//
// Validation pin for P2 of the operator-collapse refactor (Jun 2026):
// ``ed::matvec::make_cpu_symmetry_backend`` +
// ``CpuMatVecBackend<SymmetryBasisPolicy>``.
//
// Proves that the unified host backend, driven by the matrix-free kernel
// ``apply_terms<SymmetryBasisPolicy, Scalar>`` over a ``SectorBasis``,
// reproduces the legacy bespoke ``StreamingSymmetryOperator::
// applySymmetrized`` matvec byte-for-byte (to 1e-12) in every momentum
// sector of the Heisenberg Z_N ring (N=6, all 6 sectors -- including the
// complex k != 0, pi sectors that exercise the symmetry weighting's
// imaginary part).
//
// Ordering: ``SectorBasis::build`` iterates the deduplicated, ascending
// orbit-representative list and keeps the non-vanishing orbits in order --
// identical to the legacy "Pass 2" loop over ``unique_orbit_reps_`` --
// so the input/output index spaces line up element-wise and the two
// matvecs can be compared directly.
//
// This de-risks the eventual collapse of the four CPU operator classes
// into a single ``Operator<BasisPolicy, Host>``: the symmetry SpMV path
// now goes through the SAME generic backend as Full / FixedSz, with the
// only difference being the compile-time basis policy.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/symmetry_reference.h"

#include <ed/core/operator.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/matvec/term_storage.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/subspace.h>

#include <functional>

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

// Deduplicated ascending orbit representatives over the full Hilbert
// space (a state is a rep iff it is the numeric minimum of its orbit).
// Matches the legacy ``unique_orbit_reps_`` enumeration verbatim.
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

// Full-Hilbert Heisenberg PBC operator (carrier-free). Provides both the
// term list (classified into the symmetry backend's SoA) and the
// full-space ``H`` apply used by the independent symmetrized reference.
std::unique_ptr<Operator>
build_heisenberg_pbc_full(std::uint64_t N, double J) {
    auto op = std::make_unique<Operator>(N, 0.5f);
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

// Build a non-owning TermView over a locally-classified SoA store.
using TermView_t = ed::matvec::TermViewT<
    ed::matvec::DiagOneBody, ed::matvec::OffDiagOneBody,
    ed::matvec::DiagTwoBody, ed::matvec::MixedTwoBody,
    ed::matvec::OffDiagTwoBody, ed::matvec::ThreeBodyTerm>;

TermView_t make_term_view(const ed::matvec::TermStorage& soa,
                          double spin_l, bool is_real) {
    TermView_t tv;
    tv.diag_one    = &soa.diag_one_body;
    tv.offdiag_one = &soa.offdiag_one_body;
    tv.diag_two    = &soa.diag_two_body;
    tv.mixed_two   = &soa.mixed_two_body;
    tv.offdiag_two = &soa.offdiag_two_body;
    tv.three_body  = &soa.three_body;
    tv.spin_l      = spin_l;
    tv.is_real     = is_real;
    return tv;
}

} // namespace

TEST_CASE("symmetry_backend: CpuMatVecBackend<Symmetry> matches applySymmetrized (N=6)",
          "[symmetry][matvec_backend][N6]")
{
    const int N = 6;
    std::string dir = make_scratch_dir("symmetry_backend", "heisenberg_N6");
    write_zN_translation_fixtures(dir, N);

    // Full-Hilbert operator: supplies both the term list (classified into the
    // symmetry backend's SoA) and the full-space ``H`` apply used by the
    // independent symmetrized reference (no symmetry carrier required).
    auto full_op = build_heisenberg_pbc_full(N, 1.0);

    // Shared term storage: classify the operator's AoS terms once. The
    // symmetry backend consumes EXACTLY this SoA (the only difference
    // versus the full/fixed-Sz lanes is the basis policy).
    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, full_op->transform_data_, full_op->three_body_data_,
        [](const Complex& c) { return c; });
    const TermView_t tv = make_term_view(soa, /*spin_l=*/0.5, /*is_real=*/true);

    // Independent basis machinery for the new path.
    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));
    const ed::symmetry::FullSpaceSubspace full(static_cast<std::uint64_t>(N));
    const ed::symmetry::SpatialProjector  spatial(info);
    const std::vector<std::uint64_t> reps = enumerate_orbit_reps(info, N);
    const double group_size =
        static_cast<double>(info.max_clique.size());

    // Full-Hilbert H apply for the reference (FullBasisPolicy matvec).
    std::function<void(const Complex*, Complex*, std::size_t)> full_apply =
        [&full_op](const Complex* x, Complex* y, std::size_t n) {
            full_op->apply(x, y, n);
        };

    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        // New path: owning SectorBasis (kept alive for the backend's
        // borrowed policy view) + symmetry backend.
        ed::symmetry::SectorBasis sb = ed::symmetry::SectorBasis::build(
            full, spatial,
            info.sectors[s].quantum_numbers,
            info.sectors[s].phase_factors,
            reps, /*sector_id=*/s);
        const std::size_t sd = sb.dim();
        if (sd == 0) continue;

        auto backend = ed::matvec::make_cpu_symmetry_backend<
            ed::matvec::DiagOneBody, ed::matvec::OffDiagOneBody,
            ed::matvec::DiagTwoBody, ed::matvec::MixedTwoBody,
            ed::matvec::OffDiagTwoBody, ed::matvec::ThreeBodyTerm>(sb.policy());
        REQUIRE(backend->dim() == sd);

        // Two probe vectors per sector: a fixed all-ones and a random
        // complex vector (the latter exercises the imaginary channel of
        // the symmetry weighting in the complex momentum sectors).
        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) {
                std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            } else {
                x = random_unit_vector(sd, (s + 7) * 99991ULL + 13);
            }

            std::vector<Complex> y_ref(sd, Complex(0.0, 0.0));
            std::vector<Complex> y_new(sd, Complex(0.0, 0.0));

            ed_tests::apply_symmetrized_reference(
                sb.sector(), static_cast<std::uint64_t>(N), group_size,
                full_apply, x.data(), y_ref.data(), sd);
            backend->apply_complex(&tv, x.data(), y_new.data(), sd);

            double max_abs_diff = 0.0;
            double ref_scale     = 0.0;
            for (std::size_t i = 0; i < sd; ++i) {
                max_abs_diff = std::max(max_abs_diff, std::abs(y_new[i] - y_ref[i]));
                ref_scale    = std::max(ref_scale, std::abs(y_ref[i]));
            }
            INFO("sector " << s << " probe " << probe
                 << " dim " << sd
                 << " max_abs_diff " << max_abs_diff
                 << " ref_scale " << ref_scale);
            REQUIRE(max_abs_diff < 1e-12 * (1.0 + ref_scale));
        }
    }
}

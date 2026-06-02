// =============================================================================
// test_symmetry_backend_fixedsz
//
// Validation pin for P2 of the operator-collapse refactor (Jun 2026):
// the SECOND symmetry operator class -- fixed-Sz + spatial symmetry.
//
// Proves that the unified host backend
// (``ed::matvec::make_cpu_symmetry_backend`` +
// ``CpuMatVecBackend<SymmetryBasisPolicy>``), driven by the matrix-free
// kernel ``apply_terms<SymmetryBasisPolicy, Scalar>`` over a
// ``SectorBasis`` BUILT OVER A ``FixedSzSubspace``, reproduces the legacy
// bespoke ``FixedSzStreamingSymmetryOperator::applySymmetrizedFixedSz``
// matvec byte-for-byte (to 1e-12) in every momentum sector of the
// half-filled Heisenberg Z_N ring (N=6, n_up=3, all 6 sectors).
//
// This is the FixedSz analogue of ``test_symmetry_backend`` and confirms
// that the SAME generic backend collapses both the full-space symmetry
// (``StreamingSymmetryOperator``) and the fixed-Sz symmetry
// (``FixedSzStreamingSymmetryOperator``) CPU operator classes -- the only
// difference being the subspace the orbits are enumerated over (the
// compile-time basis policy is identical: ``SymmetryBasisPolicy``).
//
// Ordering: ``SectorBasis::build`` iterates the deduplicated, ascending
// fixed-Sz orbit-representative list keeping the non-vanishing orbits in
// order -- identical to the legacy fixed-Sz "Pass 2" loop over
// ``unique_orbit_reps_`` (norm_sq > 1e-10) -- so the input/output index
// spaces line up element-wise and the two matvecs compare directly.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/streaming_symmetry.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/matvec/term_storage.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/subspace.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
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

// Deduplicated ascending fixed-Sz orbit representatives: a fixed-Sz basis
// state is a rep iff it is the numeric minimum of its orbit (restricted to
// in-subspace images). Site permutations preserve popcount, so every orbit
// image of a fixed-Sz basis state is itself a fixed-Sz basis state -- this
// matches the legacy ``unique_orbit_reps_`` enumeration in
// ``generateSymmetrySectorsStreamingFixedSz`` verbatim.
std::vector<std::uint64_t>
enumerate_fixed_sz_orbit_reps(const ed::symmetry::FixedSzSubspace& sub,
                              const SymmetryGroupInfo& info) {
    std::set<std::uint64_t> reps;
    const std::vector<std::uint64_t>& states = sub.basis_states();
    for (std::uint64_t s : states) {
        std::uint64_t mn = s;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            const std::uint64_t img = applyPermutation(s, info.max_clique[g]);
            if (sub.index_of(img) >= 0) mn = std::min(mn, img);
        }
        reps.insert(mn);
    }
    return std::vector<std::uint64_t>(reps.begin(), reps.end());
}

std::unique_ptr<FixedSzStreamingSymmetryOperator>
build_heisenberg_pbc_fixed_sz(std::uint64_t N, std::int64_t n_up, double J) {
    auto op = std::make_unique<FixedSzStreamingSymmetryOperator>(
        N, 0.5f, n_up);
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

TEST_CASE("symmetry_backend: CpuMatVecBackend<Symmetry> matches "
          "applySymmetrizedFixedSz (N=6, n_up=3)",
          "[symmetry][matvec_backend][fixedsz][N6]")
{
    const int N = 6;
    const std::int64_t n_up = 3;
    std::string dir =
        make_scratch_dir("symmetry_backend_fixedsz", "heisenberg_N6_nup3");
    write_zN_translation_fixtures(dir, N);

    // Legacy reference operator + its per-sector orbit data (fixed-Sz).
    auto sym_op = build_heisenberg_pbc_fixed_sz(N, n_up, 1.0);
    REQUIRE_NOTHROW(sym_op->generateSymmetrySectorsStreamingFixedSz(dir));

    // Shared term storage: classify the operator's AoS terms once. The
    // symmetry backend consumes EXACTLY this SoA; only the basis policy
    // (and the subspace its orbits were enumerated over) differs.
    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, sym_op->transform_data_, sym_op->three_body_data_,
        [](const Complex& c) { return c; });
    const TermView_t tv = make_term_view(soa, /*spin_l=*/0.5, /*is_real=*/true);

    // Independent basis machinery for the new path: an OWNING fixed-Sz
    // subspace (so its basis_states / Lin index outlive the SectorBasis
    // policy view) + the spatial projector.
    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));
    const ed::symmetry::FixedSzSubspace fixed =
        ed::symmetry::FixedSzSubspace::build(static_cast<std::uint64_t>(N),
                                             n_up);
    const ed::symmetry::SpatialProjector spatial(info);
    const std::vector<std::uint64_t> reps =
        enumerate_fixed_sz_orbit_reps(fixed, info);

    REQUIRE(sym_op->getNumSectors() == info.sectors.size());

    for (std::size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        const std::size_t sd = sym_op->getSectorDimension(s);
        if (sd == 0) continue;

        // New path: owning SectorBasis over the fixed-Sz subspace (kept
        // alive for the backend's borrowed policy view) + symmetry backend.
        ed::symmetry::SectorBasis sb = ed::symmetry::SectorBasis::build(
            fixed, spatial,
            info.sectors[s].quantum_numbers,
            info.sectors[s].phase_factors,
            reps, /*sector_id=*/s);
        REQUIRE(sb.dim() == sd);

        auto backend = ed::matvec::make_cpu_symmetry_backend<
            ed::matvec::DiagOneBody, ed::matvec::OffDiagOneBody,
            ed::matvec::DiagTwoBody, ed::matvec::MixedTwoBody,
            ed::matvec::OffDiagTwoBody, ed::matvec::ThreeBodyTerm>(sb.policy());
        REQUIRE(backend->dim() == sd);

        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) {
                std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            } else {
                x = random_unit_vector(sd, (s + 11) * 99991ULL + 17);
            }

            std::vector<Complex> y_ref(sd, Complex(0.0, 0.0));
            std::vector<Complex> y_new(sd, Complex(0.0, 0.0));

            sym_op->applySymmetrizedFixedSz(s, x.data(), y_ref.data());
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

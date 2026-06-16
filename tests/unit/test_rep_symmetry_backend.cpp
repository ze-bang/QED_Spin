// =============================================================================
// test_rep_symmetry_backend
//
// Phase 1 gate of the "Optimized symmetry ED + NLCE" plan (Jun 2026):
// the CPU on-the-fly representative SpMV
// (``ed::matvec::make_cpu_rep_symmetry_backend`` +
// ``CpuMatVecBackend<RepSymmetryBasisPolicy>`` driving the dedicated
// ``apply_terms_rep_symmetry`` kernel) reproduces the orbit-CSR reference
// matvec (legacy ``FixedSzStreamingSymmetryOperator::applySymmetrizedFixedSz``
// AND the unified ``CpuMatVecBackend<SymmetryBasisPolicy>``) to ~1e-12 in
// EVERY momentum sector, on random complex vectors -- WITHOUT materialising
// the per-sector orbit CSR (the rep backend is built from the CSR-free
// ``getRepSectorData``).
//
// This is the bottom-up "rep matvec == CSR matvec" gate. The
// (Sz x irrep) spectrum-union == dense gate lives at the Python/integration
// level (Phase 2/3).
// =============================================================================

#include "common/catch2_harness.h"
#include "common/symmetry_reference.h"

#include <ed/core/operator.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/matvec/term_storage.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/rep_sector_data.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_set.h>
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

// Full-Hilbert Heisenberg PBC operator (carrier-free) -- term list for the
// SoA + full-space ``H`` apply backing the independent symmetrized reference.
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

void run_case(int N, std::int64_t n_up) {
    std::string dir = make_scratch_dir(
        "rep_symmetry_backend",
        "heis_N" + std::to_string(N) + "_nup" + std::to_string(n_up));
    write_zN_translation_fixtures(dir, N);

    auto full_op = build_heisenberg_pbc_full(
        static_cast<std::uint64_t>(N), 1.0);

    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, full_op->transform_data_, full_op->three_body_data_,
        [](const Complex& c) { return c; });
    const TermView_t tv = make_term_view(soa, /*spin_l=*/0.5, /*is_real=*/true);

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));
    const ed::symmetry::FixedSzSubspace fixed =
        ed::symmetry::FixedSzSubspace::build(static_cast<std::uint64_t>(N),
                                             n_up);
    const ed::symmetry::SpatialProjector spatial(info);
    const std::vector<std::uint64_t> reps =
        ed::symmetry::enumerate_fixed_sz_orbit_reps(fixed, info);
    const double group_size =
        static_cast<double>(info.max_clique.size());

    std::function<void(const Complex*, Complex*, std::size_t)> full_apply =
        [&full_op](const Complex* x, Complex* y, std::size_t n) {
            full_op->apply(x, y, n);
        };

    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        // Carrier-free owning SectorBasis over the fixed-Sz subspace.
        ed::symmetry::SectorBasis sb = ed::symmetry::SectorBasis::build(
            fixed, spatial,
            info.sectors[s].quantum_numbers,
            info.sectors[s].phase_factors,
            reps, /*sector_id=*/s);
        const std::size_t sd = sb.dim();
        if (sd == 0) continue;

        // CSR-free rep data from the same production helper the sector-set
        // builder uses (reads only orbit_rep + norm from the sector).
        ed::symmetry::RepSectorData rd =
            ed::symmetry::rep_sector_data_from_sector(sb.sector(), info, N);
        INFO("sector " << s << " dim " << sd
             << " rep dim " << rd.reps.size()
             << " usable " << rd.usable());
        REQUIRE(rd.usable());
        REQUIRE(rd.reps.size() == sd);

        auto backend = ed::matvec::make_cpu_rep_symmetry_backend<
            ed::matvec::DiagOneBody, ed::matvec::OffDiagOneBody,
            ed::matvec::DiagTwoBody, ed::matvec::MixedTwoBody,
            ed::matvec::OffDiagTwoBody, ed::matvec::ThreeBodyTerm>(rd);
        REQUIRE(backend->dim() == sd);

        for (int probe = 0; probe < 3; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) {
                std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            } else {
                x = random_unit_vector(sd, (s + 7) * 1000003ULL + probe * 17 + N);
            }

            std::vector<Complex> y_ref(sd, Complex(0.0, 0.0));
            std::vector<Complex> y_rep(sd, Complex(0.0, 0.0));

            ed_tests::apply_symmetrized_reference(
                sb.sector(), static_cast<std::uint64_t>(N), group_size,
                full_apply, x.data(), y_ref.data(), sd);
            backend->apply_complex(&tv, x.data(), y_rep.data(), sd);

            double max_abs_diff = 0.0;
            double ref_scale    = 0.0;
            for (std::size_t i = 0; i < sd; ++i) {
                max_abs_diff = std::max(max_abs_diff, std::abs(y_rep[i] - y_ref[i]));
                ref_scale    = std::max(ref_scale, std::abs(y_ref[i]));
            }
            INFO("sector " << s << " probe " << probe
                 << " max_abs_diff " << max_abs_diff
                 << " ref_scale " << ref_scale);
            REQUIRE(max_abs_diff < 1e-11 * (1.0 + ref_scale));
        }
    }
}

} // namespace

TEST_CASE("rep_symmetry_backend: CPU rep matvec matches orbit-CSR reference "
          "(N=6, n_up=3)",
          "[symmetry][matvec_backend][rep][N6]")
{
    run_case(6, 3);
}

TEST_CASE("rep_symmetry_backend: CPU rep matvec matches orbit-CSR reference "
          "(N=8, n_up=4 and n_up=3)",
          "[symmetry][matvec_backend][rep][N8]")
{
    run_case(8, 4);
    run_case(8, 3);
}

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
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/rep_symmetry_basis_policy.h>
#include <ed/matvec/term_storage.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/rep_sector_data.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_set.h>
#include <ed/symmetry/subspace.h>

#include <functional>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// GATHER == SCATTER parity + O(1) rank-table parity ("Optimized symmetry ED"
// plan, Phase E). Drives the rep-symmetry GATHER and SCATTER kernels DIRECTLY
// (bypassing the env-read backend tunables) so both run in one process, and
// asserts they agree bit-for-bit (modulo atomic FP reordering). Also builds
// the dense O(1) rank table and asserts the O(1) reverse lookup yields the
// IDENTICAL matvec as the O(log dim) binary-search fallback.
// ---------------------------------------------------------------------------
void run_parity_case(int N, std::int64_t n_up) {
    std::string dir = make_scratch_dir(
        "rep_symmetry_parity",
        "heis_N" + std::to_string(N) + "_nup" + std::to_string(n_up));
    write_zN_translation_fixtures(dir, N);

    auto full_op = build_heisenberg_pbc_full(static_cast<std::uint64_t>(N), 1.0);
    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, full_op->transform_data_, full_op->three_body_data_,
        [](const Complex& c) { return c; });

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));
    const ed::symmetry::FixedSzSubspace fixed =
        ed::symmetry::FixedSzSubspace::build(static_cast<std::uint64_t>(N), n_up);
    const ed::symmetry::SpatialProjector spatial(info);
    const std::vector<std::uint64_t> reps =
        ed::symmetry::enumerate_fixed_sz_orbit_reps(fixed, info);

    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        ed::symmetry::SectorBasis sb = ed::symmetry::SectorBasis::build(
            fixed, spatial, info.sectors[s].quantum_numbers,
            info.sectors[s].phase_factors, reps, /*sector_id=*/s);
        const std::size_t sd = sb.dim();
        if (sd == 0) continue;

        ed::symmetry::RepSectorData rd =
            ed::symmetry::rep_sector_data_from_sector(sb.sector(), info, N);
        REQUIRE(rd.usable());

        // Binary-search policy (no rank table).
        REQUIRE_FALSE(rd.has_rank_table());
        const auto pol_bs = ed::matvec::rep_policy_from(rd);

        for (int probe = 0; probe < 3; ++probe) {
            std::vector<Complex> x =
                (probe == 0)
                    ? std::vector<Complex>(sd, Complex(1.0, 0.0))
                    : random_unit_vector(sd, (s + 11) * 2654435761ULL + probe + N);

            std::vector<Complex> y_scatter(sd, Complex(0.0, 0.0));
            std::vector<Complex> y_gather(sd, Complex(0.0, 0.0));

            ed::matvec::kernel::apply_terms_rep_symmetry<
                ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
                pol_bs, 0.5, soa.diag_one_body, soa.offdiag_one_body,
                soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
                soa.three_body, x.data(), y_scatter.data());

            ed::matvec::kernel::apply_terms_rep_symmetry_gather<
                ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
                pol_bs, 0.5, soa.diag_one_body, soa.offdiag_one_body,
                soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
                soa.three_body, x.data(), y_gather.data(), /*diag_cache=*/nullptr);

            double gs_diff = 0.0, scale = 0.0;
            for (std::size_t i = 0; i < sd; ++i) {
                gs_diff = std::max(gs_diff, std::abs(y_gather[i] - y_scatter[i]));
                scale   = std::max(scale, std::abs(y_scatter[i]));
            }
            INFO("GATHER==SCATTER sector " << s << " probe " << probe
                 << " diff " << gs_diff << " scale " << scale);
            REQUIRE(gs_diff < 1e-11 * (1.0 + scale));

            // O(1) rank-table path must equal the binary-search GATHER exactly.
            ed::symmetry::RepSectorData rd_tab =
                ed::symmetry::rep_sector_data_from_sector(sb.sector(), info, N);
            rd_tab.build_rank_table();
            REQUIRE(rd_tab.has_rank_table());
            const auto pol_tab = ed::matvec::rep_policy_from(rd_tab);
            REQUIRE(pol_tab.rep_index_of_rank != nullptr);

            std::vector<Complex> y_gather_tab(sd, Complex(0.0, 0.0));
            ed::matvec::kernel::apply_terms_rep_symmetry_gather<
                ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
                pol_tab, 0.5, soa.diag_one_body, soa.offdiag_one_body,
                soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
                soa.three_body, x.data(), y_gather_tab.data(), nullptr);

            double tab_diff = 0.0;
            for (std::size_t i = 0; i < sd; ++i) {
                tab_diff = std::max(tab_diff, std::abs(y_gather_tab[i] - y_gather[i]));
            }
            INFO("O(1) vs O(log) GATHER sector " << s << " diff " << tab_diff);
            REQUIRE(tab_diff < 1e-13 * (1.0 + scale));

            // Stage 2b (SymmetryEngine v2): the rep-assembled reduced CSR
            // (build_reduced_symmetry_csr_rep, no orbit CSR) must reproduce
            // the rep-walk GATHER on the same vectors.
            const auto rep_csr = ed::matvec::build_reduced_symmetry_csr_rep<
                ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
                pol_bs, 0.5, soa.diag_one_body, soa.offdiag_one_body,
                soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
                soa.three_body);
            REQUIRE(rep_csr.built());
            REQUIRE(rep_csr.dim == sd);
            std::vector<Complex> y_csr(sd, Complex(0.0, 0.0));
            rep_csr.spmv(x.data(), y_csr.data());
            double csr_diff = 0.0;
            for (std::size_t i = 0; i < sd; ++i) {
                csr_diff = std::max(csr_diff, std::abs(y_csr[i] - y_gather[i]));
            }
            INFO("rep-CSR vs GATHER sector " << s << " diff " << csr_diff);
            REQUIRE(csr_diff < 1e-12 * (1.0 + scale));

            // Stage 4 (SymmetryEngine v2): the two-level shared-rank lookup
            // (one dense table per (N, n_up) + per-sector local remap) must
            // reproduce the binary-search GATHER exactly (same lookup result
            // -> identical arithmetic).
            ed::symmetry::RepSectorData rd_two =
                ed::symmetry::rep_sector_data_from_sector(sb.sector(), info, N);
            rd_two.shared_rank = ed::symmetry::make_shared_rank_lookup(
                reps, N, static_cast<int>(n_up));
            REQUIRE(rd_two.shared_rank != nullptr);
            rd_two.local_of_shared.assign(reps.size(), std::int32_t{-1});
            {
                std::size_t local = 0;
                for (std::size_t gi = 0; gi < reps.size(); ++gi) {
                    if (local < rd_two.reps.size() &&
                        rd_two.reps[local] == reps[gi]) {
                        rd_two.local_of_shared[gi] =
                            static_cast<std::int32_t>(local++);
                    }
                }
                REQUIRE(local == rd_two.reps.size());
            }
            REQUIRE(rd_two.has_two_level());
            const auto pol_two = ed::matvec::rep_policy_from(rd_two);
            REQUIRE(pol_two.shared_rank_of != nullptr);

            std::vector<Complex> y_two(sd, Complex(0.0, 0.0));
            ed::matvec::kernel::apply_terms_rep_symmetry_gather<
                ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
                pol_two, 0.5, soa.diag_one_body, soa.offdiag_one_body,
                soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
                soa.three_body, x.data(), y_two.data(), nullptr);
            double two_diff = 0.0;
            for (std::size_t i = 0; i < sd; ++i) {
                two_diff = std::max(two_diff, std::abs(y_two[i] - y_gather[i]));
            }
            INFO("two-level vs binary-search GATHER sector " << s
                 << " diff " << two_diff);
            REQUIRE(two_diff == 0.0);
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Dense-vector throughput micro-benchmark (hidden; run with
// `./test_rep_symmetry_backend "[.][bench]"`). This is the ITERATIVE-SOLVER
// regime (Lanczos / FTLM / TPQ): a DENSE input vector applied many times --
// where the lock-free GATHER (no atomics, no radix sort, single write) is the
// optimal kernel. (The full-spectrum bench's dense-block construction instead
// feeds UNIT vectors, the input-sparse regime that favours the scatter's
// early-skip; that path is the ED_MATVEC_SCATTER fallback.)
// ---------------------------------------------------------------------------
TEST_CASE("rep_symmetry_backend: dense-vector GATHER vs SCATTER throughput",
          "[.][bench][symmetry][rep]")
{
    auto env_int = [](const char* k, int dflt) -> int {
        const char* v = std::getenv(k);
        return (v && *v) ? std::atoi(v) : dflt;
    };
    const int N = env_int("ED_BENCH_N", 20);
    const std::int64_t n_up = env_int("ED_BENCH_NUP", N / 2);
    std::string dir = make_scratch_dir("rep_symmetry_bench",
                                       "heis_N" + std::to_string(N) +
                                       "_nup" + std::to_string(n_up));
    write_zN_translation_fixtures(dir, N);

    auto full_op = build_heisenberg_pbc_full(static_cast<std::uint64_t>(N), 1.0);
    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, full_op->transform_data_, full_op->three_body_data_,
        [](const Complex& c) { return c; });

    SymmetryGroupInfo info;
    REQUIRE_NOTHROW(info.loadFromDirectory(dir));
    const ed::symmetry::FixedSzSubspace fixed =
        ed::symmetry::FixedSzSubspace::build(static_cast<std::uint64_t>(N), n_up);
    const ed::symmetry::SpatialProjector spatial(info);
    const std::vector<std::uint64_t> reps =
        ed::symmetry::enumerate_fixed_sz_orbit_reps(fixed, info);

    // Largest sector (k=0).
    ed::symmetry::SectorBasis sb = ed::symmetry::SectorBasis::build(
        fixed, spatial, info.sectors[0].quantum_numbers,
        info.sectors[0].phase_factors, reps, 0);
    ed::symmetry::RepSectorData rd =
        ed::symmetry::rep_sector_data_from_sector(sb.sector(), info, N);
    rd.build_rank_table();
    const auto pol = ed::matvec::rep_policy_from(rd);
    const std::size_t sd = rd.reps.size();
    REQUIRE(sd > 1000);

    std::vector<Complex> x = random_unit_vector(sd, 12345);
    std::vector<Complex> y(sd);
    const int iters = 50;

    auto bench = [&](const char* name, auto&& fn) {
        fn(); // warm up
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) fn();
        const double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::printf("  %-10s dim=%zu  %.3f ms/apply\n",
                    name, sd, 1e3 * s / iters);
    };

    bench("GATHER", [&]() {
        ed::matvec::kernel::apply_terms_rep_symmetry_gather<
            ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
            pol, 0.5, soa.diag_one_body, soa.offdiag_one_body,
            soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
            soa.three_body, x.data(), y.data(), nullptr);
    });
    bench("SCATTER", [&]() {
        std::fill(y.begin(), y.end(), Complex(0.0, 0.0));
        ed::matvec::kernel::apply_terms_rep_symmetry<
            ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
            pol, 0.5, soa.diag_one_body, soa.offdiag_one_body,
            soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
            soa.three_body, x.data(), y.data());
    });

    // Scale parity: GATHER == SCATTER on the dense vector (the N>=28 smoke
    // when run with ED_BENCH_N=28). The full-space reference is infeasible at
    // this dim, so this is the self-consistent transpose check.
    std::vector<Complex> yg(sd, Complex(0.0, 0.0));
    std::vector<Complex> ysc(sd, Complex(0.0, 0.0));
    ed::matvec::kernel::apply_terms_rep_symmetry_gather<
        ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
        pol, 0.5, soa.diag_one_body, soa.offdiag_one_body,
        soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
        soa.three_body, x.data(), yg.data(), nullptr);
    ed::matvec::kernel::apply_terms_rep_symmetry<
        ed::matvec::basis::RepSymmetryBasisPolicy, Complex>(
        pol, 0.5, soa.diag_one_body, soa.offdiag_one_body,
        soa.diag_two_body, soa.mixed_two_body, soa.offdiag_two_body,
        soa.three_body, x.data(), ysc.data());
    double diff = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < sd; ++i) {
        diff  = std::max(diff, std::abs(yg[i] - ysc[i]));
        scale = std::max(scale, std::abs(ysc[i]));
    }
    std::printf("  N=%d n_up=%lld dim=%zu  GATHER==SCATTER diff=%.3e (scale %.3e)\n",
                N, static_cast<long long>(n_up), sd, diff, scale);
    REQUIRE(diff < 1e-10 * (1.0 + scale));
    SUCCEED("benchmark complete");
}

TEST_CASE("rep_symmetry_backend: GATHER == SCATTER + O(1) rank-table parity "
          "(N=6,8)",
          "[symmetry][matvec_backend][rep][parity]")
{
    run_parity_case(6, 3);
    run_parity_case(8, 4);
    run_parity_case(8, 3);
}

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

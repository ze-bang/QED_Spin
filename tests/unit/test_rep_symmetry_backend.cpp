// =============================================================================
// test_rep_symmetry_backend
//
// Phase 1 gate of the "Optimized symmetry ED + NLCE" plan (Jun 2026):
// the CPU on-the-fly representative SpMV
// (``ed::matvec::make_cpu_rep_symmetry_backend`` +
// ``CpuMatVecBackend<RepSymmetryBasisPolicy>`` driving the dedicated
// ``apply_terms_rep_symmetry`` kernel) reproduces an INDEPENDENT reference
// to ~1e-12 in EVERY momentum sector, on random complex vectors: each
// symmetric basis state is expanded over the full 2^N basis
// (``applyPermutation`` + characters), the full-space ``Operator`` is
// applied, and the result is projected back. The rep data come from the
// CSR-free orbit table, exactly as the little-group engine builds them.
//
// This is the bottom-up "rep matvec == full-space matvec" gate. The
// (Sz x irrep) spectrum-union == dense gate lives at the Python/integration
// level.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/basis_utils.h>      // applyPermutation
#include <ed/core/operator.h>
#include <ed/matvec/reduced_symmetry_csr.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/rep_symmetry_basis_policy.h>
#include <ed/matvec/term_storage.h>
#include <ed/symmetry/compiled_group.h>
#include <ed/symmetry/group.h>
#include <ed/symmetry/orbit_table.h>
#include <ed/symmetry/rep_sector_data.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace ed_tests;

namespace {

// Z_N ring translations T^m (element m = 0 .. N-1).
std::vector<std::vector<int>> zn_elements(int N) {
    std::vector<std::vector<int>> e;
    for (int m = 0; m < N; ++m) e.push_back(ed::sym::translation(N, m));
    return e;
}

// Characters of momentum sector k: chi_k(T^m) = exp(2 pi i k m / N).
std::vector<Complex> zn_characters(int N, int k) {
    std::vector<Complex> chi;
    const double two_pi = 2.0 * std::acos(-1.0);
    for (int m = 0; m < N; ++m) chi.push_back(std::polar(1.0, two_pi * k * m / N));
    return chi;
}

// Fixed-Sz orbit table of the ring (min-image representatives + stabilizers).
ed::symmetry::OrbitTable zn_orbit_table(int N, std::int64_t n_up) {
    return ed::symmetry::build_orbit_table_fixed_sz_streaming(
        static_cast<std::uint64_t>(N), static_cast<int>(n_up),
        ed::symmetry::CompiledGroup::from_permutations(zn_elements(N), N));
}

// CSR-free rep data of momentum sector k: surviving representatives with
// their closed-form norms (the little-group engine's recipe).
ed::symmetry::RepSectorData zn_sector(const ed::symmetry::OrbitTable& tab,
                                      int N, std::int64_t n_up, int k) {
    ed::symmetry::RepSectorData d;
    d.n_sites    = N;
    d.group_size = N;
    d.n_up       = static_cast<int>(n_up);
    d.characters = zn_characters(N, k);
    for (const auto& p : zn_elements(N))
        d.perms_flat.insert(d.perms_flat.end(), p.begin(), p.end());
    for (std::size_t i = 0; i < tab.size(); ++i) {
        const double nsq = ed::symmetry::projected_norm_sq(tab, i, d.characters);
        if (nsq <= 1e-12) continue;
        d.reps.push_back(tab.reps[i]);
        d.inv_norms.push_back(1.0 / std::sqrt(nsq));
    }
    return d;
}

// Independent reference: out = (1/|G|) P^dagger H P in, with the columns of P
// the expanded symmetric states inv_norm_j * sum_g conj(chi(g)) |g(r_j)>.
void apply_rep_reference(const ed::symmetry::RepSectorData& rd,
                         const Operator& H, const Complex* in, Complex* out) {
    const std::uint64_t full = 1ULL << rd.n_sites;
    const auto elems = zn_elements(rd.n_sites);
    const std::size_t dim = rd.reps.size();
    std::vector<Complex> psi(full, Complex(0.0, 0.0)), phi(full, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < dim; ++j)
        for (std::size_t g = 0; g < elems.size(); ++g)
            psi[applyPermutation(rd.reps[j], elems[g])] +=
                in[j] * rd.inv_norms[j] * std::conj(rd.characters[g]);
    H.apply(psi.data(), phi.data(), full);
    const double group_norm = 1.0 / static_cast<double>(elems.size());
    for (std::size_t k = 0; k < dim; ++k) {
        Complex acc(0.0, 0.0);
        for (std::size_t g = 0; g < elems.size(); ++g)
            acc += rd.characters[g] * phi[applyPermutation(rd.reps[k], elems[g])];
        out[k] = acc * (group_norm * rd.inv_norms[k]);
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
    auto full_op = build_heisenberg_pbc_full(
        static_cast<std::uint64_t>(N), 1.0);

    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, full_op->transform_data_, full_op->three_body_data_,
        [](const Complex& c) { return c; });
    const TermView_t tv = make_term_view(soa, /*spin_l=*/0.5, /*is_real=*/true);

    const ed::symmetry::OrbitTable tab = zn_orbit_table(N, n_up);

    for (int k = 0; k < N; ++k) {
        ed::symmetry::RepSectorData rd = zn_sector(tab, N, n_up, k);
        const std::size_t sd = rd.reps.size();
        if (sd == 0) continue;
        INFO("sector " << k << " dim " << sd << " usable " << rd.usable());
        REQUIRE(rd.usable());

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
                x = random_unit_vector(sd, (k + 7) * 1000003ULL + probe * 17 + N);
            }

            std::vector<Complex> y_ref(sd, Complex(0.0, 0.0));
            std::vector<Complex> y_rep(sd, Complex(0.0, 0.0));

            apply_rep_reference(rd, *full_op, x.data(), y_ref.data());
            backend->apply_complex(&tv, x.data(), y_rep.data(), sd);

            double max_abs_diff = 0.0;
            double ref_scale    = 0.0;
            for (std::size_t i = 0; i < sd; ++i) {
                max_abs_diff = std::max(max_abs_diff, std::abs(y_rep[i] - y_ref[i]));
                ref_scale    = std::max(ref_scale, std::abs(y_ref[i]));
            }
            INFO("sector " << k << " probe " << probe
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
    auto full_op = build_heisenberg_pbc_full(static_cast<std::uint64_t>(N), 1.0);
    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, full_op->transform_data_, full_op->three_body_data_,
        [](const Complex& c) { return c; });

    const ed::symmetry::OrbitTable tab = zn_orbit_table(N, n_up);
    const std::vector<std::uint64_t>& reps = tab.reps;

    for (int k = 0; k < N; ++k) {
        ed::symmetry::RepSectorData rd = zn_sector(tab, N, n_up, k);
        const std::size_t sd = rd.reps.size();
        if (sd == 0) continue;
        REQUIRE(rd.usable());

        // Binary-search policy (no rank table).
        REQUIRE_FALSE(rd.has_rank_table());
        const auto pol_bs = ed::matvec::rep_policy_from(rd);

        for (int probe = 0; probe < 3; ++probe) {
            std::vector<Complex> x =
                (probe == 0)
                    ? std::vector<Complex>(sd, Complex(1.0, 0.0))
                    : random_unit_vector(sd, (k + 11) * 2654435761ULL + probe + N);

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
            INFO("GATHER==SCATTER sector " << k << " probe " << probe
                 << " diff " << gs_diff << " scale " << scale);
            REQUIRE(gs_diff < 1e-11 * (1.0 + scale));

            // O(1) rank-table path must equal the binary-search GATHER exactly.
            ed::symmetry::RepSectorData rd_tab =
                zn_sector(tab, N, n_up, k);
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
            INFO("O(1) vs O(log) GATHER sector " << k << " diff " << tab_diff);
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
            INFO("rep-CSR vs GATHER sector " << k << " diff " << csr_diff);
            REQUIRE(csr_diff < 1e-12 * (1.0 + scale));

            // Stage 4 (SymmetryEngine v2): the two-level shared-rank lookup
            // (one dense table per (N, n_up) + per-sector local remap) must
            // reproduce the binary-search GATHER exactly (same lookup result
            // -> identical arithmetic).
            ed::symmetry::RepSectorData rd_two =
                zn_sector(tab, N, n_up, k);
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
            INFO("two-level vs binary-search GATHER sector " << k
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
    auto full_op = build_heisenberg_pbc_full(static_cast<std::uint64_t>(N), 1.0);
    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, full_op->transform_data_, full_op->three_body_data_,
        [](const Complex& c) { return c; });

    // Largest sector (k=0).
    const ed::symmetry::OrbitTable tab = zn_orbit_table(N, n_up);
    ed::symmetry::RepSectorData rd = zn_sector(tab, N, n_up, 0);
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

TEST_CASE("rep_symmetry_backend: CPU rep matvec matches the full-space reference "
          "(N=6, n_up=3)",
          "[symmetry][matvec_backend][rep][N6]")
{
    run_case(6, 3);
}

TEST_CASE("rep_symmetry_backend: CPU rep matvec matches the full-space reference "
          "(N=8, n_up=4 and n_up=3)",
          "[symmetry][matvec_backend][rep][N8]")
{
    run_case(8, 4);
    run_case(8, 3);
}

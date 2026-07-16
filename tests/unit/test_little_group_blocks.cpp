// =============================================================================
// tests/unit/test_little_group_blocks.cpp
//
// U1a of the lane-unification series: the little-group engine's block-set
// factory (include/ed/solvers/little_group_blocks.h) must hand out OWNED
// block operators whose structure and spectra are identical to what
// run_little_group solves internally:
//
//   1. isotypic tiling per projected star:
//        sum_blocks dim * d_sigma * (tr_folded ? 2 : 1) == dim_k0;
//   2. global covering: sum_blocks dim * multiplicity == C(N, n_up)
//      (the factory's own meta.total_dim agrees);
//   3. spectra: dense-solving every block.op() and expanding by
//      multiplicity reproduces little_group_full_spectrum().expanded()
//      exactly -- the blocks ARE the decomposition, not a relabeling;
//   4. handle invariants: op().dim() == tag.dim, projected() <=> irrep >= 0,
//      rep_data().reps.size() == the star's dim_k0.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/core/linear_operator.h>
#include <ed/core/operator.h>
#include <ed/solvers/little_group_blocks.h>
#include <ed/solvers/little_group_solve.h>

#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

using Cx = std::complex<double>;

namespace {

std::unique_ptr<Operator> heisenberg_ring(std::uint64_t N, double J) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const Cx J_real(J, 0.0), J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        op->transform_data_.push_back(t);
        t.op_type = 0; t.op_type_2 = 1; t.coefficient = J_half;
        op->transform_data_.push_back(t);
        t.op_type = 1; t.op_type_2 = 0;
        op->transform_data_.push_back(t);
    }
    return op;
}

// Closed translation group + the N reflections of the D_N ring.
std::vector<std::vector<int>> ring_translations(int N) {
    std::vector<std::vector<int>> A;
    for (int s = 0; s < N; ++s) {
        std::vector<int> p(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) p[static_cast<std::size_t>(i)] = (i + s) % N;
        A.push_back(std::move(p));
    }
    return A;
}

std::vector<std::vector<int>> ring_reflections(int N) {
    std::vector<std::vector<int>> R;
    for (int s = 0; s < N; ++s) {
        std::vector<int> p(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i)
            p[static_cast<std::size_t>(i)] = ((s - i) % N + N) % N;
        R.push_back(std::move(p));
    }
    return R;
}

std::vector<double> dense_eigs(ed::LinearOperator& op) {
    const std::size_t d = op.dim();
    Eigen::MatrixXcd H(static_cast<Eigen::Index>(d),
                       static_cast<Eigen::Index>(d));
    std::vector<Cx> e(d, Cx(0, 0)), col(d);
    for (std::size_t j = 0; j < d; ++j) {
        std::fill(e.begin(), e.end(), Cx(0, 0));
        e[j] = Cx(1, 0);
        op.apply(e.data(), col.data(), d);
        for (std::size_t i = 0; i < d; ++i)
            H(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))
                = col[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    std::vector<double> ev(es.eigenvalues().data(),
                           es.eigenvalues().data() + d);
    std::sort(ev.begin(), ev.end());
    return ev;
}

}  // namespace

TEST_CASE("little-group block factory: tiling, covering, spectra",
          "[little_group][blocks]") {
    const int N = 8, n_up = 4;
    auto H = heisenberg_ring(static_cast<std::uint64_t>(N), 1.0);
    const auto A   = ring_translations(N);
    const auto res = ring_reflections(N);

    ed::solvers::LittleGroupOptions opt;
    opt.n_up = n_up;

    const auto set = ed::solvers::build_little_group_blocks(
        *H, A, res, N, opt);
    REQUIRE(!set.blocks.empty());
    REQUIRE(set.meta.flip_engaged);   // Heisenberg at half filling

    // ---- 4. handle invariants -------------------------------------------
    std::map<int, std::uint64_t> dim_k0_of_star;
    for (const auto& s : set.meta.stars) dim_k0_of_star[s.k0] = s.dim_k0;
    bool any_projected = false;
    for (const auto& b : set.blocks) {
        const auto& t = b.tag();
        REQUIRE(b.op().dim() == t.dim);
        REQUIRE(b.projected() == (t.irrep >= 0));
        REQUIRE(b.rep_data().reps.size() == dim_k0_of_star.at(t.k0));
        any_projected |= b.projected();
    }
    REQUIRE(any_projected);   // the D8 ring projects its self-conjugate momenta

    // ---- 1. isotypic tiling per projected star --------------------------
    for (const auto& s : set.meta.stars) {
        if (!s.projected) continue;
        std::uint64_t covered = 0;
        for (const auto& b : set.blocks) {
            if (b.tag().k0 != s.k0) continue;
            covered += b.tag().dim
                     * static_cast<std::uint64_t>(b.tag().irrep_dim)
                     * (b.tag().tr_folded ? 2u : 1u);
        }
        REQUIRE(covered == s.dim_k0);
        // strict reduction: no single block as large as the sector
        for (const auto& b : set.blocks)
            if (b.tag().k0 == s.k0) REQUIRE(b.tag().dim < s.dim_k0);
    }

    // ---- 2. global covering ---------------------------------------------
    std::uint64_t states = 0;
    for (const auto& b : set.blocks)
        states += b.tag().dim * b.tag().multiplicity;
    REQUIRE(states == 70);   // C(8,4)
    REQUIRE(set.meta.total_dim == 70);

    // ---- 3. block spectra reproduce the engine output -------------------
    std::vector<double> from_blocks;
    for (const auto& b : set.blocks) {
        const auto ev = dense_eigs(b.op());
        for (double e : ev)
            for (std::uint64_t r = 0; r < b.tag().multiplicity; ++r)
                from_blocks.push_back(e);
    }
    std::sort(from_blocks.begin(), from_blocks.end());

    const auto spec = ed::solvers::little_group_full_spectrum(
        *H, A, res, N, opt);
    const auto ref = spec.expanded();
    REQUIRE(from_blocks.size() == ref.size());
    ed_tests::require_eigs_close(from_blocks, ref, ref.size(), 1e-9,
                                 "block-set vs engine spectrum");
}

TEST_CASE("U2a: lift_to_rep turns block eigenvectors into H_k0 eigenvectors",
          "[little_group][blocks][lift]") {
    const int N = 8, n_up = 4;
    auto H = heisenberg_ring(static_cast<std::uint64_t>(N), 1.0);
    const auto A   = ring_translations(N);
    const auto res = ring_reflections(N);

    ed::solvers::LittleGroupOptions opt;
    opt.n_up      = n_up;
    opt.spin_flip = 0;   // raw sectors: the GS-DSSF contract (9d v1)

    const auto set = ed::solvers::build_little_group_blocks(
        *H, A, res, N, opt);
    bool tested_projected = false;
    for (const auto& b : set.blocks) {
        if (!b.projected()) continue;
        tested_projected = true;
        // Dense GS of the isotypic block...
        const std::size_t d = b.op().dim();
        Eigen::MatrixXcd Hb(static_cast<Eigen::Index>(d),
                            static_cast<Eigen::Index>(d));
        std::vector<Cx> e(d, Cx(0, 0)), col(d);
        for (std::size_t j = 0; j < d; ++j) {
            std::fill(e.begin(), e.end(), Cx(0, 0));
            e[j] = Cx(1, 0);
            b.op().apply(e.data(), col.data(), d);
            for (std::size_t i = 0; i < d; ++i)
                Hb(static_cast<Eigen::Index>(i),
                   static_cast<Eigen::Index>(j)) = col[i];
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hb);
        const double E0 = es.eigenvalues()(0);
        std::vector<Cx> v(d);
        for (std::size_t i = 0; i < d; ++i)
            v[i] = es.eigenvectors()(static_cast<Eigen::Index>(i), 0);
        // ...lifted must be an eigenvector of the FULL momentum-sector
        // H_k0 at the same eigenvalue (norm-preserving: W orthonormal).
        auto u = b.lift_to_rep(v.data());
        const std::size_t n = u.size();
        REQUIRE(n == b.rep_data().reps.size());
        // Norm preservation (W's columns are orthonormal)...
        double norm2 = 0.0;
        for (const auto& c : u) norm2 += std::norm(c);
        REQUIRE(std::abs(norm2 - 1.0) < 1e-10);
        // ...and the REAL claim: u is an eigenvector of the FULL
        // momentum-sector H_k0 at the same eigenvalue -- the same
        // residual the engine's GS guard enforces.
        auto hk = ed::solvers::make_rep_sector_matvec(*H, b.rep_data());
        std::vector<Cx> hu(n);
        hk->apply(u.data(), hu.data(), n);
        double num = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            num += std::norm(hu[i] - E0 * u[i]);
        REQUIRE(std::sqrt(num) < 1e-9);
        break;
    }
    REQUIRE(tested_projected);
}

namespace {

// Dense N-site Heisenberg ring in the full 2^N computational basis.
Eigen::MatrixXd dense_heisenberg(int N) {
    const std::size_t dim = std::size_t{1} << N;
    Eigen::MatrixXd Hd = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
    auto sz = [](std::uint64_t s, int i) {
        return ((s >> i) & 1u) ? 0.5 : -0.5;
    };
    for (std::uint64_t s = 0; s < dim; ++s) {
        for (int i = 0; i < N; ++i) {
            const int j = (i + 1) % N;
            Hd(static_cast<Eigen::Index>(s), static_cast<Eigen::Index>(s))
                += sz(s, i) * sz(s, j);
            if (((s >> i) & 1u) != ((s >> j) & 1u)) {
                const std::uint64_t t = s ^ (1ULL << i) ^ (1ULL << j);
                Hd(static_cast<Eigen::Index>(t),
                   static_cast<Eigen::Index>(s)) += 0.5;
            }
        }
    }
    return Hd;
}

}  // namespace

TEST_CASE("U2b: flip-projected eigenvectors expand to the computational "
          "basis (the orbit-CSR lane's refusal, done arithmetically)",
          "[little_group][blocks][flip][expand]") {
    const int N = 8, n_up = 4;
    auto H = heisenberg_ring(static_cast<std::uint64_t>(N), 1.0);
    const auto A   = ring_translations(N);
    const auto res = ring_reflections(N);

    ed::solvers::LittleGroupOptions opt;
    opt.n_up = n_up;   // spin_flip auto: (k, +/-) blocks engage at N/2

    const auto set = ed::solvers::build_little_group_blocks(
        *H, A, res, N, opt);
    REQUIRE(set.meta.flip_engaged);
    const auto Hd = dense_heisenberg(N);
    const std::uint64_t mask = (1ULL << N) - 1;

    int tested = 0;
    for (const auto& b : set.blocks) {
        if (b.tag().flip_parity < 0) continue;
        REQUIRE(b.rep_data().has_flips());   // genuinely flip-extended
        // Dense lowest eigenpair of this block...
        const std::size_t d = b.op().dim();
        Eigen::MatrixXcd Hb(static_cast<Eigen::Index>(d),
                            static_cast<Eigen::Index>(d));
        std::vector<Cx> e(d, Cx(0, 0)), col(d);
        for (std::size_t j = 0; j < d; ++j) {
            std::fill(e.begin(), e.end(), Cx(0, 0));
            e[j] = Cx(1, 0);
            b.op().apply(e.data(), col.data(), d);
            for (std::size_t i = 0; i < d; ++i)
                Hb(static_cast<Eigen::Index>(i),
                   static_cast<Eigen::Index>(j)) = col[i];
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hb);
        const double E0 = es.eigenvalues()(0);
        std::vector<Cx> v(d);
        for (std::size_t i = 0; i < d; ++i)
            v[i] = es.eigenvectors()(static_cast<Eigen::Index>(i), 0);
        // ...lift to the (k, +/-) rep basis, expand to 2^N amplitudes.
        const auto u   = b.lift_to_rep(v.data());
        const auto psi = ed::solvers::expand_rep_vector_to_computational(
            b.rep_data(), u);
        // (a) eigenvector of the FULL dense H at the block eigenvalue;
        double num = 0.0;
        Eigen::VectorXcd p(static_cast<Eigen::Index>(psi.size()));
        for (std::size_t i = 0; i < psi.size(); ++i)
            p(static_cast<Eigen::Index>(i)) = psi[i];
        Eigen::VectorXcd hp = Hd * p;
        for (Eigen::Index i = 0; i < p.size(); ++i)
            num += std::norm(hp(i) - E0 * p(i));
        REQUIRE(std::sqrt(num) < 1e-9);
        // (b) eigenvector of the FLIP at the block's labelled parity;
        const double want = (b.tag().flip_parity == 0) ? 1.0 : -1.0;
        double fnum = 0.0;
        for (std::uint64_t s = 0; s < psi.size(); ++s)
            fnum += std::norm(psi[s ^ mask] - want * psi[s]);
        REQUIRE(std::sqrt(fnum) < 1e-9);
        // (c) supported only on n_up states.
        for (std::uint64_t s = 0; s < psi.size(); ++s)
            if (std::abs(psi[s]) > 1e-12)
                REQUIRE(__builtin_popcountll(s) == n_up);
        if (++tested >= 4) break;   // a few blocks of each parity suffice
    }
    REQUIRE(tested >= 4);
}

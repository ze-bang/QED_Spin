// =============================================================================
// tests/unit/test_group_sector.cpp
//
// The group-sector lane (include/ed/solvers/group_sector.h) and its dispatch inside the little-group block factory:
//
//   1. build_group_sector + solve_group_sector reproduce the spectrum of H restricted to the range of the projector
//      P = |G|^-1 sum_g chi(g)^* U(g), built independently in the full Sz sector -- for the dihedral group of a
//      J1-J2 ring (real characters, with and without the spin flip) and for its translations alone at a complex
//      momentum;
//   2. convert_group_vector (D_N -> C_N) keeps the norm and delivers an eigenvector of the subgroup sector;
//   3. ED_SYM_LG_GROUP_SECTOR = 1 (the default) and = 0 (the isotypic W path) hand out the same blocks: same keys,
//      dimensions, multiplicities and dense spectra, and lift_to_rep gives the same H_k0 eigenvector up to a phase.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/core/linear_operator.h>
#include <ed/core/operator.h>
#include <ed/solvers/group_sector.h>
#include <ed/solvers/little_group_blocks.h>
#include <ed/solvers/little_group_solve.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

using Cx = std::complex<double>;

namespace {

constexpr double kPi = 3.14159265358979323846;

void add_heisenberg_bond(Operator& op, std::uint64_t i, std::uint64_t j, double J) {
    Operator::TransformData t;
    t.op_type = 2; t.site_index = i; t.op_type_2 = 2; t.site_index_2 = j;
    t.coefficient = Cx(J, 0.0); t.is_two_body = true;
    op.transform_data_.push_back(t);
    t.op_type = 0; t.op_type_2 = 1; t.coefficient = Cx(0.5 * J, 0.0);
    op.transform_data_.push_back(t);
    t.op_type = 1; t.op_type_2 = 0;
    op.transform_data_.push_back(t);
}

// J1-J2 ring: J2 lifts the accidental degeneracies of the NN ring, so eigenvectors are unique up to a phase.
std::unique_ptr<Operator> j1j2_ring(int N, double J1, double J2) {
    auto op = std::make_unique<Operator>(static_cast<std::uint64_t>(N), 0.5f);
    for (int i = 0; i < N; ++i) {
        add_heisenberg_bond(*op, i, (i + 1) % N, J1);
        add_heisenberg_bond(*op, i, (i + 2) % N, J2);
    }
    return op;
}

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
        for (int i = 0; i < N; ++i) p[static_cast<std::size_t>(i)] = ((s - i) % N + N) % N;
        R.push_back(std::move(p));
    }
    return R;
}

// ---- independent reference: the full Sz sector ----------------------------------------------------------------------
struct SzSector {
    int N;
    std::vector<std::uint64_t> states;
    std::map<std::uint64_t, std::size_t> index;
    SzSector(int n, int n_up) : N(n) {
        for (std::uint64_t s = 0; s < (1ULL << n); ++s)
            if (__builtin_popcountll(s) == n_up) { index[s] = states.size(); states.push_back(s); }
    }
    [[nodiscard]] Eigen::Index dim() const { return static_cast<Eigen::Index>(states.size()); }
};

Eigen::MatrixXcd dense_ring(const SzSector& S, double J1, double J2) {
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(S.dim(), S.dim());
    auto bond = [&](std::uint64_t s, std::size_t a, int i, int j, double J) {
        const bool bi = (s >> i) & 1u, bj = (s >> j) & 1u;
        H(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(a)) += J * (bi == bj ? 0.25 : -0.25);
        if (bi != bj) {
            const std::size_t b = S.index.at(s ^ (1ULL << i) ^ (1ULL << j));
            H(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) += 0.5 * J;
        }
    };
    for (std::size_t a = 0; a < S.states.size(); ++a)
        for (int i = 0; i < S.N; ++i) {
            bond(S.states[a], a, i, (i + 1) % S.N, J1);
            bond(S.states[a], a, i, (i + 2) % S.N, J2);
        }
    return H;
}

// U(p, flip): site i -> p[i], then optionally the global spin flip.
Eigen::MatrixXcd site_op(const SzSector& S, const std::vector<int>& p, bool flip) {
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(S.dim(), S.dim());
    const std::uint64_t all = (1ULL << S.N) - 1ULL;
    for (std::size_t a = 0; a < S.states.size(); ++a) {
        std::uint64_t t = 0;
        for (int i = 0; i < S.N; ++i)
            if ((S.states[a] >> i) & 1u) t |= 1ULL << p[static_cast<std::size_t>(i)];
        if (flip) t ^= all;
        U(static_cast<Eigen::Index>(S.index.at(t)), static_cast<Eigen::Index>(a)) = 1.0;
    }
    return U;
}

// Spectrum of H on range(P), P = |G|^-1 sum_g conj(chi(g)) U(g) with the group-sector element layout.
std::vector<double> projected_spectrum(const SzSector& S, const Eigen::MatrixXcd& H,
                                       const std::vector<std::vector<int>>& perms, bool flip,
                                       const std::vector<Cx>& chi) {
    Eigen::MatrixXcd P = Eigen::MatrixXcd::Zero(S.dim(), S.dim());
    const std::size_t G = perms.size();
    for (std::size_t f = 0; f < (flip ? 2u : 1u); ++f)
        for (std::size_t g = 0; g < G; ++g) P += std::conj(chi[f * G + g]) * site_op(S, perms[g], f == 1);
    P /= static_cast<double>(flip ? 2 * G : G);
    REQUIRE((P * P - P).norm() < 1e-10);            // chi really is a 1-dim representation of G
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ps(0.5 * (P + P.adjoint()));
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = 0; i < S.dim(); ++i) if (ps.eigenvalues()(i) > 0.5) keep.push_back(i);
    Eigen::MatrixXcd Q(S.dim(), static_cast<Eigen::Index>(keep.size()));
    for (std::size_t c = 0; c < keep.size(); ++c) Q.col(static_cast<Eigen::Index>(c)) = ps.eigenvectors().col(keep[c]);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Q.adjoint() * H * Q);
    return {es.eigenvalues().data(), es.eigenvalues().data() + es.eigenvalues().size()};
}

std::vector<double> group_spectrum(const Operator& op, const std::vector<std::vector<int>>& perms, int N, int n_up,
                                   bool flip, const std::vector<Cx>& chi) {
    auto rd = std::make_shared<const ed::symmetry::RepSectorData>(
        ed::solvers::build_group_sector(perms, N, n_up, flip, chi));
    ed::solvers::GroupSectorSolveOptions o;
    o.levels = static_cast<int>(rd->reps.size());
    o.dense_max_dim = 1 << 20;
    o.return_vectors = false;
    const auto r = ed::solvers::solve_group_sector(op, rd, o);
    REQUIRE(r.dim == rd->reps.size());
    return r.energies;
}

void require_same_spectrum(const std::vector<double>& a, const std::vector<double>& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) REQUIRE(std::abs(a[i] - b[i]) < 1e-10);
}

Eigen::MatrixXcd materialize(ed::LinearOperator& op) {
    const std::size_t d = op.dim();
    Eigen::MatrixXcd H(static_cast<Eigen::Index>(d), static_cast<Eigen::Index>(d));
    std::vector<Cx> e(d), col(d);
    for (std::size_t j = 0; j < d; ++j) {
        std::fill(e.begin(), e.end(), Cx(0, 0)); e[j] = 1.0;
        op.apply(e.data(), col.data(), d);
        for (std::size_t i = 0; i < d; ++i) H(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = col[i];
    }
    return H;
}

struct EnvFlag {
    explicit EnvFlag(const char* v) { setenv("ED_SYM_LG_GROUP_SECTOR", v, 1); }
    ~EnvFlag() { unsetenv("ED_SYM_LG_GROUP_SECTOR"); }
};

}  // namespace

TEST_CASE("group sector: spectra equal H on the range of the full-space projector", "[group_sector]") {
    const int N = 10, n_up = 5;
    const double J1 = 1.0, J2 = 0.37;
    auto H = j1j2_ring(N, J1, J2);
    const SzSector S(N, n_up);
    const Eigen::MatrixXcd Hd = dense_ring(S, J1, J2);
    const auto T = ring_translations(N);
    const auto R = ring_reflections(N);

    SECTION("dihedral group, real characters, spin flip on and off") {
        std::vector<std::vector<int>> D = T;
        D.insert(D.end(), R.begin(), R.end());
        for (int k : {0, N / 2})                 // the two momenta whose little co-group is the whole D_N
            for (int refl : {+1, -1})
                for (int fl : {0, +1, -1}) {
                    std::vector<Cx> chi;
                    for (int s = 0; s < N; ++s) chi.emplace_back(std::cos(2 * kPi * k * s / N), 0.0);
                    for (int s = 0; s < N; ++s)          // reflection i -> s - i = translation(s) * (i -> -i)
                        chi.emplace_back(refl * std::cos(2 * kPi * k * s / N), 0.0);
                    if (fl != 0) { const auto half = chi; for (const auto& c : half) chi.push_back(double(fl) * c); }
                    INFO("k=" << k << " refl=" << refl << " flip=" << fl);
                    require_same_spectrum(group_spectrum(*H, D, N, n_up, fl != 0, chi),
                                          projected_spectrum(S, Hd, D, fl != 0, chi));
                }
    }
    SECTION("translations alone, complex momentum") {
        for (int k : {1, 3})
            for (int fl : {+1, -1}) {
                std::vector<Cx> chi;
                for (int s = 0; s < N; ++s) chi.push_back(std::polar(1.0, 2 * kPi * k * s / N));
                const auto half = chi; for (const auto& c : half) chi.push_back(double(fl) * c);
                INFO("k=" << k << " flip=" << fl);
                require_same_spectrum(group_spectrum(*H, T, N, n_up, true, chi),
                                      projected_spectrum(S, Hd, T, true, chi));
            }
    }
}

TEST_CASE("group sector: convert_group_vector D_N -> C_N keeps the state", "[group_sector]") {
    const int N = 10, n_up = 5;
    auto H = j1j2_ring(N, 1.0, 0.37);
    const auto T = ring_translations(N);
    const auto R = ring_reflections(N);
    std::vector<std::vector<int>> D = T;
    D.insert(D.end(), R.begin(), R.end());
    for (int refl : {+1, -1}) {
        std::vector<Cx> chiD(2 * D.size(), Cx(1, 0)), chiT(2 * T.size(), Cx(1, 0));
        for (std::size_t g = T.size(); g < D.size(); ++g) chiD[g] = chiD[D.size() + g] = double(refl);
        auto src = std::make_shared<const ed::symmetry::RepSectorData>(
            ed::solvers::build_group_sector(D, N, n_up, true, chiD));
        const auto dst = ed::solvers::build_group_sector(T, N, n_up, true, chiT);
        ed::solvers::GroupSectorSolveOptions o;
        o.levels = 1;
        const auto r = ed::solvers::solve_group_sector(*H, src, o);
        REQUIRE(r.residuals.at(0) < 1e-9);
        const auto w = ed::solvers::convert_group_vector(r.vectors.at(0), *src, dst, true);
        double n2 = 0.0;
        for (const auto& c : w) n2 += std::norm(c);
        REQUIRE(std::abs(n2 - 1.0) < 1e-10);
        auto hk = ed::solvers::make_rep_sector_matvec(*H, dst);
        std::vector<Cx> hw(w.size());
        hk->apply(w.data(), hw.data(), w.size());
        double res = 0.0;
        for (std::size_t i = 0; i < w.size(); ++i) res += std::norm(hw[i] - r.energies[0] * w[i]);
        REQUIRE(std::sqrt(res) < 1e-9);
    }
}

TEST_CASE("group sector: the block factory's dispatch equals the isotypic W path", "[group_sector][little_group]") {
    const int N = 12, n_up = 6;
    auto H = j1j2_ring(N, 1.0, 0.37);
    const auto A = ring_translations(N);
    const auto R = ring_reflections(N);
    ed::solvers::LittleGroupOptions opt;
    opt.n_up = n_up;

    using Key = std::tuple<int, int>;   // (k0, irrep)
    std::map<Key, std::tuple<std::uint64_t, std::uint64_t, std::vector<double>, std::vector<Cx>, int>> ref;
    {
        EnvFlag off("0");
        const auto set = ed::solvers::build_little_group_blocks(*H, A, R, N, opt);
        for (const auto& b : set.blocks) {
            if (!b.projected()) continue;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(b.op()));
            std::vector<Cx> v0(es.eigenvectors().col(0).data(),
                               es.eigenvectors().col(0).data() + es.eigenvectors().rows());
            ref[{b.tag().k0, b.tag().irrep}] = {
                b.tag().dim, b.tag().multiplicity,
                {es.eigenvalues().data(), es.eigenvalues().data() + es.eigenvalues().size()},
                b.lift_to_rep(v0.data()), b.tag().irrep_dim};
        }
    }
    REQUIRE(!ref.empty());

    EnvFlag on("1");
    const auto set = ed::solvers::build_little_group_blocks(*H, A, R, N, opt);
    std::size_t seen = 0;
    for (const auto& b : set.blocks) {
        if (!b.projected()) continue;
        const Key key{b.tag().k0, b.tag().irrep};
        INFO("k0=" << b.tag().k0 << " irrep=" << b.tag().irrep);
        REQUIRE(ref.count(key) == 1);
        const auto& [dim, mult, ev_ref, u_ref, d] = ref.at(key);
        REQUIRE(b.tag().dim == dim);
        REQUIRE(b.tag().multiplicity == mult);
        REQUIRE(b.op().dim() == b.tag().dim);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(b.op()));
        const std::vector<double> ev(es.eigenvalues().data(), es.eigenvalues().data() + es.eigenvalues().size());
        require_same_spectrum(ev, ev_ref);
        if (d == 1 && ev.size() > 1 && ev[1] - ev[0] > 1e-8) {   // nondegenerate: the lifted states agree
            std::vector<Cx> v0(es.eigenvectors().col(0).data(),
                               es.eigenvectors().col(0).data() + es.eigenvectors().rows());
            const auto u = b.lift_to_rep(v0.data());
            REQUIRE(u.size() == u_ref.size());
            Cx ov(0, 0);
            for (std::size_t i = 0; i < u.size(); ++i) ov += std::conj(u_ref[i]) * u[i];
            REQUIRE(std::abs(std::abs(ov) - 1.0) < 1e-10);
        }
        ++seen;
    }
    REQUIRE(seen == ref.size());
}

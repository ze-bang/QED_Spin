// =============================================================================
// tests/unit/test_invariance.cpp
//
// The symmetry verdicts of include/ed/ops/invariance.h on a model zoo, against the dense
// truth and against the record-level detectors they replace (commute_check.h, spin_flip.h,
// su2.h, time_reversal.h, ed::sectors::sz_content):
//   1. masked(op) is the operator the kernels apply: its dense matrix equals Operator::apply
//      on every basis state (every record shape, same-site products included);
//   2. every new verdict equals the dense truth, and is unchanged when H is scaled by 1e-6
//      or 1e6;
//   3. every old verdict equals the new one, except where the records misrepresent the
//      operator -- cancelling S+S+ / S-S- records (the builder's D_z DM term, the Cartesian
//      form of the Heisenberg bond) and same-site S_i.S_i records -- listed per model.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/input/hamiltonian_builder.h>
#include <ed/input/lattice.h>
#include <ed/ops/commute_check.h>
#include <ed/ops/invariance.h>
#include <ed/ops/operator.h>
#include <ed/ops/spin_flip.h>
#include <ed/ops/su2.h>
#include <ed/ops/time_reversal.h>
#include <ed/sectors/sectors.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <tuple>
#include <string>
#include <vector>

using Cx = std::complex<double>;
using ed::ops::MaskedOperator;

namespace {

constexpr int N = 8;
constexpr std::uint64_t kDim = 1ULL << N;

struct Model {
    std::string name;
    std::shared_ptr<Operator> H;
    std::set<std::string> old_differs;   // verdicts the record-level detectors get wrong here
};

std::vector<std::pair<std::size_t, std::size_t>> ring() {
    return ed::input::lattice::chain(N, true).nn_pairs();
}

std::shared_ptr<Operator> records() { return std::make_shared<Operator>(N, 0.5f); }

// J S_i.S_j in Cartesian form: Sx Sx and Sy Sy each as four ladder records (their S+S+ and
// S-S- parts cancel), then Sz Sz.
void cartesian_bond(Operator& H, std::uint64_t i, std::uint64_t j, double J) {
    const double q = J / 4.0;
    const std::tuple<int, int, double> recs[] = {{0, 0, q}, {0, 1, q}, {1, 0, q}, {1, 1, q},     // Sx Sx
                                                 {0, 0, -q}, {0, 1, q}, {1, 0, q}, {1, 1, -q}};  // Sy Sy
    for (const auto& [a, b, c] : recs)
        H.addTwoBodyTerm(static_cast<std::uint8_t>(a), i, static_cast<std::uint8_t>(b), j, Cx(c, 0.0));
    H.addTwoBodyTerm(2, i, 2, j, Cx(J, 0.0));
}

std::vector<Model> zoo() {
    using ed::input::HamiltonianBuilder;
    std::vector<Model> z;
    const auto bonds = ring();
    auto built = [&](const std::string& name, auto&& fill, std::set<std::string> differs = {}) {
        HamiltonianBuilder b(N);
        fill(b);
        z.push_back({name, b.to_operator(), std::move(differs)});
    };
    built("heisenberg", [&](HamiltonianBuilder& b) { b.heisenberg(bonds, 1.0); });
    built("xxz", [&](HamiltonianBuilder& b) { b.xxz(bonds, 1.0, 0.5); });
    built("xyz", [&](HamiltonianBuilder& b) { b.xyz(bonds, 1.0, 0.7, 0.4); });
    built("heisenberg+zeeman_z", [&](HamiltonianBuilder& b) { b.heisenberg(bonds, 1.0).zeeman({0.0, 0.0, 0.3}); });
    built("heisenberg+zeeman_x", [&](HamiltonianBuilder& b) { b.heisenberg(bonds, 1.0).zeeman({0.3, 0.0, 0.0}); });
    built("heisenberg+zeeman_y", [&](HamiltonianBuilder& b) { b.heisenberg(bonds, 1.0).zeeman({0.0, 0.3, 0.0}); });
    built("tfim", [&](HamiltonianBuilder& b) { b.transverse_field_ising(bonds, 1.0, 0.7); });
    built("heisenberg+dm_x", [&](HamiltonianBuilder& b) {
        b.heisenberg(bonds, 1.0).dm(bonds, std::vector<std::array<double, 3>>(bonds.size(), {0.3, 0.0, 0.0}));
    });
    // D_z conserves S^z, but the builder writes it with S+S+ / S-S- records that cancel.
    built("heisenberg+dm_z", [&](HamiltonianBuilder& b) {
        b.heisenberg(bonds, 1.0).dm(bonds, std::vector<std::array<double, 3>>(bonds.size(), {0.0, 0.0, 0.3}));
    }, {"sz"});
    {
        const auto hc = ed::input::lattice::honeycomb(2, 2, true);
        std::vector<std::pair<std::size_t, std::size_t>> hb;
        std::vector<int> axis;
        for (const auto& bd : hc.nn_bonds) { hb.emplace_back(bd.i, bd.j); axis.push_back(bd.bond_type % 3); }
        built("kitaev_honeycomb", [&](HamiltonianBuilder& b) { b.kitaev(hb, axis, 1.0); });
    }
    {   // the Heisenberg ring in Cartesian form: U(1) and SU(2), which the records hide
        auto H = records();
        for (const auto& [i, j] : bonds) cartesian_bond(*H, i, j, 1.0);
        z.push_back({"heisenberg_cartesian", H, {"sz", "su2"}});
    }
    {   // S_tot^2 as the full double sum, i == j included (a constant 3/4 per site)
        auto H = records();
        for (std::uint64_t i = 0; i < N; ++i)
            for (std::uint64_t j = 0; j < N; ++j) {
                H->addTwoBodyTerm(0, i, 1, j, Cx(0.5, 0.0));
                H->addTwoBodyTerm(1, i, 0, j, Cx(0.5, 0.0));
                H->addTwoBodyTerm(2, i, 2, j, Cx(1.0, 0.0));
            }
        z.push_back({"s_tot_squared", H, {"su2", "flip"}});
    }
    {   // Heisenberg plus the scalar chirality S_a.(S_b x S_c) on consecutive triples
        HamiltonianBuilder b(N);
        b.heisenberg(bonds, 1.0);
        auto H = b.to_operator();
        const int pattern[6][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}, {1, 0, 2}, {2, 1, 0}, {0, 2, 1}};
        for (std::uint64_t a = 0; a < N; ++a) {
            const std::uint64_t s[3] = {a, (a + 1) % N, (a + 2) % N};
            for (int p = 0; p < 6; ++p)
                H->addThreeBodyTerm(static_cast<std::uint8_t>(pattern[p][0]), s[0],
                                    static_cast<std::uint8_t>(pattern[p][1]), s[1],
                                    static_cast<std::uint8_t>(pattern[p][2]), s[2],
                                    Cx(0.0, p < 3 ? 0.2 : -0.2));
        }
        z.push_back({"heisenberg+chirality", H, {}});
    }
    {   // R + R^dagger for random records R of every shape, same-site two-body products included;
        // the adjoint records are written out (S+ <-> S-, coefficient conjugated, factors reversed)
        auto H = records();
        std::mt19937 rng(20261001);
        std::uniform_int_distribution<int> op(0, 2), site(0, N - 1);
        std::normal_distribution<double> g(0.0, 1.0);
        auto adj = [](int o) { return static_cast<std::uint8_t>(o == 2 ? 2 : 1 - o); };
        auto one = [&](int o, std::uint64_t s, Cx c) {
            H->addOneBodyTerm(static_cast<std::uint8_t>(o), s, c);
            H->addOneBodyTerm(adj(o), s, std::conj(c));
        };
        auto two = [&](int a, std::uint64_t s, int b, std::uint64_t t, Cx c) {
            H->addTwoBodyTerm(static_cast<std::uint8_t>(a), s, static_cast<std::uint8_t>(b), t, c);
            H->addTwoBodyTerm(adj(b), t, adj(a), s, std::conj(c));
        };
        for (int k = 0; k < 12; ++k) one(op(rng), static_cast<std::uint64_t>(site(rng)), Cx(g(rng), g(rng)));
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                const auto s = static_cast<std::uint64_t>(site(rng));
                two(a, s, b, s, Cx(g(rng), g(rng)));
            }
        for (int k = 0; k < 16; ++k) {
            const int a = op(rng), b = op(rng);
            const auto s = static_cast<std::uint64_t>(site(rng)), t = static_cast<std::uint64_t>(site(rng));
            two(a, s, b, t, Cx(g(rng), g(rng)));
        }
        for (int k = 0; k < 6; ++k) {
            const auto s0 = static_cast<std::uint64_t>(k), s1 = (s0 + 3) % N, s2 = (s0 + 5) % N;
            const int a = op(rng), b = op(rng), c = op(rng);
            const Cx w(g(rng), g(rng));
            H->addThreeBodyTerm(static_cast<std::uint8_t>(a), s0, static_cast<std::uint8_t>(b), s1,
                                static_cast<std::uint8_t>(c), s2, w);
            H->addThreeBodyTerm(adj(a), s0, adj(b), s1, adj(c), s2, std::conj(w));
        }
        z.push_back({"random_records", H, {}});
    }
    return z;
}

std::vector<std::vector<int>> perms() {
    std::vector<std::vector<int>> p;
    auto make = [&](auto f) { std::vector<int> v(N); for (int i = 0; i < N; ++i) v[static_cast<std::size_t>(i)] = f(i); p.push_back(v); };
    make([](int i) { return i; });
    make([](int i) { return (i + 1) % N; });
    make([](int i) { return (i + 2) % N; });
    make([](int i) { return (N - i) % N; });
    make([](int i) { return i < 2 ? 1 - i : i; });
    return p;
}

// ---- the dense truth, from the kernels' own apply ---------------------------------------

std::vector<Cx> dense_apply(const Operator& op) {   // M[t * dim + s]
    std::vector<Cx> M(kDim * kDim), e(kDim), out(kDim);
    for (std::uint64_t s = 0; s < kDim; ++s) {
        std::fill(e.begin(), e.end(), Cx(0.0, 0.0));
        e[s] = 1.0;
        op.apply(e.data(), out.data(), kDim);
        for (std::uint64_t t = 0; t < kDim; ++t) M[t * kDim + s] = out[t];
    }
    return M;
}

double max_abs(const std::vector<Cx>& M) {
    double m = 0.0;
    for (const auto& x : M) m = std::max(m, std::abs(x));
    return m;
}

bool up(std::uint64_t s, int i) { return (((s >> i) & 1ULL) != 0) != ed::ops::kSetBitIsDown; }

struct Truth { bool flip, real, su2; int sz; std::vector<bool> perm; };

Truth truth(const std::vector<Cx>& M) {
    const double tol = 1e-9 * max_abs(M);
    auto at = [&](std::uint64_t t, std::uint64_t s) { return M[t * kDim + s]; };
    Truth r{true, true, true, 0, {}};
    const std::uint64_t all = kDim - 1;
    bool u1 = true, parity = true;
    for (std::uint64_t t = 0; t < kDim; ++t)
        for (std::uint64_t s = 0; s < kDim; ++s) {
            const Cx m = at(t, s);
            if (std::abs(at(t ^ all, s ^ all) - m) > tol) r.flip = false;
            if (std::abs(m.imag()) > tol) r.real = false;
            if (std::abs(m) > tol) {
                const int d = __builtin_popcountll(t) - __builtin_popcountll(s);
                if (d != 0) u1 = false;
                if (d % 2 != 0) parity = false;
            }
            // [M, S^a_tot] for a = -, +, z, with S sparse
            double sz_t = 0.0, sz_s = 0.0;
            Cx cm(0.0, 0.0), cp(0.0, 0.0);
            for (int i = 0; i < N; ++i) {
                const std::uint64_t b = 1ULL << i;
                sz_t += up(t, i) ? 0.5 : -0.5;
                sz_s += up(s, i) ? 0.5 : -0.5;
                if (up(s, i)) cm += at(t, s ^ b); else cp += at(t, s ^ b);       // (M S)[t, s]
                if (!up(t, i)) cm -= at(t ^ b, s); else cp -= at(t ^ b, s);      // (S M)[t, s]
            }
            if (std::abs(cm) > tol || std::abs(cp) > tol || std::abs(m * (sz_s - sz_t)) > tol) r.su2 = false;
        }
    r.sz = u1 ? 0 : (parity ? 1 : 2);   // the order of SzContent { U1, Parity, None }
    for (const auto& p : perms()) {
        bool ok = true;
        for (std::uint64_t t = 0; t < kDim && ok; ++t)
            for (std::uint64_t s = 0; s < kDim; ++s) {
                const auto pt = ed::ops::permute_mask(t, p.data(), N), ps = ed::ops::permute_mask(s, p.data(), N);
                if (std::abs(at(pt, ps) - at(t, s)) > tol) { ok = false; break; }
            }
        r.perm.push_back(ok);
    }
    return r;
}

// ---- the verdicts --------------------------------------------------------------------

struct Verdicts { bool flip, real, su2; int sz; std::vector<bool> perm; };

Verdicts fresh(const MaskedOperator& H) {
    Verdicts v{ed::ops::flip_invariant(H), ed::ops::conjugation_invariant(H), ed::ops::su2_invariant(H),
               static_cast<int>(ed::ops::sz_content(H)), {}};
    for (const auto& p : perms()) v.perm.push_back(ed::ops::commutes_with_permutation(H, p));
    return v;
}

Verdicts old(const Operator& H) {
    H.commitPendingTransforms();
    const auto& soa = H.terms_;
    const auto sz = ed::sectors::sz_content(H);
    Verdicts v{ed::symmetry::hamiltonian_is_spin_flip_symmetric(soa), ed::symmetry::hamiltonian_is_real(soa),
               sz == ed::sectors::SzContent::U1 && ed::symmetry::hamiltonian_is_su2_symmetric(soa),
               static_cast<int>(sz), {}};
    for (const auto& p : perms())
        v.perm.push_back(ed::symmetry::hamiltonian_commutes_with_permutation(H.transform_data_, H.three_body_data_, p));
    return v;
}

}  // namespace

TEST_CASE("masked(op) is the operator the kernels apply", "[invariance]") {
    for (const auto& m : zoo()) {
        INFO("model " << m.name);
        const auto D = dense_apply(*m.H);
        const auto A = ed::ops::masked(*m.H).to_dense();
        double d = 0.0;
        for (std::size_t i = 0; i < D.size(); ++i) d = std::max(d, std::abs(D[i] - A[i]));
        REQUIRE(d <= 1e-13 * std::max(1.0, max_abs(D)));
    }
    auto bad = records();
    bad->addOneBodyTerm(3, 0, Cx(1.0, 0.0));
    CHECK_THROWS_AS(ed::ops::masked(*bad), std::invalid_argument);
    auto far = records();
    far->addTwoBodyTerm(2, 0, 2, N, Cx(1.0, 0.0));
    CHECK_THROWS_AS(ed::ops::masked(*far), std::invalid_argument);
    CHECK_THROWS_AS(ed::ops::commutes_with_permutation(ed::ops::masked(*records()), {0, 0, 1, 2, 3, 4, 5, 6}),
                    std::invalid_argument);
}

TEST_CASE("the verdicts are the dense truth, at any scale", "[invariance]") {
    for (const auto& m : zoo()) {
        INFO("model " << m.name);
        const auto t = truth(dense_apply(*m.H));
        const auto H = ed::ops::masked(*m.H);
        for (double scale : {1.0, 1e-6, 1e6}) {
            INFO("scale " << scale);
            const auto v = fresh(H.scaled(scale));
            CHECK(v.flip == t.flip);
            CHECK(v.real == t.real);
            CHECK(v.su2 == t.su2);
            CHECK(v.sz == t.sz);
            CHECK(v.perm == t.perm);
        }
        CHECK(ed::ops::hermitian(H) == H.equals(H.dagger(), ed::ops::kInvarianceRtol));
    }
}

TEST_CASE("the record-level detectors agree except where the records misrepresent H", "[invariance]") {
    for (const auto& m : zoo()) {
        INFO("model " << m.name);
        const auto n = fresh(ed::ops::masked(*m.H));
        const auto o = old(*m.H);
        auto expect = [&](const char* what, bool same) {
            INFO("verdict " << what);
            CHECK(same == (m.old_differs.count(what) == 0));
        };
        expect("flip", o.flip == n.flip);
        expect("real", o.real == n.real);
        expect("su2", o.su2 == n.su2);
        expect("sz", o.sz == n.sz);
        expect("perm", o.perm == n.perm);
    }
}

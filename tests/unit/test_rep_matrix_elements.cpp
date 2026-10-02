// =============================================================================
// tests/unit/test_rep_matrix_elements.cpp
//
// rep_matrix_elements() (include/ed/ops/program.h) against the full-space reference
// <E a| O |E b>, where E is the engine's own rep-basis expansion (ed::sectors::expand into
// the full space) and O acts term by term on the 2^N vector.
//
// Systems: a 12-site ring (Z_12 translations, complex characters, stabilisers of order
// 2, 3, 4, 6, 12) and a 4x4 torus (Z_4 x Z_4, stabilisers from period-2 patterns), each
// with and without the global spin flip, plus sectors built by the engine's k-sector
// stream. Cases:
//   1. same sector: random non-Hermitian operators (single terms, bonds, 3-site strings);
//   2. cross-momentum: a single bond projected onto the lambda component;
//   3. cross flip parity (k,+) <-> (k,-) with flip-odd operators;
//   4. cross S^z: S+ and S- strings between n_up and n_up -/+ 1;
//   5. identity = <a|b>, and exactly 0 between different sectors;
//   6. <a|T|b> == conj <b|T^dagger|a>;
//   7. many kets / bras / pairs in one call.
// =============================================================================
#include "common/catch2_harness.h"

#include "engine/options.h"   // little_group_k_sectors_stream

#include <ed/basis/rep_sector.h>
#include <ed/ops/operator.h>
#include <ed/ops/program.h>
#include <ed/sectors/sectors.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using Cx = std::complex<double>;
using ed::ops::compile_program;
using ed::ops::MaskedOperator;
using ed::ops::RepVectorView;
using ed::ops::rep_matrix_elements;
using ed::symmetry::RepSectorData;

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Element { std::vector<int> perm; std::uint64_t flip; std::vector<int> label; };

// Abelian group Z_L1 x Z_L2 (x Z_2 flip) acting on an L1 x L2 torus (L2 = 1: ring).
std::vector<Element> torus_group(int L1, int L2, bool with_flip) {
    const int n = L1 * L2;
    const std::uint64_t all = (1ULL << n) - 1ULL;
    std::vector<Element> G;
    for (int f = 0; f < (with_flip ? 2 : 1); ++f)
        for (int a = 0; a < L1; ++a)
            for (int b = 0; b < L2; ++b) {
                Element e;
                e.perm.resize(static_cast<std::size_t>(n));
                for (int x = 0; x < L1; ++x)
                    for (int y = 0; y < L2; ++y)
                        e.perm[static_cast<std::size_t>(x + L1 * y)] = (x + a) % L1 + L1 * ((y + b) % L2);
                e.flip = f ? all : 0ULL;
                e.label = {a, b, f};
                G.push_back(std::move(e));
            }
    return G;
}

// Sector with characters exp(2 pi i (k1 a / L1 + k2 b / L2)) * parity^f. Reps and norms
// follow the engine: rep = min over the images of the policy's apply_perm, inv_norm =
// 1 / sqrt(|sum_{Stab} chi|^2 / |Stab|), orbits with zero projected norm dropped.
RepSectorData make_sector(const std::vector<Element>& G, int L1, int L2, int k1, int k2,
                          int parity, int n_up, bool lut) {
    const int n = L1 * L2;
    RepSectorData rd;
    rd.n_sites = n;
    rd.group_size = static_cast<int>(G.size());
    rd.n_up = n_up;
    bool flips = false;
    for (const auto& e : G) {
        rd.perms_flat.insert(rd.perms_flat.end(), e.perm.begin(), e.perm.end());
        rd.flip_masks.push_back(e.flip);
        flips = flips || e.flip != 0;
        const double ph = 2.0 * kPi * (static_cast<double>(k1 * e.label[0]) / L1
                                       + static_cast<double>(k2 * e.label[1]) / L2);
        rd.characters.push_back(std::polar(1.0, ph) * ((e.label[2] && parity < 0) ? -1.0 : 1.0));
    }
    if (!flips) rd.flip_masks.clear();
    const auto pol = rd.make_policy();
    for (std::uint64_t s = 0; s < (1ULL << n); ++s) {
        if (__builtin_popcountll(s) != n_up) continue;
        bool is_rep = true;
        Cx sum(0.0, 0.0);
        int stab = 0;
        for (int g = 0; g < rd.group_size && is_rep; ++g) {
            const std::uint64_t img = pol.apply_perm(s, g);
            if (img < s) is_rep = false;
            if (img == s) { sum += rd.characters[static_cast<std::size_t>(g)]; ++stab; }
        }
        if (!is_rep) continue;
        const double nsq = std::norm(sum) / stab;
        if (nsq < 1e-12) continue;
        rd.reps.push_back(s);
        rd.inv_norms.push_back(1.0 / std::sqrt(nsq));
    }
    if (lut) rd.build_perm_lut();
    return rd;
}

std::vector<Cx> random_vec(std::size_t d, std::mt19937& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    std::vector<Cx> v(d);
    double n2 = 0.0;
    for (auto& x : v) { x = Cx(g(rng), g(rng)); n2 += std::norm(x); }
    for (auto& x : v) x /= std::sqrt(n2);
    return v;
}

// <E a| O |E b> on the full 2^N space.
Cx dense_me(const MaskedOperator& O, const RepSectorData& tgt, const std::vector<Cx>& a,
            const RepSectorData& src, const std::vector<Cx>& b) {
    const auto pa = ed::sectors::expand(tgt, a, -1);
    const auto pb = ed::sectors::expand(src, b, -1);
    Cx r(0.0, 0.0);
    const auto terms = O.terms();
    for (std::uint64_t s = 0; s < pb.size(); ++s) {
        if (pb[s] == Cx(0.0, 0.0)) continue;
        for (const auto& t : terms) {
            std::uint64_t tt;
            double sg;
            if (!ed::ops::masked_apply(t, s, tt, sg)) continue;
            r += std::conj(pa[tt]) * t.coeff * sg * pb[s];
        }
    }
    return r;
}

MaskedOperator random_operator(int n, std::mt19937& rng) {
    const std::string alphabet = "+-zxyudI";
    std::uniform_int_distribution<int> op(0, 7), site(0, n - 1), len(1, 4);
    std::normal_distribution<double> g(0.0, 1.0);
    MaskedOperator O(n);
    for (int k = 0; k < 3; ++k) {
        const int K = len(rng);
        std::string ops;
        std::vector<int> sites;
        for (int i = 0; i < K; ++i) {
            ops.push_back(alphabet[static_cast<std::size_t>(op(rng))]);
            sites.push_back(site(rng));
        }
        O.add(MaskedOperator::product(n, ops, sites, Cx(g(rng), g(rng))));
    }
    return O;
}

// Every observable of `ops` for every (bra, ket) pair, engine vs dense.
double check_all(const std::vector<MaskedOperator>& ops, const RepSectorData& src,
                 const RepSectorData& tgt, const std::vector<std::vector<Cx>>& kets,
                 const std::vector<std::vector<Cx>>& bras) {
    const auto prog = compile_program(ops, src, tgt);
    std::vector<RepVectorView> kv, bv;
    for (const auto& v : kets) kv.push_back({v.data(), v.size()});
    for (const auto& v : bras) bv.push_back({v.data(), v.size()});
    std::vector<std::pair<int, int>> pairs;
    for (int b = 0; b < static_cast<int>(bras.size()); ++b)
        for (int k = 0; k < static_cast<int>(kets.size()); ++k) pairs.emplace_back(b, k);
    const auto M = rep_matrix_elements(src, tgt, prog, kv, bv, pairs);
    double err = 0.0;
    for (std::size_t p = 0; p < pairs.size(); ++p)
        for (std::size_t o = 0; o < ops.size(); ++o) {
            const Cx ref = dense_me(ops[o], tgt, bras[static_cast<std::size_t>(pairs[p].first)],
                                    src, kets[static_cast<std::size_t>(pairs[p].second)]);
            err = std::max(err, std::abs(M[p * ops.size() + o] - ref));
        }
    return err;
}

}  // namespace

TEST_CASE("same-sector elements of random non-Hermitian operators", "[rep_me]") {
    std::mt19937 rng(11);
    struct Sys { int L1, L2; bool flip; int k1, k2, parity; };
    for (const Sys& y : {Sys{12, 1, false, 1, 0, 1}, Sys{12, 1, false, 4, 0, 1},
                         Sys{12, 1, true, 6, 0, -1}, Sys{12, 1, true, 0, 0, 1},
                         Sys{4, 4, false, 2, 2, 1}, Sys{4, 4, true, 0, 0, 1},
                         Sys{4, 4, true, 1, 3, -1}}) {
        const int n = y.L1 * y.L2;
        const auto G = torus_group(y.L1, y.L2, y.flip);
        const auto rd = make_sector(G, y.L1, y.L2, y.k1, y.k2, y.parity, n / 2, /*lut=*/n == 16);
        REQUIRE(rd.dim() > 0);
        std::vector<MaskedOperator> ops;
        for (int i = 0; i < 6; ++i) ops.push_back(random_operator(n, rng));
        ops.push_back(MaskedOperator::product(n, "zzz", {0, 1, 2}, 1.0));   // odd string
        ops.push_back(MaskedOperator::product(n, "z", {5}, 1.0));
        ops.push_back(MaskedOperator::product(n, "I", {0}, 1.0));
        const std::vector<std::vector<Cx>> kets{random_vec(rd.dim(), rng), random_vec(rd.dim(), rng)};
        const std::vector<std::vector<Cx>> bras{random_vec(rd.dim(), rng), kets[0]};
        INFO("L " << y.L1 << "x" << y.L2 << " flip " << y.flip << " k " << y.k1 << "," << y.k2);
        CHECK(check_all(ops, rd, rd, kets, bras) < 1e-12);

        // identity: <a|b>, and dagger symmetry <a|T|b> = conj <b|T^dagger|a>
        const auto T = random_operator(n, rng);
        const std::vector<MaskedOperator> ops2{T, T.dagger(), MaskedOperator::product(n, "I", {0}, 1.0)};
        const auto prog = compile_program(ops2, rd, rd);
        const std::vector<RepVectorView> v{{kets[0].data(), rd.dim()}, {kets[1].data(), rd.dim()}};
        const auto M = rep_matrix_elements(rd, rd, prog, v, v, {{0, 1}, {1, 0}});
        Cx ip(0.0, 0.0);
        for (std::size_t r = 0; r < rd.dim(); ++r) ip += std::conj(kets[0][r]) * kets[1][r];
        CHECK(std::abs(M[0 * 3 + 2] - ip) < 1e-13);
        CHECK(std::abs(M[0 * 3 + 0] - std::conj(M[1 * 3 + 1])) < 1e-13);
    }
}

TEST_CASE("cross-momentum and cross-parity elements of single bonds", "[rep_me]") {
    std::mt19937 rng(12);
    for (const bool torus : {false, true}) {
        const int L1 = torus ? 4 : 12, L2 = torus ? 4 : 1, n = L1 * L2;
        const auto G = torus_group(L1, L2, /*flip=*/true);
        const auto a = make_sector(G, L1, L2, 1, 0, +1, n / 2, false);
        const auto b = make_sector(G, L1, L2, 0, 0, +1, n / 2, false);
        const auto c = make_sector(G, L1, L2, 1, 0, -1, n / 2, false);
        std::vector<MaskedOperator> ops{
            MaskedOperator::product(n, "+-", {0, 1}, 1.0),                    // one bond, not invariant
            MaskedOperator::product(n, "zz", {0, 1}, 1.0),
            MaskedOperator::product(n, "z", {3}, 1.0),                        // flip-odd
            MaskedOperator::product(n, "zzz", {0, 1, 2}, 1.0),               // flip-odd string
            MaskedOperator::product(n, "+-z", {0, 1, 2}, Cx(0.0, 1.0)),      // flip-odd, non-Hermitian
            MaskedOperator::product(n, "I", {0}, 1.0)};                       // must vanish across sectors
        const std::vector<std::vector<Cx>> ka{random_vec(a.dim(), rng)}, kb{random_vec(b.dim(), rng)},
            kc{random_vec(c.dim(), rng), random_vec(c.dim(), rng)};
        INFO((torus ? "4x4 torus" : "12-ring"));
        CHECK(check_all(ops, b, a, kb, ka) < 1e-12);   // k=0 -> k=1, same parity
        CHECK(check_all(ops, a, c, ka, kc) < 1e-12);   // (k,+) -> (k,-)
        CHECK(check_all(ops, c, a, kc, ka) < 1e-12);   // (k,-) -> (k,+)
        CHECK(check_all(ops, b, c, kb, kc) < 1e-12);   // momentum and parity both change

        const auto prog = compile_program({ops.back()}, a, c);
        CHECK(prog.n_terms() == 0);                     // selection rule: identity cannot connect
    }
}

TEST_CASE("cross-Sz elements of ladder strings", "[rep_me]") {
    std::mt19937 rng(13);
    for (const bool torus : {false, true}) {
        const int L1 = torus ? 4 : 12, L2 = torus ? 4 : 1, n = L1 * L2;
        const auto G = torus_group(L1, L2, /*flip=*/false);
        const auto s6 = make_sector(G, L1, L2, 0, 0, 1, n / 2, true);
        const auto s5 = make_sector(G, L1, L2, 1, 0, 1, n / 2 - 1, true);
        std::vector<MaskedOperator> up{MaskedOperator::product(n, "+", {0}, 1.0),
                                       MaskedOperator::product(n, "+z", {2, 0}, 1.0),
                                       MaskedOperator::product(n, "++-", {0, 1, 3}, 1.0),
                                       MaskedOperator::product(n, "-", {0}, 1.0)};   // wrong direction: dropped
        const std::vector<std::vector<Cx>> k6{random_vec(s6.dim(), rng), random_vec(s6.dim(), rng)},
            b5{random_vec(s5.dim(), rng)};
        INFO((torus ? "4x4 torus" : "12-ring"));
        CHECK(check_all(up, s6, s5, k6, b5) < 1e-12);   // S+ removes a set bit: n_up -> n_up - 1
        const auto prog = compile_program(up, s6, s5);
        CHECK(prog.terms_per_obs[3] == 0);
        std::vector<MaskedOperator> dn{MaskedOperator::product(n, "-", {7}, 1.0),
                                       MaskedOperator::product(n, "-z-+", {0, 4, 5, 9}, 1.0)};
        CHECK(check_all(dn, s5, s6, b5, k6) < 1e-12);
    }
}

TEST_CASE("sectors built by the engine", "[rep_me]") {
    const int N = 10;
    auto op = std::make_unique<Operator>(N, 0.5f);
    for (int i = 0; i < N; ++i) {
        Operator::TransformData t;
        t.op_type = 2; t.site_index = static_cast<std::uint64_t>(i); t.op_type_2 = 2;
        t.site_index_2 = static_cast<std::uint64_t>((i + 1) % N);
        t.coefficient = Cx(1.0, 0.0); t.is_two_body = true;
        op->add_record(t);
    }
    std::vector<std::vector<int>> A;
    for (int s = 0; s < N; ++s) {
        std::vector<int> p(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) p[static_cast<std::size_t>(i)] = (i + s) % N;
        A.push_back(std::move(p));
    }
    std::vector<RepSectorData> secs;
    ed::solvers::little_group_k_sectors_stream(*op, A, N, N / 2, -1,
                                               [&](RepSectorData& rd) { secs.push_back(rd); });
    REQUIRE(secs.size() >= 2);
    std::mt19937 rng(14);
    std::vector<MaskedOperator> ops{MaskedOperator::product(N, "+-", {2, 3}, 1.0),
                                    MaskedOperator::product(N, "zzz", {1, 4, 8}, 1.0),
                                    random_operator(N, rng)};
    for (std::size_t i = 0; i < secs.size(); ++i)
        for (std::size_t j = 0; j < secs.size(); j += 3) {
            const std::vector<std::vector<Cx>> k{random_vec(secs[i].dim(), rng)},
                b{random_vec(secs[j].dim(), rng)};
            INFO("sectors " << i << " -> " << j);
            CHECK(check_all(ops, secs[i], secs[j], k, b) < 1e-12);
        }
}

TEST_CASE("program / sector mismatch is rejected", "[rep_me]") {
    const auto G = torus_group(12, 1, false);
    const auto a = make_sector(G, 12, 1, 0, 0, 1, 6, false);
    const auto b = make_sector(G, 12, 1, 1, 0, 1, 6, false);
    const auto prog = compile_program({MaskedOperator::product(12, "zz", {0, 1}, 1.0)}, a, a);
    const std::vector<Cx> va(a.dim(), 0.0), vb(b.dim(), 0.0), vbad(a.dim() + 1, 0.0);
    CHECK_THROWS(rep_matrix_elements(b, b, prog, {{vb.data(), vb.size()}}, {{vb.data(), vb.size()}}, {{0, 0}}));
    CHECK_THROWS(rep_matrix_elements(a, a, prog, {{va.data(), va.size()}}, {{va.data(), va.size()}}, {{0, 1}}));
    CHECK_THROWS(rep_matrix_elements(a, a, prog, {{vbad.data(), vbad.size()}}, {{va.data(), va.size()}}, {{0, 0}}));
}

TEST_CASE("GPU sweep equals the CPU sweep", "[rep_me][gpu]") {
    if (!ed::ops::rep_matrix_elements_gpu_available()) SKIP("no CUDA device");
    std::mt19937 rng(15);
    const auto G = torus_group(4, 4, /*flip=*/true);
    const auto a = make_sector(G, 4, 4, 0, 0, +1, 8, true);
    const auto c = make_sector(G, 4, 4, 1, 3, -1, 8, false);
    std::vector<MaskedOperator> ops;
    for (int i = 0; i < 40; ++i) ops.push_back(random_operator(16, rng));
    ops.push_back(MaskedOperator::product(16, "+-+-", {0, 1, 4, 5}, 1.0));
    ops.push_back(MaskedOperator::product(16, "zzz", {0, 1, 2}, 1.0));
    ops.push_back(MaskedOperator::product(16, "I", {0}, 1.0));
    std::vector<std::vector<Cx>> va, vc;
    for (int i = 0; i < 3; ++i) va.push_back(random_vec(a.dim(), rng));
    for (int i = 0; i < 2; ++i) vc.push_back(random_vec(c.dim(), rng));
    const auto views = [](const std::vector<std::vector<Cx>>& v) {
        std::vector<RepVectorView> r;
        for (const auto& x : v) r.push_back({x.data(), x.size()});
        return r;
    };
    ed::ops::RepMEOptions gpu;
    gpu.use_gpu = true;
    gpu.gpu_shared_bytes = 4096;   // force observable chunking
    std::vector<std::pair<int, int>> pairs;
    for (int b = 0; b < 3; ++b)
        for (int k = 0; k < 3; ++k) pairs.emplace_back(b, k);
    // same sector (diagonal fast path), then cross momentum + parity
    for (const bool cross : {false, true}) {
        const auto& tgt = cross ? c : a;
        const auto& bv = cross ? vc : va;
        std::vector<std::pair<int, int>> pp;
        for (const auto& p : pairs)
            if (p.first < static_cast<int>(bv.size())) pp.push_back(p);
        const auto prog = compile_program(ops, a, tgt);
        const auto M0 = rep_matrix_elements(a, tgt, prog, views(va), views(bv), pp);
        const auto M1 = rep_matrix_elements(a, tgt, prog, views(va), views(bv), pp, gpu);
        double err = 0.0;
        for (std::size_t i = 0; i < M0.size(); ++i) err = std::max(err, std::abs(M0[i] - M1[i]));
        INFO("cross " << cross);
        CHECK(err < 1e-12);
    }
}

namespace {
// masks = every translate of `window` (a group-closed set of site windows)
std::vector<std::uint64_t> window_masks(const std::vector<Element>& G, std::uint64_t window, int n) {
    std::vector<std::uint64_t> m;
    for (const auto& e : G) {
        const std::uint64_t w = ed::ops::permute_mask(window, e.perm.data(), n);
        if (std::find(m.begin(), m.end(), w) == m.end()) m.push_back(w);
    }
    return m;
}
bool balanced_dense(std::uint64_t s, const std::vector<std::uint64_t>& masks) {
    for (auto m : masks)
        if (2 * __builtin_popcountll(s & m) != __builtin_popcountll(m)) return false;
    return true;
}
}  // namespace

TEST_CASE("constraint projector: <a|P O P|b> against dense", "[rep_me][balanced]") {
    std::mt19937 rng(16);
    for (const bool torus : {false, true}) {
        const int L1 = torus ? 4 : 12, L2 = torus ? 4 : 1, n = L1 * L2;
        const auto G = torus_group(L1, L2, /*flip=*/true);
        const auto a = make_sector(G, L1, L2, 0, 0, +1, n / 2, true);
        const auto c = make_sector(G, L1, L2, 1, 0, -1, n / 2, false);
        const std::uint64_t window = torus ? 0x33ULL : 0xFULL;   // 2x2 plaquette / 4 consecutive sites
        const auto masks = window_masks(G, window, n);
        std::vector<MaskedOperator> ops{random_operator(n, rng), MaskedOperator::product(n, "zzz", {0, 1, 2}, 1.0),
                                        MaskedOperator::product(n, "+-", {0, 1}, 1.0),
                                        MaskedOperator::product(n, "I", {0}, 1.0)};
        const std::vector<Cx> va = random_vec(a.dim(), rng), vc = random_vec(c.dim(), rng);
        ed::ops::RepMEOptions opt;
        opt.balanced_masks = masks;
        for (const bool cross : {false, true}) {
            const auto& tgt = cross ? c : a;
            const auto& vb = cross ? vc : va;
            const auto prog = compile_program(ops, a, tgt);
            const auto M = rep_matrix_elements(a, tgt, prog, {{va.data(), va.size()}}, {{vb.data(), vb.size()}}, {{0, 0}}, opt);
            const auto pa = ed::sectors::expand(tgt, vb, -1);
            const auto pb = ed::sectors::expand(a, va, -1);
            double err = 0.0;
            for (std::size_t o = 0; o < ops.size(); ++o) {
                Cx ref(0.0, 0.0);
                for (std::uint64_t s = 0; s < pb.size(); ++s) {
                    if (pb[s] == Cx(0.0, 0.0) || !balanced_dense(s, masks)) continue;
                    for (const auto& t : ops[o].terms()) {
                        std::uint64_t tt; double sg;
                        if (!ed::ops::masked_apply(t, s, tt, sg) || !balanced_dense(tt, masks)) continue;
                        ref += std::conj(pa[tt]) * t.coeff * sg * pb[s];
                    }
                }
                err = std::max(err, std::abs(M[o] - ref));
            }
            INFO((torus ? "4x4 " : "ring ") << (cross ? "cross" : "same") << " masks " << masks.size());
            CHECK(err < 1e-12);
            if (!cross) CHECK(std::abs(M[3]) < 1.0 - 1e-6);   // <P> < 1: the projector is doing something
        }
        // a mask set the group does not preserve is refused
        ed::ops::RepMEOptions bad;
        bad.balanced_masks = {window};
        const auto prog = compile_program(ops, a, a);
        CHECK_THROWS(rep_matrix_elements(a, a, prog, {{va.data(), va.size()}}, {{va.data(), va.size()}}, {{0, 0}}, bad));
        if (ed::ops::rep_matrix_elements_gpu_available()) {
            ed::ops::RepMEOptions g = opt;
            g.use_gpu = true;
            const auto P = compile_program(ops, a, c);
            const auto M0 = rep_matrix_elements(a, c, P, {{va.data(), va.size()}}, {{vc.data(), vc.size()}}, {{0, 0}}, opt);
            const auto M1 = rep_matrix_elements(a, c, P, {{va.data(), va.size()}}, {{vc.data(), vc.size()}}, {{0, 0}}, g);
            for (std::size_t i = 0; i < M0.size(); ++i) CHECK(std::abs(M0[i] - M1[i]) < 1e-12);
        }
    }
}

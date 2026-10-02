// =============================================================================
// tests/unit/test_row_walk.cpp
//
// Every lane that applies H, against one independent reference: the dense matrix of H's
// canonical terms (MaskedOperator::to_dense), and for a symmetry sector the block
// E^dagger H E with E the engine's own rep-basis expansion (ed::sectors::expand).
//
//   full space   Operator::apply: the row walk and the CSR assembled from it, each on a real
//                and a complex input;
//   rep sectors  RepSectorMatVec: reduced CSR (the default lane), the walk (CSR budget 0),
//                reduced_csr() itself, and the device gather on host pointers when a CUDA
//                device is present; the walk is the same bit for bit whether a sector looks
//                states up through a rank table or by binary search.
//
// Operators: Hermitian models on an 8-site ring, built through the builder and through raw
// records whose same-site products and cancelling S+S+ / S-S- pairs the lanes must
// reduce (J1-J2, XXZ with a transverse field, XYZ, D_z and D_x DM, the Cartesian
// Heisenberg form, the scalar chirality, S_tot^2 as the full double sum, random
// translation-symmetrised operators). Sectors: translations Z_8, the dihedral group D_8
// (non-abelian, its four 1-dim irreps), each with and without the spin flip, at
// n_up = 4, 3 and the full space (n_up = -1) as each model's symmetries allow.
// This test pins today's lanes before they move onto one row walk (P3.2), and the walk itself:
// compile_operator keeps every term exactly, and for_each_connection over it reproduces
// MaskedOperator::to_dense bit for bit (O directly, and rows of O through O^dagger).
// =============================================================================
#include "common/catch2_harness.h"

#include "engine/internal.h"   // RepSectorMatVec

#include <ed/basis/rep_sector.h>
#include <ed/core/select_backend.h>
#include <ed/input/hamiltonian_builder.h>
#include <ed/ops/invariance.h>
#include <ed/ops/operator.h>
#include <ed/ops/program.h>
#include <ed/ops/row_walk.h>
#include <ed/sectors/sectors.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

using Cx = std::complex<double>;
using ed::ops::MaskedOperator;
using ed::solvers::lg_detail::RepSectorMatVec;
using ed::symmetry::RepSectorData;

namespace {

constexpr int N = 8;
constexpr double kPi = 3.14159265358979323846;
constexpr std::uint64_t kAll = (1ULL << N) - 1ULL;

// ---- environment switches, restored on scope exit ----------------------------------------

struct EnvGuard {
    std::vector<std::pair<std::string, std::string>> saved;
    std::vector<std::string> unset;
    void set(const char* k, const char* v) {
        if (const char* old = std::getenv(k)) saved.emplace_back(k, old); else unset.emplace_back(k);
        setenv(k, v, 1);
    }
    ~EnvGuard() {
        for (const auto& [k, v] : saved) setenv(k.c_str(), v.c_str(), 1);
        for (const auto& k : unset) unsetenv(k.c_str());
    }
};

// ---- models ------------------------------------------------------------------------------

struct Model {
    std::string name;
    std::shared_ptr<Operator> H;
    bool u1, flip, dihedral;
};

std::vector<std::pair<std::size_t, std::size_t>> ring(int d = 1) {
    std::vector<std::pair<std::size_t, std::size_t>> b;
    for (int i = 0; i < N; ++i) b.emplace_back(i, (i + d) % N);
    return b;
}

std::shared_ptr<Operator> from_terms(const MaskedOperator& m) {
    return std::make_shared<Operator>(ed::ops::to_operator(m));
}

MaskedOperator P(const char* ops, std::vector<int> sites, Cx c = 1.0) {
    return MaskedOperator::product(N, ops, sites, c);
}

// sum over the translations of U_g R U_g^dagger, plus its adjoint
MaskedOperator translation_symmetrised(const MaskedOperator& R) {
    MaskedOperator S(N);
    for (int a = 0; a < N; ++a) {
        std::vector<int> p(N);
        for (int i = 0; i < N; ++i) p[static_cast<std::size_t>(i)] = (i + a) % N;
        S.add(R.image(p.data(), 0));
    }
    return S + S.dagger();
}

std::vector<Model> zoo() {
    using ed::input::HamiltonianBuilder;
    std::vector<Model> z;
    auto built = [&](const std::string& name, bool u1, bool flip, bool dih, auto&& fill) {
        HamiltonianBuilder b(N);
        fill(b);
        z.push_back({name, b.to_operator(), u1, flip, dih});
    };
    built("j1j2", true, true, true, [](HamiltonianBuilder& b) { b.heisenberg(ring(1), 1.0).heisenberg(ring(2), 0.35); });
    built("xxz+hx", false, true, true, [](HamiltonianBuilder& b) { b.xxz(ring(1), 1.0, 0.6).zeeman({0.4, 0.0, 0.0}); });
    built("xyz", false, true, true, [](HamiltonianBuilder& b) { b.xyz(ring(1), 1.0, 0.7, 0.4); });
    built("dm_z", true, false, false, [](HamiltonianBuilder& b) {
        b.heisenberg(ring(1), 1.0).dm(ring(1), std::vector<std::array<double, 3>>(N, {0.0, 0.0, 0.3}));
    });
    built("dm_x", false, false, false, [](HamiltonianBuilder& b) {
        b.heisenberg(ring(1), 1.0).dm(ring(1), std::vector<std::array<double, 3>>(N, {0.3, 0.0, 0.0}));
    });
    {   // the Cartesian Heisenberg ring: S+S+ / S-S- records that cancel
        auto H = std::make_shared<Operator>(N, 0.5f);
        const double q = 0.25;
        const int recs[8][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}};
        const double cs[8] = {q, q, q, q, -q, q, q, -q};
        for (const auto& [i, j] : ring(1)) {
            for (int r = 0; r < 8; ++r)
                H->addTwoBodyTerm(static_cast<std::uint8_t>(recs[r][0]), i, static_cast<std::uint8_t>(recs[r][1]), j, cs[r]);
            H->addTwoBodyTerm(2, i, 2, j, 1.0);
        }
        z.push_back({"heisenberg_cartesian", H, true, true, true});
    }
    {   // J1 + the scalar chirality on consecutive triples (complex, three-body)
        HamiltonianBuilder b(N);
        b.heisenberg(ring(1), 1.0);
        auto H = b.to_operator();
        const int pattern[6][3] = {{0, 1, 2}, {2, 0, 1}, {1, 2, 0}, {1, 0, 2}, {2, 1, 0}, {0, 2, 1}};
        for (std::uint64_t a = 0; a < N; ++a) {
            const std::uint64_t s[3] = {a, (a + 1) % N, (a + 2) % N};
            for (int p = 0; p < 6; ++p)
                H->addThreeBodyTerm(static_cast<std::uint8_t>(pattern[p][0]), s[0], static_cast<std::uint8_t>(pattern[p][1]), s[1],
                                    static_cast<std::uint8_t>(pattern[p][2]), s[2], Cx(0.0, p < 3 ? 0.15 : -0.15));
        }
        z.push_back({"chirality", H, true, true, false});
    }
    {   // S_tot^2 as the full double sum, same-site records included
        auto H = std::make_shared<Operator>(N, 0.5f);
        for (std::uint64_t i = 0; i < N; ++i)
            for (std::uint64_t j = 0; j < N; ++j) {
                H->addTwoBodyTerm(0, i, 1, j, 0.5);
                H->addTwoBodyTerm(1, i, 0, j, 0.5);
                H->addTwoBodyTerm(2, i, 2, j, 1.0);
            }
        z.push_back({"s_tot_squared", H, true, true, true});
    }
    {   // random translation-symmetrised operators: one U(1), one with no S^z content
        std::mt19937 rng(20261002);
        std::normal_distribution<double> g(0.0, 1.0);
        auto cx = [&] { return Cx(g(rng), g(rng)); };
        const auto u1 = translation_symmetrised(P("+-", {0, 1}, cx()) + P("z+-", {0, 1, 3}, cx()) + P("zz", {0, 2}, cx())
                                                + P("z", {0}, cx()) + P("+z-", {0, 1, 2}, cx()));
        z.push_back({"random_u1", from_terms(u1), true, false, false});
        const auto none = translation_symmetrised(P("+", {0}, cx()) + P("++", {0, 3}, cx()) + P("z-", {1, 2}, cx())
                                                  + P("x", {0}, cx()) + P("+-z", {0, 1, 2}, cx()));
        z.push_back({"random_none", from_terms(none), false, false, false});
    }
    return z;
}

// ---- sectors -----------------------------------------------------------------------------

struct Element { std::vector<int> perm; std::uint64_t flip; int rot, refl, par; };

// D_N (or Z_N without reflections) on the ring, times the flip when asked.
std::vector<Element> ring_group(bool dihedral, bool with_flip) {
    std::vector<Element> G;
    for (int f = 0; f < (with_flip ? 2 : 1); ++f)
        for (int s = 0; s < (dihedral ? 2 : 1); ++s)
            for (int a = 0; a < N; ++a) {
                Element e;
                e.perm.resize(N);
                for (int i = 0; i < N; ++i) e.perm[static_cast<std::size_t>(i)] = s ? ((a - i) % N + N) % N : (i + a) % N;
                e.flip = f ? kAll : 0ULL;
                e.rot = a; e.refl = s; e.par = f;
                G.push_back(std::move(e));
            }
    return G;
}

// The 1-dim irreps: momentum k (Z_N), or for D_N the four characters (A1, A2, B1, B2)
// = (rotation sign (+1 or (-1)^a), reflection sign); times the flip parity.
std::vector<std::vector<Cx>> characters(const std::vector<Element>& G, bool dihedral, bool with_flip) {
    std::vector<std::vector<Cx>> out;
    const int n_irr = dihedral ? 4 : N;
    for (int k = 0; k < n_irr; ++k)
        for (int p = 0; p < (with_flip ? 2 : 1); ++p) {
            std::vector<Cx> chi;
            for (const auto& e : G) {
                Cx c;
                if (!dihedral) c = std::polar(1.0, 2.0 * kPi * k * e.rot / N);
                else {
                    const double rs = (k >= 2 && (e.rot % 2)) ? -1.0 : 1.0;   // B: (-1)^a
                    const double fs = (e.refl && (k % 2)) ? -1.0 : 1.0;      // A2, B2: reflections -1
                    c = rs * fs;
                }
                if (e.par && p) c = -c;
                chi.push_back(c);
            }
            out.push_back(std::move(chi));
        }
    return out;
}

// The engine's rep basis for (G, chi, n_up): rep = the smallest image, 1/norm from the
// stabiliser sum, orbits with zero projection dropped (as in test_rep_matrix_elements).
RepSectorData make_sector(const std::vector<Element>& G, const std::vector<Cx>& chi, int n_up) {
    RepSectorData rd;
    rd.n_sites = N;
    rd.group_size = static_cast<int>(G.size());
    rd.n_up = n_up;
    bool flips = false;
    for (std::size_t g = 0; g < G.size(); ++g) {
        rd.perms_flat.insert(rd.perms_flat.end(), G[g].perm.begin(), G[g].perm.end());
        rd.flip_masks.push_back(G[g].flip);
        flips = flips || G[g].flip != 0;
        rd.characters.push_back(chi[g]);
    }
    if (!flips) rd.flip_masks.clear();
    const auto pol = rd.make_policy();
    for (std::uint64_t s = 0; s <= kAll; ++s) {
        if (n_up >= 0 && __builtin_popcountll(s) != n_up) continue;
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
    rd.build_perm_lut();
    return rd;
}

// ---- dense helpers -----------------------------------------------------------------------

using Mat = std::vector<Cx>;   // row-major, M[r * cols + c]

double max_abs(const Mat& M) {
    double m = 0.0;
    for (const auto& x : M) m = std::max(m, std::abs(x));
    return m;
}

double max_diff(const Mat& A, const Mat& B) {
    double d = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i) d = std::max(d, std::abs(A[i] - B[i]));
    return d;
}

// The matrix of a linear map given by its action on columns: M[r][c] = (apply e_c)[r].
// With `imag_input` the columns are taken from i e_c (then divided by i), so a real
// input path cannot engage.
Mat columns(std::size_t dim, const std::function<void(const Cx*, Cx*)>& apply, bool imag_input = false) {
    Mat M(dim * dim);
    std::vector<Cx> e(dim), y(dim);
    const Cx unit = imag_input ? Cx(0.0, 1.0) : Cx(1.0, 0.0);
    for (std::size_t c = 0; c < dim; ++c) {
        std::fill(e.begin(), e.end(), Cx(0.0, 0.0));
        e[c] = unit;
        apply(e.data(), y.data());
        for (std::size_t r = 0; r < dim; ++r) M[r * dim + c] = y[r] / unit;
    }
    return M;
}

// E^dagger H E for the sector's rep basis.
Mat reference_block(const Mat& Hd, const RepSectorData& rd) {
    const std::size_t d = rd.reps.size(), D = kAll + 1;
    std::vector<std::vector<Cx>> E(d);
    for (std::size_t c = 0; c < d; ++c) {
        std::vector<Cx> u(d, Cx(0.0, 0.0));
        u[c] = 1.0;
        E[c] = ed::sectors::expand(rd, u, -1);
    }
    Mat B(d * d, Cx(0.0, 0.0));
    for (std::size_t c = 0; c < d; ++c) {
        std::vector<Cx> HE(D, Cx(0.0, 0.0));
        for (std::size_t t = 0; t < D; ++t)
            for (std::size_t s = 0; s < D; ++s)
                if (E[c][s] != Cx(0.0, 0.0)) HE[t] += Hd[t * D + s] * E[c][s];
        for (std::size_t r = 0; r < d; ++r) {
            Cx acc(0.0, 0.0);
            for (std::size_t t = 0; t < D; ++t) acc += std::conj(E[r][t]) * HE[t];
            B[r * d + c] = acc;
        }
    }
    return B;
}

}  // namespace

TEST_CASE("full space: the walk and the CSR, on real and complex inputs, are H", "[row_walk]") {
    for (const auto& m : zoo()) {
        INFO("model " << m.name);
        const Mat ref = m.H->canonical().to_dense();
        const double tol = 1e-12 * std::max(1.0, max_abs(ref));
        const std::size_t D = kAll + 1;
        for (const char* lane : {"walk", "csr"}) {
            EnvGuard env;
            env.set("ED_CSR_FORCE", std::string(lane) == "csr" ? "1" : "0");
            const Operator H(*m.H);   // a fresh copy builds its lane, reading the switch, on first use
            INFO("lane " << lane);
            CHECK(std::string(H.full_space_lane()) == lane);
            for (bool imag : {false, true}) {
                INFO("imaginary input " << imag);
                const Mat M = columns(D, [&](const Cx* in, Cx* out) { H.apply(in, out, D); }, imag);
                CHECK(max_diff(M, ref) <= tol);
            }
        }
    }
}

TEST_CASE("rep sectors: CSR, walk and device gather are the block of H", "[row_walk]") {
    const bool device = ed::have_cuda();
    int sectors = 0;
    for (const auto& m : zoo()) {
        const MaskedOperator& h = m.H->canonical();
        REQUIRE(ed::ops::hermitian(h));
        const Mat Hd = h.to_dense();
        const double tol = 1e-12 * std::max(1.0, max_abs(Hd));
        for (bool dihedral : {false, true}) {
            if (dihedral && !m.dihedral) continue;
            for (bool with_flip : {false, true}) {
                if (with_flip && !m.flip) continue;
                const auto G = ring_group(dihedral, with_flip);
                for (const auto& e : G) {       // the model really has the symmetry
                    REQUIRE(ed::ops::commutes_with_permutation(h, e.perm));
                    if (e.flip) REQUIRE(ed::ops::flip_invariant(h));
                }
                std::vector<int> n_ups{-1};
                if (m.u1) n_ups = with_flip ? std::vector<int>{N / 2, -1} : std::vector<int>{N / 2, N / 2 - 1, -1};
                for (const auto& chi : characters(G, dihedral, with_flip))
                    for (int n_up : n_ups) {
                        RepSectorData rd = make_sector(G, chi, n_up);
                        const std::size_t d = rd.reps.size();
                        if (d == 0) continue;
                        ++sectors;
                        INFO("model " << m.name << " dihedral " << dihedral << " flip " << with_flip
                             << " n_up " << n_up << " dim " << d << " chi[1] " << chi[1]);
                        const Mat ref = reference_block(Hd, rd);
                        auto rds = std::make_shared<const RepSectorData>(std::move(rd));
                        {   // default: the reduced CSR
                            RepSectorMatVec op(*m.H, rds);
                            const Mat M = columns(d, [&](const Cx* in, Cx* out) { op.apply(in, out, d); });
                            CHECK(std::string(op.lane()) == "csr");
                            CHECK(max_diff(M, ref) <= tol);
                            const auto csr = op.reduced_csr();
                            const Mat C = columns(d, [&](const Cx* in, Cx* out) { csr.spmv(in, out); });
                            CHECK(max_diff(C, ref) <= tol);
                        }
                        {   // the walk: no CSR fits a budget of 0
                            EnvGuard env;
                            env.set("ED_SYM_SECTOR_CSR_BUDGET_GIB", "0");
                            RepSectorMatVec op(*m.H, rds);
                            const Mat M = columns(d, [&](const Cx* in, Cx* out) { op.apply(in, out, d); });
                            CHECK(std::string(op.lane()) == "walk");
                            CHECK(max_diff(M, ref) <= tol);
                            if (n_up >= 0) {   // the same walk through an O(1) rank table
                                RepSectorData ranked = *rds;
                                ranked.build_rank_table();
                                RepSectorMatVec op2(*m.H, std::make_shared<const RepSectorData>(std::move(ranked)));
                                const Mat M2 = columns(d, [&](const Cx* in, Cx* out) { op2.apply(in, out, d); });
                                CHECK(max_diff(M2, M) == 0.0);
                            }
                        }
                        if (device) {
                            RepSectorMatVec op(*m.H, rds, /*force_gpu=*/true);
                            const Mat M = columns(d, [&](const Cx* in, Cx* out) { op.apply(in, out, d); });
                            CHECK(std::string(op.lane()) == "gpu-gather");
                            CHECK(max_diff(M, ref) <= tol);
                        }
                    }
            }
        }
    }
    CHECK(sectors > 200);
}

TEST_CASE("compile_operator keeps every term; the row walk is to_dense exactly", "[row_walk]") {
    std::vector<MaskedOperator> ops;
    for (const auto& m : zoo()) ops.push_back(m.H->canonical());
    std::mt19937 rng(20261003);
    const std::string alphabet = "+-zxyudI";
    std::uniform_int_distribution<int> pick_op(0, static_cast<int>(alphabet.size()) - 1), pick_site(0, N - 1);
    std::uniform_int_distribution<int> pick_len(1, 4);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int trial = 0; trial < 20; ++trial) {   // non-Hermitian, up to four sites
        MaskedOperator O(N);
        for (int t = 0; t < 8; ++t) {
            std::string o;
            std::vector<int> sites;
            for (int k = pick_len(rng); k > 0; --k) {
                o.push_back(alphabet[static_cast<std::size_t>(pick_op(rng))]);
                sites.push_back(pick_site(rng));
            }
            O.add(MaskedOperator::product(N, o, sites, Cx(gauss(rng), gauss(rng))));
        }
        ops.push_back(O);
    }
    const std::size_t D = kAll + 1;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        INFO("operator " << i);
        const MaskedOperator& O = ops[i];
        const auto P = ed::ops::compile_operator(O);
        REQUIRE(ed::ops::program_operator(P, N).equals(O, 0.0));
        for (std::size_t g = 1; g < P.n_groups(); ++g) REQUIRE(P.group_flip[g - 1] < P.group_flip[g]);
        const Mat ref = O.to_dense();
        // columns: <t|O|s> straight from the walk
        Mat M(D * D, Cx(0.0, 0.0));
        for (std::uint64_t s = 0; s < D; ++s) {
            bool first = true;
            ed::ops::for_each_connection(P.view(), s, [&](std::uint64_t t, Cx h) {
                if (first && P.n_groups() > 0 && P.group_flip[0] == 0 && t != s) FAIL("diagonal not emitted first");
                first = false;
                M[t * D + s] += h;
            });
        }
        CHECK(max_diff(M, ref) == 0.0);
        // rows: <s|O|t> = conj(<t|O^dagger|s>)
        const auto Pd = ed::ops::compile_operator(O.dagger());
        Mat R(D * D, Cx(0.0, 0.0));
        for (std::uint64_t s = 0; s < D; ++s)
            ed::ops::for_each_connection(Pd.view(), s, [&](std::uint64_t t, Cx h) { R[s * D + t] += std::conj(h); });
        CHECK(max_diff(R, ref) == 0.0);
    }
    // a transverse field: one group whose subgroups differ in popcount
    const auto Px = ed::ops::compile_operator(P("x", {0}));
    REQUIRE(Px.n_groups() == 1);
    CHECK(Px.group_setbits[0] == -1);
    CHECK(ed::ops::compile_operator(P("+-", {0, 1})).group_setbits[0] == 1);
}

#ifdef WITH_CUDA
#include <cuda_runtime.h>

TEST_CASE("device: the multi-vector walk equals single applies bit for bit", "[row_walk][cuda]") {
    if (!ed::have_cuda()) SKIP("no CUDA device");
    for (const auto& m : zoo()) {
        if (m.name != "chirality" && m.name != "random_none") continue;
        INFO("model " << m.name);
        const auto G = ring_group(false, false);
        const int n_up = m.u1 ? N / 2 : -1;
        auto rds = std::make_shared<const RepSectorData>(make_sector(G, characters(G, false, false)[1], n_up));
        const std::size_t d = rds->reps.size();
        RepSectorMatVec op(*m.H, rds);
        op.enable_device(true);
        const auto single = op.bind_cuda();
        const auto multi = op.bind_cuda_multi();
        REQUIRE(multi);
        constexpr std::size_t K = 5;               // one launch of 4 vectors, one of 1
        std::mt19937 rng(7);
        std::normal_distribution<double> g(0.0, 1.0);
        std::vector<std::vector<Cx>> x(K, std::vector<Cx>(d));
        for (auto& v : x) for (auto& z : v) z = Cx(g(rng), g(rng));
        std::vector<Cx*> din(K), dout_m(K), dout_s(K);
        for (std::size_t i = 0; i < K; ++i) {
            REQUIRE(cudaMalloc(&din[i], d * sizeof(Cx)) == cudaSuccess);
            REQUIRE(cudaMalloc(&dout_m[i], d * sizeof(Cx)) == cudaSuccess);
            REQUIRE(cudaMalloc(&dout_s[i], d * sizeof(Cx)) == cudaSuccess);
            REQUIRE(cudaMemcpy(din[i], x[i].data(), d * sizeof(Cx), cudaMemcpyHostToDevice) == cudaSuccess);
        }
        std::vector<const Cx*> cin(din.begin(), din.end());
        multi(cin.data(), dout_m.data(), d, K);
        for (std::size_t i = 0; i < K; ++i) single(din[i], dout_s[i], d);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        for (std::size_t i = 0; i < K; ++i) {
            std::vector<Cx> ym(d), ys(d), yh(d);
            REQUIRE(cudaMemcpy(ym.data(), dout_m[i], d * sizeof(Cx), cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(cudaMemcpy(ys.data(), dout_s[i], d * sizeof(Cx), cudaMemcpyDeviceToHost) == cudaSuccess);
            op.apply(x[i].data(), yh.data(), d);   // the host lane
            CHECK(max_diff(ym, ys) == 0.0);
            CHECK(max_diff(ys, yh) <= 1e-12 * std::max(1.0, max_abs(yh)));
            cudaFree(din[i]); cudaFree(dout_m[i]); cudaFree(dout_s[i]);
        }
    }
}
#endif

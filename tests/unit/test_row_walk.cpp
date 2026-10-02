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
//                states up through a rank table, a shared two-level table, or by bisection.
//
// Operators: Hermitian models on an 8-site ring, built through the builder and through raw
// records whose same-site products and cancelling S+S+ / S-S- pairs the lanes must
// reduce (J1-J2, XXZ with a transverse field, XYZ, D_z and D_x DM, the Cartesian
// Heisenberg form, the scalar chirality, S_tot^2 as the full double sum, random
// translation-symmetrised operators, and a four-site ring exchange that only the canonical
// terms hold). Sectors: translations Z_8, the dihedral group D_8
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
#include "common/model_records.h"
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
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    namespace R = ed_tests::records;
    std::vector<Model> z;
    auto built = [&](const std::string& name, bool u1, bool flip, bool dih, auto&& fill) {
        auto H = std::make_shared<Operator>(N, 0.5f);
        fill(*H);
        z.push_back({name, H, u1, flip, dih});
    };
    built("j1j2", true, true, true, [](Operator& H) { R::heisenberg(H, ring(1), 1.0); R::heisenberg(H, ring(2), 0.35); });
    built("xxz+hx", false, true, true, [](Operator& H) { R::xxz(H, ring(1), 1.0, 0.6); R::zeeman(H, 0.4, 0.0, 0.0); });
    built("xyz", false, true, true, [](Operator& H) { R::xyz(H, ring(1), 1.0, 0.7, 0.4); });
    built("dm_z", true, false, false, [](Operator& H) { R::heisenberg(H, ring(1), 1.0); R::dm(H, ring(1), 0.0, 0.0, 0.3); });
    built("dm_x", false, false, false, [](Operator& H) { R::heisenberg(H, ring(1), 1.0); R::dm(H, ring(1), 0.3, 0.0, 0.0); });
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
        auto H = std::make_shared<Operator>(N, 0.5f);
        ed_tests::records::heisenberg(*H, ring(1), 1.0);
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
    {   // J1 + the four-site ring exchange K sum_i (P + P^dagger), P = P_{i,i+1} P_{i+1,i+2} P_{i+2,i+3}
        // with the transposition P_ij = 1/2 + 2 S_i.S_j: terms on four sites, no record holds them
        auto Pij = [](int i, int j) { return P("I", {0}, 0.5) + P("zz", {i, j}, 2.0) + P("+-", {i, j}) + P("-+", {i, j}); };
        MaskedOperator H(N);
        for (int i = 0; i < N; ++i) {
            const int a = i, b = (i + 1) % N, c = (i + 2) % N, d = (i + 3) % N;
            H.add(P("zz", {a, b}) + P("+-", {a, b}, 0.5) + P("-+", {a, b}, 0.5));
            const auto ring = Pij(a, b) * Pij(b, c) * Pij(c, d);
            H.add(ring + ring.dagger(), 0.3);
        }
        auto op = from_terms(H);
        REQUIRE(op->has_extra_terms());
        z.push_back({"ring_exchange", op, true, true, true});
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
                            if (n_up >= 0) {   // the same walk through an O(1) rank table, and two-level
                                RepSectorData ranked = *rds;
                                ranked.build_rank_table();
                                RepSectorMatVec op2(*m.H, std::make_shared<const RepSectorData>(std::move(ranked)));
                                const Mat M2 = columns(d, [&](const Cx* in, Cx* out) { op2.apply(in, out, d); });
                                CHECK(max_diff(M2, M) == 0.0);
                                // two-level: one rank table over every orbit of (G, n_up) (the trivial
                                // irrep keeps them all) and this sector's remap into it
                                const auto all = make_sector(G, std::vector<Cx>(G.size(), Cx(1.0, 0.0)), n_up);
                                RepSectorData two = *rds;
                                two.shared_rank = ed::symmetry::make_shared_rank_lookup(all.reps, N, n_up);
                                two.local_of_shared.assign(all.reps.size(), -1);
                                for (std::size_t gi = 0, local = 0; gi < all.reps.size(); ++gi)
                                    if (local < two.reps.size() && two.reps[local] == all.reps[gi])
                                        two.local_of_shared[gi] = static_cast<std::int32_t>(local++);
                                REQUIRE(two.has_two_level());
                                RepSectorMatVec op3(*m.H, std::make_shared<const RepSectorData>(std::move(two)));
                                const Mat M3 = columns(d, [&](const Cx* in, Cx* out) { op3.apply(in, out, d); });
                                CHECK(max_diff(M3, M) == 0.0);
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

// The sector CSR as it was built before P6.1 step 3: every row computed twice (count, then fill).
template <class Policy>
ed::matvec::ReducedSymmetryCsr<Cx> two_pass_csr(const ed::ops::ProgramView<Cx>& P, const Policy& pol, std::uint64_t dim) {
    ed::matvec::ReducedSymmetryCsr<Cx> csr;
    csr.dim = dim;
    csr.row_ptr.assign(dim + 1, 0);
    std::vector<std::pair<std::uint64_t, Cx>> row;
    for (std::uint64_t r = 0; r < dim; ++r) {
        ed::matvec::detail::cross_row(P, pol, pol, true, r, row);
        csr.row_ptr[r + 1] = csr.row_ptr[r] + row.size();
    }
    csr.allocate_first_touch();
    for (std::uint64_t r = 0; r < dim; ++r) {
        ed::matvec::detail::cross_row(P, pol, pol, true, r, row);
        std::uint64_t e = csr.row_ptr[r];
        for (const auto& [j, v] : row) { csr.col_idx[e] = static_cast<std::uint32_t>(j); csr.val[e] = v; ++e; }
    }
    return csr;
}

TEST_CASE("rep sectors: the one-pass CSR is the two-pass CSR bit for bit", "[row_walk]") {
    // P6.1 steps 3 and 6 build each row once into per-chunk slabs of (column, value id) and keep the
    // values in a dictionary; the entries must be the ones the two-pass build made, at any thread
    // count, and these clean models must engage the dictionary.
    int sectors = 0;
    for (const auto& m : zoo()) {
        const MaskedOperator& h = m.H->canonical();
        for (bool dihedral : {false, true}) {
            if (dihedral && !m.dihedral) continue;
            const auto G = ring_group(dihedral, false);
            std::vector<int> n_ups{-1};
            if (m.u1) n_ups = {N / 2, N / 2 - 1, -1};
            for (const auto& chi : characters(G, dihedral, false))
                for (int n_up : n_ups) {
                    const RepSectorData rd = make_sector(G, chi, n_up);
                    const std::uint64_t d = rd.reps.size();
                    if (d == 0) continue;
                    ++sectors;
                    INFO("model " << m.name << " dihedral " << dihedral << " n_up " << n_up << " dim " << d);
                    const auto P = ed::ops::compile_program({h.dagger()}, rd, rd);
                    const auto pol = rd.make_policy();
                    const auto ref = two_pass_csr(P.view(), pol, d);
                    for (int threads : {1, 3, 7}) {
#ifdef _OPENMP
                        const int before = omp_get_max_threads();
                        omp_set_num_threads(threads);
#endif
                        const auto csr = ed::matvec::build_sector_csr(P.view(), pol, d);
#ifdef _OPENMP
                        omp_set_num_threads(before);
#endif
                        INFO("threads " << threads);
                        CHECK(std::equal(csr.row_ptr.begin(), csr.row_ptr.end(), ref.row_ptr.begin(), ref.row_ptr.end()));
                        CHECK(std::equal(csr.col_idx.begin(), csr.col_idx.end(), ref.col_idx.begin(), ref.col_idx.end()));
                        if (ref.nnz() >= 64) CHECK(csr.dictionary());   // a tiny one stores its values
                        bool same_bits = true;
                        for (std::uint64_t e = 0; e < ref.nnz() && same_bits; ++e) {
                            const Cx a = csr.value(e), b = ref.val[e];
                            same_bits = std::memcmp(&a, &b, sizeof(Cx)) == 0;
                        }
                        CHECK(same_bits);
                        CHECK(csr.bytes() <= ref.bytes());
                    }
                }
        }
    }
    CHECK(sectors >= 12);
}

TEST_CASE("rep sectors: too many distinct values fall back to full values, bit for bit", "[row_walk]") {
    // Disordered couplings make every diagonal entry distinct: the Sz = 0 sector of a 20-site ring
    // (184756 states) holds more values than one dictionary (65536), and the build computes the rows
    // twice into full values -- the two-pass CSR exactly.
    const int n = 20;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> J(0.5, 1.5);
    MaskedOperator h(n);
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        const double jxy = J(rng), jz = J(rng);
        h.add(MaskedOperator::product(n, "+-", {i, j}, Cx(0.5 * jxy, 0.0)));
        h.add(MaskedOperator::product(n, "-+", {i, j}, Cx(0.5 * jxy, 0.0)));
        h.add(MaskedOperator::product(n, "zz", {i, j}, Cx(jz, 0.0)));
    }
    RepSectorData rd;                       // the plain Sz sector: the trivial group
    rd.n_sites = n;
    rd.group_size = 1;
    rd.n_up = n / 2;
    rd.characters = {Cx(1.0, 0.0)};
    for (int i = 0; i < n; ++i) rd.perms_flat.push_back(i);
    for (std::uint64_t s = 0; s < (std::uint64_t{1} << n); ++s)
        if (__builtin_popcountll(s) == n / 2) {
            rd.reps.push_back(s);
            rd.inv_norms.push_back(1.0);
        }
    rd.build_perm_lut();
    const auto P = ed::ops::compile_program({h.dagger()}, rd, rd);
    const auto pol = rd.make_policy();
    const std::uint64_t d = rd.reps.size();
    const auto ref = two_pass_csr(P.view(), pol, d);
    const auto csr = ed::matvec::build_sector_csr(P.view(), pol, d);
    CHECK_FALSE(csr.dictionary());
    CHECK(std::equal(csr.row_ptr.begin(), csr.row_ptr.end(), ref.row_ptr.begin(), ref.row_ptr.end()));
    CHECK(std::equal(csr.col_idx.begin(), csr.col_idx.end(), ref.col_idx.begin(), ref.col_idx.end()));
    CHECK(std::memcmp(csr.val.data(), ref.val.data(), ref.val.size() * sizeof(Cx)) == 0);
}

TEST_CASE("rep sectors: the bucket lookup finds what the binary search finds", "[row_walk]") {
    // P6.1 step 7: a sector without a rank table looks a representative up in one bucket of its
    // sorted reps; every state of every sector must map to the same index (or -1) either way.
    int sectors = 0;
    for (bool dihedral : {false, true})
        for (bool with_flip : {false, true}) {
            const auto G = ring_group(dihedral, with_flip);
            for (const auto& chi : characters(G, dihedral, with_flip))
                for (int n_up : with_flip ? std::vector<int>{N / 2, -1} : std::vector<int>{N / 2, N / 2 - 1, -1}) {
                    RepSectorData plain = make_sector(G, chi, n_up);
                    if (plain.reps.empty()) continue;
                    RepSectorData bucketed = plain;
                    bucketed.build_buckets();
                    REQUIRE_FALSE(bucketed.bucket_off.empty());
                    ++sectors;
                    const auto p0 = plain.make_policy(), p1 = bucketed.make_policy();
                    for (std::uint64_t st = 0; st <= kAll; ++st) {
                        INFO("n_up " << n_up << " state " << st);
                        REQUIRE(p1.index_of(st) == p0.index_of(st));
                        REQUIRE(p1.index_of_rep(st) == p0.index_of_rep(st));
                    }
                }
        }
    CHECK(sectors >= 12);
    RepSectorData with_table = make_sector(ring_group(false, false), std::vector<Cx>(N, Cx(1.0, 0.0)), N / 2);
    with_table.build_rank_table();
    with_table.build_buckets();
    CHECK(with_table.bucket_off.empty());   // the rank table answers in O(1)
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

TEST_CASE("cross-sector rows: <R;j|O|C;r> for any O between two sectors of one group", "[row_walk]") {
    std::mt19937 rng(20261004);
    const std::string alphabet = "+-zxyudI";
    std::uniform_int_distribution<int> pick_op(0, static_cast<int>(alphabet.size()) - 1), pick_site(0, N - 1);
    std::uniform_int_distribution<int> pick_len(1, 4);
    std::normal_distribution<double> gauss(0.0, 1.0);
    auto random_op = [&] {
        MaskedOperator O(N);
        for (int t = 0; t < 6; ++t) {
            std::string o;
            std::vector<int> sites;
            for (int k = pick_len(rng); k > 0; --k) {
                o.push_back(alphabet[static_cast<std::size_t>(pick_op(rng))]);
                sites.push_back(pick_site(rng));
            }
            O.add(MaskedOperator::product(N, o, sites, Cx(gauss(rng), gauss(rng))));
        }
        return O;
    };
    const std::size_t D = kAll + 1;
    // E_R^dagger O E_C
    auto reference = [&](const Mat& Od, const RepSectorData& R, const RepSectorData& C) {
        const std::size_t dr = R.reps.size(), dc = C.reps.size();
        auto expanded = [&](const RepSectorData& S) {
            std::vector<std::vector<Cx>> E(S.reps.size());
            for (std::size_t c = 0; c < S.reps.size(); ++c) {
                std::vector<Cx> u(S.reps.size(), Cx(0.0, 0.0));
                u[c] = 1.0;
                E[c] = ed::sectors::expand(S, u, -1);
            }
            return E;
        };
        const auto ER = expanded(R), EC = expanded(C);
        Mat B(dr * dc, Cx(0.0, 0.0));
        for (std::size_t c = 0; c < dc; ++c) {
            std::vector<Cx> OE(D, Cx(0.0, 0.0));
            for (std::size_t t = 0; t < D; ++t)
                for (std::size_t s = 0; s < D; ++s)
                    if (EC[c][s] != Cx(0.0, 0.0)) OE[t] += Od[t * D + s] * EC[c][s];
            for (std::size_t r = 0; r < dr; ++r) {
                Cx acc(0.0, 0.0);
                for (std::size_t t = 0; t < D; ++t) acc += std::conj(ER[r][t]) * OE[t];
                B[r * dc + c] = acc;
            }
        }
        return B;
    };
    auto by_columns = [&](std::size_t dr, std::size_t dc, const std::function<void(const Cx*, Cx*)>& apply) {
        Mat M(dr * dc);
        std::vector<Cx> e(dc), y(dr);
        for (std::size_t c = 0; c < dc; ++c) {
            std::fill(e.begin(), e.end(), Cx(0.0, 0.0));
            e[c] = 1.0;
            apply(e.data(), y.data());
            for (std::size_t r = 0; r < dr; ++r) M[r * dc + c] = y[r];
        }
        return M;
    };
    struct Pair { bool flip; int kR, nR, kC, nC; };
    const Pair pairs[] = {{false, 0, 4, 0, 4}, {false, 1, 4, 3, 4}, {false, 1, 3, 1, 4}, {false, 2, 4, 5, 3},
                          {false, 3, -1, 0, -1}, {false, 6, -1, 6, -1}, {true, 0, 4, 1, 4}, {true, 5, -1, 2, -1}};
    int checked = 0;
    for (int trial = 0; trial < 4; ++trial) {
        const MaskedOperator O = random_op();
        const Mat Od = O.to_dense();
        const double tol = 1e-12 * std::max(1.0, max_abs(Od));
        for (const Pair& pr : pairs) {
            const auto G = ring_group(false, pr.flip);
            const auto chis = characters(G, false, pr.flip);
            const RepSectorData R = make_sector(G, chis[static_cast<std::size_t>(pr.kR)], pr.nR);
            const RepSectorData C = make_sector(G, chis[static_cast<std::size_t>(pr.kC)], pr.nC);
            if (R.reps.empty() || C.reps.empty()) continue;
            INFO("trial " << trial << " flip " << pr.flip << " rows (" << pr.kR << ", " << pr.nR << ") cols ("
                 << pr.kC << ", " << pr.nC << ")");
            const auto P = ed::ops::compile_program({O.dagger()}, R, C);   // (O_lambda)^dagger, ket R, bra C
            const auto polR = R.make_policy(), polC = C.make_policy();
            const std::size_t dr = R.reps.size(), dc = C.reps.size();
            const Mat ref = reference(Od, R, C);
            const Mat G1 = by_columns(dr, dc, [&](const Cx* in, Cx* out) {
                ed::matvec::cross_gather(P.view(), polR, polC, false, dr, in, out);
            });
            CHECK(max_diff(G1, ref) <= tol);
            const auto csr = ed::matvec::build_cross_csr(P.view(), polR, polC, false, dr);
            const Mat C1 = by_columns(dr, dc, [&](const Cx* in, Cx* out) { csr.spmv(in, out); });
            CHECK(max_diff(C1, ref) <= tol);
            ++checked;
        }
        // one sector with itself: the diagonal shortcut
        const auto G = ring_group(false, false);
        const RepSectorData S = make_sector(G, characters(G, false, false)[2], 4);
        const auto P = ed::ops::compile_program({O.dagger()}, S, S);
        const auto pol = S.make_policy();
        const std::size_t d = S.reps.size();
        const Mat M = by_columns(d, d, [&](const Cx* in, Cx* out) { ed::matvec::cross_gather(P.view(), pol, pol, true, d, in, out); });
        CHECK(max_diff(M, reference(Od, S, S)) <= tol);
    }
    CHECK(checked >= 24);
}

#ifdef WITH_CUDA
TEST_CASE("device: the cross-sector walk is the host's", "[row_walk][cuda]") {
    if (!ed::have_cuda()) SKIP("no CUDA device");
    std::mt19937 rng(20261007);
    std::normal_distribution<double> gauss(0.0, 1.0);
    MaskedOperator O(N);   // three-body, four-site and Sz-changing terms
    O.add(P("z+z", {0, 1, 2}, Cx(0.7, 0.2)));
    O.add(P("+-zz", {1, 3, 4, 6}, Cx(-0.4, 0.9)));
    O.add(P("x", {5}, Cx(0.3, 0.0)));
    O.add(P("-", {7}, Cx(0.0, 1.1)));
    struct Pair { bool flip; int kR, nR, kC, nC; };
    const Pair pairs[] = {{false, 1, 4, 3, 4}, {false, 1, 3, 1, 4}, {false, 3, -1, 0, -1}, {true, 5, -1, 2, -1}};
    for (const Pair& pr : pairs) {
        const auto G = ring_group(false, pr.flip);
        const auto chis = characters(G, false, pr.flip);
        const RepSectorData R = make_sector(G, chis[static_cast<std::size_t>(pr.kR)], pr.nR);
        const RepSectorData C = make_sector(G, chis[static_cast<std::size_t>(pr.kC)], pr.nC);
        if (R.reps.empty() || C.reps.empty()) continue;
        INFO("flip " << pr.flip << " rows (" << pr.kR << ", " << pr.nR << ") cols (" << pr.kC << ", " << pr.nC << ")");
        const auto Pg = ed::ops::compile_program({O.dagger()}, R, C);
        const std::size_t dr = R.reps.size(), dc = C.reps.size();
        std::vector<Cx> x(dc), yh(dr), yd(dr);
        for (auto& z : x) z = Cx(gauss(rng), gauss(rng));
        ed::matvec::cross_gather(Pg.view(), R.make_policy(), C.make_policy(), false, dr, x.data(), yh.data());
        const auto fn = ed::symmetry::make_cross_matvec_gpu_rep(C, R, Pg);
        Cx *din = nullptr, *dout = nullptr;
        REQUIRE(cudaMalloc(&din, dc * sizeof(Cx)) == cudaSuccess);
        REQUIRE(cudaMalloc(&dout, dr * sizeof(Cx)) == cudaSuccess);
        REQUIRE(cudaMemcpy(din, x.data(), dc * sizeof(Cx), cudaMemcpyHostToDevice) == cudaSuccess);
        fn(din, dout, dr);
        REQUIRE(cudaMemcpy(yd.data(), dout, dr * sizeof(Cx), cudaMemcpyDeviceToHost) == cudaSuccess);
        cudaFree(din); cudaFree(dout);
        CHECK(max_abs(yh) > 0.0);
        CHECK(max_diff(yd, yh) <= 1e-12 * std::max(1.0, max_abs(yh)));
    }
}
#endif

TEST_CASE("orbit_matrix_element: <bra|O|ket> between sectors of different groups", "[row_walk]") {
    std::mt19937 rng(20261006);
    const std::string alphabet = "+-zxyudI";
    std::uniform_int_distribution<int> pick_op(0, static_cast<int>(alphabet.size()) - 1), pick_site(0, N - 1);
    std::uniform_int_distribution<int> pick_len(1, 4);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const std::size_t D = kAll + 1;
    auto rnd_vec = [&](std::size_t d) {   // unit norm: expand() returns a normalised vector
        std::vector<Cx> v(d);
        double n2 = 0.0;
        for (auto& x : v) { x = Cx(gauss(rng), gauss(rng)); n2 += std::norm(x); }
        for (auto& x : v) x /= std::sqrt(n2);
        return v;
    };
    struct Side { bool dihedral, flip; int irrep, n_up; };
    struct Pair { Side ket, bra; };
    const Pair pairs[] = {
        {{false, false, 1, 4}, {true, false, 1, 4}},      // Z_8 k = 1  vs  D_8 A2
        {{true, false, 2, 4}, {false, false, 0, 3}},      // D_8 B1 (n_up 4)  vs  Z_8 k = 0 (n_up 3)
        {{false, true, 3, 4}, {false, false, 1, 4}},      // Z_8 x flip  vs  Z_8
        {{true, true, 5, -1}, {false, false, 2, -1}},     // D_8 x flip  vs  Z_8, full space
        {{false, false, 2, 4}, {false, false, 6, 4}},     // one group: compare with rep_matrix_elements
    };
    for (int trial = 0; trial < 3; ++trial) {
        MaskedOperator O(N);
        for (int t = 0; t < 6; ++t) {
            std::string o;
            std::vector<int> sites;
            for (int k = pick_len(rng); k > 0; --k) {
                o.push_back(alphabet[static_cast<std::size_t>(pick_op(rng))]);
                sites.push_back(pick_site(rng));
            }
            O.add(MaskedOperator::product(N, o, sites, Cx(gauss(rng), gauss(rng))));
        }
        const Mat Od = O.to_dense();
        const auto P = ed::ops::compile_operator(O);
        for (const Pair& pr : pairs) {
            auto sector = [&](const Side& sd) {
                const auto G = ring_group(sd.dihedral, sd.flip);
                return make_sector(G, characters(G, sd.dihedral, sd.flip)[static_cast<std::size_t>(sd.irrep)], sd.n_up);
            };
            const RepSectorData S = sector(pr.ket), T = sector(pr.bra);
            if (S.reps.empty() || T.reps.empty()) continue;
            INFO("trial " << trial << " ket dihedral " << pr.ket.dihedral << " flip " << pr.ket.flip << " bra dihedral "
                 << pr.bra.dihedral << " flip " << pr.bra.flip);
            const auto ket = rnd_vec(S.reps.size()), bra = rnd_vec(T.reps.size());
            const auto ek = ed::sectors::expand(S, ket, -1), eb = ed::sectors::expand(T, bra, -1);
            Cx ref(0.0, 0.0);
            for (std::size_t t = 0; t < D; ++t)
                for (std::size_t s = 0; s < D; ++s)
                    if (ek[s] != Cx(0.0, 0.0)) ref += std::conj(eb[t]) * Od[t * D + s] * ek[s];
            const Cx got = ed::ops::orbit_matrix_element(P, S, T, ket, bra);
            const double tol = 1e-11 * std::max(1.0, std::abs(ref));
            CHECK(std::abs(got - ref) <= tol);
            if (ed::ops::same_group(S, T)) {
                const auto prog = ed::ops::compile_program({O}, S, T);
                const ed::ops::RepVectorView k{ket.data(), ket.size()}, b{bra.data(), bra.size()};
                const Cx me = ed::ops::rep_matrix_elements(S, T, prog, {k}, {b}, {{0, 0}})[0];
                CHECK(std::abs(me - ref) <= tol);
            }
        }
    }
}

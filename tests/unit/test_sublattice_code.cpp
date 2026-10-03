// =============================================================================
// tests/unit/test_sublattice_code.cpp
//
// The representative search by sublattices (<ed/basis/sublattice_code.h>), against brute force
// over the whole group in the same key order:
//
//   1. A block system is found, and the key order is a bijection of the sites.
//   2. The candidates of a state ascend and hold every element that reaches its key-least image.
//   3. An orbit table built through the code holds one key-least representative per orbit, with
//      its stabiliser.
//   4. The policy finds the representative and the projection bit for bit (the running sum over
//      the elements reaching it, ascending).
//   5. ED_SYM_SUBLATTICE: unset engages at N >= 24 with >= 64 permutations (16 under the relaxed rule), 0 never.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/basis/compiled_group.h>
#include <ed/basis/group.h>
#include <ed/basis/orbit_table.h>
#include <ed/basis/rep_sector.h>
#include <ed/basis/sublattice_code.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ed::symmetry;

namespace {

using Cx = std::complex<double>;
using Perms = std::vector<std::vector<int>>;

struct EnvGuard {   // ED_SYM_SUBLATTICE for one test, restored after
    explicit EnvGuard(const char* v) {
        if (v) setenv("ED_SYM_SUBLATTICE", v, 1);
        else   unsetenv("ED_SYM_SUBLATTICE");
    }
    ~EnvGuard() { unsetenv("ED_SYM_SUBLATTICE"); }
};

// D_N on a ring: N translations and N reflections.
Perms dihedral(int N) {
    return ed::sym::generate_group({ed::sym::translation(N, 1), ed::sym::reflection_1d(N)});
}

// p4m on an L x L square torus: translations, the fourfold rotation and a mirror.
Perms square_p4m(int L) {
    const int N = L * L;
    auto site = [L](int x, int y) { return ((x % L + L) % L) + L * ((y % L + L) % L); };
    std::vector<int> tx(N), ty(N), rot(N), mir(N);
    for (int y = 0; y < L; ++y)
        for (int x = 0; x < L; ++x) {
            tx[site(x, y)]  = site(x + 1, y);
            ty[site(x, y)]  = site(x, y + 1);
            rot[site(x, y)] = site(-y, x);
            mir[site(x, y)] = site(y, x);
        }
    return ed::sym::generate_group({tx, ty, rot, mir});
}

std::vector<int> flat(const Perms& p) {
    std::vector<int> f;
    for (const auto& q : p) f.insert(f.end(), q.begin(), q.end());
    return f;
}

std::uint64_t apply(const std::vector<int>& p, std::uint64_t s, std::uint64_t flip) {
    std::uint64_t r = 0;
    for (std::size_t i = 0; i < p.size(); ++i) r |= ((s >> p[i]) & 1ULL) << i;
    return r ^ flip;
}

std::uint64_t random_state(std::mt19937_64& rng, int N, int n_up) {
    std::vector<int> sites(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) sites[static_cast<std::size_t>(i)] = i;
    std::shuffle(sites.begin(), sites.end(), rng);
    std::uint64_t s = 0;
    for (int i = 0; i < n_up; ++i) s |= 1ULL << sites[static_cast<std::size_t>(i)];
    return s;
}

// The groups the tests run on, as element lists with flips (the second D24 is flip-extended).
struct Case {
    std::string name;
    Perms perms;
    std::vector<std::uint64_t> flips;
    int N;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    const Perms d24 = dihedral(24);
    out.push_back({"D24", d24, std::vector<std::uint64_t>(d24.size(), 0ULL), 24});
    Perms d24f = d24;
    d24f.insert(d24f.end(), d24.begin(), d24.end());
    std::vector<std::uint64_t> fl(d24f.size(), 0ULL);
    for (std::size_t g = d24.size(); g < fl.size(); ++g) fl[g] = (1ULL << 24) - 1;
    out.push_back({"D24 x flip", d24f, fl, 24});
    const Perms p4m = square_p4m(6);
    out.push_back({"p4m 6x6", p4m, std::vector<std::uint64_t>(p4m.size(), 0ULL), 36});
    return out;
}

std::shared_ptr<const SublatticeCode> code_of(const Case& c) {
    const auto f = flat(c.perms);
    return SublatticeCode::of(f.data(), c.flips.data(), static_cast<int>(c.perms.size()), c.N);
}

}  // namespace

TEST_CASE("sublattice code: a block system, and a key order that is a bijection", "[sublattice]") {
    EnvGuard on("1");
    for (const Case& c : cases()) {
        INFO(c.name);
        const auto code = code_of(c);
        REQUIRE(code);
        REQUIRE(c.N % code->block_size() == 0);
        REQUIRE(code->blocks() == c.N / code->block_size());
        REQUIRE(code->blocks() >= 2);
        REQUIRE(code->blocks() <= kSublatticeMaxBlocks);
        std::vector<int> kb = code->key_bit();
        std::sort(kb.begin(), kb.end());
        for (int i = 0; i < c.N; ++i) REQUIRE(kb[static_cast<std::size_t>(i)] == i);
        // every element maps blocks onto blocks: the image bits of one block come from one block
        const SublatticeView v = code->view();
        for (std::size_t g = 0; g < c.perms.size(); ++g)
            for (int j = 0; j < code->blocks(); ++j) {
                std::set<int> src;
                for (int i = 0; i < c.N; ++i)
                    if (code->key_bit()[static_cast<std::size_t>(i)] / code->block_size() == code->blocks() - 1 - j)
                        src.insert(code->key_bit()[static_cast<std::size_t>(c.perms[g][static_cast<std::size_t>(i)])]
                                   / code->block_size());
                REQUIRE(src.size() == 1);
            }
        (void)v;
    }
}

TEST_CASE("sublattice code: the candidates hold every element reaching the key-least image", "[sublattice]") {
    EnvGuard on("1");
    std::mt19937_64 rng(7);
    for (const Case& c : cases()) {
        INFO(c.name);
        const auto code = code_of(c);
        REQUIRE(code);
        const SublatticeView v = code->view();
        for (int trial = 0; trial < 3000; ++trial) {
            const int n_up = static_cast<int>(rng() % static_cast<std::uint64_t>(c.N + 1));
            const std::uint64_t s = random_state(rng, c.N, n_up);
            const std::uint64_t k = v.key(s);
            std::uint64_t best = ~0ULL;
            std::vector<int> reach;
            for (std::size_t g = 0; g < c.perms.size(); ++g) {
                const std::uint64_t ik = v.key(apply(c.perms[g], s, c.flips[g]));
                if (ik < best) { best = ik; reach.clear(); }
                if (ik == best) reach.push_back(static_cast<int>(g));
            }
            std::vector<int> cand;
            v.for_each_candidate(k, [&](int g) { cand.push_back(g); });
            REQUIRE(std::is_sorted(cand.begin(), cand.end()));
            REQUIRE(std::adjacent_find(cand.begin(), cand.end()) == cand.end());
            for (int g : reach) REQUIRE(std::binary_search(cand.begin(), cand.end(), g));
            REQUIRE(v.least_lead(k) <= v.pattern(k, 0));
            REQUIRE(v.least_lead(k) == static_cast<std::uint32_t>(best >> (c.N - code->block_size())));
        }
    }
}

TEST_CASE("sublattice code: orbit tables hold one key-least representative per orbit", "[sublattice]") {
    EnvGuard on("1");
    for (const Case& c : cases()) {
        if (c.N > 24) continue;                                // C(24, 12) = 2.7e6 states: enough
        INFO(c.name);
        const CompiledGroup cg = CompiledGroup::from_elements(c.perms, c.flips, c.N);
        REQUIRE(cg.sublattice() != nullptr);
        const SublatticeView v = cg.sublattice()->view();
        const OrbitTable tab = build_orbit_table_fixed_sz_streaming(static_cast<std::uint64_t>(c.N), c.N / 2, cg);
        // every rep is the key-least member of its orbit, with its stabiliser ascending
        std::size_t orbit_states = 0;
        for (std::size_t i = 0; i < tab.reps.size(); ++i) {
            const std::uint64_t r = tab.reps[i];
            std::vector<std::uint16_t> stab;
            std::set<std::uint64_t> orbit;
            for (std::size_t g = 0; g < c.perms.size(); ++g) {
                const std::uint64_t img = apply(c.perms[g], r, c.flips[g]);
                REQUIRE(v.key(img) >= v.key(r));
                if (img == r) stab.push_back(static_cast<std::uint16_t>(g));
                orbit.insert(img);
            }
            REQUIRE(stab == tab.stabilizer_of(i));
            orbit_states += orbit.size();
        }
        REQUIRE(std::is_sorted(tab.reps.begin(), tab.reps.end()));
        REQUIRE(orbit_states == tab.subspace_dim);             // the orbits tile the subspace
        REQUIRE(cg.is_canonical(tab.reps.front()));
    }
}

TEST_CASE("sublattice code: the policy finds the representative and the projection bit for bit", "[sublattice]") {
    EnvGuard on("1");
    std::mt19937_64 rng(11);
    for (const Case& c : cases()) {
        if (c.N > 24) continue;
        INFO(c.name);
        const CompiledGroup cg = CompiledGroup::from_elements(c.perms, c.flips, c.N);
        const OrbitTable tab = build_orbit_table_fixed_sz_streaming(static_cast<std::uint64_t>(c.N), c.N / 2, cg);
        RepSectorData rd;
        rd.reps = tab.reps;
        rd.inv_norms.assign(tab.reps.size(), 1.0);
        rd.group_size = static_cast<int>(c.perms.size());
        rd.n_sites = c.N;
        rd.n_up = c.N / 2;
        rd.perms_flat = flat(c.perms);
        rd.flip_masks = c.flips;
        rd.slc = tab.slc;                                      // the table's rule (filter_reps does this)
        REQUIRE(rd.slc);
        for (std::size_t g = 0; g < c.perms.size(); ++g)   // unit-modulus values, one per element
            rd.characters.push_back(std::polar(1.0, 0.37 * static_cast<double>(g) + 0.1));
        rd.build_perm_lut();
        const auto pol = rd.make_policy();
        REQUIRE(pol.slc.engaged());
        const SublatticeView v = pol.slc;
        for (int trial = 0; trial < 3000; ++trial) {
            const std::uint64_t s = random_state(rng, c.N, c.N / 2);
            std::uint64_t best = ~0ULL, rep = 0;
            double re = 0.0, im = 0.0;
            for (std::size_t g = 0; g < c.perms.size(); ++g) {
                const std::uint64_t img = apply(c.perms[g], s, c.flips[g]);
                const std::uint64_t ik = v.key(img);
                if (ik < best) {
                    best = ik;
                    rep = img;
                    re = 0.0 + rd.characters[g].real();
                    im = 0.0 - rd.characters[g].imag();
                } else if (ik == best) {
                    re += rd.characters[g].real();
                    im -= rd.characters[g].imag();
                }
            }
            REQUIRE(pol.representative(s) == rep);
            Cx proj;
            const std::int64_t idx = pol.index_and_projection(s, proj);
            REQUIRE(idx >= 0);
            REQUIRE(rd.reps[static_cast<std::size_t>(idx)] == rep);
            REQUIRE(proj.real() == re);   // the same terms, in the same order
            REQUIRE(proj.imag() == im);
        }
    }
}

TEST_CASE("sublattice code: a table keeps the rule it was built with", "[sublattice]") {
    // The rule is read when the group is compiled; the table records it, and a sector made from the
    // table carries it, so a later change of ED_SYM_SUBLATTICE changes nothing for them.
    const Case d24 = cases()[0];
    OrbitTable tab;
    {
        EnvGuard on("1");
        const CompiledGroup cg = CompiledGroup::from_elements(d24.perms, d24.flips, d24.N);
        tab = build_orbit_table_fixed_sz_streaming(static_cast<std::uint64_t>(d24.N), 4, cg);
        REQUIRE(tab.slc);
    }
    EnvGuard off("0");
    RepSectorData rd;
    rd.reps = tab.reps;
    rd.inv_norms.assign(tab.reps.size(), 1.0);
    rd.group_size = static_cast<int>(d24.perms.size());
    rd.n_sites = d24.N;
    rd.n_up = 4;
    rd.perms_flat = flat(d24.perms);
    rd.characters.assign(d24.perms.size(), Cx(1.0, 0.0));
    rd.slc = tab.slc;
    const auto pol = rd.make_policy();
    REQUIRE(pol.slc.engaged());
    for (std::uint64_t r : tab.reps) REQUIRE(pol.representative(r) == r);
    RepSectorData plain = rd;
    plain.slc = nullptr;                                       // hand-built: the plain order
    REQUIRE_FALSE(plain.make_policy().slc.engaged());
}

TEST_CASE("sublattice code: ED_SYM_SUBLATTICE unset engages large groups only, 0 never", "[sublattice]") {
    const Case d24 = cases()[0];
    {
        EnvGuard unset(nullptr);
        REQUIRE_FALSE(code_of(d24));                           // N = 24 but 48 permutations: too few
        const Perms d32 = dihedral(32);                        // N = 32, 64 permutations
        const auto f32 = flat(d32);
        REQUIRE(SublatticeCode::of(f32.data(), nullptr, static_cast<int>(d32.size()), 32));
        const Case p4m = cases()[2];                           // N = 36, 288 permutations
        REQUIRE(code_of(p4m));
        const Perms d8 = dihedral(8);
        const auto f = flat(d8);
        REQUIRE_FALSE(SublatticeCode::of(f.data(), nullptr, static_cast<int>(d8.size()), 8));
        {   // the relaxed rule (device verbs, eigs, spectrum) engages from 16 permutations: D24's 48 suffice
            const SublatticeRuleScope relaxed(true);
            REQUIRE(code_of(d24));
            REQUIRE_FALSE(SublatticeCode::of(f.data(), nullptr, static_cast<int>(d8.size()), 8));   // N < 24
        }
        REQUIRE_FALSE(code_of(d24));                           // the host rule again
    }
    {
        EnvGuard off("0");
        REQUIRE_FALSE(code_of(d24));
        const CompiledGroup cg = CompiledGroup::from_elements(d24.perms, d24.flips, d24.N);
        REQUIRE(cg.sublattice() == nullptr);
    }
}

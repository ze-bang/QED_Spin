// =============================================================================
// src/symmetry/group.cpp
//
// Implementation of `ed::sym::generate_group`, `group_from_generators`,
// and the `translation_group_1d` / `translation_group_with_reflection_1d`
// convenience builders declared in `ed/symmetry/group.h` (P2.11 /
// audit §3.10).
//
// All abelian-sector enumeration math (and the
// `power_representation` BFS) is borrowed verbatim from
// `SymmetryGroupInfo::computePowerRepresentation` /
// `filterInvalidSectors` in `ed/core/construct_ham.h` so the on-disk
// JSON path and the programmatic path produce IDENTICAL
// `SymmetryGroupInfo` blobs (modulo deterministic ordering). That
// equivalence is what makes the DSL safe to mix with the legacy
// JSON-fed code paths -- something we lock down in
// `tests/unit/test_symmetry_dsl.cpp`.
// =============================================================================

#include <ed/symmetry/group.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ed::sym {

namespace {

// Identity-permutation lookup table; saves a heap allocation in the BFS hot loop.
Permutation identity_internal(std::size_t n) {
    Permutation p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<int>(i);
    return p;
}

// `BFS` representation of `target` as `g_0^{p_0} o ... o g_{k-1}^{p_{k-1}}`.
// Mirrors `SymmetryGroupInfo::representAsGeneratorPowers` so the output
// vectors are bit-identical with the legacy JSON-loading path.
std::vector<int> represent_as_generator_powers(
        const std::vector<Permutation>& generators,
        const Permutation& target) {
    if (generators.empty()) return {};

    const std::size_t n = target.size();
    const Permutation id = identity_internal(n);
    if (target == id) return std::vector<int>(generators.size(), 0);

    struct State {
        std::vector<int> powers;
        Permutation perm;
    };

    std::queue<State> queue;
    std::set<Permutation> visited;
    queue.push({std::vector<int>(generators.size(), 0), id});
    visited.insert(id);

    while (!queue.empty()) {
        State curr = std::move(queue.front());
        queue.pop();
        for (std::size_t g = 0; g < generators.size(); ++g) {
            Permutation new_perm(n);
            for (std::size_t i = 0; i < n; ++i) {
                new_perm[i] = curr.perm[generators[g][i]];
            }
            if (new_perm == target) {
                std::vector<int> result = curr.powers;
                result[g]++;
                return result;
            }
            if (visited.insert(new_perm).second) {
                std::vector<int> next = curr.powers;
                next[g]++;
                queue.push({std::move(next), std::move(new_perm)});
            }
        }
    }
    throw std::runtime_error(
        "ed::sym::represent_as_generator_powers: target permutation is not "
        "in the group spanned by the given generators (BFS exhausted)");
}

} // namespace

bool is_abelian(const std::vector<Permutation>& generators) {
    for (std::size_t i = 0; i < generators.size(); ++i) {
        for (std::size_t j = i + 1; j < generators.size(); ++j) {
            if (compose(generators[i], generators[j]) !=
                compose(generators[j], generators[i])) {
                return false;
            }
        }
    }
    return true;
}

std::vector<Permutation>
maximal_abelian_subgroup_generators(
        const std::vector<Permutation>& preferred,
        const std::vector<Permutation>& group_elements) {
    if (group_elements.empty()) return {};
    const int n = static_cast<int>(group_elements.front().size());
    const Permutation id = identity_internal(static_cast<std::size_t>(n));

    // A plain greedy returns a subgroup maximal *by inclusion*, which need NOT
    // be maximal *by cardinality*: committing to a few commuting involutions
    // gives e.g. Z2^3 (order 8) and blocks the strictly larger pure-translation
    // subgroup Z3^2 (order 9) on a 3x3 periodic kagome cluster. We fix this two
    // ways: (1) sweep candidates translation-first -- fixed-point-free (lattice
    // translations have no fixed site) and higher group-order before low-order
    // involutions; (2) restart the greedy from every element and keep the
    // abelian subgroup of MAXIMUM order. |G| (an automorphism group) is small,
    // so the O(|G|^2) restart sweep is cheap and runs once at setup.
    auto fixed_point_free = [&](const Permutation& p) {
        for (int i = 0; i < n; ++i)
            if (p[static_cast<std::size_t>(i)] == i) return false;
        return true;
    };

    std::vector<Permutation> sweep = group_elements;
    std::stable_sort(sweep.begin(), sweep.end(),
        [&](const Permutation& a, const Permutation& bb) {
            const bool fa = fixed_point_free(a), fb = fixed_point_free(bb);
            if (fa != fb) return fa;                 // translations (fpf) first
            return order(a) > order(bb);             // then higher order first
        });

    // Greedy build from a seed list (then the ordered sweep); returns the
    // selected generators and the order of the abelian subgroup they span.
    auto build = [&](const std::vector<Permutation>& seed)
            -> std::pair<std::vector<Permutation>, std::size_t> {
        std::vector<Permutation> gens;
        std::set<Permutation>    closure{ id };
        auto try_add = [&](const Permutation& c) {
            if (c == id || closure.count(c)) return;
            for (const auto& g : gens)
                if (compose(c, g) != compose(g, c)) return;
            gens.push_back(c);
            closure.clear();
            for (const auto& e : generate_group(gens)) closure.insert(e);
        };
        for (const auto& c : seed)  try_add(c);
        for (const auto& c : sweep) try_add(c);
        return { gens, closure.size() };
    };

    auto best = build(preferred);                    // honor the caller's set first
    for (const auto& s : sweep) {                    // ... then restart from each
        if (s == id) continue;
        auto cand = build({ s });
        if (cand.second > best.second) best = std::move(cand);
    }
    return best.first;
}

namespace {
[[nodiscard]] bool require_abelian_env() {
    const char* v = std::getenv("ED_SYM_REQUIRE_ABELIAN");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}
} // namespace

std::vector<Permutation>
generate_group(const std::vector<Permutation>& generators) {
    if (generators.empty()) {
        throw std::invalid_argument(
            "ed::sym::generate_group: need at least one generator");
    }
    const std::size_t n = generators.front().size();
    for (const auto& g : generators) {
        validate(g, static_cast<int>(n));
    }

    std::set<Permutation> seen;
    std::queue<Permutation> queue;
    Permutation id = identity_internal(n);
    seen.insert(id);
    queue.push(id);

    while (!queue.empty()) {
        Permutation curr = std::move(queue.front());
        queue.pop();
        for (const auto& g : generators) {
            Permutation next(n);
            for (std::size_t i = 0; i < n; ++i) next[i] = g[curr[i]];
            if (seen.insert(next).second) {
                queue.push(std::move(next));
            }
        }
    }

    std::vector<Permutation> result(seen.begin(), seen.end());
    std::sort(result.begin(), result.end());
    return result;
}

SymmetryGroupInfo group_from_generators(
        int n_sites,
        std::vector<Permutation> generators,
        std::vector<std::vector<int>> sector_quantum_numbers) {
    if (n_sites <= 0) {
        throw std::invalid_argument(
            "ed::sym::group_from_generators: n_sites must be positive (got " +
            std::to_string(n_sites) + ")");
    }
    if (generators.empty()) {
        throw std::invalid_argument(
            "ed::sym::group_from_generators: need at least one generator");
    }
    for (const auto& g : generators) {
        validate(g, n_sites);
    }

    // ---------------------------------------------------------------------
    // Non-abelian safety guard. The sector/projection layer below only
    // implements 1-D (abelian) irreps. If the generators do NOT pairwise
    // commute, reducing by the full non-abelian group while projecting onto
    // 1-D characters would DROP the d>=2 irrep content -> an incomplete,
    // silently-wrong spectrum. We instead restrict to a maximal abelian
    // subgroup (a COMPLETE, correct reduction by a coarser factor |A|<=|G|),
    // unless the caller opts into a hard error via ED_SYM_REQUIRE_ABELIAN=1.
    // ---------------------------------------------------------------------
    if (!is_abelian(generators)) {
        const auto full = generate_group(generators);
        if (require_abelian_env()) {
            throw std::invalid_argument(
                "ed::sym::group_from_generators: the supplied generators do NOT "
                "commute (the generated group is non-abelian, |G|=" +
                std::to_string(full.size()) + "). The symmetry-projection layer "
                "only supports 1-D (abelian) irreps, so the d>=2 irreps would be "
                "dropped. Supply a commuting (abelian) generator set, or unset "
                "ED_SYM_REQUIRE_ABELIAN to auto-restrict to a maximal abelian "
                "subgroup.");
        }
        if (!sector_quantum_numbers.empty()) {
            throw std::invalid_argument(
                "ed::sym::group_from_generators: explicit sector_quantum_numbers "
                "were given for a NON-abelian generator set. Per-generator sector "
                "labels are not well-defined after the maximal-abelian-subgroup "
                "restriction; pass commuting generators, or omit "
                "sector_quantum_numbers to auto-enumerate the abelian sectors.");
        }
        const auto abelian_gens =
            maximal_abelian_subgroup_generators(generators, full);
        const auto abelian_size = generate_group(abelian_gens).size();
        std::cerr << "WARNING: ed::sym::group_from_generators: non-abelian "
                     "generators (|G|=" << full.size() << "); the projection "
                     "layer is abelian-only, so restricting to a maximal abelian "
                     "subgroup (|A|=" << abelian_size << "). The reduction factor "
                     "drops from " << full.size() << " to " << abelian_size
                  << " but the result is complete and correct. Set "
                     "ED_SYM_REQUIRE_ABELIAN=1 to error instead."
                  << std::endl;
        // Recurse with the abelian generators (now is_abelian => no restriction).
        return group_from_generators(n_sites, abelian_gens, {});
    }

    std::vector<int> generator_orders;
    generator_orders.reserve(generators.size());
    for (const auto& g : generators) generator_orders.push_back(order(g));

    auto max_clique = generate_group(generators);

    // power_representation[i] = (p_0, ..., p_{k-1}) such that
    //   max_clique[i] = g_0^{p_0} o ... o g_{k-1}^{p_{k-1}}
    std::vector<std::vector<int>> power_representation;
    power_representation.reserve(max_clique.size());
    for (const auto& a : max_clique) {
        power_representation.push_back(
            represent_as_generator_powers(generators, a));
    }

    // ---------------- Sector enumeration ----------------
    // Default: enumerate the full abelian product Z_{o_0} x ... x Z_{o_{k-1}};
    // any "phantom irreps" produced by generator relations get filtered by
    // the legacy filter loop below. This path is exactly what
    // `loadSectorMetadata` would feed in for a generated sector_metadata.json.
    if (sector_quantum_numbers.empty()) {
        std::uint64_t product_orders = 1;
        for (int o : generator_orders) product_orders *= static_cast<std::uint64_t>(o);
        if (product_orders > 1'000'000ull) {
            throw std::invalid_argument(
                "ed::sym::group_from_generators: product of generator orders "
                "(" + std::to_string(product_orders) + ") exceeds 1e6; pass "
                "sector_quantum_numbers explicitly to limit enumeration");
        }
        std::vector<int> q(generators.size(), 0);
        sector_quantum_numbers.reserve(static_cast<std::size_t>(product_orders));
        sector_quantum_numbers.push_back(q);
        while (true) {
            int carry = 1;
            for (int k = static_cast<int>(generators.size()) - 1;
                 k >= 0 && carry; --k) {
                q[static_cast<std::size_t>(k)] += carry;
                if (q[static_cast<std::size_t>(k)] >= generator_orders[
                        static_cast<std::size_t>(k)]) {
                    q[static_cast<std::size_t>(k)] = 0;
                    carry = 1;
                } else {
                    carry = 0;
                }
            }
            if (carry) break;
            sector_quantum_numbers.push_back(q);
        }
    } else {
        for (const auto& q : sector_quantum_numbers) {
            if (q.size() != generators.size()) {
                throw std::invalid_argument(
                    "ed::sym::group_from_generators: each sector must list one "
                    "quantum number per generator (got " +
                    std::to_string(q.size()) + " vs " +
                    std::to_string(generators.size()) + ")");
            }
            for (std::size_t k = 0; k < q.size(); ++k) {
                if (q[k] < 0 || q[k] >= generator_orders[k]) {
                    throw std::invalid_argument(
                        "ed::sym::group_from_generators: quantum number " +
                        std::to_string(q[k]) + " out of range [0, " +
                        std::to_string(generator_orders[k]) + ")");
                }
            }
        }
    }

    // Build SectorMetadata: phase_factors[a] = prod_k exp(-2 pi i q_k * p_{a,k} / o_k)
    // = χ_q(max_clique[a]). This is the PER-ELEMENT layout (length |G|) that the
    // distributed-symmetry path requires; the CPU consumers
    // (``compute_orbit_for_state`` / ``sector_characters_from``) detect this
    // length and read χ(g) directly (vs the per-generator JSON layout, length
    // num_generators, which they reconstruct via power_representation).
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    std::vector<SectorMetadata> sectors;
    sectors.reserve(sector_quantum_numbers.size());
    for (std::size_t s_idx = 0; s_idx < sector_quantum_numbers.size(); ++s_idx) {
        const auto& q = sector_quantum_numbers[s_idx];
        SectorMetadata sec;
        sec.sector_id = static_cast<std::uint64_t>(s_idx);
        sec.quantum_numbers = q;
        sec.phase_factors.reserve(max_clique.size());
        for (const auto& powers : power_representation) {
            double phase = 0.0;
            for (std::size_t k = 0; k < q.size(); ++k) {
                phase += static_cast<double>(q[k]) *
                         static_cast<double>(powers[k]) /
                         static_cast<double>(generator_orders[k]);
            }
            const double angle = -kTwoPi * phase;
            sec.phase_factors.emplace_back(std::cos(angle), std::sin(angle));
        }
        sectors.push_back(std::move(sec));
    }

    // Assemble the SymmetryGroupInfo. The remaining work
    // (filterInvalidSectors) is a public method we have to call manually
    // because the on-disk path runs it inside loadFromDirectory.
    SymmetryGroupInfo info;
    info.num_generators     = generators.size();
    info.generator_orders   = std::move(generator_orders);
    info.generators         = std::move(generators);
    info.max_clique         = std::move(max_clique);
    info.power_representation = std::move(power_representation);
    info.sectors            = std::move(sectors);

    // Inline phantom-irrep filter (mirrors filterInvalidSectors). We keep
    // the math here rather than calling the private member because the
    // private filter is gated on `sectors.size() > max_clique.size()`
    // which would skip the case where a relation makes the *enumerated*
    // sectors outnumber |G| on the very first run.
    if (info.sectors.size() > info.max_clique.size() &&
        !info.generators.empty()) {
        const std::size_t num_gen = info.generators.size();
        const std::size_t n_sites_local = info.generators.front().size();
        const Permutation id = identity_internal(n_sites_local);

        // Find relations: tuples (r_0,...,r_{k-1}) != 0 with
        //   g_0^{r_0} o ... o g_{k-1}^{r_{k-1}} = identity.
        std::vector<std::vector<int>> relations;
        std::uint64_t product_orders = 1;
        for (int o : info.generator_orders) product_orders *= static_cast<std::uint64_t>(o);
        if (product_orders <= 100'000ull) {
            std::vector<int> r(num_gen, 0);
            while (true) {
                bool all_zero = true;
                for (int v : r) if (v != 0) { all_zero = false; break; }
                if (!all_zero) {
                    Permutation result = id;
                    for (std::size_t k = 0; k < num_gen; ++k) {
                        Permutation gp = power(info.generators[k], r[k]);
                        Permutation tmp(n_sites_local);
                        for (std::size_t i = 0; i < n_sites_local; ++i) {
                            tmp[i] = result[gp[i]];
                        }
                        result = std::move(tmp);
                    }
                    if (result == id) relations.push_back(r);
                }
                int carry = 1;
                for (int k = static_cast<int>(num_gen) - 1; k >= 0 && carry; --k) {
                    r[static_cast<std::size_t>(k)] += carry;
                    if (r[static_cast<std::size_t>(k)] >= info.generator_orders[
                            static_cast<std::size_t>(k)]) {
                        r[static_cast<std::size_t>(k)] = 0;
                        carry = 1;
                    } else {
                        carry = 0;
                    }
                }
                if (carry) break;
            }
        }

        if (!relations.empty()) {
            std::vector<SectorMetadata> kept;
            kept.reserve(info.sectors.size());
            for (const auto& s : info.sectors) {
                bool valid = true;
                for (const auto& rel : relations) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < num_gen; ++k) {
                        sum += static_cast<double>(s.quantum_numbers[k]) *
                               static_cast<double>(rel[k]) /
                               static_cast<double>(info.generator_orders[k]);
                    }
                    if (std::abs(sum - std::round(sum)) > 1e-10) {
                        valid = false;
                        break;
                    }
                }
                if (valid) kept.push_back(s);
            }
            for (std::size_t i = 0; i < kept.size(); ++i) {
                kept[i].sector_id = static_cast<std::uint64_t>(i);
            }
            info.sectors = std::move(kept);
        }
    }

    return info;
}

SymmetryGroupInfo translation_group_1d(int n_sites) {
    return group_from_generators(n_sites, {translation(n_sites, 1)});
}

SymmetryGroupInfo translation_group_with_reflection_1d(int n_sites) {
    return group_from_generators(n_sites,
        {translation(n_sites, 1), reflection_1d(n_sites)});
}

} // namespace ed::sym

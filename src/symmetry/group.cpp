// =============================================================================
// src/symmetry/group.cpp
//
// Implementation of `ed::sym::generate_group` declared in
// `ed/symmetry/group.h` (P2.11 / audit §3.10).
// =============================================================================

#include <ed/symmetry/group.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <set>
#include <stdexcept>
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

} // namespace ed::sym

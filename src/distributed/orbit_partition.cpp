// =============================================================================
// src/distributed/orbit_partition.cpp    (Phase 3b #7, stage 1)
//
// LPT greedy orbit partitioner. See header for design and bounds.
// =============================================================================

#include <ed/distributed/orbit_partition.h>

#include <algorithm>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace ed::distributed {

namespace {

// Min-heap entry: (current rank weight, rank index). std::priority_queue
// is a max-heap by default, so we negate the comparator. We also need a
// stable tie-break on rank index (ascending) so the partition is
// reproducible across calls and across MPI ranks.
struct RankLoad {
    std::uint64_t weight;
    int rank;
};

struct RankLoadCmp {
    bool operator()(const RankLoad& a, const RankLoad& b) const noexcept {
        if (a.weight != b.weight) return a.weight > b.weight;
        return a.rank > b.rank;
    }
};

}  // namespace

OrbitPartition balanced_orbit_slab(
    const std::vector<std::uint64_t>& orbit_weights,
    int n_ranks) {

    if (n_ranks <= 0) {
        throw std::invalid_argument(
            "balanced_orbit_slab: n_ranks must be > 0 (got "
            + std::to_string(n_ranks) + ")");
    }

    OrbitPartition part;
    part.n_ranks = n_ranks;
    const std::size_t n_orbits = orbit_weights.size();
    part.orbit_owner.assign(n_orbits, 0);
    part.rank_orbits.assign(static_cast<std::size_t>(n_ranks),
                            std::vector<std::size_t>{});
    part.rank_weights.assign(static_cast<std::size_t>(n_ranks), 0);
    part.rank_offsets.assign(static_cast<std::size_t>(n_ranks) + 1, 0);

    if (n_orbits == 0) {
        // Empty orbit set is a valid input (degenerate sector). Every
        // rank gets zero orbits; rank_offsets is all zeros.
        return part;
    }

    // Sort orbit indices by weight, descending. Stable secondary key on
    // ascending orbit index to make ties deterministic.
    std::vector<std::size_t> order(n_orbits);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) {
                  if (orbit_weights[a] != orbit_weights[b]) {
                      return orbit_weights[a] > orbit_weights[b];
                  }
                  return a < b;
              });

    // Min-heap of (rank weight, rank index). Initialise every rank at
    // weight 0; ties broken by ascending rank index per RankLoadCmp.
    std::priority_queue<RankLoad, std::vector<RankLoad>, RankLoadCmp> heap;
    for (int r = 0; r < n_ranks; ++r) {
        heap.push({0, r});
    }

    // LPT pass: assign each (descending) orbit to the lightest rank.
    for (std::size_t orbit_idx : order) {
        RankLoad top = heap.top();
        heap.pop();
        const int r = top.rank;
        const std::uint64_t w = orbit_weights[orbit_idx];
        part.orbit_owner[orbit_idx] = r;
        part.rank_orbits[static_cast<std::size_t>(r)].push_back(orbit_idx);
        part.rank_weights[static_cast<std::size_t>(r)] = top.weight + w;
        heap.push({top.weight + w, r});
    }

    // Per-rank sorting: rank_orbits[r] in ascending orbit-index order so
    // downstream callers can build a SortedUint64Index without re-sorting.
    for (auto& v : part.rank_orbits) {
        std::sort(v.begin(), v.end());
    }

    // Prefix-sum offsets.
    std::size_t run = 0;
    for (int r = 0; r < n_ranks; ++r) {
        part.rank_offsets[static_cast<std::size_t>(r)] = run;
        run += part.rank_orbits[static_cast<std::size_t>(r)].size();
    }
    part.rank_offsets[static_cast<std::size_t>(n_ranks)] = run;

    // Per-orbit reverse lookup table (Phase 3b #7 stage 2 prep). Built
    // here so callers don't have to do an O(log local_size) binary search
    // at every owner_local_index() call -- the table is one int per
    // orbit, total ~n_orbits * 8 B which is dwarfed by the rank_orbits
    // storage itself.
    part.orbit_local_index.assign(n_orbits, 0);
    for (int r = 0; r < n_ranks; ++r) {
        const auto& v = part.rank_orbits[static_cast<std::size_t>(r)];
        for (std::size_t k = 0; k < v.size(); ++k) {
            part.orbit_local_index[v[k]] = k;
        }
    }

    return part;
}

double load_imbalance(const OrbitPartition& part) noexcept {
    if (part.n_ranks <= 0 || part.rank_weights.empty()) return 1.0;
    std::uint64_t total = 0;
    std::uint64_t mx = 0;
    for (auto w : part.rank_weights) {
        total += w;
        if (w > mx) mx = w;
    }
    if (total == 0) return 1.0;
    const double mean = static_cast<double>(total) /
                        static_cast<double>(part.n_ranks);
    return static_cast<double>(mx) / mean;
}

}  // namespace ed::distributed

// =============================================================================
// src/bfg/topology.cpp
//
// Implementations of `ed::bfg::find_triangles` / `ed::bfg::find_bowties`
// (P2.1 topology slice). These were previously file-local helpers in
// `src/apps/compute_bfg_order_parameters.cpp` (CPU driver) and were not
// reachable from `compute_bfg_order_parameters_gpu.cu` or from Python; they
// now live in the `ed_bfg` static library so all three callers share the
// same authoritative implementation.
// =============================================================================

#include "ed/bfg/topology.h"

#include <algorithm>
#include <iterator>
#include <set>
#include <tuple>

namespace ed::bfg {

std::vector<std::array<int, 3>> find_triangles(const Cluster& cluster) {
    std::vector<std::array<int, 3>> triangles;

    std::vector<std::set<int>> nn_adj(cluster.n_sites);
    for (const auto& [i, j] : cluster.edges_nn) {
        nn_adj[i].insert(j);
        nn_adj[j].insert(i);
    }

    for (int s1 = 0; s1 < cluster.n_sites; ++s1) {
        std::vector<int> neighbors;
        for (int n : nn_adj[s1]) {
            if (n > s1) neighbors.push_back(n);
        }

        for (size_t i = 0; i < neighbors.size(); ++i) {
            int s2 = neighbors[i];
            for (size_t k = i + 1; k < neighbors.size(); ++k) {
                int s3 = neighbors[k];
                if (nn_adj[s2].count(s3)) {
                    triangles.push_back({s1, s2, s3});
                }
            }
        }
    }

    return triangles;
}

std::vector<Bowtie> find_bowties(const Cluster& cluster) {
    std::vector<Bowtie> bowties;

    auto triangles = find_triangles(cluster);

    std::vector<std::vector<int>> vertex_to_triangles(cluster.n_sites);
    for (int idx = 0; idx < static_cast<int>(triangles.size()); ++idx) {
        const auto& tri = triangles[idx];
        vertex_to_triangles[tri[0]].push_back(idx);
        vertex_to_triangles[tri[1]].push_back(idx);
        vertex_to_triangles[tri[2]].push_back(idx);
    }

    std::set<std::tuple<int, int, int, int, int>> found_set;

    for (int shared_vertex = 0; shared_vertex < cluster.n_sites; ++shared_vertex) {
        const auto& tri_indices = vertex_to_triangles[shared_vertex];
        if (tri_indices.size() < 2) continue;

        for (size_t i = 0; i < tri_indices.size(); ++i) {
            for (size_t j = i + 1; j < tri_indices.size(); ++j) {
                const auto& t1 = triangles[tri_indices[i]];
                const auto& t2 = triangles[tri_indices[j]];

                std::vector<int> other1, other2;
                for (int v : {t1[0], t1[1], t1[2]}) {
                    if (v != shared_vertex) other1.push_back(v);
                }
                for (int v : {t2[0], t2[1], t2[2]}) {
                    if (v != shared_vertex) other2.push_back(v);
                }

                std::set<int> set1(other1.begin(), other1.end());
                std::set<int> set2(other2.begin(), other2.end());
                std::vector<int> intersection;
                std::set_intersection(set1.begin(), set1.end(),
                                      set2.begin(), set2.end(),
                                      std::back_inserter(intersection));
                if (!intersection.empty()) continue;

                int s0 = shared_vertex;
                int s1 = other1[0];
                int s2 = other1[1];
                int s3 = other2[0];
                int s4 = other2[1];

                if (s1 > s2) std::swap(s1, s2);
                if (s3 > s4) std::swap(s3, s4);
                if (std::make_pair(s1, s2) > std::make_pair(s3, s4)) {
                    std::swap(s1, s3);
                    std::swap(s2, s4);
                }

                auto key = std::make_tuple(s0, s1, s2, s3, s4);
                if (found_set.count(key)) continue;
                found_set.insert(key);

                std::array<double, 2> center = {0.0, 0.0};
                for (int site : {s0, s1, s2, s3, s4}) {
                    center[0] += cluster.positions[site][0];
                    center[1] += cluster.positions[site][1];
                }
                center[0] /= 5.0;
                center[1] /= 5.0;

                int orientation = cluster.sublattice[s0];

                bowties.push_back({s0, s1, s2, s3, s4, center, orientation});
            }
        }
    }

    return bowties;
}

}  // namespace ed::bfg

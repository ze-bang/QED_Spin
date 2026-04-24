// =============================================================================
// src/bfg/correlations.cpp
//
// Implementations of the BFG two-body spin correlations / bond expectations
// (P2.1 correlations slice). Lifted verbatim from
// `src/apps/compute_bfg_order_parameters.cpp` and put inside `ed_bfg` so the
// CPU driver, the GPU driver, and the future Python bindings all call the
// same authoritative implementation.
//
// The bit conventions match the rest of the codebase: bit=0 -> spin UP
// (Sz=+1/2), bit=1 -> spin DOWN (Sz=-1/2). Console diagnostics from the
// previous file-local implementations are intentionally dropped here -- the
// library version stays quiet and the CPU driver re-introduces the
// timing/progress prints if the user wants them.
// =============================================================================

#include "ed/bfg/correlations.h"

#include <cmath>
#include <complex>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::bfg {

namespace {

inline int get_bit(uint64_t state, int site) {
    return static_cast<int>((state >> site) & 1ULL);
}

inline uint64_t flip_bit(uint64_t state, int site) {
    return state ^ (1ULL << site);
}

inline double sz_value(uint64_t state, int site) {
    return get_bit(state, site) == 0 ? 0.5 : -0.5;
}

}  // namespace

std::vector<std::vector<Complex>> compute_smsp_correlations(
    const std::vector<Complex>& psi,
    int n_sites) {
    const uint64_t n_states = psi.size();
    std::vector<std::vector<Complex>> corr(n_sites, std::vector<Complex>(n_sites, 0.0));

    #pragma omp parallel
    {
        std::vector<std::vector<Complex>> local_corr(n_sites, std::vector<Complex>(n_sites, 0.0));

        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;

            for (int i = 0; i < n_sites; ++i) {
                for (int j = 0; j < n_sites; ++j) {
                    if (i == j) {
                        if (get_bit(state, i) == 1) {
                            local_corr[i][i] += std::norm(coeff);
                        }
                    } else {
                        if (get_bit(state, j) == 1 && get_bit(state, i) == 0) {
                            uint64_t new_state = flip_bit(flip_bit(state, i), j);
                            local_corr[i][j] += std::conj(psi[new_state]) * coeff;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < n_sites; ++i) {
                for (int j = 0; j < n_sites; ++j) {
                    corr[i][j] += local_corr[i][j];
                }
            }
        }
    }

    return corr;
}

std::vector<std::vector<double>> compute_szsz_correlations(
    const std::vector<Complex>& psi,
    int n_sites) {
    const uint64_t n_states = psi.size();
    std::vector<std::vector<double>> corr(n_sites, std::vector<double>(n_sites, 0.0));

    #pragma omp parallel
    {
        std::vector<std::vector<double>> local_corr(n_sites, std::vector<double>(n_sites, 0.0));

        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            double prob = std::norm(psi[state]);
            if (prob < 1e-30) continue;

            for (int i = 0; i < n_sites; ++i) {
                double sz_i = sz_value(state, i);
                for (int j = 0; j < n_sites; ++j) {
                    double sz_j = sz_value(state, j);
                    local_corr[i][j] += prob * sz_i * sz_j;
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < n_sites; ++i) {
                for (int j = 0; j < n_sites; ++j) {
                    corr[i][j] += local_corr[i][j];
                }
            }
        }
    }

    return corr;
}

std::map<std::pair<int, int>, Complex> compute_xy_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster) {
    const uint64_t n_states = psi.size();
    std::map<std::pair<int, int>, Complex> bonds;

    for (const auto& [i, j] : cluster.edges_nn) {
        bonds[{i, j}] = 0.0;
    }

    std::vector<std::pair<int, int>> edge_list(cluster.edges_nn.begin(), cluster.edges_nn.end());
    const int n_edges = static_cast<int>(edge_list.size());
    std::vector<Complex> bond_values(n_edges, 0.0);

    #pragma omp parallel
    {
        std::vector<Complex> local_bonds(n_edges, 0.0);

        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;

            for (int e = 0; e < n_edges; ++e) {
                int i = edge_list[e].first;
                int j = edge_list[e].second;

                if (get_bit(state, j) == 0 && get_bit(state, i) == 1) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    local_bonds[e] += std::conj(psi[new_state]) * coeff;
                }
                if (get_bit(state, i) == 0 && get_bit(state, j) == 1) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    local_bonds[e] += std::conj(psi[new_state]) * coeff;
                }
            }
        }

        #pragma omp critical
        {
            for (int e = 0; e < n_edges; ++e) {
                bond_values[e] += local_bonds[e];
            }
        }
    }

    for (int e = 0; e < n_edges; ++e) {
        bonds[edge_list[e]] = bond_values[e];
    }

    return bonds;
}

std::map<std::pair<int, int>, Complex> compute_spsm_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster) {
    const uint64_t n_states = psi.size();
    std::map<std::pair<int, int>, Complex> bonds;

    for (const auto& [i, j] : cluster.edges_nn) {
        bonds[{i, j}] = 0.0;
    }

    std::vector<std::pair<int, int>> edge_list(cluster.edges_nn.begin(), cluster.edges_nn.end());
    const int n_edges = static_cast<int>(edge_list.size());
    std::vector<Complex> bond_values(n_edges, 0.0);

    #pragma omp parallel
    {
        std::vector<Complex> local_bonds(n_edges, 0.0);

        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;

            for (int e = 0; e < n_edges; ++e) {
                int i = edge_list[e].first;
                int j = edge_list[e].second;

                if (get_bit(state, i) == 1 && get_bit(state, j) == 0) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    local_bonds[e] += std::conj(psi[new_state]) * coeff;
                }
            }
        }

        #pragma omp critical
        {
            for (int e = 0; e < n_edges; ++e) {
                bond_values[e] += local_bonds[e];
            }
        }
    }

    for (int e = 0; e < n_edges; ++e) {
        bonds[edge_list[e]] = bond_values[e];
    }

    return bonds;
}

std::map<std::pair<int, int>, double> compute_szsz_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster) {
    const uint64_t n_states = psi.size();
    std::map<std::pair<int, int>, double> bonds;

    for (const auto& [i, j] : cluster.edges_nn) {
        bonds[{i, j}] = 0.0;
    }

    std::vector<std::pair<int, int>> edge_list(cluster.edges_nn.begin(), cluster.edges_nn.end());
    const int n_edges = static_cast<int>(edge_list.size());
    std::vector<double> bond_values(n_edges, 0.0);

    #pragma omp parallel
    {
        std::vector<double> local_bonds(n_edges, 0.0);

        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            double prob = std::norm(psi[state]);
            if (prob < 1e-30) continue;

            for (int e = 0; e < n_edges; ++e) {
                int i = edge_list[e].first;
                int j = edge_list[e].second;

                double sz_i = sz_value(state, i);
                double sz_j = sz_value(state, j);
                local_bonds[e] += prob * sz_i * sz_j;
            }
        }

        #pragma omp critical
        {
            for (int e = 0; e < n_edges; ++e) {
                bond_values[e] += local_bonds[e];
            }
        }
    }

    for (int e = 0; e < n_edges; ++e) {
        bonds[edge_list[e]] = bond_values[e];
    }

    return bonds;
}

std::map<std::pair<int, int>, double> compute_heisenberg_bond_expectations(
    const std::map<std::pair<int, int>, double>& szsz_bonds,
    const std::map<std::pair<int, int>, Complex>& xy_bonds) {
    std::map<std::pair<int, int>, double> bonds;

    for (const auto& [edge, szsz] : szsz_bonds) {
        double xy = 0.0;
        if (xy_bonds.count(edge)) {
            xy = std::real(xy_bonds.at(edge));
        }
        bonds[edge] = szsz + 0.5 * xy;
    }

    return bonds;
}

}  // namespace ed::bfg

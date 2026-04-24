// =============================================================================
// src/bfg/cluster.cpp
//
// Implementation of `ed::bfg::Cluster` and `load_cluster(...)`.
//
// Logic moved verbatim from `src/apps/compute_bfg_order_parameters.cpp` so
// the binary's pre-refactor behaviour is preserved bit-for-bit. Only
// substantive change vs. the original: the struct + loader are now in the
// `ed::bfg` namespace.
//
// Audit ref: P2.1 (BFG library extraction).
// =============================================================================

#include "ed/bfg/cluster.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace ed::bfg {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

std::array<double, 2>
Cluster::minimum_image_displacement(int i, int j) const {
    if (n_cells_x <= 0 || n_cells_y <= 0 || cell_coords.empty()) {
        return {positions[j][0] - positions[i][0],
                positions[j][1] - positions[i][1]};
    }

    int n1_i = cell_coords[i][0], n2_i = cell_coords[i][1];
    int n1_j = cell_coords[j][0], n2_j = cell_coords[j][1];
    int sub_i = sublattice[i], sub_j = sublattice[j];

    int dn1 = n1_j - n1_i;
    int dn2 = n2_j - n2_i;

    if (dn1 > n_cells_x / 2)  dn1 -= n_cells_x;
    else if (dn1 < -n_cells_x / 2) dn1 += n_cells_x;
    if (dn2 > n_cells_y / 2)  dn2 -= n_cells_y;
    else if (dn2 < -n_cells_y / 2) dn2 += n_cells_y;

    if (n_cells_x % 2 == 1 && dn1 == (n_cells_x + 1) / 2)
        dn1 = -(n_cells_x - 1) / 2;
    if (n_cells_y % 2 == 1 && dn2 == (n_cells_y + 1) / 2)
        dn2 = -(n_cells_y - 1) / 2;

    double dx = dn1 * a1[0] + dn2 * a2[0]
              + sublattice_offsets[sub_j][0] - sublattice_offsets[sub_i][0];
    double dy = dn1 * a1[1] + dn2 * a2[1]
              + sublattice_offsets[sub_j][1] - sublattice_offsets[sub_i][1];

    return {dx, dy};
}

std::array<double, 2>
Cluster::bond_center_pbc(int i, int j) const {
    auto dr = minimum_image_displacement(i, j);
    return {positions[i][0] + 0.5 * dr[0],
            positions[i][1] + 0.5 * dr[1]};
}

Cluster load_cluster(const std::string& cluster_dir) {
    Cluster cluster;

    std::string lattice_file;
    for (const auto& suffix : {"_lattice_parameters.dat",
                               "_pbc_lattice_parameters.dat",
                               "_obc_lattice_parameters.dat"}) {
        for (const auto& prefix : {"kagome_bfg_2x2", "kagome_bfg_3x3",
                                   "kagome_bfg_4x4", "kagome_bfg_3x2",
                                   "kagome_bfg_2x3",
                                   "pyrochlore_super_1x1x1_pbc_kramer"}) {
            std::string test_file = cluster_dir + "/" + prefix + suffix;
            std::ifstream test(test_file);
            if (test.good()) {
                lattice_file = test_file;
                break;
            }
        }
        if (!lattice_file.empty()) break;
    }

    if (!lattice_file.empty()) {
        std::cout << "Loading lattice parameters from: " << lattice_file
                  << std::endl;
        std::ifstream lat_in(lattice_file);
        std::string line;
        bool reading_unit_vectors = false;
        bool reading_reciprocal_vectors = false;
        bool reading_kpoints = false;
        int vector_count = 0;

        while (std::getline(lat_in, line)) {
            if (line.empty() || line[0] == '#') {
                if (line.find("# Unit cell lattice vectors") != std::string::npos) {
                    reading_unit_vectors = true;
                    reading_reciprocal_vectors = false;
                    reading_kpoints = false;
                    vector_count = 0;
                } else if (line.find("# Reciprocal lattice vectors") != std::string::npos) {
                    reading_unit_vectors = false;
                    reading_reciprocal_vectors = true;
                    reading_kpoints = false;
                    vector_count = 0;
                } else if (line.find("# Format: k_index") != std::string::npos ||
                           line.find("# Allowed momentum points") != std::string::npos) {
                    reading_unit_vectors = false;
                    reading_reciprocal_vectors = false;
                    reading_kpoints = true;
                } else if (line.find("# Unit cells:") != std::string::npos) {
                    std::regex grid_regex(R"(#\s*Unit cells:\s*(\d+)\s*x\s*(\d+))");
                    std::smatch match;
                    if (std::regex_search(line, match, grid_regex)) {
                        cluster.n_cells_x = std::stoi(match[1]);
                        cluster.n_cells_y = std::stoi(match[2]);
                    }
                }
                continue;
            }

            std::istringstream iss(line);
            if (reading_unit_vectors && vector_count < 2) {
                int idx;
                double x, y;
                if (iss >> idx >> x >> y) {
                    if (idx == 0) cluster.a1 = {x, y};
                    else if (idx == 1) cluster.a2 = {x, y};
                    vector_count++;
                }
            } else if (reading_reciprocal_vectors && vector_count < 2) {
                int idx;
                double kx, ky;
                if (iss >> idx >> kx >> ky) {
                    if (idx == 0) cluster.b1 = {kx, ky};
                    else if (idx == 1) cluster.b2 = {kx, ky};
                    vector_count++;
                }
            } else if (reading_kpoints) {
                int k_idx, n1, n2;
                double kx, ky;
                if (iss >> k_idx >> n1 >> n2 >> kx >> ky) {
                    cluster.k_points.push_back({kx, ky});
                }
            }
        }

        std::cout << "  Lattice vectors: a1=(" << cluster.a1[0] << ","
                  << cluster.a1[1] << "), a2=(" << cluster.a2[0] << ","
                  << cluster.a2[1] << ")" << std::endl;
        std::cout << "  Reciprocal vectors: b1=(" << cluster.b1[0] << ","
                  << cluster.b1[1] << "), b2=(" << cluster.b2[0] << ","
                  << cluster.b2[1] << ")" << std::endl;
        std::cout << "  Read " << cluster.k_points.size()
                  << " k-points from lattice file" << std::endl;

        if (cluster.n_cells_x > 0 && cluster.n_cells_y > 0) {
            cluster.k_points.clear();
            int n_max_x = cluster.n_cells_x + 1;
            int n_max_y = cluster.n_cells_y + 1;
            for (int n1 = 0; n1 < n_max_x; ++n1) {
                for (int n2 = 0; n2 < n_max_y; ++n2) {
                    double kx = (static_cast<double>(n1) / cluster.n_cells_x) * cluster.b1[0]
                              + (static_cast<double>(n2) / cluster.n_cells_y) * cluster.b2[0];
                    double ky = (static_cast<double>(n1) / cluster.n_cells_x) * cluster.b1[1]
                              + (static_cast<double>(n2) / cluster.n_cells_y) * cluster.b2[1];
                    cluster.k_points.push_back({kx, ky});
                }
            }
            std::cout << "  Grid dimensions: " << cluster.n_cells_x << "x"
                      << cluster.n_cells_y << std::endl;
            std::cout << "  Extended to " << cluster.k_points.size()
                      << " k-points: n1=0.." << (n_max_x - 1)
                      << ", n2=0.." << (n_max_y - 1) << std::endl;
        } else {
            std::cout << "  Warning: Grid dimensions not found in file, using fallback"
                      << std::endl;
            int n_cells = cluster.n_sites / 3;
            int dim = static_cast<int>(std::sqrt(n_cells) + 0.5);
            int n_max = 4;
            cluster.k_points.clear();
            for (int n1 = 0; n1 < n_max; ++n1) {
                for (int n2 = 0; n2 < n_max; ++n2) {
                    double kx = (static_cast<double>(n1) / dim) * cluster.b1[0]
                              + (static_cast<double>(n2) / dim) * cluster.b2[0];
                    double ky = (static_cast<double>(n1) / dim) * cluster.b1[1]
                              + (static_cast<double>(n2) / dim) * cluster.b2[1];
                    cluster.k_points.push_back({kx, ky});
                }
            }
            std::cout << "  Extended to " << cluster.k_points.size()
                      << " k-points: (n1,n2)/" << dim
                      << " where n1,n2=0.." << (n_max - 1) << std::endl;
        }
    }

    std::string pos_file = cluster_dir + "/positions.dat";
    std::ifstream pos_in(pos_file);
    if (!pos_in.is_open()) {
        throw std::runtime_error("Cannot open positions.dat");
    }

    std::string line;
    while (std::getline(pos_in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int site_id, matrix_idx, sub_idx;
        double x, y;
        if (iss >> site_id >> matrix_idx >> sub_idx >> x >> y) {
            if (site_id >= static_cast<int>(cluster.positions.size())) {
                cluster.positions.resize(site_id + 1);
                cluster.sublattice.resize(site_id + 1);
            }
            cluster.positions[site_id]  = {x, y};
            cluster.sublattice[site_id] = sub_idx;
        }
    }
    cluster.n_sites = static_cast<int>(cluster.positions.size());

    std::string nn_file;
    for (const auto& prefix : {"kagome_bfg_2x2_pbc", "kagome_bfg_3x3_pbc",
                                "kagome_bfg_4x4_pbc",
                                "kagome_bfg_2x2_obc", "kagome_bfg_3x3_obc",
                                "kagome_bfg_4x4_obc",
                                "kagome_bfg_3x2_pbc", "kagome_bfg_3x2_obc"}) {
        std::string test_file = cluster_dir + "/" + prefix + "_nn_list.dat";
        std::ifstream test(test_file);
        if (test.good()) {
            nn_file = test_file;
            break;
        }
    }

    if (nn_file.empty()) {
        std::cout << "Warning: NN list file not found, constructing from positions"
                  << std::endl;
        double nn_dist = 0.5 + 0.01;
        for (int i = 0; i < cluster.n_sites; ++i) {
            for (int j = i + 1; j < cluster.n_sites; ++j) {
                double dx = cluster.positions[j][0] - cluster.positions[i][0];
                double dy = cluster.positions[j][1] - cluster.positions[i][1];
                double d  = std::sqrt(dx * dx + dy * dy);
                if (d < nn_dist) {
                    cluster.nn_list[i].push_back(j);
                    cluster.nn_list[j].push_back(i);
                    cluster.edges_nn.push_back({std::min(i, j), std::max(i, j)});
                }
            }
        }
    } else {
        std::ifstream nn_in(nn_file);
        while (std::getline(nn_in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            int site_id, n_neighbors;
            if (iss >> site_id >> n_neighbors) {
                for (int k = 0; k < n_neighbors; ++k) {
                    int neighbor;
                    if (iss >> neighbor) {
                        cluster.nn_list[site_id].push_back(neighbor);
                        if (site_id < neighbor) {
                            cluster.edges_nn.push_back({site_id, neighbor});
                        }
                    }
                }
            }
        }
    }

    if (cluster.a1[0] == 0.0 && cluster.a1[1] == 0.0) {
        std::cout << "Warning: Using default kagome lattice vectors (not loaded from file)"
                  << std::endl;
        cluster.a1 = {1.0, 0.0};
        cluster.a2 = {0.5, std::sqrt(3.0) / 2.0};
    }

    if (cluster.b1[0] == 0.0 && cluster.b1[1] == 0.0) {
        std::cout << "Warning: Computing reciprocal vectors (not loaded from file)"
                  << std::endl;
        double det = cluster.a1[0] * cluster.a2[1] - cluster.a1[1] * cluster.a2[0];
        cluster.b1 = {2.0 * kPi * cluster.a2[1] / det,
                      -2.0 * kPi * cluster.a2[0] / det};
        cluster.b2 = {-2.0 * kPi * cluster.a1[1] / det,
                      2.0 * kPi * cluster.a1[0] / det};
    }

    if (cluster.k_points.empty()) {
        std::cout << "Warning: Generating k-points grid (not loaded from file)"
                  << std::endl;
        int n_cells = cluster.n_sites / 3;
        int dim = static_cast<int>(std::sqrt(n_cells) + 0.5);
        if (dim * dim != n_cells) dim = n_cells;

        for (int n1 = 0; n1 < dim; ++n1) {
            for (int n2 = 0; n2 < dim; ++n2) {
                double kx = (static_cast<double>(n1) / dim) * cluster.b1[0]
                          + (static_cast<double>(n2) / dim) * cluster.b2[0];
                double ky = (static_cast<double>(n1) / dim) * cluster.b1[1]
                          + (static_cast<double>(n2) / dim) * cluster.b2[1];
                cluster.k_points.push_back({kx, ky});
            }
        }
    }

    for (const auto& [i, j] : cluster.edges_nn) {
        double dx = cluster.positions[j][0] - cluster.positions[i][0];
        double dy = cluster.positions[j][1] - cluster.positions[i][1];
        double angle = std::atan2(dy, dx);
        double angle_deg = std::fmod(angle * 180.0 / kPi + 180.0, 180.0);

        int orientation;
        if (angle_deg < 30.0 || angle_deg >= 150.0) {
            orientation = 0;
        } else if (angle_deg < 90.0) {
            orientation = 1;
        } else {
            orientation = 2;
        }
        cluster.bond_orientation[{i, j}] = orientation;
        cluster.bond_orientation[{j, i}] = orientation;
    }

    cluster.sites_per_cell = 3;
    cluster.sublattice_offsets = {
        {0.0, 0.0},
        {0.5, 0.0},
        {0.25, std::sqrt(3.0) / 4.0}
    };

    if (cluster.n_cells_x <= 0 || cluster.n_cells_y <= 0) {
        int n_cells = cluster.n_sites / cluster.sites_per_cell;
        int dim = static_cast<int>(std::sqrt(n_cells) + 0.5);
        if (dim * dim == n_cells) {
            cluster.n_cells_x = dim;
            cluster.n_cells_y = dim;
        } else {
            for (int d1 = dim + 1; d1 >= 1; --d1) {
                if (n_cells % d1 == 0) {
                    int d2 = n_cells / d1;
                    cluster.n_cells_x = d1;
                    cluster.n_cells_y = d2;
                    break;
                }
            }
        }
    }

    cluster.cell_coords.resize(cluster.n_sites);
    double det = cluster.a1[0] * cluster.a2[1] - cluster.a1[1] * cluster.a2[0];
    if (std::abs(det) > 1e-10) {
        for (int site = 0; site < cluster.n_sites; ++site) {
            int sub = cluster.sublattice[site];
            double rx = cluster.positions[site][0] - cluster.sublattice_offsets[sub][0];
            double ry = cluster.positions[site][1] - cluster.sublattice_offsets[sub][1];

            double n1_f = (rx * cluster.a2[1] - ry * cluster.a2[0]) / det;
            double n2_f = (ry * cluster.a1[0] - rx * cluster.a1[1]) / det;

            int n1 = static_cast<int>(std::round(n1_f));
            int n2 = static_cast<int>(std::round(n2_f));

            cluster.cell_coords[site] = {n1, n2};
        }
        std::cout << "  Computed unit cell coordinates for PBC-correct displacements"
                  << std::endl;
        std::cout << "  Grid: " << cluster.n_cells_x << " x " << cluster.n_cells_y
                  << " unit cells" << std::endl;
    } else {
        std::cout << "  Warning: Could not compute unit cell coordinates (det=0)"
                  << std::endl;
    }

    std::cout << "Loaded cluster: " << cluster.n_sites << " sites, "
              << cluster.edges_nn.size() << " NN bonds" << std::endl;

    return cluster;
}

}  // namespace ed::bfg

/**
 * @file compute_bfg_order_parameters.cpp
 * @brief Fast C++ computation of BFG order parameters from wavefunctions
 * 
 * Computes:
 * 1. S(q) - Spin structure factor using S^-S^+ correlations at ALL k-points + 2D grid
 * 2. Nematic order - Bond orientation anisotropy (C3 → C1 breaking)
 *    - Variants: XY (S+S- + S-S+), S+S-, SzSz, Heisenberg (S·S)
 * 3. VBS (Valence Bond Solid) order - S_D(q) with proper 4-site correlations
 *    - Computes actual ⟨D_b D_b'⟩ dimer-dimer correlations
 * 
 * Output includes:
 * - Order parameters at special k-points (Γ, K, M, etc.)
 * - Full 2D structure factor grids for visualization
 * - Per-bond expectations for spatial visualization
 * - Detailed HDF5 output compatible with Python plotting
 * 
 * Usage:
 *   ./compute_bfg_order_parameters <wavefunction.h5> <cluster_dir> [output.h5]
 * 
 * Compile with:
 *   g++ -O3 -march=native -fopenmp -std=c++17 compute_bfg_order_parameters.cpp -o compute_bfg_order_parameters -lhdf5 -lhdf5_cpp
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <complex>
#include <cmath>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <regex>
#include <mutex>
#include <atomic>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <H5Cpp.h>

namespace fs = std::filesystem;

using Complex = std::complex<double>;
const double PI = 3.14159265358979323846;
const Complex I(0.0, 1.0);

// -----------------------------------------------------------------------------
// Bit manipulation helpers (inlined for speed)
// -----------------------------------------------------------------------------

inline int get_bit(uint64_t state, int site) {
    return (state >> site) & 1;
}

inline uint64_t set_bit(uint64_t state, int site, int value) {
    if (value) {
        return state | (1ULL << site);
    } else {
        return state & ~(1ULL << site);
    }
}

inline uint64_t flip_bit(uint64_t state, int site) {
    return state ^ (1ULL << site);
}

// -----------------------------------------------------------------------------
// Spin operators on basis states
// Returns: (new_state, coefficient) or (-1, 0) if annihilated
// -----------------------------------------------------------------------------

// S^+ raises spin: |↓⟩ → |↑⟩, |↑⟩ → 0
// ED convention: bit=0 is UP (Sz=+1/2), bit=1 is DOWN (Sz=-1/2)
inline std::pair<int64_t, double> apply_sp(uint64_t state, int site) {
    if (get_bit(state, site) == 1) {  // spin down (bit=1 in ED convention)
        return {static_cast<int64_t>(state & ~(1ULL << site)), 1.0};  // flip to 0 (up)
    }
    return {-1, 0.0};
}

// S^- lowers spin: |↑⟩ → |↓⟩, |↓⟩ → 0
// ED convention: bit=0 is UP (Sz=+1/2), bit=1 is DOWN (Sz=-1/2)
inline std::pair<int64_t, double> apply_sm(uint64_t state, int site) {
    if (get_bit(state, site) == 0) {  // spin up (bit=0 in ED convention)
        return {static_cast<int64_t>(state | (1ULL << site)), 1.0};  // flip to 1 (down)
    }
    return {-1, 0.0};
}

// S^z eigenvalue: |↑⟩ → +1/2, |↓⟩ → -1/2
// ED convention: bit=0 is UP (Sz=+1/2), bit=1 is DOWN (Sz=-1/2)
inline double sz_value(uint64_t state, int site) {
    return get_bit(state, site) ? -0.5 : 0.5;
}

// -----------------------------------------------------------------------------
// Cluster data structure
// -----------------------------------------------------------------------------

struct Cluster {
    int n_sites;
    std::vector<std::array<double, 2>> positions;
    std::vector<int> sublattice;
    std::vector<std::pair<int, int>> edges_nn;
    std::map<int, std::vector<int>> nn_list;
    std::array<double, 2> a1, a2;  // lattice vectors
    std::array<double, 2> b1, b2;  // reciprocal vectors
    std::vector<std::array<double, 2>> k_points;
    int n_cells_x = 0, n_cells_y = 0;  // Grid dimensions (e.g., 3x3 or 2x3)
    
    // Derived: bond orientations (0, 1, 2)
    std::map<std::pair<int, int>, int> bond_orientation;
    
    // Unit cell representation for PBC-correct displacements
    // Each site i has: position[i] = R_{cell} + delta_{sublattice}
    // where R_{cell} = cell_coords[i][0] * a1 + cell_coords[i][1] * a2
    std::vector<std::array<int, 2>> cell_coords;     // (n1, n2) unit cell indices
    std::vector<std::array<double, 2>> sublattice_offsets;  // delta_alpha within unit cell
    int sites_per_cell = 3;  // For kagome
    
    // Compute minimum-image displacement from site i to site j
    // Returns the actual displacement vector accounting for PBC
    std::array<double, 2> minimum_image_displacement(int i, int j) const {
        if (n_cells_x <= 0 || n_cells_y <= 0 || cell_coords.empty()) {
            // Fallback: direct subtraction (old behavior, may be wrong for PBC)
            return {positions[j][0] - positions[i][0], positions[j][1] - positions[i][1]};
        }
        
        // Get unit cell coordinates
        int n1_i = cell_coords[i][0], n2_i = cell_coords[i][1];
        int n1_j = cell_coords[j][0], n2_j = cell_coords[j][1];
        int sub_i = sublattice[i], sub_j = sublattice[j];
        
        // Compute cell displacement with minimum image convention
        int dn1 = n1_j - n1_i;
        int dn2 = n2_j - n2_i;
        
        // Apply minimum image: shift to range [-N/2, N/2)
        if (dn1 > n_cells_x / 2) dn1 -= n_cells_x;
        else if (dn1 < -n_cells_x / 2) dn1 += n_cells_x;
        if (dn2 > n_cells_y / 2) dn2 -= n_cells_y;
        else if (dn2 < -n_cells_y / 2) dn2 += n_cells_y;
        
        // For odd dimensions, need to handle the boundary case
        // e.g., for N=3: -1, 0, 1 are the valid range
        // dn=2 should map to -1, dn=-2 should map to 1
        if (n_cells_x % 2 == 1 && dn1 == (n_cells_x + 1) / 2) dn1 = -(n_cells_x - 1) / 2;
        if (n_cells_y % 2 == 1 && dn2 == (n_cells_y + 1) / 2) dn2 = -(n_cells_y - 1) / 2;
        
        // Displacement = dn1*a1 + dn2*a2 + (delta_j - delta_i)
        double dx = dn1 * a1[0] + dn2 * a2[0] + sublattice_offsets[sub_j][0] - sublattice_offsets[sub_i][0];
        double dy = dn1 * a1[1] + dn2 * a2[1] + sublattice_offsets[sub_j][1] - sublattice_offsets[sub_i][1];
        
        return {dx, dy};
    }
    
    // Compute bond center position using minimum-image convention
    // For bond (i,j), returns: r_i + displacement(i,j)/2
    // This gives the correct midpoint even for PBC-crossing bonds
    std::array<double, 2> bond_center_pbc(int i, int j) const {
        auto dr = minimum_image_displacement(i, j);
        return {positions[i][0] + 0.5 * dr[0], positions[i][1] + 0.5 * dr[1]};
    }
};

// -----------------------------------------------------------------------------
// Load cluster from files
// -----------------------------------------------------------------------------

Cluster load_cluster(const std::string& cluster_dir) {
    Cluster cluster;
    
    // Find lattice parameters file
    std::string lattice_file;
    for (const auto& suffix : {"_lattice_parameters.dat", "_pbc_lattice_parameters.dat", "_obc_lattice_parameters.dat"}) {
        std::string pattern = cluster_dir + "/*" + suffix;
        // Simple approach: try to open common names
        for (const auto& prefix : {"kagome_bfg_2x2", "kagome_bfg_3x3", "kagome_bfg_4x4", "kagome_bfg_3x2", "kagome_bfg_2x3", "pyrochlore_super_1x1x1_pbc_kramer"}) {
            std::string test_file = cluster_dir + "/" + prefix + suffix;
            std::ifstream test(test_file);
            if (test.good()) {
                lattice_file = test_file;
                break;
            }
        }
        if (!lattice_file.empty()) break;
    }
    
    // Load lattice parameters (if found)
    if (!lattice_file.empty()) {
        std::cout << "Loading lattice parameters from: " << lattice_file << std::endl;
        std::ifstream lat_in(lattice_file);
        std::string line;
        bool reading_unit_vectors = false;
        bool reading_reciprocal_vectors = false;
        bool reading_kpoints = false;
        int vector_count = 0;
        
        while (std::getline(lat_in, line)) {
            if (line.empty() || line[0] == '#') {
                // Check for section headers
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
                    // Parse grid dimensions: "# Unit cells: 3 x 3" or "# Unit cells: 2 x 3"
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
                    if (idx == 0) {
                        cluster.a1 = {x, y};
                    } else if (idx == 1) {
                        cluster.a2 = {x, y};
                    }
                    vector_count++;
                }
            } else if (reading_reciprocal_vectors && vector_count < 2) {
                int idx;
                double kx, ky;
                if (iss >> idx >> kx >> ky) {
                    if (idx == 0) {
                        cluster.b1 = {kx, ky};
                    } else if (idx == 1) {
                        cluster.b2 = {kx, ky};
                    }
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
        
        std::cout << "  Lattice vectors: a1=(" << cluster.a1[0] << "," << cluster.a1[1] << "), "
                  << "a2=(" << cluster.a2[0] << "," << cluster.a2[1] << ")" << std::endl;
        std::cout << "  Reciprocal vectors: b1=(" << cluster.b1[0] << "," << cluster.b1[1] << "), "
                  << "b2=(" << cluster.b2[0] << "," << cluster.b2[1] << ")" << std::endl;
        std::cout << "  Read " << cluster.k_points.size() << " k-points from lattice file" << std::endl;
        
        // Override with extended grid: k = (n1/dim_x)b1 + (n2/dim_y)b2
        // Use grid dimensions from file if available
        if (cluster.n_cells_x > 0 && cluster.n_cells_y > 0) {
            cluster.k_points.clear();
            int n_max_x = cluster.n_cells_x + 1;  // Extended range: 0 to n_cells_x
            int n_max_y = cluster.n_cells_y + 1;  // Extended range: 0 to n_cells_y
            for (int n1 = 0; n1 < n_max_x; ++n1) {
                for (int n2 = 0; n2 < n_max_y; ++n2) {
                    double kx = (static_cast<double>(n1) / cluster.n_cells_x) * cluster.b1[0] + 
                                (static_cast<double>(n2) / cluster.n_cells_y) * cluster.b2[0];
                    double ky = (static_cast<double>(n1) / cluster.n_cells_x) * cluster.b1[1] + 
                                (static_cast<double>(n2) / cluster.n_cells_y) * cluster.b2[1];
                    cluster.k_points.push_back({kx, ky});
                }
            }
            std::cout << "  Grid dimensions: " << cluster.n_cells_x << "x" << cluster.n_cells_y << std::endl;
            std::cout << "  Extended to " << cluster.k_points.size() << " k-points: n1=0.." << (n_max_x-1) 
                      << ", n2=0.." << (n_max_y-1) << std::endl;
        } else {
            // Fallback: infer from site count
            std::cout << "  Warning: Grid dimensions not found in file, using fallback" << std::endl;
            int n_cells = cluster.n_sites / 3;
            int dim = static_cast<int>(std::sqrt(n_cells) + 0.5);
            int n_max = 4;
            cluster.k_points.clear();
            for (int n1 = 0; n1 < n_max; ++n1) {
                for (int n2 = 0; n2 < n_max; ++n2) {
                    double kx = (static_cast<double>(n1) / dim) * cluster.b1[0] + 
                                (static_cast<double>(n2) / dim) * cluster.b2[0];
                    double ky = (static_cast<double>(n1) / dim) * cluster.b1[1] + 
                                (static_cast<double>(n2) / dim) * cluster.b2[1];
                    cluster.k_points.push_back({kx, ky});
                }
            }
            std::cout << "  Extended to " << cluster.k_points.size() << " k-points: (n1,n2)/" << dim << " where n1,n2=0.." << (n_max-1) << std::endl;
        }
    }
    
    // Load positions
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
            cluster.positions[site_id] = {x, y};
            cluster.sublattice[site_id] = sub_idx;
        }
    }
    cluster.n_sites = cluster.positions.size();
    
    // Load NN list
    std::string nn_file;
    for (const auto& prefix : {"kagome_bfg_2x2_pbc", "kagome_bfg_3x3_pbc", "kagome_bfg_4x4_pbc",
                                "kagome_bfg_2x2_obc", "kagome_bfg_3x3_obc", "kagome_bfg_4x4_obc",
                                "kagome_bfg_3x2_pbc", "kagome_bfg_3x2_obc"}) {
        std::string test_file = cluster_dir + "/" + prefix + "_nn_list.dat";
        std::ifstream test(test_file);
        if (test.good()) {
            nn_file = test_file;
            break;
        }
    }
    
    if (nn_file.empty()) {
        // Try to find any nn_list file
        std::ifstream dir_test(cluster_dir);
        // Fallback: construct from positions using distance
        std::cout << "Warning: NN list file not found, constructing from positions" << std::endl;
        double nn_dist = 0.5 + 0.01;  // NN distance on kagome + tolerance
        for (int i = 0; i < cluster.n_sites; ++i) {
            for (int j = i + 1; j < cluster.n_sites; ++j) {
                double dx = cluster.positions[j][0] - cluster.positions[i][0];
                double dy = cluster.positions[j][1] - cluster.positions[i][1];
                double d = std::sqrt(dx * dx + dy * dy);
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
    
    // Fallback: If lattice vectors not loaded, use default kagome values
    if (cluster.a1[0] == 0.0 && cluster.a1[1] == 0.0) {
        std::cout << "Warning: Using default kagome lattice vectors (not loaded from file)" << std::endl;
        cluster.a1 = {1.0, 0.0};
        cluster.a2 = {0.5, std::sqrt(3.0) / 2.0};
    }
    
    // Fallback: If reciprocal vectors not loaded, compute them
    if (cluster.b1[0] == 0.0 && cluster.b1[1] == 0.0) {
        std::cout << "Warning: Computing reciprocal vectors (not loaded from file)" << std::endl;
        double det = cluster.a1[0] * cluster.a2[1] - cluster.a1[1] * cluster.a2[0];
        cluster.b1 = {2.0 * PI * cluster.a2[1] / det, -2.0 * PI * cluster.a2[0] / det};
        cluster.b2 = {-2.0 * PI * cluster.a1[1] / det, 2.0 * PI * cluster.a1[0] / det};
    }
    
    // Fallback: If k-points not loaded, generate them
    if (cluster.k_points.empty()) {
        std::cout << "Warning: Generating k-points grid (not loaded from file)" << std::endl;
        // Assume square grid based on site count
        int n_cells = cluster.n_sites / 3;
        int dim = static_cast<int>(std::sqrt(n_cells) + 0.5);
        if (dim * dim != n_cells) dim = n_cells;  // Non-square
        
        for (int n1 = 0; n1 < dim; ++n1) {
            for (int n2 = 0; n2 < dim; ++n2) {
                double kx = (static_cast<double>(n1) / dim) * cluster.b1[0] + 
                            (static_cast<double>(n2) / dim) * cluster.b2[0];
                double ky = (static_cast<double>(n1) / dim) * cluster.b1[1] + 
                            (static_cast<double>(n2) / dim) * cluster.b2[1];
                cluster.k_points.push_back({kx, ky});
            }
        }
    }
    
    // Compute bond orientations
    for (const auto& [i, j] : cluster.edges_nn) {
        double dx = cluster.positions[j][0] - cluster.positions[i][0];
        double dy = cluster.positions[j][1] - cluster.positions[i][1];
        double angle = std::atan2(dy, dx);
        double angle_deg = std::fmod(angle * 180.0 / PI + 180.0, 180.0);
        
        int orientation;
        if (angle_deg < 30.0 || angle_deg >= 150.0) {
            orientation = 0;  // ~0° (horizontal)
        } else if (angle_deg < 90.0) {
            orientation = 1;  // ~60°
        } else {
            orientation = 2;  // ~120°
        }
        cluster.bond_orientation[{i, j}] = orientation;
        cluster.bond_orientation[{j, i}] = orientation;
    }
    
    // =========================================================================
    // Compute unit cell coordinates for PBC-correct minimum-image displacements
    // Each site is represented as: r = n1*a1 + n2*a2 + delta_sublattice
    // =========================================================================
    cluster.sites_per_cell = 3;  // Kagome lattice
    
    // Standard kagome sublattice offsets (within unit cell)
    cluster.sublattice_offsets = {
        {0.0, 0.0},                        // Sublattice 0
        {0.5, 0.0},                        // Sublattice 1
        {0.25, std::sqrt(3.0) / 4.0}       // Sublattice 2
    };
    
    // Infer grid dimensions if not already set
    if (cluster.n_cells_x <= 0 || cluster.n_cells_y <= 0) {
        int n_cells = cluster.n_sites / cluster.sites_per_cell;
        // Try to factorize (prefer square or close to square)
        int dim = static_cast<int>(std::sqrt(n_cells) + 0.5);
        if (dim * dim == n_cells) {
            cluster.n_cells_x = dim;
            cluster.n_cells_y = dim;
        } else {
            // Try common non-square: 3x2, 4x3, etc.
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
    
    // Compute unit cell coordinates for each site
    // Method: solve r = n1*a1 + n2*a2 + delta for (n1, n2)
    // Since we know sublattice, we can subtract delta first, then solve for n1, n2
    cluster.cell_coords.resize(cluster.n_sites);
    
    double det = cluster.a1[0] * cluster.a2[1] - cluster.a1[1] * cluster.a2[0];
    if (std::abs(det) > 1e-10) {
        for (int site = 0; site < cluster.n_sites; ++site) {
            int sub = cluster.sublattice[site];
            // Position relative to sublattice offset
            double rx = cluster.positions[site][0] - cluster.sublattice_offsets[sub][0];
            double ry = cluster.positions[site][1] - cluster.sublattice_offsets[sub][1];
            
            // Solve: rx = n1*a1[0] + n2*a2[0], ry = n1*a1[1] + n2*a2[1]
            // n1 = (rx*a2[1] - ry*a2[0]) / det
            // n2 = (ry*a1[0] - rx*a1[1]) / det
            double n1_f = (rx * cluster.a2[1] - ry * cluster.a2[0]) / det;
            double n2_f = (ry * cluster.a1[0] - rx * cluster.a1[1]) / det;
            
            // Round to nearest integer (should be exact for valid lattice)
            int n1 = static_cast<int>(std::round(n1_f));
            int n2 = static_cast<int>(std::round(n2_f));
            
            cluster.cell_coords[site] = {n1, n2};
        }
        std::cout << "  Computed unit cell coordinates for PBC-correct displacements" << std::endl;
        std::cout << "  Grid: " << cluster.n_cells_x << " x " << cluster.n_cells_y << " unit cells" << std::endl;
    } else {
        std::cout << "  Warning: Could not compute unit cell coordinates (det=0)" << std::endl;
    }
    
    std::cout << "Loaded cluster: " << cluster.n_sites << " sites, "
              << cluster.edges_nn.size() << " NN bonds" << std::endl;
    
    return cluster;
}

// -----------------------------------------------------------------------------
// Load wavefunction from HDF5
// -----------------------------------------------------------------------------

std::vector<Complex> load_wavefunction(const std::string& filename) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        
        // Try different dataset paths (including nested groups)
        std::vector<std::string> dataset_paths = {
            "eigendata/eigenvector_0",  // Common structure: /eigendata/eigenvector_0
            "eigenvector_0", 
            "eigendata/eigenvectors",
            "eigenvectors", 
            "psi", 
            "wavefunction", 
            "ground_state"
        };
        
        H5::DataSet dataset;
        bool found = false;
        
        for (const auto& path : dataset_paths) {
            try {
                dataset = file.openDataSet(path);
                found = true;
                std::cout << "Found wavefunction in dataset: " << path << std::endl;
                break;
            } catch (...) {
                continue;
            }
        }
        
        if (!found) {
            // List available datasets/groups
            std::cout << "Available objects in file:" << std::endl;
            hsize_t num_objs = file.getNumObjs();
            for (hsize_t i = 0; i < num_objs; ++i) {
                std::string obj_name = file.getObjnameByIdx(i);
                std::cout << "  " << obj_name;
                // If it's a group, list its contents
                try {
                    H5::Group grp = file.openGroup(obj_name);
                    std::cout << "/" << std::endl;
                    for (hsize_t j = 0; j < grp.getNumObjs(); ++j) {
                        std::cout << "    " << grp.getObjnameByIdx(j) << std::endl;
                    }
                } catch (...) {
                    std::cout << std::endl;
                }
            }
            throw std::runtime_error("Wavefunction dataset not found");
        }
        
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data());
        
        hsize_t total_size = 1;
        for (int i = 0; i < rank; ++i) {
            total_size *= dims[i];
        }
        
        // Check if complex or real
        H5::DataType dtype = dataset.getDataType();
        bool is_complex = (dtype.getSize() == 16);  // 2 doubles
        
        std::vector<Complex> psi;
        if (is_complex) {
            // Read as compound type with "real" and "imag" fields
            H5::CompType complex_type(sizeof(Complex));
            complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
            complex_type.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);
            
            psi.resize(total_size);
            try {
                dataset.read(psi.data(), complex_type);
            } catch (...) {
                // Try alternative field names "r" and "i"
                H5::CompType complex_type_alt(sizeof(Complex));
                complex_type_alt.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
                complex_type_alt.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
                try {
                    dataset.read(psi.data(), complex_type_alt);
                } catch (...) {
                    // Fall back to reading as raw doubles
                    std::vector<double> buffer(total_size * 2);
                    dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
                    for (hsize_t i = 0; i < total_size; ++i) {
                        psi[i] = Complex(buffer[2 * i], buffer[2 * i + 1]);
                    }
                }
            }
        } else {
            std::vector<double> real_data(total_size);
            dataset.read(real_data.data(), H5::PredType::NATIVE_DOUBLE);
            psi.resize(total_size);
            for (hsize_t i = 0; i < total_size; ++i) {
                psi[i] = Complex(real_data[i], 0.0);
            }
        }
        
        std::cout << "Loaded wavefunction: " << psi.size() << " amplitudes" << std::endl;
        return psi;
        
    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
    }
}

// -----------------------------------------------------------------------------
// Load TPQ state from HDF5 (finds lowest temperature = highest beta)
// -----------------------------------------------------------------------------

std::pair<std::vector<Complex>, double> load_tpq_state(const std::string& filename, int sample_idx = 0) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        
        // Navigate to TPQ samples
        std::string sample_path = "tpq/samples/sample_" + std::to_string(sample_idx) + "/states";
        H5::Group states_group;
        
        try {
            states_group = file.openGroup(sample_path);
        } catch (...) {
            throw std::runtime_error("TPQ states not found at: " + sample_path);
        }
        
        // Find all beta_* datasets and determine highest beta (lowest T)
        std::vector<std::pair<std::string, double>> beta_states;
        hsize_t num_objs = states_group.getNumObjs();
        
        for (hsize_t i = 0; i < num_objs; ++i) {
            std::string name = states_group.getObjnameByIdx(i);
            if (name.substr(0, 5) == "beta_") {
                try {
                    double beta = std::stod(name.substr(5));
                    beta_states.push_back({name, beta});
                } catch (...) {
                    continue;
                }
            }
        }
        
        if (beta_states.empty()) {
            throw std::runtime_error("No beta_* states found in TPQ data");
        }
        
        // Sort by beta (ascending) and pick the highest (lowest temperature)
        std::sort(beta_states.begin(), beta_states.end(), 
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        std::string best_state = beta_states.back().first;
        double best_beta = beta_states.back().second;
        double temperature = 1.0 / best_beta;
        
        std::cout << "Selected TPQ state: " << best_state 
                  << " (beta=" << best_beta << ", T=" << temperature << ")" << std::endl;
        
        // Load the state
        std::string dataset_path = sample_path + "/" + best_state;
        H5::DataSet dataset = file.openDataSet(dataset_path);
        
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data());
        
        hsize_t total_size = 1;
        for (int i = 0; i < rank; ++i) {
            total_size *= dims[i];
        }
        
        // Read complex data
        H5::CompType complex_type(sizeof(Complex));
        complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);
        
        std::vector<Complex> psi(total_size);
        dataset.read(psi.data(), complex_type);
        
        std::cout << "Loaded TPQ state: " << psi.size() << " amplitudes" << std::endl;
        return {psi, temperature};
        
    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 error loading TPQ: " + std::string(e.getCDetailMsg()));
    }
}

// -----------------------------------------------------------------------------
// Compute S^-_i S^+_j correlation matrix
// This matches TPQ_DSSF "0,0" convention: ⟨(S⁺)†S⁺⟩ = ⟨S⁻S⁺⟩
// -----------------------------------------------------------------------------

std::vector<std::vector<Complex>> compute_smsp_correlations(
    const std::vector<Complex>& psi, 
    int n_sites
) {
    uint64_t n_states = psi.size();
    std::vector<std::vector<Complex>> corr(n_sites, std::vector<Complex>(n_sites, 0.0));
    
    std::cout << "Computing S^-S^+ correlations (matches TPQ_DSSF '0,0')..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        // Thread-local storage
        std::vector<std::vector<Complex>> local_corr(n_sites, std::vector<Complex>(n_sites, 0.0));
        
        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;
            
            for (int i = 0; i < n_sites; ++i) {
                for (int j = 0; j < n_sites; ++j) {
                    if (i == j) {
                        // Diagonal term: S^-_i S^+_i = (1/2 - Sz_i) for spin-1/2
                        // ⟨ψ|S^-_i S^+_i|ψ⟩ = Σ_state |ψ(state)|² × (1/2 - Sz_i(state))
                        // For bit=0 (UP): Sz = +1/2, so S^-S^+ = 0
                        // For bit=1 (DOWN): Sz = -1/2, so S^-S^+ = 1
                        if (get_bit(state, i) == 1) {  // spin down
                            local_corr[i][i] += std::norm(coeff);  // contributes 1
                        }
                        // bit=0 (up) contributes 0, so no else branch needed
                    } else {
                        // Off-diagonal: S^-_i S^+_j |state⟩ with i ≠ j
                        // S^+_j acts first: needs site j to be DOWN (bit=1), flips to UP (bit=0)
                        // S^-_i acts second: needs site i to be UP (bit=0), flips to DOWN (bit=1)
                        // ED convention: bit=0 is UP, bit=1 is DOWN
                        if (get_bit(state, j) == 1 && get_bit(state, i) == 0) {
                            uint64_t new_state = flip_bit(flip_bit(state, i), j);
                            local_corr[i][j] += std::conj(psi[new_state]) * coeff;
                        }
                    }
                }
            }
        }
        
        // Reduce
        #pragma omp critical
        {
            for (int i = 0; i < n_sites; ++i) {
                for (int j = 0; j < n_sites; ++j) {
                    corr[i][j] += local_corr[i][j];
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    
    return corr;
}

// -----------------------------------------------------------------------------
// Compute XY bond expectation ⟨S^+_i S^-_j + S^-_i S^+_j⟩
// -----------------------------------------------------------------------------

std::map<std::pair<int, int>, Complex> compute_xy_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster
) {
    uint64_t n_states = psi.size();
    int n_sites = cluster.n_sites;
    std::map<std::pair<int, int>, Complex> bonds;
    
    // Initialize bonds
    for (const auto& [i, j] : cluster.edges_nn) {
        bonds[{i, j}] = 0.0;
    }
    
    std::cout << "Computing XY bond expectations..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert to vector for OpenMP
    std::vector<std::pair<int, int>> edge_list(cluster.edges_nn.begin(), cluster.edges_nn.end());
    int n_edges = edge_list.size();
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
                
                // S^+_i S^-_j term: j must be UP (bit=0), i must be DOWN (bit=1)
                // ED convention: bit=0 is UP, bit=1 is DOWN
                if (get_bit(state, j) == 0 && get_bit(state, i) == 1) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    local_bonds[e] += std::conj(psi[new_state]) * coeff;
                }
                
                // S^-_i S^+_j term: i must be UP (bit=0), j must be DOWN (bit=1)
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
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    
    return bonds;
}

// -----------------------------------------------------------------------------
// Compute S^+_i S^-_j bond expectation (NOT symmetrized)
// -----------------------------------------------------------------------------

std::map<std::pair<int, int>, Complex> compute_spsm_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster
) {
    uint64_t n_states = psi.size();
    std::map<std::pair<int, int>, Complex> bonds;
    
    for (const auto& [i, j] : cluster.edges_nn) {
        bonds[{i, j}] = 0.0;
    }
    
    std::cout << "Computing S+S- bond expectations..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::pair<int, int>> edge_list(cluster.edges_nn.begin(), cluster.edges_nn.end());
    int n_edges = edge_list.size();
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
                
                // S^+_i S^-_j: i must be DOWN (bit=1), j must be UP (bit=0)
                // S^+ raises: |↓⟩ → |↑⟩, S^- lowers: |↑⟩ → |↓⟩
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
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    
    return bonds;
}

// -----------------------------------------------------------------------------
// Compute S^z_i S^z_j bond expectation
// -----------------------------------------------------------------------------

std::map<std::pair<int, int>, double> compute_szsz_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster
) {
    uint64_t n_states = psi.size();
    std::map<std::pair<int, int>, double> bonds;
    
    for (const auto& [i, j] : cluster.edges_nn) {
        bonds[{i, j}] = 0.0;
    }
    
    std::cout << "Computing SzSz bond expectations..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::pair<int, int>> edge_list(cluster.edges_nn.begin(), cluster.edges_nn.end());
    int n_edges = edge_list.size();
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
                
                // S^z_i S^z_j is diagonal
                double sz_i = sz_value(state, i);  // ±0.5
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
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    
    return bonds;
}

// -----------------------------------------------------------------------------
// Compute full Heisenberg bond S_i · S_j = S^z_i S^z_j + (1/2)(S^+_i S^-_j + S^-_i S^+_j)
// -----------------------------------------------------------------------------

std::map<std::pair<int, int>, double> compute_heisenberg_bond_expectations(
    const std::map<std::pair<int, int>, double>& szsz_bonds,
    const std::map<std::pair<int, int>, Complex>& xy_bonds
) {
    std::map<std::pair<int, int>, double> bonds;
    
    for (const auto& [edge, szsz] : szsz_bonds) {
        double xy = 0.0;
        if (xy_bonds.count(edge)) {
            xy = std::real(xy_bonds.at(edge));
        }
        // S·S = SzSz + (1/2)(S+S- + S-S+) = SzSz + (1/2)*XY
        bonds[edge] = szsz + 0.5 * xy;
    }
    
    return bonds;
}

// -----------------------------------------------------------------------------
// Memory-efficient structure factor computation
// Computes ⟨ψ|D†(q₁)D(q₂)|ψ⟩ directly without materializing O(Hilbert) vectors
// This avoids the O(n_threads * n_states) memory explosion
// -----------------------------------------------------------------------------

struct DimerSFResult {
    Complex overlap;      // ⟨D(q₁)ψ|D(q₂)ψ⟩
    Complex expect_q1;    // ⟨D(q₁)⟩
    Complex expect_q2;    // ⟨D(q₂)⟩
};

// Compute dimer structure factor directly: S_D(q) = ⟨D†(q)D(q)⟩ - |⟨D(q)⟩|²
// For XY dimer operator D_b = S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ
// This is O(N_bonds * Hilbert) time and O(1) extra memory per thread
DimerSFResult compute_dimer_sf_direct(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
) {
    uint64_t n_states = psi.size();
    int n_bonds = bonds.size();
    
    // Precompute phases
    std::vector<Complex> phases(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        double phase_arg = q[0] * bond_centers[b][0] + q[1] * bond_centers[b][1];
        phases[b] = std::exp(I * phase_arg);
    }
    
    double overlap_real = 0.0, overlap_imag = 0.0;
    double expect_real = 0.0, expect_imag = 0.0;
    
    // Key insight: ⟨D(q)ψ|D(q)ψ⟩ = Σ_{b,b'} exp(i(q·r_{b'} - q·r_b)) × ⟨ψ|D_b D_{b'}|ψ⟩
    // where D_b D_{b'} is a 4-spin operator (or 2-spin if b = b')
    // We compute this by iterating over |ψ⟩ once, collecting all contributions
    
    #pragma omp parallel reduction(+:overlap_real, overlap_imag, expect_real, expect_imag)
    {
        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            double prob = std::norm(coeff);
            if (prob < 1e-30) continue;
            
            // For each pair of bonds (b, b'), compute contribution to ⟨D_b† D_{b'}⟩
            // D_b = S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ (XY dimer)
            
            for (int b1 = 0; b1 < n_bonds; ++b1) {
                int i1 = bonds[b1].first, j1 = bonds[b1].second;
                int si1 = get_bit(state, i1), sj1 = get_bit(state, j1);
                Complex phase1_conj = std::conj(phases[b1]);
                
                for (int b2 = 0; b2 < n_bonds; ++b2) {
                    int i2 = bonds[b2].first, j2 = bonds[b2].second;
                    int si2 = get_bit(state, i2), sj2 = get_bit(state, j2);
                    Complex phase2 = phases[b2];
                    Complex phase_factor = phase1_conj * phase2;
                    
                    // Compute ⟨ψ|D_b1 D_b2|ψ⟩ contribution from |state⟩
                    // D_b1 D_b2 has 4 terms each (S⁺S⁻ or S⁻S⁺) × (S⁺S⁻ or S⁻S⁺)
                    // Need to check if b1 and b2 share sites (special handling)
                    
                    bool share = (i1 == i2 || i1 == j2 || j1 == i2 || j1 == j2);
                    
                    if (!share) {
                        // Independent bonds: 4 terms
                        // Term 1: (S⁺_{i1}S⁻_{j1})(S⁺_{i2}S⁻_{j2})
                        if (sj1 == 0 && si1 == 1 && sj2 == 0 && si2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        // Term 2: (S⁺_{i1}S⁻_{j1})(S⁻_{i2}S⁺_{j2})
                        if (sj1 == 0 && si1 == 1 && si2 == 0 && sj2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        // Term 3: (S⁻_{i1}S⁺_{j1})(S⁺_{i2}S⁻_{j2})
                        if (si1 == 0 && sj1 == 1 && sj2 == 0 && si2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        // Term 4: (S⁻_{i1}S⁺_{j1})(S⁻_{i2}S⁺_{j2})
                        if (si1 == 0 && sj1 == 1 && si2 == 0 && sj2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                    } else if (b1 == b2) {
                        // Same bond: D_b D_b = (S⁺S⁻ + S⁻S⁺)² 
                        // = S⁺S⁻S⁺S⁻ + S⁺S⁻S⁻S⁺ + S⁻S⁺S⁺S⁻ + S⁻S⁺S⁻S⁺
                        // Acting on same sites, this simplifies based on current state
                        // Diagonal contributions (same in/out state)
                        if ((si1 == 1 && sj1 == 0) || (si1 == 0 && sj1 == 1)) {
                            // D_b|state⟩ = |flipped⟩, D_b|flipped⟩ = |state⟩
                            // So D_b D_b|state⟩ = |state⟩
                            Complex c = prob * phase_factor;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                    } else {
                        // Overlapping but different bonds - careful handling
                        // This is more complex; for now use the general approach
                        // by checking each of the 4 operator combinations
                        std::array<int, 4> sites = {i1, j1, i2, j2};
                        std::set<int> unique_sites(sites.begin(), sites.end());
                        
                        // Build the 4 terms explicitly
                        // Term 1: S⁺_{i1}S⁻_{j1}S⁺_{i2}S⁻_{j2}
                        {
                            uint64_t s = state;
                            bool valid = true;
                            // Apply operators right to left: S⁻_{j2}, S⁺_{i2}, S⁻_{j1}, S⁺_{i1}
                            if (get_bit(s, j2) == 0) { s = flip_bit(s, j2); } else valid = false;
                            if (valid && get_bit(s, i2) == 1) { s = flip_bit(s, i2); } else valid = false;
                            if (valid && get_bit(s, j1) == 0) { s = flip_bit(s, j1); } else valid = false;
                            if (valid && get_bit(s, i1) == 1) { s = flip_bit(s, i1); } else valid = false;
                            if (valid) {
                                Complex c = std::conj(psi[s]) * coeff * phase_factor;
                                overlap_real += c.real(); overlap_imag += c.imag();
                            }
                        }
                        // Term 2: S⁺_{i1}S⁻_{j1}S⁻_{i2}S⁺_{j2}
                        {
                            uint64_t s = state;
                            bool valid = true;
                            if (get_bit(s, j2) == 1) { s = flip_bit(s, j2); } else valid = false;
                            if (valid && get_bit(s, i2) == 0) { s = flip_bit(s, i2); } else valid = false;
                            if (valid && get_bit(s, j1) == 0) { s = flip_bit(s, j1); } else valid = false;
                            if (valid && get_bit(s, i1) == 1) { s = flip_bit(s, i1); } else valid = false;
                            if (valid) {
                                Complex c = std::conj(psi[s]) * coeff * phase_factor;
                                overlap_real += c.real(); overlap_imag += c.imag();
                            }
                        }
                        // Term 3: S⁻_{i1}S⁺_{j1}S⁺_{i2}S⁻_{j2}
                        {
                            uint64_t s = state;
                            bool valid = true;
                            if (get_bit(s, j2) == 0) { s = flip_bit(s, j2); } else valid = false;
                            if (valid && get_bit(s, i2) == 1) { s = flip_bit(s, i2); } else valid = false;
                            if (valid && get_bit(s, j1) == 1) { s = flip_bit(s, j1); } else valid = false;
                            if (valid && get_bit(s, i1) == 0) { s = flip_bit(s, i1); } else valid = false;
                            if (valid) {
                                Complex c = std::conj(psi[s]) * coeff * phase_factor;
                                overlap_real += c.real(); overlap_imag += c.imag();
                            }
                        }
                        // Term 4: S⁻_{i1}S⁺_{j1}S⁻_{i2}S⁺_{j2}
                        {
                            uint64_t s = state;
                            bool valid = true;
                            if (get_bit(s, j2) == 1) { s = flip_bit(s, j2); } else valid = false;
                            if (valid && get_bit(s, i2) == 0) { s = flip_bit(s, i2); } else valid = false;
                            if (valid && get_bit(s, j1) == 1) { s = flip_bit(s, j1); } else valid = false;
                            if (valid && get_bit(s, i1) == 0) { s = flip_bit(s, i1); } else valid = false;
                            if (valid) {
                                Complex c = std::conj(psi[s]) * coeff * phase_factor;
                                overlap_real += c.real(); overlap_imag += c.imag();
                            }
                        }
                    }
                }
                
                // Expectation value ⟨D(q)⟩: diagonal terms only from D_b|state⟩ → flipped
                // D_b|state⟩ = |new_state⟩ iff state has appropriate spins
                if (sj1 == 0 && si1 == 1) {
                    // S⁺_{i1}S⁻_{j1} term
                    uint64_t ns = flip_bit(flip_bit(state, i1), j1);
                    Complex c = std::conj(psi[ns]) * coeff * phases[b1];
                    expect_real += c.real(); expect_imag += c.imag();
                }
                if (si1 == 0 && sj1 == 1) {
                    // S⁻_{i1}S⁺_{j1} term
                    uint64_t ns = flip_bit(flip_bit(state, i1), j1);
                    Complex c = std::conj(psi[ns]) * coeff * phases[b1];
                    expect_real += c.real(); expect_imag += c.imag();
                }
            }
        }
    }
    
    return {Complex(overlap_real, overlap_imag), 
            Complex(expect_real, expect_imag), 
            Complex(expect_real, expect_imag)};  // q1 = q2 case
}

// Memory-efficient Heisenberg dimer structure factor
// D_b = S_i·S_j = SzSz + (1/2)(S⁺S⁻ + S⁻S⁺)
DimerSFResult compute_heisenberg_sf_direct(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
) {
    uint64_t n_states = psi.size();
    int n_bonds = bonds.size();
    
    std::vector<Complex> phases(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        double phase_arg = q[0] * bond_centers[b][0] + q[1] * bond_centers[b][1];
        phases[b] = std::exp(I * phase_arg);
    }
    
    double overlap_real = 0.0, overlap_imag = 0.0;
    double expect_real = 0.0, expect_imag = 0.0;
    
    #pragma omp parallel reduction(+:overlap_real, overlap_imag, expect_real, expect_imag)
    {
        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            double prob = std::norm(coeff);
            if (prob < 1e-30) continue;
            
            for (int b1 = 0; b1 < n_bonds; ++b1) {
                int i1 = bonds[b1].first, j1 = bonds[b1].second;
                double sz_i1 = get_bit(state, i1) ? -0.5 : 0.5;
                double sz_j1 = get_bit(state, j1) ? -0.5 : 0.5;
                double szsz1 = sz_i1 * sz_j1;
                Complex phase1_conj = std::conj(phases[b1]);
                
                // Expectation: ⟨D(q)⟩ diagonal part (SzSz)
                Complex exp_contrib = phases[b1] * szsz1 * prob;
                expect_real += exp_contrib.real();
                expect_imag += exp_contrib.imag();
                
                // XY part of expectation
                int si1 = get_bit(state, i1), sj1 = get_bit(state, j1);
                if (sj1 == 0 && si1 == 1) {
                    uint64_t ns = flip_bit(flip_bit(state, i1), j1);
                    Complex c = std::conj(psi[ns]) * coeff * phases[b1] * 0.5;
                    expect_real += c.real(); expect_imag += c.imag();
                }
                if (si1 == 0 && sj1 == 1) {
                    uint64_t ns = flip_bit(flip_bit(state, i1), j1);
                    Complex c = std::conj(psi[ns]) * coeff * phases[b1] * 0.5;
                    expect_real += c.real(); expect_imag += c.imag();
                }
                
                for (int b2 = 0; b2 < n_bonds; ++b2) {
                    int i2 = bonds[b2].first, j2 = bonds[b2].second;
                    double sz_i2 = get_bit(state, i2) ? -0.5 : 0.5;
                    double sz_j2 = get_bit(state, j2) ? -0.5 : 0.5;
                    double szsz2 = sz_i2 * sz_j2;
                    Complex phase2 = phases[b2];
                    Complex phase_factor = phase1_conj * phase2;
                    
                    // SzSz × SzSz diagonal contribution
                    Complex diag_contrib = phase_factor * szsz1 * szsz2 * prob;
                    overlap_real += diag_contrib.real();
                    overlap_imag += diag_contrib.imag();
                    
                    // Mixed and XY×XY terms need more careful handling
                    // For simplicity, compute key terms
                    bool share = (i1 == i2 || i1 == j2 || j1 == i2 || j1 == j2);
                    int si2 = get_bit(state, i2), sj2 = get_bit(state, j2);
                    
                    if (!share) {
                        // SzSz × XY terms (where XY on b2)
                        if (sj2 == 0 && si2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(state, i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor * szsz1 * 0.5;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        if (si2 == 0 && sj2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(state, i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor * szsz1 * 0.5;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        
                        // XY × SzSz terms (where XY on b1)
                        if (sj1 == 0 && si1 == 1) {
                            uint64_t ns = flip_bit(flip_bit(state, i1), j1);
                            double sz_ns_i2 = get_bit(ns, i2) ? -0.5 : 0.5;
                            double sz_ns_j2 = get_bit(ns, j2) ? -0.5 : 0.5;
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor * sz_ns_i2 * sz_ns_j2 * 0.5;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        if (si1 == 0 && sj1 == 1) {
                            uint64_t ns = flip_bit(flip_bit(state, i1), j1);
                            double sz_ns_i2 = get_bit(ns, i2) ? -0.5 : 0.5;
                            double sz_ns_j2 = get_bit(ns, j2) ? -0.5 : 0.5;
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor * sz_ns_i2 * sz_ns_j2 * 0.5;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        
                        // XY × XY terms (4 combinations)
                        // (S⁺S⁻)_{b1} × (S⁺S⁻)_{b2}
                        if (sj1 == 0 && si1 == 1 && sj2 == 0 && si2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor * 0.25;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        if (sj1 == 0 && si1 == 1 && si2 == 0 && sj2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor * 0.25;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        if (si1 == 0 && sj1 == 1 && sj2 == 0 && si2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor * 0.25;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        if (si1 == 0 && sj1 == 1 && si2 == 0 && sj2 == 1) {
                            uint64_t ns = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                            Complex c = std::conj(psi[ns]) * coeff * phase_factor * 0.25;
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                    } else if (b1 == b2) {
                        // Same bond: D_b² terms
                        // XY part: already diagonal if flippable
                        if ((si1 == 1 && sj1 == 0) || (si1 == 0 && sj1 == 1)) {
                            Complex c = prob * phase_factor * 0.25;  // (1/2)² × 1
                            overlap_real += c.real(); overlap_imag += c.imag();
                        }
                        // Cross SzSz × XY terms average out for same bond
                    }
                    // Overlapping different bonds: more complex, skip for now
                }
            }
        }
    }
    
    return {Complex(overlap_real, overlap_imag),
            Complex(expect_real, expect_imag),
            Complex(expect_real, expect_imag)};
}

// Flag to enable memory-efficient mode (controlled by system memory check)
static bool g_use_memory_efficient_mode = false;
static uint64_t g_memory_threshold = 1ULL << 24;  // ~16M states threshold

void set_memory_efficient_mode(uint64_t n_states) {
    // Enable memory-efficient mode if Hilbert space is large
    // Threshold: when thread-local arrays would exceed ~4GB total
    int n_threads = 1;
    #ifdef _OPENMP
    n_threads = omp_get_max_threads();
    #endif
    
    uint64_t mem_per_thread = n_states * sizeof(Complex);  // 16 bytes per complex
    uint64_t total_thread_mem = n_threads * mem_per_thread;
    
    // Enable if total thread-local memory would exceed 4GB
    g_use_memory_efficient_mode = (total_thread_mem > 4ULL * 1024 * 1024 * 1024);
    
    if (g_use_memory_efficient_mode) {
        std::cout << "[Memory] Enabling memory-efficient mode for " << n_states 
                  << " states (" << n_threads << " threads would need "
                  << (total_thread_mem / (1024*1024*1024)) << " GB)" << std::endl;
    }
}

// -----------------------------------------------------------------------------
// Apply Fourier-transformed XY dimer operator: D_α(q)|ψ⟩ = Σ_{b∈α} exp(iq·r_b) D_b|ψ⟩
// where D_b = S⁺ᵢS⁻ⱼ + S⁻ᵢS⁺ⱼ
// This is O(N_bonds * Hilbert) instead of O(N_bonds² * Hilbert) for pairwise correlations
// -----------------------------------------------------------------------------

std::vector<Complex> apply_dimer_fourier(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,  // Bonds in this orientation
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
) {
    uint64_t n_states = psi.size();
    std::vector<Complex> result(n_states, 0.0);
    int n_bonds = bonds.size();
    
    // Precompute phases
    std::vector<Complex> phases(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        double phase_arg = q[0] * bond_centers[b][0] + q[1] * bond_centers[b][1];
        phases[b] = std::exp(I * phase_arg);
    }
    
    if (g_use_memory_efficient_mode) {
        // Memory-efficient mode: use atomic updates instead of thread-local arrays
        // Split result into real/imag parts for atomic operations
        std::vector<double> result_real(n_states, 0.0);
        std::vector<double> result_imag(n_states, 0.0);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;
            
            for (int b = 0; b < n_bonds; ++b) {
                int i = bonds[b].first;
                int j = bonds[b].second;
                int s_i = get_bit(state, i);
                int s_j = get_bit(state, j);
                Complex phase = phases[b];
                
                // S⁺ᵢS⁻ⱼ: needs i=DOWN(1), j=UP(0)
                if (s_i == 1 && s_j == 0) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    Complex val = phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
                
                // S⁻ᵢS⁺ⱼ: needs i=UP(0), j=DOWN(1)
                if (s_i == 0 && s_j == 1) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    Complex val = phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
            }
        }
        
        // Combine real and imaginary parts
        for (uint64_t s = 0; s < n_states; ++s) {
            result[s] = Complex(result_real[s], result_imag[s]);
        }
    } else {
        // Original mode: thread-local arrays (fast but memory-hungry)
        #pragma omp parallel
        {
            std::vector<Complex> local_result(n_states, 0.0);
            
            #pragma omp for schedule(dynamic, 1024)
            for (uint64_t state = 0; state < n_states; ++state) {
                Complex coeff = psi[state];
                if (std::abs(coeff) < 1e-15) continue;
                
                for (int b = 0; b < n_bonds; ++b) {
                    int i = bonds[b].first;
                    int j = bonds[b].second;
                    int s_i = get_bit(state, i);
                    int s_j = get_bit(state, j);
                    Complex phase = phases[b];
                    
                    // S⁺ᵢS⁻ⱼ: needs i=DOWN(1), j=UP(0)
                    if (s_i == 1 && s_j == 0) {
                        uint64_t new_state = flip_bit(flip_bit(state, i), j);
                        local_result[new_state] += phase * coeff;
                    }
                    
                    // S⁻ᵢS⁺ⱼ: needs i=UP(0), j=DOWN(1)
                    if (s_i == 0 && s_j == 1) {
                        uint64_t new_state = flip_bit(flip_bit(state, i), j);
                        local_result[new_state] += phase * coeff;
                    }
                }
            }
            
            #pragma omp critical
            {
                for (uint64_t s = 0; s < n_states; ++s) {
                    result[s] += local_result[s];
                }
            }
        }
    }
    
    return result;
}

// -----------------------------------------------------------------------------
// Apply Fourier-transformed Heisenberg dimer operator: D_α(q)|ψ⟩
// D_b = S_i·S_j = SzSz + (1/2)(S⁺S⁻ + S⁻S⁺)
// For structure factor we need the fluctuation part, so we apply the full operator
// and subtract the mean later.
// Returns {D(q)|ψ⟩, ⟨D(q)⟩} where D(q) = Σ_b exp(iq·r_b) D_b
// -----------------------------------------------------------------------------

std::pair<std::vector<Complex>, Complex> apply_heisenberg_dimer_fourier(
    const std::vector<Complex>& psi,
    const std::vector<std::pair<int, int>>& bonds,
    const std::vector<std::array<double, 2>>& bond_centers,
    const std::array<double, 2>& q
) {
    uint64_t n_states = psi.size();
    std::vector<Complex> result(n_states, 0.0);
    int n_bonds = bonds.size();
    
    // Precompute phases
    std::vector<Complex> phases(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        double phase_arg = q[0] * bond_centers[b][0] + q[1] * bond_centers[b][1];
        phases[b] = std::exp(I * phase_arg);
    }
    
    double expect_real = 0.0, expect_imag = 0.0;
    
    if (g_use_memory_efficient_mode) {
        // Memory-efficient mode: use atomic updates
        std::vector<double> result_real(n_states, 0.0);
        std::vector<double> result_imag(n_states, 0.0);
        
        #pragma omp parallel for reduction(+:expect_real, expect_imag) schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;
            
            for (int b = 0; b < n_bonds; ++b) {
                int i = bonds[b].first;
                int j = bonds[b].second;
                int s_i = get_bit(state, i);
                int s_j = get_bit(state, j);
                Complex phase = phases[b];
                
                // SzSz term: diagonal
                double sz_i = s_i ? -0.5 : 0.5;
                double sz_j = s_j ? -0.5 : 0.5;
                double szsz = sz_i * sz_j;
                
                Complex diag_contrib = phase * szsz * coeff;
                #pragma omp atomic
                result_real[state] += diag_contrib.real();
                #pragma omp atomic
                result_imag[state] += diag_contrib.imag();
                
                Complex exp_contrib = phase * szsz * std::norm(coeff);
                expect_real += exp_contrib.real();
                expect_imag += exp_contrib.imag();
                
                // (1/2) S⁺ᵢS⁻ⱼ: needs i=DOWN(1), j=UP(0)
                if (s_i == 1 && s_j == 0) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    Complex val = 0.5 * phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
                
                // (1/2) S⁻ᵢS⁺ⱼ: needs i=UP(0), j=DOWN(1)
                if (s_i == 0 && s_j == 1) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    Complex val = 0.5 * phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
            }
        }
        
        for (uint64_t s = 0; s < n_states; ++s) {
            result[s] = Complex(result_real[s], result_imag[s]);
        }
    } else {
        // Original mode with thread-local arrays
        #pragma omp parallel reduction(+:expect_real, expect_imag)
        {
            std::vector<Complex> local_result(n_states, 0.0);
            double local_expect_real = 0.0, local_expect_imag = 0.0;
            
            #pragma omp for schedule(dynamic, 1024)
            for (uint64_t state = 0; state < n_states; ++state) {
                Complex coeff = psi[state];
                if (std::abs(coeff) < 1e-15) continue;
                
                for (int b = 0; b < n_bonds; ++b) {
                    int i = bonds[b].first;
                    int j = bonds[b].second;
                    int s_i = get_bit(state, i);
                    int s_j = get_bit(state, j);
                    Complex phase = phases[b];
                    
                    // SzSz term: diagonal, contributes (s_i - 0.5)(s_j - 0.5) = ±0.25
                    double sz_i = s_i ? -0.5 : 0.5;  // bit=0 is UP (+0.5)
                    double sz_j = s_j ? -0.5 : 0.5;
                    double szsz = sz_i * sz_j;
                    
                    // Diagonal contribution to D(q)|ψ⟩
                    local_result[state] += phase * szsz * coeff;
                    
                    // Contribution to ⟨D(q)⟩
                    Complex contrib = phase * szsz * std::norm(coeff);
                    local_expect_real += contrib.real();
                    local_expect_imag += contrib.imag();
                    
                    // (1/2) S⁺ᵢS⁻ⱼ: needs i=DOWN(1), j=UP(0)
                    if (s_i == 1 && s_j == 0) {
                        uint64_t new_state = flip_bit(flip_bit(state, i), j);
                        local_result[new_state] += 0.5 * phase * coeff;
                    }
                    
                    // (1/2) S⁻ᵢS⁺ⱼ: needs i=UP(0), j=DOWN(1)
                    if (s_i == 0 && s_j == 1) {
                        uint64_t new_state = flip_bit(flip_bit(state, i), j);
                        local_result[new_state] += 0.5 * phase * coeff;
                    }
                }
            }
            
            expect_real += local_expect_real;
            expect_imag += local_expect_imag;
            
            #pragma omp critical
            {
                for (uint64_t s = 0; s < n_states; ++s) {
                    result[s] += local_result[s];
                }
            }
        }
    }
    
    // ⟨D(q)⟩ = ⟨ψ|D(q)|ψ⟩
    Complex expect = Complex(0, 0);
    for (uint64_t s = 0; s < n_states; ++s) {
        expect += std::conj(psi[s]) * result[s];
    }
    
    return {result, expect};
}

// -----------------------------------------------------------------------------
// Apply Fourier-transformed bowtie resonance operator: P_α(q)|ψ⟩
// P_bt = S⁺₁S⁻₂S⁺₃S⁻₄ + S⁻₁S⁺₂S⁻₃S⁺₄ (4-spin ring flip on outer corners)
// -----------------------------------------------------------------------------

struct BowtieData {
    int s1, s2, s3, s4;  // Outer corner sites
    std::array<double, 2> center;
};

std::vector<Complex> apply_bowtie_fourier(
    const std::vector<Complex>& psi,
    const std::vector<BowtieData>& bowties,  // Bowties in this orientation
    const std::array<double, 2>& q
) {
    uint64_t n_states = psi.size();
    std::vector<Complex> result(n_states, 0.0);
    int n_bowties = bowties.size();
    
    // Precompute phases
    std::vector<Complex> phases(n_bowties);
    for (int p = 0; p < n_bowties; ++p) {
        double phase_arg = q[0] * bowties[p].center[0] + q[1] * bowties[p].center[1];
        phases[p] = std::exp(I * phase_arg);
    }
    
    if (g_use_memory_efficient_mode) {
        // Memory-efficient mode: use atomic updates
        std::vector<double> result_real(n_states, 0.0);
        std::vector<double> result_imag(n_states, 0.0);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;
            
            for (int p = 0; p < n_bowties; ++p) {
                int s1 = bowties[p].s1;
                int s2 = bowties[p].s2;
                int s3 = bowties[p].s3;
                int s4 = bowties[p].s4;
                
                int b1 = get_bit(state, s1);
                int b2 = get_bit(state, s2);
                int b3 = get_bit(state, s3);
                int b4 = get_bit(state, s4);
                
                Complex phase = phases[p];
                
                // S⁺₁S⁻₂S⁺₃S⁻₄: needs s1=DOWN(1), s2=UP(0), s3=DOWN(1), s4=UP(0)
                if (b1 == 1 && b2 == 0 && b3 == 1 && b4 == 0) {
                    uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                    Complex val = phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
                
                // S⁻₁S⁺₂S⁻₃S⁺₄: needs s1=UP(0), s2=DOWN(1), s3=UP(0), s4=DOWN(1)
                if (b1 == 0 && b2 == 1 && b3 == 0 && b4 == 1) {
                    uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                    Complex val = phase * coeff;
                    #pragma omp atomic
                    result_real[new_state] += val.real();
                    #pragma omp atomic
                    result_imag[new_state] += val.imag();
                }
            }
        }
        
        for (uint64_t s = 0; s < n_states; ++s) {
            result[s] = Complex(result_real[s], result_imag[s]);
        }
    } else {
        // Original mode with thread-local arrays
        #pragma omp parallel
        {
            std::vector<Complex> local_result(n_states, 0.0);
            
            #pragma omp for schedule(dynamic, 1024)
            for (uint64_t state = 0; state < n_states; ++state) {
                Complex coeff = psi[state];
                if (std::abs(coeff) < 1e-15) continue;
                
                for (int p = 0; p < n_bowties; ++p) {
                    int s1 = bowties[p].s1;
                    int s2 = bowties[p].s2;
                    int s3 = bowties[p].s3;
                    int s4 = bowties[p].s4;
                    
                    int b1 = get_bit(state, s1);
                    int b2 = get_bit(state, s2);
                    int b3 = get_bit(state, s3);
                    int b4 = get_bit(state, s4);
                    
                    Complex phase = phases[p];
                    
                    // S⁺₁S⁻₂S⁺₃S⁻₄: needs s1=DOWN(1), s2=UP(0), s3=DOWN(1), s4=UP(0)
                    if (b1 == 1 && b2 == 0 && b3 == 1 && b4 == 0) {
                        uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                        local_result[new_state] += phase * coeff;
                    }
                    
                    // S⁻₁S⁺₂S⁻₃S⁺₄: needs s1=UP(0), s2=DOWN(1), s3=UP(0), s4=DOWN(1)
                    if (b1 == 0 && b2 == 1 && b3 == 0 && b4 == 1) {
                        uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                        local_result[new_state] += phase * coeff;
                    }
                }
            }
            
            #pragma omp critical
            {
                for (uint64_t s = 0; s < n_states; ++s) {
                    result[s] += local_result[s];
                }
            }
        }
    }
    
    return result;
}

// -----------------------------------------------------------------------------
// Compute dimer-dimer correlation ⟨D_b1 D_b2⟩ for proper VBS order
// D = S^+_i S^-_j + S^-_i S^+_j (XY dimer operator)
// This is a 4-site spin correlation
// -----------------------------------------------------------------------------

Complex compute_dimer_dimer_correlation(
    const std::vector<Complex>& psi,
    int i1, int j1, int i2, int j2
) {
    uint64_t n_states = psi.size();
    double result_real = 0.0;
    double result_imag = 0.0;
    
    #pragma omp parallel reduction(+:result_real,result_imag)
    {
        double local_real = 0.0;
        double local_imag = 0.0;
        
        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;
            
            int s_i1 = get_bit(state, i1);
            int s_j1 = get_bit(state, j1);
            int s_i2 = get_bit(state, i2);
            int s_j2 = get_bit(state, j2);
            
            // Term 1: S^+_{i1} S^-_{j1} S^+_{i2} S^-_{j2}
            // S^-_j acts first (needs j=UP=0), S^+_i acts second (needs i=DOWN=1)
            // ED convention: bit=0 is UP, bit=1 is DOWN
            if (s_j1 == 0 && s_i1 == 1 && s_j2 == 0 && s_i2 == 1) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, j1);  // j1: up(0) -> down(1)
                new_state = flip_bit(new_state, i1);  // i1: down(1) -> up(0)
                new_state = flip_bit(new_state, j2);  // j2: up(0) -> down(1)
                new_state = flip_bit(new_state, i2);  // i2: down(1) -> up(0)
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            
            // Term 2: S^+_{i1} S^-_{j1} S^-_{i2} S^+_{j2}
            // S^+S^-: j1=UP(0), i1=DOWN(1); S^-S^+: i2=UP(0), j2=DOWN(1)
            if (s_j1 == 0 && s_i1 == 1 && s_i2 == 0 && s_j2 == 1) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, j1);
                new_state = flip_bit(new_state, i1);
                new_state = flip_bit(new_state, i2);
                new_state = flip_bit(new_state, j2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            
            // Term 3: S^-_{i1} S^+_{j1} S^+_{i2} S^-_{j2}
            // S^-S^+: i1=UP(0), j1=DOWN(1); S^+S^-: j2=UP(0), i2=DOWN(1)
            if (s_i1 == 0 && s_j1 == 1 && s_j2 == 0 && s_i2 == 1) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, i1);
                new_state = flip_bit(new_state, j1);
                new_state = flip_bit(new_state, j2);
                new_state = flip_bit(new_state, i2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            
            // Term 4: S^-_{i1} S^+_{j1} S^-_{i2} S^+_{j2}
            // Both S^-S^+: i1=UP(0), j1=DOWN(1), i2=UP(0), j2=DOWN(1)
            if (s_i1 == 0 && s_j1 == 1 && s_i2 == 0 && s_j2 == 1) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, i1);
                new_state = flip_bit(new_state, j1);
                new_state = flip_bit(new_state, i2);
                new_state = flip_bit(new_state, j2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
        }
        
        result_real += local_real;
        result_imag += local_imag;
    }
    
    return Complex(result_real, result_imag);
}

// -----------------------------------------------------------------------------
// Compute Heisenberg dimer-dimer correlation ⟨(S_i·S_j)(S_k·S_l)⟩
// This is a proper 4-site correlation for Heisenberg VBS order
// S·S = SzSz + (1/2)(S+S- + S-S+)
// -----------------------------------------------------------------------------

double compute_heisenberg_dimer_dimer_correlation(
    const std::vector<Complex>& psi,
    int i1, int j1, int i2, int j2
) {
    uint64_t n_states = psi.size();
    double result = 0.0;
    
    #pragma omp parallel reduction(+:result)
    {
        double local_result = 0.0;
        
        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            double prob = std::norm(coeff);
            if (prob < 1e-30) continue;
            
            int s_i1 = get_bit(state, i1);
            int s_j1 = get_bit(state, j1);
            int s_i2 = get_bit(state, i2);
            int s_j2 = get_bit(state, j2);
            
            // Sz values: bit=0 -> +1/2, bit=1 -> -1/2
            double sz_i1 = s_i1 ? -0.5 : 0.5;
            double sz_j1 = s_j1 ? -0.5 : 0.5;
            double sz_i2 = s_i2 ? -0.5 : 0.5;
            double sz_j2 = s_j2 ? -0.5 : 0.5;
            
            // ========================================================
            // (S_i1·S_j1)(S_i2·S_j2) expansion:
            // = (SzSz + 1/2(S+S- + S-S+))_bond1 × (SzSz + 1/2(S+S- + S-S+))_bond2
            // 
            // Diagonal terms (SzSz)×(SzSz):
            double szsz_1 = sz_i1 * sz_j1;
            double szsz_2 = sz_i2 * sz_j2;
            local_result += prob * szsz_1 * szsz_2;
            
            // Cross terms (SzSz)×(1/2 XY) and (1/2 XY)×(SzSz):
            // These require off-diagonal matrix elements on one bond only
            
            // (SzSz)_1 × (1/2)(S+S- + S-S+)_2:
            // S+_i2 S-_j2: need i2=DOWN(1), j2=UP(0)
            if (s_i2 == 1 && s_j2 == 0) {
                uint64_t new_state = flip_bit(flip_bit(state, i2), j2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.5 * szsz_1 * contrib.real();
            }
            // S-_i2 S+_j2: need i2=UP(0), j2=DOWN(1)
            if (s_i2 == 0 && s_j2 == 1) {
                uint64_t new_state = flip_bit(flip_bit(state, i2), j2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.5 * szsz_1 * contrib.real();
            }
            
            // (1/2)(S+S- + S-S+)_1 × (SzSz)_2:
            // S+_i1 S-_j1: need i1=DOWN(1), j1=UP(0)
            if (s_i1 == 1 && s_j1 == 0) {
                uint64_t new_state = flip_bit(flip_bit(state, i1), j1);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.5 * szsz_2 * contrib.real();
            }
            // S-_i1 S+_j1: need i1=UP(0), j1=DOWN(1)
            if (s_i1 == 0 && s_j1 == 1) {
                uint64_t new_state = flip_bit(flip_bit(state, i1), j1);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.5 * szsz_2 * contrib.real();
            }
            
            // (1/4)(XY)_1 × (XY)_2 terms:
            // These are 4-spin off-diagonal terms, similar to XY dimer-dimer
            // Factor is 1/4 from (1/2)×(1/2) prefactors
            
            // Term: S+_i1 S-_j1 S+_i2 S-_j2 (need i1=1,j1=0,i2=1,j2=0)
            if (s_i1 == 1 && s_j1 == 0 && s_i2 == 1 && s_j2 == 0) {
                uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.25 * contrib.real();
            }
            
            // Term: S+_i1 S-_j1 S-_i2 S+_j2 (need i1=1,j1=0,i2=0,j2=1)
            if (s_i1 == 1 && s_j1 == 0 && s_i2 == 0 && s_j2 == 1) {
                uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.25 * contrib.real();
            }
            
            // Term: S-_i1 S+_j1 S+_i2 S-_j2 (need i1=0,j1=1,i2=1,j2=0)
            if (s_i1 == 0 && s_j1 == 1 && s_i2 == 1 && s_j2 == 0) {
                uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.25 * contrib.real();
            }
            
            // Term: S-_i1 S+_j1 S-_i2 S+_j2 (need i1=0,j1=1,i2=0,j2=1)
            if (s_i1 == 0 && s_j1 == 1 && s_i2 == 0 && s_j2 == 1) {
                uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, i1), j1), i2), j2);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_result += 0.25 * contrib.real();
            }
        }
        
        result += local_result;
    }
    
    return result;
}

// -----------------------------------------------------------------------------
// Compute spin structure factor S(q)
// -----------------------------------------------------------------------------

struct StructureFactorResult {
    std::vector<Complex> s_q;  // S(q) at each k-point
    int q_max_idx;
    Complex s_q_max;
    std::array<double, 2> q_max;
    double m_translation;
};

StructureFactorResult compute_spin_structure_factor(
    const std::vector<std::vector<Complex>>& smsp_corr,
    const Cluster& cluster
) {
    StructureFactorResult result;
    int n_k = cluster.k_points.size();
    int n_sites = cluster.n_sites;
    
    result.s_q.resize(n_k, 0.0);
    
    std::cout << "Computing S(q) at " << n_k << " k-points..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int ik = 0; ik < n_k; ++ik) {
        const auto& q = cluster.k_points[ik];
        Complex s_q = 0.0;
        
        for (int i = 0; i < n_sites; ++i) {
            for (int j = 0; j < n_sites; ++j) {
                // S(q) = (1/N) Σᵢⱼ ⟨S⁻ᵢ S⁺ⱼ⟩ e^(iq·(Rⱼ-Rᵢ))
                // Matches TPQ_DSSF "0,0": ⟨(S⁺(q))† S⁺(q)⟩ = ⟨S⁻(-q) S⁺(q)⟩
                // Use minimum-image displacement for PBC-correct phases
                auto dr = cluster.minimum_image_displacement(i, j);
                double phase_arg = q[0] * dr[0] + q[1] * dr[1];
                s_q += smsp_corr[i][j] * std::exp(I * phase_arg);
            }
        }
        result.s_q[ik] = s_q / static_cast<double>(n_sites);
    }
    
    // Find maximum
    double max_val = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        double val = std::abs(result.s_q[ik]);
        if (val > max_val) {
            max_val = val;
            result.q_max_idx = ik;
            result.s_q_max = result.s_q[ik];
            result.q_max = cluster.k_points[ik];
        }
    }
    
    result.m_translation = std::sqrt(max_val / n_sites);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Compute nematic order
// -----------------------------------------------------------------------------

struct NematicResult {
    Complex psi_nem;
    double m_nem;
    std::array<Complex, 3> O_bar;  // Average bond energy per orientation
    double anisotropy;
    std::string bond_type;  // "xy", "szsz", or "heisenberg"
};

// Generic nematic order from complex bond expectations
NematicResult compute_nematic_order(
    const std::map<std::pair<int, int>, Complex>& bond_exp,
    const Cluster& cluster,
    const std::string& bond_type = "xy"
) {
    NematicResult result;
    result.bond_type = bond_type;
    std::array<Complex, 3> sum_by_orient = {0.0, 0.0, 0.0};
    std::array<int, 3> count_by_orient = {0, 0, 0};
    
    for (const auto& [edge, exp_val] : bond_exp) {
        int alpha = cluster.bond_orientation.at(edge);
        sum_by_orient[alpha] += exp_val;
        count_by_orient[alpha]++;
    }
    
    for (int alpha = 0; alpha < 3; ++alpha) {
        if (count_by_orient[alpha] > 0) {
            result.O_bar[alpha] = sum_by_orient[alpha] / static_cast<double>(count_by_orient[alpha]);
        }
    }
    
    // ψ_nem = Σ ω^α O̅_α, where ω = exp(2πi/3)
    Complex omega = std::exp(2.0 * PI * I / 3.0);
    result.psi_nem = result.O_bar[0] + omega * result.O_bar[1] + omega * omega * result.O_bar[2];
    result.m_nem = std::abs(result.psi_nem);
    
    // Anisotropy
    std::array<double, 3> mags = {std::abs(result.O_bar[0]), std::abs(result.O_bar[1]), std::abs(result.O_bar[2])};
    double max_mag = *std::max_element(mags.begin(), mags.end());
    double min_mag = *std::min_element(mags.begin(), mags.end());
    result.anisotropy = (max_mag > 1e-10) ? (max_mag - min_mag) / max_mag : 0.0;
    
    std::cout << "Nematic order (" << bond_type << "): m_nem = " << result.m_nem 
              << ", anisotropy = " << result.anisotropy << std::endl;
    
    return result;
}

// Nematic order from real-valued bond expectations (SzSz, Heisenberg)
NematicResult compute_nematic_order_real(
    const std::map<std::pair<int, int>, double>& bond_exp,
    const Cluster& cluster,
    const std::string& bond_type = "szsz"
) {
    // Convert to complex map and use the generic function
    std::map<std::pair<int, int>, Complex> bond_exp_complex;
    for (const auto& [edge, val] : bond_exp) {
        bond_exp_complex[edge] = Complex(val, 0.0);
    }
    return compute_nematic_order(bond_exp_complex, cluster, bond_type);
}

// -----------------------------------------------------------------------------
// Compute VBS (Valence Bond Solid) order with PROPER 4-site correlations
// S_D(q) = (1/N_b) Σ_{b,b'} exp(iq·(r_b - r_{b'})) ⟨δD_b δD_{b'}⟩_connected
// where D_b = S^+_i S^-_j + S^-_i S^+_j (XY dimer operator)
//    or D_b = S_i · S_j = SzSz + (1/2)(S+S- + S-S+) (Heisenberg dimer)
// 
// The connected correlator is: ⟨D_b D_{b'}⟩ - ⟨D_b⟩⟨D_{b'}⟩
// This requires computing ACTUAL 4-site spin correlations!
// -----------------------------------------------------------------------------

struct VBSResult {
    // XY dimer VBS
    std::vector<Complex> S_d_xy;          // S_D(q) at each k-point (XY dimers)
    std::vector<std::vector<Complex>> S_d_xy_2d;  // 2D grid for visualization
    std::vector<std::vector<Complex>> dimer_corr_xy;  // Raw ⟨D_b D_b'⟩ matrix (bond-resolved)
    std::vector<std::vector<Complex>> connected_corr_xy;  // Connected ⟨δD_b δD_b'⟩
    int q_max_idx_xy;
    Complex s_d_max_xy;
    std::array<double, 2> q_max_xy;
    double m_vbs_xy;
    double D_mean_xy;                      // Mean XY bond value
    
    // Heisenberg dimer VBS
    std::vector<double> S_d_heis;           // S_D(q) at each k-point (Heisenberg dimers)
    std::vector<std::vector<double>> S_d_heis_2d;  // 2D grid for visualization
    std::vector<std::vector<double>> dimer_corr_heis;  // Raw ⟨(S·S)_b (S·S)_b'⟩ matrix
    std::vector<std::vector<double>> connected_corr_heis;  // Connected correlator
    int q_max_idx_heis;
    double s_d_max_heis;
    std::array<double, 2> q_max_heis;
    double m_vbs_heis;
    double D_mean_heis;                     // Mean Heisenberg bond value
    
    // Orientation-resolved structure factors (3x3 matrix per q, stored as 6 independent components)
    // Index mapping: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,1), 4=(1,2), 5=(2,2)
    // S_D^{αβ}(q) = (1/N_b) Σ_{b∈α, b'∈β} exp(iq·(r_b - r_{b'})) ⟨δD_b δD_{b'}⟩
    std::vector<std::array<Complex, 6>> S_d_xy_oriented;  // [n_k][6] XY orientation-resolved
    std::vector<std::array<double, 6>> S_d_heis_oriented;  // [n_k][6] Heisenberg orientation-resolved
    std::array<int, 3> n_bonds_per_orientation;  // Number of bonds per orientation
    
    int n_q_grid;                       // Size of 2D grid
    int n_bonds;                        // Number of bonds
    
    // Backward compatible accessors (default to XY)
    const std::vector<Complex>& S_d() const { return S_d_xy; }
    const std::vector<std::vector<Complex>>& S_d_2d() const { return S_d_xy_2d; }
    int q_max_idx() const { return q_max_idx_xy; }
    Complex s_d_max() const { return s_d_max_xy; }
    std::array<double, 2> q_max() const { return q_max_xy; }
    double m_vbs() const { return m_vbs_xy; }
    double D_mean() const { return D_mean_xy; }
};

VBSResult compute_vbs_order(
    const std::vector<Complex>& psi,
    const std::map<std::pair<int, int>, Complex>& xy_bond_exp,
    const std::map<std::pair<int, int>, double>& heisenberg_bond_exp,
    const Cluster& cluster,
    int n_q_grid = 50
) {
    VBSResult result;
    result.n_q_grid = n_q_grid;
    int n_bonds = cluster.edges_nn.size();
    result.n_bonds = n_bonds;
    int n_k = cluster.k_points.size();
    
    if (n_bonds == 0) {
        result.m_vbs_xy = 0.0;
        result.m_vbs_heis = 0.0;
        return result;
    }
    
    std::cout << "Computing VBS order using efficient Fourier-space method..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert edges to vector for indexed access
    std::vector<std::pair<int, int>> edges(cluster.edges_nn.begin(), cluster.edges_nn.end());
    
    // Group bonds by orientation and compute bond centers
    std::array<std::vector<std::pair<int, int>>, 3> edges_by_orient;
    std::array<std::vector<std::array<double, 2>>, 3> centers_by_orient;
    std::vector<std::array<double, 2>> all_bond_centers(n_bonds);
    
    for (int b = 0; b < n_bonds; ++b) {
        int i = edges[b].first;
        int j = edges[b].second;
        auto center = cluster.bond_center_pbc(i, j);
        all_bond_centers[b] = center;
        
        int orient = cluster.bond_orientation.at(edges[b]);
        edges_by_orient[orient].push_back(edges[b]);
        centers_by_orient[orient].push_back(center);
    }
    
    result.n_bonds_per_orientation = {
        static_cast<int>(edges_by_orient[0].size()),
        static_cast<int>(edges_by_orient[1].size()),
        static_cast<int>(edges_by_orient[2].size())
    };
    
    std::cout << "  Bonds per orientation: " << result.n_bonds_per_orientation[0] << ", "
              << result.n_bonds_per_orientation[1] << ", "
              << result.n_bonds_per_orientation[2] << std::endl;
    
    // =========================================================================
    // Mean bond values ⟨D_α⟩ for each orientation
    // =========================================================================
    std::array<Complex, 3> D_mean_xy_orient = {};
    std::array<double, 3> D_mean_heis_orient = {};
    
    double sum_xy = 0.0, sum_heis = 0.0;
    for (int b = 0; b < n_bonds; ++b) {
        sum_xy += std::real(xy_bond_exp.at(edges[b]));
        sum_heis += heisenberg_bond_exp.at(edges[b]);
    }
    result.D_mean_xy = sum_xy / n_bonds;
    result.D_mean_heis = sum_heis / n_bonds;
    
    for (int alpha = 0; alpha < 3; ++alpha) {
        Complex sum_a = 0.0;
        double sum_h = 0.0;
        for (const auto& edge : edges_by_orient[alpha]) {
            sum_a += xy_bond_exp.at(edge);
            sum_h += heisenberg_bond_exp.at(edge);
        }
        if (!edges_by_orient[alpha].empty()) {
            D_mean_xy_orient[alpha] = sum_a / static_cast<double>(edges_by_orient[alpha].size());
            D_mean_heis_orient[alpha] = sum_h / edges_by_orient[alpha].size();
        }
    }
    
    // =========================================================================
    // Efficient Fourier-space computation of S_D^{αβ}(q):
    // D_α(q)|ψ⟩ = Σ_{b∈α} exp(iq·r_b) D_b|ψ⟩
    // S_D^{αβ}(q) = (1/√(N_α N_β)) [⟨D_α(q)ψ|D_β(q)ψ⟩ - ⟨D_α(q)⟩*⟨D_β(q)⟩]
    //
    // This is O(3 * n_k * N_bonds * Hilbert) instead of O(N_bonds² * Hilbert)
    // =========================================================================
    
    result.S_d_xy.resize(n_k, 0.0);
    result.S_d_heis.resize(n_k, 0.0);
    result.S_d_xy_oriented.resize(n_k);
    result.S_d_heis_oriented.resize(n_k);
    
    // Map from (α, β) pair to index in the 6-element array
    auto orient_pair_idx = [](int alpha, int beta) -> int {
        if (alpha > beta) std::swap(alpha, beta);
        if (alpha == 0) return beta;
        if (alpha == 1) return 2 + beta;
        return 5;
    };
    
    std::cout << "  Computing S_D(q) at " << n_k << " k-points using Fourier method..." << std::flush;
    
    if (g_use_memory_efficient_mode) {
        // Memory-efficient mode: use direct structure factor computation
        // This computes ⟨D†(q)D(q)⟩ without storing O(Hilbert) intermediate vectors
        std::cout << "\n  [Memory-efficient mode: using direct SF computation]" << std::endl;
        
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            
            // Compute XY dimer structure factor directly
            auto sf_xy = compute_dimer_sf_direct(psi, edges, all_bond_centers, q);
            Complex s_d_total_xy = sf_xy.overlap - std::norm(sf_xy.expect_q1);
            result.S_d_xy[ik] = s_d_total_xy / static_cast<double>(n_bonds);
            
            // Compute Heisenberg dimer structure factor directly
            auto sf_heis = compute_heisenberg_sf_direct(psi, edges, all_bond_centers, q);
            Complex s_d_total_heis = sf_heis.overlap - std::norm(sf_heis.expect_q1);
            result.S_d_heis[ik] = s_d_total_heis.real() / n_bonds;
            
            // For orientation-resolved, compute each pair separately
            std::array<Complex, 6> s_d_xy_orient = {};
            std::array<double, 6> s_d_heis_orient = {};
            
            // Skip detailed orientation-resolved computation in memory-efficient mode
            // (would require O(9 * n_bonds² * Hilbert) which is expensive)
            result.S_d_xy_oriented[ik] = s_d_xy_orient;
            result.S_d_heis_oriented[ik] = s_d_heis_orient;
            
            if ((ik + 1) % 5 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_D(q): " << (ik + 1) << "/" << n_k << " k-points..." << std::flush;
            }
        }
    } else {
        // Original mode: use Fourier-transformed operators (faster but more memory)
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            std::array<double, 2> neg_q = {-q[0], -q[1]};
            
            // Apply D_α(q)|ψ⟩ for each orientation (XY dimer)
            std::array<std::vector<Complex>, 3> D_q_psi_xy;
            std::array<Complex, 3> D_q_expect_xy = {};
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                if (edges_by_orient[alpha].empty()) {
                    D_q_psi_xy[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                D_q_psi_xy[alpha] = apply_dimer_fourier(psi, edges_by_orient[alpha], 
                                                        centers_by_orient[alpha], q);
                // ⟨D_α(q)⟩ = Σ_b exp(iq·r_b) ⟨D_b⟩
                for (size_t b = 0; b < edges_by_orient[alpha].size(); ++b) {
                    double phase_arg = q[0] * centers_by_orient[alpha][b][0] + 
                                       q[1] * centers_by_orient[alpha][b][1];
                    D_q_expect_xy[alpha] += std::exp(I * phase_arg) * 
                                            xy_bond_exp.at(edges_by_orient[alpha][b]);
                }
            }
            
            // Compute S_D^{αβ}(q) for all orientation pairs
            std::array<Complex, 6> s_d_xy_orient = {};
            Complex s_d_total = 0.0;
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_bonds_per_orientation[alpha];
                    int N_b = result.n_bonds_per_orientation[beta];
                    
                    if (N_a == 0 || N_b == 0) continue;
                    
                    // ⟨D_α(q)ψ|D_β(q)ψ⟩
                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap += std::conj(D_q_psi_xy[alpha][s]) * D_q_psi_xy[beta][s];
                    }
                    
                    // Connected: subtract ⟨D_α(q)⟩*⟨D_β(q)⟩
                    Complex connected = overlap - std::conj(D_q_expect_xy[alpha]) * D_q_expect_xy[beta];
                    
                    // Normalize by sqrt(N_α * N_β)
                    double norm = std::sqrt(static_cast<double>(N_a) * N_b);
                    s_d_xy_orient[idx] = connected / norm;
                    
                    // Contribution to total (accounting for symmetry α↔β)
                    if (alpha == beta) {
                        s_d_total += connected;
                    } else {
                        s_d_total += 2.0 * connected;  // α,β and β,α
                    }
                }
            }
            
            result.S_d_xy_oriented[ik] = s_d_xy_orient;
            result.S_d_xy[ik] = s_d_total / static_cast<double>(n_bonds);
            
            // For Heisenberg, use the Heisenberg dimer Fourier transform
            auto [D_q_psi_heis_all, D_q_expect_heis_all] = apply_heisenberg_dimer_fourier(
                psi, edges, all_bond_centers, q);
            
            // For orientation-resolved Heisenberg, we need separate applications
            std::array<double, 6> s_d_heis_orient = {};
            double s_d_heis_total = 0.0;
            
            // Compute per-orientation Heisenberg structure factors
            std::array<std::vector<Complex>, 3> D_q_psi_heis;
            std::array<Complex, 3> D_q_expect_heis = {};
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                if (edges_by_orient[alpha].empty()) {
                    D_q_psi_heis[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                auto [dpsi, dexp] = apply_heisenberg_dimer_fourier(
                    psi, edges_by_orient[alpha], centers_by_orient[alpha], q);
                D_q_psi_heis[alpha] = std::move(dpsi);
                D_q_expect_heis[alpha] = dexp;
            }
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_bonds_per_orientation[alpha];
                    int N_b = result.n_bonds_per_orientation[beta];
                    
                    if (N_a == 0 || N_b == 0) continue;
                    
                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap += std::conj(D_q_psi_heis[alpha][s]) * D_q_psi_heis[beta][s];
                    }
                    
                    Complex connected = overlap - std::conj(D_q_expect_heis[alpha]) * D_q_expect_heis[beta];
                    double norm = std::sqrt(static_cast<double>(N_a) * N_b);
                    s_d_heis_orient[idx] = connected.real() / norm;
                    
                    if (alpha == beta) {
                        s_d_heis_total += connected.real();
                    } else {
                        s_d_heis_total += 2.0 * connected.real();
                    }
                }
            }
            
            result.S_d_heis_oriented[ik] = s_d_heis_orient;
            result.S_d_heis[ik] = s_d_heis_total / n_bonds;
            
            if ((ik + 1) % 10 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_D(q): " << (ik + 1) << "/" << n_k << " k-points..." << std::flush;
            }
        }
    }
    std::cout << " done" << std::endl;
    
    // Find maxima
    double max_val_xy = 0.0, max_val_heis = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        double val_xy = std::abs(result.S_d_xy[ik]);
        if (val_xy > max_val_xy) {
            max_val_xy = val_xy;
            result.q_max_idx_xy = ik;
            result.s_d_max_xy = result.S_d_xy[ik];
            result.q_max_xy = cluster.k_points[ik];
        }
        
        double val_heis = std::abs(result.S_d_heis[ik]);
        if (val_heis > max_val_heis) {
            max_val_heis = val_heis;
            result.q_max_idx_heis = ik;
            result.s_d_max_heis = result.S_d_heis[ik];
            result.q_max_heis = cluster.k_points[ik];
        }
    }
    result.m_vbs_xy = std::sqrt(max_val_xy / n_bonds);
    result.m_vbs_heis = std::sqrt(max_val_heis / n_bonds);
    
    // =========================================================================
    // Also compute on dense 2D grid for visualization (total S_D only)
    // Skip in memory-efficient mode for very large systems
    // =========================================================================
    if (g_use_memory_efficient_mode && n_q_grid > 10) {
        std::cout << "  [Memory-efficient mode: skipping dense 2D VBS grid (use --n-q-grid 10 to enable)]" << std::endl;
        // Just initialize empty grids
        result.S_d_xy_2d.resize(0);
        result.S_d_heis_2d.resize(0);
    } else {
        result.S_d_xy_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
        result.S_d_heis_2d.resize(n_q_grid, std::vector<double>(n_q_grid, 0.0));
        
        std::cout << "  Computing S_D(q) on " << n_q_grid << "x" << n_q_grid << " grid..." << std::flush;
        
        // For 2D grid, use total bonds (not orientation-resolved for speed)
        for (int i1 = 0; i1 < n_q_grid; ++i1) {
            for (int i2 = 0; i2 < n_q_grid; ++i2) {
                double q1 = -1.0 + 2.0 * i1 / (n_q_grid - 1);
                double q2 = -1.0 + 2.0 * i2 / (n_q_grid - 1);
                std::array<double, 2> qvec = {
                    q1 * cluster.b1[0] + q2 * cluster.b2[0],
                    q1 * cluster.b1[1] + q2 * cluster.b2[1]
                };
                
                if (g_use_memory_efficient_mode) {
                    // Use direct structure factor computation
                    auto sf_xy = compute_dimer_sf_direct(psi, edges, all_bond_centers, qvec);
                    result.S_d_xy_2d[i1][i2] = (sf_xy.overlap - std::norm(sf_xy.expect_q1)) / static_cast<double>(n_bonds);
                    
                    auto sf_heis = compute_heisenberg_sf_direct(psi, edges, all_bond_centers, qvec);
                    result.S_d_heis_2d[i1][i2] = (sf_heis.overlap - std::norm(sf_heis.expect_q1)).real() / n_bonds;
                } else {
                    // XY
                    auto D_q_psi = apply_dimer_fourier(psi, edges, all_bond_centers, qvec);
                    Complex D_q_expect = 0.0;
                    for (int b = 0; b < n_bonds; ++b) {
                        double phase_arg = qvec[0] * all_bond_centers[b][0] + qvec[1] * all_bond_centers[b][1];
                        D_q_expect += std::exp(I * phase_arg) * xy_bond_exp.at(edges[b]);
                    }
                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap += std::conj(D_q_psi[s]) * D_q_psi[s];
                    }
                    result.S_d_xy_2d[i1][i2] = (overlap - std::norm(D_q_expect)) / static_cast<double>(n_bonds);
                    
                    // Heisenberg
                    auto [D_q_psi_h, D_q_expect_h] = apply_heisenberg_dimer_fourier(psi, edges, all_bond_centers, qvec);
                    Complex overlap_h = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap_h += std::conj(D_q_psi_h[s]) * D_q_psi_h[s];
                    }
                    result.S_d_heis_2d[i1][i2] = (overlap_h - std::norm(D_q_expect_h)).real() / n_bonds;
                }
            }
            
            if ((i1 + 1) % 10 == 0 || i1 == n_q_grid - 1) {
                std::cout << "\r  Computing S_D(q) 2D grid: " << (i1 + 1) << "/" << n_q_grid << " rows..." << std::flush;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done" << std::endl;
    std::cout << "  VBS order (XY):         m_vbs = " << result.m_vbs_xy 
              << " at q = (" << result.q_max_xy[0] << ", " << result.q_max_xy[1] << ")" << std::endl;
    std::cout << "  VBS order (Heisenberg): m_vbs = " << result.m_vbs_heis
              << " at q = (" << result.q_max_heis[0] << ", " << result.q_max_heis[1] << ")"
              << " [" << duration.count() << " ms]" << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Find all NN triangles in the kagome lattice
// A triangle is 3 sites where all pairs are NN connected.
// Returns list of (s1, s2, s3) tuples with s1 < s2 < s3.
// -----------------------------------------------------------------------------

std::vector<std::array<int, 3>> find_triangles(const Cluster& cluster) {
    std::vector<std::array<int, 3>> triangles;
    
    // Build NN adjacency set
    std::vector<std::set<int>> nn_adj(cluster.n_sites);
    for (const auto& [i, j] : cluster.edges_nn) {
        nn_adj[i].insert(j);
        nn_adj[j].insert(i);
    }
    
    // For each site, look for triangles where it's the smallest index
    for (int s1 = 0; s1 < cluster.n_sites; ++s1) {
        std::vector<int> neighbors;
        for (int n : nn_adj[s1]) {
            if (n > s1) neighbors.push_back(n);
        }
        
        // Check all pairs of neighbors > s1
        for (size_t i = 0; i < neighbors.size(); ++i) {
            int s2 = neighbors[i];
            for (size_t k = i + 1; k < neighbors.size(); ++k) {
                int s3 = neighbors[k];
                // Check if s2-s3 are also NN
                if (nn_adj[s2].count(s3)) {
                    triangles.push_back({s1, s2, s3});
                }
            }
        }
    }
    
    return triangles;
}

// -----------------------------------------------------------------------------
// Bowtie structure: two triangles sharing exactly one vertex
// -----------------------------------------------------------------------------

struct Bowtie {
    int s0;  // Shared center vertex
    int s1, s2;  // Triangle 1 outer vertices
    int s3, s4;  // Triangle 2 outer vertices
    std::array<double, 2> center;  // Center position for Fourier transform
    int orientation;  // Orientation based on sublattice of center vertex (0, 1, or 2)
};

// Find all bowties in the kagome lattice
std::vector<Bowtie> find_bowties(const Cluster& cluster) {
    std::vector<Bowtie> bowties;
    
    // Find all triangles
    auto triangles = find_triangles(cluster);
    
    // Build index: which triangles contain each vertex
    std::vector<std::vector<int>> vertex_to_triangles(cluster.n_sites);
    for (int idx = 0; idx < static_cast<int>(triangles.size()); ++idx) {
        const auto& tri = triangles[idx];
        vertex_to_triangles[tri[0]].push_back(idx);
        vertex_to_triangles[tri[1]].push_back(idx);
        vertex_to_triangles[tri[2]].push_back(idx);
    }
    
    // Track found bowties to avoid duplicates
    std::set<std::tuple<int, int, int, int, int>> found_set;
    
    // For each vertex, find all pairs of triangles sharing ONLY that vertex
    for (int shared_vertex = 0; shared_vertex < cluster.n_sites; ++shared_vertex) {
        const auto& tri_indices = vertex_to_triangles[shared_vertex];
        if (tri_indices.size() < 2) continue;
        
        // Check all pairs of triangles at this vertex
        for (size_t i = 0; i < tri_indices.size(); ++i) {
            for (size_t j = i + 1; j < tri_indices.size(); ++j) {
                const auto& t1 = triangles[tri_indices[i]];
                const auto& t2 = triangles[tri_indices[j]];
                
                // Get the other two vertices of each triangle
                std::vector<int> other1, other2;
                for (int v : {t1[0], t1[1], t1[2]}) {
                    if (v != shared_vertex) other1.push_back(v);
                }
                for (int v : {t2[0], t2[1], t2[2]}) {
                    if (v != shared_vertex) other2.push_back(v);
                }
                
                // Check that triangles share ONLY the shared_vertex (no edge sharing)
                std::set<int> set1(other1.begin(), other1.end());
                std::set<int> set2(other2.begin(), other2.end());
                std::vector<int> intersection;
                std::set_intersection(set1.begin(), set1.end(), 
                                      set2.begin(), set2.end(),
                                      std::back_inserter(intersection));
                if (!intersection.empty()) continue;  // They share more than one vertex
                
                // This is a valid bowtie
                int s0 = shared_vertex;
                int s1 = other1[0], s2 = other1[1];
                int s3 = other2[0], s4 = other2[1];
                
                // Create canonical key to avoid duplicates
                // Sort each triangle's other vertices, then sort triangles
                if (s1 > s2) std::swap(s1, s2);
                if (s3 > s4) std::swap(s3, s4);
                if (std::make_pair(s1, s2) > std::make_pair(s3, s4)) {
                    std::swap(s1, s3);
                    std::swap(s2, s4);
                }
                
                auto key = std::make_tuple(s0, s1, s2, s3, s4);
                if (found_set.count(key)) continue;
                found_set.insert(key);
                
                // Center position (mean of all 5 sites)
                std::array<double, 2> center = {0.0, 0.0};
                for (int site : {s0, s1, s2, s3, s4}) {
                    center[0] += cluster.positions[site][0];
                    center[1] += cluster.positions[site][1];
                }
                center[0] /= 5.0;
                center[1] /= 5.0;
                
                // Orientation based on sublattice of center vertex
                int orientation = cluster.sublattice[s0];
                
                bowties.push_back({s0, s1, s2, s3, s4, center, orientation});
            }
        }
    }
    
    return bowties;
}

// -----------------------------------------------------------------------------
// Compute bowtie ring-flip expectation:
// P_r = ⟨S⁺₁S⁻₂S⁺₃S⁻₄ + h.c.⟩
// Acts on the 4 OUTER corners (s1, s2, s3, s4), EXCLUDING center s0
// -----------------------------------------------------------------------------

Complex compute_bowtie_resonance(
    const std::vector<Complex>& psi,
    int s1, int s2, int s3, int s4
) {
    uint64_t n_states = psi.size();
    double result_real = 0.0;
    double result_imag = 0.0;
    
    #pragma omp parallel reduction(+:result_real,result_imag)
    {
        double local_real = 0.0;
        double local_imag = 0.0;
        
        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;
            
            // S⁺_s1 S⁻_s2 S⁺_s3 S⁻_s4 term
            // Requires: bit(s1)=1 (down), bit(s2)=0 (up), bit(s3)=1 (down), bit(s4)=0 (up)
            int b1 = get_bit(state, s1);
            int b2 = get_bit(state, s2);
            int b3 = get_bit(state, s3);
            int b4 = get_bit(state, s4);
            
            // S+ raises: needs spin down (bit=1), S- lowers: needs spin up (bit=0)
            if (b1 == 1 && b2 == 0 && b3 == 1 && b4 == 0) {
                // All operators succeed with coefficient 1 each
                uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            
            // Hermitian conjugate: S⁻_s1 S⁺_s2 S⁻_s3 S⁺_s4
            // Requires: bit(s1)=0, bit(s2)=1, bit(s3)=0, bit(s4)=1
            if (b1 == 0 && b2 == 1 && b3 == 0 && b4 == 1) {
                uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
        }
        
        result_real += local_real;
        result_imag += local_imag;
    }
    
    return Complex(result_real, result_imag);
}

// -----------------------------------------------------------------------------
// Compute triangle chiral/resonance expectation:
// χ = ⟨S⁺₁S⁻₂S⁺₃ + S⁻₁S⁺₂S⁻₃⟩
// Measures 3-site ring exchange around a triangle
// -----------------------------------------------------------------------------

Complex compute_triangle_chiral(
    const std::vector<Complex>& psi,
    int s1, int s2, int s3
) {
    uint64_t n_states = psi.size();
    double result_real = 0.0;
    double result_imag = 0.0;
    
    #pragma omp parallel reduction(+:result_real,result_imag)
    {
        double local_real = 0.0;
        double local_imag = 0.0;
        
        #pragma omp for schedule(dynamic, 1024)
        for (uint64_t state = 0; state < n_states; ++state) {
            Complex coeff = psi[state];
            if (std::abs(coeff) < 1e-15) continue;
            
            int b1 = get_bit(state, s1);
            int b2 = get_bit(state, s2);
            int b3 = get_bit(state, s3);
            
            // S⁺_s1 S⁻_s2 S⁺_s3: needs b1=1, b2=0, b3=1
            if (b1 == 1 && b2 == 0 && b3 == 1) {
                uint64_t new_state = flip_bit(flip_bit(flip_bit(state, s1), s2), s3);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            
            // S⁻_s1 S⁺_s2 S⁻_s3: needs b1=0, b2=1, b3=0
            if (b1 == 0 && b2 == 1 && b3 == 0) {
                uint64_t new_state = flip_bit(flip_bit(flip_bit(state, s1), s2), s3);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
        }
        
        result_real += local_real;
        result_imag += local_imag;
    }
    
    return Complex(result_real, result_imag);
}

// -----------------------------------------------------------------------------
// Plaquette order parameter result structure
// -----------------------------------------------------------------------------

struct PlaquetteResult {
    // Bowtie resonance order
    std::vector<Complex> S_p;           // S_P(q) at each k-point
    std::vector<std::vector<Complex>> S_p_2d;  // 2D grid for visualization
    int q_max_idx;
    Complex s_p_max;
    std::array<double, 2> q_max;
    double m_plaquette;                 // Order parameter sqrt(S_P(q_max)/N_bt)
    
    // Orientation-resolved structure factors (3x3 matrix per q, stored as 6 independent components)
    // Index mapping: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,1), 4=(1,2), 5=(2,2)
    // S_P^{αβ}(q) = (1/N_bt) Σ_{bt∈α, bt'∈β} exp(iq·(r_bt - r_{bt'})) ⟨δP_bt δP_{bt'}⟩
    std::vector<std::array<Complex, 6>> S_p_oriented;  // [n_k][6] orientation-resolved
    std::array<int, 3> n_plaquettes_per_orientation;   // Number of bowties per orientation
    
    // Raw plaquette data
    int n_plaquettes;
    double P_mean;                      // Mean bowtie resonance
    double resonance_strength;          // Mean |P_bt|
    std::vector<Complex> P_r;           // Individual bowtie resonances
    std::vector<std::array<double, 2>> centers;  // Bowtie centers
    std::vector<int> orientations;      // Orientation of each bowtie (sublattice of center)
    
    // Triangle chiral order
    int n_triangles;
    double chi_mean;                    // Mean triangle chiral
    std::vector<Complex> chi_r;         // Individual triangle chirals
};

// -----------------------------------------------------------------------------
// Compute plaquette/bowtie resonance order
// S_P(q) = (1/N_bt) Σ_{bt,bt'} exp(iq·(R_bt - R_{bt'})) ⟨δP_bt δP_{bt'}⟩
// where δP_bt = P_bt - ⟨P⟩
// -----------------------------------------------------------------------------

PlaquetteResult compute_plaquette_order(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    int n_q_grid = 50
) {
    PlaquetteResult result;
    int n_k = cluster.k_points.size();
    
    std::cout << "Computing plaquette/bowtie resonance order using efficient Fourier method..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Find all triangles (for chiral order)
    auto triangles = find_triangles(cluster);
    result.n_triangles = triangles.size();
    std::cout << "  Found " << result.n_triangles << " triangular plaquettes" << std::endl;
    
    // Find all bowties (for resonance order)
    auto bowties = find_bowties(cluster);
    result.n_plaquettes = bowties.size();
    std::cout << "  Found " << result.n_plaquettes << " bowtie plaquettes (5-site, 2 triangles sharing corner)" << std::endl;
    
    if (result.n_plaquettes == 0) {
        result.m_plaquette = 0.0;
        result.P_mean = 0.0;
        result.resonance_strength = 0.0;
        return result;
    }
    
    // Group bowties by orientation and prepare BowtieData for Fourier transform
    std::array<std::vector<BowtieData>, 3> bowties_by_orient;
    std::vector<BowtieData> all_bowties(result.n_plaquettes);
    
    result.P_r.resize(result.n_plaquettes);
    result.centers.resize(result.n_plaquettes);
    result.orientations.resize(result.n_plaquettes);
    
    for (int idx = 0; idx < result.n_plaquettes; ++idx) {
        const auto& bt = bowties[idx];
        all_bowties[idx] = {bt.s1, bt.s2, bt.s3, bt.s4, bt.center};
        result.centers[idx] = bt.center;
        result.orientations[idx] = bt.orientation;
        bowties_by_orient[bt.orientation].push_back(all_bowties[idx]);
    }
    
    result.n_plaquettes_per_orientation = {
        static_cast<int>(bowties_by_orient[0].size()),
        static_cast<int>(bowties_by_orient[1].size()),
        static_cast<int>(bowties_by_orient[2].size())
    };
    
    std::cout << "  Bowties per orientation: " << result.n_plaquettes_per_orientation[0] << ", "
              << result.n_plaquettes_per_orientation[1] << ", "
              << result.n_plaquettes_per_orientation[2] << std::endl;
    
    // Compute single-plaquette expectations ⟨P_bt⟩ for each bowtie (needed for means)
    std::cout << "  Computing individual bowtie expectations..." << std::flush;
    
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < result.n_plaquettes; ++idx) {
        const auto& bt = bowties[idx];
        result.P_r[idx] = compute_bowtie_resonance(psi, bt.s1, bt.s2, bt.s3, bt.s4);
    }
    std::cout << " done" << std::endl;
    
    // Mean plaquette value and resonance strength per orientation
    std::array<Complex, 3> P_mean_orient = {};
    double sum_P = 0.0, sum_abs_P = 0.0;
    
    for (int idx = 0; idx < result.n_plaquettes; ++idx) {
        sum_P += result.P_r[idx].real();
        sum_abs_P += std::abs(result.P_r[idx]);
        P_mean_orient[result.orientations[idx]] += result.P_r[idx];
    }
    result.P_mean = sum_P / result.n_plaquettes;
    result.resonance_strength = sum_abs_P / result.n_plaquettes;
    
    for (int alpha = 0; alpha < 3; ++alpha) {
        if (result.n_plaquettes_per_orientation[alpha] > 0) {
            P_mean_orient[alpha] /= static_cast<double>(result.n_plaquettes_per_orientation[alpha]);
        }
    }
    
    // =========================================================================
    // Efficient Fourier-space computation of S_P^{αβ}(q):
    // P_α(q)|ψ⟩ = Σ_{p∈α} exp(iq·r_p) P_p|ψ⟩
    // S_P^{αβ}(q) = (1/√(N_α N_β)) [⟨P_α(q)ψ|P_β(q)ψ⟩ - ⟨P_α(q)⟩*⟨P_β(q)⟩]
    // =========================================================================
    
    result.S_p.resize(n_k, 0.0);
    result.S_p_oriented.resize(n_k);
    
    auto orient_pair_idx = [](int alpha, int beta) -> int {
        if (alpha > beta) std::swap(alpha, beta);
        if (alpha == 0) return beta;
        if (alpha == 1) return 2 + beta;
        return 5;
    };
    
    std::cout << "  Computing S_P(q) at " << n_k << " k-points using Fourier method..." << std::flush;
    
    if (g_use_memory_efficient_mode) {
        // Memory-efficient mode: compute structure factor without storing full O(Hilbert) vectors
        // For plaquette/bowtie, this is more complex - we skip detailed orientation resolution
        std::cout << "\n  [Memory-efficient mode: using simplified plaquette SF computation]" << std::endl;
        
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            
            // For memory efficiency, compute ⟨P†(q)P(q)⟩ directly
            // P(q)|ψ⟩ = Σ_p exp(iq·r_p) P_p|ψ⟩
            // We need ⟨ψ|P†(q)P(q)|ψ⟩ = Σ_{p,p'} exp(i q·(r_{p'} - r_p)) ⟨ψ|P_p P_{p'}|ψ⟩
            // For large systems, approximate by using individual expectations
            
            Complex P_q_expect = 0.0;
            Complex P_q_expect_sq = 0.0;  // For variance approximation
            
            for (int p = 0; p < result.n_plaquettes; ++p) {
                double phase_arg = q[0] * result.centers[p][0] + q[1] * result.centers[p][1];
                Complex phase = std::exp(I * phase_arg);
                P_q_expect += phase * result.P_r[p];
                P_q_expect_sq += std::norm(result.P_r[p]);  // |⟨P_p⟩|²
            }
            
            // Approximate S_P(q) using ⟨P⟩ values (underestimates true structure factor)
            // True value would require ⟨P_p P_{p'}⟩ four-point functions
            result.S_p[ik] = (P_q_expect_sq - std::norm(P_q_expect) / result.n_plaquettes) / 
                            static_cast<double>(result.n_plaquettes);
            
            result.S_p_oriented[ik] = {};  // Skip orientation-resolved in memory mode
            
            if ((ik + 1) % 10 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_P(q): " << (ik + 1) << "/" << n_k << " k-points..." << std::flush;
            }
        }
    } else {
        // Original mode with full vector storage
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            
            // Apply P_α(q)|ψ⟩ for each orientation
            std::array<std::vector<Complex>, 3> P_q_psi;
            std::array<Complex, 3> P_q_expect = {};
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                if (bowties_by_orient[alpha].empty()) {
                    P_q_psi[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                P_q_psi[alpha] = apply_bowtie_fourier(psi, bowties_by_orient[alpha], q);
                
                // ⟨P_α(q)⟩ = Σ_p exp(iq·r_p) ⟨P_p⟩
                int local_idx = 0;
                for (int p = 0; p < result.n_plaquettes; ++p) {
                    if (result.orientations[p] == alpha) {
                        double phase_arg = q[0] * result.centers[p][0] + q[1] * result.centers[p][1];
                        P_q_expect[alpha] += std::exp(I * phase_arg) * result.P_r[p];
                    }
                }
            }
            
            // Compute S_P^{αβ}(q) for all orientation pairs
            std::array<Complex, 6> s_p_orient = {};
            Complex s_p_total = 0.0;
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_plaquettes_per_orientation[alpha];
                    int N_b = result.n_plaquettes_per_orientation[beta];
                    
                    if (N_a == 0 || N_b == 0) continue;
                    
                    // ⟨P_α(q)ψ|P_β(q)ψ⟩
                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap += std::conj(P_q_psi[alpha][s]) * P_q_psi[beta][s];
                    }
                    
                    // Connected: subtract ⟨P_α(q)⟩*⟨P_β(q)⟩
                    Complex connected = overlap - std::conj(P_q_expect[alpha]) * P_q_expect[beta];
                    
                    // Normalize by sqrt(N_α * N_β)
                    double norm = std::sqrt(static_cast<double>(N_a) * N_b);
                    s_p_orient[idx] = connected / norm;
                    
                    // Contribution to total
                    if (alpha == beta) {
                        s_p_total += connected;
                    } else {
                        s_p_total += 2.0 * connected;
                    }
                }
            }
            
            result.S_p_oriented[ik] = s_p_orient;
            result.S_p[ik] = s_p_total / static_cast<double>(result.n_plaquettes);
            
            if ((ik + 1) % 10 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_P(q): " << (ik + 1) << "/" << n_k << " k-points..." << std::flush;
            }
        }
    }
    std::cout << " done" << std::endl;
    
    // Find maximum
    double max_val = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        double val = std::abs(result.S_p[ik]);
        if (val > max_val) {
            max_val = val;
            result.q_max_idx = ik;
            result.s_p_max = result.S_p[ik];
            result.q_max = cluster.k_points[ik];
        }
    }
    result.m_plaquette = std::sqrt(max_val / result.n_plaquettes);
    
    // =========================================================================
    // Also compute on dense 2D grid for visualization
    // Skip in memory-efficient mode for large systems
    // =========================================================================
    if (g_use_memory_efficient_mode && n_q_grid > 10) {
        std::cout << "  [Memory-efficient mode: skipping dense 2D plaquette grid]" << std::endl;
        result.S_p_2d.resize(0);
    } else {
        result.S_p_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
        
        std::cout << "  Computing S_P(q) on " << n_q_grid << "x" << n_q_grid << " grid..." << std::flush;
        
        for (int i1 = 0; i1 < n_q_grid; ++i1) {
            for (int i2 = 0; i2 < n_q_grid; ++i2) {
                double q1 = -1.0 + 2.0 * i1 / (n_q_grid - 1);
                double q2 = -1.0 + 2.0 * i2 / (n_q_grid - 1);
                std::array<double, 2> qvec = {
                    q1 * cluster.b1[0] + q2 * cluster.b2[0],
                    q1 * cluster.b1[1] + q2 * cluster.b2[1]
                };
                
                auto P_q_psi = apply_bowtie_fourier(psi, all_bowties, qvec);
                Complex P_q_expect = 0.0;
                for (int p = 0; p < result.n_plaquettes; ++p) {
                    double phase_arg = qvec[0] * result.centers[p][0] + qvec[1] * result.centers[p][1];
                    P_q_expect += std::exp(I * phase_arg) * result.P_r[p];
                }
                
                Complex overlap = 0.0;
                for (size_t s = 0; s < psi.size(); ++s) {
                    overlap += std::conj(P_q_psi[s]) * P_q_psi[s];
                }
                result.S_p_2d[i1][i2] = (overlap - std::norm(P_q_expect)) / static_cast<double>(result.n_plaquettes);
            }
            
            if ((i1 + 1) % 10 == 0 || i1 == n_q_grid - 1) {
                std::cout << "\r  Computing S_P(q) 2D grid: " << (i1 + 1) << "/" << n_q_grid << " rows..." << std::flush;
            }
        }
    }
    
    // =========================================================================
    // Compute triangle chiral order
    // =========================================================================
    result.chi_r.resize(result.n_triangles);
    
    std::cout << " done" << std::endl;
    std::cout << "  Computing triangle chiral expectations..." << std::flush;
    
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < result.n_triangles; ++idx) {
        const auto& tri = triangles[idx];
        result.chi_r[idx] = compute_triangle_chiral(psi, tri[0], tri[1], tri[2]);
    }
    
    double sum_chi = 0.0;
    for (int idx = 0; idx < result.n_triangles; ++idx) {
        sum_chi += result.chi_r[idx].real();
    }
    result.chi_mean = sum_chi / result.n_triangles;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << " done" << std::endl;
    std::cout << "  Plaquette order: m_plaquette = " << result.m_plaquette 
              << " at q = (" << result.q_max[0] << ", " << result.q_max[1] << ")" << std::endl;
    std::cout << "  Mean resonance: <P> = " << result.P_mean 
              << ", strength = " << result.resonance_strength << std::endl;
    std::cout << "  Triangle chiral: <χ> = " << result.chi_mean
              << " [" << duration.count() << " ms]" << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Compute 2D S(q) grid for visualization
// -----------------------------------------------------------------------------

std::vector<std::vector<Complex>> compute_sq_2d_grid(
    const std::vector<std::vector<Complex>>& smsp_corr,
    const Cluster& cluster,
    int n_q_grid = 50
) {
    int n_sites = cluster.n_sites;
    std::vector<std::vector<Complex>> s_q_2d(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
    
    std::cout << "Computing S(q) on " << n_q_grid << "x" << n_q_grid << " grid..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for collapse(2)
    for (int i1 = 0; i1 < n_q_grid; ++i1) {
        for (int i2 = 0; i2 < n_q_grid; ++i2) {
            double q1 = -1.0 + 2.0 * i1 / (n_q_grid - 1);
            double q2 = -1.0 + 2.0 * i2 / (n_q_grid - 1);
            double qx = q1 * cluster.b1[0] + q2 * cluster.b2[0];
            double qy = q1 * cluster.b1[1] + q2 * cluster.b2[1];
            
            Complex s_q = 0.0;
            for (int i = 0; i < n_sites; ++i) {
                for (int j = 0; j < n_sites; ++j) {
                    // S(q) = (1/N) Σᵢⱼ ⟨S⁻ᵢ S⁺ⱼ⟩ e^(iq·(Rⱼ-Rᵢ))
                    // Matches TPQ_DSSF "0,0" convention
                    // Use minimum-image displacement for PBC-correct phases
                    auto dr = cluster.minimum_image_displacement(i, j);
                    double phase_arg = qx * dr[0] + qy * dr[1];
                    s_q += smsp_corr[i][j] * std::exp(I * phase_arg);
                }
            }
            s_q_2d[i1][i2] = s_q / static_cast<double>(n_sites);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    
    return s_q_2d;
}

// -----------------------------------------------------------------------------
// Save results to HDF5 (full version with 2D grids and VBS)
// -----------------------------------------------------------------------------

void save_results(
    const std::string& filename,
    const StructureFactorResult& sf,
    const NematicResult& nem,
    const NematicResult& nem_spsm,
    const NematicResult& nem_szsz,
    const NematicResult& nem_heisenberg,
    const VBSResult& vbs,
    const PlaquetteResult& plaq,
    const Cluster& cluster,
    const std::vector<std::vector<Complex>>& s_q_2d,
    int n_q_grid,
    const std::map<std::pair<int, int>, Complex>& spsm_bonds = {},
    const std::map<std::pair<int, int>, double>& szsz_bonds = {},
    const std::map<std::pair<int, int>, double>& heisenberg_bonds = {}
) {
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        
        // Create compound type for complex numbers
        H5::CompType complex_type(sizeof(Complex));
        complex_type.insertMember("r", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("i", sizeof(double), H5::PredType::NATIVE_DOUBLE);
        
        // Save structure factor S(q) at k-points
        {
            hsize_t dims[1] = {sf.s_q.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_q", complex_type, dataspace);
            dataset.write(sf.s_q.data(), complex_type);
        }
        
        // Save 2D S(q) grid for visualization
        if (!s_q_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(n_q_grid), static_cast<hsize_t>(n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("S_q_2d", complex_type, dataspace);
            // Flatten for HDF5
            std::vector<Complex> flat(n_q_grid * n_q_grid);
            for (int i = 0; i < n_q_grid; ++i) {
                for (int j = 0; j < n_q_grid; ++j) {
                    flat[i * n_q_grid + j] = s_q_2d[i][j];
                }
            }
            dataset.write(flat.data(), complex_type);
        }
        
        // Save VBS S_D(q) at k-points (XY)
        if (!vbs.S_d_xy.empty()) {
            hsize_t dims[1] = {vbs.S_d_xy.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_D_q_xy", complex_type, dataspace);
            dataset.write(vbs.S_d_xy.data(), complex_type);
        }
        
        // Save VBS S_D(q) at k-points (Heisenberg)
        if (!vbs.S_d_heis.empty()) {
            hsize_t dims[1] = {vbs.S_d_heis.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_D_q_heis", H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(vbs.S_d_heis.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        // Save 2D S_D(q) grid for VBS visualization (XY)
        if (!vbs.S_d_xy_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(vbs.n_q_grid), static_cast<hsize_t>(vbs.n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("S_D_q_xy_2d", complex_type, dataspace);
            std::vector<Complex> flat(vbs.n_q_grid * vbs.n_q_grid);
            for (int i = 0; i < vbs.n_q_grid; ++i) {
                for (int j = 0; j < vbs.n_q_grid; ++j) {
                    flat[i * vbs.n_q_grid + j] = vbs.S_d_xy_2d[i][j];
                }
            }
            dataset.write(flat.data(), complex_type);
        }
        
        // Save 2D S_D(q) grid for VBS visualization (Heisenberg)
        if (!vbs.S_d_heis_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(vbs.n_q_grid), static_cast<hsize_t>(vbs.n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("S_D_q_heis_2d", H5::PredType::NATIVE_DOUBLE, dataspace);
            std::vector<double> flat(vbs.n_q_grid * vbs.n_q_grid);
            for (int i = 0; i < vbs.n_q_grid; ++i) {
                for (int j = 0; j < vbs.n_q_grid; ++j) {
                    flat[i * vbs.n_q_grid + j] = vbs.S_d_heis_2d[i][j];
                }
            }
            dataset.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        // Save orientation-resolved S_D^{αβ}(q) - 6 unique components per k-point
        // Index: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,1), 4=(1,2), 5=(2,2)
        if (!vbs.S_d_xy_oriented.empty()) {
            int n_k = vbs.S_d_xy_oriented.size();
            
            // XY orientation-resolved
            {
                std::vector<Complex> flat_data(n_k * 6);
                for (int ik = 0; ik < n_k; ++ik) {
                    for (int c = 0; c < 6; ++c) {
                        flat_data[ik * 6 + c] = vbs.S_d_xy_oriented[ik][c];
                    }
                }
                hsize_t dims[2] = {static_cast<hsize_t>(n_k), 6};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet("S_D_q_xy_oriented", complex_type, dataspace);
                dataset.write(flat_data.data(), complex_type);
            }
            
            // Heisenberg orientation-resolved
            {
                std::vector<double> flat_data(n_k * 6);
                for (int ik = 0; ik < n_k; ++ik) {
                    for (int c = 0; c < 6; ++c) {
                        flat_data[ik * 6 + c] = vbs.S_d_heis_oriented[ik][c];
                    }
                }
                hsize_t dims[2] = {static_cast<hsize_t>(n_k), 6};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet("S_D_q_heis_oriented", H5::PredType::NATIVE_DOUBLE, dataspace);
                dataset.write(flat_data.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // n_bonds_per_orientation
            {
                hsize_t dims[1] = {3};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = file.createDataSet("n_bonds_per_orientation", H5::PredType::NATIVE_INT, dataspace);
                dataset.write(vbs.n_bonds_per_orientation.data(), H5::PredType::NATIVE_INT);
            }
            
            std::cout << "  Orientation-resolved S_D^{αβ}(q) saved: " << n_k << " x 6" << std::endl;
        }
        
        // Save bond-resolved dimer correlation matrices
        if (!vbs.dimer_corr_xy.empty()) {
            int nb = vbs.n_bonds;
            
            // XY dimer-dimer correlation matrix ⟨D_b D_b'⟩
            {
                hsize_t dims[2] = {static_cast<hsize_t>(nb), static_cast<hsize_t>(nb)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet("dimer_corr_xy", complex_type, dataspace);
                std::vector<Complex> flat(nb * nb);
                for (int i = 0; i < nb; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        flat[i * nb + j] = vbs.dimer_corr_xy[i][j];
                    }
                }
                dataset.write(flat.data(), complex_type);
            }
            
            // XY connected correlation matrix ⟨δD_b δD_b'⟩
            {
                hsize_t dims[2] = {static_cast<hsize_t>(nb), static_cast<hsize_t>(nb)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet("connected_corr_xy", complex_type, dataspace);
                std::vector<Complex> flat(nb * nb);
                for (int i = 0; i < nb; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        flat[i * nb + j] = vbs.connected_corr_xy[i][j];
                    }
                }
                dataset.write(flat.data(), complex_type);
            }
            
            // Heisenberg dimer-dimer correlation matrix ⟨(S·S)_b (S·S)_b'⟩
            {
                hsize_t dims[2] = {static_cast<hsize_t>(nb), static_cast<hsize_t>(nb)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet("dimer_corr_heis", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> flat(nb * nb);
                for (int i = 0; i < nb; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        flat[i * nb + j] = vbs.dimer_corr_heis[i][j];
                    }
                }
                dataset.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // Heisenberg connected correlation matrix
            {
                hsize_t dims[2] = {static_cast<hsize_t>(nb), static_cast<hsize_t>(nb)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = file.createDataSet("connected_corr_heis", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> flat(nb * nb);
                for (int i = 0; i < nb; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        flat[i * nb + j] = vbs.connected_corr_heis[i][j];
                    }
                }
                dataset.write(flat.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            std::cout << "Saved bond-resolved dimer correlations: " << nb << " x " << nb << " matrices" << std::endl;
        }
        
        // Save k-points
        {
            hsize_t dims[2] = {cluster.k_points.size(), 2};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("k_points", H5::PredType::NATIVE_DOUBLE, dataspace);
            std::vector<double> k_flat(cluster.k_points.size() * 2);
            for (size_t i = 0; i < cluster.k_points.size(); ++i) {
                k_flat[2 * i] = cluster.k_points[i][0];
                k_flat[2 * i + 1] = cluster.k_points[i][1];
            }
            dataset.write(k_flat.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        // Save q-grid parameters for 2D visualization
        {
            std::vector<double> q_vals(n_q_grid);
            for (int i = 0; i < n_q_grid; ++i) {
                q_vals[i] = -1.0 + 2.0 * i / (n_q_grid - 1);
            }
            hsize_t dims[1] = {static_cast<hsize_t>(n_q_grid)};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("q_grid_vals", H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(q_vals.data(), H5::PredType::NATIVE_DOUBLE);
        }
        
        // Save scalar results as attributes
        H5::Group root = file.openGroup("/");
        
        {
            H5::DataSpace scalar(H5S_SCALAR);
            
            auto write_scalar = [&](const std::string& name, double val) {
                H5::Attribute attr = root.createAttribute(name, H5::PredType::NATIVE_DOUBLE, scalar);
                attr.write(H5::PredType::NATIVE_DOUBLE, &val);
            };
            
            auto write_int = [&](const std::string& name, int val) {
                H5::Attribute attr = root.createAttribute(name, H5::PredType::NATIVE_INT, scalar);
                attr.write(H5::PredType::NATIVE_INT, &val);
            };
            
            // Translation order
            write_scalar("m_translation", sf.m_translation);
            write_scalar("s_q_max", std::abs(sf.s_q_max));
            write_int("q_max_idx", sf.q_max_idx);
            write_scalar("q_max_x", sf.q_max[0]);
            write_scalar("q_max_y", sf.q_max[1]);
            
            // Nematic order
            // Nematic order (XY - default)
            write_scalar("m_nematic", nem.m_nem);
            write_scalar("nematic_anisotropy", nem.anisotropy);
            
            // Nematic order variants
            write_scalar("m_nematic_spsm", nem_spsm.m_nem);
            write_scalar("nematic_anisotropy_spsm", nem_spsm.anisotropy);
            write_scalar("m_nematic_szsz", nem_szsz.m_nem);
            write_scalar("nematic_anisotropy_szsz", nem_szsz.anisotropy);
            write_scalar("m_nematic_heisenberg", nem_heisenberg.m_nem);
            write_scalar("nematic_anisotropy_heisenberg", nem_heisenberg.anisotropy);
            
            // VBS order (proper 4-site correlations)
            // XY dimer VBS
            write_scalar("m_vbs_xy", vbs.m_vbs_xy);
            write_scalar("D_mean_xy", vbs.D_mean_xy);
            write_scalar("s_d_max_xy", std::abs(vbs.s_d_max_xy));
            write_int("vbs_q_max_idx_xy", vbs.q_max_idx_xy);
            write_scalar("vbs_q_max_x_xy", vbs.q_max_xy[0]);
            write_scalar("vbs_q_max_y_xy", vbs.q_max_xy[1]);
            
            // Heisenberg dimer VBS
            write_scalar("m_vbs_heis", vbs.m_vbs_heis);
            write_scalar("D_mean_heis", vbs.D_mean_heis);
            write_scalar("s_d_max_heis", std::abs(vbs.s_d_max_heis));
            write_int("vbs_q_max_idx_heis", vbs.q_max_idx_heis);
            write_scalar("vbs_q_max_x_heis", vbs.q_max_heis[0]);
            write_scalar("vbs_q_max_y_heis", vbs.q_max_heis[1]);
            
            // Backward compatibility (default to XY)
            write_scalar("m_vbs", vbs.m_vbs_xy);
            write_scalar("D_mean", vbs.D_mean_xy);
            
            // Cluster info
            write_int("n_sites", cluster.n_sites);
            write_int("n_bonds", static_cast<int>(cluster.edges_nn.size()));
            write_int("n_q_grid", n_q_grid);
        }
        
        // Save bond data for visualization
        if (!spsm_bonds.empty()) {
            H5::Group bonds_group = file.createGroup("/bonds");
            
            // Save site positions
            {
                hsize_t dims[2] = {static_cast<hsize_t>(cluster.n_sites), 2};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = bonds_group.createDataSet("positions", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> pos_flat(cluster.n_sites * 2);
                for (int i = 0; i < cluster.n_sites; ++i) {
                    pos_flat[2 * i] = cluster.positions[i][0];
                    pos_flat[2 * i + 1] = cluster.positions[i][1];
                }
                dataset.write(pos_flat.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // Save edge list
            int n_bonds = cluster.edges_nn.size();
            {
                hsize_t dims[2] = {static_cast<hsize_t>(n_bonds), 2};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet dataset = bonds_group.createDataSet("edges", H5::PredType::NATIVE_INT, dataspace);
                std::vector<int> edge_flat(n_bonds * 2);
                int e = 0;
                for (const auto& [i, j] : cluster.edges_nn) {
                    edge_flat[2 * e] = i;
                    edge_flat[2 * e + 1] = j;
                    e++;
                }
                dataset.write(edge_flat.data(), H5::PredType::NATIVE_INT);
            }
            
            // Save S+S- bond expectations
            {
                hsize_t dims[1] = {static_cast<hsize_t>(n_bonds)};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = bonds_group.createDataSet("spsm", complex_type, dataspace);
                std::vector<Complex> vals(n_bonds);
                int e = 0;
                for (const auto& edge : cluster.edges_nn) {
                    vals[e++] = spsm_bonds.at(edge);
                }
                dataset.write(vals.data(), complex_type);
            }
            
            // Save SzSz bond expectations
            if (!szsz_bonds.empty()) {
                hsize_t dims[1] = {static_cast<hsize_t>(n_bonds)};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = bonds_group.createDataSet("szsz", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> vals(n_bonds);
                int e = 0;
                for (const auto& edge : cluster.edges_nn) {
                    vals[e++] = szsz_bonds.at(edge);
                }
                dataset.write(vals.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // Save Heisenberg S·S bond expectations
            if (!heisenberg_bonds.empty()) {
                hsize_t dims[1] = {static_cast<hsize_t>(n_bonds)};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = bonds_group.createDataSet("heisenberg", H5::PredType::NATIVE_DOUBLE, dataspace);
                std::vector<double> vals(n_bonds);
                int e = 0;
                for (const auto& edge : cluster.edges_nn) {
                    vals[e++] = heisenberg_bonds.at(edge);
                }
                dataset.write(vals.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // Save bond orientations (0, 1, 2 for kagome)
            {
                hsize_t dims[1] = {static_cast<hsize_t>(n_bonds)};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = bonds_group.createDataSet("orientation", H5::PredType::NATIVE_INT, dataspace);
                std::vector<int> vals(n_bonds);
                int e = 0;
                for (const auto& edge : cluster.edges_nn) {
                    vals[e++] = cluster.bond_orientation.at(edge);
                }
                dataset.write(vals.data(), H5::PredType::NATIVE_INT);
            }
            
            // Print bond statistics
            double spsm_sum = 0, szsz_sum = 0, heis_sum = 0;
            for (const auto& edge : cluster.edges_nn) {
                spsm_sum += std::real(spsm_bonds.at(edge));
                if (!szsz_bonds.empty()) szsz_sum += szsz_bonds.at(edge);
                if (!heisenberg_bonds.empty()) heis_sum += heisenberg_bonds.at(edge);
            }
            std::cout << "Bond expectations saved:" << std::endl;
            std::cout << "  <S+S->_avg = " << spsm_sum / n_bonds << std::endl;
            std::cout << "  <SzSz>_avg = " << szsz_sum / n_bonds << std::endl;
            std::cout << "  <S·S>_avg  = " << heis_sum / n_bonds << std::endl;
        }
        
        // =========================================================================
        // Save plaquette/bowtie resonance data
        // =========================================================================
        if (plaq.n_plaquettes > 0) {
            H5::Group plaq_group = file.createGroup("/plaquette");
            
            // Scalars
            {
                hsize_t dims[1] = {1};
                H5::DataSpace dataspace(1, dims);
                
                H5::DataSet ds = plaq_group.createDataSet("m_plaquette", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(&plaq.m_plaquette, H5::PredType::NATIVE_DOUBLE);
                
                ds = plaq_group.createDataSet("P_mean", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(&plaq.P_mean, H5::PredType::NATIVE_DOUBLE);
                
                ds = plaq_group.createDataSet("resonance_strength", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(&plaq.resonance_strength, H5::PredType::NATIVE_DOUBLE);
                
                ds = plaq_group.createDataSet("chi_mean", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(&plaq.chi_mean, H5::PredType::NATIVE_DOUBLE);
                
                ds = plaq_group.createDataSet("n_plaquettes", H5::PredType::NATIVE_INT, dataspace);
                ds.write(&plaq.n_plaquettes, H5::PredType::NATIVE_INT);
                
                ds = plaq_group.createDataSet("n_triangles", H5::PredType::NATIVE_INT, dataspace);
                ds.write(&plaq.n_triangles, H5::PredType::NATIVE_INT);
            }
            
            // q_max
            {
                hsize_t dims[1] = {2};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet("q_max", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(plaq.q_max.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // S_P(q) at k-points
            if (!plaq.S_p.empty()) {
                hsize_t dims[1] = {plaq.S_p.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet("S_p", complex_type, dataspace);
                ds.write(plaq.S_p.data(), complex_type);
            }
            
            // S_P(q) 2D grid
            if (!plaq.S_p_2d.empty()) {
                int grid_size = plaq.S_p_2d.size();
                std::vector<Complex> flat_data(grid_size * grid_size);
                for (int i = 0; i < grid_size; ++i) {
                    for (int j = 0; j < grid_size; ++j) {
                        flat_data[i * grid_size + j] = plaq.S_p_2d[i][j];
                    }
                }
                hsize_t dims[2] = {static_cast<hsize_t>(grid_size), static_cast<hsize_t>(grid_size)};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet ds = plaq_group.createDataSet("S_p_2d", complex_type, dataspace);
                ds.write(flat_data.data(), complex_type);
            }
            
            // Orientation-resolved S_P^{αβ}(q) - 6 unique components per k-point
            // Index: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,1), 4=(1,2), 5=(2,2)
            if (!plaq.S_p_oriented.empty()) {
                int n_k = plaq.S_p_oriented.size();
                std::vector<Complex> flat_data(n_k * 6);
                for (int ik = 0; ik < n_k; ++ik) {
                    for (int c = 0; c < 6; ++c) {
                        flat_data[ik * 6 + c] = plaq.S_p_oriented[ik][c];
                    }
                }
                hsize_t dims[2] = {static_cast<hsize_t>(n_k), 6};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet ds = plaq_group.createDataSet("S_p_oriented", complex_type, dataspace);
                ds.write(flat_data.data(), complex_type);
                
                // Also save n_plaquettes_per_orientation
                hsize_t dims3[1] = {3};
                H5::DataSpace ds3(1, dims3);
                H5::DataSet orient_ds = plaq_group.createDataSet("n_plaquettes_per_orientation", H5::PredType::NATIVE_INT, ds3);
                orient_ds.write(plaq.n_plaquettes_per_orientation.data(), H5::PredType::NATIVE_INT);
                
                std::cout << "  Orientation-resolved S_P^{αβ}(q) saved: " << n_k << " x 6" << std::endl;
            }
            
            // Plaquette orientations
            if (!plaq.orientations.empty()) {
                hsize_t dims[1] = {plaq.orientations.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet("orientations", H5::PredType::NATIVE_INT, dataspace);
                ds.write(plaq.orientations.data(), H5::PredType::NATIVE_INT);
            }
            
            // Individual plaquette resonances
            if (!plaq.P_r.empty()) {
                hsize_t dims[1] = {plaq.P_r.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet("P_r", complex_type, dataspace);
                ds.write(plaq.P_r.data(), complex_type);
            }
            
            // Plaquette centers
            if (!plaq.centers.empty()) {
                std::vector<double> flat_centers(plaq.centers.size() * 2);
                for (size_t i = 0; i < plaq.centers.size(); ++i) {
                    flat_centers[2*i] = plaq.centers[i][0];
                    flat_centers[2*i + 1] = plaq.centers[i][1];
                }
                hsize_t dims[2] = {plaq.centers.size(), 2};
                H5::DataSpace dataspace(2, dims);
                H5::DataSet ds = plaq_group.createDataSet("centers", H5::PredType::NATIVE_DOUBLE, dataspace);
                ds.write(flat_centers.data(), H5::PredType::NATIVE_DOUBLE);
            }
            
            // Triangle chiral expectations
            if (!plaq.chi_r.empty()) {
                hsize_t dims[1] = {plaq.chi_r.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet ds = plaq_group.createDataSet("chi_r", complex_type, dataspace);
                ds.write(plaq.chi_r.data(), complex_type);
            }
            
            std::cout << "Plaquette data saved:" << std::endl;
            std::cout << "  n_plaquettes = " << plaq.n_plaquettes << std::endl;
            std::cout << "  n_triangles = " << plaq.n_triangles << std::endl;
            std::cout << "  m_plaquette = " << plaq.m_plaquette << std::endl;
        }
        
        std::cout << "Results saved to: " << filename << std::endl;
        
    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 write error: " + std::string(e.getCDetailMsg()));
    }
}

// -----------------------------------------------------------------------------
// Results structure for scan mode
// -----------------------------------------------------------------------------

struct OrderParameterResults {
    double jpm;
    double temperature = 0.0;  // For TPQ mode, T=0 for ground state
    double m_translation;
    double m_nematic;
    double m_nematic_spsm;
    double m_nematic_szsz;
    double m_nematic_heisenberg;
    double m_vbs;           // XY VBS (for backward compatibility)
    double m_vbs_xy;        // XY VBS (explicit)
    double m_vbs_heis;      // Heisenberg VBS
    double anisotropy;
    double D_mean;          // XY dimer mean (backward compat)
    double D_mean_xy;
    double D_mean_heis;
    // Plaquette/bowtie resonance order
    double m_plaquette;
    double P_mean;           // Mean bowtie resonance
    double resonance_strength;  // Mean |P_bt|
    double chi_mean;         // Mean triangle chiral
    int n_plaquettes;
    int n_triangles;
};

// -----------------------------------------------------------------------------
// Compute all order parameters (for scan mode)
// -----------------------------------------------------------------------------

OrderParameterResults compute_all_order_parameters(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    double jpm_value
) {
    OrderParameterResults results;
    results.jpm = jpm_value;
    
    // Compute correlations
    auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);

    // Structure factor
    auto sf_result = compute_spin_structure_factor(smsp_corr, cluster);
    results.m_translation = sf_result.m_translation;
    
    // Bond expectations
    auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
    auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
    auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
    auto heisenberg_bond_exp = compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);
    
    // Nematic (all variants)
    auto nem_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
    auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
    auto nem_szsz_result = compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
    auto nem_heis_result = compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");
    
    results.m_nematic = nem_result.m_nem;
    results.anisotropy = nem_result.anisotropy;
    results.m_nematic_spsm = nem_spsm_result.m_nem;
    results.m_nematic_szsz = nem_szsz_result.m_nem;
    results.m_nematic_heisenberg = nem_heis_result.m_nem;
    
    // VBS order (proper 4-site correlations, both XY and Heisenberg)
    auto vbs_result = compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster);
    results.m_vbs = vbs_result.m_vbs_xy;
    results.m_vbs_xy = vbs_result.m_vbs_xy;
    results.m_vbs_heis = vbs_result.m_vbs_heis;
    results.D_mean = vbs_result.D_mean_xy;
    results.D_mean_xy = vbs_result.D_mean_xy;
    results.D_mean_heis = vbs_result.D_mean_heis;
    
    // Plaquette/bowtie resonance order
    auto plaq_result = compute_plaquette_order(psi, cluster);
    results.m_plaquette = plaq_result.m_plaquette;
    results.P_mean = plaq_result.P_mean;
    results.resonance_strength = plaq_result.resonance_strength;
    results.chi_mean = plaq_result.chi_mean;
    results.n_plaquettes = plaq_result.n_plaquettes;
    results.n_triangles = plaq_result.n_triangles;
    
    return results;
}

// -----------------------------------------------------------------------------
// Scan directory mode
// -----------------------------------------------------------------------------

std::vector<OrderParameterResults> scan_jpm_directories(
    const std::string& scan_dir,
    const std::string& output_dir,
    int n_workers,
    int n_q_grid,
    bool save_full,
    bool use_tpq
) {
    // Get MPI rank and size
    int mpi_rank = 0, mpi_size = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    
    // Find all Jpm=* directories (all ranks do this)
    std::vector<std::pair<double, std::string>> jpm_dirs;
    std::regex jpm_regex("Jpm=([+-]?[0-9]*\\.?[0-9]+)");
    
    for (const auto& entry : fs::directory_iterator(scan_dir)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        std::smatch match;
        if (std::regex_search(dirname, match, jpm_regex)) {
            double jpm = std::stod(match[1].str());
            jpm_dirs.push_back({jpm, entry.path().string()});
        }
    }
    
    std::sort(jpm_dirs.begin(), jpm_dirs.end());
    
    if (mpi_rank == 0) {
        std::cout << "Found " << jpm_dirs.size() << " Jpm directories" << std::endl;
#ifdef USE_MPI
        std::cout << "MPI enabled: " << mpi_size << " processes" << std::endl;
#endif
        if (save_full) {
            std::cout << "Full output mode: saving S(q) and S_D(q) 2D grids per directory" << std::endl;
        }
    }
    
    if (jpm_dirs.empty()) {
        return {};
    }
    
    // Load cluster from first directory
    Cluster cluster = load_cluster(jpm_dirs[0].second);
    if (mpi_rank == 0) {
        std::cout << "Cluster: " << cluster.n_sites << " sites, " 
                  << cluster.edges_nn.size() << " bonds" << std::endl;
    }
    
    // Process directories - each MPI rank processes a subset
    std::vector<OrderParameterResults> all_results(jpm_dirs.size());
    std::mutex print_mutex;
    std::atomic<int> completed(0);
    
    // Distribute work: rank processes indices where (i % mpi_size == mpi_rank)
    #pragma omp parallel for schedule(dynamic) num_threads(n_workers)
    for (size_t i = 0; i < jpm_dirs.size(); ++i) {
        // MPI work distribution: only process if this index belongs to this rank
#ifdef USE_MPI
        if (static_cast<int>(i % mpi_size) != mpi_rank) {
            continue;
        }
#endif
        double jpm = jpm_dirs[i].first;
        const std::string& dir = jpm_dirs[i].second;
        
        try {
            // Find wavefunction file
            std::string wf_file;
            
            // First check output/ subdirectory for ed_results.h5
            std::string output_subdir = dir + "/output";
            if (fs::exists(output_subdir)) {
                std::string ed_results = output_subdir + "/ed_results.h5";
                if (fs::exists(ed_results)) {
                    wf_file = ed_results;
                } else {
                    // Search for any .h5 file in output/
                    for (const auto& entry : fs::directory_iterator(output_subdir)) {
                        if (entry.path().extension() == ".h5") {
                            wf_file = entry.path().string();
                            break;
                        }
                    }
                }
            }
            
            // If not found in output/, try main directory
            if (wf_file.empty()) {
                for (const auto& entry : fs::directory_iterator(dir)) {
                    std::string fname = entry.path().filename().string();
                    if (fname.find(".h5") != std::string::npos && 
                        fname.find("eigenvector") != std::string::npos) {
                        wf_file = entry.path().string();
                        break;
                    }
                }
            }
            
            if (wf_file.empty()) {
                // Try other patterns in main directory
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().extension() == ".h5") {
                        wf_file = entry.path().string();
                        break;
                    }
                }
            }
            
            if (wf_file.empty()) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "No wavefunction file found in " << dir << std::endl;
                continue;
            }
            
            // Load wavefunction (ground state or TPQ lowest temperature)
            std::vector<Complex> psi;
            double temperature = 0.0;
            
            if (use_tpq) {
                try {
                    auto [tpq_psi, T] = load_tpq_state(wf_file);
                    psi = std::move(tpq_psi);
                    temperature = T;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cerr << "TPQ load failed for " << dir << ": " << e.what() 
                              << ", falling back to ground state" << std::endl;
                    psi = load_wavefunction(wf_file);
                }
            } else {
                psi = load_wavefunction(wf_file);
            }
            
            // Set memory-efficient mode on first file processed by this thread
            // (all files should have same Hilbert space size)
            #pragma omp critical(set_memory_mode)
            {
                static bool memory_mode_set = false;
                if (!memory_mode_set) {
                    set_memory_efficient_mode(psi.size());
                    memory_mode_set = true;
                }
            }
            
            OrderParameterResults results;
            results.jpm = jpm;
            results.temperature = temperature;
            
            if (save_full) {
                // Full computation with 2D grids
                auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);
                auto sf_result = compute_spin_structure_factor(smsp_corr, cluster);
                auto s_q_2d = compute_sq_2d_grid(smsp_corr, cluster, n_q_grid);
                
                auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
                auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
                auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
                auto heisenberg_bond_exp = compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);
                
                auto nem_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
                auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
                auto nem_szsz_result = compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
                auto nem_heisenberg_result = compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");
                
                auto vbs_result = compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster, n_q_grid);
                auto plaq_result = compute_plaquette_order(psi, cluster, n_q_grid);
                
                // Save full results to per-Jpm file
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << jpm;
                std::string out_file = output_dir + "/order_params_Jpm=" + oss.str() + ".h5";
                save_results(out_file, sf_result, nem_result, nem_spsm_result, nem_szsz_result,
                            nem_heisenberg_result, vbs_result, plaq_result,
                            cluster, s_q_2d, n_q_grid,
                            spsm_bond_exp, szsz_bond_exp, heisenberg_bond_exp);
                
                // Fill scalar results for summary
                results.m_translation = sf_result.m_translation;
                results.m_nematic = nem_result.m_nem;
                results.m_nematic_spsm = nem_spsm_result.m_nem;
                results.m_nematic_szsz = nem_szsz_result.m_nem;
                results.m_nematic_heisenberg = nem_heisenberg_result.m_nem;
                results.anisotropy = nem_result.anisotropy;
                results.m_vbs = vbs_result.m_vbs_xy;
                results.m_vbs_xy = vbs_result.m_vbs_xy;
                results.m_vbs_heis = vbs_result.m_vbs_heis;
                results.D_mean = vbs_result.D_mean_xy;
                results.D_mean_xy = vbs_result.D_mean_xy;
                results.D_mean_heis = vbs_result.D_mean_heis;
                results.m_plaquette = plaq_result.m_plaquette;
                results.P_mean = plaq_result.P_mean;
                results.resonance_strength = plaq_result.resonance_strength;
                results.chi_mean = plaq_result.chi_mean;
                results.n_plaquettes = plaq_result.n_plaquettes;
                results.n_triangles = plaq_result.n_triangles;
                results.D_mean_xy = vbs_result.D_mean_xy;
                results.D_mean_heis = vbs_result.D_mean_heis;
            } else {
                // Quick scalar-only computation
                results = compute_all_order_parameters(psi, cluster, jpm);
                results.temperature = temperature;  // Preserve temperature from TPQ
            }
            
            all_results[i] = results;
            
            int done = ++completed;
            {
                std::lock_guard<std::mutex> lock(print_mutex);
#ifdef USE_MPI
                std::cout << "[Rank " << mpi_rank << "] ";
#endif
                std::cout << "[" << done << "/" << jpm_dirs.size() << "] "
                          << "Jpm=" << std::fixed << std::setprecision(4) << jpm;
                if (use_tpq) {
                    std::cout << " T=" << std::setprecision(6) << results.temperature;
                }
                std::cout << " | m_trans=" << std::setprecision(6) << results.m_translation
                          << " | m_nem=" << results.m_nematic
                          << " | m_vbs_xy=" << results.m_vbs_xy
                          << " | m_vbs_heis=" << results.m_vbs_heis
                          << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(print_mutex);
#ifdef USE_MPI
            std::cerr << "[Rank " << mpi_rank << "] ";
#endif
            std::cerr << "Error processing " << dir << ": " << e.what() << std::endl;
        }
    }
    
#ifdef USE_MPI
    // Gather results from all ranks to rank 0
    if (mpi_rank == 0) {
        std::cout << "\nGathering results from all MPI ranks..." << std::endl;
    }
    
    // Create a flat buffer for MPI communication (just the scalar values)
    std::vector<double> local_buffer;
    std::vector<int> local_indices;
    
    for (size_t i = 0; i < jpm_dirs.size(); ++i) {
        if (static_cast<int>(i % mpi_size) == mpi_rank && all_results[i].jpm != 0.0) {
            local_indices.push_back(i);
            local_buffer.push_back(all_results[i].jpm);
            local_buffer.push_back(all_results[i].temperature);
            local_buffer.push_back(all_results[i].m_translation);
            local_buffer.push_back(all_results[i].m_nematic);
            local_buffer.push_back(all_results[i].m_nematic_spsm);
            local_buffer.push_back(all_results[i].m_nematic_szsz);
            local_buffer.push_back(all_results[i].m_nematic_heisenberg);
            local_buffer.push_back(all_results[i].m_vbs);
            local_buffer.push_back(all_results[i].m_vbs_xy);
            local_buffer.push_back(all_results[i].m_vbs_heis);
            local_buffer.push_back(all_results[i].anisotropy);
            local_buffer.push_back(all_results[i].D_mean);
            local_buffer.push_back(all_results[i].D_mean_xy);
            local_buffer.push_back(all_results[i].D_mean_heis);
        }
    }
    
    // Gather all results to rank 0
    std::vector<int> recv_counts(mpi_size);
    int local_count = local_buffer.size();
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> displs;
    std::vector<double> recv_buffer;
    if (mpi_rank == 0) {
        displs.resize(mpi_size);
        int total = 0;
        for (int i = 0; i < mpi_size; ++i) {
            displs[i] = total;
            total += recv_counts[i];
        }
        recv_buffer.resize(total);
    }
    
    MPI_Gatherv(local_buffer.data(), local_count, MPI_DOUBLE,
                recv_buffer.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Gather indices
    std::vector<int> all_indices;
    std::vector<int> index_counts(mpi_size);
    int local_idx_count = local_indices.size();
    MPI_Gather(&local_idx_count, 1, MPI_INT, index_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> idx_displs;
    if (mpi_rank == 0) {
        idx_displs.resize(mpi_size);
        int total = 0;
        for (int i = 0; i < mpi_size; ++i) {
            idx_displs[i] = total;
            total += index_counts[i];
        }
        all_indices.resize(total);
    }
    
    MPI_Gatherv(local_indices.data(), local_idx_count, MPI_INT,
                all_indices.data(), index_counts.data(), idx_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Rank 0 unpacks the results
    if (mpi_rank == 0) {
        const int vals_per_result = 14;
        for (size_t i = 0; i < all_indices.size(); ++i) {
            int idx = all_indices[i];
            size_t offset = i * vals_per_result;
            all_results[idx].jpm = recv_buffer[offset + 0];
            all_results[idx].temperature = recv_buffer[offset + 1];
            all_results[idx].m_translation = recv_buffer[offset + 2];
            all_results[idx].m_nematic = recv_buffer[offset + 3];
            all_results[idx].m_nematic_spsm = recv_buffer[offset + 4];
            all_results[idx].m_nematic_szsz = recv_buffer[offset + 5];
            all_results[idx].m_nematic_heisenberg = recv_buffer[offset + 6];
            all_results[idx].m_vbs = recv_buffer[offset + 7];
            all_results[idx].m_vbs_xy = recv_buffer[offset + 8];
            all_results[idx].m_vbs_heis = recv_buffer[offset + 9];
            all_results[idx].anisotropy = recv_buffer[offset + 10];
            all_results[idx].D_mean = recv_buffer[offset + 11];
            all_results[idx].D_mean_xy = recv_buffer[offset + 12];
            all_results[idx].D_mean_heis = recv_buffer[offset + 13];
        }
    }
#endif
    
    return all_results;
}

// -----------------------------------------------------------------------------
// Save scan results to HDF5
// -----------------------------------------------------------------------------

void save_scan_results(
    const std::vector<OrderParameterResults>& results,
    const std::string& output_file
) {
    try {
        H5::H5File file(output_file, H5F_ACC_TRUNC);
        
        size_t n = results.size();
        std::vector<double> jpm_vals(n), temperature_vals(n);
        std::vector<double> m_trans(n);
        std::vector<double> m_nem(n), m_nem_spsm(n), m_nem_szsz(n), m_nem_heis(n), aniso(n);
        std::vector<double> m_vbs_vals(n), m_vbs_xy_vals(n), m_vbs_heis_vals(n);
        std::vector<double> D_mean_vals(n), D_mean_xy_vals(n), D_mean_heis_vals(n);
        std::vector<double> m_plaq(n), P_mean_vals(n), res_strength(n), chi_mean_vals(n);
        std::vector<int> n_plaq(n), n_tri(n);
        
        for (size_t i = 0; i < n; ++i) {
            jpm_vals[i] = results[i].jpm;
            temperature_vals[i] = results[i].temperature;
            m_trans[i] = results[i].m_translation;
            
            m_nem[i] = results[i].m_nematic;
            m_nem_spsm[i] = results[i].m_nematic_spsm;
            m_nem_szsz[i] = results[i].m_nematic_szsz;
            m_nem_heis[i] = results[i].m_nematic_heisenberg;
            aniso[i] = results[i].anisotropy;
            
            m_vbs_vals[i] = results[i].m_vbs;
            m_vbs_xy_vals[i] = results[i].m_vbs_xy;
            m_vbs_heis_vals[i] = results[i].m_vbs_heis;
            
            D_mean_vals[i] = results[i].D_mean;
            D_mean_xy_vals[i] = results[i].D_mean_xy;
            D_mean_heis_vals[i] = results[i].D_mean_heis;
            
            m_plaq[i] = results[i].m_plaquette;
            P_mean_vals[i] = results[i].P_mean;
            res_strength[i] = results[i].resonance_strength;
            chi_mean_vals[i] = results[i].chi_mean;
            n_plaq[i] = results[i].n_plaquettes;
            n_tri[i] = results[i].n_triangles;
        }
        
        auto write_dataset = [&](const std::string& name, const std::vector<double>& data) {
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        
        auto write_dataset_int = [&](const std::string& name, const std::vector<int>& data) {
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(name, H5::PredType::NATIVE_INT, dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_INT);
        };
        
        write_dataset("jpm_values", jpm_vals);
        write_dataset("temperature", temperature_vals);
        write_dataset("m_translation", m_trans);
        
        // Nematic order (all variants)
        write_dataset("m_nematic", m_nem);
        write_dataset("m_nematic_spsm", m_nem_spsm);
        write_dataset("m_nematic_szsz", m_nem_szsz);
        write_dataset("m_nematic_heisenberg", m_nem_heis);
        write_dataset("anisotropy", aniso);
        
        // VBS order (both XY and Heisenberg)
        write_dataset("m_vbs", m_vbs_vals);
        write_dataset("m_vbs_xy", m_vbs_xy_vals);
        write_dataset("m_vbs_heis", m_vbs_heis_vals);
        write_dataset("D_mean", D_mean_vals);
        write_dataset("D_mean_xy", D_mean_xy_vals);
        write_dataset("D_mean_heis", D_mean_heis_vals);
        
        // Plaquette/bowtie resonance order
        write_dataset("m_plaquette", m_plaq);
        write_dataset("P_mean", P_mean_vals);
        write_dataset("resonance_strength", res_strength);
        write_dataset("chi_mean", chi_mean_vals);
        write_dataset_int("n_plaquettes", n_plaq);
        write_dataset_int("n_triangles", n_tri);
        
        std::cout << "Scan results saved to: " << output_file << std::endl;
        
    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 write error: " + std::string(e.getCDetailMsg()));
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  Single file mode:\n"
              << "    " << prog << " <wavefunction.h5> <cluster_dir> [output.h5]\n\n"
              << "  Scan directory mode:\n"
              << "    " << prog << " --scan-dir <dir> [options]\n\n"
              << "Options:\n"
              << "  --scan-dir <dir>     Directory containing Jpm=* subdirectories\n"
              << "  --output-dir <dir>   Output directory for results\n"
              << "  --n-workers <n>      Number of parallel workers (default: 4)\n"
              << "  --n-q-grid <n>       2D q-grid size for visualization (default: 50)\n"
              << "  --save-full          Save full S(q), S_D(q), S_P(q) 2D grids and bond-resolved data per Jpm\n"
              << "  --tpq                Use TPQ states (lowest temperature) instead of ground state\n"
              << "\nComputes BFG order parameters from ground state or TPQ wavefunction:\n"
              << "  1. S(q) - Spin structure factor (translation order)\n"
              << "  2. Nematic order - Bond orientation anisotropy (C3 breaking)\n"
              << "     - Variants: XY, S+S-, SzSz, Heisenberg\n"
              << "  3. VBS order - Valence bond solid with proper 4-site dimer correlations\n"
              << "     - Variants: XY dimer (S+S- + S-S+), Heisenberg dimer (S·S)\n"
              << "     - Bond-resolved: full dimer-dimer correlation matrices\n"
              << "  4. Plaquette/bowtie resonance order - BFG-native 4-spin ring flip\n"
              << "     - S_P(q) structure factor for bowtie plaquettes\n"
              << "     - Triangle chiral order\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif
    
    if (argc < 2) {
        if (mpi_rank == 0) {
            print_usage(argv[0]);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    std::string scan_dir, output_dir;
    std::string wf_file, cluster_dir, output_file = "bfg_order_parameters.h5";
    int n_workers = 4;
    int n_q_grid = 50;
    bool scan_mode = false;
    bool save_full = false;
    bool use_tpq = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scan-dir" && i + 1 < argc) {
            scan_dir = argv[++i];
            scan_mode = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--n-workers" && i + 1 < argc) {
            n_workers = std::stoi(argv[++i]);
        } else if (arg == "--n-q-grid" && i + 1 < argc) {
            n_q_grid = std::stoi(argv[++i]);
        } else if (arg == "--save-full") {
            save_full = true;
        } else if (arg == "--tpq") {
            use_tpq = true;
        } else if (arg == "-h" || arg == "--help") {
            if (mpi_rank == 0) {
                print_usage(argv[0]);
            }
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        } else if (wf_file.empty()) {
            wf_file = arg;
        } else if (cluster_dir.empty()) {
            cluster_dir = arg;
        } else {
            output_file = arg;
        }
    }
    
    #ifdef _OPENMP
    if (mpi_rank == 0) {
        std::cout << "OpenMP enabled with " << omp_get_max_threads() << " max threads" << std::endl;
    }
    #endif
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    try {
        if (scan_mode) {
            // Scan directory mode
            if (scan_dir.empty()) {
                if (mpi_rank == 0) {
                    std::cerr << "Error: --scan-dir required in scan mode" << std::endl;
                }
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 1;
            }
            if (output_dir.empty()) {
                output_dir = scan_dir + "/order_parameter_results";
            }
            
            // Only rank 0 creates output directory
            if (mpi_rank == 0) {
                fs::create_directories(output_dir);
                
                std::cout << "========================================\n"
                          << "BFG ORDER PARAMETER SCAN (CPU)\n"
                          << "========================================\n"
                          << "Scan directory: " << scan_dir << "\n"
                          << "Output directory: " << output_dir << "\n"
                          << "Workers: " << n_workers << "\n"
                          << "Mode: " << (use_tpq ? "TPQ (lowest temperature)" : "Ground state") << "\n"
                          << "Save full: " << (save_full ? "yes (2D grids)" : "no (scalars only)") << "\n"
                          << "========================================" << std::endl;
            }
            
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);  // Wait for directory creation
#endif
            
            auto results = scan_jpm_directories(scan_dir, output_dir, n_workers, n_q_grid, save_full, use_tpq);
            
            // Only rank 0 saves the combined results
            if (mpi_rank == 0 && !results.empty()) {
                std::string results_file = output_dir + "/scan_results.h5";
                save_scan_results(results, results_file);
            }
            
        } else {
            // Single file mode - only run on rank 0
#ifdef USE_MPI
            if (mpi_rank != 0) {
                MPI_Finalize();
                return 0;  // Other ranks exit
            }
#endif
            
            if (wf_file.empty() || cluster_dir.empty()) {
                print_usage(argv[0]);
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 1;
            }
            
            std::cout << "========================================\n"
                      << "BFG ORDER PARAMETER COMPUTATION (CPU)\n"
                      << "========================================" << std::endl;
            
            Cluster cluster = load_cluster(cluster_dir);
            
            std::vector<Complex> psi;
            double temperature = 0.0;
            
            if (use_tpq) {
                auto [tpq_psi, tpq_temp] = load_tpq_state(wf_file);
                psi = std::move(tpq_psi);
                temperature = tpq_temp;
                std::cout << "Using TPQ state (T=" << temperature << ")" << std::endl;
            } else {
                psi = load_wavefunction(wf_file);
                std::cout << "Using ground state wavefunction" << std::endl;
            }
            
            // Set memory-efficient mode based on Hilbert space size
            set_memory_efficient_mode(psi.size());
            
            // Verify size
            uint64_t expected_size = 1ULL << cluster.n_sites;
            if (psi.size() != expected_size) {
                std::cerr << "Warning: wavefunction size " << psi.size() 
                          << " != expected 2^" << cluster.n_sites << " = " << expected_size << std::endl;
            }
            
            // Compute correlations
            auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);
            
            // Compute S(q) at k-points
            auto sf_result = compute_spin_structure_factor(smsp_corr, cluster);
            
            // Compute 2D S(q) grid for visualization
            auto s_q_2d = compute_sq_2d_grid(smsp_corr, cluster, n_q_grid);
            
            // Compute bond expectations for nematic and VBS
            auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
            
            // Compute additional bond operators for visualization
            auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
            auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
            auto heisenberg_bond_exp = compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);
            
            // Compute nematic order for all three bond types
            auto nem_xy_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
            auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
            auto nem_szsz_result = compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
            auto nem_heisenberg_result = compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");
            
            // Use XY nematic as primary (for backwards compatibility)
            auto nem_result = nem_xy_result;
            
            // VBS order with proper 4-site correlations (with 2D grid)
            auto vbs_result = compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster, n_q_grid);
            
            // Plaquette/bowtie resonance order
            auto plaq_result = compute_plaquette_order(psi, cluster, n_q_grid);
            
            // Save results with full 2D grids and bond data
            save_results(output_file, sf_result, nem_result, nem_spsm_result, nem_szsz_result, 
                        nem_heisenberg_result, vbs_result, plaq_result,
                        cluster, s_q_2d, n_q_grid,
                        spsm_bond_exp, szsz_bond_exp, heisenberg_bond_exp);
            
            // Print summary
            std::cout << "\n========== ORDER PARAMETER SUMMARY ==========" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Translation order:  m = " << sf_result.m_translation 
                      << " at q = (" << sf_result.q_max[0] << ", " << sf_result.q_max[1] << ")" << std::endl;
            std::cout << "Nematic order (XY):         m = " << nem_result.m_nem 
                      << ", anisotropy = " << nem_result.anisotropy << std::endl;
            std::cout << "Nematic order (S+S-):       m = " << nem_spsm_result.m_nem 
                      << ", anisotropy = " << nem_spsm_result.anisotropy << std::endl;
            std::cout << "Nematic order (SzSz):       m = " << nem_szsz_result.m_nem 
                      << ", anisotropy = " << nem_szsz_result.anisotropy << std::endl;
            std::cout << "Nematic order (Heisenberg): m = " << nem_heisenberg_result.m_nem 
                      << ", anisotropy = " << nem_heisenberg_result.anisotropy << std::endl;
            std::cout << "VBS order (XY, 4-site):         m = " << vbs_result.m_vbs_xy 
                      << " at q = (" << vbs_result.q_max_xy[0] << ", " << vbs_result.q_max_xy[1] << ")" << std::endl;
            std::cout << "VBS order (Heisenberg, 4-site): m = " << vbs_result.m_vbs_heis 
                      << " at q = (" << vbs_result.q_max_heis[0] << ", " << vbs_result.q_max_heis[1] << ")" << std::endl;
            std::cout << "Plaquette order (bowtie):   m = " << plaq_result.m_plaquette 
                      << " at q = (" << plaq_result.q_max[0] << ", " << plaq_result.q_max[1] << ")" << std::endl;
            std::cout << "  Resonance: <P> = " << plaq_result.P_mean 
                      << ", strength = " << plaq_result.resonance_strength << std::endl;
            std::cout << "  Triangle chiral: <χ> = " << plaq_result.chi_mean << std::endl;
            std::cout << "==============================================" << std::endl;
        }
        
    } catch (const std::exception& e) {
        if (mpi_rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    if (mpi_rank == 0) {
        std::cout << "\nTotal runtime: " << total_duration.count() << " seconds" << std::endl;
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    return 0;
}

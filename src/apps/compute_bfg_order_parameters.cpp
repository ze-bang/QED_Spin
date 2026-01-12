/**
 * @file compute_bfg_order_parameters.cpp
 * @brief Fast C++ computation of BFG order parameters from wavefunctions
 * 
 * Computes:
 * 1. S(q) - Spin structure factor using S^+S^- correlations at ALL k-points + 2D grid
 * 2. Nematic order - Bond orientation anisotropy (C6 → C2 breaking)
 * 3. Stripe structure factor - Bond-bond correlations with orientation phase  
 * 4. VBS (Valence Bond Solid) order - S_D(q) bond dimer structure factor
 * 5. Plaquette/bowtie resonance - Ring-flip correlations
 * 
 * Output includes:
 * - Order parameters at special k-points (Γ, K, M, etc.)
 * - Full 2D structure factor grids for visualization
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
inline std::pair<int64_t, double> apply_sp(uint64_t state, int site) {
    if (get_bit(state, site) == 0) {  // spin down
        return {static_cast<int64_t>(state | (1ULL << site)), 1.0};
    }
    return {-1, 0.0};
}

// S^- lowers spin: |↑⟩ → |↓⟩, |↓⟩ → 0
inline std::pair<int64_t, double> apply_sm(uint64_t state, int site) {
    if (get_bit(state, site) == 1) {  // spin up
        return {static_cast<int64_t>(state & ~(1ULL << site)), 1.0};
    }
    return {-1, 0.0};
}

// S^z eigenvalue: |↑⟩ → +1/2, |↓⟩ → -1/2
inline double sz_value(uint64_t state, int site) {
    return get_bit(state, site) ? 0.5 : -0.5;
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
    
    // Derived: bond orientations (0, 1, 2)
    std::map<std::pair<int, int>, int> bond_orientation;
    
    // Bowties: (center, s1, s2, s3, s4, center_pos)
    struct Bowtie {
        int s0, s1, s2, s3, s4;
        std::array<double, 2> center;
    };
    std::vector<Bowtie> bowties;
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
        for (const auto& prefix : {"kagome_bfg_2x2", "kagome_bfg_3x3", "kagome_bfg_4x4", "kagome_bfg_3x2"}) {
            std::string test_file = cluster_dir + "/" + prefix + suffix;
            std::ifstream test(test_file);
            if (test.good()) {
                lattice_file = test_file;
                break;
            }
        }
        if (!lattice_file.empty()) break;
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
    
    // Set lattice vectors (kagome)
    cluster.a1 = {1.0, 0.0};
    cluster.a2 = {0.5, std::sqrt(3.0) / 2.0};
    
    // Reciprocal vectors
    double det = cluster.a1[0] * cluster.a2[1] - cluster.a1[1] * cluster.a2[0];
    cluster.b1 = {2.0 * PI * cluster.a2[1] / det, -2.0 * PI * cluster.a2[0] / det};
    cluster.b2 = {-2.0 * PI * cluster.a1[1] / det, 2.0 * PI * cluster.a1[0] / det};
    
    // Generate k-points (assume square grid based on site count)
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
    
    // Find bowties (5-site structures: 2 triangles sharing a corner)
    // Build triangle list first
    std::vector<std::array<int, 3>> triangles;
    for (int i = 0; i < cluster.n_sites; ++i) {
        const auto& ni = cluster.nn_list[i];
        for (size_t a = 0; a < ni.size(); ++a) {
            int j = ni[a];
            if (j <= i) continue;
            for (size_t b = a + 1; b < ni.size(); ++b) {
                int k = ni[b];
                if (k <= j) continue;
                // Check if j and k are neighbors
                const auto& nj = cluster.nn_list[j];
                if (std::find(nj.begin(), nj.end(), k) != nj.end()) {
                    triangles.push_back({i, j, k});
                }
            }
        }
    }
    
    // Find pairs of triangles sharing exactly one vertex
    for (size_t t1 = 0; t1 < triangles.size(); ++t1) {
        for (size_t t2 = t1 + 1; t2 < triangles.size(); ++t2) {
            std::set<int> s1(triangles[t1].begin(), triangles[t1].end());
            std::set<int> s2(triangles[t2].begin(), triangles[t2].end());
            std::vector<int> shared;
            std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                                  std::back_inserter(shared));
            
            if (shared.size() == 1) {
                int s0 = shared[0];  // Center vertex
                std::vector<int> outer;
                for (int v : triangles[t1]) if (v != s0) outer.push_back(v);
                for (int v : triangles[t2]) if (v != s0) outer.push_back(v);
                
                if (outer.size() == 4) {
                    Cluster::Bowtie bt;
                    bt.s0 = s0;
                    bt.s1 = outer[0];
                    bt.s2 = outer[1];
                    bt.s3 = outer[2];
                    bt.s4 = outer[3];
                    
                    // Center position
                    bt.center[0] = (cluster.positions[bt.s0][0] + 
                                    cluster.positions[bt.s1][0] +
                                    cluster.positions[bt.s2][0] +
                                    cluster.positions[bt.s3][0] +
                                    cluster.positions[bt.s4][0]) / 5.0;
                    bt.center[1] = (cluster.positions[bt.s0][1] + 
                                    cluster.positions[bt.s1][1] +
                                    cluster.positions[bt.s2][1] +
                                    cluster.positions[bt.s3][1] +
                                    cluster.positions[bt.s4][1]) / 5.0;
                    
                    cluster.bowties.push_back(bt);
                }
            }
        }
    }
    
    std::cout << "Loaded cluster: " << cluster.n_sites << " sites, "
              << cluster.edges_nn.size() << " NN bonds, "
              << cluster.bowties.size() << " bowties" << std::endl;
    
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
// Compute S^+_i S^-_j correlation matrix
// -----------------------------------------------------------------------------

std::vector<std::vector<Complex>> compute_spsm_correlations(
    const std::vector<Complex>& psi, 
    int n_sites
) {
    uint64_t n_states = psi.size();
    std::vector<std::vector<Complex>> corr(n_sites, std::vector<Complex>(n_sites, 0.0));
    
    std::cout << "Computing S^+S^- correlations..." << std::flush;
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
                    // S^+_i S^-_j: need j=up, i=down
                    // After: j=down, i=up
                    if (get_bit(state, j) == 1 && get_bit(state, i) == 0) {
                        uint64_t new_state = flip_bit(flip_bit(state, i), j);
                        local_corr[i][j] += std::conj(psi[new_state]) * coeff;
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
                
                // S^+_i S^-_j term
                if (get_bit(state, j) == 1 && get_bit(state, i) == 0) {
                    uint64_t new_state = flip_bit(flip_bit(state, i), j);
                    local_bonds[e] += std::conj(psi[new_state]) * coeff;
                }
                
                // S^-_i S^+_j term
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
// Compute bond-bond correlation ⟨B_1 B_2⟩
// B = S^+_i S^-_j + S^-_i S^+_j
// -----------------------------------------------------------------------------

Complex compute_bond_bond_correlation(
    const std::vector<Complex>& psi,
    int /* n_sites */,
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
            // Requires: j1=up, i1=down, j2=up, i2=down
            if (s_j1 == 1 && s_i1 == 0 && s_j2 == 1 && s_i2 == 0) {
                uint64_t new_state = state;
                new_state = flip_bit(new_state, j1);  // j1: up -> down
                new_state = flip_bit(new_state, i1);  // i1: down -> up
                new_state = flip_bit(new_state, j2);  // j2: up -> down
                new_state = flip_bit(new_state, i2);  // i2: down -> up
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            
            // Term 2: S^+_{i1} S^-_{j1} S^-_{i2} S^+_{j2}
            if (s_j1 == 1 && s_i1 == 0 && s_i2 == 1 && s_j2 == 0) {
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
            if (s_i1 == 1 && s_j1 == 0 && s_j2 == 1 && s_i2 == 0) {
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
            if (s_i1 == 1 && s_j1 == 0 && s_i2 == 1 && s_j2 == 0) {
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
// Compute bowtie ring-flip expectation ⟨S^+_1 S^-_2 S^+_3 S^-_4 + h.c.⟩
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
            
            int b1 = get_bit(state, s1);
            int b2 = get_bit(state, s2);
            int b3 = get_bit(state, s3);
            int b4 = get_bit(state, s4);
            
            // S^+_1 S^-_2 S^+_3 S^-_4: need s1=down, s2=up, s3=down, s4=up
            if (b1 == 0 && b2 == 1 && b3 == 0 && b4 == 1) {
                uint64_t new_state = flip_bit(flip_bit(flip_bit(flip_bit(state, s1), s2), s3), s4);
                Complex contrib = std::conj(psi[new_state]) * coeff;
                local_real += contrib.real();
                local_imag += contrib.imag();
            }
            
            // S^-_1 S^+_2 S^-_3 S^+_4: need s1=up, s2=down, s3=up, s4=down
            if (b1 == 1 && b2 == 0 && b3 == 1 && b4 == 0) {
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
    const std::vector<std::vector<Complex>>& spsm_corr,
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
                double dr_x = cluster.positions[i][0] - cluster.positions[j][0];
                double dr_y = cluster.positions[i][1] - cluster.positions[j][1];
                double phase_arg = q[0] * dr_x + q[1] * dr_y;
                s_q += spsm_corr[i][j] * std::exp(I * phase_arg);
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
    std::array<Complex, 3> O_bar;  // Average XY energy per orientation
    double anisotropy;
};

NematicResult compute_nematic_order(
    const std::map<std::pair<int, int>, Complex>& bond_exp,
    const Cluster& cluster
) {
    NematicResult result;
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
    
    std::cout << "Nematic order: m_nem = " << result.m_nem << ", anisotropy = " << result.anisotropy << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Compute stripe structure factor
// -----------------------------------------------------------------------------

struct StripeResult {
    Complex S_stripe;
    double m_stripe;
    std::array<Complex, 3> O_bar_by_orientation;
};

StripeResult compute_stripe_structure_factor(
    const std::vector<Complex>& psi,
    const Cluster& cluster
) {
    StripeResult result;
    int n_edges = cluster.edges_nn.size();
    
    if (n_edges == 0) {
        result.S_stripe = 0.0;
        result.m_stripe = 0.0;
        return result;
    }
    
    std::cout << "Computing stripe structure factor (bond-bond correlations)..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Get bond expectations for diagnostics
    auto bond_exp = compute_xy_bond_expectations(psi, cluster);
    
    // Compute O_bar by orientation
    std::array<Complex, 3> sum_by_orient = {0.0, 0.0, 0.0};
    std::array<int, 3> count_by_orient = {0, 0, 0};
    for (const auto& [edge, exp_val] : bond_exp) {
        int alpha = cluster.bond_orientation.at(edge);
        sum_by_orient[alpha] += exp_val;
        count_by_orient[alpha]++;
    }
    for (int alpha = 0; alpha < 3; ++alpha) {
        if (count_by_orient[alpha] > 0) {
            result.O_bar_by_orientation[alpha] = sum_by_orient[alpha] / static_cast<double>(count_by_orient[alpha]);
        }
    }
    
    // Compute bond-bond correlations with orientation phase
    Complex omega = std::exp(2.0 * PI * I / 3.0);
    Complex S_stripe = 0.0;
    
    std::vector<std::pair<int, int>> edges(cluster.edges_nn.begin(), cluster.edges_nn.end());
    
    for (size_t b1 = 0; b1 < edges.size(); ++b1) {
        int i1 = edges[b1].first;
        int j1 = edges[b1].second;
        int alpha1 = cluster.bond_orientation.at(edges[b1]);
        
        for (size_t b2 = 0; b2 < edges.size(); ++b2) {
            int i2 = edges[b2].first;
            int j2 = edges[b2].second;
            int alpha2 = cluster.bond_orientation.at(edges[b2]);
            
            Complex corr = compute_bond_bond_correlation(psi, cluster.n_sites, i1, j1, i2, j2);
            Complex phase = std::pow(omega, alpha2 - alpha1);
            S_stripe += corr * phase;
        }
        
        if ((b1 + 1) % 10 == 0) {
            std::cout << "\r  Bond " << (b1 + 1) << "/" << edges.size() << std::flush;
        }
    }
    
    S_stripe /= static_cast<double>(n_edges);
    result.S_stripe = S_stripe;
    result.m_stripe = std::sqrt(std::abs(S_stripe) / n_edges);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "\rStripe order: S_stripe = " << std::abs(S_stripe) 
              << ", m_stripe = " << result.m_stripe
              << " (" << duration.count() << " s)" << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Compute VBS (Valence Bond Solid) order - Bond dimer structure factor
// S_D(q) = (1/N_b) Σ_{b,b'} exp(iq·(r_b - r_{b'})) <δD_b δD_{b'}>
// where δD_b = D_b - <D> and D_b = S^+_i S^-_j + h.c.
// -----------------------------------------------------------------------------

struct VBSResult {
    std::vector<Complex> S_d;          // S_D(q) at each k-point
    std::vector<std::vector<Complex>> S_d_2d;  // 2D grid for visualization
    int q_max_idx;
    Complex s_d_max;
    std::array<double, 2> q_max;
    double m_vbs;
    double D_mean;                      // Mean bond value
    int n_q_grid;                       // Size of 2D grid
};

VBSResult compute_vbs_order(
    const std::map<std::pair<int, int>, Complex>& bond_exp,
    const Cluster& cluster,
    int n_q_grid = 50
) {
    VBSResult result;
    result.n_q_grid = n_q_grid;
    int n_bonds = cluster.edges_nn.size();
    int n_k = cluster.k_points.size();
    
    if (n_bonds == 0) {
        result.m_vbs = 0.0;
        return result;
    }
    
    std::cout << "Computing VBS order (bond dimer structure factor)..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Compute bond centers
    std::vector<std::pair<int, int>> edges(cluster.edges_nn.begin(), cluster.edges_nn.end());
    std::vector<std::array<double, 2>> bond_centers(n_bonds);
    
    for (int b = 0; b < n_bonds; ++b) {
        int i = edges[b].first;
        int j = edges[b].second;
        bond_centers[b][0] = 0.5 * (cluster.positions[i][0] + cluster.positions[j][0]);
        bond_centers[b][1] = 0.5 * (cluster.positions[i][1] + cluster.positions[j][1]);
    }
    
    // Mean bond value
    double sum_real = 0.0;
    for (int b = 0; b < n_bonds; ++b) {
        sum_real += std::real(bond_exp.at(edges[b]));
    }
    result.D_mean = sum_real / n_bonds;
    
    // Connected bond correlations δD
    std::vector<Complex> delta_D(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        delta_D[b] = bond_exp.at(edges[b]) - result.D_mean;
    }
    
    // =========================================================================
    // Compute S_D(q) at discrete allowed k-points
    // =========================================================================
    result.S_d.resize(n_k, 0.0);
    
    #pragma omp parallel for
    for (int ik = 0; ik < n_k; ++ik) {
        const auto& q = cluster.k_points[ik];
        Complex s_d = 0.0;
        
        for (int b1 = 0; b1 < n_bonds; ++b1) {
            for (int b2 = 0; b2 < n_bonds; ++b2) {
                double dr_x = bond_centers[b1][0] - bond_centers[b2][0];
                double dr_y = bond_centers[b1][1] - bond_centers[b2][1];
                double phase_arg = q[0] * dr_x + q[1] * dr_y;
                s_d += delta_D[b1] * std::conj(delta_D[b2]) * std::exp(I * phase_arg);
            }
        }
        result.S_d[ik] = s_d / static_cast<double>(n_bonds);
    }
    
    // Find maximum
    double max_val = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        double val = std::abs(result.S_d[ik]);
        if (val > max_val) {
            max_val = val;
            result.q_max_idx = ik;
            result.s_d_max = result.S_d[ik];
            result.q_max = cluster.k_points[ik];
        }
    }
    result.m_vbs = std::sqrt(max_val / n_bonds);
    
    // =========================================================================
    // Also compute on dense 2D grid for visualization
    // =========================================================================
    result.S_d_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
    
    #pragma omp parallel for collapse(2)
    for (int i1 = 0; i1 < n_q_grid; ++i1) {
        for (int i2 = 0; i2 < n_q_grid; ++i2) {
            double q1 = -1.0 + 2.0 * i1 / (n_q_grid - 1);
            double q2 = -1.0 + 2.0 * i2 / (n_q_grid - 1);
            double qx = q1 * cluster.b1[0] + q2 * cluster.b2[0];
            double qy = q1 * cluster.b1[1] + q2 * cluster.b2[1];
            
            Complex s_d = 0.0;
            for (int b1 = 0; b1 < n_bonds; ++b1) {
                for (int b2 = 0; b2 < n_bonds; ++b2) {
                    double dr_x = bond_centers[b1][0] - bond_centers[b2][0];
                    double dr_y = bond_centers[b1][1] - bond_centers[b2][1];
                    double phase_arg = qx * dr_x + qy * dr_y;
                    s_d += delta_D[b1] * std::conj(delta_D[b2]) * std::exp(I * phase_arg);
                }
            }
            result.S_d_2d[i1][i2] = s_d / static_cast<double>(n_bonds);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    std::cout << "  VBS order: m_vbs = " << result.m_vbs 
              << " at q = (" << result.q_max[0] << ", " << result.q_max[1] << ")" << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Compute 2D S(q) grid for visualization
// -----------------------------------------------------------------------------

std::vector<std::vector<Complex>> compute_sq_2d_grid(
    const std::vector<std::vector<Complex>>& spsm_corr,
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
                    double dr_x = cluster.positions[i][0] - cluster.positions[j][0];
                    double dr_y = cluster.positions[i][1] - cluster.positions[j][1];
                    double phase_arg = qx * dr_x + qy * dr_y;
                    s_q += spsm_corr[i][j] * std::exp(I * phase_arg);
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
// Compute plaquette/bowtie order
// -----------------------------------------------------------------------------

struct PlaquetteResult {
    std::vector<Complex> P_r;  // Resonance per bowtie
    double P_mean;
    double resonance_strength;
    std::vector<Complex> S_p;  // Structure factor at k-points
    double m_plaquette;
    int q_max_idx;
    std::array<double, 2> q_max;
};

PlaquetteResult compute_plaquette_order(
    const std::vector<Complex>& psi,
    const Cluster& cluster
) {
    PlaquetteResult result;
    int n_bowties = cluster.bowties.size();
    
    if (n_bowties == 0) {
        result.m_plaquette = 0.0;
        result.resonance_strength = 0.0;
        return result;
    }
    
    std::cout << "Computing plaquette resonance for " << n_bowties << " bowties..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    
    result.P_r.resize(n_bowties);
    
    #pragma omp parallel for
    for (int b = 0; b < n_bowties; ++b) {
        const auto& bt = cluster.bowties[b];
        result.P_r[b] = compute_bowtie_resonance(psi, bt.s1, bt.s2, bt.s3, bt.s4);
    }
    
    // Mean resonance
    double sum_real = 0.0;
    double sum_abs = 0.0;
    for (int b = 0; b < n_bowties; ++b) {
        sum_real += std::real(result.P_r[b]);
        sum_abs += std::abs(result.P_r[b]);
    }
    result.P_mean = sum_real / n_bowties;
    result.resonance_strength = sum_abs / n_bowties;
    
    // Structure factor at k-points
    int n_k = cluster.k_points.size();
    result.S_p.resize(n_k, 0.0);
    
    // Connected correlations
    std::vector<Complex> delta_P(n_bowties);
    for (int b = 0; b < n_bowties; ++b) {
        delta_P[b] = result.P_r[b] - result.P_mean;
    }
    
    #pragma omp parallel for
    for (int ik = 0; ik < n_k; ++ik) {
        const auto& q = cluster.k_points[ik];
        Complex s_p = 0.0;
        
        for (int b1 = 0; b1 < n_bowties; ++b1) {
            for (int b2 = 0; b2 < n_bowties; ++b2) {
                double dr_x = cluster.bowties[b1].center[0] - cluster.bowties[b2].center[0];
                double dr_y = cluster.bowties[b1].center[1] - cluster.bowties[b2].center[1];
                double phase_arg = q[0] * dr_x + q[1] * dr_y;
                s_p += delta_P[b1] * std::conj(delta_P[b2]) * std::exp(I * phase_arg);
            }
        }
        result.S_p[ik] = s_p / static_cast<double>(n_bowties);
    }
    
    // Find maximum
    double max_val = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        double val = std::abs(result.S_p[ik]);
        if (val > max_val) {
            max_val = val;
            result.q_max_idx = ik;
            result.q_max = cluster.k_points[ik];
        }
    }
    result.m_plaquette = std::sqrt(max_val / n_bowties);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    std::cout << "  P_mean = " << result.P_mean << ", resonance_strength = " << result.resonance_strength
              << ", m_plaquette = " << result.m_plaquette << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Save results to HDF5 (full version with 2D grids and VBS)
// -----------------------------------------------------------------------------

void save_results(
    const std::string& filename,
    const StructureFactorResult& sf,
    const NematicResult& nem,
    const StripeResult& stripe,
    const VBSResult& vbs,
    const PlaquetteResult& plaq,
    const Cluster& cluster,
    const std::vector<std::vector<Complex>>& s_q_2d,
    int n_q_grid
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
        
        // Save VBS S_D(q) at k-points
        if (!vbs.S_d.empty()) {
            hsize_t dims[1] = {vbs.S_d.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_D_q", complex_type, dataspace);
            dataset.write(vbs.S_d.data(), complex_type);
        }
        
        // Save 2D S_D(q) grid for VBS visualization
        if (!vbs.S_d_2d.empty()) {
            hsize_t dims[2] = {static_cast<hsize_t>(vbs.n_q_grid), static_cast<hsize_t>(vbs.n_q_grid)};
            H5::DataSpace dataspace(2, dims);
            H5::DataSet dataset = file.createDataSet("S_D_q_2d", complex_type, dataspace);
            std::vector<Complex> flat(vbs.n_q_grid * vbs.n_q_grid);
            for (int i = 0; i < vbs.n_q_grid; ++i) {
                for (int j = 0; j < vbs.n_q_grid; ++j) {
                    flat[i * vbs.n_q_grid + j] = vbs.S_d_2d[i][j];
                }
            }
            dataset.write(flat.data(), complex_type);
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
            write_scalar("m_nematic", nem.m_nem);
            write_scalar("nematic_anisotropy", nem.anisotropy);
            
            // Stripe order
            write_scalar("m_stripe", stripe.m_stripe);
            write_scalar("S_stripe_abs", std::abs(stripe.S_stripe));
            
            // VBS order
            write_scalar("m_vbs", vbs.m_vbs);
            write_scalar("D_mean", vbs.D_mean);
            write_scalar("s_d_max", std::abs(vbs.s_d_max));
            write_int("vbs_q_max_idx", vbs.q_max_idx);
            write_scalar("vbs_q_max_x", vbs.q_max[0]);
            write_scalar("vbs_q_max_y", vbs.q_max[1]);
            
            // Plaquette order
            write_scalar("m_plaquette", plaq.m_plaquette);
            write_scalar("P_mean", plaq.P_mean);
            write_scalar("resonance_strength", plaq.resonance_strength);
            write_int("plaquette_q_max_idx", plaq.q_max_idx);
            
            // Cluster info
            write_int("n_sites", cluster.n_sites);
            write_int("n_bonds", static_cast<int>(cluster.edges_nn.size()));
            write_int("n_bowties", static_cast<int>(cluster.bowties.size()));
            write_int("n_q_grid", n_q_grid);
        }
        
        // Save plaquette resonances
        if (!plaq.P_r.empty()) {
            hsize_t dims[1] = {plaq.P_r.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("P_r", complex_type, dataspace);
            dataset.write(plaq.P_r.data(), complex_type);
        }
        
        // Save plaquette structure factor
        if (!plaq.S_p.empty()) {
            hsize_t dims[1] = {plaq.S_p.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("S_plaquette", complex_type, dataspace);
            dataset.write(plaq.S_p.data(), complex_type);
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
    double m_stripe;
    double m_vbs;
    double m_plaquette;
    double resonance_strength;
    double anisotropy;
    double P_mean;
    double D_mean;
};

// -----------------------------------------------------------------------------
// Compute all order parameters (for scan mode)
// -----------------------------------------------------------------------------

OrderParameterResults compute_all_order_parameters(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    double jpm_value,
    bool skip_stripe
) {
    OrderParameterResults results;
    results.jpm = jpm_value;
    
    // Compute correlations
    auto spsm_corr = compute_spsm_correlations(psi, cluster.n_sites);
    
    // Structure factor
    auto sf_result = compute_spin_structure_factor(spsm_corr, cluster);
    results.m_translation = sf_result.m_translation;
    
    // Nematic
    auto bond_exp = compute_xy_bond_expectations(psi, cluster);
    auto nem_result = compute_nematic_order(bond_exp, cluster);
    results.m_nematic = nem_result.m_nem;
    results.anisotropy = nem_result.anisotropy;
    
    // VBS order
    auto vbs_result = compute_vbs_order(bond_exp, cluster);
    results.m_vbs = vbs_result.m_vbs;
    results.D_mean = vbs_result.D_mean;
    
    // Stripe (can be slow)
    if (!skip_stripe && cluster.edges_nn.size() <= 50) {
        auto stripe_result = compute_stripe_structure_factor(psi, cluster);
        results.m_stripe = stripe_result.m_stripe;
    } else {
        results.m_stripe = 0.0;
    }
    
    // Plaquette
    auto plaq_result = compute_plaquette_order(psi, cluster);
    results.m_plaquette = plaq_result.m_plaquette;
    results.resonance_strength = plaq_result.resonance_strength;
    results.P_mean = plaq_result.P_mean;
    
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
    bool skip_stripe,
    bool save_full,
    bool use_tpq
) {
    // Find all Jpm=* directories
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
    
    std::cout << "Found " << jpm_dirs.size() << " Jpm directories" << std::endl;
    if (save_full) {
        std::cout << "Full output mode: saving S(q) and S_D(q) 2D grids per directory" << std::endl;
    }
    
    if (jpm_dirs.empty()) {
        return {};
    }
    
    // Load cluster from first directory
    Cluster cluster = load_cluster(jpm_dirs[0].second);
    std::cout << "Cluster: " << cluster.n_sites << " sites, " 
              << cluster.edges_nn.size() << " bonds, "
              << cluster.bowties.size() << " bowties" << std::endl;
    
    // Process directories
    std::vector<OrderParameterResults> all_results(jpm_dirs.size());
    std::mutex print_mutex;
    std::atomic<int> completed(0);
    
    #pragma omp parallel for schedule(dynamic) num_threads(n_workers)
    for (size_t i = 0; i < jpm_dirs.size(); ++i) {
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
            
            OrderParameterResults results;
            results.jpm = jpm;
            results.temperature = temperature;
            
            if (save_full) {
                // Full computation with 2D grids
                auto spsm_corr = compute_spsm_correlations(psi, cluster.n_sites);
                auto sf_result = compute_spin_structure_factor(spsm_corr, cluster);
                auto s_q_2d = compute_sq_2d_grid(spsm_corr, cluster, n_q_grid);
                
                auto bond_exp = compute_xy_bond_expectations(psi, cluster);
                auto nem_result = compute_nematic_order(bond_exp, cluster);
                auto vbs_result = compute_vbs_order(bond_exp, cluster, n_q_grid);
                
                StripeResult stripe_result;
                if (!skip_stripe && cluster.edges_nn.size() <= 50) {
                    stripe_result = compute_stripe_structure_factor(psi, cluster);
                }
                
                auto plaq_result = compute_plaquette_order(psi, cluster);
                
                // Save full results to per-Jpm file
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << jpm;
                std::string out_file = output_dir + "/order_params_Jpm=" + oss.str() + ".h5";
                save_results(out_file, sf_result, nem_result, stripe_result, vbs_result,
                            plaq_result, cluster, s_q_2d, n_q_grid);
                
                // Fill scalar results for summary
                results.m_translation = sf_result.m_translation;
                results.m_nematic = nem_result.m_nem;
                results.anisotropy = nem_result.anisotropy;
                results.m_vbs = vbs_result.m_vbs;
                results.D_mean = vbs_result.D_mean;
                results.m_stripe = stripe_result.m_stripe;
                results.m_plaquette = plaq_result.m_plaquette;
                results.resonance_strength = plaq_result.resonance_strength;
                results.P_mean = plaq_result.P_mean;
            } else {
                // Quick scalar-only computation
                results = compute_all_order_parameters(psi, cluster, jpm, skip_stripe);
                results.temperature = temperature;  // Preserve temperature from TPQ
            }
            
            all_results[i] = results;
            
            int done = ++completed;
            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[" << done << "/" << jpm_dirs.size() << "] "
                          << "Jpm=" << std::fixed << std::setprecision(4) << jpm;
                if (use_tpq) {
                    std::cout << " T=" << std::setprecision(6) << results.temperature;
                }
                std::cout << " | m_trans=" << std::setprecision(6) << results.m_translation
                          << " | m_vbs=" << results.m_vbs
                          << " | m_plaq=" << results.m_plaquette
                          << " | res=" << results.resonance_strength
                          << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "Error processing " << dir << ": " << e.what() << std::endl;
        }
    }
    
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
        std::vector<double> jpm_vals(n), m_trans(n), m_nem(n), m_stripe(n);
        std::vector<double> m_plaq(n), resonance(n), aniso(n), P_mean(n);
        
        for (size_t i = 0; i < n; ++i) {
            jpm_vals[i] = results[i].jpm;
            m_trans[i] = results[i].m_translation;
            m_nem[i] = results[i].m_nematic;
            m_stripe[i] = results[i].m_stripe;
            m_plaq[i] = results[i].m_plaquette;
            resonance[i] = results[i].resonance_strength;
            aniso[i] = results[i].anisotropy;
            P_mean[i] = results[i].P_mean;
        }
        
        auto write_dataset = [&](const std::string& name, const std::vector<double>& data) {
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        };
        
        write_dataset("jpm_values", jpm_vals);
        write_dataset("m_translation", m_trans);
        write_dataset("m_nematic", m_nem);
        write_dataset("m_stripe", m_stripe);
        write_dataset("m_plaquette", m_plaq);
        write_dataset("resonance_strength", resonance);
        write_dataset("anisotropy", aniso);
        write_dataset("P_mean", P_mean);
        
        // Add VBS order parameters
        std::vector<double> m_vbs_vals(n), D_mean_vals(n), temperature_vals(n);
        for (size_t i = 0; i < n; ++i) {
            m_vbs_vals[i] = results[i].m_vbs;
            D_mean_vals[i] = results[i].D_mean;
            temperature_vals[i] = results[i].temperature;
        }
        write_dataset("m_vbs", m_vbs_vals);
        write_dataset("D_mean", D_mean_vals);
        write_dataset("temperature", temperature_vals);
        
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
              << "  --skip-stripe        Skip stripe order computation (faster)\n"
              << "  --save-full          Save full S(q), S_D(q) 2D grids per Jpm directory\n"
              << "  --tpq                Use TPQ states (lowest temperature) instead of ground state\n"
              << "\nComputes BFG order parameters from ground state or TPQ wavefunction:\n"
              << "  1. S(q) - Spin structure factor (translation order)\n"
              << "  2. Nematic order - Bond orientation anisotropy (C6→C2)\n"
              << "  3. Stripe order - Bond-bond correlations with C3 phase\n"
              << "  4. VBS order - Valence bond solid (dimer structure factor)\n"
              << "  5. Plaquette order - Bowtie ring-flip correlations\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string scan_dir, output_dir;
    std::string wf_file, cluster_dir, output_file = "bfg_order_parameters.h5";
    int n_workers = 4;
    int n_q_grid = 50;
    bool skip_stripe = false;
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
        } else if (arg == "--skip-stripe") {
            skip_stripe = true;
        } else if (arg == "--save-full") {
            save_full = true;
        } else if (arg == "--tpq") {
            use_tpq = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
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
    std::cout << "OpenMP enabled with " << omp_get_max_threads() << " max threads" << std::endl;
    #endif
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    try {
        if (scan_mode) {
            // Scan directory mode
            if (scan_dir.empty()) {
                std::cerr << "Error: --scan-dir required in scan mode" << std::endl;
                return 1;
            }
            if (output_dir.empty()) {
                output_dir = scan_dir + "/order_parameter_results";
            }
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
            
            auto results = scan_jpm_directories(scan_dir, output_dir, n_workers, n_q_grid, skip_stripe, save_full, use_tpq);
            
            if (!results.empty()) {
                std::string results_file = output_dir + "/scan_results.h5";
                save_scan_results(results, results_file);
            }
            
        } else {
            // Single file mode
            if (wf_file.empty() || cluster_dir.empty()) {
                print_usage(argv[0]);
                return 1;
            }
            
            std::cout << "========================================\n"
                      << "BFG ORDER PARAMETER COMPUTATION (CPU)\n"
                      << "========================================" << std::endl;
            
            Cluster cluster = load_cluster(cluster_dir);
            std::vector<Complex> psi = load_wavefunction(wf_file);
            
            // Verify size
            uint64_t expected_size = 1ULL << cluster.n_sites;
            if (psi.size() != expected_size) {
                std::cerr << "Warning: wavefunction size " << psi.size() 
                          << " != expected 2^" << cluster.n_sites << " = " << expected_size << std::endl;
            }
            
            // Compute correlations
            auto spsm_corr = compute_spsm_correlations(psi, cluster.n_sites);
            
            // Compute S(q) at k-points
            auto sf_result = compute_spin_structure_factor(spsm_corr, cluster);
            
            // Compute 2D S(q) grid for visualization
            auto s_q_2d = compute_sq_2d_grid(spsm_corr, cluster, n_q_grid);
            
            // Compute bond expectations for nematic and VBS
            auto bond_exp = compute_xy_bond_expectations(psi, cluster);
            auto nem_result = compute_nematic_order(bond_exp, cluster);
            
            // VBS order (with 2D grid)
            auto vbs_result = compute_vbs_order(bond_exp, cluster, n_q_grid);
            
            // Stripe order (slower)
            StripeResult stripe_result;
            if (!skip_stripe) {
                stripe_result = compute_stripe_structure_factor(psi, cluster);
            }
            
            // Plaquette order
            auto plaq_result = compute_plaquette_order(psi, cluster);
            
            // Save results with full 2D grids
            save_results(output_file, sf_result, nem_result, stripe_result, vbs_result, 
                        plaq_result, cluster, s_q_2d, n_q_grid);
            
            // Print summary
            std::cout << "\n========== ORDER PARAMETER SUMMARY ==========" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Translation order:  m = " << sf_result.m_translation 
                      << " at q = (" << sf_result.q_max[0] << ", " << sf_result.q_max[1] << ")" << std::endl;
            std::cout << "Nematic order:      m = " << nem_result.m_nem 
                      << ", anisotropy = " << nem_result.anisotropy << std::endl;
            std::cout << "VBS order:          m = " << vbs_result.m_vbs 
                      << " at q = (" << vbs_result.q_max[0] << ", " << vbs_result.q_max[1] << ")" << std::endl;
            if (!skip_stripe) {
                std::cout << "Stripe order:       m = " << stripe_result.m_stripe << std::endl;
            }
            std::cout << "Plaquette order:    m = " << plaq_result.m_plaquette 
                      << ", resonance = " << plaq_result.resonance_strength << std::endl;
            std::cout << "==============================================" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    std::cout << "\nTotal runtime: " << total_duration.count() << " seconds" << std::endl;
    
    return 0;
}

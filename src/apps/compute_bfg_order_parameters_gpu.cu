/**
 * @file compute_bfg_order_parameters_gpu.cu
 * @brief GPU-accelerated computation of BFG order parameters from wavefunctions
 * 
 * Features:
 * - CUDA GPU acceleration for correlation computations
 * - scan-dir mode for processing multiple Jpm directories
 * - OpenMP parallelization across directories
 * - HDF5 I/O for wavefunctions and results
 * 
 * Computes:
 * 1. S(q) - Spin structure factor using S^+S^- correlations
 * 2. Nematic order - Bond orientation anisotropy
 * 3. Stripe structure factor - Bond-bond correlations with orientation phase
 * 4. Plaquette/bowtie resonance - Ring-flip correlations
 * 
 * Usage:
 *   Single file:  ./compute_bfg_order_parameters_gpu <wavefunction.h5> <cluster_dir> [output.h5]
 *   Scan mode:    ./compute_bfg_order_parameters_gpu --scan-dir <dir> --output-dir <out>
 * 
 * Compile with:
 *   nvcc -O3 -std=c++17 -arch=sm_80 compute_bfg_order_parameters_gpu.cu -o compute_bfg_order_parameters_gpu -lhdf5 -lhdf5_cpp -Xcompiler -fopenmp
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
#include <thread>
#include <mutex>
#include <atomic>

#include <cuda_runtime.h>
#include <cuComplex.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <H5Cpp.h>

namespace fs = std::filesystem;

using Complex = std::complex<double>;
const double PI = 3.14159265358979323846;
const Complex I_CPU(0.0, 1.0);

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// -----------------------------------------------------------------------------
// GPU Kernels
// -----------------------------------------------------------------------------

// Complex number helpers for CUDA (cuCmul, cuConj already defined in cuComplex.h)
__device__ __forceinline__ cuDoubleComplex make_cuDoubleComplex_from_parts(double r, double i) {
    return make_cuDoubleComplex(r, i);
}

__device__ __forceinline__ int get_bit_gpu(uint64_t state, int site) {
    return (state >> site) & 1;
}

__device__ __forceinline__ uint64_t flip_bit_gpu(uint64_t state, int site) {
    return state ^ (1ULL << site);
}

// Kernel for computing S^+_i S^-_j correlations
__global__ void compute_spsm_correlation_kernel(
    const cuDoubleComplex* __restrict__ psi,
    double* __restrict__ corr_real,
    double* __restrict__ corr_imag,
    int n_sites,
    uint64_t n_states,
    int i, int j
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    
    cuDoubleComplex coeff = psi[idx];
    double abs_coeff = cuCreal(coeff) * cuCreal(coeff) + cuCimag(coeff) * cuCimag(coeff);
    if (abs_coeff < 1e-30) return;
    
    uint64_t state = idx;
    
    // S^+_i S^-_j: need j=up, i=down
    if (get_bit_gpu(state, j) == 1 && get_bit_gpu(state, i) == 0) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(state, i), j);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(corr_real, cuCreal(contrib));
        atomicAdd(corr_imag, cuCimag(contrib));
    }
}

// Kernel for computing XY bond expectation
__global__ void compute_xy_bond_kernel(
    const cuDoubleComplex* __restrict__ psi,
    double* __restrict__ result_real,
    double* __restrict__ result_imag,
    uint64_t n_states,
    int site_i, int site_j
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    
    cuDoubleComplex coeff = psi[idx];
    double abs_coeff = cuCreal(coeff) * cuCreal(coeff) + cuCimag(coeff) * cuCimag(coeff);
    if (abs_coeff < 1e-30) return;
    
    uint64_t state = idx;
    
    // S^+_i S^-_j term
    if (get_bit_gpu(state, site_j) == 1 && get_bit_gpu(state, site_i) == 0) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(state, site_i), site_j);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(result_real, cuCreal(contrib));
        atomicAdd(result_imag, cuCimag(contrib));
    }
    
    // S^-_i S^+_j term
    if (get_bit_gpu(state, site_i) == 1 && get_bit_gpu(state, site_j) == 0) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(state, site_i), site_j);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(result_real, cuCreal(contrib));
        atomicAdd(result_imag, cuCimag(contrib));
    }
}

// Kernel for computing bowtie resonance
__global__ void compute_bowtie_resonance_kernel(
    const cuDoubleComplex* __restrict__ psi,
    double* __restrict__ result_real,
    double* __restrict__ result_imag,
    uint64_t n_states,
    int s1, int s2, int s3, int s4
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    
    cuDoubleComplex coeff = psi[idx];
    double abs_coeff = cuCreal(coeff) * cuCreal(coeff) + cuCimag(coeff) * cuCimag(coeff);
    if (abs_coeff < 1e-30) return;
    
    uint64_t state = idx;
    int b1 = get_bit_gpu(state, s1);
    int b2 = get_bit_gpu(state, s2);
    int b3 = get_bit_gpu(state, s3);
    int b4 = get_bit_gpu(state, s4);
    
    // S^+_1 S^-_2 S^+_3 S^-_4: need s1=down, s2=up, s3=down, s4=up
    if (b1 == 0 && b2 == 1 && b3 == 0 && b4 == 1) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(state, s1), s2), s3), s4);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(result_real, cuCreal(contrib));
        atomicAdd(result_imag, cuCimag(contrib));
    }
    
    // S^-_1 S^+_2 S^-_3 S^+_4: need s1=up, s2=down, s3=up, s4=down
    if (b1 == 1 && b2 == 0 && b3 == 1 && b4 == 0) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(state, s1), s2), s3), s4);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(result_real, cuCreal(contrib));
        atomicAdd(result_imag, cuCimag(contrib));
    }
}

// Kernel for bond-bond correlation (4-point function)
__global__ void compute_bond_bond_correlation_kernel(
    const cuDoubleComplex* __restrict__ psi,
    double* __restrict__ result_real,
    double* __restrict__ result_imag,
    uint64_t n_states,
    int i1, int j1, int i2, int j2
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    
    cuDoubleComplex coeff = psi[idx];
    double abs_coeff = cuCreal(coeff) * cuCreal(coeff) + cuCimag(coeff) * cuCimag(coeff);
    if (abs_coeff < 1e-30) return;
    
    uint64_t state = idx;
    int s_i1 = get_bit_gpu(state, i1);
    int s_j1 = get_bit_gpu(state, j1);
    int s_i2 = get_bit_gpu(state, i2);
    int s_j2 = get_bit_gpu(state, j2);
    
    // Term 1: S^+_{i1} S^-_{j1} S^+_{i2} S^-_{j2}
    if (s_j1 == 1 && s_i1 == 0 && s_j2 == 1 && s_i2 == 0) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(state, j1), i1), j2), i2);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(result_real, cuCreal(contrib));
        atomicAdd(result_imag, cuCimag(contrib));
    }
    
    // Term 2: S^+_{i1} S^-_{j1} S^-_{i2} S^+_{j2}
    if (s_j1 == 1 && s_i1 == 0 && s_i2 == 1 && s_j2 == 0) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(state, j1), i1), i2), j2);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(result_real, cuCreal(contrib));
        atomicAdd(result_imag, cuCimag(contrib));
    }
    
    // Term 3: S^-_{i1} S^+_{j1} S^+_{i2} S^-_{j2}
    if (s_i1 == 1 && s_j1 == 0 && s_j2 == 1 && s_i2 == 0) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(state, i1), j1), j2), i2);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(result_real, cuCreal(contrib));
        atomicAdd(result_imag, cuCimag(contrib));
    }
    
    // Term 4: S^-_{i1} S^+_{j1} S^-_{i2} S^+_{j2}
    if (s_i1 == 1 && s_j1 == 0 && s_i2 == 1 && s_j2 == 0) {
        uint64_t new_state = flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(flip_bit_gpu(state, i1), j1), i2), j2);
        cuDoubleComplex contrib = cuCmul(cuConj(psi[new_state]), coeff);
        atomicAdd(result_real, cuCreal(contrib));
        atomicAdd(result_imag, cuCimag(contrib));
    }
}

// -----------------------------------------------------------------------------
// GPU Wrapper Classes
// -----------------------------------------------------------------------------

class GPUWavefunction {
public:
    cuDoubleComplex* d_psi;
    uint64_t n_states;
    bool allocated;
    
    GPUWavefunction() : d_psi(nullptr), n_states(0), allocated(false) {}
    
    void allocate(uint64_t n) {
        if (allocated) free();
        n_states = n;
        CUDA_CHECK(cudaMalloc(&d_psi, n_states * sizeof(cuDoubleComplex)));
        allocated = true;
    }
    
    void upload(const std::vector<Complex>& psi) {
        if (!allocated || psi.size() != n_states) {
            allocate(psi.size());
        }
        CUDA_CHECK(cudaMemcpy(d_psi, psi.data(), n_states * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    }
    
    void free() {
        if (allocated && d_psi) {
            cudaFree(d_psi);
            d_psi = nullptr;
            allocated = false;
        }
    }
    
    ~GPUWavefunction() { free(); }
};

class GPUResultAccumulator {
public:
    double* d_result_real;
    double* d_result_imag;
    bool allocated;
    
    GPUResultAccumulator() : d_result_real(nullptr), d_result_imag(nullptr), allocated(false) {}
    
    void allocate() {
        if (allocated) return;
        CUDA_CHECK(cudaMalloc(&d_result_real, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_result_imag, sizeof(double)));
        allocated = true;
    }
    
    void reset() {
        double zero = 0.0;
        CUDA_CHECK(cudaMemcpy(d_result_real, &zero, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_result_imag, &zero, sizeof(double), cudaMemcpyHostToDevice));
    }
    
    Complex get_result() {
        double real, imag;
        CUDA_CHECK(cudaMemcpy(&real, d_result_real, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&imag, d_result_imag, sizeof(double), cudaMemcpyDeviceToHost));
        return Complex(real, imag);
    }
    
    void free() {
        if (allocated) {
            if (d_result_real) cudaFree(d_result_real);
            if (d_result_imag) cudaFree(d_result_imag);
            d_result_real = nullptr;
            d_result_imag = nullptr;
            allocated = false;
        }
    }
    
    ~GPUResultAccumulator() { free(); }
};

// -----------------------------------------------------------------------------
// Cluster data structure
// -----------------------------------------------------------------------------

struct Cluster {
    int n_sites;
    std::vector<std::array<double, 2>> positions;
    std::vector<int> sublattice;
    std::vector<std::pair<int, int>> edges_nn;
    std::map<int, std::vector<int>> nn_list;
    std::array<double, 2> a1, a2;
    std::array<double, 2> b1, b2;
    std::vector<std::array<double, 2>> k_points;
    std::map<std::pair<int, int>, int> bond_orientation;
    
    struct Bowtie {
        int s0, s1, s2, s3, s4;
        std::array<double, 2> center;
    };
    std::vector<Bowtie> bowties;
};

// -----------------------------------------------------------------------------
// Results structure
// -----------------------------------------------------------------------------

struct OrderParameterResults {
    double jpm;
    double m_translation;
    double m_nematic;
    double m_stripe;
    double m_vbs;             // VBS order parameter
    double D_mean;            // Mean bond value
    double m_plaquette;
    double resonance_strength;
    double anisotropy;
    std::array<double, 2> q_max;
    int q_max_idx;
    double s_q_max;
    double P_mean;
};

// -----------------------------------------------------------------------------
// Load cluster from files
// -----------------------------------------------------------------------------

Cluster load_cluster(const std::string& cluster_dir) {
    Cluster cluster;
    
    // Load positions
    std::string pos_file = cluster_dir + "/positions.dat";
    std::ifstream pos_in(pos_file);
    if (!pos_in.is_open()) {
        throw std::runtime_error("Cannot open positions.dat in " + cluster_dir);
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
    
    // Load NN list - try to find any matching file
    std::vector<std::string> nn_patterns = {"_nn_list.dat"};
    std::string nn_file;
    
    for (const auto& entry : fs::directory_iterator(cluster_dir)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("_nn_list.dat") != std::string::npos) {
            nn_file = entry.path().string();
            break;
        }
    }
    
    if (!nn_file.empty()) {
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
    } else {
        // Construct from positions
        double nn_dist = 0.51;
        for (int i = 0; i < cluster.n_sites; ++i) {
            for (int j = i + 1; j < cluster.n_sites; ++j) {
                double dx = cluster.positions[j][0] - cluster.positions[i][0];
                double dy = cluster.positions[j][1] - cluster.positions[i][1];
                double d = std::sqrt(dx * dx + dy * dy);
                if (d < nn_dist) {
                    cluster.nn_list[i].push_back(j);
                    cluster.nn_list[j].push_back(i);
                    cluster.edges_nn.push_back({i, j});
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
    
    // Generate k-points
    int n_cells = cluster.n_sites / 3;
    int dim = static_cast<int>(std::sqrt(n_cells) + 0.5);
    if (dim * dim != n_cells) dim = std::max(1, static_cast<int>(std::sqrt(n_cells)));
    
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
        
        int orientation = (angle_deg < 30.0 || angle_deg >= 150.0) ? 0 : 
                          (angle_deg < 90.0) ? 1 : 2;
        cluster.bond_orientation[{i, j}] = orientation;
        cluster.bond_orientation[{j, i}] = orientation;
    }
    
    // Find bowties
    std::vector<std::array<int, 3>> triangles;
    for (int i = 0; i < cluster.n_sites; ++i) {
        const auto& ni = cluster.nn_list[i];
        for (size_t a = 0; a < ni.size(); ++a) {
            int j = ni[a];
            if (j <= i) continue;
            for (size_t b = a + 1; b < ni.size(); ++b) {
                int k = ni[b];
                if (k <= j) continue;
                const auto& nj = cluster.nn_list[j];
                if (std::find(nj.begin(), nj.end(), k) != nj.end()) {
                    triangles.push_back({i, j, k});
                }
            }
        }
    }
    
    for (size_t t1 = 0; t1 < triangles.size(); ++t1) {
        for (size_t t2 = t1 + 1; t2 < triangles.size(); ++t2) {
            std::set<int> s1(triangles[t1].begin(), triangles[t1].end());
            std::set<int> s2(triangles[t2].begin(), triangles[t2].end());
            std::vector<int> shared;
            std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                                  std::back_inserter(shared));
            
            if (shared.size() == 1) {
                int s0 = shared[0];
                std::vector<int> outer;
                for (int v : triangles[t1]) if (v != s0) outer.push_back(v);
                for (int v : triangles[t2]) if (v != s0) outer.push_back(v);
                
                if (outer.size() == 4) {
                    Cluster::Bowtie bt;
                    bt.s0 = s0;
                    bt.s1 = outer[0]; bt.s2 = outer[1];
                    bt.s3 = outer[2]; bt.s4 = outer[3];
                    bt.center[0] = (cluster.positions[bt.s0][0] + cluster.positions[bt.s1][0] +
                                    cluster.positions[bt.s2][0] + cluster.positions[bt.s3][0] +
                                    cluster.positions[bt.s4][0]) / 5.0;
                    bt.center[1] = (cluster.positions[bt.s0][1] + cluster.positions[bt.s1][1] +
                                    cluster.positions[bt.s2][1] + cluster.positions[bt.s3][1] +
                                    cluster.positions[bt.s4][1]) / 5.0;
                    cluster.bowties.push_back(bt);
                }
            }
        }
    }
    
    return cluster;
}

// -----------------------------------------------------------------------------
// Load wavefunction from HDF5
// -----------------------------------------------------------------------------

std::vector<Complex> load_wavefunction(const std::string& filename, int eigenvector_idx = 0) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        
        std::vector<std::string> dataset_names = {
            "eigenvector_" + std::to_string(eigenvector_idx),
            "eigenvectors", "psi", "wavefunction", "ground_state"
        };
        
        H5::DataSet dataset;
        bool found = false;
        
        for (const auto& name : dataset_names) {
            try {
                dataset = file.openDataSet(name);
                found = true;
                break;
            } catch (...) { continue; }
        }
        
        if (!found) {
            throw std::runtime_error("Wavefunction dataset not found in " + filename);
        }
        
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data());
        
        hsize_t total_size = 1;
        for (int i = 0; i < rank; ++i) total_size *= dims[i];
        
        std::vector<Complex> psi(total_size);
        
        H5::DataType dtype = dataset.getDataType();
        if (dtype.getSize() == 16) {
            std::vector<double> buffer(total_size * 2);
            dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
            for (hsize_t i = 0; i < total_size; ++i) {
                psi[i] = Complex(buffer[2 * i], buffer[2 * i + 1]);
            }
        } else {
            std::vector<double> real_data(total_size);
            dataset.read(real_data.data(), H5::PredType::NATIVE_DOUBLE);
            for (hsize_t i = 0; i < total_size; ++i) {
                psi[i] = Complex(real_data[i], 0.0);
            }
        }
        
        return psi;
        
    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 error reading " + filename + ": " + e.getCDetailMsg());
    }
}

// -----------------------------------------------------------------------------
// GPU-accelerated computation functions
// -----------------------------------------------------------------------------

std::vector<std::vector<Complex>> compute_spsm_correlations_gpu(
    GPUWavefunction& gpu_psi,
    GPUResultAccumulator& acc,
    int n_sites
) {
    std::vector<std::vector<Complex>> corr(n_sites, std::vector<Complex>(n_sites, 0.0));
    
    int block_size = 256;
    int num_blocks = (gpu_psi.n_states + block_size - 1) / block_size;
    
    acc.allocate();
    
    for (int i = 0; i < n_sites; ++i) {
        for (int j = 0; j < n_sites; ++j) {
            acc.reset();
            compute_spsm_correlation_kernel<<<num_blocks, block_size>>>(
                gpu_psi.d_psi, acc.d_result_real, acc.d_result_imag,
                n_sites, gpu_psi.n_states, i, j
            );
            CUDA_CHECK(cudaDeviceSynchronize());
            corr[i][j] = acc.get_result();
        }
    }
    
    return corr;
}

std::map<std::pair<int, int>, Complex> compute_xy_bond_expectations_gpu(
    GPUWavefunction& gpu_psi,
    GPUResultAccumulator& acc,
    const Cluster& cluster
) {
    std::map<std::pair<int, int>, Complex> bonds;
    
    int block_size = 256;
    int num_blocks = (gpu_psi.n_states + block_size - 1) / block_size;
    
    acc.allocate();
    
    for (const auto& [i, j] : cluster.edges_nn) {
        acc.reset();
        compute_xy_bond_kernel<<<num_blocks, block_size>>>(
            gpu_psi.d_psi, acc.d_result_real, acc.d_result_imag,
            gpu_psi.n_states, i, j
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        bonds[{i, j}] = acc.get_result();
    }
    
    return bonds;
}

Complex compute_bond_bond_correlation_gpu(
    GPUWavefunction& gpu_psi,
    GPUResultAccumulator& acc,
    int i1, int j1, int i2, int j2
) {
    int block_size = 256;
    int num_blocks = (gpu_psi.n_states + block_size - 1) / block_size;
    
    acc.reset();
    compute_bond_bond_correlation_kernel<<<num_blocks, block_size>>>(
        gpu_psi.d_psi, acc.d_result_real, acc.d_result_imag,
        gpu_psi.n_states, i1, j1, i2, j2
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    return acc.get_result();
}

std::vector<Complex> compute_bowtie_resonances_gpu(
    GPUWavefunction& gpu_psi,
    GPUResultAccumulator& acc,
    const Cluster& cluster
) {
    std::vector<Complex> P_r(cluster.bowties.size());
    
    int block_size = 256;
    int num_blocks = (gpu_psi.n_states + block_size - 1) / block_size;
    
    acc.allocate();
    
    for (size_t b = 0; b < cluster.bowties.size(); ++b) {
        const auto& bt = cluster.bowties[b];
        acc.reset();
        compute_bowtie_resonance_kernel<<<num_blocks, block_size>>>(
            gpu_psi.d_psi, acc.d_result_real, acc.d_result_imag,
            gpu_psi.n_states, bt.s1, bt.s2, bt.s3, bt.s4
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        P_r[b] = acc.get_result();
    }
    
    return P_r;
}

// -----------------------------------------------------------------------------
// Compute all order parameters (GPU version)
// -----------------------------------------------------------------------------

OrderParameterResults compute_order_parameters_gpu(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    double jpm_value = 0.0,
    bool skip_stripe = false
) {
    OrderParameterResults results;
    results.jpm = jpm_value;
    
    GPUWavefunction gpu_psi;
    GPUResultAccumulator acc;
    
    gpu_psi.upload(psi);
    acc.allocate();
    
    std::cout << "  Computing S^+S^- correlations (GPU)..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    auto spsm_corr = compute_spsm_correlations_gpu(gpu_psi, acc, cluster.n_sites);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " (" << duration.count() << " ms)" << std::endl;
    
    // Compute S(q)
    std::cout << "  Computing S(q)..." << std::flush;
    start = std::chrono::high_resolution_clock::now();
    
    int n_k = cluster.k_points.size();
    std::vector<Complex> s_q(n_k, 0.0);
    
    for (int ik = 0; ik < n_k; ++ik) {
        const auto& q = cluster.k_points[ik];
        Complex s = 0.0;
        for (int i = 0; i < cluster.n_sites; ++i) {
            for (int j = 0; j < cluster.n_sites; ++j) {
                double dr_x = cluster.positions[i][0] - cluster.positions[j][0];
                double dr_y = cluster.positions[i][1] - cluster.positions[j][1];
                double phase_arg = q[0] * dr_x + q[1] * dr_y;
                s += spsm_corr[i][j] * std::exp(I_CPU * phase_arg);
            }
        }
        s_q[ik] = s / static_cast<double>(cluster.n_sites);
    }
    
    double max_val = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        double val = std::abs(s_q[ik]);
        if (val > max_val) {
            max_val = val;
            results.q_max_idx = ik;
            results.s_q_max = val;
            results.q_max = cluster.k_points[ik];
        }
    }
    results.m_translation = std::sqrt(max_val / cluster.n_sites);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " (" << duration.count() << " ms)" << std::endl;
    
    // Compute nematic order
    std::cout << "  Computing XY bonds (GPU)..." << std::flush;
    start = std::chrono::high_resolution_clock::now();
    auto bond_exp = compute_xy_bond_expectations_gpu(gpu_psi, acc, cluster);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " (" << duration.count() << " ms)" << std::endl;
    
    std::array<Complex, 3> O_bar = {0.0, 0.0, 0.0};
    std::array<int, 3> count = {0, 0, 0};
    for (const auto& [edge, exp_val] : bond_exp) {
        int alpha = cluster.bond_orientation.at(edge);
        O_bar[alpha] += exp_val;
        count[alpha]++;
    }
    for (int a = 0; a < 3; ++a) {
        if (count[a] > 0) O_bar[a] /= static_cast<double>(count[a]);
    }
    
    Complex omega = std::exp(2.0 * PI * I_CPU / 3.0);
    Complex psi_nem = O_bar[0] + omega * O_bar[1] + omega * omega * O_bar[2];
    results.m_nematic = std::abs(psi_nem);
    
    std::array<double, 3> mags = {std::abs(O_bar[0]), std::abs(O_bar[1]), std::abs(O_bar[2])};
    double max_mag = *std::max_element(mags.begin(), mags.end());
    double min_mag = *std::min_element(mags.begin(), mags.end());
    results.anisotropy = (max_mag > 1e-10) ? (max_mag - min_mag) / max_mag : 0.0;
    
    // Compute VBS order (bond dimer structure factor)
    std::cout << "  Computing VBS order..." << std::flush;
    start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::pair<int, int>> edges(cluster.edges_nn.begin(), cluster.edges_nn.end());
    int n_bonds = edges.size();
    
    // Compute bond centers
    std::vector<std::array<double, 2>> bond_centers(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        int i = edges[b].first, j = edges[b].second;
        bond_centers[b][0] = 0.5 * (cluster.positions[i][0] + cluster.positions[j][0]);
        bond_centers[b][1] = 0.5 * (cluster.positions[i][1] + cluster.positions[j][1]);
    }
    
    // Mean bond value
    double sum_bond = 0.0;
    for (const auto& [edge, exp_val] : bond_exp) {
        sum_bond += std::real(exp_val);
    }
    results.D_mean = sum_bond / n_bonds;
    
    // Connected bond correlations δD
    std::vector<Complex> delta_D(n_bonds);
    for (int b = 0; b < n_bonds; ++b) {
        delta_D[b] = bond_exp.at(edges[b]) - results.D_mean;
    }
    
    // S_D(q) at discrete k-points
    double s_d_max = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        const auto& q = cluster.k_points[ik];
        Complex s_d = 0.0;
        for (int b1 = 0; b1 < n_bonds; ++b1) {
            for (int b2 = 0; b2 < n_bonds; ++b2) {
                double dr_x = bond_centers[b1][0] - bond_centers[b2][0];
                double dr_y = bond_centers[b1][1] - bond_centers[b2][1];
                double phase_arg = q[0] * dr_x + q[1] * dr_y;
                s_d += delta_D[b1] * std::conj(delta_D[b2]) * std::exp(I_CPU * phase_arg);
            }
        }
        s_d /= static_cast<double>(n_bonds);
        if (std::abs(s_d) > s_d_max) s_d_max = std::abs(s_d);
    }
    results.m_vbs = std::sqrt(s_d_max / n_bonds);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " (" << duration.count() << " ms)" << std::endl;
    
    // Compute stripe order (bond-bond correlations) - can be slow
    if (!skip_stripe && cluster.edges_nn.size() <= 50) {
        std::cout << "  Computing stripe order (GPU)..." << std::flush;
        start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::pair<int, int>> edges(cluster.edges_nn.begin(), cluster.edges_nn.end());
        Complex S_stripe = 0.0;
        
        for (size_t b1 = 0; b1 < edges.size(); ++b1) {
            int i1 = edges[b1].first, j1 = edges[b1].second;
            int alpha1 = cluster.bond_orientation.at(edges[b1]);
            for (size_t b2 = 0; b2 < edges.size(); ++b2) {
                int i2 = edges[b2].first, j2 = edges[b2].second;
                int alpha2 = cluster.bond_orientation.at(edges[b2]);
                
                Complex corr = compute_bond_bond_correlation_gpu(gpu_psi, acc, i1, j1, i2, j2);
                Complex phase = std::pow(omega, alpha2 - alpha1);
                S_stripe += corr * phase;
            }
        }
        
        S_stripe /= static_cast<double>(edges.size());
        results.m_stripe = std::sqrt(std::abs(S_stripe) / edges.size());
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << " (" << duration.count() << " s)" << std::endl;
    } else {
        results.m_stripe = 0.0;
        if (skip_stripe) std::cout << "  Skipping stripe order (--skip-stripe)" << std::endl;
        else std::cout << "  Skipping stripe order (too many bonds)" << std::endl;
    }
    
    // Compute plaquette/bowtie order
    if (!cluster.bowties.empty()) {
        std::cout << "  Computing bowtie resonance (GPU)..." << std::flush;
        start = std::chrono::high_resolution_clock::now();
        
        auto P_r = compute_bowtie_resonances_gpu(gpu_psi, acc, cluster);
        
        double sum_real = 0.0, sum_abs = 0.0;
        for (const auto& p : P_r) {
            sum_real += std::real(p);
            sum_abs += std::abs(p);
        }
        results.P_mean = sum_real / P_r.size();
        results.resonance_strength = sum_abs / P_r.size();
        
        // Plaquette structure factor
        int n_bowties = cluster.bowties.size();
        std::vector<Complex> delta_P(n_bowties);
        for (int b = 0; b < n_bowties; ++b) {
            delta_P[b] = P_r[b] - results.P_mean;
        }
        
        double s_p_max = 0.0;
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            Complex s_p = 0.0;
            for (int b1 = 0; b1 < n_bowties; ++b1) {
                for (int b2 = 0; b2 < n_bowties; ++b2) {
                    double dr_x = cluster.bowties[b1].center[0] - cluster.bowties[b2].center[0];
                    double dr_y = cluster.bowties[b1].center[1] - cluster.bowties[b2].center[1];
                    double phase_arg = q[0] * dr_x + q[1] * dr_y;
                    s_p += delta_P[b1] * std::conj(delta_P[b2]) * std::exp(I_CPU * phase_arg);
                }
            }
            s_p /= static_cast<double>(n_bowties);
            if (std::abs(s_p) > s_p_max) s_p_max = std::abs(s_p);
        }
        results.m_plaquette = std::sqrt(s_p_max / n_bowties);
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << " (" << duration.count() << " ms)" << std::endl;
    } else {
        results.m_plaquette = 0.0;
        results.resonance_strength = 0.0;
        results.P_mean = 0.0;
    }
    
    return results;
}

// -----------------------------------------------------------------------------
// Scan directory mode
// -----------------------------------------------------------------------------

std::vector<OrderParameterResults> scan_jpm_directories(
    const std::string& scan_dir,
    const std::string& output_dir,
    int n_workers,
    bool skip_stripe
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
    
    // Get number of GPUs
    int n_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    std::cout << "Available GPUs: " << n_gpus << std::endl;
    
    #pragma omp parallel for schedule(dynamic) num_threads(std::min(n_workers, n_gpus))
    for (size_t i = 0; i < jpm_dirs.size(); ++i) {
        int thread_id = 0;
        #ifdef _OPENMP
        thread_id = omp_get_thread_num();
        #endif
        
        // Set GPU for this thread
        int gpu_id = thread_id % n_gpus;
        CUDA_CHECK(cudaSetDevice(gpu_id));
        
        double jpm = jpm_dirs[i].first;
        const std::string& dir = jpm_dirs[i].second;
        
        try {
            // Find wavefunction file
            std::string wf_file;
            for (const auto& entry : fs::directory_iterator(dir)) {
                std::string fname = entry.path().filename().string();
                if (fname.find(".h5") != std::string::npos && 
                    fname.find("eigenvector") != std::string::npos) {
                    wf_file = entry.path().string();
                    break;
                }
            }
            
            if (wf_file.empty()) {
                // Try other patterns
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
            
            // Load wavefunction
            auto psi = load_wavefunction(wf_file);
            
            // Compute order parameters
            auto results = compute_order_parameters_gpu(psi, cluster, jpm, skip_stripe);
            all_results[i] = results;
            
            int done = ++completed;
            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "[" << done << "/" << jpm_dirs.size() << "] "
                          << "Jpm=" << std::fixed << std::setprecision(4) << jpm
                          << " | m_trans=" << std::setprecision(6) << results.m_translation
                          << " | m_vbs=" << results.m_vbs
                          << " | m_plaq=" << results.m_plaquette
                          << " | res=" << results.resonance_strength
                          << " (GPU " << gpu_id << ")" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "Error processing " << dir << ": " << e.what() << std::endl;
        }
    }
    
    return all_results;
}

// -----------------------------------------------------------------------------
// Save results to HDF5
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
        std::vector<double> m_vbs_vals(n), D_mean_vals(n);
        
        for (size_t i = 0; i < n; ++i) {
            jpm_vals[i] = results[i].jpm;
            m_trans[i] = results[i].m_translation;
            m_nem[i] = results[i].m_nematic;
            m_stripe[i] = results[i].m_stripe;
            m_vbs_vals[i] = results[i].m_vbs;
            D_mean_vals[i] = results[i].D_mean;
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
        write_dataset("m_vbs", m_vbs_vals);
        write_dataset("D_mean", D_mean_vals);
        write_dataset("m_plaquette", m_plaq);
        write_dataset("resonance_strength", resonance);
        write_dataset("anisotropy", aniso);
        write_dataset("P_mean", P_mean);
        
        std::cout << "Results saved to: " << output_file << std::endl;
        
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
              << "    " << prog << " --scan-dir <dir> --output-dir <out> [options]\n\n"
              << "Options:\n"
              << "  --scan-dir <dir>     Directory containing Jpm=* subdirectories\n"
              << "  --output-dir <dir>   Output directory for results\n"
              << "  --n-workers <n>      Number of parallel workers (default: num_gpus)\n"
              << "  --skip-stripe        Skip stripe order computation (faster)\n"
              << "  --eigenvector <i>    Index of eigenvector to analyze (default: 0)\n"
              << "\nComputes BFG order parameters (GPU accelerated):\n"
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
    int n_workers = 0;  // 0 = auto (use num_gpus)
    int eigenvector_idx = 0;
    bool skip_stripe = false;
    bool scan_mode = false;
    
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
        } else if (arg == "--skip-stripe") {
            skip_stripe = true;
        } else if (arg == "--eigenvector" && i + 1 < argc) {
            eigenvector_idx = std::stoi(argv[++i]);
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
    
    // Check CUDA
    int n_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (n_gpus == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    std::cout << "CUDA devices available: " << n_gpus << std::endl;
    
    if (n_workers == 0) n_workers = n_gpus;
    
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
                      << "BFG ORDER PARAMETER SCAN (GPU)\n"
                      << "========================================\n"
                      << "Scan directory: " << scan_dir << "\n"
                      << "Output directory: " << output_dir << "\n"
                      << "Workers: " << n_workers << " (GPUs: " << n_gpus << ")\n"
                      << "========================================" << std::endl;
            
            auto results = scan_jpm_directories(scan_dir, output_dir, n_workers, skip_stripe);
            
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
                      << "BFG ORDER PARAMETER COMPUTATION (GPU)\n"
                      << "========================================" << std::endl;
            
            Cluster cluster = load_cluster(cluster_dir);
            std::cout << "Cluster: " << cluster.n_sites << " sites, "
                      << cluster.edges_nn.size() << " bonds, "
                      << cluster.bowties.size() << " bowties" << std::endl;
            
            auto psi = load_wavefunction(wf_file, eigenvector_idx);
            std::cout << "Wavefunction: " << psi.size() << " amplitudes" << std::endl;
            
            auto results = compute_order_parameters_gpu(psi, cluster, 0.0, skip_stripe);
            
            std::cout << "\n========== RESULTS ==========\n"
                      << std::fixed << std::setprecision(6)
                      << "m_translation:     " << results.m_translation << "\n"
                      << "m_nematic:         " << results.m_nematic << "\n"
                      << "m_stripe:          " << results.m_stripe << "\n"
                      << "m_plaquette:       " << results.m_plaquette << "\n"
                      << "resonance_strength:" << results.resonance_strength << "\n"
                      << "P_mean:            " << results.P_mean << "\n"
                      << "anisotropy:        " << results.anisotropy << "\n"
                      << "q_max:             (" << results.q_max[0] << ", " << results.q_max[1] << ")\n"
                      << "=============================" << std::endl;
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

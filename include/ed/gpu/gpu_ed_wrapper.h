#pragma once

// Forward declaration to avoid including construct_ham.h which has CUDA-incompatible code
class Operator;

#include <vector>
#include <complex>
#include <functional>
#include <string>
#include <tuple>
#include <map>
#include <cstdint>

// Forward declarations only - don't include CUDA headers in this header
// They will be included in the .cu implementation file

/**
 * Wrapper class to integrate GPU operators with existing ED code
 * Provides a unified interface that works with both CPU and GPU implementations
 */
class GPUEDWrapper {
public:
    /**
     * Run Lanczos algorithm on GPU
     * Compatible with existing Lanczos interface.
     * ``seed`` (defaulted 0) sets the GPU Lanczos starting-vector RNG;
     * 0 keeps the legacy deterministic seed (42), nonzero passes
     * through verbatim. Closes the audit S1 #21 silent CPU/GPU
     * divergence gap.
     */
    static void runGPULanczos(void* gpu_op_handle,
                             int N, int max_iter, int num_eigs,
                             double tol,
                             std::vector<double>& eigenvalues,
                             std::string dir = "",
                             bool eigenvectors = false,
                             unsigned long long seed = 0ULL);

    /**
     * Run GPU Lanczos for Fixed Sz sector. ``seed`` semantics match
     * the full-Hilbert overload above.
     */
    static void runGPULanczosFixedSz(void* gpu_op_handle,
                                    int n_up,
                                    int max_iter, int num_eigs,
                                    double tol,
                                    std::vector<double>& eigenvalues,
                                    std::string dir = "",
                                    bool eigenvectors = false,
                                    unsigned long long seed = 0ULL);
    
    /**
     * Run Block Lanczos algorithm on GPU
     * Better for degenerate eigenvalues and improved convergence
     */
    static void runGPUBlockLanczos(void* gpu_op_handle,
                                  int N, int max_iter, int num_eigs,
                                  int block_size,
                                  double tol,
                                  std::vector<double>& eigenvalues,
                                  std::string dir = "",
                                  bool eigenvectors = false);
    
    /**
     * Run GPU Block Lanczos for Fixed Sz sector
     */
    static void runGPUBlockLanczosFixedSz(void* gpu_op_handle,
                                         int n_up,
                                         int max_iter, int num_eigs,
                                         int block_size,
                                         double tol,
                                         std::vector<double>& eigenvalues,
                                         std::string dir = "",
                                         bool eigenvectors = false);
    
    /**
     * Run GPU TPQ for Fixed Sz sector (microcanonical)
     */
    static void runGPUMicrocanonicalTPQFixedSz(void* gpu_op_handle,
                                              int n_up,
                                              int max_iter, int num_samples,
                                              int temp_interval,
                                              std::vector<double>& eigenvalues,
                                              std::string dir = "",
                                              double large_value = 1e5,
                                              bool continue_quenching = false,
                                              int continue_sample = 0,
                                              double continue_beta = 0.0,
                                              bool save_thermal_states = false,
                                              double target_beta = 1000.0,
                                              int num_measure_points = 20,
                                              double measure_beta_min = 1.0,
                                              double measure_beta_max = 1000.0);
    
    /**
     * Run GPU TPQ for Fixed Sz sector (canonical)
     */
    static void runGPUCanonicalTPQFixedSz(void* gpu_op_handle,
                                         int n_up,
                                         double beta_max, int num_samples,
                                         int temp_interval,
                                         std::vector<double>& energies,
                                         std::string dir = "",
                                         double delta_beta = 0.1,
                                         int taylor_order = 50,
                                         int num_measure_points = 20,
                                         double measure_beta_min = 1.0,
                                         double measure_beta_max = 1000.0);
    
    /**
     * Create GPU Fixed Sz operator directly from interaction lists
     */
    static void* createGPUFixedSzOperatorDirect(int n_sites, int n_up, float spin_l,
                                               const std::vector<std::tuple<int, int, char, char, double>>& interactions,
                                               const std::vector<std::tuple<int, char, double>>& single_site_ops);
    
    /**
     * Create GPU operator directly from interaction lists
     */
    static void* createGPUOperatorDirect(int n_sites,
                                        const std::vector<std::tuple<int, int, char, char, double>>& interactions,
                                        const std::vector<std::tuple<int, char, double>>& single_site_ops);
    
    /**
     * Create GPU operator from InterAll.dat and Trans.dat files
     * Standard format used by the ED pipeline
     */
    static void* createGPUOperatorFromFiles(int n_sites,
                                           const std::string& interall_file,
                                           const std::string& trans_file);

    /**
     * Clean up GPU resources
     */
    static void destroyGPUOperator(void* gpu_op_handle);
    
    /**
     * Check if GPU is available and ready
     */
    static bool isGPUAvailable();
    
    /**
     * Get GPU device information
     */
    static void printGPUInfo();
    
    /**
     * Run GPU-accelerated microcanonical TPQ
     */
    static void runGPUMicrocanonicalTPQ(void* gpu_op_handle,
                                        int N, int max_iter, int num_samples,
                                        int temp_interval,
                                        std::vector<double>& eigenvalues,
                                        std::string dir = "",
                                        double large_value = 1e5,
                                        bool continue_quenching = false,
                                        int continue_sample = 0,
                                        double continue_beta = 0.0,
                                        bool save_thermal_states = false,
                                        double target_beta = 1000.0,
                                        int num_measure_points = 20,
                                        double measure_beta_min = 1.0,
                                        double measure_beta_max = 1000.0);
    
    /**
     * Run GPU-accelerated canonical TPQ
     */
    static void runGPUCanonicalTPQ(void* gpu_op_handle,
                                   int N, double beta_max, int num_samples,
                                   int temp_interval,
                                   std::vector<double>& energies,
                                   std::string dir = "",
                                   double delta_beta = 0.1,
                                   int taylor_order = 50,
                                   int num_measure_points = 20,
                                   double measure_beta_min = 1.0,
                                   double measure_beta_max = 1000.0);
    
    /**
     * Run GPU-accelerated Krylov-Schur algorithm
     * Restarted eigenvalue solver optimal for computing many eigenvalues
     * 
     * @param gpu_op_handle GPU Hamiltonian operator handle
     * @param N Hilbert space dimension
     * @param num_eigenvalues Number of eigenvalues to compute
     * @param max_iter Maximum Krylov subspace size per restart cycle
     * @param tol Convergence tolerance
     * @param eigenvalues Output: computed eigenvalues
     * @param dir Output directory for results
     * @param compute_eigenvectors Whether to compute and save eigenvectors
     */
    static void runGPUKrylovSchur(void* gpu_op_handle,
                                 int N, int num_eigenvalues, int max_iter,
                                 double tol,
                                 std::vector<double>& eigenvalues,
                                 std::string dir = "",
                                 bool compute_eigenvectors = false);
    
    /**
     * Run GPU-accelerated Krylov-Schur for Fixed Sz sector
     */
    static void runGPUKrylovSchurFixedSz(void* gpu_op_handle,
                                        int n_up,
                                        int num_eigenvalues, int max_iter,
                                        double tol,
                                        std::vector<double>& eigenvalues,
                                        std::string dir = "",
                                        bool compute_eigenvectors = false);
    
    /**
     * Run GPU-accelerated Finite Temperature Lanczos Method (FTLM)
     */
    static void runGPUFTLM(void* gpu_op_handle,
                          int N,
                          int krylov_dim,
                          int num_samples,
                          double temp_min,
                          double temp_max,
                          int num_temp_bins,
                          double tolerance,
                          std::string dir = "",
                          bool full_reorth = false,
                          int reorth_freq = 10,
                          unsigned int random_seed = 0);
    
    // The following entry points were retired in the minimalist-architecture rev
    // (May 2026): `runGPUFTLMFixedSz`, `runGPUDynamicalResponse`,
    // `runGPUDynamicalResponseThermal`, `runGPUDynamicalCorrelation`,
    // `runGPUDynamicalCorrelationState`, `runGPUDynamicalCorrelationStateCF`,
    // `runGPUThermalExpectation`. The live GPU DSSF surface is now just
    // `runGPUDynamicalCorrelationMultiTemp` (cross-correlator, thermal, multi-T)
    // and `runGPUStaticCorrelation` (static thermal correlator), reached via
    // `src/cli/workflows.cpp`. CPU consumers should use
    // `ed::observables::cf_dynamical_correlator` etc.

    /**
     * Run GPU-accelerated multi-temperature dynamical correlation (OPTIMIZED)
     * Runs Lanczos once per sample, then computes all temperatures efficiently
     * Equivalent to compute_dynamical_correlation_multi_sample_multi_temperature on CPU
     * 
     * @param gpu_op_handle GPU Hamiltonian operator handle
     * @param gpu_obs1_handle First GPU observable operator handle
     * @param gpu_obs2_handle Second GPU observable operator handle
     * @param N Hilbert space dimension
     * @param num_samples Number of random samples
     * @param krylov_dim Lanczos order
     * @param omega_min Minimum frequency
     * @param omega_max Maximum frequency
     * @param num_omega_bins Number of frequency points
     * @param broadening Lorentzian broadening parameter
     * @param temperatures Vector of temperature points
     * @param random_seed Random seed (0 = random)
     * @param ground_state_energy Ground state energy for frequency shift (0 = auto-detect)
     * @return map<temperature, tuple(frequencies, S_real, S_imag)>
     */
    static std::map<double, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>
    runGPUDynamicalCorrelationMultiTemp(void* gpu_op_handle,
                                       void* gpu_obs1_handle,
                                       void* gpu_obs2_handle,
                                       int N,
                                       int num_samples,
                                       int krylov_dim,
                                       double omega_min,
                                       double omega_max,
                                       int num_omega_bins,
                                       double broadening,
                                       const std::vector<double>& temperatures,
                                       unsigned int random_seed = 0,
                                       double ground_state_energy = 0.0);
    
    /**
     * Run GPU-accelerated static correlation function calculation
     * Computes ⟨O₁†O₂⟩_T via FTLM
     * 
     * @param gpu_op_handle GPU Hamiltonian operator handle
     * @param gpu_obs1_handle First GPU observable operator handle
     * @param gpu_obs2_handle Second GPU observable operator handle
     * @param N Hilbert space dimension
     * @param num_samples Number of random samples
     * @param krylov_dim Lanczos order
     * @param temp_min Minimum temperature
     * @param temp_max Maximum temperature
     * @param num_temp_bins Number of temperature points
     * @param random_seed Random seed (0 = random)
     * @return tuple(temperatures, corr_real, corr_imag, error_real, error_imag)
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
                     std::vector<double>, std::vector<double>>
    runGPUStaticCorrelation(void* gpu_op_handle,
                           void* gpu_obs1_handle,
                           void* gpu_obs2_handle,
                           int N,
                           int num_samples,
                           int krylov_dim,
                           double temp_min,
                           double temp_max,
                           int num_temp_bins,
                           unsigned int random_seed = 0);
    
    /**
     * Create GPU Symmetrized operator for a single symmetry sector
     * 
     * @param n_sites Number of lattice sites
     * @param spin_l Spin quantum number (default 0.5)
     * @param sector_dim Number of symmetrized basis states in this sector
     * @param orbit_elements Flattened orbit elements (all orbits concatenated)
     * @param orbit_coefficients Matching complex coefficients
     * @param orbit_offsets CSR-style offsets into orbit arrays per basis state
     * @param orbit_norms Normalization factor for each basis state
     * @param group_size |G|, the order of the symmetry group
     * @param interall_file Path to InterAll.dat
     * @param trans_file Path to Trans.dat
     * @return Opaque handle to GPUSymmetrizedOperator (caller must destroyGPUOperator)
     */
    static void* createGPUSymmetrizedOperator(
        int n_sites, float spin_l,
        int sector_dim,
        const std::vector<uint64_t>& orbit_elements,
        const std::vector<std::complex<double>>& orbit_coefficients,
        const std::vector<int>& orbit_offsets,
        const std::vector<double>& orbit_norms,
        int group_size,
        const std::string& interall_file,
        const std::string& trans_file);

    /**
     * Run full (dense) diagonalization on GPU using cuSOLVER zheevd
     * 
     * Builds the dense Hamiltonian matrix on GPU column-by-column via matvec,
     * then computes ALL eigenvalues (and optionally eigenvectors) using the
     * divide-and-conquer algorithm.
     *
     * @param gpu_op_handle Opaque handle to GPUOperator (any subclass)
     * @param N Hilbert space dimension (or sector dimension)
     * @param num_eigenvalues Number of lowest eigenvalues to return (0 = all)
     * @param eigenvalues Output vector of eigenvalues
     * @param dir Output directory for HDF5 results (empty = no save)
     * @param compute_eigenvectors Whether to compute and save eigenvectors
     */
    static void runGPUFullDiag(void* gpu_op_handle,
                               int N, int num_eigenvalues,
                               std::vector<double>& eigenvalues,
                               std::string dir = "",
                               bool compute_eigenvectors = true);

private:
    static int getGPUCount();
    static size_t getAvailableGPUMemory(int device = 0);
};

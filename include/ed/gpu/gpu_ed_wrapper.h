#ifndef GPU_ED_WRAPPER_H
#define GPU_ED_WRAPPER_H

// Forward declaration to avoid including construct_ham.h which has CUDA-incompatible code
class Operator;

#include <vector>
#include <complex>
#include <functional>
#include <string>
#include <tuple>
#include <map>

// Forward declarations only - don't include CUDA headers in this header
// They will be included in the .cu implementation file

/**
 * Wrapper class to integrate GPU operators with existing ED code
 * Provides a unified interface that works with both CPU and GPU implementations
 */
class GPUEDWrapper {
public:
    /**
     * Create GPU operator from CPU Operator
     * Transfers interaction data to GPU
     */
    static bool createGPUOperatorFromCPU(const Operator& cpu_op, 
                                        void** gpu_op_handle,
                                        int n_sites);
    
    /**
     * Run Lanczos algorithm on GPU
     * Compatible with existing Lanczos interface
     */
    static void runGPULanczos(void* gpu_op_handle,
                             int N, int max_iter, int num_eigs,
                             double tol,
                             std::vector<double>& eigenvalues,
                             std::string dir = "",
                             bool eigenvectors = false);
    
    /**
     * Run GPU Lanczos for Fixed Sz sector
     */
    static void runGPULanczosFixedSz(void* gpu_op_handle,
                                    int n_up,
                                    int max_iter, int num_eigs,
                                    double tol,
                                    std::vector<double>& eigenvalues,
                                    std::string dir = "",
                                    bool eigenvectors = false);
    
    /**
     * Run GPU Davidson for Fixed Sz sector
     */
    static void runGPUDavidsonFixedSz(void* gpu_op_handle,
                                     int n_up,
                                     int num_eigenvalues, int max_iter,
                                     int max_subspace, double tol,
                                     std::vector<double>& eigenvalues,
                                     std::string dir = "",
                                     bool compute_eigenvectors = false);
    
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
                                              bool save_thermal_states = false);
    
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
                                         int taylor_order = 50);
    
    /**
     * Create GPU Fixed Sz operator directly from interaction lists
     */
    static void* createGPUFixedSzOperatorDirect(int n_sites, int n_up, float spin_l,
                                               const std::vector<std::tuple<int, int, char, char, double>>& interactions,
                                               const std::vector<std::tuple<int, char, double>>& single_site_ops);
    
    /**
     * Matrix-vector product using GPU
     * y = H * x
     */
    static void gpuMatVec(void* gpu_op_handle,
                         const std::complex<double>* x,
                         std::complex<double>* y,
                         int N);
    
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
     * Create GPU operator from CSR arrays (host-side)
     * row_ptr size = N+1, col_ind size = nnz, values size = nnz
     */
    static void* createGPUOperatorFromCSR(int n_sites,
                                         int N,
                                         const std::vector<int>& row_ptr,
                                         const std::vector<int>& col_ind,
                                         const std::vector<std::complex<double>>& values);
    
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
     * Estimate memory requirements for GPU computation
     */
    static size_t estimateGPUMemory(int n_sites, bool fixed_sz = false, int n_up = 0);
    
    /**
     * Automatic decision: use GPU or CPU based on problem size and available resources
     */
    static bool shouldUseGPU(int n_sites, bool fixed_sz = false);
    
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
                                        bool save_thermal_states = false);
    
    /**
     * Run GPU-accelerated canonical TPQ
     */
    static void runGPUCanonicalTPQ(void* gpu_op_handle,
                                   int N, double beta_max, int num_samples,
                                   int temp_interval,
                                   std::vector<double>& energies,
                                   std::string dir = "",
                                   double delta_beta = 0.1,
                                   int taylor_order = 50);
    
    /**
     * Run GPU-accelerated Davidson method
     */
    static void runGPUDavidson(void* gpu_op_handle,
                              int N, int num_eigenvalues, int max_iter,
                              int max_subspace, double tol,
                              std::vector<double>& eigenvalues,
                              std::string dir = "",
                              bool compute_eigenvectors = false);
    
    /**
     * Run GPU-accelerated LOBPCG method
     * @deprecated This method now redirects to Davidson GPU for better stability
     */
    static void runGPULOBPCG(void* gpu_op_handle,
                            int N, int num_eigenvalues, int max_iter,
                            double tol,
                            std::vector<double>& eigenvalues,
                            std::string dir = "",
                            bool compute_eigenvectors = false);
    
    /**
     * Run GPU-accelerated LOBPCG method for Fixed Sz sector
     * @deprecated This method now redirects to Davidson GPU for better stability
     */
    static void runGPULOBPCGFixedSz(void* gpu_op_handle,
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
    
    /**
     * Run GPU-accelerated FTLM for Fixed Sz sector
     */
    static void runGPUFTLMFixedSz(void* gpu_op_handle,
                                 int n_up,
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
    
    /**
     * Run GPU-accelerated dynamical response (spectral function) for a single state
     * Computes S(ω) = <ψ|O†δ(ω - H)O|ψ>
     * 
     * @param gpu_op_handle GPU Hamiltonian operator handle
     * @param gpu_obs_handle GPU observable operator handle (nullptr = identity)
     * @param d_psi_state Device pointer to initial state |ψ> (normalized)
     * @param N Hilbert space dimension
     * @param krylov_dim Lanczos order
     * @param omega_min Minimum frequency
     * @param omega_max Maximum frequency
     * @param num_omega_bins Number of frequency points
     * @param broadening Lorentzian broadening parameter (eta)
     * @param temperature Temperature for thermal weighting (0 = none)
     * @param ground_state_energy Ground state energy for frequency shift
     * @return tuple(frequencies, spectral_function)
     */
    static std::pair<std::vector<double>, std::vector<double>>
    runGPUDynamicalResponse(void* gpu_op_handle,
                           void* gpu_obs_handle,
                           void* d_psi_state,
                           int N,
                           int krylov_dim,
                           double omega_min,
                           double omega_max,
                           int num_omega_bins,
                           double broadening,
                           double temperature = 0.0,
                           double ground_state_energy = 0.0);
    
    /**
     * Run GPU-accelerated thermal dynamical response (spectral function) with FTLM averaging
     * Computes S(ω,T) averaged over random samples
     * 
     * @param gpu_op_handle GPU Hamiltonian operator handle
     * @param gpu_obs_handle GPU observable operator handle
     * @param N Hilbert space dimension
     * @param num_samples Number of random samples for thermal averaging
     * @param krylov_dim Lanczos order
     * @param omega_min Minimum frequency
     * @param omega_max Maximum frequency
     * @param num_omega_bins Number of frequency points
     * @param broadening Lorentzian broadening parameter
     * @param temperature Temperature
     * @param random_seed Random seed (0 = random)
     * @param ground_state_energy Ground state energy for frequency shift
     * @return tuple(frequencies, avg_spectral_function, error_bars)
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    runGPUDynamicalResponseThermal(void* gpu_op_handle,
                                  void* gpu_obs_handle,
                                  int N,
                                  int num_samples,
                                  int krylov_dim,
                                  double omega_min,
                                  double omega_max,
                                  int num_omega_bins,
                                  double broadening,
                                  double temperature,
                                  unsigned int random_seed = 0,
                                  double ground_state_energy = 0.0);
    
    /**
     * Run GPU-accelerated dynamical correlation between two operators
     * Computes S_{O1,O2}(ω) = Σ_n ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ δ(ω - E_n)
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
     * @param temperature Temperature
     * @param random_seed Random seed (0 = random)
     * @param ground_state_energy Ground state energy for frequency shift
     * @return tuple(frequencies, S_real, S_imag, error_real, error_imag)
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
                     std::vector<double>, std::vector<double>>
    runGPUDynamicalCorrelation(void* gpu_op_handle,
                              void* gpu_obs1_handle,
                              void* gpu_obs2_handle,
                              int N,
                              int num_samples,
                              int krylov_dim,
                              double omega_min,
                              double omega_max,
                              int num_omega_bins,
                              double broadening,
                              double temperature,
                              unsigned int random_seed = 0,
                              double ground_state_energy = 0.0);
    
    /**
     * Run GPU-accelerated dynamical correlation for a specific state (no sampling)
     * Computes S_{O1,O2}(ω) = Σ_n ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ δ(ω - E_n) for given state |ψ⟩
     * 
     * @param gpu_op_handle GPU Hamiltonian operator handle
     * @param gpu_obs1_handle First GPU observable operator handle
     * @param gpu_obs2_handle Second GPU observable operator handle
     * @param d_psi_state Device pointer to initial state |ψ> (must be on GPU, normalized)
     * @param N Hilbert space dimension
     * @param krylov_dim Lanczos order
     * @param omega_min Minimum frequency
     * @param omega_max Maximum frequency
     * @param num_omega_bins Number of frequency points
     * @param broadening Lorentzian broadening parameter
     * @param temperature Temperature for thermal weighting (0 = none)
     * @param ground_state_energy Ground state energy for frequency shift (0 = auto-detect)
     * @param operators_identical -1 = auto-detect via pointer comparison, 0 = false, 1 = true
     * @return tuple(frequencies, S_real, S_imag)
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    runGPUDynamicalCorrelationState(void* gpu_op_handle,
                                   void* gpu_obs1_handle,
                                   void* gpu_obs2_handle,
                                   void* d_psi_state,
                                   int N,
                                   int krylov_dim,
                                   double omega_min,
                                   double omega_max,
                                   int num_omega_bins,
                                   double broadening,
                                   double temperature = 0.0,
                                   double ground_state_energy = 0.0,
                                   int operators_identical = -1);
    
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
     * Run GPU-accelerated thermal expectation value calculation
     * Computes ⟨O⟩_T and susceptibility χ_T = β(⟨O²⟩ - ⟨O⟩²) via FTLM
     * 
     * @param gpu_op_handle GPU Hamiltonian operator handle
     * @param gpu_obs_handle GPU observable operator handle
     * @param N Hilbert space dimension
     * @param num_samples Number of random samples
     * @param krylov_dim Lanczos order
     * @param temp_min Minimum temperature
     * @param temp_max Maximum temperature
     * @param num_temp_bins Number of temperature points
     * @param random_seed Random seed (0 = random)
     * @return tuple(temperatures, expectation, susceptibility, exp_error, sus_error)
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
                     std::vector<double>, std::vector<double>>
    runGPUThermalExpectation(void* gpu_op_handle,
                            void* gpu_obs_handle,
                            int N,
                            int num_samples,
                            int krylov_dim,
                            double temp_min,
                            double temp_max,
                            int num_temp_bins,
                            unsigned int random_seed = 0);
    
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
    
private:
    static int getGPUCount();
    static size_t getAvailableGPUMemory(int device = 0);
};

#endif // GPU_ED_WRAPPER_H

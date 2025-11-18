#ifndef GPU_FTLM_CUH
#define GPU_FTLM_CUH

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <vector>
#include <complex>
#include <functional>
#include "gpu_operator.cuh"
#include "../core/thermal_types.h"

// Forward declare CPU functions for averaging (defined in ftlm.cpp)
void average_ftlm_samples(const std::vector<ThermodynamicData>& sample_data, FTLMResults& results);
void save_ftlm_results(const FTLMResults& results, const std::string& filename);

/**
 * @brief GPU-accelerated Finite Temperature Lanczos Method (FTLM)
 * 
 * Implements FTLM on GPU for computing finite-temperature thermodynamic properties.
 * Uses Lanczos method to build Krylov subspace and extract thermal properties from
 * the microcanonical spectrum approximated by Ritz values.
 */
class GPUFTLMSolver {
public:
    /**
     * @brief Constructor
     * @param op GPU operator (Hamiltonian)
     * @param N Hilbert space dimension
     * @param krylov_dim Maximum Krylov subspace dimension
     * @param tolerance Convergence tolerance for Lanczos
     */
    GPUFTLMSolver(GPUOperator* op, int N, int krylov_dim = 100, double tolerance = 1e-10);
    
    /**
     * @brief Destructor - cleanup GPU memory
     */
    ~GPUFTLMSolver();
    
    /**
     * @brief Run FTLM calculation
     * @param num_samples Number of random initial states to sample
     * @param temp_min Minimum temperature
     * @param temp_max Maximum temperature
     * @param num_temp_bins Number of temperature points
     * @param output_dir Output directory for results
     * @param full_reorth Use full reorthogonalization (slower but more accurate)
     * @param reorth_freq Selective reorthogonalization frequency (0 = none)
     * @param random_seed Random seed (0 = use system clock)
     * @return FTLMResults structure with thermodynamic data
     */
    FTLMResults run(int num_samples,
                   double temp_min, double temp_max, int num_temp_bins,
                   const std::string& output_dir = "",
                   bool full_reorth = false,
                   int reorth_freq = 10,
                   unsigned int random_seed = 0);
    
    /**
     * @brief Run FTLM for a single random sample
     * @param seed Random seed for initial state
     * @param alpha Output: diagonal elements of tridiagonal matrix
     * @param beta Output: off-diagonal elements of tridiagonal matrix
     * @return Number of Lanczos iterations performed
     */
    int runSingleSample(unsigned int seed,
                       std::vector<double>& alpha,
                       std::vector<double>& beta);
    
    /**
     * @brief Compute thermodynamics from Lanczos tridiagonal matrix
     * @param alpha Diagonal elements
     * @param beta Off-diagonal elements
     * @param temperatures Temperature grid
     * @return ThermodynamicData structure
     */
    ThermodynamicData computeThermodynamics(
        const std::vector<double>& alpha,
        const std::vector<double>& beta,
        const std::vector<double>& temperatures);
    
    /**
     * @brief Get performance statistics
     */
    struct Stats {
        double total_time;
        double lanczos_time;
        double diag_time;
        double thermo_time;
        int total_iterations;
        int num_samples_completed;
    };
    
    Stats getStats() const { return stats_; }
    
    /**
     * @brief Compute dynamical response S(ω) for operator O applied to a given state
     * 
     * GPU-accelerated version of compute_dynamical_response.
     * Computes S(ω) = <ψ|O†δ(ω - H)O|ψ> where:
     * - H is the Hamiltonian (via GPU operator)
     * - O is an operator (via GPU operator)
     * - |ψ> is an initial state vector (on GPU)
     * 
     * @param d_psi Initial state vector on GPU (normalized)
     * @param op_O GPU operator for O (if nullptr, uses H)
     * @param omega_min Minimum frequency
     * @param omega_max Maximum frequency
     * @param num_omega_bins Number of frequency points
     * @param broadening Lorentzian broadening parameter (η)
     * @param temperature Temperature for thermal weighting (0 = none)
     * @return Spectral function data (frequencies and S(ω))
     */
    std::pair<std::vector<double>, std::vector<double>> 
    computeDynamicalResponse(const cuDoubleComplex* d_psi,
                           GPUOperator* op_O,
                           double omega_min,
                           double omega_max,
                           int num_omega_bins,
                           double broadening,
                           double temperature = 0.0);
    
    /**
     * @brief Compute dynamical response averaged over random initial states
     * 
     * GPU-accelerated thermal dynamical response using FTLM approach.
     * Averages S(ω) over multiple random initial states to approximate
     * finite temperature response.
     * 
     * @param num_samples Number of random initial states
     * @param op_O GPU operator for O (if nullptr, uses H)
     * @param omega_min Minimum frequency
     * @param omega_max Maximum frequency
     * @param num_omega_bins Number of frequency points
     * @param broadening Lorentzian broadening parameter
     * @param temperature Temperature for thermal weighting
     * @param random_seed Random seed (0 = random)
     * @return Spectral function with error bars
     */
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    computeDynamicalResponseThermal(int num_samples,
                                  GPUOperator* op_O,
                                  double omega_min,
                                  double omega_max,
                                  int num_omega_bins,
                                  double broadening,
                                  double temperature = 0.0,
                                  unsigned int random_seed = 0);
    
    /**
     * @brief Compute dynamical correlation S_{O1,O2}(ω) = <O₁†(ω)O₂>
     * 
     * GPU-accelerated computation of cross-correlation spectral function.
     * For O1=O2, gives auto-correlation/spectral density.
     * 
     * @param num_samples Number of random samples for thermal average
     * @param op_O1 GPU operator for O₁
     * @param op_O2 GPU operator for O₂
     * @param omega_min Minimum frequency
     * @param omega_max Maximum frequency
     * @param num_omega_bins Number of frequency points
     * @param broadening Lorentzian broadening parameter
     * @param temperature Temperature for thermal weighting
     * @param energy_shift Ground state energy shift (0 = auto-detect)
     * @param random_seed Random seed (0 = random)
     * @param output_dir Output directory for intermediate files (empty = no output)
     * @param store_intermediate Whether to save per-sample spectra
     * @return Spectral function with real/imaginary parts and errors
     */
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
               std::vector<double>, std::vector<double>>
    computeDynamicalCorrelation(int num_samples,
                              GPUOperator* op_O1,
                              GPUOperator* op_O2,
                              double omega_min,
                              double omega_max,
                              int num_omega_bins,
                              double broadening,
                              double temperature = 0.0,
                              double energy_shift = 0.0,
                              unsigned int random_seed = 0,
                              const std::string& output_dir = "",
                              bool store_intermediate = false);
    
    /**
     * @brief Compute thermal expectation value ⟨O⟩_T and susceptibility
     * 
     * GPU-accelerated computation of thermal averages:
     * - ⟨O⟩_T = Tr(O exp(-βH)) / Z
     * - χ_T = β(⟨O²⟩ - ⟨O⟩²)  [generalized susceptibility]
     * 
     * Uses FTLM approach with multiple random samples.
     * 
     * @param num_samples Number of random samples for thermal average
     * @param op_O GPU operator for O (if nullptr, uses H for energy)
     * @param temp_min Minimum temperature
     * @param temp_max Maximum temperature
     * @param num_temp_bins Number of temperature points
     * @param random_seed Random seed (0 = random)
     * @param output_dir Output directory for intermediate files (empty = no output)
     * @param store_intermediate Whether to save per-sample data
     * @return Tuple of (temperatures, expectation values, errors)
     */
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    computeThermalExpectation(int num_samples,
                            GPUOperator* op_O,
                            double temp_min,
                            double temp_max,
                            int num_temp_bins,
                            unsigned int random_seed = 0,
                            const std::string& output_dir = "",
                            bool store_intermediate = false);
    
    /**
     * @brief Compute static correlation function ⟨O₁†O₂⟩_T
     * 
     * GPU-accelerated computation of static two-point correlation at finite temperature.
     * Computes ⟨O₁†O₂⟩ = Tr(O₁†O₂ exp(-βH)) / Z
     * 
     * This is the static (ω=0) version of the dynamical correlation.
     * Useful for:
     * - Structure factors at q=0
     * - Equal-time correlation functions
     * - Connected correlations (subtract ⟨O₁⟩*⟨O₂⟩*)
     * 
     * @param num_samples Number of random samples for thermal average
     * @param op_O1 GPU operator for O₁
     * @param op_O2 GPU operator for O₂
     * @param temp_min Minimum temperature
     * @param temp_max Maximum temperature
     * @param num_temp_bins Number of temperature points
     * @param random_seed Random seed (0 = random)
     * @param output_dir Output directory for intermediate files (empty = no output)
     * @param store_intermediate Whether to save per-sample data
     * @return Tuple of (temperatures, correlation values, errors)
     */
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    computeStaticCorrelation(int num_samples,
                           GPUOperator* op_O1,
                           GPUOperator* op_O2,
                           double temp_min,
                           double temp_max,
                           int num_temp_bins,
                           unsigned int random_seed = 0,
                           const std::string& output_dir = "",
                           bool store_intermediate = false);
    
    /**
     * @brief Compute dynamical correlation for a given state (single-state version)
     * 
     * GPU-accelerated version of compute_dynamical_correlation_state.
     * Computes S(ω) = Σₙ ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ δ(ω - Eₙ) for a specific state |ψ⟩.
     * 
     * This is the single-state version (no random sampling/thermal averaging).
     * Use this when you have a specific quantum state, such as:
     * - Ground state from exact diagonalization
     * - Excited state from Lanczos
     * - Time-evolved state
     * - Specific symmetry sector state
     * 
     * The function uses the Lehmann representation computed via Lanczos:
     * 1. Applies O₂ to the given state: |φ⟩ = O₂|ψ⟩
     * 2. Builds Krylov subspace starting from |φ⟩
     * 3. Diagonalizes H in the Krylov basis to get approximate eigenstates
     * 4. Computes weights: ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩
     * 5. Constructs spectral function with Lorentzian broadening
     * 
     * For O1=O2, gives spectral density. For different operators, gives cross-correlation.
     * 
     * @param d_psi Input quantum state on GPU (must be normalized)
     * @param op_O1 GPU operator for O₁
     * @param op_O2 GPU operator for O₂
     * @param omega_min Minimum frequency
     * @param omega_max Maximum frequency
     * @param num_omega_bins Number of frequency points
     * @param broadening Lorentzian broadening parameter (η)
     * @param temperature Temperature for Boltzmann weighting (0 = no weighting)
     * @param energy_shift Ground state energy shift (0 = auto-detect from Krylov)
     * @return Tuple of (frequencies, Re[S(ω)], Im[S(ω)])
     */
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    computeDynamicalCorrelationState(const cuDoubleComplex* d_psi,
                                    GPUOperator* op_O1,
                                    GPUOperator* op_O2,
                                    double omega_min,
                                    double omega_max,
                                    int num_omega_bins,
                                    double broadening,
                                    double temperature = 0.0,
                                    double energy_shift = 0.0);
    
private:
    GPUOperator* op_;
    int N_;  // Hilbert space dimension
    int krylov_dim_;
    double tolerance_;
    
    // GPU memory for Lanczos vectors
    cuDoubleComplex* d_v_current_;   // Current Lanczos vector
    cuDoubleComplex* d_v_prev_;      // Previous Lanczos vector
    cuDoubleComplex* d_w_;           // Work vector (H*v)
    cuDoubleComplex* d_temp_;        // Temporary vector
    
    // Stored Lanczos vectors for reorthogonalization (if needed)
    cuDoubleComplex** d_lanczos_basis_;
    int num_stored_vectors_;
    bool store_basis_;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
    
    // Performance statistics
    Stats stats_;
    
    // Memory management
    bool gpu_memory_allocated_;
    void allocateMemory();
    void freeMemory();
    
    // Lanczos iteration helpers
    void initializeRandomVector(cuDoubleComplex* d_vec, unsigned int seed);
    void normalizeVector(cuDoubleComplex* d_vec);
    double vectorNorm(const cuDoubleComplex* d_vec);
    void vectorCopy(const cuDoubleComplex* src, cuDoubleComplex* dst);
    void vectorScale(cuDoubleComplex* d_vec, double scale);
    void vectorAxpy(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                   const cuDoubleComplex& alpha);
    std::complex<double> vectorDot(const cuDoubleComplex* d_x,
                                   const cuDoubleComplex* d_y);
    
    // Orthogonalization
    void orthogonalizeAgainstBasis(cuDoubleComplex* d_vec, int num_basis_vecs);
    void gramSchmidt(cuDoubleComplex* d_vec, int iter);
    
    // Build Lanczos tridiagonal matrix
    int buildLanczosTridiagonal(unsigned int seed,
                               bool full_reorth,
                               int reorth_freq,
                               std::vector<double>& alpha,
                               std::vector<double>& beta);
    
    // Diagonalize tridiagonal matrix (on CPU)
    void diagonalizeTridiagonal(const std::vector<double>& alpha,
                               const std::vector<double>& beta,
                               std::vector<double>& ritz_values,
                               std::vector<double>& weights);

    // Helper functions for spectral calculations
    /**
     * @brief Build Lanczos tridiagonal from a given starting vector
     */
    int buildLanczosTridiagonalFromVector(const cuDoubleComplex* d_start_vec,
                                         bool full_reorth,
                                         int reorth_freq,
                                         std::vector<double>& alpha,
                                         std::vector<double>& beta);
    
    /**
     * @brief Compute spectral function from Ritz values and weights
     */
    void computeSpectralFunction(const std::vector<double>& ritz_values,
                                const std::vector<double>& weights,
                                const std::vector<double>& frequencies,
                                double broadening,
                                double temperature,
                                std::vector<double>& spectral_func);
    
    /**
     * @brief Build Lanczos tridiagonal and store basis vectors
     * 
     * Extended version that stores all Lanczos basis vectors for 
     * computing matrix elements with operators.
     * 
     * @param d_start_vec Starting vector on device
     * @param full_reorth Whether to do full reorthogonalization
     * @param reorth_freq Reorthogonalization frequency (0 = none)
     * @param alpha Output: diagonal elements
     * @param beta Output: off-diagonal elements
     * @param d_basis_out Output: pointer to array of basis vectors (allocated by this function)
     * @return Number of iterations performed
     */
    int buildLanczosTridiagonalWithBasis(const cuDoubleComplex* d_start_vec,
                                        bool full_reorth,
                                        int reorth_freq,
                                        std::vector<double>& alpha,
                                        std::vector<double>& beta,
                                        cuDoubleComplex*** d_basis_out);
    
    /**
     * @brief Compute spectral function from complex weights
     * 
     * For cross-correlation, weights can be complex. This computes both
     * real and imaginary parts of the spectral function.
     */
    void computeSpectralFunctionComplex(const std::vector<double>& ritz_values,
                                       const std::vector<std::complex<double>>& complex_weights,
                                       const std::vector<double>& frequencies,
                                       double broadening,
                                       double temperature,
                                       std::vector<double>& spectral_func_real,
                                       std::vector<double>& spectral_func_imag);
};

/**
 * @brief GPU kernels for FTLM operations
 */
namespace GPUFTLMKernels {

/**
 * @brief Initialize random vector on GPU
 */
__global__ void initRandomVectorKernel(cuDoubleComplex* vec, int N, 
                                      unsigned long long seed);

/**
 * @brief Normalize vector kernel
 */
__global__ void normalizeKernel(cuDoubleComplex* vec, int N, double norm);

/**
 * @brief Vector AXPY: y = alpha*x + y
 */
__global__ void axpyKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                          cuDoubleComplex alpha, int N);

/**
 * @brief Vector scaling: x = alpha*x
 */
__global__ void scaleKernel(cuDoubleComplex* x, double alpha, int N);

} // namespace GPUFTLMKernels

#endif // WITH_CUDA

#endif // GPU_FTLM_CUH

#ifndef GPU_TPQ_CUH
#define GPU_TPQ_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>
#include <complex>
#include <string>
#include <functional>

// Forward declarations
class GPUOperator;

/**
 * @brief GPU-accelerated Thermal Pure Quantum states implementation
 * 
 * Implements both microcanonical and canonical TPQ methods on GPU
 * for efficient finite temperature calculations on large systems.
 */
class GPUTPQSolver {
public:
    /**
     * @brief Constructor
     * @param gpu_op Pointer to GPU operator (Hamiltonian)
     * @param N Hilbert space dimension
     */
    GPUTPQSolver(GPUOperator* gpu_op, int N);
    
    /**
     * @brief Destructor - frees GPU resources
     */
    ~GPUTPQSolver();
    
    /**
     * @brief Run microcanonical TPQ
     * @param max_iter Maximum number of iterations
     * @param num_samples Number of random initial states
     * @param temp_interval Interval for measurements
     * @param eigenvalues Output energies
     * @param dir Output directory
     * @param large_value Large value for high energy cutoff
     * @param fixed_sz_op Optional GPUFixedSzOperator for embedding to full space
     */
    void runMicrocanonicalTPQ(
        int max_iter,
        int num_samples,
        int temp_interval,
        std::vector<double>& eigenvalues,
        const std::string& dir = "",
        double large_value = 1e5,
        class GPUFixedSzOperator* fixed_sz_op = nullptr
    );
    
    /**
     * @brief Run canonical TPQ with imaginary time evolution
     * @param beta_max Maximum inverse temperature
     * @param num_samples Number of random initial states
     * @param temp_interval Measurement interval
     * @param energies Output energies
     * @param dir Output directory
     * @param delta_beta Time step for imaginary time evolution
     * @param taylor_order Order of Taylor expansion
     * @param fixed_sz_op Optional GPUFixedSzOperator for embedding to full space
     */
    void runCanonicalTPQ(
        double beta_max,
        int num_samples,
        int temp_interval,
        std::vector<double>& energies,
        const std::string& dir = "",
        double delta_beta = 0.1,
        int taylor_order = 50,
        class GPUFixedSzOperator* fixed_sz_op = nullptr
    );
    
    /**
     * @brief Get statistics about GPU TPQ execution
     */
    struct TPQStats {
        double total_time;
        double matvec_time;
        double normalize_time;
        int iterations;
        double throughput;
    };
    
    TPQStats getStats() const { return stats_; }

private:
    GPUOperator* gpu_op_;
    int N_;
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_gen_;
    
    // Device vectors
    cuDoubleComplex* d_state_;
    cuDoubleComplex* d_temp_;
    cuDoubleComplex* d_h_state_;
    double* d_real_scratch_;
    
    // Statistics
    TPQStats stats_;
    
    // Helper functions
    void allocateMemory();
    void freeMemory();
    void generateRandomState(unsigned int seed);
    void normalizeState();
    double computeNorm();
    std::pair<double, double> computeEnergyAndVariance();
    void imaginaryTimeEvolve(double delta_beta, int taylor_order);
    void writeTPQData(const std::string& filename, double inv_temp, 
                     double energy, double variance, double norm, int step);
    bool saveTPQState(const std::string& filename);
    bool saveTPQState(const std::string& filename, class GPUFixedSzOperator* fixed_sz_op);
};

#endif // GPU_TPQ_CUH

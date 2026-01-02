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
#include "ed/core/hdf5_io.h"

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
     * @param continue_quenching If true, continue from saved state
     * @param continue_sample Sample index to continue from (0 = auto-detect)
     * @param continue_beta Beta value to continue from (0.0 = auto-detect)
     * @param save_thermal_states If true, save binary state files at target temperatures
     */
    void runMicrocanonicalTPQ(
        int max_iter,
        int num_samples,
        int temp_interval,
        std::vector<double>& eigenvalues,
        const std::string& dir = "",
        double large_value = 1e5,
        class GPUFixedSzOperator* fixed_sz_op = nullptr,
        bool continue_quenching = false,
        int continue_sample = 0,
        double continue_beta = 0.0,
        bool save_thermal_states = false
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
    
    // CUDA streams for pipelining
    cudaStream_t compute_stream_;    // Main computation stream
    cudaStream_t transfer_stream_;   // Data transfer stream
    bool streams_initialized_;
    
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
    void writeTPQDataHDF5(const std::string& h5_file, size_t sample,
                          double inv_temp, double energy, double variance, 
                          double doublon, uint64_t step);
    bool saveTPQStateHDF5(const std::string& dir, size_t sample, double beta, class GPUFixedSzOperator* fixed_sz_op);
    bool loadTPQStateFromHDF5(const std::string& h5_file, 
                               const std::string& dataset_name,
                               class GPUFixedSzOperator* fixed_sz_op);
    
    /**
     * @brief Find the TPQ state with highest beta (lowest energy) in HDF5
     * 
     * Searches the HDF5 file for all stored TPQ states and returns info about
     * the state with the highest beta value.
     * 
     * @param dir Output directory containing ed_results.h5
     * @param sample_filter Sample index to filter (or -1 for all samples)
     * @return TPQStateInfo struct with sample, beta, and dataset name
     */
    HDF5IO::TPQStateInfo findLowestEnergyTPQStateHDF5(const std::string& dir, int sample_filter);
    
    /**
     * @brief Legacy function for backward compatibility - now uses HDF5
     * 
     * @param dir Output directory to search
     * @param sample Sample index (or 0 for auto-detect)
     * @param beta_out Output parameter for the beta value found
     * @param step_out Output parameter for the step number found
     * @return Path to HDF5 file, or empty string if none found
     */
    std::string findLowestEnergyTPQState(const std::string& dir, int sample, 
                                        double& beta_out, int& step_out);
};

#endif // GPU_TPQ_CUH

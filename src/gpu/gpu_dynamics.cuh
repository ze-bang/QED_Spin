#ifndef GPU_DYNAMICS_CUH
#define GPU_DYNAMICS_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <vector>
#include <complex>
#include <string>
#include <functional>

// Forward declaration
class GPUOperator;

/**
 * @brief GPU-accelerated dynamical correlation computations
 * 
 * Computes time-dependent correlation functions on GPU for spectral function calculations.
 * Operator applications are done on CPU for simplicity and compatibility.
 */
class GPUDynamicsSolver {
public:
    /**
     * @brief Constructor
     * @param gpu_op Pointer to GPU Hamiltonian operator
     * @param N Hilbert space dimension
     */
    GPUDynamicsSolver(GPUOperator* gpu_op, int N);
    
    /**
     * @brief Destructor - frees GPU resources
     */
    ~GPUDynamicsSolver();
    
    /**
     * @brief Compute dynamical correlations using Krylov time evolution
     * 
     * Computes <O_2(t) O_1(0)> = <ψ(t)|O_2|ψ_O1(t)> for multiple operators
     * 
     * @param initial_state Initial state vector (on host)
     * @param operators_1 First set of operators (as CPU functions)
     * @param operators_2 Second set of operators (as CPU functions)
     * @param operator_names Names for output files
     * @param dir Output directory
     * @param sample Sample index for filename
     * @param inv_temp Inverse temperature (β) for labeling
     * @param t_end Maximum time
     * @param dt Time step
     * @param krylov_dim Krylov subspace dimension
     */
    void computeKrylovCorrelations(
        const std::vector<std::complex<double>>& initial_state,
        const std::vector<std::function<void(const std::complex<double>*, std::complex<double>*, int)>>& operators_1,
        const std::vector<std::function<void(const std::complex<double>*, std::complex<double>*, int)>>& operators_2,
        const std::vector<std::string>& operator_names,
        const std::string& dir,
        int sample,
        double inv_temp,
        double t_end,
        double dt,
        int krylov_dim
    );
    
    /**
     * @brief Compute dynamical correlations using Taylor time evolution
     * 
     * Uses pre-computed time evolution operator U(dt)
     * 
     * @param U_t_cpu Time evolution operator (CPU function)
     * @param initial_state Initial state vector (on host)
     * @param operators_1 First set of operators (as CPU functions)
     * @param operators_2 Second set of operators (as CPU functions)
     * @param operator_names Names for output files
     * @param dir Output directory
     * @param sample Sample index
     * @param inv_temp Inverse temperature
     * @param t_end Maximum time
     * @param dt Time step
     */
    void computeTaylorCorrelations(
        std::function<void(const std::complex<double>*, std::complex<double>*, int)> U_t_cpu,
        const std::vector<std::complex<double>>& initial_state,
        const std::vector<std::function<void(const std::complex<double>*, std::complex<double>*, int)>>& operators_1,
        const std::vector<std::function<void(const std::complex<double>*, std::complex<double>*, int)>>& operators_2,
        const std::vector<std::string>& operator_names,
        const std::string& dir,
        int sample,
        double inv_temp,
        double t_end,
        double dt
    );
    
    struct DynamicsStats {
        double total_time;
        double evolution_time;
        double operator_time;
        int num_steps;
    };
    
    DynamicsStats getStats() const { return stats_; }

private:
    GPUOperator* gpu_op_;
    int N_;
    cublasHandle_t cublas_handle_;
    
    // Device vectors for computation
    cuDoubleComplex* d_state_;
    cuDoubleComplex* d_evolved_state_;
    cuDoubleComplex* d_O1_state_;
    cuDoubleComplex* d_evolved_O1_state_;
    cuDoubleComplex* d_O2_temp_;
    cuDoubleComplex* d_temp_;
    
    // Krylov workspace
    cuDoubleComplex* d_krylov_basis_;
    cuDoubleComplex* d_krylov_temp_;
    double* d_krylov_H_;  // Tridiagonal Hamiltonian
    
    DynamicsStats stats_;
    
    // Helper functions
    void allocateMemory(int max_krylov_dim);
    void freeMemory();
    void copyStateToGPU(const std::vector<std::complex<double>>& state);
    void copyStateFromGPU(const cuDoubleComplex* d_state, std::vector<std::complex<double>>& h_state);
    void copyStateToGPU(const std::vector<std::complex<double>>& h_state, cuDoubleComplex* d_state);
    std::complex<double> dotProductGPU(const cuDoubleComplex* v1, const cuDoubleComplex* v2);
    void krylovTimeStep(cuDoubleComplex* state, double dt, int krylov_dim);
};

#endif // GPU_DYNAMICS_CUH

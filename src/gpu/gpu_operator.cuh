#ifndef GPU_OPERATOR_CUH
#define GPU_OPERATOR_CUH

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <vector>
#include <complex>
#include <functional>
#include <memory>
#include "kernel_config.h"
#include "bit_operations.cuh"

// Forward declare only - don't include construct_ham.h to avoid CUDA compilation issues
// The CPU Operator class uses C++ features incompatible with NVCC

// Define complex type
using Complex = std::complex<double>;

/**
 * GPU-accelerated Operator class for large-scale exact diagonalization
 * Supports up to 32 sites (4.3 billion basis states)
 * Uses chunked processing and on-the-fly matrix element computation
 */
class GPUOperator {
public:
    // Constructor
    GPUOperator(int n_sites, float spin_l = 0.5f);
    
    // Destructor
    ~GPUOperator();
    
    // Add transform (operator term)
    void addTransform(const std::function<std::pair<int, Complex>(int)>& transform);
    
    // Matrix-vector product: y = H * x (core operation for Lanczos)
    void matVec(const std::complex<double>* x, std::complex<double>* y, int N);
    
    // GPU-accelerated matrix-vector product
    void matVecGPU(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N);
    
    // Set interaction parameters
    void setInteraction(int site1, int site2, char op1, char op2, double coupling);
    
    // Set single-site operator
    void setSingleSite(int site, char op, double coupling);
    
    // Build sparse matrix representation (if memory allows)
    bool buildSparseMatrix(int N);
    
    // Load pre-built CSR matrix (for symmetrized blocks)
    bool loadCSRMatrix(int N, const int* row_ptr, const int* col_ind, 
                      const std::complex<double>* values, size_t nnz);
    
    // Get dimension
    int getDimension() const { return dimension_; }
    
    // Memory management
    size_t estimateMemoryRequirement(int N) const;
    bool allocateGPUMemory(int N);
    void freeGPUMemory();
    
    // Performance monitoring
    struct PerformanceStats {
        double matVecTime;
        double memoryUsed;
        int numChunks;
        double throughput;  // GFLOPS
    };
    
    PerformanceStats getStats() const { return stats_; }
    
protected:
    int n_sites_;
    float spin_l_;
    int dimension_;
    
    // Interaction storage
    struct Interaction {
        int site1, site2;
        char op1, op2;
        double coupling;
    };
    
    struct SingleSiteOp {
        int site;
        char op;
        double coupling;
    };
    
    std::vector<Interaction> interactions_;
    std::vector<SingleSiteOp> single_site_ops_;
    
    // GPU memory pointers
    cuDoubleComplex* d_vector_in_;
    cuDoubleComplex* d_vector_out_;
    cuDoubleComplex* d_temp_;
    
    // Sparse matrix storage (CSR format)
    int* d_csr_row_ptr_;
    int* d_csr_col_ind_;
    cuDoubleComplex* d_csr_values_;
    size_t nnz_;
    
    // cuSPARSE and cuBLAS handles
    cusparseHandle_t cusparse_handle_;
    cublasHandle_t cublas_handle_;
    cusparseSpMatDescr_t mat_descriptor_;
    cusparseDnVecDescr_t vec_x_descriptor_;
    cusparseDnVecDescr_t vec_y_descriptor_;
    
    // Device interaction data
    Interaction* d_interactions_;
    SingleSiteOp* d_single_site_ops_;
    int num_interactions_;
    int num_single_site_ops_;
    
    // Memory management
    bool gpu_memory_allocated_;
    bool sparse_matrix_built_;
    size_t available_gpu_memory_;
    
    // Performance stats
    PerformanceStats stats_;
    
    // Chunked processing for large systems
    struct ChunkInfo {
        uint64_t start_state;
        uint64_t end_state;
        int start_idx;
        int size;
    };
    
    std::vector<ChunkInfo> chunks_;
    void setupChunks(int N);
    void processChunk(const ChunkInfo& chunk, const cuDoubleComplex* d_x, 
                     cuDoubleComplex* d_y);
    
    // Helper functions
    void copyInteractionsToDevice();
    void initializeCUSPARSE();
    void initializeCUBLAS();
};

/**
 * GPU-accelerated Fixed Sz Operator class
 * Optimized for fixed magnetization sectors
 */
class GPUFixedSzOperator : public GPUOperator {
public:
    GPUFixedSzOperator(int n_sites, int n_up, float spin_l = 0.5f);
    
    // Override matrix-vector product for fixed Sz basis
    void matVecFixedSz(const cuDoubleComplex* d_x, cuDoubleComplex* d_y);
    
    // Build basis states on GPU
    void buildBasisOnGPU();
    
    // Get basis dimension
    int getFixedSzDimension() const { return fixed_sz_dim_; }
    
private:
    int n_up_;
    int fixed_sz_dim_;
    
    // Basis states stored on GPU
    uint64_t* d_basis_states_;
    
    // Hash table for state lookup
    struct HashEntry {
        uint64_t state;
        int index;
    };
    
    HashEntry* d_hash_table_;
    int hash_table_size_;
    
    // Build hash table on GPU
    void buildHashTableOnGPU();
    
    // Hash function
    __device__ static uint64_t hash_function(uint64_t state, int table_size);
};

// CUDA kernel declarations
namespace GPUKernels {

// Matrix-vector product kernel (on-the-fly computation)
__global__ void matVecKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                             int N, int n_sites,
                             const void* interactions, int num_interactions,
                             const void* single_site_ops, int num_single_site_ops);

// Sparse matrix-vector product (uses cuSPARSE internally)
__global__ void sparseMatVecKernel(const int* row_ptr, const int* col_ind,
                                   const cuDoubleComplex* values,
                                   const cuDoubleComplex* x, cuDoubleComplex* y,
                                   int N);

// Fixed Sz matrix-vector product kernel
__global__ void matVecFixedSzKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                                    const uint64_t* basis_states,
                                    const void* hash_table, int hash_size,
                                    int N, int n_sites,
                                    const void* interactions, int num_interactions,
                                    const void* single_site_ops, int num_single_site_ops);

// Basis generation kernel for fixed Sz
__global__ void generateFixedSzBasisKernel(uint64_t* basis_states, int n_bits, int n_up,
                                          uint64_t start_state, int num_states);

// Hash table construction kernel
__global__ void buildHashTableKernel(const uint64_t* basis_states, void* hash_table,
                                    int hash_size, int num_states);

// State lookup kernel
__device__ int lookupState(uint64_t state, const void* hash_table, int hash_size);

} // namespace GPUKernels

#endif // WITH_CUDA

#endif // GPU_OPERATOR_CUH

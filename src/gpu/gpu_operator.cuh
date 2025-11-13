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
 * Optimized transform data structure (Structure-of-Arrays)
 * Matches CPU implementation for consistency
 */
struct GPUTransformData {
    uint8_t op_type;        // 0=S+, 1=S-, 2=Sz
    uint32_t site_index;    // Which site to act on
    cuDoubleComplex coefficient;  // Coupling constant
    uint32_t site_index_2;  // Second site for two-body operators
    uint8_t op_type_2;      // Second operator type for two-body
    uint8_t is_two_body;    // Flag for two-body vs one-body
    uint8_t _padding[2];    // Align to 32 bytes
    
    __host__ __device__ GPUTransformData() 
        : op_type(0), site_index(0), site_index_2(0), 
          op_type_2(0), is_two_body(0) {
        coefficient = make_cuDoubleComplex(0.0, 0.0);
        _padding[0] = _padding[1] = 0;
    }
};

/**
 * GPU-accelerated Operator class for large-scale exact diagonalization
 * Supports up to 32 sites (4.3 billion basis states)
 * Uses chunked processing and on-the-fly matrix element computation
 * 
 * OPTIMIZED: Uses Structure-of-Arrays to eliminate std::function overhead
 */
class GPUOperator {
public:
    // Constructor
    GPUOperator(int n_sites, float spin_l = 0.5f);
    
    // Destructor
    ~GPUOperator();
    
    // OPTIMIZED: Direct data population (no std::function overhead)
    void addOneBodyTerm(uint8_t op_type, uint32_t site, const std::complex<double>& coeff);
    void addTwoBodyTerm(uint8_t op1, uint32_t site1, uint8_t op2, uint32_t site2, 
                       const std::complex<double>& coeff);
    
    // Matrix-vector product: y = H * x (core operation for Lanczos)
    virtual void matVec(const std::complex<double>* x, std::complex<double>* y, int N);
    
    // GPU-accelerated matrix-vector product
    virtual void matVecGPU(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N);
    
    // Set interaction parameters
    void setInteraction(int site1, int site2, char op1, char op2, double coupling);
    
    // Set single-site operator
    void setSingleSite(int site, char op, double coupling);
    
    // Load CSR arrays (host) into GPUOperator and construct cuSPARSE descriptors
    bool loadCSR(int N, const std::vector<int>& row_ptr,
                 const std::vector<int>& col_ind,
                 const std::vector<std::complex<double>>& values);
    
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
    
    // OPTIMIZED: Structure-of-Arrays storage
    std::vector<GPUTransformData> transform_data_;  // Host storage
    GPUTransformData* d_transform_data_;            // Device storage
    int num_transforms_;
    
    // Legacy interaction storage (deprecated, kept for compatibility)
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
    
    // Texture object for optimized random access to input vector
    cudaTextureObject_t tex_input_vector_;
    
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
    void copyTransformDataToDevice();
    void createTextureObject(cuDoubleComplex* d_data, int size);
    void destroyTextureObject();
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
    ~GPUFixedSzOperator();
    
    // Override matrix-vector product for fixed Sz basis
    void matVecFixedSz(const cuDoubleComplex* d_x, cuDoubleComplex* d_y);
    
    // Override base class methods to use fixed Sz
    void matVecGPU(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N) override;
    void matVec(const std::complex<double>* x, std::complex<double>* y, int N) override;
    
    // Build basis states on GPU
    void buildBasisOnGPU();
    
    // Get basis dimension
    int getFixedSzDimension() const { return fixed_sz_dim_; }
    
private:
    int n_up_;
    int fixed_sz_dim_;
    
    // Basis states stored on GPU
    uint64_t* d_basis_states_;
};

// CUDA kernel declarations
namespace GPUKernels {

// GPU-NATIVE: Transform-parallel kernel (maximum GPU utilization)
// Launches N×T threads where each computes one (state,transform) contribution
__global__ void matVecTransformParallel(const cuDoubleComplex* x, cuDoubleComplex* y,
                                        const GPUTransformData* transforms,
                                        int num_transforms, int N, int n_sites, float spin_l);

// OPTIMIZED: State-parallel kernel with shared memory
__global__ void matVecKernelOptimized(cudaTextureObject_t tex_x_unused, cuDoubleComplex* y,
                                      int N, int n_sites, float spin_l,
                                      const GPUTransformData* transforms, int num_transforms,
                                      const cuDoubleComplex* x);

// Legacy: Matrix-vector product kernel (on-the-fly computation)
__global__ void matVecKernel(const cuDoubleComplex* x, cuDoubleComplex* y,
                             int N, int n_sites,
                             const void* interactions, int num_interactions,
                             const void* single_site_ops, int num_single_site_ops);

// Sparse matrix-vector product (uses cuSPARSE internally)
__global__ void sparseMatVecKernel(const int* row_ptr, const int* col_ind,
                                   const cuDoubleComplex* values,
                                   const cuDoubleComplex* x, cuDoubleComplex* y,
                                   int N);

// OPTIMIZED: Fixed-Sz matrix-vector product using Structure-of-Arrays
// GPU-NATIVE: Transform-parallel Fixed-Sz kernel
__global__ void matVecFixedSzTransformParallel(const cuDoubleComplex* x, cuDoubleComplex* y,
                                               const uint64_t* basis_states,
                                               const GPUTransformData* transforms,
                                               int num_transforms, int N, int n_sites, float spin_l);

__global__ void matVecFixedSzKernelOptimized(const cuDoubleComplex* x, cuDoubleComplex* y,
                                             const uint64_t* basis_states,
                                             int N, int n_sites, float spin_l,
                                             const GPUTransformData* transforms, int num_transforms);

// Legacy: Fixed Sz matrix-vector product kernel
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

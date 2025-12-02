#ifndef GPU_KERNEL_CONFIG_H
#define GPU_KERNEL_CONFIG_H

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusparse.h>
#include <cublas_v2.h>
#endif

// GPU Configuration Constants
namespace GPUConfig {
    // Thread block dimensions
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int MAX_BLOCKS = 65535;
    
    // Memory management
    constexpr size_t MAX_GPU_MEMORY_GB = 32;  // Adjust based on your GPU
    constexpr size_t CHUNK_SIZE = 1ULL << 28;  // 256M states per chunk
    
    // Sparse matrix construction
    constexpr int NNZ_PER_STATE_ESTIMATE = 32;  // Average non-zeros per row
    constexpr int HASH_TABLE_SIZE = 1ULL << 20;  // 1M entries
    
    // Multi-GPU settings
    constexpr int MAX_GPUS = 8;
    
    // Bit manipulation constants
    constexpr int MAX_SITES = 32;
    constexpr uint64_t MAX_BASIS_SIZE = 1ULL << 32;  // 4.3 billion states
    
    // Performance tuning
    constexpr int PREFETCH_DISTANCE = 4;
    constexpr int VECTOR_WIDTH = 4;  // For vectorized operations
    constexpr bool USE_UNIFIED_MEMORY = false;  // Set true if GPU supports it
    constexpr bool USE_PEER_ACCESS = true;  // For multi-GPU
}

// Error checking macro
#ifdef WITH_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t err = call; \
        if (err != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#else
#define CUDA_CHECK(call) call
#define CUSPARSE_CHECK(call) call
#define CUBLAS_CHECK(call) call
#endif

#endif // GPU_KERNEL_CONFIG_H

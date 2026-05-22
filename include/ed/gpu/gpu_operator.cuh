#pragma once

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <vector>
#include <complex>
#include <functional>
#include <memory>
#include <ed/gpu/kernel_config.h>
#include <ed/gpu/bit_operations.cuh>
#include <ed/matvec/matvec.h>

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

// ============================================================================
// Branch-free separated transform storage (v2 optimization)
// Matches CPU implementation - eliminates warp divergence in hot loops
// ============================================================================

/** One-body diagonal (Sz only) - no bit flips, just multiply */
struct GPUDiagonalOneBody {
    uint32_t site_index;
    cuDoubleComplex coefficient;
    
    __host__ __device__ GPUDiagonalOneBody() : site_index(0) {
        coefficient = make_cuDoubleComplex(0.0, 0.0);
    }
};

/** One-body off-diagonal (S+ or S-) - flips one bit */
struct GPUOffDiagonalOneBody {
    uint32_t site_index;
    uint8_t op_type;  // 0=S+, 1=S-
    cuDoubleComplex coefficient;
    
    __host__ __device__ GPUOffDiagonalOneBody() : site_index(0), op_type(0) {
        coefficient = make_cuDoubleComplex(0.0, 0.0);
    }
};

/** Two-body purely diagonal (Sz_i Sz_j) - no bit flips */
struct GPUDiagonalTwoBody {
    uint32_t site_index_1;
    uint32_t site_index_2;
    cuDoubleComplex coefficient;
    
    __host__ __device__ GPUDiagonalTwoBody() : site_index_1(0), site_index_2(0) {
        coefficient = make_cuDoubleComplex(0.0, 0.0);
    }
};

/** Two-body mixed (one Sz, one S+/S-) - flips one bit */
struct GPUMixedTwoBody {
    uint32_t sz_site;        // Site with Sz operator
    uint32_t flip_site;      // Site with S+/S- operator
    uint8_t flip_op_type;    // 0=S+, 1=S-
    cuDoubleComplex coefficient;
    
    __host__ __device__ GPUMixedTwoBody() : sz_site(0), flip_site(0), flip_op_type(0) {
        coefficient = make_cuDoubleComplex(0.0, 0.0);
    }
};

/** Two-body off-diagonal (S+_i S-_j or S-_i S+_j) - flips two bits */
struct GPUOffDiagonalTwoBody {
    uint32_t site_index_1;
    uint32_t site_index_2;
    uint8_t op_type_1;  // 0=S+, 1=S-
    uint8_t op_type_2;  // 0=S+, 1=S-
    cuDoubleComplex coefficient;
    
    __host__ __device__ GPUOffDiagonalTwoBody() 
        : site_index_1(0), site_index_2(0), op_type_1(0), op_type_2(0) {
        coefficient = make_cuDoubleComplex(0.0, 0.0);
    }
};

/**
 * Three-body transform data structure
 * For interactions like S^α_i S^β_j S^γ_k
 */
struct GPUThreeBodyTransformData {
    uint8_t op_type_1;      // First operator type
    uint32_t site_index_1;  // First site
    uint8_t op_type_2;      // Second operator type
    uint32_t site_index_2;  // Second site
    uint8_t op_type_3;      // Third operator type
    uint32_t site_index_3;  // Third site
    cuDoubleComplex coefficient;  // Coupling constant
    uint8_t _padding[6];    // Alignment padding
    
    __host__ __device__ GPUThreeBodyTransformData()
        : op_type_1(0), site_index_1(0), op_type_2(0),
          site_index_2(0), op_type_3(0), site_index_3(0) {
        coefficient = make_cuDoubleComplex(0.0, 0.0);
        for (int i = 0; i < 6; ++i) _padding[i] = 0;
    }
};

/**
 * GPU-accelerated Operator class for large-scale exact diagonalization
 * Supports up to 32 sites (4.3 billion basis states)
 * Uses chunked processing and on-the-fly matrix element computation
 * 
 * OPTIMIZED: Uses Structure-of-Arrays to eliminate std::function overhead
 */
class GPUOperator : public ed::matvec::MatVecOperator {
public:
    // Constructor
    GPUOperator(int n_sites, float spin_l = 0.5f);
    
    // Destructor - virtual for correct polymorphic deletion
    ~GPUOperator() override;

    // -------------------------------------------------------------------
    // MatVecOperator interface (Phase 2 of matvec-unification revamp).
    // GPUOperator advertises CudaDevice memory space; `in` and `out`
    // are treated as device pointers (cuDoubleComplex and
    // std::complex<double> are layout-compatible). matVecGPU is
    // already virtual, so this dispatches through to GPUFixedSz /
    // GPUSymmetrized overrides without further intervention.
    // -------------------------------------------------------------------
    void apply(const std::complex<double>* in, std::complex<double>* out,
               std::size_t size) const override {
        const_cast<GPUOperator*>(this)->matVecGPU(
            reinterpret_cast<const cuDoubleComplex*>(in),
            reinterpret_cast<cuDoubleComplex*>(out),
            static_cast<int>(size));
    }
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(dimension_);
    }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return ed::matvec::MemorySpace::CudaDevice;
    }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "GPUOperator(n_sites=" + std::to_string(n_sites_)
            + ", dim=" + std::to_string(dimension_) + ")";
    }
    
    // OPTIMIZED: Direct data population (no std::function overhead)
    void addOneBodyTerm(uint8_t op_type, uint32_t site, const std::complex<double>& coeff);
    void addTwoBodyTerm(uint8_t op1, uint32_t site1, uint8_t op2, uint32_t site2, 
                       const std::complex<double>& coeff);
    void addThreeBodyTerm(uint8_t op1, uint32_t site1, uint8_t op2, uint32_t site2,
                         uint8_t op3, uint32_t site3, const std::complex<double>& coeff);
    
    // Load three-body terms from file
    void loadThreeBodyFile(const std::string& filename);
    
    // Copy three-body data to device
    void copyThreeBodyDataToDevice();
    
    // Matrix-vector product: y = H * x (core operation for Lanczos)
    virtual void matVec(const std::complex<double>* x, std::complex<double>* y, int N);
    
    // GPU-accelerated matrix-vector product
    virtual void matVecGPU(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N);
    
    // GPU-accelerated matrix-vector product with stream (for parallel block operations)
    virtual void matVecGPUAsync(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N, cudaStream_t stream);
    
    // Check if operator supports async matVec
    virtual bool supportsAsyncMatVec() const { return true; }
    
    // Legacy char-based interface (wraps addOneBodyTerm/addTwoBodyTerm)
    void setInteraction(int site1, int site2, char op1, char op2, double coupling);
    void setSingleSite(int site, char op, double coupling);
    
    // Get dimension
    int64_t getDimension() const { return dimension_; }
    
    // Memory management
    size_t estimateMemoryRequirement(int N) const;
    bool allocateGPUMemory(int N);
    void freeGPUMemory();
    
    // Copy transform data to device (public for operator conversion)
    void copyTransformDataToDevice();
    
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
    int64_t dimension_;  // int64_t to support > 2^31 states; GPU kernels still use int N per call
    
    // OPTIMIZED: Structure-of-Arrays storage
    std::vector<GPUTransformData> transform_data_;  // Host storage
    GPUTransformData* d_transform_data_;            // Device storage
    int num_transforms_;
    
    // Three-body term storage
    std::vector<GPUThreeBodyTransformData> three_body_data_;  // Host storage
    GPUThreeBodyTransformData* d_three_body_data_;            // Device storage
    int num_three_body_;
    
    // ========================================================================
    // Branch-free separated storage (v2 optimization)
    // Eliminates warp divergence in kernels - each kernel processes uniform ops
    // ========================================================================
    std::vector<GPUDiagonalOneBody> diag_one_body_;      // Sz terms
    std::vector<GPUOffDiagonalOneBody> offdiag_one_body_; // S+/S- terms
    std::vector<GPUDiagonalTwoBody> diag_two_body_;      // Sz_i Sz_j terms
    std::vector<GPUMixedTwoBody> mixed_two_body_;        // Sz * S+/S- terms
    std::vector<GPUOffDiagonalTwoBody> offdiag_two_body_; // S+/S- * S+/S- terms
    
    // Device pointers for separated storage
    GPUDiagonalOneBody* d_diag_one_body_ = nullptr;
    GPUOffDiagonalOneBody* d_offdiag_one_body_ = nullptr;
    GPUDiagonalTwoBody* d_diag_two_body_ = nullptr;
    GPUMixedTwoBody* d_mixed_two_body_ = nullptr;
    GPUOffDiagonalTwoBody* d_offdiag_two_body_ = nullptr;
    
    // Counts for separated transforms
    size_t num_diag_one_body_ = 0;
    size_t num_offdiag_one_body_ = 0;
    size_t num_diag_two_body_ = 0;
    size_t num_mixed_two_body_ = 0;
    size_t num_offdiag_two_body_ = 0;
    
    bool transforms_separated_ = false;
    bool separated_on_device_ = false;
    
    // ========================================================================
    // Kernel pathway selection (cached to avoid branch overhead per matVec)
    // ========================================================================
    enum class KernelPathway {
        UNINITIALIZED = 0,     // Not yet selected
        WARP_REDUCTION,        // Gather pattern, no atomics (T >= 1024, N >= 8192)
        BRANCH_FREE_SCATTER,   // Separated kernels with atomics (T >= 64)
        SHARED_MEMORY,         // State-parallel with shared mem (T < 64)
        CUSPARSE_CSR           // Assembled CSR with cuSPARSE SpMV (fastest when matrix fits)
    };
    
    KernelPathway selected_pathway_ = KernelPathway::UNINITIALIZED;
    int cached_N_ = 0;  // Dimension for which pathway was selected
    
    // Launch configuration (cached to avoid recomputation)
    struct LaunchConfig {
        int num_blocks = 0;
        int threads_per_block = 0;
        size_t shared_mem_size = 0;
        dim3 grid_2d = dim3(0,0,0);
        dim3 block_2d = dim3(0,0,0);
    };
    LaunchConfig launch_config_;
    
    // Select optimal kernel pathway based on problem characteristics
    void selectKernelPathway(int N);
    
    // Separate transforms by type (call before kernel launch)
    void separateTransformsByType();
    void copySeparatedTransformsToDevice();

    // Drop ALL caches derived from transform_data_ / three_body_data_ when
    // the operator definition mutates. See addOneBodyTerm() for the full
    // contract — call this from any setter that touches the host-side data.
    void invalidateDerivedCaches();

    // ========================================================================
    // cuSPARSE assembled-CSR fast path
    //
    // For sufficiently small N where the full sparse Hamiltonian fits in GPU
    // memory, we assemble it once into CSR format and reuse cuSPARSE's
    // hand-tuned SpMV kernels. This avoids the per-matvec transform-loop
    // overhead and the atomic contention in the matrix-free pathways.
    //
    // Build is lazy: triggered on first matVecGPU call when the pathway is
    // selected. Returns true if the matrix was successfully assembled (i.e.
    // there is enough device memory). If false, caller should fall back to
    // a matrix-free pathway.
    //
    // Cache invariants: invalidated whenever the operator definition mutates
    // (transform_data_ changes). The host-side flag `csr_assembled_` is the
    // single source of truth for "device CSR is up-to-date".
    // ========================================================================
    bool buildCsrOnDevice(int N);
    void freeCsrDeviceData();
    void applyCusparse(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N,
                       cudaStream_t stream = 0);

    // ------------------------------------------------------------------------
    // Mixed-precision FP32 CSR cache (Phase 3a #3).
    // See include/ed/gpu/gpu_mixed_precision.h for the design rationale and
    // the env knob that gates the FP32 path.
    //
    // Built lazily by buildCsrFp32OnDevice() the first time applyCusparse()
    // is asked to use the mixed-precision path. Aliases the integer index
    // arrays of the FP64 CSR (d_csr_row_offsets_ / d_csr_col_idx_); only
    // d_csr_values_fp32_ is a fresh FP32 copy of d_csr_values_.
    //
    // Workspace vectors d_x_fp32_workspace_ / d_y_fp32_workspace_ are
    // allocated to length csr_dim_fp32_ at build time and reused on every
    // mixed-precision SpMV call.
    // ------------------------------------------------------------------------
    bool buildCsrFp32OnDevice(int N);
    void freeCsrFp32DeviceData();
    void applyCusparseMixed(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                            int N, cudaStream_t stream = 0);

    cuFloatComplex*       d_csr_values_fp32_       = nullptr;
    cuFloatComplex*       d_x_fp32_workspace_      = nullptr;
    cuFloatComplex*       d_y_fp32_workspace_      = nullptr;
    cusparseSpMatDescr_t  csr_descr_fp32_          = nullptr;
    cusparseDnVecDescr_t  vec_x_descr_fp32_        = nullptr;
    cusparseDnVecDescr_t  vec_y_descr_fp32_        = nullptr;
    void*                 cusparse_workspace_fp32_ = nullptr;
    size_t                cusparse_workspace_bytes_fp32_ = 0;
    int                   csr_dim_fp32_            = 0;
    bool                  fp32_csr_assembled_      = false;

    // CSR storage on device (column-flattened: row_offsets[N+1], col_idx[nnz],
    // values[nnz]). Owned by this object.
    int*               d_csr_row_offsets_ = nullptr;
    int*               d_csr_col_idx_     = nullptr;
    cuDoubleComplex*   d_csr_values_      = nullptr;
    int64_t            csr_nnz_           = 0;
    int                csr_dim_           = 0;        // N for which CSR was built
    bool               csr_assembled_     = false;

    // cuSPARSE descriptors and workspace buffer. cuSPARSE 11+ uses opaque
    // sparse/dense descriptors plus a per-call workspace whose size is
    // queried via cusparseSpMV_bufferSize(); we cache it across calls.
    cusparseHandle_t   cusparse_handle_   = nullptr;
    cusparseSpMatDescr_t csr_descr_       = nullptr;
    cusparseDnVecDescr_t vec_x_descr_     = nullptr;
    cusparseDnVecDescr_t vec_y_descr_     = nullptr;
    void*              cusparse_workspace_ = nullptr;
    size_t             cusparse_workspace_bytes_ = 0;
    
    // GPU memory pointers
    cuDoubleComplex* d_vector_in_;
    cuDoubleComplex* d_vector_out_;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
    
    // Memory management
    bool gpu_memory_allocated_;
    size_t available_gpu_memory_;
    
    // OPTIMIZATION: Pre-allocated CUDA events (avoid create/destroy per matVec)
    cudaEvent_t timing_start_;
    cudaEvent_t timing_stop_;
    bool events_initialized_ = false;
    
    // Performance stats
    PerformanceStats stats_;
    
    // Helper functions
    void initializeCUBLAS();
};

/**
 * Open-addressing hash entry for fixed-Sz state -> basis-index lookup.
 *
 * Stored device-side. Empty slots use key == UINT64_MAX (set via cudaMemset 0xFF).
 * Layout: 16 bytes (key 8 + value 4 + pad 4), naturally aligned for 64-bit loads.
 */
struct GPUStateLookupEntry {
    uint64_t key;     // basis state bitmask (UINT64_MAX = empty)
    int32_t  value;   // index into basis_states[] (-1 = unused)
    int32_t  _pad;    // align to 16 bytes

    __host__ __device__ GPUStateLookupEntry()
        : key(static_cast<uint64_t>(-1)), value(-1), _pad(0) {}
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
    void matVecGPUAsync(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N, cudaStream_t stream) override;

    // MatVecOperator overrides (Phase 2): dim() and description() reflect the
    // projected sector. apply / memory_space / is_hermitian are inherited
    // from GPUOperator and dispatch to GPUFixedSz::matVecGPU virtually.
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(fixed_sz_dim_);
    }
    [[nodiscard]] std::string description() const override {
        return "GPUFixedSzOperator(n_sites=" + std::to_string(n_sites_)
            + ", n_up=" + std::to_string(n_up_)
            + ", dim=" + std::to_string(fixed_sz_dim_) + ")";
    }

    // Fixed-Sz kernels use shared basis table and atomic accumulation,
    // so concurrent multi-stream execution is not safe.
    bool supportsAsyncMatVec() const override { return false; }
    
    // Build basis states on GPU
    void buildBasisOnGPU();

    // Build open-addressing hash table state -> basis_idx on GPU
    // (sized to next power of two >= 2 * fixed_sz_dim_, load factor <= 0.5).
    // No-op if hash already built or env ED_GPU_FIXED_SZ_HASH=0.
    void buildStateHashOnGPU();
    
    // Get basis dimension
    int getFixedSzDimension() const { return fixed_sz_dim_; }
    
    // Get full Hilbert space dimension
    size_t getFullDim() const { return 1ULL << n_sites_; }
    
    // Transform vector from fixed-Sz basis to full Hilbert space
    std::vector<std::complex<double>> embedToFull(const std::vector<std::complex<double>>& fixed_sz_vec);
    
    // Project vector from full Hilbert space to fixed-Sz basis
    std::vector<std::complex<double>> projectToReduced(const std::vector<std::complex<double>>& full_vec);
    
private:
    int n_up_;
    int fixed_sz_dim_;
    
    // Basis states stored on GPU
    uint64_t* d_basis_states_;

    // Open-addressing hash table for state -> basis_idx (O(1) avg lookup)
    // d_state_hash_ == nullptr signals "not built" -> kernels fall back to
    // binary search on basis_states. state_hash_size_ is a power of two.
    GPUStateLookupEntry* d_state_hash_      = nullptr;
    int                  state_hash_size_   = 0;  // power of 2, mask = size-1
    bool                 use_hash_          = true;
};

// CUDA kernel declarations
namespace GPUKernels {

// State-parallel kernel with shared memory (used for small T)
__global__ void matVecKernelOptimized(cudaTextureObject_t tex_x_unused, cuDoubleComplex* y,
                                      int N, int n_sites, float spin_l,
                                      const GPUTransformData* transforms, int num_transforms,
                                      const cuDoubleComplex* x);

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

// Basis generation kernel for fixed Sz
__global__ void generateFixedSzBasisKernel(uint64_t* basis_states, int n_bits, int n_up,
                                          uint64_t start_state, int num_states);

// Upload Pascal triangle into __constant__ d_pascal[][]. Required before
// the first generateFixedSzBasisKernel launch; idempotent across calls.
void ensure_pascal_uploaded();

// State lookup (binary search)
__device__ int lookupState(uint64_t state, const void* basis_states_ptr, int num_states);

// ============================================================================
// Hash-table accelerated fixed-Sz matvec kernels (Phase X optimization)
// Replace per-element O(log N) binary search with O(1) avg open-addressing
// hash lookup. Hash table is built once in GPUFixedSzOperator::buildStateHashOnGPU().
// Hash size is a power of two; modulo replaced by bitmask in the lookup.
// ============================================================================

// One-pass build kernel: each thread inserts one basis state into the hash
// using atomicCAS on the 64-bit key. Empty key sentinel = UINT64_MAX.
__global__ void buildStateHashKernel(GPUStateLookupEntry* table,
                                     int table_size,        // power of 2
                                     uint32_t table_mask,   // = table_size - 1
                                     const uint64_t* basis_states,
                                     int num_states);

// Hash-lookup variant of matVecFixedSzTransformParallel. Same 2D launch grid.
__global__ void matVecFixedSzTransformParallelHash(const cuDoubleComplex* x,
                                                   cuDoubleComplex* y,
                                                   const uint64_t* basis_states,
                                                   const GPUStateLookupEntry* hash_table,
                                                   int hash_table_size,
                                                   uint32_t hash_table_mask,
                                                   const GPUTransformData* transforms,
                                                   int num_transforms,
                                                   int N, int n_sites, float spin_l);

// Hash-lookup variant of matVecFixedSzKernelOptimized.
__global__ void matVecFixedSzKernelOptimizedHash(const cuDoubleComplex* x,
                                                 cuDoubleComplex* y,
                                                 const uint64_t* basis_states,
                                                 const GPUStateLookupEntry* hash_table,
                                                 int hash_table_size,
                                                 uint32_t hash_table_mask,
                                                 int N, int n_sites, float spin_l,
                                                 const GPUTransformData* transforms,
                                                 int num_transforms);

// ============================================================================
// MIXED-PRECISION CAST KERNELS (Phase 3a #3)
// Element-wise FP64 <-> FP32 complex casts used by the mixed-precision
// SpMV path (see include/ed/gpu/gpu_mixed_precision.h).
// ============================================================================
__global__ void castDoubleToFloatComplex(const cuDoubleComplex* in,
                                         cuFloatComplex* out, int N);
__global__ void castFloatToDoubleComplex(const cuFloatComplex* in,
                                         cuDoubleComplex* out, int N);
__global__ void castDoubleToFloatComplexValues(const cuDoubleComplex* in,
                                               cuFloatComplex* out, int64_t nnz);

// ============================================================================
// BRANCH-FREE KERNELS (v2 optimization)
// Each kernel handles one operator type - no warp divergence
// ============================================================================

// Full Hilbert space branch-free kernels
__global__ void matVecDiagonalOneBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                      const GPUDiagonalOneBody* transforms,
                                      int num_transforms, int N, float spin_l);

__global__ void matVecOffDiagonalOneBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                         const GPUOffDiagonalOneBody* transforms,
                                         int num_transforms, int N);

__global__ void matVecDiagonalTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                      const GPUDiagonalTwoBody* transforms,
                                      int num_transforms, int N, float spin_l);

__global__ void matVecMixedTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                   const GPUMixedTwoBody* transforms,
                                   int num_transforms, int N, float spin_l);

__global__ void matVecOffDiagonalTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                         const GPUOffDiagonalTwoBody* transforms,
                                         int num_transforms, int N);

// Fixed-Sz branch-free kernels (with binary search for state lookup)
__global__ void matVecFixedSzDiagonalOneBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                             const uint64_t* basis_states,
                                             const GPUDiagonalOneBody* transforms,
                                             int num_transforms, int N, float spin_l);

__global__ void matVecFixedSzDiagonalTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                             const uint64_t* basis_states,
                                             const GPUDiagonalTwoBody* transforms,
                                             int num_transforms, int N, float spin_l);

__global__ void matVecFixedSzOffDiagonalTwoBody(const cuDoubleComplex* x, cuDoubleComplex* y,
                                                const uint64_t* basis_states,
                                                const GPUOffDiagonalTwoBody* transforms,
                                                int num_transforms, int N);

// ============================================================================
// WARP-REDUCTION (GATHER) KERNEL - Atomic-free output
// Each warp computes one output element by gathering contributions from all inputs
// ============================================================================

__global__ void matVecWarpReductionFused(
    const cuDoubleComplex* __restrict__ x,
    cuDoubleComplex* __restrict__ y,
    const GPUDiagonalOneBody* diag1, int num_diag1,
    const GPUDiagonalTwoBody* diag2, int num_diag2,
    const GPUOffDiagonalOneBody* offdiag1, int num_offdiag1,
    const GPUMixedTwoBody* mixed2, int num_mixed2,
    const GPUOffDiagonalTwoBody* offdiag2, int num_offdiag2,
    int N, float spin_l);

} // namespace GPUKernels

// ============================================================================
// GPU Symmetrized Operator — matrix-free H*v in symmetry-projected sectors
// ============================================================================

/**
 * @brief Open-addressing hash table entry for state → basis index lookup
 *
 * Each computational basis state s that belongs to some symmetrized basis
 * state |φ_k⟩ is stored with:
 *   key   = s  (computational state, EMPTY_KEY = UINT64_MAX means vacant)
 *   value = k  (symmetrized basis index in sector)
 *   projection_factor = conj(β_s) * group_norm / norm_k
 *     where β_s is the orbit coefficient of s in |φ_k⟩
 *
 * Pre-computing the projection factor avoids per-lookup division/conjugation.
 */
struct GPUHashEntry {
    uint64_t key;                    // Computational basis state (UINT64_MAX = empty)
    int32_t  value;                  // Symmetrized basis index
    cuDoubleComplex projection;      // conj(coeff) * group_norm / norm
    
    __host__ __device__ GPUHashEntry()
        : key(UINT64_MAX), value(-1) {
        projection = make_cuDoubleComplex(0.0, 0.0);
    }
};

/**
 * @brief GPU-accelerated operator for symmetry-projected sectors
 *
 * Implements the streaming symmetrized matvec on GPU:
 *   out[k] += Σ_j c_j Σ_{s∈orbit_j} (α_s/norm_j) Σ_t h(s,s') · proj(s',k)
 *
 * Data stored on device:
 *   - orbit_elements[orbit_offsets[j]..orbit_offsets[j+1]] = states in orbit j
 *   - orbit_coefficients[..] = α_s coefficients
 *   - orbit_norms[j] = normalization of |φ_j⟩
 *   - hash_table[] = open-addressing lookup: state → (basis_idx, projection_factor)
 *   - transform_data_/separated storage from GPUOperator base
 *
 * Parallelize over (j, orbit_idx) pairs via 2D grid; scatter to out[k] with atomics.
 */
class GPUSymmetrizedOperator : public GPUOperator {
public:
    GPUSymmetrizedOperator(int n_sites, float spin_l = 0.5f);
    ~GPUSymmetrizedOperator();
    
    /**
     * @brief Set up sector data from CPU-side orbit information
     *
     * @param sector_dim       Number of symmetrized basis states in this sector
     * @param orbit_elements   Flattened orbit elements (all orbits concatenated)
     * @param orbit_coefficients Matching complex coefficients
     * @param orbit_offsets    CSR-style offsets: orbit_offsets[j] = start of orbit j
     * @param orbit_norms      Normalization factor for each basis state
     * @param group_size       |G|, the order of the symmetry group
     */
    void setSectorData(
        int sector_dim,
        const std::vector<uint64_t>& orbit_elements,
        const std::vector<std::complex<double>>& orbit_coefficients,
        const std::vector<int>& orbit_offsets,
        const std::vector<double>& orbit_norms,
        int group_size);
    
    // Override matvec for symmetrized operation
    void matVecGPU(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N) override;
    void matVecGPUAsync(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N, cudaStream_t stream) override;
    void matVec(const std::complex<double>* x, std::complex<double>* y, int N) override;

    // MatVecOperator overrides (Phase 2): symmetrized sector dim + label.
    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(sector_dim_);
    }
    [[nodiscard]] std::string description() const override {
        return "GPUSymmetrizedOperator(n_sites=" + std::to_string(n_sites_)
            + ", sector_dim=" + std::to_string(sector_dim_) + ")";
    }

    // Fixed-Sz kernels use shared basis table and atomic accumulation
    bool supportsAsyncMatVec() const override { return false; }
    
    int getSectorDimension() const { return sector_dim_; }
    
private:
    // Build the GPU open-addressing hash table
    void buildHashTable();
    
    // Copy orbit data to device
    void copyOrbitDataToDevice();
    
    // Sector properties
    int sector_dim_ = 0;
    int group_size_ = 1;
    int total_orbit_elements_ = 0;
    
    // Host-side orbit data
    std::vector<uint64_t> h_orbit_elements_;
    std::vector<std::complex<double>> h_orbit_coefficients_;
    std::vector<int> h_orbit_offsets_;
    std::vector<double> h_orbit_norms_;
    
    // Device-side orbit data (CSR layout)
    uint64_t*        d_orbit_elements_    = nullptr;
    cuDoubleComplex* d_orbit_coefficients_ = nullptr;
    int*             d_orbit_offsets_     = nullptr;
    double*          d_orbit_norms_       = nullptr;
    
    // Device-side hash table for state → (basis_idx, projection_factor)
    GPUHashEntry*    d_hash_table_        = nullptr;
    int              hash_table_size_     = 0;
    
    bool             sector_data_on_device_ = false;
};

// ============================================================================
// Symmetrized matvec kernel declarations
// ============================================================================
namespace GPUKernels {

/**
 * @brief Symmetrized matvec kernel — scatter pattern
 *
 * 2D grid: x-dim = orbit elements across all basis states, y-dim = transforms.
 * Each thread applies one transform to one orbit element, then looks up the
 * result in the hash table and atomically accumulates into the output.
 */
__global__ void matVecSymmetrized(
    const cuDoubleComplex* __restrict__ x,
    cuDoubleComplex* __restrict__ y,
    const uint64_t* orbit_elements,
    const cuDoubleComplex* orbit_coefficients,
    const int* orbit_offsets,
    const double* orbit_norms,
    int sector_dim,
    const GPUTransformData* transforms, int num_transforms,
    const GPUHashEntry* hash_table, int hash_table_size,
    int n_sites, float spin_l,
    int total_orbit_elements);

} // namespace GPUKernels

// ============================================================================
// CPU → GPU Conversion Helper
// ============================================================================

// Forward declaration to avoid circular dependency
class Operator;

/**
 * @brief Convert CPU Operator to GPUOperator
 * 
 * Extracts sparse matrix from CPU operator and loads into GPU memory.
 * 
 * @param cpu_op CPU operator to convert
 * @param gpu_op GPU operator to populate
 * @return true if successful
 */
bool convertOperatorToGPU(const Operator& cpu_op, GPUOperator& gpu_op);

#endif // WITH_CUDA

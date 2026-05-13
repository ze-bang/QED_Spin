#ifdef WITH_CUDA

// Prevent inclusion of CPU Operator class
#define CONSTRUCT_HAM_H

#include <ed/gpu/gpu_operator.cuh>
#include <iostream>
#include <cmath>

using namespace GPUConfig;

// ============================================================================
// GPUFixedSzOperator Implementation
// ============================================================================

GPUFixedSzOperator::GPUFixedSzOperator(int n_sites, int n_up, float spin_l)
    : GPUOperator(n_sites, spin_l), n_up_(n_up),
      d_basis_states_(nullptr) {
    
    // Calculate binomial coefficient C(n_sites, n_up) for dimension
    auto binomial = [](int n, int k) -> int64_t {
        if (k > n - k) k = n - k;
        int64_t result = 1;
        for (int i = 0; i < k; ++i) {
            result *= (n - i);
            result /= (i + 1);
        }
        return result;
    };
    
    fixed_sz_dim_ = binomial(n_sites, n_up);
    dimension_ = fixed_sz_dim_;  // Override full dimension

    // Hash table opt-in (default ON). Set ED_GPU_FIXED_SZ_HASH=0 to fall back
    // to binary-search lookup (used for benchmarking / debugging).
    {
        const char* s = std::getenv("ED_GPU_FIXED_SZ_HASH");
        use_hash_ = !(s && s[0] == '0');
    }

    std::cout << "GPU Fixed Sz Operator initialized (OPTIMIZED)\n";
    std::cout << "  Sites: " << n_sites << ", N_up: " << n_up << "\n";
    std::cout << "  Fixed Sz dimension: " << fixed_sz_dim_ << "\n";
    std::cout << "  Reduction factor: " << (1 << n_sites) / (double)fixed_sz_dim_ << "x\n";
    std::cout << "  State lookup: " << (use_hash_ ? "Hash table (open addressing, O(1) avg)"
                                                  : "Binary search (warp-coherent)") << "\n";

    // Build basis on GPU
    buildBasisOnGPU();

    // Build hash table after basis is on device. Skip if disabled or if
    // we cannot fit the table in remaining device memory.
    if (use_hash_) {
        buildStateHashOnGPU();
    }
}

GPUFixedSzOperator::~GPUFixedSzOperator() {
    if (d_basis_states_) {
        cudaFree(d_basis_states_);
        d_basis_states_ = nullptr;
    }
    if (d_state_hash_) {
        cudaFree(d_state_hash_);
        d_state_hash_ = nullptr;
        state_hash_size_ = 0;
    }
}

void GPUFixedSzOperator::buildBasisOnGPU() {
    std::cout << "Building fixed Sz basis on GPU...\n";

    // Combinadic-unrank kernel needs the Pascal triangle in __constant__
    // memory. ensure_pascal_uploaded() is idempotent across operators.
    GPUKernels::ensure_pascal_uploaded();

    CUDA_CHECK(cudaMalloc(&d_basis_states_, fixed_sz_dim_ * sizeof(uint64_t)));

    // start_state retained in the kernel signature for ABI compatibility,
    // but the unrank implementation roots its enumeration at colex rank 0
    // = (1 << n_up) - 1, which is the same starting state.
    uint64_t start_state = (1ULL << n_up_) - 1;

    int num_blocks = (fixed_sz_dim_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GPUKernels::generateFixedSzBasisKernel<<<num_blocks, BLOCK_SIZE>>>(
        d_basis_states_, n_sites_, n_up_, start_state, fixed_sz_dim_);
    CUDA_CHECK(cudaGetLastError());
    // No cudaDeviceSynchronize here: subsequent matVec calls are issued on
    // the default stream too, so the dependency is implicit. Forcing a
    // host sync at construction time only serialized solver initialization
    // for no benefit. (Errors are caught by the next CUDA_CHECK on any
    // downstream API call.)

    std::cout << "  Basis generation complete (combinadic unrank, naturally sorted)\n";
}

void GPUFixedSzOperator::buildStateHashOnGPU() {
    if (!use_hash_) return;
    if (d_state_hash_ != nullptr) return;  // already built

    // Pick power-of-two table size with load factor <= 0.5.
    // (load factor 0.5 keeps avg probe count ~1.5; max ~log N very rare.)
    int target = static_cast<int>(2 * static_cast<int64_t>(fixed_sz_dim_));
    int p = 1;
    while (p < target) p <<= 1;
    state_hash_size_ = p;
    uint32_t mask = static_cast<uint32_t>(p - 1);

    size_t hash_bytes = static_cast<size_t>(state_hash_size_) * sizeof(GPUStateLookupEntry);

    // Memory guard: skip hash if it would exceed remaining device memory.
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    // Leave a 2 GB buffer for transient allocations (Lanczos vectors etc).
    const size_t kSafetyBuffer = static_cast<size_t>(2) << 30;
    if (hash_bytes + kSafetyBuffer > free_bytes) {
        std::cout << "  [hash] insufficient device memory ("
                  << (hash_bytes >> 20) << " MiB needed, "
                  << (free_bytes  >> 20) << " MiB free) — falling back to binary search\n";
        use_hash_ = false;
        state_hash_size_ = 0;
        return;
    }

    cudaError_t err = cudaMalloc(&d_state_hash_, hash_bytes);
    if (err != cudaSuccess) {
        std::cout << "  [hash] cudaMalloc failed for "
                  << (hash_bytes >> 20) << " MiB ("
                  << cudaGetErrorString(err) << ") — falling back to binary search\n";
        use_hash_ = false;
        state_hash_size_ = 0;
        d_state_hash_ = nullptr;
        return;
    }

    // Initialize all keys to UINT64_MAX (= 0xFF byte fill) and values to -1.
    // Both fields are -1 under 0xFF, which matches GPUStateLookupEntry default.
    CUDA_CHECK(cudaMemset(d_state_hash_, 0xFF, hash_bytes));

    int build_blocks = (fixed_sz_dim_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GPUKernels::buildStateHashKernel<<<build_blocks, BLOCK_SIZE>>>(
        d_state_hash_, state_hash_size_, mask,
        d_basis_states_, fixed_sz_dim_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "  [hash] built table: " << state_hash_size_ << " slots, "
              << (hash_bytes >> 20) << " MiB, load factor "
              << (static_cast<double>(fixed_sz_dim_) / state_hash_size_) << "\n";
}

void GPUFixedSzOperator::matVecFixedSz(const cuDoubleComplex* d_x, cuDoubleComplex* d_y) {
    // Per-call timing forces a host/device sync via cudaEventSynchronize
    // and, in the previous implementation, also a cudaDeviceSynchronize.
    // Both completely serialize the Lanczos pipeline (every iteration
    // waits on the GPU before launching the next BLAS call), which on
    // typical Heisenberg models was costing ~30% of wall time. We keep
    // the timing path as opt-in via ED_GPU_TIMING=1, matching the
    // convention used by GPULanczos::orthogonalize() and the dense GPU
    // matVec in gpu_operator.cu.
    static const bool timing_enabled = []() {
        const char* s = std::getenv("ED_GPU_TIMING");
        return (s && s[0] == '1');
    }();

    cudaEvent_t start = nullptr, stop = nullptr;
    if (timing_enabled) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
    }

    int num_blocks = (fixed_sz_dim_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = std::min(num_blocks, MAX_BLOCKS);

    if (!transform_data_.empty()) {
        if (d_transform_data_ == nullptr) {
            copyTransformDataToDevice();
        }

        const bool hash_ready = (d_state_hash_ != nullptr) && (state_hash_size_ > 0);
        const uint32_t hash_mask = hash_ready ? static_cast<uint32_t>(state_hash_size_ - 1) : 0u;

        const int TRANSFORM_PARALLEL_THRESHOLD = 64;
        if (num_transforms_ > TRANSFORM_PARALLEL_THRESHOLD) {
            CUDA_CHECK(cudaMemset(d_y, 0, fixed_sz_dim_ * sizeof(cuDoubleComplex)));
            dim3 block(16, 16);
            dim3 grid((fixed_sz_dim_ + block.x - 1) / block.x,
                     (num_transforms_ + block.y - 1) / block.y);
            if (hash_ready) {
                GPUKernels::matVecFixedSzTransformParallelHash<<<grid, block>>>(
                    d_x, d_y, d_basis_states_,
                    d_state_hash_, state_hash_size_, hash_mask,
                    d_transform_data_, num_transforms_, fixed_sz_dim_, n_sites_, spin_l_);
            } else {
                GPUKernels::matVecFixedSzTransformParallel<<<grid, block>>>(
                    d_x, d_y, d_basis_states_,
                    d_transform_data_, num_transforms_, fixed_sz_dim_, n_sites_, spin_l_);
            }
        } else {
            CUDA_CHECK(cudaMemset(d_y, 0, fixed_sz_dim_ * sizeof(cuDoubleComplex)));
            size_t shared_mem_size = std::min(num_transforms_, 4096) * sizeof(GPUTransformData);
            if (hash_ready) {
                GPUKernels::matVecFixedSzKernelOptimizedHash<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
                    d_x, d_y, d_basis_states_,
                    d_state_hash_, state_hash_size_, hash_mask,
                    fixed_sz_dim_, n_sites_, spin_l_,
                    d_transform_data_, num_transforms_);
            } else {
                GPUKernels::matVecFixedSzKernelOptimized<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
                    d_x, d_y, d_basis_states_,
                    fixed_sz_dim_, n_sites_, spin_l_,
                    d_transform_data_, num_transforms_);
            }
        }
    } else {
        std::cerr << "Error: GPUFixedSzOperator::matVecFixedSz called with no transform data" << std::endl;
        CUDA_CHECK(cudaMemset(d_y, 0, fixed_sz_dim_ * sizeof(cuDoubleComplex)));
    }

    CUDA_CHECK(cudaGetLastError());
    // No cudaDeviceSynchronize here: callers are responsible for
    // ordering with respect to subsequent reads of d_y. All consumers
    // (GPULanczos, GPUFTLM, GPUTPQ) chain operations on the same default
    // stream, so the dependency is implicit.

    if (timing_enabled) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        stats_.matVecTime = milliseconds / 1000.0;
        double flops = static_cast<double>(fixed_sz_dim_) * NNZ_PER_STATE_ESTIMATE * 8;
        stats_.throughput = flops / (stats_.matVecTime * 1e9);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

// Override matVecGPU to use fixed Sz version
void GPUFixedSzOperator::matVecGPU(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N) {
    if (N != fixed_sz_dim_) {
        throw std::runtime_error("GPUFixedSzOperator::matVecGPU: dimension mismatch");
    }
    matVecFixedSz(d_x, d_y);
}

// Override async matVec - fall back to synchronous fixed-Sz version
// The fixed-Sz kernels use shared d_basis_states_ and atomic accumulation,
// making concurrent multi-stream execution unsafe.
void GPUFixedSzOperator::matVecGPUAsync(const cuDoubleComplex* d_x, cuDoubleComplex* d_y, int N, cudaStream_t stream) {
    (void)stream;  // Cannot safely use custom stream with fixed-Sz kernels
    if (N != fixed_sz_dim_) {
        throw std::runtime_error("GPUFixedSzOperator::matVecGPUAsync: dimension mismatch");
    }
    matVecFixedSz(d_x, d_y);
}

// Override host-side matVec to use fixed Sz version
void GPUFixedSzOperator::matVec(const std::complex<double>* x, std::complex<double>* y, int N) {
    if (N != fixed_sz_dim_) {
        throw std::runtime_error("GPUFixedSzOperator::matVec: dimension mismatch");
    }
    
    if (!gpu_memory_allocated_) {
        allocateGPUMemory(N);
    }
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_vector_in_, x, N * sizeof(cuDoubleComplex), 
                         cudaMemcpyHostToDevice));
    
    // Perform matrix-vector product
    matVecFixedSz(d_vector_in_, d_vector_out_);
    
    // Copy output to host
    CUDA_CHECK(cudaMemcpy(y, d_vector_out_, N * sizeof(cuDoubleComplex),
                         cudaMemcpyDeviceToHost));
}

// Transform vector from fixed-Sz basis to full Hilbert space
std::vector<std::complex<double>> GPUFixedSzOperator::embedToFull(
    const std::vector<std::complex<double>>& fixed_sz_vec) {
    
    if (fixed_sz_vec.size() != static_cast<size_t>(fixed_sz_dim_)) {
        throw std::invalid_argument("Input vector size mismatch with fixed Sz dimension");
    }
    
    // Copy basis states from GPU to host
    std::vector<uint64_t> h_basis_states(fixed_sz_dim_);
    CUDA_CHECK(cudaMemcpy(h_basis_states.data(), d_basis_states_, 
                         fixed_sz_dim_ * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    // Create full-space vector
    uint64_t full_dim = 1ULL << n_sites_;
    std::vector<std::complex<double>> full_vec(full_dim, std::complex<double>(0.0, 0.0));
    
    // Map fixed-Sz coefficients to full-space positions
    for (int i = 0; i < fixed_sz_dim_; ++i) {
        uint64_t state = h_basis_states[i];
        full_vec[state] = fixed_sz_vec[i];
    }
    
    return full_vec;
}

std::vector<std::complex<double>> GPUFixedSzOperator::projectToReduced(
    const std::vector<std::complex<double>>& full_vec) {
    
    uint64_t full_dim = 1ULL << n_sites_;
    if (full_vec.size() != full_dim) {
        throw std::invalid_argument("Input vector size mismatch with full Hilbert space dimension");
    }
    
    // Copy basis states from GPU to host
    std::vector<uint64_t> h_basis_states(fixed_sz_dim_);
    CUDA_CHECK(cudaMemcpy(h_basis_states.data(), d_basis_states_, 
                         fixed_sz_dim_ * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    // Create reduced-space vector
    std::vector<std::complex<double>> reduced_vec(fixed_sz_dim_);
    
    // Extract coefficients at fixed-Sz basis state positions
    for (int i = 0; i < fixed_sz_dim_; ++i) {
        uint64_t state = h_basis_states[i];
        reduced_vec[i] = full_vec[state];
    }
    
    return reduced_vec;
}

#endif // WITH_CUDA

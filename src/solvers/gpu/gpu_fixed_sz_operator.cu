#ifdef WITH_CUDA

// Prevent inclusion of CPU Operator class
#define CONSTRUCT_HAM_H

#include <ed/gpu/gpu_operator.cuh>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdint>

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
    // Combinadic-unrank kernel needs the Pascal triangle in __constant__
    // memory. ensure_pascal_uploaded() is idempotent across operators.
    GPUKernels::ensure_pascal_uploaded();

    // Phase C of the "Kill the GPU State-Lookup Hash" plan (May 2026):
    // ``ED_GPU_STORE_BASIS=0`` opts out of the dim x 8 B basis-state
    // array on device. The Rank matvec kernels (Phase A.2) handle a
    // ``basis_states == nullptr`` input by computing the state on the
    // fly via ``unrank_combination_dev`` from constant-cache Pascal
    // entries -- same arithmetic that produced the array in the first
    // place, so the cost is moved from a global ``__ldg`` to a few
    // constant-cache reads. Saves 1.8 GB at dim = 225M.
    //
    // Refuses to drop the array when the Hash path is active or when
    // ``ED_GPU_FIXED_SZ_HASH=1`` (default-ish): the Hash kernels still
    // need basis_states for their initial state read, and the
    // host-side ``embedToFull`` / ``projectToReduced`` helpers also
    // do a single bulk D2H copy of it. Those helpers fall back to a
    // host-side unrank when the array is absent (see implementations
    // below).
    static const bool keep_basis_env = []() {
        const char* s = std::getenv("ED_GPU_STORE_BASIS");
        // Default: keep the basis (compat / fastest at typical N).
        // ``ED_GPU_STORE_BASIS=0`` -> drop it.
        return !(s != nullptr && s[0] == '0');
    }();
    // Hash variants still need the array for the initial-state read,
    // so if the user explicitly asked for the legacy hash path keep
    // the basis around regardless.
    static const bool legacy_hash_requested = []() {
        const char* s = std::getenv("ED_GPU_USE_HASH");
        return (s != nullptr && s[0] == '1');
    }();
    const bool will_store_basis = keep_basis_env || legacy_hash_requested;

    std::cout << "Building fixed Sz basis on GPU"
              << (will_store_basis ? "..." : " [device array dropped; unrank on demand]...")
              << "\n";

    if (!will_store_basis) {
        d_basis_states_ = nullptr;
        std::cout << "  Basis generation: skipped device storage "
                     "(saves " << (fixed_sz_dim_ * sizeof(uint64_t) >> 20)
                  << " MiB); states computed via combinadic unrank.\n";
        return;
    }

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

    // Phase A.3 of the "Kill the GPU State-Lookup Hash" plan (May 2026):
    // skip the entire hash build (malloc + 0xFF memset over 8 - 32 GiB +
    // 225M atomicCAS inserts + cudaDeviceSynchronize) unless the user
    // explicitly asks for the legacy lookup path via ED_GPU_USE_HASH=1.
    // The matvec dispatch defaults to the Rank kernels (Phase A.2) so
    // the hash is dead weight in the common case.
    //
    // Cached so a per-operator construction does not call getenv()
    // every time; matches the convention used by the matvec dispatch
    // above and the existing ED_GPU_TIMING knob.
    static const bool legacy_hash_requested = []() {
        const char* s = std::getenv("ED_GPU_USE_HASH");
        return (s != nullptr && s[0] == '1');
    }();
    if (!legacy_hash_requested) {
        // Quiet by default; one-shot log at the first skip so a user
        // who set ED_GPU_USE_HASH=1 expecting the hash but mistyped the
        // value still gets a hint in the run log.
        static bool logged_once = false;
        if (!logged_once) {
            std::cout << "  [hash] skipping ``buildStateHashOnGPU`` -- "
                         "matvec uses combinadic rank (set ED_GPU_USE_HASH=1 "
                         "to restore the legacy 8-32 GiB hash table).\n";
            logged_once = true;
        }
        // Leave use_hash_ alone (ABI compat) but record that we did
        // not build, so the matvec dispatcher's ``hash_ready`` check
        // correctly falls through to the Rank kernel.
        state_hash_size_ = 0;
        d_state_hash_    = nullptr;
        return;
    }

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

        // Phase A.2 of the "Kill the GPU State-Lookup Hash" plan
        // (May 2026): the Rank variants replace the 8 - 32 GiB
        // ``d_state_hash_`` lookup with a constant-cache combinadic
        // rank. They are the default; the legacy Hash path stays
        // available for diagnostic compares behind ``ED_GPU_USE_HASH=1``.
        //
        // We cache the env var read in a function-local static so the
        // matvec hot loop does not call getenv() per call; the cache
        // is invalidated at process exit, matching how the other ED
        // GPU env-var knobs work (e.g. ED_GPU_TIMING above).
        static const bool use_hash = []() {
            const char* s = std::getenv("ED_GPU_USE_HASH");
            return (s != nullptr && s[0] == '1');
        }();
        const bool hash_ready = use_hash
                                && (d_state_hash_ != nullptr)
                                && (state_hash_size_ > 0);
        const uint32_t hash_mask = hash_ready
                                   ? static_cast<uint32_t>(state_hash_size_ - 1)
                                   : 0u;

        // Phase B of the "Kill the GPU State-Lookup Hash" plan (May 2026):
        // route between the 2D (state x transform) parallel kernel and
        // the 1D-per-state kernel based on both transform count AND
        // dimension. The legacy heuristic was ``num_transforms > 64
        // -> 2D``, which is wrong at huge dim: the 2D kernel re-reads
        // ``__ldg(x[state_idx])`` ``num_transforms`` times per state
        // (different ``blockIdx.y`` -> different L1/L2 cache), turning
        // input-vector reads into the dominant HBM traffic. At
        // dim=225M, num_transforms=288 that is ~1 TB of redundant reads
        // per matvec.
        //
        // The 1D kernel keeps each state on a single thread that
        // iterates transforms in shared memory and reads ``x[idx]``
        // exactly once. Once dim crosses ~64M the 1D pattern wins
        // decisively even at large num_transforms; below that
        // crossover the 2D kernel still wins via more grid-level
        // parallelism.
        //
        // The crossover threshold is a single env-var knob
        // (``ED_GPU_FIXEDSZ_1D_DIM_THRESHOLD``) so a benchmark sweep
        // can pin it without rebuilding. The default (1 << 26 = ~64M)
        // is a conservative initial guess to be refined in Phase F.
        static const std::uint64_t kHugeDimThreshold = []() {
            const char* s = std::getenv("ED_GPU_FIXEDSZ_1D_DIM_THRESHOLD");
            if (s == nullptr || s[0] == '\0') return static_cast<std::uint64_t>(1ULL << 26);
            const long long v = std::atoll(s);
            return (v > 0) ? static_cast<std::uint64_t>(v)
                           : static_cast<std::uint64_t>(1ULL << 26);
        }();
        const int  TRANSFORM_PARALLEL_THRESHOLD = 64;
        const bool huge_dim = (static_cast<std::uint64_t>(fixed_sz_dim_) > kHugeDimThreshold);
        const bool use_2d   = (num_transforms_ > TRANSFORM_PARALLEL_THRESHOLD) && !huge_dim;
        if (use_2d) {
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
                // Default: Rank variant -- zero-HBM-traffic lookup.
                GPUKernels::matVecFixedSzTransformParallelRank<<<grid, block>>>(
                    d_x, d_y, d_basis_states_,
                    d_transform_data_, num_transforms_, fixed_sz_dim_, n_sites_, n_up_, spin_l_);
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
                // Default: Rank variant.
                GPUKernels::matVecFixedSzKernelOptimizedRank<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
                    d_x, d_y, d_basis_states_,
                    fixed_sz_dim_, n_sites_, n_up_, spin_l_,
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

// Phase C of the "Kill the GPU State-Lookup Hash" plan (May 2026):
// host-side combinadic unrank used by ``embedToFull`` /
// ``projectToReduced`` when ``ED_GPU_STORE_BASIS=0`` dropped the device
// array. Same colex convention as ``unrank_combination_dev`` in
// gpu_kernels.cu. Builds a small Pascal table on the host (cheap,
// ~33 KiB) and walks the rank top-down for each basis index.
//
// Cost: O(n_sites) per state, host-side; the function is called at
// most a few times in a typical run (vector boundary conversions),
// not in the matvec hot loop.
namespace {
struct HostPascal {
    std::uint64_t v[65][65];
    HostPascal() {
        for (int n = 0; n <= 64; ++n) {
            v[n][0] = 1ULL;
            for (int k = 1; k <= n; ++k) {
                v[n][k] = ((k - 1 >= 0) ? v[n-1][k-1] : 0ULL)
                       +  ((k < n)      ? v[n-1][k]   : 0ULL);
            }
            for (int k = n + 1; k <= 64; ++k) v[n][k] = 0ULL;
        }
    }
    std::uint64_t at(int n, int k) const {
        if (k < 0 || k > n || n < 0 || n > 64) return 0ULL;
        return v[n][k];
    }
};
inline const HostPascal& host_pascal() {
    static const HostPascal P;
    return P;
}

inline std::uint64_t host_unrank(std::uint64_t rank, int n_bits, int k) {
    const HostPascal& P = host_pascal();
    std::uint64_t state = 0ULL;
    for (int i = k - 1; i >= 0; --i) {
        int p = i;
        while (p + 1 < n_bits && P.at(p + 1, i + 1) <= rank) ++p;
        state |= (1ULL << p);
        rank -= P.at(p, i + 1);
    }
    return state;
}

// Materialise the basis-state array on the host, either by copying
// from device (if stored) or by walking the unrank.
std::vector<std::uint64_t> materialize_host_basis(const std::uint64_t* d_basis_states,
                                                  int fixed_sz_dim,
                                                  int n_sites, int n_up) {
    std::vector<std::uint64_t> h_basis(fixed_sz_dim);
    if (d_basis_states != nullptr) {
        CUDA_CHECK(cudaMemcpy(h_basis.data(), d_basis_states,
                              fixed_sz_dim * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));
    } else {
        for (int i = 0; i < fixed_sz_dim; ++i) {
            h_basis[i] = host_unrank(static_cast<std::uint64_t>(i), n_sites, n_up);
        }
    }
    return h_basis;
}
}  // namespace

// Transform vector from fixed-Sz basis to full Hilbert space
std::vector<std::complex<double>> GPUFixedSzOperator::embedToFull(
    const std::vector<std::complex<double>>& fixed_sz_vec) {
    
    if (fixed_sz_vec.size() != static_cast<size_t>(fixed_sz_dim_)) {
        throw std::invalid_argument("Input vector size mismatch with fixed Sz dimension");
    }
    
    auto h_basis_states = materialize_host_basis(d_basis_states_, fixed_sz_dim_,
                                                 n_sites_, n_up_);
    
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
    
    auto h_basis_states = materialize_host_basis(d_basis_states_, fixed_sz_dim_,
                                                 n_sites_, n_up_);
    
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

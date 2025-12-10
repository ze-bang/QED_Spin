/**
 * @file cuda_raii.cuh
 * @brief RAII wrappers for CUDA handles and resources
 * 
 * Provides automatic resource management for CUDA handles (cuBLAS, cuSPARSE, etc.)
 * and GPU memory allocations. Using these wrappers ensures proper cleanup even
 * when exceptions are thrown.
 * 
 * Example usage:
 *   CuBlasHandle blas;  // Automatically creates handle
 *   cublasZaxpy(blas.get(), ...);  // Use handle
 *   // Handle automatically destroyed when blas goes out of scope
 */

#ifndef CUDA_RAII_CUH
#define CUDA_RAII_CUH

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>
#include <stdexcept>
#include <string>

namespace cuda_raii {

// ============================================================================
// cuBLAS Handle RAII Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for cuBLAS handle
 * 
 * Automatically creates handle on construction and destroys on destruction.
 * Non-copyable, move-only semantics.
 */
class CuBlasHandle {
public:
    CuBlasHandle() : handle_(nullptr) {
        cublasStatus_t status = cublasCreate(&handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle: " + 
                                    std::to_string(static_cast<int>(status)));
        }
    }
    
    ~CuBlasHandle() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }
    
    // Non-copyable
    CuBlasHandle(const CuBlasHandle&) = delete;
    CuBlasHandle& operator=(const CuBlasHandle&) = delete;
    
    // Movable
    CuBlasHandle(CuBlasHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    
    CuBlasHandle& operator=(CuBlasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                cublasDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    // Access underlying handle
    cublasHandle_t get() const { return handle_; }
    operator cublasHandle_t() const { return handle_; }
    
    // Set stream
    void setStream(cudaStream_t stream) {
        cublasSetStream(handle_, stream);
    }
    
private:
    cublasHandle_t handle_;
};

// ============================================================================
// cuSPARSE Handle RAII Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for cuSPARSE handle
 */
class CuSparseHandle {
public:
    CuSparseHandle() : handle_(nullptr) {
        cusparseStatus_t status = cusparseCreate(&handle_);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSPARSE handle: " +
                                    std::to_string(static_cast<int>(status)));
        }
    }
    
    ~CuSparseHandle() {
        if (handle_) {
            cusparseDestroy(handle_);
        }
    }
    
    // Non-copyable
    CuSparseHandle(const CuSparseHandle&) = delete;
    CuSparseHandle& operator=(const CuSparseHandle&) = delete;
    
    // Movable
    CuSparseHandle(CuSparseHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    
    CuSparseHandle& operator=(CuSparseHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                cusparseDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    cusparseHandle_t get() const { return handle_; }
    operator cusparseHandle_t() const { return handle_; }
    
private:
    cusparseHandle_t handle_;
};

// ============================================================================
// cuRAND Generator RAII Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for cuRAND generator
 */
class CuRandGenerator {
public:
    explicit CuRandGenerator(curandRngType_t rng_type = CURAND_RNG_PSEUDO_DEFAULT) 
        : generator_(nullptr) {
        curandStatus_t status = curandCreateGenerator(&generator_, rng_type);
        if (status != CURAND_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuRAND generator: " +
                                    std::to_string(static_cast<int>(status)));
        }
    }
    
    ~CuRandGenerator() {
        if (generator_) {
            curandDestroyGenerator(generator_);
        }
    }
    
    // Non-copyable
    CuRandGenerator(const CuRandGenerator&) = delete;
    CuRandGenerator& operator=(const CuRandGenerator&) = delete;
    
    // Movable
    CuRandGenerator(CuRandGenerator&& other) noexcept : generator_(other.generator_) {
        other.generator_ = nullptr;
    }
    
    CuRandGenerator& operator=(CuRandGenerator&& other) noexcept {
        if (this != &other) {
            if (generator_) {
                curandDestroyGenerator(generator_);
            }
            generator_ = other.generator_;
            other.generator_ = nullptr;
        }
        return *this;
    }
    
    curandGenerator_t get() const { return generator_; }
    operator curandGenerator_t() const { return generator_; }
    
    void setSeed(unsigned long long seed) {
        curandSetPseudoRandomGeneratorSeed(generator_, seed);
    }
    
private:
    curandGenerator_t generator_;
};

// ============================================================================
// CUDA Stream RAII Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for CUDA stream
 */
class CudaStream {
public:
    CudaStream() : stream_(nullptr) {
        cudaError_t err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
    
    explicit CudaStream(unsigned int flags) : stream_(nullptr) {
        cudaError_t err = cudaStreamCreateWithFlags(&stream_, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream with flags: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
    
    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Non-copyable
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    
    // Movable
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
    void synchronize() {
        cudaStreamSynchronize(stream_);
    }
    
private:
    cudaStream_t stream_;
};

// ============================================================================
// CUDA Event RAII Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for CUDA event
 */
class CudaEvent {
public:
    CudaEvent() : event_(nullptr) {
        cudaError_t err = cudaEventCreate(&event_);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA event: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
    
    explicit CudaEvent(unsigned int flags) : event_(nullptr) {
        cudaError_t err = cudaEventCreateWithFlags(&event_, flags);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA event with flags: " +
                                    std::string(cudaGetErrorString(err)));
        }
    }
    
    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }
    
    // Non-copyable
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    
    // Movable
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }
    
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }
    
    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }
    
    void record(cudaStream_t stream = nullptr) {
        cudaEventRecord(event_, stream);
    }
    
    void synchronize() {
        cudaEventSynchronize(event_);
    }
    
    static float elapsedTime(const CudaEvent& start, const CudaEvent& stop) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start.event_, stop.event_);
        return ms;
    }
    
private:
    cudaEvent_t event_;
};

// ============================================================================
// GPU Memory RAII Wrapper
// ============================================================================

/**
 * @brief RAII wrapper for GPU memory allocation
 * 
 * Template class for type-safe GPU memory management.
 */
template<typename T>
class GpuMemory {
public:
    GpuMemory() : ptr_(nullptr), size_(0) {}
    
    explicit GpuMemory(size_t count) : ptr_(nullptr), size_(count) {
        if (count > 0) {
            cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate GPU memory: " +
                                        std::string(cudaGetErrorString(err)));
            }
        }
    }
    
    ~GpuMemory() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    // Non-copyable
    GpuMemory(const GpuMemory&) = delete;
    GpuMemory& operator=(const GpuMemory&) = delete;
    
    // Movable
    GpuMemory(GpuMemory&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    GpuMemory& operator=(GpuMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() const { return ptr_; }
    operator T*() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }
    
    // Copy from host
    void copyFromHost(const T* host_ptr, size_t count) {
        cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
    }
    
    // Copy to host
    void copyToHost(T* host_ptr, size_t count) const {
        cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    // Async copy from host
    void copyFromHostAsync(const T* host_ptr, size_t count, cudaStream_t stream) {
        cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream);
    }
    
    // Async copy to host
    void copyToHostAsync(T* host_ptr, size_t count, cudaStream_t stream) const {
        cudaMemcpyAsync(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
    }
    
    // Set memory to zero
    void zero() {
        cudaMemset(ptr_, 0, size_ * sizeof(T));
    }
    
private:
    T* ptr_;
    size_t size_;
};

} // namespace cuda_raii

#endif // WITH_CUDA

#endif // CUDA_RAII_CUH

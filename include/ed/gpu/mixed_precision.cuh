/**
 * @file mixed_precision.cuh
 * @brief Mixed precision utilities for GPU solvers
 * 
 * Provides options for using lower precision (FP32, TF32) in iterative methods
 * to improve performance. The final results are computed in FP64 for accuracy.
 * 
 * Performance gains:
 * - TF32 (Tensor Float 32): ~2x speedup on Ampere+ GPUs with minimal accuracy loss
 * - FP32: ~2x speedup, suitable for initial iterations with FP64 refinement
 * 
 * Usage:
 *   MixedPrecisionConfig config;
 *   config.use_tf32 = true;  // Enable TF32 for GEMM operations
 *   config.apply(cublas_handle);
 */

#ifndef MIXED_PRECISION_CUH
#define MIXED_PRECISION_CUH

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

namespace mixed_precision {

/**
 * @brief Configuration for mixed precision computation
 */
struct MixedPrecisionConfig {
    // Use TF32 for Tensor Core acceleration (Ampere+ GPUs)
    // TF32 provides FP32-like range with reduced precision (10-bit mantissa)
    // Minimal accuracy loss for iterative methods
    bool use_tf32 = false;
    
    // Use FP32 accumulation for complex operations (faster on some GPUs)
    bool use_fp32_accumulator = false;
    
    // Number of FP64 refinement iterations after mixed precision solve
    int refinement_iterations = 0;
    
    // Threshold for switching from mixed to full precision
    double refinement_threshold = 1e-8;
    
    /**
     * @brief Apply configuration to cuBLAS handle
     * 
     * @param handle cuBLAS handle to configure
     * @return true if configuration was applied successfully
     */
    bool apply(cublasHandle_t handle) const {
        if (!handle) return false;
        
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
        
        if (use_tf32) {
            // Enable TF32 for tensor core operations
            // This affects ZGEMM and similar operations on Ampere+ GPUs
            status = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
            if (status == CUBLAS_STATUS_SUCCESS) {
                std::cout << "Mixed Precision: TF32 Tensor Core math enabled\n";
            } else {
                std::cerr << "Warning: Failed to enable TF32 math mode\n";
                return false;
            }
        } else {
            // Use default (FP64) math
            status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }
        
        return (status == CUBLAS_STATUS_SUCCESS);
    }
    
    /**
     * @brief Reset to default precision mode
     */
    bool reset(cublasHandle_t handle) const {
        if (!handle) return false;
        cublasStatus_t status = cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        return (status == CUBLAS_STATUS_SUCCESS);
    }
    
    /**
     * @brief Print current configuration
     */
    void print() const {
        std::cout << "Mixed Precision Configuration:\n";
        std::cout << "  TF32 (Tensor Core): " << (use_tf32 ? "enabled" : "disabled") << "\n";
        std::cout << "  FP32 accumulator: " << (use_fp32_accumulator ? "enabled" : "disabled") << "\n";
        if (refinement_iterations > 0) {
            std::cout << "  Refinement iterations: " << refinement_iterations << "\n";
            std::cout << "  Refinement threshold: " << refinement_threshold << "\n";
        }
    }
    
    /**
     * @brief Check if GPU supports TF32
     * 
     * TF32 is supported on Ampere (compute capability 8.0+) GPUs
     */
    static bool isTF32Supported() {
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        
        // TF32 requires compute capability >= 8.0 (Ampere)
        bool supported = (props.major >= 8);
        
        if (supported) {
            std::cout << "GPU " << props.name << " supports TF32 (compute " 
                     << props.major << "." << props.minor << ")\n";
        }
        
        return supported;
    }
    
    /**
     * @brief Create optimal configuration for current GPU
     */
    static MixedPrecisionConfig autoDetect() {
        MixedPrecisionConfig config;
        
        if (isTF32Supported()) {
            config.use_tf32 = true;
            config.refinement_iterations = 1;  // One FP64 refinement pass
            std::cout << "Auto-detected: Enabling TF32 with FP64 refinement\n";
        }
        
        return config;
    }
};

/**
 * @brief RAII wrapper for temporary precision mode changes
 * 
 * Automatically restores original precision mode when destroyed.
 */
class ScopedMixedPrecision {
public:
    ScopedMixedPrecision(cublasHandle_t handle, const MixedPrecisionConfig& config)
        : handle_(handle), config_(config), original_mode_(CUBLAS_DEFAULT_MATH) {
        
        // Save current mode
        cublasGetMathMode(handle_, &original_mode_);
        
        // Apply new mode
        config_.apply(handle_);
    }
    
    ~ScopedMixedPrecision() {
        // Restore original mode
        cublasSetMathMode(handle_, original_mode_);
    }
    
    // Non-copyable
    ScopedMixedPrecision(const ScopedMixedPrecision&) = delete;
    ScopedMixedPrecision& operator=(const ScopedMixedPrecision&) = delete;
    
private:
    cublasHandle_t handle_;
    MixedPrecisionConfig config_;
    cublasMath_t original_mode_;
};

} // namespace mixed_precision

#endif // WITH_CUDA

#endif // MIXED_PRECISION_CUH

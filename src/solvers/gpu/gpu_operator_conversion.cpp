// ============================================================================
// CPU Operator → GPU Operator Conversion (Host-only compilation)
// ============================================================================

#ifdef WITH_CUDA

#include <ed/gpu/gpu_operator.cuh>
#include <ed/core/construct_ham.h>
#include <iostream>

/**
 * @brief Convert CPU Operator to GPUOperator using transform_data_ approach
 * 
 * This function bridges the gap between CPU and GPU implementations by:
 * 1. Extracting interaction terms from CPU Operator
 * 2. Converting to GPUTransformData format
 * 3. Populating GPU operator's transform_data_ and copying to device
 * 
 * This approach is more efficient than building a full sparse matrix because:
 * - GPU kernels compute matrix elements on-the-fly
 * - Avoids O(2^N × 2^N) sparse matrix construction
 * - Uses the same optimized kernels as direct GPU operator construction
 * 
 * @param cpu_op The CPU Operator to convert
 * @param gpu_op The GPUOperator to populate
 * @return true if conversion successful, false otherwise
 */
bool convertOperatorToGPU(const Operator& cpu_op, GPUOperator& gpu_op) {
    std::cout << "Converting CPU Operator to GPU (transform_data approach)..." << std::endl;
    
    // Access the CPU operator's transform data (Structure-of-Arrays format)
    // Both CPU and GPU use the same optimized representation!
    const auto& cpu_transforms = cpu_op.transform_data_;
    
    std::cout << "  Number of transform operations: " << cpu_transforms.size() << std::endl;
    
    if (cpu_transforms.empty()) {
        std::cerr << "Warning: CPU Operator has no transform data. "
                  << "Make sure the operator was loaded correctly." << std::endl;
        return false;
    }
    
    // Convert each CPU transform to GPU transform
    // The data structures are nearly identical, just need type conversions
    int one_body_count = 0;
    int two_body_count = 0;
    
    for (const auto& cpu_transform : cpu_transforms) {
        if (cpu_transform.is_two_body) {
            // Two-body interaction
            gpu_op.addTwoBodyTerm(
                cpu_transform.op_type,      // Already in 0=S+, 1=S-, 2=Sz format
                static_cast<uint32_t>(cpu_transform.site_index),
                cpu_transform.op_type_2,
                static_cast<uint32_t>(cpu_transform.site_index_2),
                cpu_transform.coefficient
            );
            two_body_count++;
        } else {
            // One-body term
            gpu_op.addOneBodyTerm(
                cpu_transform.op_type,      // Already in 0=S+, 1=S-, 2=Sz format
                static_cast<uint32_t>(cpu_transform.site_index),
                cpu_transform.coefficient
            );
            one_body_count++;
        }
    }
    
    std::cout << "  One-body terms: " << one_body_count << std::endl;
    std::cout << "  Two-body terms: " << two_body_count << std::endl;
    
    // Copy transform data to GPU device memory
    gpu_op.copyTransformDataToDevice();
    
    // NOTE: We no longer allocate GPU state vectors here.
    // The caller is responsible for allocating their own vectors if needed.
    // This prevents double-allocation when the caller manages their own memory.
    
    std::cout << "  GPU operator ready! (dimension: " << gpu_op.getDimension() << ")" << std::endl;
    std::cout << "  NOTE: State vectors NOT allocated - caller manages memory" << std::endl;
    
    return true;
}

#endif // WITH_CUDA

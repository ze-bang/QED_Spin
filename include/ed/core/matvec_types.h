#pragma once
// =============================================================================
// include/ed/core/matvec_types.h
//
// Lightweight callable/type aliases for matrix-free kernels and operator
// factories. Keep this header dependency-free except for STL primitives so
// modules such as ed/operators/spin_ops.h do not need to include heavy
// Operator class machinery from construct_ham/operator.h.
// =============================================================================

#include <complex>
#include <functional>

namespace ed::types {

using Complex = std::complex<double>;
using MatVec  = std::function<void(const Complex*, Complex*, int)>;

} // namespace ed::types

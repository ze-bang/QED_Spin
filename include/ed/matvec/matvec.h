#pragma once
// =============================================================================
// include/ed/matvec/matvec.h
//
// MatVecOperator: the single polymorphic interface every solver in the ED
// library consumes. Concrete operators subclass it and advertise their
// MemorySpace; solvers consume the base class.
//
// Design (hybrid pattern):
//   * Virtual at the boundary --- one virtual call per matvec, free at ED
//     dimensions (matvec body is microseconds to seconds).
//   * Internal kernels stay templated (see term_kernels.h) so the inner
//     bit-flip loops are fully inlined and SIMD-vectorisable.
//
// The interface intentionally exposes only the four pieces of metadata
// every Krylov / thermal solver in this codebase actually needs:
//   * dim()          : size of the local input/output buffer (in elements)
//   * global_dim()   : total problem dimension (defaults to dim())
//   * memory_space() : where the bytes live
//   * is_hermitian() : whether the surrounding solver may use Hermitian
//                      shortcuts (real eigenvalues, two-term Lanczos, ...)
//
// The matvec consumers (Lanczos, FTLM, TPQ, CG/LOBPCG, time evolution)
// are expressed in terms of this base class plus a matching Backend
// (axpy/dot/norm/scale/copy).
// =============================================================================

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <ed/matvec/memory_space.h>

namespace ed::matvec {

using Complex = std::complex<double>;

class MatVecOperator {
public:
    virtual ~MatVecOperator() = default;

    // -------------------------------------------------------------------
    // Hot path. Implementations may assume:
    //   * in != out (solvers always supply distinct buffers)
    //   * size == dim() (we check in debug; release skips for speed)
    //   * `in` / `out` are aligned to at least alignof(Complex) and live
    //     in the MemorySpace returned by memory_space(); the solver
    //     constructs them with a matching Backend.
    //
    // Semantics: `out = H * in`. Implementations MAY overwrite `out`
    // (they are not required to accumulate). Callers must zero `out`
    // before this if they want H * in + (previous out).
    // -------------------------------------------------------------------
    virtual void apply(const Complex* in,
                       Complex* out,
                       std::size_t size) const = 0;

    // -------------------------------------------------------------------
    // Metadata. All four are `virtual` so subclasses that compose other
    // operators (e.g. a basis-shift wrapper) can override them; the
    // implementations below are sensible defaults.
    // -------------------------------------------------------------------
    [[nodiscard]] virtual std::size_t dim() const = 0;
    [[nodiscard]] virtual std::size_t global_dim() const { return dim(); }
    [[nodiscard]] virtual MemorySpace memory_space() const {
        return MemorySpace::Host;
    }
    [[nodiscard]] virtual bool is_hermitian() const { return true; }

    // -------------------------------------------------------------------
    // Optional fast dense assembly. When supported, fill the `dim()` x `dim()`
    // matrix `dense` (COLUMN-MAJOR, pre-zeroed by the caller) directly from the
    // operator's sparse term structure -- O(nnz) total, vs O(dim) full matvecs
    // (= O(dim * nnz)) when the caller instead builds columns via `apply`. This
    // is REENTRANT (it reads only const term data + does const basis lookups; no
    // backend CSR/scratch is touched), so implementations may parallelize over
    // columns. Returns true if it built the matrix; false (the default) tells
    // the caller to fall back to the matvec column build. Operator implements
    // it for the full Hilbert space.
    [[nodiscard]] virtual bool try_build_dense_columns(Complex* /*dense*/,
                                                       std::size_t /*N*/) const {
        return false;
    }

    // Human-readable type tag for diagnostics / dispatch printouts.
    // Defaulted so subclasses can omit it; ours all override.
    [[nodiscard]] virtual std::string description() const {
        return "MatVecOperator";
    }

    // -------------------------------------------------------------------
    // Sanity helpers shared by all subclasses. Inlined into the hot
    // path; release builds compile to nothing when NDEBUG is set.
    // -------------------------------------------------------------------
    void check_size(std::size_t size) const {
#ifndef NDEBUG
        if (size != dim()) {
            throw std::invalid_argument(
                "MatVecOperator::apply: size " + std::to_string(size)
                + " != dim() " + std::to_string(dim())
                + " (operator " + description() + ")");
        }
#else
        (void)size;
#endif
    }
};

// Rectangular (cross-sector) operators such as ``CrossSectorOrbitObservable``
// carry their own ``apply(in, out)`` and do NOT derive from MatVecOperator.

// =============================================================================
// Callable adapter. Some CPU solvers take a
// `std::function<void(const Complex*, Complex*, int)>` for the matvec;
// this small free function turns any MatVecOperator into that callable
// shape:
//
//     full_diagonalization(as_apply_function(H_op), N, ...);
//
// The returned callable holds a reference to the passed operator; the
// caller must keep the operator alive for the lifetime of the callable.
// Cost: ~ one virtual call per matvec, identical to what a direct
// MatVecOperator& would pay. NO additional std::function allocation
// overhead beyond what the caller already had.
// =============================================================================
[[nodiscard]] inline auto as_apply_function(const MatVecOperator& op) {
    return [&op](const Complex* in, Complex* out, int n) {
        op.apply(in, out, static_cast<std::size_t>(n));
    };
}

} // namespace ed::matvec

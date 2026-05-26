#pragma once
// =============================================================================
// include/ed/matvec/matvec.h
//
// MatVecOperator: the single polymorphic interface every solver in the ED
// library consumes. Replaces the previous parallel hierarchies:
//
//   - std::function<void(const Complex*, Complex*, int)> (CPU solvers)
//   - GPUOperator* with device pointers (GPU solvers)
//   - DistributedOperator::apply(v_local, y_local) (MPI solvers)
//
// Each is now a concrete subclass of MatVecOperator advertising its
// MemorySpace; solvers consume the base class.
//
// Design (Hybrid pattern, per architecture decision May 2026):
//   * Virtual at the boundary --- one virtual call per matvec, free at ED
//     dimensions (matvec body is microseconds to seconds).
//   * Internal kernels stay templated (see term_kernels.h) so the inner
//     bit-flip loops are fully inlined and SIMD-vectorisable.
//
// The interface intentionally exposes only the four pieces of metadata
// every Krylov / thermal solver in this codebase actually needs:
//   * dim()          : size of the local input/output buffer (in elements)
//   * global_dim()   : sum of dim() across all ranks (== dim() if not MPI)
//   * memory_space() : where the bytes live
//   * is_hermitian() : whether the surrounding solver may use Hermitian
//                      shortcuts (real eigenvalues, two-term Lanczos, ...)
//
// All five existing matvec consumers in the codebase (Lanczos, FTLM, LTLM,
// TPQ, CG/LOBPCG, KPM-DOS, time evolution) are expressible in terms of
// this base class plus a matching Backend (axpy/dot/norm/scale/copy).
//
// Phase 1 of the matvec-unification revamp.
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

    // Human-readable type tag for diagnostics / dispatch printouts.
    // Defaulted so subclasses can omit it; ours all override.
    [[nodiscard]] virtual std::string description() const {
        return "MatVecOperator";
    }

    // nnz_per_row_estimate() was retired in the minimalist-architecture
    // rev (May 2026): no concrete subclass overrode it and no solver
    // path called it. The assembled-CSR decision lives inside
    // ``GPUOperator::selectPathway`` (which queries its own counters);
    // the matrix-free decision is implicit in ``MemorySpace``. For CPU
    // estimates use ``Operator::getTransformData().size()`` /
    // ``getThreeBodyData().size()``; for GPU use
    // ``GPUOperator::getNumTransforms()``.

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

// CrossSectorMatVecOperator (rectangular matvec base for DSSF) was
// retired in the minimalist-architecture rev (May 2026): zero
// implementors in tree across its full lifetime. ``CrossSectorObservable``
// (DSSF) carries its own ``apply(in, out)`` and does NOT derive from
// this. The DSSF + spatial symmetry workstream (S1 #37) will need a
// rectangular orbit-basis operator -- it will land as a fresh
// ``CrossSectorOrbitObservable`` rather than reviving this base class.

// =============================================================================
// Legacy-bridge adapter. Many existing CPU solvers in this codebase take a
// `std::function<void(const Complex*, Complex*, int)>` for the matvec.
// Rather than touch every solver signature, this small free function turns
// any MatVecOperator into that callable shape --- so callers can write:
//
//     ed::workflows::solve(H, ...)                // H is a LinearOperator
//        -> uses H.apply(...) via virtual dispatch                  (best)
//
//     legacy_solver(as_apply_function(some_matvec_op), N, ...);    (bridge)
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

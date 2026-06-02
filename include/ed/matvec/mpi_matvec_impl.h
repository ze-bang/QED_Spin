#pragma once
// =============================================================================
// include/ed/matvec/mpi_matvec_impl.h
//
// MpiMatVecImpl: the MemSpace=DistributedHost matrix-free SpMV, built on the
// reusable ``CommPlan`` halo schedule. P4 of the operator-collapse refactor
// (Jun 2026).
//
// This is the second half of the P4 extraction. ``CommPlan`` (comm_plan.h)
// lifted the MPI_Alltoallv halo *schedule* out of
// ``ed::distributed::DistributedOperator``; ``MpiMatVecImpl`` lifts the
// per-apply *SpMV body* out (``DistributedOperator::apply_local_``) and
// composes the two:
//
//   y_local = (H * v_global)[local_offset, local_offset + local_n)
//
// where every rank owns only its slab of v and y. Per apply:
//
//   1. ``plan_.exchange<Scalar>(v_local, recv_buf_)`` -- ONE MPI_Alltoallv of
//      Scalar values (complex OR real -- the datatype is dispatched at
//      compile time by CommPlan).
//   2. GATHER-form local SpMV: for each local output row r, accumulate
//      ``y_local[r] = sum_c <r|H|c> v[c]`` via
//      ``ed::matvec::kernel::gather_row`` (the single source of truth for
//      the GATHER bit-flip algebra). ``get_v(c)`` reads v[c] from the local
//      slab when c is local, else from the halo recv buffer via
//      ``plan_.recv_index_of(c)``.
//
// Scope: this implements the FULL-Hilbert distributed lane (global array
// index == bitstring), which is exactly the regime
// ``DistributedOperator`` supports today -- so it is a faithful, drop-in
// equivalent. The class is intentionally free of the heavy ``Operator``
// header: it borrows a ``TermStorage`` + spin length, which is all the
// GATHER kernel reads. The eventual unified ``Operator<BasisPolicy,
// DistributedHost>`` constructs one of these and forwards ``apply()`` to it.
//
// Thread-safety: a single instance is NOT safe to call concurrently (the
// recv buffer is a reused member); the inner GATHER loop is OpenMP-
// parallel over rows, matching the DistributedOperator contract.
// =============================================================================

#ifdef WITH_MPI

#include <mpi.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <ed/matvec/comm_plan.h>
#include <ed/matvec/memory_space.h>
#include <ed/matvec/term_kernels_gather.h>
#include <ed/matvec/term_storage.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace ed::matvec::dist {

class MpiMatVecImpl {
public:
    using Complex = std::complex<double>;

    MpiMatVecImpl() = default;

    // Collective construction. Every rank in ``comm`` must pass the SAME
    // n_bits + an equivalent ``TermStorage`` (same Hamiltonian definition).
    // The term storage is BORROWED (pointer-stable for the lifetime of
    // *this); the caller keeps it alive. ``spin_l`` is the spin length
    // (0.5 for spin-1/2). The flip-pattern set is derived from the terms.
    void build(MPI_Comm comm, std::uint64_t n_bits,
               const TermStorage& terms, double spin_l) {
        terms_  = &terms;
        spin_l_ = spin_l;
        const std::uint64_t global_dim = (n_bits == 0) ? 1ULL : (1ULL << n_bits);
        plan_.build(comm, global_dim, flip_patterns_from_terms(terms));
    }

    // -------- geometry (queryable on every rank) ---------------------------
    [[nodiscard]] std::uint64_t global_dim()   const noexcept { return plan_.global_dim(); }
    [[nodiscard]] std::uint64_t local_offset() const noexcept { return plan_.local_offset(); }
    [[nodiscard]] std::uint64_t local_n()      const noexcept { return plan_.local_n(); }
    [[nodiscard]] int           rank()         const noexcept { return plan_.rank(); }
    [[nodiscard]] int           comm_size()    const noexcept { return plan_.size(); }
    [[nodiscard]] MPI_Comm      comm()         const noexcept { return plan_.comm(); }
    [[nodiscard]] const CommPlan& plan()       const noexcept { return plan_; }
    [[nodiscard]] MemorySpace   memory_space() const noexcept {
        return MemorySpace::DistributedHost;
    }

    // -------- complex SpMV: y_local = (H v_global) on our slab -------------
    // v_local / y_local are sized exactly ``local_n()``.
    void apply_complex(const Complex* v_local, Complex* y_local) const {
        plan_.exchange<Complex>(v_local, recv_buf_);
        const std::uint64_t off = plan_.local_offset();
        const std::uint64_t loc = plan_.local_n();

        auto get_v = [&](std::uint64_t c) -> Complex {
            if (c >= off && c < off + loc) return v_local[c - off];
            const std::size_t k = plan_.recv_index_of(c);
            // build() guarantees every visited column is local OR in the
            // halo; a miss is a comm-plan bug.
            return recv_buf_[k];
        };

        std::fill(y_local, y_local + loc, Complex(0.0, 0.0));
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 4096)
#endif
        for (std::uint64_t r_local = 0; r_local < loc; ++r_local) {
            const std::uint64_t r = off + r_local;
            y_local[r_local] = ed::matvec::kernel::gather_row(
                r, v_local[r_local], *terms_, spin_l_, get_v);
        }
    }

    // -------- real SpMV (real-Hermitian fast path, MPI_DOUBLE halo) --------
    // Caller must have verified the Hamiltonian is real (no complex
    // couplings); the gather kernel projects coefficients to their real
    // part via the legacy gather_row complex path, so we run the complex
    // kernel and take the real component. The halo itself is exchanged as
    // doubles (half the bandwidth) and v[c] is reconstructed as a real
    // Complex for the kernel.
    void apply_real(const double* v_local, double* y_local) const {
        plan_.exchange<double>(v_local, recv_real_buf_);
        const std::uint64_t off = plan_.local_offset();
        const std::uint64_t loc = plan_.local_n();

        auto get_v = [&](std::uint64_t c) -> Complex {
            if (c >= off && c < off + loc) return Complex(v_local[c - off], 0.0);
            const std::size_t k = plan_.recv_index_of(c);
            return Complex(recv_real_buf_[k], 0.0);
        };

        std::fill(y_local, y_local + loc, 0.0);
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 4096)
#endif
        for (std::uint64_t r_local = 0; r_local < loc; ++r_local) {
            const std::uint64_t r = off + r_local;
            const Complex acc = ed::matvec::kernel::gather_row(
                r, Complex(v_local[r_local], 0.0), *terms_, spin_l_, get_v);
            y_local[r_local] = acc.real();
        }
    }

private:
    CommPlan             plan_;
    const TermStorage*   terms_  = nullptr;  // borrowed
    double               spin_l_ = 0.5;

    // Reused halo staging buffers (resized inside CommPlan::exchange).
    mutable std::vector<Complex> recv_buf_;
    mutable std::vector<double>  recv_real_buf_;
};

}  // namespace ed::matvec::dist

#endif  // WITH_MPI

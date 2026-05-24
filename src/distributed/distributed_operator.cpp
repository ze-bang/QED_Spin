// =============================================================================
// src/distributed/distributed_operator.cpp
//
// Phase 3b #1 implementation. See include/ed/distributed/distributed_operator.h
// for the design rationale and the GATHER vs SCATTER discussion.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_operator.h>

// Pull in the full Operator definition (we need the SoA term storage).
#include <ed/core/construct_ham.h>

// GATHER-form bit-flip kernel (the dual of ed::matvec::kernel::apply_terms).
#include <ed/matvec/term_kernels_gather.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace ed::distributed {

namespace {

// MPI datatype for std::complex<double>. We use MPI_C_DOUBLE_COMPLEX rather
// than MPI_DOUBLE_COMPLEX (the Fortran symbol) because std::complex<double>
// is layout-compatible with C99 _Complex double per [complex.numbers] /
// MPI 3.0 §3.4 (MPI_C_DOUBLE_COMPLEX is the C-binding type).
//
// OpenMPI implements the predefined datatype handles as `(void*)` casts to
// runtime symbols, which means they're not constexpr. We therefore declare
// these as `const` (resolved at first use; ODR-safe within a TU).
const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

// Saturating cast guarding the (Complex element count) -> (MPI int count)
// conversion at MPI_Alltoallv boundaries. We never expect to overflow in
// the bounded test regime, but a hard error here is much friendlier than a
// silent wraparound at honest-40.
inline int as_mpi_int(std::size_t n, const char* what) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string("DistributedOperator: ") + what
            + " count " + std::to_string(n) + " exceeds INT_MAX. "
            "Phase 3b honest-40 needs symmetry-aware slabbing or MPI_Count "
            "support; see include/ed/distributed/distributed_operator.h.");
    }
    return static_cast<int>(n);
}

}  // namespace

// -----------------------------------------------------------------------------
// Static helpers
// -----------------------------------------------------------------------------
void DistributedOperator::balanced_slab(std::uint64_t global_dim, int rank,
                                        int size,
                                        std::uint64_t& out_offset,
                                        std::uint64_t& out_n) noexcept {
    const std::uint64_t base = global_dim / static_cast<std::uint64_t>(size);
    const std::uint64_t rem  = global_dim % static_cast<std::uint64_t>(size);
    const std::uint64_t r    = static_cast<std::uint64_t>(rank);
    if (r < rem) {
        out_n      = base + 1;
        out_offset = r * (base + 1);
    } else {
        out_n      = base;
        out_offset = rem * (base + 1) + (r - rem) * base;
    }
}

int DistributedOperator::balanced_owner_rank(std::uint64_t global_idx,
                                             std::uint64_t global_dim,
                                             int size) noexcept {
    if (size <= 0 || global_dim == 0) return 0;
    const std::uint64_t base = global_dim / static_cast<std::uint64_t>(size);
    const std::uint64_t rem  = global_dim % static_cast<std::uint64_t>(size);
    // First `rem` ranks each have (base + 1) rows occupying the leading
    // rem * (base + 1) global indices. The remaining ranks each have
    // `base` rows.
    const std::uint64_t cutoff = rem * (base + 1);
    if (global_idx < cutoff) {
        return static_cast<int>(global_idx / (base + 1));
    }
    if (base == 0) {
        // Defensive: extra index past the populated rank range. Pin to last.
        return size - 1;
    }
    return static_cast<int>(rem + (global_idx - cutoff) / base);
}

int DistributedOperator::owner_rank(std::uint64_t global_idx) const noexcept {
    // Fast path via prefix lookup in rank_offsets_.
    auto it = std::upper_bound(rank_offsets_.begin(), rank_offsets_.end(),
                               global_idx);
    return static_cast<int>(std::distance(rank_offsets_.begin(), it)) - 1;
}

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
DistributedOperator::DistributedOperator(std::shared_ptr<Operator> op,
                                         MPI_Comm comm)
    : op_(std::move(op)), comm_(comm) {
    if (!op_) {
        throw std::invalid_argument(
            "DistributedOperator: null operator");
    }
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    const std::uint64_t n_bits = op_->getNumBits();
    if (n_bits >= 64) {
        throw std::runtime_error(
            "DistributedOperator: n_bits >= 64 unsupported");
    }
    global_dim_ = (n_bits == 0) ? 1ULL : (1ULL << n_bits);

    balanced_slab(global_dim_, rank_, size_, local_offset_, local_n_);

    // Build prefix table for O(log size_) owner_rank() lookups.
    rank_offsets_.assign(size_ + 1, 0);
    for (int r = 0; r < size_; ++r) {
        std::uint64_t off, n;
        balanced_slab(global_dim_, r, size_, off, n);
        rank_offsets_[r] = off;
    }
    rank_offsets_[size_] = global_dim_;

    // Make sure SoA term storage is fresh before we read it (idempotent).
    op_->commitPendingTransforms();

    extract_flip_patterns_();
    build_comm_pattern_();
}

// -----------------------------------------------------------------------------
// Bit-flip pattern extraction
// -----------------------------------------------------------------------------
//
// Walks the operator's separated SoA storage and records every distinct
// "column = row XOR p" pattern p. The trick is that every term type has a
// fixed pattern shape:
//
//   diag_one_body_      : p = 0                         (Sz)
//   offdiag_one_body_   : p = (1 << site)               (S+ / S-)
//   diag_two_body_      : p = 0                         (Sz Sz)
//   mixed_two_body_     : p = (1 << flip_site)          (Sz S+/-, S+/- Sz)
//   offdiag_two_body_   : p = (1 << s1) ^ (1 << s2)     (S+ S-, etc.)
//   three_body_data_    : p = XOR over (1 << site_k) for op_type_k != 2
//
// We dedupe via a small std::unordered_set. For Heisenberg with O(N) bonds
// the number of unique patterns is O(N) -- tiny.
void DistributedOperator::extract_flip_patterns_() {
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(64);
    auto add = [&](std::uint64_t p) { seen.insert(p); };

    // Always include 0 -- the diagonal / "do nothing" case is structural.
    add(0);

    // SoA bins live on Operator::terms_ after the May-2026 term-storage
    // unification. Rebuild the cache from the canonical AoS (cheap when
    // already fresh) before reading the binned views.
    op_->commitPendingTransforms();
    for (const auto& t : op_->terms_.diag_one_body)    { add(0); (void)t; }
    for (const auto& t : op_->terms_.offdiag_one_body) { add(1ULL << t.site_index); }
    for (const auto& t : op_->terms_.diag_two_body)    { add(0); (void)t; }
    for (const auto& t : op_->terms_.mixed_two_body)   { add(1ULL << t.flip_site); }
    for (const auto& t : op_->terms_.offdiag_two_body) {
        add((1ULL << t.site_index_1) ^ (1ULL << t.site_index_2));
    }
    for (const auto& t : op_->three_body_data_) {
        std::uint64_t p = 0;
        if (t.op_type_1 != 2) p ^= (1ULL << t.site_index_1);
        if (t.op_type_2 != 2) p ^= (1ULL << t.site_index_2);
        if (t.op_type_3 != 2) p ^= (1ULL << t.site_index_3);
        add(p);
    }

    flip_patterns_.assign(seen.begin(), seen.end());
    std::sort(flip_patterns_.begin(), flip_patterns_.end());
}

// -----------------------------------------------------------------------------
// Communication pattern build
// -----------------------------------------------------------------------------
//
// Algorithm:
//   pass 1 -- enumerate (local row, pattern) and bucket every off-rank
//             column into recv_global_per_rank[owner].
//   sort + dedupe each bucket. Lengths give recv_counts_.
//   MPI_Alltoall(recv_counts_, send_counts_).
//   MPI_Alltoallv exchanges the *global indices* we want with each peer.
//   Each rank converts the indices it received from peer R into local
//   slab offsets in v_local; that is its send_local_idx_ contribution
//   for peer R's column requests.
//   Build recv_lookup_ = SortedUint64Index over the concatenated recv
//   global-index list with values = position in recv_buf.
void DistributedOperator::build_comm_pattern_() {
    const int S = size_;
    std::vector<std::vector<std::uint64_t>> recv_global_per_rank(S);

    // -- pass 1: enumerate ----------------------------------------------------
    // Reserve heuristically: each off-rank column shows up at most once per
    // pattern per local row. The strict upper bound is local_n_ * patterns,
    // but in practice the per-rank halo is dominated by neighbours.
    const std::size_t reserve_hint =
        std::min<std::size_t>(local_n_ * flip_patterns_.size(),
                              static_cast<std::size_t>(global_dim_));
    for (auto& v : recv_global_per_rank) {
        v.reserve(std::min<std::size_t>(reserve_hint, 1ULL << 20));
    }

    for (std::uint64_t r_local = 0; r_local < local_n_; ++r_local) {
        const std::uint64_t r = local_offset_ + r_local;
        for (std::uint64_t p : flip_patterns_) {
            const std::uint64_t c = r ^ p;
            if (c >= global_dim_) continue;  // defensive: 0 pattern always inside
            // local slab?
            if (c >= local_offset_ && c < local_offset_ + local_n_) continue;
            const int owner = owner_rank(c);
            recv_global_per_rank[owner].push_back(c);
        }
    }

    // -- dedupe + sort each bucket -------------------------------------------
    recv_counts_.assign(S, 0);
    recv_displs_.assign(S, 0);
    for (int r = 0; r < S; ++r) {
        auto& bucket = recv_global_per_rank[r];
        std::sort(bucket.begin(), bucket.end());
        bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
        recv_counts_[r] = as_mpi_int(bucket.size(), "recv_counts");
    }
    int run = 0;
    for (int r = 0; r < S; ++r) {
        recv_displs_[r] = run;
        run += recv_counts_[r];
    }
    total_recv_ = run;

    // -- exchange counts ------------------------------------------------------
    send_counts_.assign(S, 0);
    send_displs_.assign(S, 0);
    MPI_Alltoall(recv_counts_.data(), 1, MPI_INT,
                 send_counts_.data(), 1, MPI_INT, comm_);
    int run2 = 0;
    for (int r = 0; r < S; ++r) {
        send_displs_[r] = run2;
        run2 += send_counts_[r];
    }
    total_send_ = run2;

    // -- exchange the global indices we want from each peer ------------------
    std::vector<std::uint64_t> recv_global_concat;
    recv_global_concat.reserve(static_cast<std::size_t>(total_recv_));
    for (int r = 0; r < S; ++r) {
        recv_global_concat.insert(recv_global_concat.end(),
                                  recv_global_per_rank[r].begin(),
                                  recv_global_per_rank[r].end());
    }

    std::vector<std::uint64_t> peer_wants_from_us(static_cast<std::size_t>(total_send_));

    // Custom datatype for uint64_t. MPI_UINT64_T is required by MPI 3.0
    // and supported by every implementation we link against (OpenMPI
    // >= 1.6, MPICH >= 3.0, Intel MPI >= 5).
    MPI_Alltoallv(recv_global_concat.data(),
                  recv_counts_.data(), recv_displs_.data(), MPI_UINT64_T,
                  peer_wants_from_us.data(),
                  send_counts_.data(), send_displs_.data(), MPI_UINT64_T,
                  comm_);

    // -- translate "global idx peer wants" -> "local idx in v_local" ---------
    send_local_idx_.assign(static_cast<std::size_t>(total_send_), 0);
    for (int r = 0; r < S; ++r) {
        const int off = send_displs_[r];
        const int n   = send_counts_[r];
        for (int k = 0; k < n; ++k) {
            const std::uint64_t g = peer_wants_from_us[off + k];
            if (g < local_offset_ || g >= local_offset_ + local_n_) {
                throw std::logic_error(
                    "DistributedOperator: peer requested out-of-slab index "
                    + std::to_string(g) + " from rank " + std::to_string(rank_)
                    + " (slab=[" + std::to_string(local_offset_) + ","
                    + std::to_string(local_offset_ + local_n_) + "))");
            }
            send_local_idx_[off + k] = static_cast<int>(g - local_offset_);
        }
    }

    // -- build the recv-side lookup ------------------------------------------
    recv_lookup_.clear();
    recv_lookup_.reserve(static_cast<std::size_t>(total_recv_));
    for (std::size_t k = 0; k < recv_global_concat.size(); ++k) {
        recv_lookup_.insert(recv_global_concat[k], k);
    }
    recv_lookup_.finalize();

    // -- pre-size the reusable halo staging buffers (Phase 8) ----------------
    // total_send_ / total_recv_ are fixed for the lifetime of *this; sizing
    // the buffers here means apply() does no allocation in the hot path.
    send_buf_.assign(static_cast<std::size_t>(total_send_), Complex(0.0, 0.0));
    recv_buf_.assign(static_cast<std::size_t>(total_recv_), Complex(0.0, 0.0));
}

// -----------------------------------------------------------------------------
// SpMV -- canonical implementation. Both the 2-arg ``apply(v, y)`` and
// the 3-arg MatVecOperator ``apply(v, y, size)`` forward here.
// (Audit STRUCTURAL_AUDIT.md S1 #10.)
// -----------------------------------------------------------------------------
void DistributedOperator::apply_local_(const Complex* v_local,
                                       Complex* y_local) const {
    // Pack send buffer. The destination buffer is the persistent
    // send_buf_ allocated once in build_comm_pattern_ -- no per-call
    // heap activity in the hot path (Phase 8 carryover of the
    // memory-pool optimisation that the Phase 6.1 audit already applied
    // to the CPU code paths).
    Complex* const send_buf = send_buf_.data();
    for (int k = 0; k < total_send_; ++k) {
        send_buf[k] = v_local[send_local_idx_[k]];
    }

    // Halo exchange into the matching persistent recv_buf_.
    Complex* const recv_buf = recv_buf_.data();
    MPI_Alltoallv(send_buf,
                  send_counts_.data(), send_displs_.data(), kComplexDatatype,
                  recv_buf,
                  recv_counts_.data(), recv_displs_.data(), kComplexDatatype,
                  comm_);

    // Helper to read v[c] for any global column c that appears in our
    // bit-flip-pattern enumeration. Local hits are O(1); off-rank hits are
    // O(log total_recv_) via the SortedUint64Index binary search.
    auto get_v = [&](std::uint64_t c) -> Complex {
        if (c >= local_offset_ && c < local_offset_ + local_n_) {
            return v_local[c - local_offset_];
        }
        const std::size_t idx = recv_lookup_.find(c);
        // If build_comm_pattern_ ran correctly, every visited column is
        // either local OR present in recv_lookup_. A miss is a bug.
        assert(idx != ed::core::SortedUint64Index::kNotFound &&
               "DistributedOperator::apply: lookup miss "
               "(comm plan does not cover all visited columns)");
        return recv_buf[idx];
    };

    // Ensure the SoA term cache is fresh (after May-2026 term-storage
    // unification the SoA bins live on Operator::terms_, regenerated
    // from the canonical AoS by commitPendingTransforms()).
    op_->commitPendingTransforms();

    // Zero the output slab.
    std::fill(y_local, y_local + local_n_, Complex(0.0, 0.0));

    // -------------------------------------------------------------------------
    // GATHER-form matrix-free SpMV.
    //
    // For each LOCAL output row r, accumulate
    //   y_local[r] = sum_c <r|H|c> v[c]
    // by dispatching to ``ed::matvec::kernel::gather_row``, which holds the
    // single source of truth for the GATHER direction of the bit-flip
    // algebra (the SCATTER counterpart lives in
    // ``ed::matvec::kernel::apply_terms``). Reading ``v[c]`` for any
    // global column c -- local OR off-rank -- is the ``get_v`` closure
    // captured above; the kernel is unaware of MPI / NCCL / etc.
    // -------------------------------------------------------------------------

    const double spin_l = static_cast<double>(op_->getSpin());

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4096)
#endif
    for (std::uint64_t r_local = 0; r_local < local_n_; ++r_local) {
        const std::uint64_t r = local_offset_ + r_local;
        y_local[r_local] = ed::matvec::kernel::gather_row(
            r, v_local[r_local], op_->terms_, spin_l, get_v);
    }
}

// -----------------------------------------------------------------------------
// Diagnostics
// -----------------------------------------------------------------------------
std::size_t DistributedOperator::plan_bytes() const noexcept {
    std::size_t b = 0;
    b += rank_offsets_.capacity() * sizeof(std::uint64_t);
    b += flip_patterns_.capacity() * sizeof(std::uint64_t);
    b += send_counts_.capacity()   * sizeof(int);
    b += send_displs_.capacity()   * sizeof(int);
    b += recv_counts_.capacity()   * sizeof(int);
    b += recv_displs_.capacity()   * sizeof(int);
    b += send_local_idx_.capacity() * sizeof(int);
    b += recv_lookup_.size_bytes();
    return b;
}

}  // namespace ed::distributed

#endif  // WITH_MPI

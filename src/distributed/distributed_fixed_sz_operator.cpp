// =============================================================================
// src/distributed/distributed_fixed_sz_operator.cpp
//
// Wave 2 of the "Unify all 16 matvec cells" plan (May 2026). Cell 2C
// (Distributed Fixed-Sz). Mirror of distributed_operator.cpp, with two
// changes:
//
//   1. Flip-pattern enumeration walks BITSTRINGS, not array indices.
//      Each local row's basis_state determines whether a pattern yields
//      an in-basis (Sz-preserving) column.
//   2. The matvec hot-path uses ``gather_row_basis<FixedSzBasisPolicy>``
//      instead of the bitstring-keyed ``gather_row``; ``index_of`` goes
//      through the parent ``FixedSzOperator``'s Lin (1990) table.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_fixed_sz_operator.h>

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/construct_ham.h>
#include <ed/matvec/term_kernels_gather.h>
#include <ed/matvec/basis_policy.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace ed::distributed {

namespace {

const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

inline int as_mpi_int(std::size_t n, const char* what) {
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string("DistributedFixedSzOperator: ")
            + what + " count " + std::to_string(n) + " exceeds INT_MAX.");
    }
    return static_cast<int>(n);
}

inline int popcount64(std::uint64_t x) noexcept {
    return __builtin_popcountll(x);
}

}  // namespace

// -----------------------------------------------------------------------------
// Static helpers (mirror DistributedOperator::balanced_slab /
// balanced_owner_rank).
// -----------------------------------------------------------------------------
void DistributedFixedSzOperator::balanced_slab(std::uint64_t global_dim,
                                               int rank, int size,
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

int DistributedFixedSzOperator::balanced_owner_rank(std::uint64_t global_idx,
                                                    std::uint64_t global_dim,
                                                    int size) noexcept {
    if (size <= 0 || global_dim == 0) return 0;
    const std::uint64_t base = global_dim / static_cast<std::uint64_t>(size);
    const std::uint64_t rem  = global_dim % static_cast<std::uint64_t>(size);
    const std::uint64_t cutoff = rem * (base + 1);
    if (global_idx < cutoff) {
        return static_cast<int>(global_idx / (base + 1));
    }
    if (base == 0) return size - 1;
    return static_cast<int>(rem + (global_idx - cutoff) / base);
}

int DistributedFixedSzOperator::owner_rank(std::uint64_t global_idx) const noexcept {
    auto it = std::upper_bound(rank_offsets_.begin(), rank_offsets_.end(),
                               global_idx);
    return static_cast<int>(std::distance(rank_offsets_.begin(), it)) - 1;
}

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
DistributedFixedSzOperator::DistributedFixedSzOperator(
    std::shared_ptr<FixedSzOperator> op,
    MPI_Comm comm)
    : op_(std::move(op)), comm_(comm)
{
    if (!op_) {
        throw std::invalid_argument(
            "DistributedFixedSzOperator: null operator");
    }
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    const std::uint64_t n_bits = op_->getNumBits();
    if (n_bits >= 64) {
        throw std::runtime_error(
            "DistributedFixedSzOperator: n_bits >= 64 unsupported");
    }

    global_dim_ = op_->getFixedSzDim();
    balanced_slab(global_dim_, rank_, size_, local_offset_, local_n_);

    rank_offsets_.assign(size_ + 1, 0);
    for (int r = 0; r < size_; ++r) {
        std::uint64_t off, n;
        balanced_slab(global_dim_, r, size_, off, n);
        rank_offsets_[r] = off;
    }
    rank_offsets_[size_] = global_dim_;

    op_->commitPendingTransforms();
    build_comm_pattern_();
}

// -----------------------------------------------------------------------------
// Comm-pattern build
//
// Pattern enumeration: for each local row r_local, walk the operator's
// SoA term storage and emit XOR patterns as the SCATTER kernel would.
// Test popcount preservation; if preserved, lookup c_idx via the
// FixedSzOperator's LinIndexTable. Remote c_idx (not in our slab) is
// added to the per-peer recv list.
// -----------------------------------------------------------------------------
void DistributedFixedSzOperator::build_comm_pattern_() {
    const int S = size_;
    std::vector<std::vector<std::uint64_t>> recv_global_per_rank(S);

    const auto& basis_states = op_->getBasisStates();
    const auto& lin_index    = op_->lin_index_table();

    // Pre-compute the set of unique flip patterns, skipping odd-popcount
    // ones (which never preserve Sz). diag terms add pattern 0; offdiag
    // pairs add even-popcount patterns; single S+/- adds odd-popcount
    // (which we drop -- they can never yield in-basis columns in any
    // Fixed-Sz sector).
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(64);
    seen.insert(0);  // diagonal always

    for (const auto& t : op_->terms_.offdiag_two_body) {
        const std::uint64_t p = (1ULL << t.site_index_1) ^ (1ULL << t.site_index_2);
        if (popcount64(p) % 2 == 0) seen.insert(p);
    }
    for (const auto& t : op_->three_body_data_) {
        std::uint64_t p = 0;
        if (t.op_type_1 != 2) p ^= (1ULL << t.site_index_1);
        if (t.op_type_2 != 2) p ^= (1ULL << t.site_index_2);
        if (t.op_type_3 != 2) p ^= (1ULL << t.site_index_3);
        if (popcount64(p) % 2 == 0) seen.insert(p);
    }
    // offdiag_one_body / mixed_two_body each flip a single bit -- odd
    // popcount; always leaves the Fixed-Sz basis. Skip.

    std::vector<std::uint64_t> patterns(seen.begin(), seen.end());
    std::sort(patterns.begin(), patterns.end());

    // -- pass 1: enumerate ---------------------------------------------------
    for (std::uint64_t r_local = 0; r_local < local_n_; ++r_local) {
        const std::uint64_t r_idx  = local_offset_ + r_local;
        const std::uint64_t r_state = basis_states[r_idx];
        for (std::uint64_t p : patterns) {
            const std::uint64_t c_state = r_state ^ p;
            // popcount preservation (NB: r_state^p preserves popcount
            // iff popcount(r_state AND p) equals popcount(p)/2).
            if (popcount64(r_state) != popcount64(c_state)) continue;
            const std::int64_t c_idx = lin_index.lookup(c_state);
            if (c_idx < 0) continue;  // defensive: shouldn't fire after popcount check
            const std::uint64_t c = static_cast<std::uint64_t>(c_idx);
            if (c >= local_offset_ && c < local_offset_ + local_n_) continue;
            const int owner = owner_rank(c);
            recv_global_per_rank[owner].push_back(c);
        }
    }

    // -- dedupe + counts ----------------------------------------------------
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

    std::vector<std::uint64_t> recv_global_concat;
    recv_global_concat.reserve(static_cast<std::size_t>(total_recv_));
    for (int r = 0; r < S; ++r) {
        recv_global_concat.insert(recv_global_concat.end(),
                                  recv_global_per_rank[r].begin(),
                                  recv_global_per_rank[r].end());
    }

    std::vector<std::uint64_t> peer_wants_from_us(
        static_cast<std::size_t>(total_send_));

    MPI_Alltoallv(recv_global_concat.data(),
                  recv_counts_.data(), recv_displs_.data(), MPI_UINT64_T,
                  peer_wants_from_us.data(),
                  send_counts_.data(), send_displs_.data(), MPI_UINT64_T,
                  comm_);

    send_local_idx_.assign(static_cast<std::size_t>(total_send_), 0);
    for (int r = 0; r < S; ++r) {
        const int off = send_displs_[r];
        const int n   = send_counts_[r];
        for (int k = 0; k < n; ++k) {
            const std::uint64_t g = peer_wants_from_us[off + k];
            if (g < local_offset_ || g >= local_offset_ + local_n_) {
                throw std::logic_error(
                    "DistributedFixedSzOperator: peer requested "
                    "out-of-slab index " + std::to_string(g) + " from rank "
                    + std::to_string(rank_));
            }
            send_local_idx_[off + k] = static_cast<int>(g - local_offset_);
        }
    }

    recv_lookup_.clear();
    recv_lookup_.reserve(static_cast<std::size_t>(total_recv_));
    for (std::size_t k = 0; k < recv_global_concat.size(); ++k) {
        recv_lookup_.insert(recv_global_concat[k], k);
    }
    recv_lookup_.finalize();

    send_buf_.assign(static_cast<std::size_t>(total_send_), Complex(0.0, 0.0));
    recv_buf_.assign(static_cast<std::size_t>(total_recv_), Complex(0.0, 0.0));
}

// -----------------------------------------------------------------------------
// SpMV
// -----------------------------------------------------------------------------
void DistributedFixedSzOperator::apply_local_(const Complex* v_local,
                                              Complex* y_local) const {
    Complex* const send_buf = send_buf_.data();
    for (int k = 0; k < total_send_; ++k) {
        send_buf[k] = v_local[send_local_idx_[k]];
    }

    Complex* const recv_buf = recv_buf_.data();
    MPI_Alltoallv(send_buf,
                  send_counts_.data(), send_displs_.data(), kComplexDatatype,
                  recv_buf,
                  recv_counts_.data(), recv_displs_.data(), kComplexDatatype,
                  comm_);

    auto get_v = [&](std::uint64_t c_idx) -> Complex {
        if (c_idx >= local_offset_ && c_idx < local_offset_ + local_n_) {
            return v_local[c_idx - local_offset_];
        }
        const std::size_t idx = recv_lookup_.find(c_idx);
        assert(idx != ed::core::SortedUint64Index::kNotFound &&
               "DistributedFixedSzOperator: lookup miss");
        return recv_buf[idx];
    };

    op_->commitPendingTransforms();

    std::fill(y_local, y_local + local_n_, Complex(0.0, 0.0));

    const double spin_l = static_cast<double>(op_->getSpin());

    // Build a FixedSzBasisPolicy view onto the parent's basis tables.
    const ed::matvec::basis::FixedSzBasisPolicy basis =
        ed::matvec::basis::make_fixed_sz_basis(
            op_->getBasisStates(), op_->lin_index_table());

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 4096)
#endif
    for (std::uint64_t r_local = 0; r_local < local_n_; ++r_local) {
        const std::uint64_t r_idx = local_offset_ + r_local;
        y_local[r_local] = ed::matvec::kernel::gather_row_basis<
            ed::matvec::basis::FixedSzBasisPolicy, Complex>(
                r_idx, v_local[r_local], basis,
                op_->terms_, spin_l, get_v);
    }
}

// -----------------------------------------------------------------------------
// Wave 4 (May 2026, "Unify all 16 matvec cells" plan): real-arithmetic
// overrides. Delegates to the underlying serial FixedSzOperator's
// ``isReal()`` real-coefficient check.
// -----------------------------------------------------------------------------
bool DistributedFixedSzOperator::is_real_hermitian() const noexcept {
    try {
        return op_ && const_cast<FixedSzOperator&>(*op_).isReal();
    } catch (...) {
        return false;
    }
}

}  // namespace ed::distributed

#endif  // WITH_MPI

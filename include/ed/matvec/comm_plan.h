#pragma once
// =============================================================================
// include/ed/matvec/comm_plan.h
//
// CommPlan: a reusable, policy-agnostic halo-exchange schedule for the
// distributed matrix-free SpMV. P4 of the operator-collapse refactor
// (Jun 2026).
//
// Today the MPI_Alltoallv halo plan lives inline inside
// ``ed::distributed::DistributedOperator`` (src/distributed/
// distributed_operator.cpp, ``build_comm_pattern_``). That class is one of
// the ~13 bespoke operator types the refactor collapses into a single
// ``Operator<BasisPolicy, MemSpace>``. The MemSpace=DistributedHost lane of
// that template needs the SAME comm-plan logic, but driven by a
// ``DistributedBasisPolicy<InnerPolicy>`` rather than hard-wired to the
// full-Hilbert 1D row slab.
//
// ``CommPlan`` is that logic, lifted out verbatim and made standalone:
//
//   * ``build(comm, global_dim, flip_patterns)`` performs the balanced 1D
//     slab decomposition, enumerates every off-rank column each local row
//     touches (row XOR pattern), and runs the two-phase
//     MPI_Alltoall(counts) + MPI_Alltoallv(global indices) exchange to
//     resolve the send/recv schedule -- byte-for-byte the same algorithm
//     ``DistributedOperator::build_comm_pattern_`` uses.
//
//   * ``exchange<Scalar>(local_in, recv_buf)`` performs ONE MPI_Alltoallv of
//     ``Scalar`` values (complex OR real -- the datatype is dispatched at
//     compile time) using the resolved schedule. This is the per-apply hot
//     path; ``recv_buf`` is sized to ``total_recv()`` and filled with the
//     remote columns in the plan's canonical order.
//
//   * ``recv_index_of(global_col)`` maps a remote global column to its
//     position in ``recv_buf`` (O(log) via SortedUint64Index), so the
//     GATHER-form kernel can read off-rank amplitudes.
//
// ``CommPlan`` is deliberately free of any operator / term-storage / basis
// dependency: the caller passes the flip-pattern set (a free helper
// ``flip_patterns_from_terms`` extracts it from a ``TermStorage``). This
// keeps it reusable for the eventual ``MpiMatVecImpl<BasisPolicy>`` without
// pulling the heavy operator headers into the distributed lane.
//
// Scope note (mirrors distributed_operator.h): the balanced 1D row slab is
// the bounded-regime decomposition. Symmetry-aware slabbing (keep each
// fixed-Sz orbit on one rank) is the honest-40 follow-up and is orthogonal
// to this plan's ABI.
// =============================================================================

#ifdef WITH_MPI

#include <mpi.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <ed/core/sorted_uint64_index.h>
#include <ed/matvec/term_storage.h>

namespace ed::matvec::dist {

// ---------------------------------------------------------------------------
// MPI datatype dispatch for the value halo. std::complex<double> is
// layout-compatible with C99 _Complex double (MPI 3.0 §3.4), so
// MPI_C_DOUBLE_COMPLEX is the correct handle. OpenMPI's predefined handles
// are not constexpr, hence the function form resolved at first use.
// ---------------------------------------------------------------------------
template <class Scalar>
inline MPI_Datatype comm_plan_mpi_datatype() noexcept;

template <>
inline MPI_Datatype comm_plan_mpi_datatype<double>() noexcept {
    return MPI_DOUBLE;
}
template <>
inline MPI_Datatype comm_plan_mpi_datatype<std::complex<double>>() noexcept {
    return MPI_C_DOUBLE_COMPLEX;
}

// ---------------------------------------------------------------------------
// flip_patterns_from_terms: the unique set { p : column == row XOR p } a
// Hamiltonian's SoA term storage induces. Identical extraction to
// ``DistributedOperator::extract_flip_patterns_`` -- each term type has a
// fixed pattern shape:
//
//   diag_one_body    : p = 0
//   offdiag_one_body : p = 1 << site_index
//   diag_two_body    : p = 0
//   mixed_two_body   : p = 1 << flip_site
//   offdiag_two_body : p = (1 << site_index_1) ^ (1 << site_index_2)
//   three_body       : p = XOR of (1 << site_index_k) for op_type_k != 2
//
// p == 0 (the diagonal "do nothing") is always present and structural.
// ---------------------------------------------------------------------------
inline std::vector<std::uint64_t>
flip_patterns_from_terms(const TermStorage& terms) {
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(64);
    seen.insert(0);  // structural diagonal

    for (const auto& t : terms.offdiag_one_body) {
        seen.insert(1ULL << t.site_index);
    }
    for (const auto& t : terms.mixed_two_body) {
        seen.insert(1ULL << t.flip_site);
    }
    for (const auto& t : terms.offdiag_two_body) {
        seen.insert((1ULL << t.site_index_1) ^ (1ULL << t.site_index_2));
    }
    for (const auto& t : terms.three_body) {
        std::uint64_t p = 0;
        if (t.op_type_1 != 2) p ^= (1ULL << t.site_index_1);
        if (t.op_type_2 != 2) p ^= (1ULL << t.site_index_2);
        if (t.op_type_3 != 2) p ^= (1ULL << t.site_index_3);
        seen.insert(p);
    }

    std::vector<std::uint64_t> patterns(seen.begin(), seen.end());
    std::sort(patterns.begin(), patterns.end());
    return patterns;
}

// ---------------------------------------------------------------------------
// CommPlan
// ---------------------------------------------------------------------------
class CommPlan {
public:
    CommPlan() = default;

    // -------- balanced 1D row slab (collective-friendly static helper) ------
    // First (global_dim mod size) ranks get ceil(global_dim/size) rows, the
    // rest get floor. Matches DistributedOperator::balanced_slab exactly.
    static void balanced_slab(std::uint64_t global_dim, int rank, int size,
                              std::uint64_t& out_offset,
                              std::uint64_t& out_n) noexcept {
        const std::uint64_t sz   = static_cast<std::uint64_t>(size);
        const std::uint64_t base = global_dim / sz;
        const std::uint64_t rem  = global_dim % sz;
        const std::uint64_t r    = static_cast<std::uint64_t>(rank);
        if (r < rem) {
            out_n      = base + 1;
            out_offset = r * (base + 1);
        } else {
            out_n      = base;
            out_offset = rem * (base + 1) + (r - rem) * base;
        }
    }

    // -------- collective construction --------------------------------------
    // Every rank in ``comm`` must call with the SAME global_dim and the SAME
    // flip-pattern set (i.e. the same Hamiltonian definition).
    void build(MPI_Comm comm, std::uint64_t global_dim,
               const std::vector<std::uint64_t>& flip_patterns) {
        comm_       = comm;
        global_dim_ = global_dim;
        flip_patterns_ = flip_patterns;
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);

        balanced_slab(global_dim_, rank_, size_, local_offset_, local_n_);

        rank_offsets_.assign(static_cast<std::size_t>(size_) + 1, 0);
        for (int r = 0; r < size_; ++r) {
            std::uint64_t off, n;
            balanced_slab(global_dim_, r, size_, off, n);
            rank_offsets_[r] = off;
        }
        rank_offsets_[size_] = global_dim_;

        build_comm_pattern_();
    }

    // -------- owner / locality queries -------------------------------------
    [[nodiscard]] int owner_rank(std::uint64_t global_idx) const noexcept {
        auto it = std::upper_bound(rank_offsets_.begin(), rank_offsets_.end(),
                                   global_idx);
        return static_cast<int>(std::distance(rank_offsets_.begin(), it)) - 1;
    }
    [[nodiscard]] bool is_local(std::uint64_t global_idx) const noexcept {
        return global_idx >= local_offset_
            && global_idx <  local_offset_ + local_n_;
    }

    // -------- geometry accessors -------------------------------------------
    [[nodiscard]] std::uint64_t global_dim()   const noexcept { return global_dim_; }
    [[nodiscard]] std::uint64_t local_offset() const noexcept { return local_offset_; }
    [[nodiscard]] std::uint64_t local_n()      const noexcept { return local_n_; }
    [[nodiscard]] int           rank()         const noexcept { return rank_; }
    [[nodiscard]] int           size()         const noexcept { return size_; }
    [[nodiscard]] MPI_Comm      comm()         const noexcept { return comm_; }
    [[nodiscard]] int           total_send()   const noexcept { return total_send_; }
    [[nodiscard]] int           total_recv()   const noexcept { return total_recv_; }
    [[nodiscard]] std::size_t   num_flip_patterns() const noexcept {
        return flip_patterns_.size();
    }

    // -------- recv-buffer index of a remote global column ------------------
    // Returns SortedUint64Index::kNotFound when ``global_col`` is not in the
    // halo (caller should then read it from the local slab).
    [[nodiscard]] std::size_t recv_index_of(std::uint64_t global_col) const noexcept {
        return recv_lookup_.find(global_col);
    }

    // -------- per-apply halo exchange (the hot path) -----------------------
    // local_in : length local_n() amplitudes owned by this rank.
    // recv_buf : resized to total_recv(); on return holds the remote columns
    //            this rank requested, in the plan's canonical order (so
    //            ``recv_index_of`` indexes directly into it).
    template <class Scalar>
    void exchange(const Scalar* local_in, std::vector<Scalar>& recv_buf) const {
        const MPI_Datatype dt = comm_plan_mpi_datatype<Scalar>();
        std::vector<Scalar> send_buf(static_cast<std::size_t>(total_send_));
        for (int k = 0; k < total_send_; ++k) {
            send_buf[static_cast<std::size_t>(k)] =
                local_in[static_cast<std::size_t>(send_local_idx_[k])];
        }
        recv_buf.assign(static_cast<std::size_t>(total_recv_), Scalar{});
        MPI_Alltoallv(send_buf.data(),
                      send_counts_.data(), send_displs_.data(), dt,
                      recv_buf.data(),
                      recv_counts_.data(), recv_displs_.data(), dt,
                      comm_);
    }

private:
    static int as_mpi_int(std::size_t n, const char* what) {
        if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(std::string("CommPlan: ") + what
                + " count " + std::to_string(n) + " exceeds INT_MAX.");
        }
        return static_cast<int>(n);
    }

    void build_comm_pattern_() {
        const int S = size_;
        std::vector<std::vector<std::uint64_t>> recv_global_per_rank(
            static_cast<std::size_t>(S));

        // -- pass 1: enumerate off-rank columns -----------------------------
        for (std::uint64_t r_local = 0; r_local < local_n_; ++r_local) {
            const std::uint64_t r = local_offset_ + r_local;
            for (std::uint64_t p : flip_patterns_) {
                const std::uint64_t c = r ^ p;
                if (c >= global_dim_) continue;
                if (c >= local_offset_ && c < local_offset_ + local_n_) continue;
                recv_global_per_rank[static_cast<std::size_t>(owner_rank(c))]
                    .push_back(c);
            }
        }

        // -- dedupe + sort each bucket --------------------------------------
        recv_counts_.assign(S, 0);
        recv_displs_.assign(S, 0);
        for (int r = 0; r < S; ++r) {
            auto& bucket = recv_global_per_rank[static_cast<std::size_t>(r)];
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

        // -- exchange counts ------------------------------------------------
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

        // -- exchange the global indices we want from each peer -------------
        std::vector<std::uint64_t> recv_global_concat;
        recv_global_concat.reserve(static_cast<std::size_t>(total_recv_));
        for (int r = 0; r < S; ++r) {
            recv_global_concat.insert(
                recv_global_concat.end(),
                recv_global_per_rank[static_cast<std::size_t>(r)].begin(),
                recv_global_per_rank[static_cast<std::size_t>(r)].end());
        }

        std::vector<std::uint64_t> peer_wants_from_us(
            static_cast<std::size_t>(total_send_));
        MPI_Alltoallv(recv_global_concat.data(),
                      recv_counts_.data(), recv_displs_.data(), MPI_UINT64_T,
                      peer_wants_from_us.data(),
                      send_counts_.data(), send_displs_.data(), MPI_UINT64_T,
                      comm_);

        // -- translate peer's wanted global idx -> our local slab offset ----
        send_local_idx_.assign(static_cast<std::size_t>(total_send_), 0);
        for (int r = 0; r < S; ++r) {
            const int off = send_displs_[r];
            const int n   = send_counts_[r];
            for (int k = 0; k < n; ++k) {
                const std::uint64_t g = peer_wants_from_us[off + k];
                if (g < local_offset_ || g >= local_offset_ + local_n_) {
                    throw std::logic_error(
                        "CommPlan: peer requested out-of-slab index "
                        + std::to_string(g) + " from rank "
                        + std::to_string(rank_));
                }
                send_local_idx_[off + k] =
                    static_cast<int>(g - local_offset_);
            }
        }

        // -- build the recv-side lookup -------------------------------------
        recv_lookup_.clear();
        recv_lookup_.reserve(static_cast<std::size_t>(total_recv_));
        for (std::size_t k = 0; k < recv_global_concat.size(); ++k) {
            recv_lookup_.insert(recv_global_concat[k], k);
        }
        recv_lookup_.finalize();
    }

    MPI_Comm      comm_       = MPI_COMM_NULL;
    int           rank_       = 0;
    int           size_       = 1;
    std::uint64_t global_dim_   = 0;
    std::uint64_t local_offset_ = 0;
    std::uint64_t local_n_      = 0;

    std::vector<std::uint64_t> rank_offsets_;   // length size_+1
    std::vector<std::uint64_t> flip_patterns_;  // sorted

    std::vector<int> send_counts_;
    std::vector<int> send_displs_;
    std::vector<int> recv_counts_;
    std::vector<int> recv_displs_;
    std::vector<int> send_local_idx_;  // length total_send_

    ed::core::SortedUint64Index recv_lookup_;  // remote global col -> recv pos
    int total_send_ = 0;
    int total_recv_ = 0;
};

}  // namespace ed::matvec::dist

#endif  // WITH_MPI

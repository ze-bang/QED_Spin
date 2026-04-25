// =============================================================================
// src/distributed/orbit_halo_plan.cpp    (Phase 3b #7, stage 2 prep)
//
// Build / exchange logic for OrbitHaloPlan. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/orbit_halo_plan.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace ed::distributed {

namespace {

const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

[[noreturn]] void throw_mpi(int code, const char* where) {
    char buf[MPI_MAX_ERROR_STRING] = {0};
    int  len = 0;
    MPI_Error_string(code, buf, &len);
    throw std::runtime_error(
        std::string("OrbitHaloPlan: MPI failure in ") + where + ": " + buf);
}

inline void check_mpi(int code, const char* where) {
    if (code != MPI_SUCCESS) throw_mpi(code, where);
}

}  // namespace

OrbitHaloPlan::OrbitHaloPlan(const OrbitPartition& part,
                             const std::vector<std::size_t>& needed_orbits,
                             MPI_Comm comm)
    : comm_(comm) {

    if (comm == MPI_COMM_NULL) {
        throw std::invalid_argument("OrbitHaloPlan: comm is MPI_COMM_NULL");
    }
    check_mpi(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(comm_, &size_), "MPI_Comm_size");

    if (part.n_ranks != size_) {
        throw std::invalid_argument(
            "OrbitHaloPlan: part.n_ranks (" + std::to_string(part.n_ranks)
            + ") != comm_size (" + std::to_string(size_) + ")");
    }

    // ---- 1. Filter `needed_orbits` to only remote ones, group by owner
    // rank, then sort + dedupe per owner. Remote = not owned by this
    // rank under `part`.
    std::vector<std::vector<std::uint64_t>> remote_per_owner(
        static_cast<std::size_t>(size_));
    for (std::size_t orbit_id : needed_orbits) {
        const int owner = part.owner_rank(orbit_id);
        if (owner < 0) continue;             // out-of-range -> ignore
        if (owner == rank_) continue;        // locally owned -> no halo
        remote_per_owner[static_cast<std::size_t>(owner)].push_back(orbit_id);
    }
    for (auto& v : remote_per_owner) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // ---- 2. Build recv_counts / recv_displs / recv_orbit_id (rank-major).
    recv_counts_.assign(static_cast<std::size_t>(size_), 0);
    recv_displs_.assign(static_cast<std::size_t>(size_), 0);
    std::size_t total_recv = 0;
    for (int r = 0; r < size_; ++r) {
        const auto& v = remote_per_owner[static_cast<std::size_t>(r)];
        // INT_MAX overflow check: bounded basis for the bounded-N tests.
        if (v.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "OrbitHaloPlan: recv_count from rank " + std::to_string(r)
                + " exceeds INT_MAX");
        }
        recv_counts_[r] = static_cast<int>(v.size());
        recv_displs_[r] = static_cast<int>(total_recv);
        total_recv += v.size();
    }
    recv_orbit_id_.resize(total_recv);
    for (int r = 0; r < size_; ++r) {
        const auto& v = remote_per_owner[static_cast<std::size_t>(r)];
        std::copy(v.begin(), v.end(),
                  recv_orbit_id_.begin() + recv_displs_[r]);
    }

    // ---- 3. Tell every other rank how many orbits we expect from them.
    // After this, send_counts_[r] = "rank r told me it wants this many
    // orbits FROM me", which is exactly the count I owe rank r.
    send_counts_.assign(static_cast<std::size_t>(size_), 0);
    check_mpi(MPI_Alltoall(recv_counts_.data(), 1, MPI_INT,
                           send_counts_.data(), 1, MPI_INT, comm_),
              "MPI_Alltoall(counts)");
    std::size_t total_send = 0;
    send_displs_.assign(static_cast<std::size_t>(size_), 0);
    for (int r = 0; r < size_; ++r) {
        send_displs_[r] = static_cast<int>(total_send);
        total_send += static_cast<std::size_t>(send_counts_[r]);
    }
    send_orbit_id_.resize(total_send);
    send_local_idx_.resize(total_send);

    // ---- 4. Tell every other rank WHICH orbits we want; receive their
    // requests as the things we need to send back. recv-side orbit_ids
    // are already sorted ascending per-rank (step 1 dedup), and the
    // send-side orbit_ids will be sorted per source-rank too because
    // peer A sends its sorted "I want" list to us and we receive it as
    // "I owe these to A".
    check_mpi(MPI_Alltoallv(recv_orbit_id_.data(),
                            recv_counts_.data(), recv_displs_.data(),
                            MPI_UINT64_T,
                            send_orbit_id_.data(),
                            send_counts_.data(), send_displs_.data(),
                            MPI_UINT64_T, comm_),
              "MPI_Alltoallv(orbit_ids)");

    // ---- 5. For every send slot, resolve the orbit's local index on
    // this rank (constant-time via OrbitPartition::owner_local_index).
    for (std::size_t k = 0; k < send_orbit_id_.size(); ++k) {
        const std::uint64_t orbit = send_orbit_id_[k];
        const int owner = part.owner_rank(orbit);
        if (owner != rank_) {
            // Defensive: peer asked us for an orbit we don't own. Should
            // never happen if every rank built `part` deterministically
            // from the same inputs.
            throw std::runtime_error(
                "OrbitHaloPlan: peer requested orbit "
                + std::to_string(orbit) + " not owned by this rank "
                + std::to_string(rank_) + " (owner=" + std::to_string(owner)
                + ")");
        }
        send_local_idx_[k] = part.owner_local_index(orbit);
    }
}

void OrbitHaloPlan::exchange(const Complex* local_amplitudes,
                             Complex* halo_amplitudes) const {
    // Pack send buffer: send_buf[k] = local_amplitudes[send_local_idx_[k]].
    std::vector<Complex> send_buf(send_orbit_id_.size(), Complex(0.0, 0.0));
    for (std::size_t k = 0; k < send_orbit_id_.size(); ++k) {
        send_buf[k] = local_amplitudes[send_local_idx_[k]];
    }

    // Single Alltoallv of complex<double>. recv side is already sized
    // by the caller (recv_total()).
    check_mpi(MPI_Alltoallv(send_buf.data(),
                            send_counts_.data(), send_displs_.data(),
                            kComplexDatatype,
                            halo_amplitudes,
                            recv_counts_.data(), recv_displs_.data(),
                            kComplexDatatype, comm_),
              "MPI_Alltoallv(amplitudes)");
}

std::size_t OrbitHaloPlan::plan_bytes() const noexcept {
    return sizeof(int) * (send_counts_.capacity() + send_displs_.capacity()
                          + recv_counts_.capacity() + recv_displs_.capacity())
         + sizeof(std::uint64_t) * (send_orbit_id_.capacity()
                                    + recv_orbit_id_.capacity())
         + sizeof(std::size_t) * send_local_idx_.capacity();
}

}  // namespace ed::distributed

#endif  // WITH_MPI

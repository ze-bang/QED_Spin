#pragma once

// Phase 3b (bootstrap): MPI-parallel matrix-free H·v on a 1D row decomposition
// of the computational basis. Each rank owns a contiguous row slab of the
// output vector; input is assembled from per-rank slices via MPI_Allgatherv.
//
// Memory note: every rank still holds a full replica of v in the workspace
// after gather_input() — this bootstrap step validates correctness and row
// slicing (Operator::apply_row_range) before replacing the Allgatherv with a
// sparse halo exchange (next Phase 3b milestone).

#include <cstdint>
#include <vector>

#include <ed/core/construct_ham.h>

#if defined(WITH_MPI) && WITH_MPI

#include <mpi.h>

namespace ed::distributed {

class DistributedMatrixFreeOperator {
public:
    explicit DistributedMatrixFreeOperator(Operator& op,
                                           MPI_Comm comm = MPI_COMM_WORLD);

    uint64_t global_dim() const { return global_dim_; }
    int mpi_rank() const { return rank_; }
    int mpi_size() const { return size_; }
    uint64_t local_row_offset() const { return local_row_offset_; }
    uint64_t local_row_count() const { return local_row_count_; }

    /// Reconstruct the full input vector from per-rank contiguous slices
    /// (same row partitioning as local matvec ownership).
    void gather_input(const Complex* v_local, std::vector<Complex>& v_full) const;

    /// v_full_workspace is resized to global_dim(); y_local must have length
    /// local_row_count() (zeroed by caller is optional; apply_row_range zeros
    /// internally before accumulating).
    void apply(const Complex* v_local, std::vector<Complex>& v_full_workspace,
               Complex* y_local) const;

private:
    Operator* op_;
    MPI_Comm comm_;
    int rank_ = 0;
    int size_ = 1;
    uint64_t global_dim_ = 0;
    uint64_t local_row_offset_ = 0;
    uint64_t local_row_count_ = 0;
};

}  // namespace ed::distributed

#endif  // WITH_MPI

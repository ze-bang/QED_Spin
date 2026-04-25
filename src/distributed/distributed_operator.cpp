#if defined(WITH_MPI) && WITH_MPI

#include <ed/distributed/distributed_operator.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ed::distributed {

namespace {

void balanced_row_split(uint64_t dim, int rank, int size,
                        uint64_t& offset, uint64_t& count) {
    if (size <= 0) {
        throw std::invalid_argument("balanced_row_split: invalid MPI size");
    }
    const uint64_t sz = static_cast<uint64_t>(size);
    const uint64_t q = dim / sz;
    const uint64_t r = dim % sz;
    const uint64_t ur = static_cast<uint64_t>(rank);
    count = q + (ur < r ? 1u : 0u);
    offset = ur * q + std::min(ur, r);
}

}  // namespace

DistributedMatrixFreeOperator::DistributedMatrixFreeOperator(Operator& op,
                                                             MPI_Comm comm)
    : op_(&op), comm_(comm) {
    int init = 0;
    MPI_Initialized(&init);
    if (!init) {
        throw std::runtime_error(
            "DistributedMatrixFreeOperator: MPI_Init must be called first");
    }
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    global_dim_ = 1ULL << op_->getNumBits();
    balanced_row_split(global_dim_, rank_, size_, local_row_offset_,
                       local_row_count_);
}

void DistributedMatrixFreeOperator::gather_input(
    const Complex* v_local, std::vector<Complex>& v_full) const {

    v_full.resize(global_dim_);

    std::vector<int> counts(size_);
    std::vector<int> displs(size_);
    for (int r = 0; r < size_; ++r) {
        uint64_t off = 0, cnt = 0;
        balanced_row_split(global_dim_, r, size_, off, cnt);
        if (cnt > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("gather_input: local slice exceeds INT_MAX");
        }
        counts[r] = static_cast<int>(cnt);
        displs[r] = static_cast<int>(off);
    }

    static_assert(sizeof(Complex) == 16, "MPI BYTE packing assumes 16B complex");
    std::vector<int> counts_bytes(size_);
    std::vector<int> displs_bytes(size_);
    for (int r = 0; r < size_; ++r) {
        counts_bytes[r] = counts[r] * static_cast<int>(sizeof(Complex));
        displs_bytes[r] = displs[r] * static_cast<int>(sizeof(Complex));
    }

    MPI_Allgatherv(v_local, static_cast<int>(local_row_count_ * sizeof(Complex)),
                   MPI_BYTE, v_full.data(), counts_bytes.data(), displs_bytes.data(),
                   MPI_BYTE, comm_);
}

void DistributedMatrixFreeOperator::apply(
    const Complex* v_local, std::vector<Complex>& v_full_workspace,
    Complex* y_local) const {
    gather_input(v_local, v_full_workspace);
    op_->apply_row_range(v_full_workspace.data(), y_local, global_dim_,
                        local_row_offset_, local_row_count_);
}

}  // namespace ed::distributed

#endif  // WITH_MPI

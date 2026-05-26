#pragma once
// =============================================================================
// include/ed/matvec/distributed_basis_policy.h
//
// DistributedBasisPolicy<InnerPolicy>. Wave 2 of the
// "Unify all 16 matvec cells under apply_terms<BasisPolicy, Scalar, Backend>"
// plan (May 2026).
//
// A DistributedBasisPolicy wraps any other BasisPolicy and adds slab
// metadata so the SCATTER / GATHER kernels and the comm-pattern build
// can distinguish "local row" from "remote row" without changing the
// underlying ``state_of`` / ``index_of`` semantics.
//
// Slab convention (matches DistributedOperator):
//
//   * Every rank owns a contiguous range of GLOBAL array indices
//     ``[local_offset_, local_offset_ + local_dim_)``.
//   * ``state_of(global_idx)`` and ``index_of(state)`` delegate to the
//     inner policy. The inner policy is responsible for the
//     (bitstring <-> global_array_idx) mapping; the distributed
//     wrapper only annotates which slice of the global index range we
//     own.
//
// The kernel ``apply_terms`` interprets ``is_distributed = true`` as a
// signal that the SCATTER form may emit into off-rank rows -- those
// cases must instead use the GATHER kernel (``gather_row_basis``) on
// each rank. Today the SCATTER body still operates row-locally; the
// distributed-aware emit / off-rank routing lives in the operator
// classes (``DistributedOperator``, etc.). The trait is provided here
// for forward compatibility so future kernel changes can branch on it
// without an ABI break.
// =============================================================================

#include <cstdint>
#include <utility>

#include <ed/matvec/basis_policy.h>

namespace ed::matvec::basis {

template <class InnerPolicy>
struct DistributedBasisPolicy {
    InnerPolicy    inner;
    std::uint64_t  local_offset_;
    std::uint64_t  local_dim_;
    std::uint64_t  global_dim_;

    // -------- delegated bitstring <-> index mapping -------------------
    [[nodiscard]] inline std::uint64_t dim() const noexcept {
        return local_dim_;
    }
    [[nodiscard]] inline std::uint64_t global_dim() const noexcept {
        return global_dim_;
    }
    [[nodiscard]] inline std::uint64_t state_of(std::uint64_t local_idx) const noexcept {
        // local_idx is interpreted as a position in our slab. We
        // delegate ``state_of`` on the GLOBAL index.
        return inner.state_of(local_offset_ + local_idx);
    }
    [[nodiscard]] inline std::int64_t index_of(std::uint64_t state) const noexcept {
        // Returns the GLOBAL array index. The caller (kernel) decides
        // whether to route it locally or via the halo plan.
        return inner.index_of(state);
    }

    // -------- distributed trait + helpers ------------------------------
    static constexpr bool may_leave_basis    = InnerPolicy::may_leave_basis;
    static constexpr bool needs_orbit_walk   = InnerPolicy::needs_orbit_walk;
    static constexpr bool has_coeff_modifier = InnerPolicy::has_coeff_modifier;
    static constexpr bool is_distributed     = true;

    template <class Callback>
    inline void iter_orbit(std::uint64_t local_idx, Callback&& cb) const {
        inner.iter_orbit(local_offset_ + local_idx, std::forward<Callback>(cb));
    }

    template <class Scalar>
    [[nodiscard]] inline Scalar coeff_modifier(std::uint64_t s,
                                               std::uint64_t s_prime,
                                               std::uint64_t src_idx,
                                               std::uint64_t dst_idx) const noexcept
    {
        return inner.template coeff_modifier<Scalar>(s, s_prime, src_idx, dst_idx);
    }

    [[nodiscard]] inline bool is_local(std::uint64_t global_idx) const noexcept {
        return global_idx >= local_offset_
            && global_idx <  local_offset_ + local_dim_;
    }
    [[nodiscard]] inline std::uint64_t local_offset() const noexcept {
        return local_offset_;
    }
};

// Convenience factory.
template <class InnerPolicy>
[[nodiscard]] inline DistributedBasisPolicy<InnerPolicy>
make_distributed_basis(InnerPolicy inner,
                       std::uint64_t local_offset,
                       std::uint64_t local_dim,
                       std::uint64_t global_dim) noexcept
{
    return DistributedBasisPolicy<InnerPolicy>{
        std::move(inner), local_offset, local_dim, global_dim,
    };
}

} // namespace ed::matvec::basis

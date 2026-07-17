#pragma once
// =============================================================================
// include/ed/symmetry/subspace.h
//
// Subspace: which computational basis states make up a given sector.
//
// Part of the "Orthogonal symmetry composition" wave (May 2026). A Subspace
// is the FIRST orthogonal axis of the new (Subspace x ProjectorChain)
// decomposition that supersedes the four-mode (none / Sz / Symm /
// Sz+Symm) taxonomy:
//
//   * FullSpaceSubspace : every length-N bitstring (dim = 2^N).
//   * FixedSzSubspace   : every bitstring with popcount = n_up
//                         (dim = C(N, n_up)).
//   * (future) FixedS2Subspace : SU(2) total-S = s coupled basis.
//   * (future) any other state-only filter (particle number,
//     sublattice grading, ...).
//
// A Subspace OWNS the description of its state list. For full-space
// the description is just ``n_bits``; for fixed-Sz it is the sorted
// state vector plus a Lin (1990) O(1) index table. The two POD
// "basis-policy" view types in ``ed/matvec/basis_policy.h``
// (``FullBasisPolicy`` / ``FixedSzBasisPolicy``) remain unchanged --
// they are the value-type _views_ that the unified matvec kernels
// (``ed::matvec::kernel::apply_terms``) consume. ``Subspace::policy()``
// returns those views directly.
//
// Why we need an owning abstraction even though the POD views already
// work: the next wave (``ProjectorChain``) builds orbits by walking
// the subspace's state list during host-side sector construction.
// That construction needs ownership semantics (especially for the
// FixedSz case, which carries ~ C(N, n_up) sorted states + a 24-byte
// LinIndexTable per sector); the POD policies are non-owning and
// therefore cannot drive sector generation by themselves.
//
// ``FixedSzOperator`` exposes a ``subspace()`` method returning a
// non-owning ``FixedSzSubspace`` view over its existing fields (the
// legacy monolithic streaming operators are gone). Future refactors may
// migrate ownership in here without disturbing the new abstraction's
// public surface.
//
// =============================================================================

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ed/core/basis_utils.h>   // LinIndexTable, generateFixedSzBasis
#include <ed/core/combinadic.h>    // BinomialTable (tableless fixed-Sz mode)
#include <ed/matvec/basis_policy.h>

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// FullSpaceSubspace -- every length-N bitstring is in the basis.
//
// dim = 2^N; state_of(i) = i; index_of(s) = s for s < 2^N else -1.
// Holds only ``n_bits``. Trivially copyable.
// ---------------------------------------------------------------------------
class FullSpaceSubspace {
public:
    explicit FullSpaceSubspace(std::uint64_t n_bits) : n_bits_(n_bits) {
        if (n_bits >= 64) {
            throw std::invalid_argument(
                "FullSpaceSubspace: n_bits >= 64 is not supported "
                "(1ULL << n_bits would overflow).");
        }
    }

    [[nodiscard]] std::uint64_t n_bits() const noexcept { return n_bits_; }

    [[nodiscard]] std::uint64_t dim() const noexcept {
        return 1ULL << n_bits_;
    }

    [[nodiscard]] std::uint64_t state_of(std::uint64_t idx) const noexcept {
        return idx;
    }

    [[nodiscard]] std::int64_t index_of(std::uint64_t state) const noexcept {
        return state < dim() ? static_cast<std::int64_t>(state)
                             : std::int64_t{-1};
    }

    /// Returns the POD view consumed by ``ed::matvec::kernel::apply_terms``.
    /// ``FullBasisPolicy`` is a non-owning value type holding only
    /// ``n_bits``, so it stays valid as long as this subspace value is
    /// reachable.
    [[nodiscard]] ed::matvec::basis::FullBasisPolicy policy() const noexcept {
        return ed::matvec::basis::make_full_basis(n_bits_);
    }

    // ABI alignment with ed::matvec::basis::* policies: a Subspace can
    // be passed directly to algorithms that already accept those POD
    // views via the same trait-check pattern.
    static constexpr bool may_leave_basis    = false;
    static constexpr bool needs_orbit_walk   = false;
    static constexpr bool has_coeff_modifier = false;
    static constexpr bool is_distributed     = false;

private:
    std::uint64_t n_bits_ = 0;
};

// ---------------------------------------------------------------------------
// FixedSzSubspace -- only bitstrings with popcount = n_up.
//
// Holds the sorted state list and a Lin (1990) O(1) state->index table.
// Two flavours of construction:
//
//   * ``FixedSzSubspace::build(n_bits, n_up)``    -- owning, allocates
//                                                    the sorted basis
//                                                    + Lin table.
//   * ``FixedSzSubspace::view(n_bits, n_up,
//                              const vector<uint64_t>& basis,
//                              const LinIndexTable& lin)``
//                                                  -- non-owning view
//                                                    over storage held
//                                                    elsewhere (used by
//                                                    ``FixedSzOperator``
//                                                    to expose its
//                                                    internal storage).
// ---------------------------------------------------------------------------
class FixedSzSubspace {
public:
    FixedSzSubspace() = default;

    // ------------------------------------------------------------------
    // Copy / move with pointer re-homing (operator-collapse Phase 4).
    //
    // A built (owning) FixedSzSubspace stores its tables in
    // ``owned_basis_`` / ``owned_lin_`` and points ``basis_ptr_`` /
    // ``lin_ptr_`` at them. A naive default copy/move would copy the
    // vectors but leave the pointers aimed at the SOURCE's storage,
    // dangling once the source dies. Now that ``SubspaceOperator`` holds
    // a FixedSzSubspace as an owned member (and that operator is copyable
    // -- e.g. ``std::vector<Operator>`` in the DSSF spec), we re-home the
    // pointers to ``this`` on copy/move whenever the source was owning.
    // View-only subspaces (pointers aimed at external storage) copy the
    // external pointers verbatim, exactly as before.
    // ------------------------------------------------------------------
    FixedSzSubspace(const FixedSzSubspace& o)
        : n_bits_(o.n_bits_), n_up_(o.n_up_),
          tableless_(o.tableless_), tableless_dim_(o.tableless_dim_),
          owned_basis_(o.owned_basis_), owned_lin_(o.owned_lin_),
          owned_binom_(o.owned_binom_) {
        rehome_from_(o);
    }
    FixedSzSubspace(FixedSzSubspace&& o) noexcept
        : n_bits_(o.n_bits_), n_up_(o.n_up_),
          tableless_(o.tableless_), tableless_dim_(o.tableless_dim_),
          owned_basis_(std::move(o.owned_basis_)),
          owned_lin_(std::move(o.owned_lin_)),
          owned_binom_(std::move(o.owned_binom_)) {
        rehome_from_(o);
    }
    FixedSzSubspace& operator=(const FixedSzSubspace& o) {
        if (this != &o) {
            n_bits_      = o.n_bits_;
            n_up_        = o.n_up_;
            tableless_   = o.tableless_;
            tableless_dim_ = o.tableless_dim_;
            owned_basis_ = o.owned_basis_;
            owned_lin_   = o.owned_lin_;
            owned_binom_ = o.owned_binom_;
            rehome_from_(o);
        }
        return *this;
    }
    FixedSzSubspace& operator=(FixedSzSubspace&& o) noexcept {
        if (this != &o) {
            n_bits_      = o.n_bits_;
            n_up_        = o.n_up_;
            tableless_   = o.tableless_;
            tableless_dim_ = o.tableless_dim_;
            owned_basis_ = std::move(o.owned_basis_);
            owned_lin_   = std::move(o.owned_lin_);
            owned_binom_ = std::move(o.owned_binom_);
            rehome_from_(o);
        }
        return *this;
    }

    /// Owning build: enumerates basis states with popcount = n_up and
    /// builds the Lin index table.
    [[nodiscard]] static FixedSzSubspace
    build(std::uint64_t n_bits, std::int64_t n_up) {
        if (n_bits >= 64) {
            throw std::invalid_argument(
                "FixedSzSubspace: n_bits >= 64 is not supported.");
        }
        if (n_up < 0 || n_up > static_cast<std::int64_t>(n_bits)) {
            throw std::invalid_argument(
                "FixedSzSubspace: n_up must satisfy 0 <= n_up <= n_bits.");
        }

        FixedSzSubspace s;
        s.n_bits_       = n_bits;
        s.n_up_         = n_up;
        s.owned_basis_  = generateFixedSzBasis(n_bits, n_up);
        s.owned_lin_.build(n_bits, n_up, s.owned_basis_);
        s.basis_ptr_    = &s.owned_basis_;
        s.lin_ptr_      = &s.owned_lin_;
        return s;
    }

    /// Tableless combinadic build (Track A): no C(N,n_up) basis vector and no
    /// Lin table -- only an O(N^2) BinomialTable. ``state_of`` / ``index_of`` /
    /// ``policy`` go through combinadic rank/unrank. Use when the materialized
    /// basis would not fit (the planner selects this). ``basis_states()`` /
    /// ``lin_index()`` MUST NOT be called in this mode (no materialized arrays).
    [[nodiscard]] static FixedSzSubspace
    build_tableless(std::uint64_t n_bits, std::int64_t n_up) {
        if (n_bits >= 64) {
            throw std::invalid_argument(
                "FixedSzSubspace: n_bits >= 64 is not supported.");
        }
        if (n_up < 0 || n_up > static_cast<std::int64_t>(n_bits)) {
            throw std::invalid_argument(
                "FixedSzSubspace: n_up must satisfy 0 <= n_up <= n_bits.");
        }
        FixedSzSubspace s;
        s.n_bits_        = n_bits;
        s.n_up_          = n_up;
        s.tableless_     = true;
        s.owned_binom_.resize(static_cast<int>(n_bits));
        s.tableless_dim_ = s.owned_binom_.at(static_cast<int>(n_bits),
                                             static_cast<int>(n_up));
        return s;
    }

    /// Non-owning view: ``basis`` + ``lin`` must outlive this subspace.
    /// Used by ``FixedSzOperator::subspace()`` so the operator keeps
    /// ownership of its tables and the Subspace abstraction is a thin
    /// observation layer on top.
    [[nodiscard]] static FixedSzSubspace
    view(std::uint64_t n_bits,
         std::int64_t  n_up,
         const std::vector<std::uint64_t>& basis,
         const LinIndexTable& lin) noexcept
    {
        FixedSzSubspace s;
        s.n_bits_    = n_bits;
        s.n_up_      = n_up;
        s.basis_ptr_ = &basis;
        s.lin_ptr_   = &lin;
        return s;
    }

    [[nodiscard]] std::uint64_t n_bits() const noexcept { return n_bits_; }
    [[nodiscard]] std::int64_t  n_up()   const noexcept { return n_up_;   }

    [[nodiscard]] bool is_tableless() const noexcept { return tableless_; }
    [[nodiscard]] const ed::core::combinadic::BinomialTable& binom() const noexcept {
        return owned_binom_;
    }

    [[nodiscard]] std::uint64_t dim() const noexcept {
        if (tableless_) return tableless_dim_;
        return basis_ptr_ != nullptr ? basis_ptr_->size() : 0;
    }

    [[nodiscard]] std::uint64_t state_of(std::uint64_t idx) const noexcept {
        if (tableless_)
            return ed::core::combinadic::unrank_to_state(
                idx, static_cast<int>(n_bits_), static_cast<int>(n_up_), owned_binom_);
        return (*basis_ptr_)[idx];
    }

    [[nodiscard]] std::int64_t index_of(std::uint64_t state) const noexcept {
        if (tableless_) {
            return (__builtin_popcountll(state) == static_cast<int>(n_up_))
                ? ed::core::combinadic::rank_state(
                      state, static_cast<int>(n_bits_), static_cast<int>(n_up_), owned_binom_)
                : std::int64_t{-1};
        }
        return lin_ptr_->lookup(state);
    }

    [[nodiscard]] const std::vector<std::uint64_t>& basis_states() const noexcept {
        return *basis_ptr_;
    }

    [[nodiscard]] const LinIndexTable& lin_index() const noexcept {
        return *lin_ptr_;
    }

    /// POD view consumed by ``ed::matvec::kernel::apply_terms``. Backed
    /// by the same ``basis_states_`` / ``LinIndexTable`` whose stability
    /// the caller guaranteed at construction (or that this object owns
    /// when built via ``build``).
    [[nodiscard]] ed::matvec::basis::FixedSzBasisPolicy policy() const noexcept {
        if (tableless_) {
            return ed::matvec::basis::make_combinadic_fixed_sz_basis(
                static_cast<int>(n_bits_), static_cast<int>(n_up_),
                owned_binom_, tableless_dim_);
        }
        return ed::matvec::basis::make_fixed_sz_basis(*basis_ptr_, *lin_ptr_);
    }

    static constexpr bool may_leave_basis    = true;
    static constexpr bool needs_orbit_walk   = false;
    static constexpr bool has_coeff_modifier = false;
    static constexpr bool is_distributed     = false;

private:
    // Re-home the non-owning pointers after a copy/move from ``o``: if the
    // source pointed at its own ``owned_*`` storage we point at OUR copies;
    // otherwise we adopt the source's external pointers verbatim.
    void rehome_from_(const FixedSzSubspace& o) noexcept {
        if (o.basis_ptr_ == &o.owned_basis_) {
            basis_ptr_ = &owned_basis_;
            lin_ptr_   = &owned_lin_;
        } else {
            basis_ptr_ = o.basis_ptr_;
            lin_ptr_   = o.lin_ptr_;
        }
    }

    std::uint64_t n_bits_ = 0;
    std::int64_t  n_up_   = 0;
    // Tableless combinadic mode (build_tableless): no materialized basis/Lin
    // table, just owned_binom_ + the cached sector dimension.
    bool          tableless_     = false;
    std::uint64_t tableless_dim_ = 0;
    // Owned tables (populated only by ``build``); viewers initialize
    // these as empty and set the pointers below to external storage.
    std::vector<std::uint64_t> owned_basis_;
    LinIndexTable              owned_lin_;
    ed::core::combinadic::BinomialTable owned_binom_;  // tableless mode only
    // Non-owning observations -- always set to ``&owned_*`` for built
    // subspaces and to external storage for view-only ones.
    const std::vector<std::uint64_t>* basis_ptr_ = nullptr;
    const LinIndexTable*              lin_ptr_   = nullptr;
};

// ---------------------------------------------------------------------------
// Subspace "tag" enum -- runtime discriminator for code paths that want
// to take any subspace. Used by ProjectorChain (Phase R3) when building
// orbits without having to template on the concrete type.
// ---------------------------------------------------------------------------
enum class SubspaceKind : std::uint8_t {
    FullSpace = 0,
    FixedSz   = 1,
    // Future: FixedS2, Coupled, etc.
};

} // namespace ed::symmetry

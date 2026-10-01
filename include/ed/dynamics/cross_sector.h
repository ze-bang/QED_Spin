#pragma once
// =============================================================================
// include/ed/dynamics/cross_sector.h
//
// Rectangular cross-sector observable between two rep sectors.
//
// Maps a vector in the rep basis of a *source* sector (one momentum /
// little-group sector, a ``RepSectorData``) to a vector in the rep basis of
// a *target* sector on the same lattice. With symmetric kets
//
//   |psi^{k_src}_alpha> = sum_g chi^*_{k_src}(g) g |r_alpha> / norm_alpha
//
// the matrix element of an observable ``O = sum_t T_t`` (one-, two-body spin
// terms in the ``Operator::TransformData`` layout) is accumulated by walking
// the source orbit per group element, applying each term (bit-flip / Sz-sign
// action ``s -> s'``) and projecting ``s'`` into the target sector with the
// rep policy's ``index_and_projection``; ``group_norm = 1/sqrt(|G_src||G_dst|)``.
//
// Threading: OpenMP-parallel over source rep indices with per-thread
// accumulators. The rectangular matrix is cached as a CSR on first use
// (within ED_XSEC_CSR_BUDGET_GIB).
// =============================================================================

#include <ed/matvec/linear_operator.h>
#include <ed/ops/operator.h>                       // Operator::TransformData
#include <ed/basis/rep_sector.h>                   // RepSectorData
#include <ed/matvec/rep_symmetry_basis_policy.h>   // host rep policy view

#include <complex>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <vector>

namespace ed::dssf {

/// Rectangular orbit-basis observable mapping a source rep sector to a
/// target rep sector.
///
/// ``src`` and ``dst`` may wrap the same sector or different sectors on
/// the same lattice (e.g. fixed-Sz x irrep -> fixed-Sz x irrep with
/// different ``n_up``).
class CrossSectorOrbitObservable {
public:
    using Complex       = std::complex<double>;
    using TransformData = Operator::TransformData;

    /// Non-owning handle onto one CSR-free ``RepSectorData`` sector (the
    /// little-group engine's block / momentum sectors). The ref wraps
    /// exactly ONE sector (``num_sectors() == 1``), so the observable is
    /// built with ``src_sector == dst_sector == 0``. ``sb_bits`` carries the
    /// lattice site count. The rep lane regenerates the source orbit per
    /// group element (``apply_perm`` + conj(chi(g))) and projects into the
    /// destination with ``index_and_projection`` -- the same arithmetic the
    /// rep matvec kernel uses. The RepSectorData must outlive the observable.
    struct OperatorRef {
        const ed::symmetry::RepSectorData*  rd      = nullptr;
        std::uint64_t                       sb_bits = 0;

        /// Wrap a CSR-free RepSectorData.
        static OperatorRef from_rep(const ed::symmetry::RepSectorData& rep,
                                    std::uint64_t                      num_bits) {
            OperatorRef r;
            r.rd      = &rep;
            r.sb_bits = num_bits;
            return r;
        }

        bool valid()  const { return rd != nullptr; }
        std::size_t num_sectors() const {
            return 1;
        }
        std::uint64_t num_bits() const {
            return sb_bits;
        }
        std::size_t dim() const {
            return rd->reps.size();
        }
        std::uint64_t group_size() const {
            return static_cast<std::uint64_t>(rd->group_size);
        }
    };

    /// @param src        Source sector (provides the input orbit basis).
    /// @param src_sector Sector index within ``src`` (0: a ref wraps one sector).
    /// @param dst        Target sector (provides the output orbit basis).
    /// @param dst_sector Sector index within ``dst`` (0: a ref wraps one sector).
    /// @param transforms Observable terms (one/two-body; same layout as
    ///                   ``Operator::transform_data_``).
    /// @param spin_l     Spin S (0.5 for spin-1/2). Used for the
    ///                   diagonal/off-diagonal matrix-element
    ///                   prefactors.
    CrossSectorOrbitObservable(OperatorRef                 src,
                                std::size_t                 src_sector,
                                OperatorRef                 dst,
                                std::size_t                 dst_sector,
                                std::vector<TransformData>  transforms,
                                float                       spin_l = 0.5f);

    /// Apply the rectangular operator to a source-sector orbit-basis
    /// vector. ``out`` is zero-filled by the callee, then accumulated.
    ///
    /// @param in         length ``dim_src()``
    /// @param out        length ``dim_dst()`` (zero-filled internally)
    /// @param dst_size   must equal ``dim_dst()`` (defensive check)
    void apply(const Complex* in, Complex* out, std::size_t dst_size) const;

    /// The rectangular matrix as CSR (row per target state), built on first use.
    /// Empty pointers when the build was refused by ED_XSEC_CSR_BUDGET_GIB.
    struct CsrView {
        const std::int64_t*  row_ptr = nullptr;
        const std::uint32_t* col     = nullptr;
        const Complex*       val     = nullptr;
        std::size_t          rows = 0, cols = 0, nnz = 0;
    };
    [[nodiscard]] CsrView csr() const;
    std::size_t dim_src() const { return dim_src_; }
    std::size_t dim_dst() const { return dim_dst_; }


private:
    OperatorRef                  src_;
    std::size_t                  src_sector_;
    OperatorRef                  dst_;
    std::size_t                  dst_sector_;
    std::vector<TransformData>   transforms_;
    float                        spin_l_     = 0.5f;
    std::uint64_t                n_bits_     = 0;
    std::size_t                  dim_src_    = 0;
    std::size_t                  dim_dst_    = 0;
    double                       group_norm_ = 1.0;
    // Rep policy views (POD pointers into the refs'
    // RepSectorData).
    ed::matvec::basis::RepSymmetryBasisPolicy src_pol_{};
    ed::matvec::basis::RepSymmetryBasisPolicy dst_pol_{};

    // Cached rectangular reduced matrix A[k, alpha] (dst rows x src cols),
    // assembled on the first apply() by the same walk and reused afterwards.
    // The finite-T FTLM lane applies the probe M x R times per sector pair
    // and the walk costs |G|^2 x terms x index lookups per source row
    // (~70 ms per apply at N = 20); the CSR apply is a memory-bound gather. Refused (walk kept) when the
    // pre-merge triplet estimate exceeds ED_XSEC_CSR_BUDGET_GIB (default 4).
    mutable std::mutex                 csr_mutex_;
    mutable bool                       csr_built_   = false;
    mutable bool                       csr_refused_ = false;
    mutable std::vector<std::int64_t>  csr_row_ptr_;
    mutable std::vector<std::uint32_t> csr_col_;
    mutable std::vector<Complex>       csr_val_;
    template <class Emit> void walk_columns_(Emit&& emit) const;
    void build_csr_() const;
    void apply_walk_(const Complex* in, Complex* out) const;
};

}  // namespace ed::dssf

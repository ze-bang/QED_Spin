#pragma once
// =============================================================================
// include/ed/dssf/cross_sector_orbit_observable.h
//
// Rectangular orbit-basis cross-sector observable.
//
// Maps a vector expressed in the orbit basis of a *source* irrep
// (sector ``src_sector_idx`` of the underlying ``StreamingSymmetryOperator``
// operators) to a vector expressed in
// the orbit basis of a *target* irrep (``dst_sector_idx`` of the same
// or a different streaming operator on the same lattice).
//
// This is the cross-irrep generalisation of
// ``ed::dssf::CrossSectorObservable`` (which handles fixed-Sz
// cross-sector only) and the missing piece that closes the SOTA gap
// flagged in ``docs/architecture/SYMMETRY.md`` Section 3 (the
// dynamical S(Q, omega) selection rule walker for spatial irreps).
//
// Math
// ----
// Given an observable with terms ``O = sum_t T_t`` where each ``T_t``
// is a one-, two-, or three-body spin operator (the same
// ``Operator::TransformData`` layout used by H), and orbit-basis
// kets
//
//   |psi^{k_src}_alpha> = sum_g chi^*_{k_src}(g) g |s_alpha>
//   |psi^{k_dst}_beta>  = sum_h chi^*_{k_dst}(h) h |s_beta>
//
// the matrix element folds exactly the way the same-sector matvec
// (``StreamingSymmetryOperator::applyHamiltonianTermsFullSpace``) does:
//
//   out[k_dst, beta] += in[k_src, alpha]
//                    * alpha_s                         (orbit coeff of s in orbit_alpha)
//                    * h_element(s -> s')              (term coefficient)
//                    * conj(beta_{s'})                 (orbit coeff of s' in orbit_beta)
//                    * group_norm / state_beta.norm
//                    / state_alpha.norm
//
// where ``s -> s'`` is the bit-flip / Sz-sign action of ``T_t``,
// ``group_norm = 1 / |G|``, and the orbit-coefficient lookups use
// ``state_to_sector_basis_[dst_sector_idx]`` + ``findCoeff``. This
// mirrors the exact projection used by ``applySymmetrized``, so
// matrix elements computed by this class are consistent with the
// rest of the streaming-symmetry pipeline by construction.
//
// Threading
// ---------
// OpenMP-parallel over source orbit-basis indices (same dynamic
// schedule as ``CrossSectorObservable``). Per-thread accumulators are
// merged at the end, avoiding any atomic accumulation on the hot
// path (this matters for groups with hundreds of elements where the
// inner ``findCoeff`` is non-trivial).
//
// Cost
// ----
// ``apply()`` is
//   O(dim_src * <orbit size> * |transforms|)
// matrix-element evaluations, with a single ``findCoeff`` lookup per
// projected state (O(log |orbit|)) and a single
// ``lookupBasisIndex`` lookup (O(log dim_dst)). Identical asymptotics
// to ``applySymmetrized``.
//
// SOTA upgrade, May 2026.
// =============================================================================

#include <ed/core/linear_operator.h>
#include <ed/core/operator.h>             // Operator::TransformData
#include <ed/core/sorted_uint64_index.h>  // SortedUint64Index::kNotFound
#include <ed/symmetry/symmetry_sector_data.h>  // SymmetrySector
#include <ed/symmetry/sector_basis.h>     // SectorBasis (operator-collapse lane)
#include <ed/symmetry/rep_sector_data.h>  // RepSectorData (Stage 8d rep lane)
#include <ed/matvec/rep_symmetry_basis_policy.h>  // host rep policy view

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::dssf {

/// Rectangular orbit-basis observable mapping
///   (src_sector_idx of src_op) -> (dst_sector_idx of dst_op)
///
/// ``src_op`` and ``dst_op`` may be the same instance (homogeneous
/// case, same streaming operator) or different instances with the
/// same lattice (heterogeneous case, e.g. fixed-Sz x irrep ->
/// fixed-Sz x irrep with different ``n_up``). Both must define
/// ``num_sites()`` and have generated their per-sector basis +
/// state-to-orbit lookup tables.
class CrossSectorOrbitObservable {
public:
    using Complex       = std::complex<double>;
    using TransformData = Operator::TransformData;

    /// Non-owning handle onto a single materialised symmetry sector,
    /// owned by an ``ed::symmetry::SectorBasis`` (the producer behind
    /// ``make_sector_operators_tagged``). The ref wraps exactly ONE
    /// sector (``num_sectors() == 1``), so the CrossSectorOrbitObservable
    /// is built with ``src_sector == dst_sector == 0``. ``sb_bits``
    /// carries the lattice site count (the SectorBasis itself does not
    /// store it). Callers must keep the underlying SectorBasis /
    /// SectorOperator alive for the observable's lifetime.
    ///
    /// Operator-collapse Phase 3 (Jun 2026): the StreamingSymmetryOperator /
    /// FixedSzStreamingSymmetryOperator carrier lanes have been retired; the
    /// SectorBasis lane is the sole path.
    ///
    /// Stage 8d (SymmetryEngine v2, Jul 2026): second lane -- a CSR-free
    /// ``RepSectorData`` ref. Flip-extended and Sz-parity sectors never
    /// materialise an orbit CSR (their csr_provider throws by design), and
    /// the default rep-lazy fixed-Sz lane shouldn't have to either. The rep
    /// lane regenerates the source orbit per group element
    /// (``apply_perm`` + conj(chi(g))) and projects into the destination
    /// with ``index_and_projection`` -- the same arithmetic the rep matvec
    /// kernel uses, so both refs produce identical matrix elements (pinned
    /// by ``test_rep_cross_sector.cpp``).
    struct OperatorRef {
        const ed::symmetry::SectorBasis*    sb      = nullptr;
        const ed::symmetry::RepSectorData*  rd      = nullptr;
        std::uint64_t                       sb_bits = 0;

        /// Wrap a single materialised SectorBasis.
        /// ``num_bits`` is the lattice site count.
        static OperatorRef from(const ed::symmetry::SectorBasis& basis,
                                std::uint64_t                    num_bits) {
            OperatorRef r;
            r.sb      = &basis;
            r.sb_bits = num_bits;
            return r;
        }

        /// Wrap a CSR-free RepSectorData (rep-lazy / flip / parity
        /// sectors). The RepSectorData must outlive the observable.
        static OperatorRef from_rep(const ed::symmetry::RepSectorData& rep,
                                    std::uint64_t                      num_bits) {
            OperatorRef r;
            r.rd      = &rep;
            r.sb_bits = num_bits;
            return r;
        }

        bool valid()  const { return sb != nullptr || rd != nullptr; }
        bool is_rep() const { return rd != nullptr; }

        const SymmetrySector& sector(std::size_t /*k*/) const {
            return sb->sector();   // orbit lane only; single sector, k == 0
        }
        std::size_t num_sectors() const {
            return 1;
        }
        std::uint64_t num_bits() const {
            return sb_bits;
        }
        std::size_t dim() const {
            return is_rep() ? rd->reps.size()
                            : sb->sector().basis_states.size();
        }
        std::size_t lookupBasisIndex(std::size_t /*k*/,
                                     std::uint64_t s) const {
            const std::int64_t idx = sb->index_of(s);
            return (idx < 0)
                ? ed::core::SortedUint64Index::kNotFound
                : static_cast<std::size_t>(idx);
        }
        std::uint64_t group_size() const {
            return is_rep() ? static_cast<std::uint64_t>(rd->group_size)
                            : static_cast<std::uint64_t>(sb->group_size());
        }
    };

    /// @param src        Source streaming operator (provides the input orbit basis).
    /// @param src_sector Index into ``src``'s sectors_; the source irrep.
    /// @param dst        Target streaming operator (provides the output orbit basis).
    /// @param dst_sector Index into ``dst``'s sectors_; the target irrep.
    /// @param transforms Observable terms (one/two-body; mirror of
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

    /// std::function adapter for callers that pass operator
    /// applications as lambdas (the FTLM/CF matvec API). Matches
    /// ``CrossSectorObservable::as_apply_function``.
    auto as_apply_function() const {
        return [this](const Complex* in, Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }

    std::size_t dim_src() const { return dim_src_; }
    std::size_t dim_dst() const { return dim_dst_; }

    /// Per-sector quantum-number labels for the source / target
    /// (mirror of ``SymBasisState::quantum_numbers``). Used by the
    /// streaming-symmetry spectral workflow to populate
    /// ``SpectralSectorEntry`` tags and the selection-rule label.
    /// Rep-lane refs carry no SymmetrySector metadata -- callers on
    /// that lane label from the handle's SectorTag instead (the
    /// cross-irrep bindings already do).
    const std::vector<int>& src_quantum_numbers() const {
        static const std::vector<int> kEmpty;
        return src_.is_rep() ? kEmpty
                             : src_.sector(src_sector_).quantum_numbers;
    }
    const std::vector<int>& dst_quantum_numbers() const {
        static const std::vector<int> kEmpty;
        return dst_.is_rep() ? kEmpty
                             : dst_.sector(dst_sector_).quantum_numbers;
    }

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
    // Stage 8d rep-lane policy views (POD pointers into the refs'
    // RepSectorData; valid only when the matching ref ``is_rep()``).
    ed::matvec::basis::RepSymmetryBasisPolicy src_pol_{};
    ed::matvec::basis::RepSymmetryBasisPolicy dst_pol_{};
};

}  // namespace ed::dssf

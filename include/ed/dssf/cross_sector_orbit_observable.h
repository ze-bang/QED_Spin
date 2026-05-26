#pragma once
// =============================================================================
// include/ed/dssf/cross_sector_orbit_observable.h
//
// Rectangular orbit-basis cross-sector observable.
//
// Maps a vector expressed in the orbit basis of a *source* irrep
// (sector ``src_sector_idx`` of the underlying ``StreamingSymmetryOperator``
// or ``FixedSzStreamingSymmetryOperator``) to a vector expressed in
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
#include <ed/core/streaming_symmetry.h>   // StreamingSymmetryOperator,
                                          // FixedSzStreamingSymmetryOperator

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

    /// Polymorphic source/target handle so a single class works
    /// against ``StreamingSymmetryOperator`` and
    /// ``FixedSzStreamingSymmetryOperator``. The handle is non-owning
    /// -- callers must keep the underlying operator alive for the
    /// CrossSectorOrbitObservable's lifetime.
    struct OperatorRef {
        StreamingSymmetryOperator*        sym = nullptr;
        FixedSzStreamingSymmetryOperator* fsz = nullptr;
        static OperatorRef from(StreamingSymmetryOperator& op) {
            OperatorRef r;
            r.sym = &op;
            return r;
        }
        static OperatorRef from(FixedSzStreamingSymmetryOperator& op) {
            OperatorRef r;
            r.fsz = &op;
            return r;
        }
        /// Either ``sym`` or ``fsz`` must be non-null. The "raw"
        /// pointer interpretation is what consumers use; the typed
        /// references just give them a static-polymorphic API.
        bool valid() const { return sym != nullptr || fsz != nullptr; }

        const SymmetrySector& sector(std::size_t k) const {
            if (sym) return sym->getSector(k);
            return fsz->getSector(k);
        }
        std::size_t num_sectors() const {
            if (sym) return sym->num_sectors();
            return fsz->num_sectors();
        }
        std::uint64_t num_bits() const {
            if (sym) return sym->getNumBits();
            return fsz->getNumBits();
        }
        std::size_t lookupBasisIndex(std::size_t k,
                                     std::uint64_t s) const {
            if (sym) return sym->lookupBasisIndex(k, s);
            return fsz->lookupBasisIndex(k, s);
        }
        std::uint64_t group_size() const {
            if (sym) return sym->getGroupSize();
            return fsz->getGroupSize();
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
    const std::vector<int>& src_quantum_numbers() const {
        return src_.sector(src_sector_).quantum_numbers;
    }
    const std::vector<int>& dst_quantum_numbers() const {
        return dst_.sector(dst_sector_).quantum_numbers;
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
};

}  // namespace ed::dssf

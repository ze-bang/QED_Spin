#pragma once
// =============================================================================
// include/ed/core/sector_loop.h
//
// Shared plumbing for the streaming-symmetry sector loop used by all
// three workflows (``ed::workflows::solve``, ``::thermal``,
// ``::spectral``). The orchestrator entry points themselves operate on
// a single ``LinearOperator`` (one irrep block at a time); the per-
// sector iteration lives one level up (CLI's
// ``run_streaming_symmetry_workflow``, the Python Pybind11 bindings
// ``workflows_{solve,thermal,spectral}_streaming_symmetry_directory``,
// and a small amount of glue in the SOTA spectral path).
//
// To keep all of those callers in lock-step, this header centralises:
//
//   * ``StreamingSymmetryHandle`` -- a thin downcast wrapper that
//     exposes ``num_sectors()``, ``sector(k)``, ``sector_tag(k)`` for
//     both the full streaming-symmetry operator and its fixed-Sz
//     sibling (``FixedSzStreamingSymmetryOperator``). Hides the
//     ``dynamic_cast`` boilerplate every caller used to repeat.
//
//   * ``filter_sectors(...)`` -- canonical resolution of the
//     ``opts.selected_sectors`` filter against the set of non-empty
//     sectors. Single source of truth so the CLI / Python / GPU paths
//     all expose the same ``sector=`` semantics.
//
// SOTA upgrade, May 2026.
// =============================================================================

#include <ed/core/results.h>            // SectorTag
#include <ed/core/streaming_symmetry.h> // StreamingSymmetryOperator + FixedSzStreamingSymmetryOperator
#include <ed/symmetry/sector_set.h>     // make_sector_operator_adopt (P5b gate)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ed::core {

// ---------------------------------------------------------------------------
// Operator-collapse refactor (Phase C, Jun 2026).
//
// ``StreamingSymmetryHandle::sector(k)`` returns a standalone
// ``ed::symmetry::SectorOperator`` (owning its sector's orbit data + a copy
// of the Hamiltonian terms, driven by the unified
// ``CpuMatVecBackend<SymmetryBasisPolicy>`` on the CPU and the GPU sector
// mirror via ``bind_cuda()`` on CUDA builds). This routes the ENTIRE
// production sector loop (CLI, Python bindings, thermal, spectral) through
// the collapse-target operator.
//
// Phase C (Jun 2026): the cutover gate (``ED_SYMMETRY_SECTOR_OPERATOR``) and
// the legacy back-referencing ``SectorView`` escape hatch have been REMOVED.
// ``SectorOperator`` is now the only sector-matvec path on every backend
// (CPU/GPU parity landed in Phase A; production cutover validated in Phase B).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CSR-free lazy rep path gate ("scan other region" optimisation, Jun 2026).
//
// On CUDA builds the fixed-Sz sector loop hands out CSR-free lazy
// SectorOperators (``make_rep_sector_operator_lazy``): the GPU on-the-fly
// representative matvec never materialises the per-sector host orbit CSR
// (~24 GiB/sector at N=32), and a CPU fallback lazily materialises it only if
// ``apply`` is actually called. Tied to the SAME ``ED_GPU_SYMMETRY_REP`` gate
// as the device rep matvec so ``=0`` restores the legacy adopt/orbit-CSR path
// end-to-end. On non-CUDA builds there is no GPU rep matvec, so the CPU always
// needs the orbit CSR -- keep the eager adopt path (return false).
// ---------------------------------------------------------------------------
inline bool rep_lazy_sector_path_enabled() {
#ifdef WITH_CUDA
    static const bool enabled = [] {
        const char* e = std::getenv("ED_GPU_SYMMETRY_REP");
        if (e == nullptr || e[0] == '\0') return true;   // default ON
        if (e[0] == '0' && e[1] == '\0')  return false;  // "0" -> OFF
        return true;                                     // anything else -> ON
    }();
    return enabled;
#else
    return false;
#endif
}

/// Lightweight handle over either flavour of streaming-symmetry
/// operator. The handle does NOT own the operator -- callers must keep
/// the underlying ``ed::LinearOperator`` alive for the handle's
/// lifetime.
class StreamingSymmetryHandle {
public:
    explicit StreamingSymmetryHandle(ed::LinearOperator* op) {
        if (!op) {
            throw std::invalid_argument(
                "ed::core::StreamingSymmetryHandle: null operator.");
        }
        fsz_sym_op_ = dynamic_cast<FixedSzStreamingSymmetryOperator*>(op);
        sym_op_     = dynamic_cast<StreamingSymmetryOperator*>(op);
        if (!fsz_sym_op_ && !sym_op_) {
            throw std::invalid_argument(
                "ed::core::StreamingSymmetryHandle: operator is neither a "
                "StreamingSymmetryOperator nor a "
                "FixedSzStreamingSymmetryOperator. Build it via "
                "ed::make_operator(OperatorSpec{ .streaming_symmetry = true }).");
        }
    }

    [[nodiscard]] std::size_t num_sectors() const {
        return fsz_sym_op_
            ? fsz_sym_op_->num_sectors()
            : sym_op_->num_sectors();
    }

    /// Per-sector matvec view (one irrep block). The returned
    /// LinearOperator is owned and must outlive its uses.
    ///
    /// Returns a standalone ``ed::symmetry::SectorOperator``
    /// (collapse-target path) built by adopting the underlying operator's
    /// materialised sector data. It satisfies the ``ed::LinearOperator``
    /// contract the orchestrator consumes.
    [[nodiscard]] std::unique_ptr<ed::LinearOperator>
    sector(std::size_t k) const {
        if (fsz_sym_op_) {
            // CSR-free lazy path (CUDA, default ON): hand out a SectorOperator
            // that defers the host orbit CSR -- the GPU rep matvec never
            // materialises it (~24 GiB/sector saved at N=32). Falls back to a
            // lazy CPU materialisation only if ``apply`` is invoked.
            if (rep_lazy_sector_path_enabled()) {
                return ed::symmetry::make_rep_sector_operator_lazy(
                    *fsz_sym_op_, k);
            }
            SymmetrySector sec = fsz_sym_op_->getSector(k);  // copy (materialises)
            return ed::symmetry::make_sector_operator_adopt(
                *fsz_sym_op_, std::move(sec),
                static_cast<std::size_t>(fsz_sym_op_->getGroupSize()));
        }
        SymmetrySector sec = sym_op_->getSector(k);          // copy (materialises)
        return ed::symmetry::make_sector_operator_adopt(
            *sym_op_, std::move(sec),
            static_cast<std::size_t>(sym_op_->getGroupSize()));
    }

    /// Quantum-number tag for sector ``k``. Pulls
    /// ``SymmetrySector::quantum_numbers`` + the basis dimension out
    /// of the underlying operator's sector metadata.
    [[nodiscard]] SectorTag sector_tag(std::size_t k) const {
        SectorTag t;
        t.sector_index = k;
        if (fsz_sym_op_) {
            // CSR-free: dim from Pass 1.5, quantum numbers from generation --
            // neither triggers orbit-CSR materialisation (see the cheap
            // accessors on FixedSzStreamingSymmetryOperator).
            t.sector_dim      = fsz_sym_op_->getSectorDimension(k);
            t.quantum_numbers = fsz_sym_op_->getSectorQuantumNumbers(k);
            t.n_up            = static_cast<int>(fsz_sym_op_->getNUp());
        } else {
            const auto& s = sym_op_->getSector(k);
            t.sector_dim      = static_cast<std::uint64_t>(s.basis_states.size());
            t.quantum_numbers = s.quantum_numbers;
            t.n_up            = -1;
        }
        return t;
    }

private:
    FixedSzStreamingSymmetryOperator* fsz_sym_op_ = nullptr;
    StreamingSymmetryOperator*        sym_op_     = nullptr;
};

/// Canonical filter for the ``selected_sectors`` axis. Returns the
/// list of sector indices (in ascending order) that the streaming
/// loop should visit, given:
///
///   * ``num_sectors``      -- total sector count from the handle
///   * ``selected_sectors`` -- caller-supplied filter (empty => keep all)
///
/// Out-of-range indices are silently dropped (matches the legacy
/// behaviour for ``EDParameters::selected_sectors``); duplicates are
/// deduplicated; the output is sorted ascending so that downstream
/// HDF5 paths and per-sector tags are stable across runs.
inline std::vector<std::size_t>
filter_sectors(std::size_t                       num_sectors,
               const std::vector<std::size_t>&   selected_sectors) {
    std::vector<std::size_t> out;
    if (selected_sectors.empty()) {
        out.reserve(num_sectors);
        for (std::size_t k = 0; k < num_sectors; ++k) out.push_back(k);
        return out;
    }
    out.reserve(selected_sectors.size());
    for (std::size_t k : selected_sectors) {
        if (k < num_sectors) out.push_back(k);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// ----------------------------------------------------------------------------
// Cross-irrep selection-rule helpers.
//
// ``SectorTag::quantum_numbers`` is stored as one integer per
// minimal generator (matches ``SymmetrySector::quantum_numbers`` in
// ``include/ed/core/streaming_symmetry.h``). For a translation
// generator of order ``N``, ``q in {0, ..., N-1}`` is the irrep
// label and the Bloch phase used in the orbit projection is
// ``chi_q(T^a) = exp(-i 2 pi q a / N)`` (see
// streaming_symmetry.h around line 1234 and the QuSpin-convention
// test pin in ``tests/unit/test_symmetry_dsl.cpp``).
//
// The selection rule for a momentum-carrying observable ``O_Q``
// follows the standard ``k_final = k_initial + Q`` rule in PHYSICAL
// units. With the streaming-symmetry convention
// ``chi_q(T) = exp(-2 pi i q / N)``, a Bloch state in irrep ``q``
// satisfies ``T |psi_q> = exp(-2 pi i q / N) |psi_q>``, so under the
// ``T = exp(+i k)`` convention the irrep label ``q`` corresponds to
// PHYSICAL momentum ``k = -2 pi q / N``. Applying
// ``S^Q = (1/sqrt N) sum_j exp(-i Q j) S_j`` then carries the state
// to ``k_new = k_old + Q``, which in irrep labels reads
//
//   q_dst = (q_src - round(Q * order / (2 pi))) mod order
//
// when ``Q`` is given as a physical momentum. The orchestrator
// docstring on ``SpectralOptions::momentum_transfer`` describes the
// argument as "fractional reciprocal-lattice units" with one full
// reciprocal-lattice vector being ``G = 2 pi`` -- so the scaling
// becomes ``delta_q_label = -round(Q_frac * order)``. The minus
// sign comes purely from the ``q = -k * N / (2 pi)`` mapping
// inherited from the QuSpin-style phase convention used by the
// streaming-symmetry builder and pinned by
// ``tests/unit/test_symmetry_dsl.cpp``. We empirically verified the
// sign by projecting ``S^z_{Q=2pi/N} |psi_0>`` onto each irrep
// subspace and confirming it lands in ``q_src - 1`` for the N=6
// AFM Heisenberg ring (see ``test_cross_irrep_spectral_matches_lehmann_reference``).
//
// The helpers below
//   * compute the per-generator order from the set of all sector
//     quantum_numbers,
//   * convert a fractional reciprocal-lattice vector ``Q_frac`` into
//     integer ``delta_q``,
//   * walk the handle and return the unique sector index whose
//     quantum_numbers match the target (or
//     ``kSectorNotFound`` if no surviving sector has that
//     irrep label).
//
// Multi-generator groups (e.g. D_4: T x C_4): we restrict to abelian
// (or strictly translation-only) groups in this first cut. The
// helper still returns the right answer for any *abelian* generator
// set as long as each generator has a well-defined Z_n order.
// ----------------------------------------------------------------------------

inline constexpr std::size_t kSectorNotFound =
    static_cast<std::size_t>(-1);

/// Infer per-generator orders by taking ``max(q[g]) + 1`` across all
/// sectors. Conservative -- if every sector has ``q[g] = 0`` (trivial
/// generator) the inferred order is 1, which makes the selection
/// rule act as the identity on that axis. For Z_N translations the
/// inferred order matches ``N`` exactly when at least one sector hits
/// the maximal irrep, which is always true for the streaming-symmetry
/// builder (it enumerates ``N`` distinct sectors per generator).
inline std::vector<int>
infer_generator_orders(const StreamingSymmetryHandle& handle) {
    const std::size_t S = handle.num_sectors();
    if (S == 0) return {};
    std::vector<int> orders;
    for (std::size_t k = 0; k < S; ++k) {
        const auto& q = handle.sector_tag(k).quantum_numbers;
        if (orders.size() < q.size()) orders.resize(q.size(), 1);
        for (std::size_t g = 0; g < q.size(); ++g) {
            if (q[g] + 1 > orders[g]) orders[g] = q[g] + 1;
        }
    }
    return orders;
}

/// Convert a fractional reciprocal-lattice vector ``Q_frac`` (Q in
/// units of the reciprocal-lattice vector ``G = 2 pi``, e.g.
/// ``Q_frac = 1/N`` for one irrep step on an N-site translation
/// group) into the integer ``delta_q[g]`` irrep-label shift used to
/// resolve the target sector.
///
/// The map is ``delta_q[g] = -round(Q_frac[g] * orders[g])`` with
/// the rounding residual (in label units) reported via
/// ``out_residual``. The minus sign is the streaming-symmetry
/// phase-convention compensation -- see the header docstring above
/// for the derivation. Reducing modulo ``orders[g]`` ensures the
/// resulting label sits in ``[0, orders[g])``.
inline std::vector<int>
quantize_momentum_transfer(const std::vector<double>& Q_frac,
                           const std::vector<int>&    orders,
                           double*                    out_residual = nullptr) {
    const std::size_t G = orders.size();
    std::vector<int> dq(G, 0);
    double max_resid = 0.0;
    for (std::size_t g = 0; g < G; ++g) {
        const double q_real = (g < Q_frac.size()) ? Q_frac[g] : 0.0;
        const int    order  = std::max(orders[g], 1);
        // Multiply by the generator order to convert from "fraction
        // of a reciprocal-lattice vector" (the SpectralOptions
        // documented unit) to "integer irrep-label step". The
        // physical momentum k = -2 pi q / N means a positive Q in
        // physical units corresponds to a NEGATIVE q-label shift,
        // hence the unary minus below.
        const double q_in_labels = -q_real * static_cast<double>(order);
        const double rounded     = std::round(q_in_labels);
        max_resid = std::max(max_resid, std::abs(q_in_labels - rounded));
        int shift = static_cast<int>(rounded);
        shift = ((shift % order) + order) % order;
        dq[g] = shift;
    }
    if (out_residual) *out_residual = max_resid;
    return dq;
}

/// Apply the per-generator shift ``delta_q`` to a source
/// ``quantum_numbers`` vector and return the target irrep label.
/// Missing generators (``src_qn`` shorter than ``orders``) default
/// to ``q[g] = 0``.
inline std::vector<int>
add_quantum_numbers(const std::vector<int>& src_qn,
                    const std::vector<int>& delta_q,
                    const std::vector<int>& orders) {
    const std::size_t G = orders.size();
    std::vector<int> dst(G, 0);
    for (std::size_t g = 0; g < G; ++g) {
        const int src   = (g < src_qn.size()) ? src_qn[g] : 0;
        const int delta = (g < delta_q.size()) ? delta_q[g] : 0;
        const int order = std::max(orders[g], 1);
        int next = ((src + delta) % order + order) % order;
        dst[g] = next;
    }
    return dst;
}

/// Find the sector whose ``quantum_numbers`` match ``target_qn``.
/// Returns ``kSectorNotFound`` when no such sector exists or when
/// multiple disagree -- the streaming-symmetry builder guarantees
/// uniqueness for a fixed (n_up, irrep) tuple so duplicates would
/// indicate a metadata bug upstream.
inline std::size_t
find_sector_by_quantum_numbers(
        const StreamingSymmetryHandle& handle,
        const std::vector<int>&        target_qn) {
    const std::size_t S = handle.num_sectors();
    std::size_t hit  = kSectorNotFound;
    for (std::size_t k = 0; k < S; ++k) {
        const auto& q = handle.sector_tag(k).quantum_numbers;
        if (q.size() != target_qn.size()) continue;
        bool match = true;
        for (std::size_t g = 0; g < q.size(); ++g) {
            if (q[g] != target_qn[g]) { match = false; break; }
        }
        if (match) {
            if (hit != kSectorNotFound) return kSectorNotFound;
            hit = k;
        }
    }
    return hit;
}

/// Convenience wrapper: resolve the target sector index from a
/// source sector + a fractional momentum-transfer vector. Returns
/// ``kSectorNotFound`` if the selection rule lands outside the
/// builder's sector set (e.g. when the user specifies an
/// incommensurate Q).
inline std::size_t
resolve_target_sector(const StreamingSymmetryHandle& handle,
                      std::size_t                    src_sector,
                      const std::vector<double>&     Q_frac,
                      double*                        out_residual = nullptr) {
    if (src_sector >= handle.num_sectors()) return kSectorNotFound;
    const auto orders   = infer_generator_orders(handle);
    const auto delta_q  = quantize_momentum_transfer(Q_frac, orders, out_residual);
    const auto src_qn   = handle.sector_tag(src_sector).quantum_numbers;
    const auto tgt_qn   = add_quantum_numbers(src_qn, delta_q, orders);
    return find_sector_by_quantum_numbers(handle, tgt_qn);
}

}  // namespace ed::core

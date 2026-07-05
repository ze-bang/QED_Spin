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
// Templated on the handle type so the same logic drives both the legacy
// ``StreamingSymmetryHandle`` and the operator-collapse ``SectorSetView``
// (``ed/core/make_operator.h``). Any type exposing ``num_sectors()`` +
// ``sector_tag(k).quantum_numbers`` for EVERY raw sector (including empty
// irreps) satisfies the contract.
template <class HandleT>
inline std::vector<int>
infer_generator_orders(const HandleT& handle) {
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
template <class HandleT>
inline std::size_t
find_sector_by_quantum_numbers(
        const HandleT&                 handle,
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
template <class HandleT>
inline std::size_t
resolve_target_sector(const HandleT&                 handle,
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

/// Stage 8d (SymmetryEngine v2): selection rule over SYNTHETIC sector sets.
///
/// Sz-parity halves and prod-sigma^x flip sectors append +-1 slot labels
/// AFTER the raw spatial irrep labels (see ``make_operator.h``: parity
/// first, then flip sign). Those labels are NOT Z_n momentum axes -- the
/// modular walk in ``resolve_target_sector`` would corrupt a -1 label
/// ((-1 + 0) mod 2 == +1) and silently land in the wrong slot. This
/// variant applies the momentum shift to the leading
/// ``qn.size() - n_slots`` labels only and maps each trailing slot label
/// by an explicit sign:
///
///   dst_slot[j] = src_slot[j] * slot_signs[j]
///
/// where the caller derives ``slot_signs`` from the probe observable:
/// parity slot sign = -1 iff the probe toggles set-bit parity
/// (``delta_n_up_parity``), flip slot sign = the probe's
/// ``spin_flip_character`` (must be +-1; a character-less probe cannot
/// be routed through flip slots).
template <class HandleT>
inline std::size_t
resolve_target_sector_slotted(const HandleT&              handle,
                              std::size_t                 src_sector,
                              const std::vector<double>&  Q_frac,
                              std::size_t                 n_slots,
                              const std::vector<int>&     slot_signs,
                              double*                     out_residual = nullptr) {
    if (n_slots == 0) {
        return resolve_target_sector(handle, src_sector, Q_frac, out_residual);
    }
    const std::size_t S = handle.num_sectors();
    if (src_sector >= S || slot_signs.size() < n_slots) return kSectorNotFound;

    // Split every tag into (raw momentum prefix, slot tail); infer the
    // per-generator orders over the PREFIX only.
    std::vector<int> orders;
    auto split = [&](std::size_t k, std::vector<int>& prefix,
                     std::vector<int>& tail) -> bool {
        const auto& q = handle.sector_tag(k).quantum_numbers;
        if (q.size() < n_slots) return false;
        const std::size_t np = q.size() - n_slots;
        prefix.assign(q.begin(), q.begin() + static_cast<std::ptrdiff_t>(np));
        tail.assign(q.begin() + static_cast<std::ptrdiff_t>(np), q.end());
        return true;
    };
    for (std::size_t k = 0; k < S; ++k) {
        std::vector<int> prefix, tail;
        if (!split(k, prefix, tail)) continue;
        if (orders.size() < prefix.size()) orders.resize(prefix.size(), 1);
        for (std::size_t g = 0; g < prefix.size(); ++g) {
            if (prefix[g] + 1 > orders[g]) orders[g] = prefix[g] + 1;
        }
    }

    std::vector<int> src_prefix, src_tail;
    if (!split(src_sector, src_prefix, src_tail)) return kSectorNotFound;
    const auto delta_q    = quantize_momentum_transfer(Q_frac, orders, out_residual);
    const auto tgt_prefix = add_quantum_numbers(src_prefix, delta_q, orders);
    std::vector<int> tgt_tail(n_slots);
    for (std::size_t j = 0; j < n_slots; ++j) {
        tgt_tail[j] = src_tail[j] * slot_signs[j];
    }

    std::size_t hit = kSectorNotFound;
    for (std::size_t k = 0; k < S; ++k) {
        std::vector<int> prefix, tail;
        if (!split(k, prefix, tail)) continue;
        if (prefix != tgt_prefix || tail != tgt_tail) continue;
        if (hit != kSectorNotFound) return kSectorNotFound;  // ambiguous
        hit = k;
    }
    return hit;
}

}  // namespace ed::core

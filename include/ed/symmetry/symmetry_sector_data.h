#pragma once
// =============================================================================
// include/ed/symmetry/symmetry_sector_data.h
//
// Orbit-based symmetry-sector data structures shared across the
// symmetry stack (``SectorBasis`` / ``SectorOperator`` / the unified
// ``CpuMatVecBackend<SymmetryBasisPolicy>`` / the GPU sector mirror /
// the cross-irrep observable).
//
// These three POD-ish structs used to live in the (now-deleted)
// ``ed/core/streaming_symmetry.h`` alongside the
// ``StreamingSymmetryOperator`` carrier classes. They are the part of
// that header that survives the operator-collapse carrier retirement
// (Phase 3, Jun 2026): they describe ONE symmetry sector's orbit data
// and the per-sector state->basis-index lookup, independent of any
// operator class.
//
//   * SymBasisState     -- one symmetrised basis ket (orbit reps,
//                          orbit elements, phase coefficients, norm).
//   * SymmetrySector    -- a vector of SymBasisState plus irrep labels.
//   * SectorLookupHandle -- O(1)/O(log N) computational-state ->
//                          orbit-basis-index lookup used in the SpMV
//                          hot loop.
//
// Kept in the GLOBAL namespace to match every existing reference
// (``::SymmetrySector`` etc.) across the codebase.
// =============================================================================

#include <ed/core/linear_operator.h>        // ed::Complex
#include <ed/core/basis_utils.h>            // LinIndexTable
#include <ed/core/sorted_uint64_index.h>    // SortedUint64Index

#include <algorithm>
#include <complex>
#include <cstdint>
#include <numeric>
#include <vector>

// These structs live in the GLOBAL namespace (matching every existing
// ``::SymmetrySector`` reference) and use a bare ``Complex``. The legacy
// ``ed/core/streaming_symmetry.h`` relied on a global ``Complex`` alias
// brought in transitively; declare it explicitly here so the structs are
// self-contained. (An identical global ``using`` redeclaration is benign.)
using Complex = std::complex<double>;

// ============================================================================
// SectorLookupHandle  (Wave A2, May 2026)
// ----------------------------------------------------------------------------
// Bundles the existing ``SortedUint64Index`` fallback (O(log N) per find)
// with an optional dense O(1) side-table built per sector. Used inside
// the symmetry SpMV hot loop:
//
//   const std::size_t k = lookup.find(s_prime);  // <-- the hot lookup
//
// which inlines to a single branch on ``handle.dense != nullptr``. The
// dense table is an int32_t array of length ``index_dim`` (= fixed-Sz
// dim, or 2^N for the full-space variant); ``-1`` marks "not in this
// sector's orbits". Building it is O(orbit_total) and the memory cost
// is 4 bytes per slot per sector -- a 4x reduction over the
// ``SortedUint64Index`` storage when the sector covers the bulk of the
// fixed-Sz / full Hilbert space.
//
// Memory budget is gated by ``ED_SYM_DENSE_LOOKUP_BYTES_MAX``
// (default 512 MB total across all sectors); above that the operator
// stays on the SortedUint64Index path.
// ============================================================================
struct SectorLookupHandle {
    const ed::core::SortedUint64Index* fallback = nullptr;
    const std::int32_t* dense                   = nullptr;
    // For the fixed-Sz variant this points to the parent's LinIndexTable;
    // for the full-space variant it is nullptr and ``s`` itself is the
    // dense-table index (since the full Hilbert space is enumerable by
    // ``state``).
    const LinIndexTable* lin                    = nullptr;

    inline std::size_t find(std::uint64_t s) const {
        if (dense != nullptr) {
            std::int64_t li;
            if (lin != nullptr) {
                li = lin->lookup(s);
                if (li < 0) {
                    return ed::core::SortedUint64Index::kNotFound;
                }
            } else {
                li = static_cast<std::int64_t>(s);
            }
            const std::int32_t v = dense[li];
            return (v < 0) ? ed::core::SortedUint64Index::kNotFound
                           : static_cast<std::size_t>(v);
        }
        return fallback->find(s);
    }
};

// ============================================================================
// Orbit-Based Symmetry Data Structures
// ============================================================================

/**
 * @brief Compact representation of a symmetrized basis state
 *
 * Stores orbit elements and their phase coefficients for efficient H*v computation.
 * Memory: O(orbit_size) per basis state instead of O(full_dim).
 */
struct SymBasisState {
    uint64_t orbit_rep;                    // Representative element (smallest in orbit)
    std::vector<int> quantum_numbers;      // Quantum numbers for this sector
    std::vector<uint64_t> orbit_elements;  // All states in the orbit (sorted ascending after sortOrbit())
    std::vector<Complex> orbit_coefficients;  // Coefficient of each orbit element in symmetrized state (parallel to orbit_elements)
    double norm;                           // Normalization factor
    // Cached reciprocal of `norm` (0 when norm == 0). Derived, NOT persisted:
    // recomputed by sortOrbit() -- the universal finalize point invoked once
    // after `norm` is assigned on every construction / HDF5-load path. Lets the
    // matvec hot loop turn the per-emit `group_norm / norm_k` division into a
    // multiply (see SymmetryBasisPolicy::coeff_modifier / iter_orbit).
    double inv_norm;

    SymBasisState() : orbit_rep(0), norm(0.0), inv_norm(0.0) {}

    SymBasisState(uint64_t rep, const std::vector<int>& qn, double n = 1.0)
        : orbit_rep(rep), quantum_numbers(qn), norm(n),
          inv_norm(n != 0.0 ? 1.0 / n : 0.0) {}

    /**
     * Sort orbit_elements ascending and parallel-sort orbit_coefficients to
     * keep them aligned. After this, findCoeff() does an O(log |orbit|)
     * binary search instead of an O(|orbit|) linear scan -- a 5-50x speedup
     * for large symmetry groups (e.g. 768 = 48 * 16 cubic + translation).
     *
     * Idempotent and safe to call multiple times; cheap if already sorted.
     */
    void sortOrbit() {
        // Refresh the cached reciprocal norm at the canonical finalize point.
        // Done before the early-out so size-0/1 orbits also get a valid
        // inv_norm (their `norm` is still set by the construction path).
        inv_norm = (norm != 0.0) ? (1.0 / norm) : 0.0;
        const size_t n = orbit_elements.size();
        if (n < 2) return;
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return orbit_elements[a] < orbit_elements[b]; });
        // In-place permutation with one extra buffer.
        std::vector<uint64_t> e(n);
        std::vector<Complex> c(n);
        for (size_t k = 0; k < n; ++k) {
            e[k] = orbit_elements[idx[k]];
            c[k] = orbit_coefficients[idx[k]];
        }
        orbit_elements = std::move(e);
        orbit_coefficients = std::move(c);
    }

    /**
     * O(log |orbit|) lookup of the coefficient of |s_prime> in this
     * symmetrised basis state. Returns 0 if s_prime is not in the orbit.
     * Requires orbit_elements to be sorted ascending (call sortOrbit()
     * exactly once after the orbit is fully populated).
     *
     * NOTE: kept as std::lower_bound deliberately. Unlike
     * ed::core::SortedUint64Index::find() (which searches multi-GB,
     * cache-cold arrays where a branch-free + prefetch form wins ~2x),
     * an orbit is SMALL -- bounded by the group order |G| (typically ~N
     * up to ~768) and almost always L1/L2-resident. There the manual
     * prefetch hints are pure overhead and the branch-free cmov form is
     * no faster than the well-predicted std::lower_bound, so the simple
     * form is retained.
     */
    inline Complex findCoeff(uint64_t s_prime) const {
        auto it = std::lower_bound(orbit_elements.begin(), orbit_elements.end(), s_prime);
        if (it == orbit_elements.end() || *it != s_prime) return Complex(0.0, 0.0);
        return orbit_coefficients[static_cast<size_t>(it - orbit_elements.begin())];
    }
};

/**
 * @brief Sector information with orbit representatives
 */
struct SymmetrySector {
    uint64_t sector_id;
    std::vector<int> quantum_numbers;
    std::vector<Complex> phase_factors;
    std::vector<SymBasisState> basis_states;  // Compact basis representation

    SymmetrySector() : sector_id(0) {}
};

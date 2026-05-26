#pragma once
// =============================================================================
// include/ed/matvec/term_storage.h
//
// TermStorage: the canonical, single-source-of-truth Structure-of-Arrays
// term schema for a spin-system Hamiltonian. Owned by `Operator` (and by
// any future `GpuOperator` / `DistributedOperator` once they migrate).
//
// Background
// ----------
// Before the term-storage unification (May 2026), `Operator` carried THREE
// coexisting term representations:
//
//   1. ``transforms_``         std::vector<std::function<...>>  -- legacy
//   2. ``transform_data_``     std::vector<TransformData> (AoS) -- legacy
//   3. ``diag_one_body_``,
//      ``offdiag_one_body_``,
//      ``diag_two_body_``,
//      ``mixed_two_body_``,
//      ``offdiag_two_body_``,
//      ``three_body_data_``    std::vector<...>           (SoA) -- hot path
//
// External code pushed into (2); a lazy ``separateTransformsByType()`` call
// on first ``apply()`` fanned the AoS list into the SoA bins, guarded by a
// ``transforms_separated_`` flag. ``invalidateMatrixCaches()`` did NOT clear
// that flag, so a sequence like::
//
//     op.transform_data_.push_back(t);
//     op.invalidateMatrixCaches();
//     op.apply(in, out, n);
//
// silently dropped the new term: the SoA bins (populated on first apply) were
// not refreshed, and the matvec kernel reads only the SoA. The lazy fan-out
// also needed a ``const_cast`` inside the otherwise-const apply() chain.
//
// This header collapses the storage to a single SoA `TermStorage` struct,
// exposes typed setters that classify-and-route at the call site (no lazy
// fan-out), and keeps the bit-flip schema (field names, op_type encodings)
// identical to what ``ed::matvec::kernel::apply_terms`` and
// ``ed::matvec::CpuMatVecBackend`` already consume.
//
// op_type encoding (unchanged from the legacy API):
//   0 = S+   (raising spin operator)
//   1 = S-   (lowering spin operator)
//   2 = Sz   (diagonal spin operator)
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::matvec {

using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// Term-record structs (one per kernel branch). Field names are duck-typed by
// ``term_kernels.h`` and the CSR triplet emitter in ``matvec_backend.h``, so
// callers that previously referenced ``Operator::DiagonalOneBody`` etc. keep
// working through type aliases in ``operator.h``.
// -----------------------------------------------------------------------------

/// One-body diagonal (Sz only). New basis state == input basis state.
struct DiagOneBody {
    std::uint64_t site_index{0};
    Complex       coefficient{0.0, 0.0};
};

/// One-body off-diagonal (S+ or S-). Flips one bit; gated by op_type.
struct OffDiagOneBody {
    std::uint64_t site_index{0};
    std::uint8_t  op_type{0};  ///< 0 = S+, 1 = S-
    Complex       coefficient{0.0, 0.0};
};

/// Two-body purely diagonal (Sz Sz). No bit flips.
struct DiagTwoBody {
    std::uint64_t site_index_1{0};
    std::uint64_t site_index_2{0};
    Complex       coefficient{0.0, 0.0};
};

/// Two-body mixed (one Sz, one S+/S-). Flips one bit.
///
/// The Sz factor and the S+/S- flip commute (they act on different
/// sites), so the kernel only needs to know which bit to flip, which
/// bit carries Sz, and the flip op_type. No "Sz-first" flag is kept:
/// ``Sz_i S+_j`` and ``S+_j Sz_i`` produce identical matrix elements.
struct MixedTwoBody {
    std::uint64_t sz_site{0};       ///< Site carrying the Sz factor
    std::uint64_t flip_site{0};     ///< Site carrying the S+/S- factor
    std::uint8_t  flip_op_type{0};  ///< 0 = S+, 1 = S-
    Complex       coefficient{0.0, 0.0};
};

/// Two-body off-diagonal (S+/-_i S+/-_j). Flips two bits, both gated.
struct OffDiagTwoBody {
    std::uint64_t site_index_1{0};
    std::uint64_t site_index_2{0};
    std::uint8_t  op_type_1{0};  ///< 0 = S+, 1 = S-
    std::uint8_t  op_type_2{0};  ///< 0 = S+, 1 = S-
    Complex       coefficient{0.0, 0.0};
};

/// Three-body term of arbitrary mixed type (S^a_i S^b_j S^c_k).
struct ThreeBodyTerm {
    std::uint8_t  op_type_1{0};
    std::uint64_t site_index_1{0};
    std::uint8_t  op_type_2{0};
    std::uint64_t site_index_2{0};
    std::uint8_t  op_type_3{0};
    std::uint64_t site_index_3{0};
    Complex       coefficient{0.0, 0.0};
};

// -----------------------------------------------------------------------------
// TermStorage: SoA bins plus typed/classified setters.
//
// The struct is intentionally a plain data carrier (no virtuals, no caches).
// Owning operators ALWAYS invalidate their derived caches (assembled CSR,
// ``isReal()`` token, etc.) whenever they mutate this struct; the storage
// itself doesn't know about those caches.
// -----------------------------------------------------------------------------
struct TermStorage {
    std::vector<DiagOneBody>     diag_one_body;
    std::vector<OffDiagOneBody>  offdiag_one_body;
    std::vector<DiagTwoBody>     diag_two_body;
    std::vector<MixedTwoBody>    mixed_two_body;
    std::vector<OffDiagTwoBody>  offdiag_two_body;
    std::vector<ThreeBodyTerm>   three_body;

    // -------- Typed setters: zero-classification path ---------------------

    void add_diag_one_body(std::uint64_t site, Complex coeff) {
        diag_one_body.push_back({site, coeff});
    }
    void add_offdiag_one_body(std::uint64_t site, std::uint8_t op_type, Complex coeff) {
        offdiag_one_body.push_back({site, op_type, coeff});
    }
    void add_diag_two_body(std::uint64_t site_1, std::uint64_t site_2, Complex coeff) {
        diag_two_body.push_back({site_1, site_2, coeff});
    }
    void add_mixed_two_body(std::uint64_t sz_site, std::uint64_t flip_site,
                            std::uint8_t flip_op_type, Complex coeff) {
        mixed_two_body.push_back({sz_site, flip_site, flip_op_type, coeff});
    }
    void add_offdiag_two_body(std::uint64_t site_1, std::uint64_t site_2,
                              std::uint8_t op_type_1, std::uint8_t op_type_2, Complex coeff) {
        offdiag_two_body.push_back({site_1, site_2, op_type_1, op_type_2, coeff});
    }
    void add_three_body(std::uint8_t op_type_1, std::uint64_t site_1,
                        std::uint8_t op_type_2, std::uint64_t site_2,
                        std::uint8_t op_type_3, std::uint64_t site_3,
                        Complex coeff) {
        three_body.push_back({op_type_1, site_1, op_type_2, site_2,
                              op_type_3, site_3, coeff});
    }

    // -------- Classify-and-route helpers ----------------------------------
    //
    // The "raw" interface that mirrors the legacy ``TransformData`` shape.
    // Routes the term to the right SoA bin based on op_type{,_2} values.
    // This is exactly the classification logic that lived in the old
    // ``Operator::separateTransformsByType()`` -- now executed eagerly at
    // the point of insertion, with no lazy fan-out and no flag to keep in
    // sync with cache invalidation.

    /**
     * @brief Append a one-body term. `op_type` is the op encoding
     *        (0 = S+, 1 = S-, 2 = Sz). Sz routes to ``diag_one_body``;
     *        S+/S- route to ``offdiag_one_body``.
     */
    void add_one_body(std::uint8_t op_type, std::uint64_t site, Complex coeff) {
        if (op_type == 2) {
            add_diag_one_body(site, coeff);
        } else {
            if (op_type != 0 && op_type != 1) {
                throw std::invalid_argument(
                    "TermStorage::add_one_body: op_type must be 0 (S+), 1 (S-), "
                    "or 2 (Sz); got " + std::to_string(static_cast<int>(op_type)));
            }
            add_offdiag_one_body(site, op_type, coeff);
        }
    }

    /**
     * @brief Append a two-body term. Routes to:
     *          - ``diag_two_body``   if both ops are Sz
     *          - ``mixed_two_body``  if exactly one op is Sz
     *          - ``offdiag_two_body`` if neither op is Sz
     */
    void add_two_body(std::uint8_t op_type_1, std::uint64_t site_1,
                      std::uint8_t op_type_2, std::uint64_t site_2,
                      Complex coeff) {
        auto check = [](std::uint8_t t) {
            if (t > 2) {
                throw std::invalid_argument(
                    "TermStorage::add_two_body: op_type must be 0 (S+), 1 (S-), "
                    "or 2 (Sz); got " + std::to_string(static_cast<int>(t)));
            }
        };
        check(op_type_1);
        check(op_type_2);

        if (op_type_1 == 2 && op_type_2 == 2) {
            add_diag_two_body(site_1, site_2, coeff);
        } else if (op_type_1 == 2) {
            // Sz at site_1, S+/- at site_2.
            add_mixed_two_body(/*sz_site=*/site_1, /*flip_site=*/site_2,
                               /*flip_op_type=*/op_type_2, coeff);
        } else if (op_type_2 == 2) {
            // S+/- at site_1, Sz at site_2.
            add_mixed_two_body(/*sz_site=*/site_2, /*flip_site=*/site_1,
                               /*flip_op_type=*/op_type_1, coeff);
        } else {
            add_offdiag_two_body(site_1, site_2, op_type_1, op_type_2, coeff);
        }
    }

    // -------- Bulk operations ---------------------------------------------

    void clear() noexcept {
        diag_one_body.clear();
        offdiag_one_body.clear();
        diag_two_body.clear();
        mixed_two_body.clear();
        offdiag_two_body.clear();
        three_body.clear();
    }

    [[nodiscard]] std::size_t total_terms() const noexcept {
        return diag_one_body.size() + offdiag_one_body.size()
             + diag_two_body.size() + mixed_two_body.size()
             + offdiag_two_body.size() + three_body.size();
    }

    [[nodiscard]] bool empty() const noexcept { return total_terms() == 0; }

    /**
     * @brief Generic classify-and-route over an arbitrary AoS term iterable.
     *
     * @tparam Sink   Any object exposing the six typed setters used below
     *                (``add_diag_one_body``, ``add_offdiag_one_body``,
     *                ``add_diag_two_body``, ``add_mixed_two_body``,
     *                ``add_offdiag_two_body``, ``add_three_body``). Both
     *                ``ed::matvec::TermStorage`` (host) and the GPU's
     *                ``GpuSoaBinSink`` adapter (which forwards to the
     *                cuDoubleComplex SoA vectors) satisfy this concept.
     * @tparam Aos    Range of structs exposing ``op_type``, ``site_index``,
     *                ``coefficient``, ``site_index_2``, ``op_type_2``,
     *                ``is_two_body`` (the legacy ``TransformData`` shape;
     *                ``GPUTransformData`` also satisfies it).
     * @tparam Aos3   Range of structs matching the three-body schema:
     *                ``op_type_{1,2,3}``, ``site_index_{1,2,3}``,
     *                ``coefficient``.
     * @tparam Conv   Callable that converts an AoS scalar coefficient
     *                (e.g. ``std::complex<double>`` or ``cuDoubleComplex``)
     *                to the sink's coefficient type. Default = identity.
     *
     * This is the SINGLE source of truth for the
     *   op_type \in {0,1,2} -> {diag,offdiag,mixed} \times {one,two}body
     * decision tree. CPU ``commitPendingTransforms`` and GPU
     * ``separateTransformsByType`` both delegate here so the classification
     * cannot drift between backends.
     */
    template <class Sink, class Aos, class Aos3, class Conv>
    static void classify_route(Sink& sink, const Aos& aos, const Aos3& aos3,
                               Conv&& conv) {
        for (const auto& t : aos) {
            const auto coeff = conv(t.coefficient);
            if (!t.is_two_body) {
                if (t.op_type == 2) {
                    sink.add_diag_one_body(t.site_index, coeff);
                } else {
                    sink.add_offdiag_one_body(t.site_index, t.op_type, coeff);
                }
            } else {
                if (t.op_type == 2 && t.op_type_2 == 2) {
                    sink.add_diag_two_body(t.site_index, t.site_index_2, coeff);
                } else if (t.op_type == 2) {
                    sink.add_mixed_two_body(t.site_index, t.site_index_2,
                                            t.op_type_2, coeff);
                } else if (t.op_type_2 == 2) {
                    sink.add_mixed_two_body(t.site_index_2, t.site_index,
                                            t.op_type, coeff);
                } else {
                    sink.add_offdiag_two_body(t.site_index, t.site_index_2,
                                              t.op_type, t.op_type_2, coeff);
                }
            }
        }
        for (const auto& t : aos3) {
            sink.add_three_body(t.op_type_1, t.site_index_1,
                                t.op_type_2, t.site_index_2,
                                t.op_type_3, t.site_index_3,
                                conv(t.coefficient));
        }
    }

    /**
     * @brief Returns true iff every coupling in every bin is purely real
     *        (|imag| <= tol). Linear in the number of terms.
     */
    [[nodiscard]] bool is_real(double tol = 1e-15) const noexcept {
        auto real_run = [tol](const auto& vec) {
            for (const auto& t : vec) {
                if (std::abs(t.coefficient.imag()) > tol) return false;
            }
            return true;
        };
        return real_run(diag_one_body)
            && real_run(offdiag_one_body)
            && real_run(diag_two_body)
            && real_run(mixed_two_body)
            && real_run(offdiag_two_body)
            && real_run(three_body);
    }
};

} // namespace ed::matvec

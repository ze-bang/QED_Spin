// =============================================================================
// src/symmetry/streaming_symmetry_unified.cpp
//
// Wave 1 of the "Unify all 16 matvec cells under apply_terms<BasisPolicy,
// Scalar, Backend>" plan (May 2026).
//
// Provides the unified symmetric matvec entry points declared in
// ``streaming_symmetry.h``:
//
//   * StreamingSymmetryOperator::applySymmetrizedUnified(sector_idx,
//     in, out)
//   * StreamingSymmetryOperator::applySymmetrizedUnifiedReal(...)
//   * FixedSzStreamingSymmetryOperator::applySymmetrizedFixedSzUnified(...)
//   * FixedSzStreamingSymmetryOperator::applySymmetrizedFixedSzUnifiedReal(...)
//
// All four dispatch through ``ed::matvec::kernel::apply_terms`` with a
// ``SymmetryBasisPolicy`` (Wave 0 ABI extension): the kernel walks the
// orbit via ``iter_orbit`` and applies the per-emit
// ``conj(beta_{s'}) * group_norm / norm_k`` modifier via
// ``coeff_modifier`` so the bit-exact semantics of the legacy bespoke
// path (``applyHamiltonianTermsFullSpace`` / ``applyHamiltonianTerms``)
// are preserved.
//
// The legacy bespoke implementations remain in ``streaming_symmetry.h``
// behind ``ED_SYMMETRY_LEGACY_MATVEC`` for one release cycle as a
// fall-back / cross-check (see ``SectorView::apply`` /
// ``SectorView::apply_real``).
// =============================================================================

#include <ed/core/streaming_symmetry.h>
#include <ed/matvec/symmetry_basis_policy.h>
#include <ed/matvec/term_kernels.h>

#include <cstdlib>
#include <algorithm>

namespace {

// One-shot environment-variable read shared by every SectorView in the
// process. ``ED_SYMMETRY_LEGACY_MATVEC=1`` forces the bespoke pre-Wave-1
// path; default = unified.
inline bool read_legacy_symmetric_env_once() {
    const char* env = std::getenv("ED_SYMMETRY_LEGACY_MATVEC");
    if (env == nullptr) return false;
    if (env[0] == '\0') return false;
    if (env[0] == '0' && env[1] == '\0') return false;
    return true;
}

} // namespace

// =============================================================================
// Shared env-var gate
// =============================================================================

bool StreamingSymmetryOperator::useLegacySymmetricMatvec() {
    static const bool kLegacy = read_legacy_symmetric_env_once();
    return kLegacy;
}

// =============================================================================
// StreamingSymmetryOperator (full Hilbert, complex)
// =============================================================================

void StreamingSymmetryOperator::applySymmetrizedUnified(
    std::size_t sector_idx,
    const Complex* in,
    Complex* out) const
{
    if (sector_idx >= sectors_.size()) {
        throw std::runtime_error(
            "StreamingSymmetryOperator::applySymmetrizedUnified: "
            "invalid sector index");
    }

    const auto& sector            = sectors_[sector_idx];
    const std::size_t sector_dim  = sector.basis_states.size();
    const SectorLookupHandle lookup = makeSectorLookup_(sector_idx);
    const double group_size       = static_cast<double>(getGroupSize());

    // ``apply_terms`` atomic-adds into ``out``; caller-zeroed.
    std::fill(out, out + sector_dim, Complex(0.0, 0.0));

    commitPendingTransforms();

    const ed::matvec::basis::SymmetryBasisPolicy basis =
        ed::matvec::basis::make_symmetry_basis(sector, lookup, group_size);

    ed::matvec::kernel::apply_terms<
        ed::matvec::basis::SymmetryBasisPolicy, Complex>(
            basis,
            static_cast<double>(spin_l_),
            terms_.diag_one_body,
            terms_.offdiag_one_body,
            terms_.diag_two_body,
            terms_.mixed_two_body,
            terms_.offdiag_two_body,
            terms_.three_body,
            in,
            out);
}

void StreamingSymmetryOperator::applySymmetrizedUnifiedReal(
    std::size_t sector_idx,
    const double* in,
    double* out) const
{
    if (sector_idx >= sectors_.size()) {
        throw std::runtime_error(
            "StreamingSymmetryOperator::applySymmetrizedUnifiedReal: "
            "invalid sector index");
    }

    const auto& sector            = sectors_[sector_idx];
    const std::size_t sector_dim  = sector.basis_states.size();
    const SectorLookupHandle lookup = makeSectorLookup_(sector_idx);
    const double group_size       = static_cast<double>(getGroupSize());

    std::fill(out, out + sector_dim, 0.0);

    commitPendingTransforms();

    const ed::matvec::basis::SymmetryBasisPolicy basis =
        ed::matvec::basis::make_symmetry_basis(sector, lookup, group_size);

    ed::matvec::kernel::apply_terms<
        ed::matvec::basis::SymmetryBasisPolicy, double>(
            basis,
            static_cast<double>(spin_l_),
            terms_.diag_one_body,
            terms_.offdiag_one_body,
            terms_.diag_two_body,
            terms_.mixed_two_body,
            terms_.offdiag_two_body,
            terms_.three_body,
            in,
            out);
}

// =============================================================================
// FixedSzStreamingSymmetryOperator (Sz + symmetry, complex/real)
// =============================================================================

void FixedSzStreamingSymmetryOperator::applySymmetrizedFixedSzUnified(
    std::size_t sector_idx,
    const Complex* in,
    Complex* out) const
{
    if (sector_idx >= sectors_.size()) {
        throw std::runtime_error(
            "FixedSzStreamingSymmetryOperator::"
            "applySymmetrizedFixedSzUnified: invalid sector index");
    }

    const auto& sector            = sectors_[sector_idx];
    const std::size_t sector_dim  = sector.basis_states.size();
    const SectorLookupHandle lookup = makeSectorLookup_(sector_idx);
    const double group_size       = static_cast<double>(getGroupSize());

    std::fill(out, out + sector_dim, Complex(0.0, 0.0));

    commitPendingTransforms();

    const ed::matvec::basis::SymmetryBasisPolicy basis =
        ed::matvec::basis::make_symmetry_basis(sector, lookup, group_size);

    ed::matvec::kernel::apply_terms<
        ed::matvec::basis::SymmetryBasisPolicy, Complex>(
            basis,
            static_cast<double>(spin_l_),
            terms_.diag_one_body,
            terms_.offdiag_one_body,
            terms_.diag_two_body,
            terms_.mixed_two_body,
            terms_.offdiag_two_body,
            terms_.three_body,
            in,
            out);
}

void FixedSzStreamingSymmetryOperator::applySymmetrizedFixedSzUnifiedReal(
    std::size_t sector_idx,
    const double* in,
    double* out) const
{
    if (sector_idx >= sectors_.size()) {
        throw std::runtime_error(
            "FixedSzStreamingSymmetryOperator::"
            "applySymmetrizedFixedSzUnifiedReal: invalid sector index");
    }

    const auto& sector            = sectors_[sector_idx];
    const std::size_t sector_dim  = sector.basis_states.size();
    const SectorLookupHandle lookup = makeSectorLookup_(sector_idx);
    const double group_size       = static_cast<double>(getGroupSize());

    std::fill(out, out + sector_dim, 0.0);

    commitPendingTransforms();

    const ed::matvec::basis::SymmetryBasisPolicy basis =
        ed::matvec::basis::make_symmetry_basis(sector, lookup, group_size);

    ed::matvec::kernel::apply_terms<
        ed::matvec::basis::SymmetryBasisPolicy, double>(
            basis,
            static_cast<double>(spin_l_),
            terms_.diag_one_body,
            terms_.offdiag_one_body,
            terms_.diag_two_body,
            terms_.mixed_two_body,
            terms_.offdiag_two_body,
            terms_.three_body,
            in,
            out);
}

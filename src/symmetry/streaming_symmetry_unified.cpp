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
#include <ed/matvec/matvec_backend.h>
#include <ed/matvec/symmetry_basis_policy.h>
#include <ed/matvec/term_kernels.h>

#include <cstdlib>
#include <algorithm>

namespace {

// ---------------------------------------------------------------------------
// Operator-collapse refactor (P2b, Jun 2026): the unified symmetric matvec
// now drives the SAME host backend object the eventual
// ``Operator<SymmetryBasisPolicy, Host>`` collapse will hold --
// ``CpuMatVecBackend<SymmetryBasisPolicy, ...>`` -- instead of calling the
// ``apply_terms`` kernel directly. For the symmetry policy the backend
// always takes the matrix-free kernel (CSR is compiled out via
// ``if constexpr (needs_orbit_walk)``), so the result is bit-identical to
// the former direct call while exercising the production backend in the
// default solve path. The backend is constructed on the stack per call --
// it only stores a (POD) policy + tunables, so there is no heap traffic in
// the Lanczos / FTLM / TPQ inner loop.
// ---------------------------------------------------------------------------
using SymBackend = ed::matvec::CpuMatVecBackend<
    ed::matvec::basis::SymmetryBasisPolicy,
    ed::matvec::DiagOneBody, ed::matvec::OffDiagOneBody,
    ed::matvec::DiagTwoBody, ed::matvec::MixedTwoBody,
    ed::matvec::OffDiagTwoBody, ed::matvec::ThreeBodyTerm>;
using SymTermView = SymBackend::term_view_t;

inline void sym_backend_apply_complex(
    const ed::matvec::basis::SymmetryBasisPolicy& basis,
    const SymTermView& tv,
    const Complex* in, Complex* out, std::size_t dim)
{
    SymBackend backend(basis, ed::matvec::detail::read_symmetry_tunables(),
                       "CpuSymmetry(unified)");
    backend.apply_complex(&tv, in, out, dim);
}

inline void sym_backend_apply_real(
    const ed::matvec::basis::SymmetryBasisPolicy& basis,
    const SymTermView& tv,
    const double* in, double* out, std::size_t dim)
{
    SymBackend backend(basis, ed::matvec::detail::read_symmetry_tunables(),
                       "CpuSymmetry(unified)");
    backend.apply_real(&tv, in, out, dim);
}

} // namespace

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

    // ``term_view_()`` runs commitPendingTransforms() and stamps is_real;
    // the backend zero-fills ``out`` before the matrix-free scatter.
    const ed::matvec::basis::SymmetryBasisPolicy basis =
        ed::matvec::basis::make_symmetry_basis(sector, lookup, group_size);
    sym_backend_apply_complex(basis, term_view_(), in, out, sector_dim);
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

    const ed::matvec::basis::SymmetryBasisPolicy basis =
        ed::matvec::basis::make_symmetry_basis(sector, lookup, group_size);
    sym_backend_apply_real(basis, term_view_(), in, out, sector_dim);
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

    const ed::matvec::basis::SymmetryBasisPolicy basis =
        ed::matvec::basis::make_symmetry_basis(sector, lookup, group_size);
    sym_backend_apply_complex(basis, term_view_(), in, out, sector_dim);
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

    const ed::matvec::basis::SymmetryBasisPolicy basis =
        ed::matvec::basis::make_symmetry_basis(sector, lookup, group_size);
    sym_backend_apply_real(basis, term_view_(), in, out, sector_dim);
}

#pragma once
// =============================================================================
// include/ed/symmetry/sector_gpu_mirror.h
//
// Pure declaration of the reusable per-sector GPU matvec entry point.
//
// This header carries NO CUDA includes, so CPU translation units (e.g.
// ``sector_operator.h`` consumed by ed_core / ed_solvers_cpu) can include
// it freely. The definition lives in
// ``src/symmetry/streaming_symmetry_gpu_mirror.cu`` (compiled into
// ``ed_solvers_gpu`` when WITH_CUDA is ON); a throwing stub lives in the
// ``.cpp`` sibling for non-CUDA builds.
//
// ``make_sector_matvec_gpu`` builds a one-shot, fully device-resident
// mirror of a single ``SymmetrySector`` (orbit CSR + state->orbit lookup
// + term SoA) and returns a ``MatvecFn`` that runs the unified
// ``apply_terms_gpu_scatter<DeviceSymmetryBasisPolicy, cuDoubleComplex>``
// kernel on DEVICE pointers (per the ``bind_cuda()`` contract). The
// mirror is captured by the returned callable (build-once / reuse across
// the many matvecs of a Lanczos / FTLM sweep), mirroring the legacy
// ``StreamingSymmetryOperator::bind_cuda_for_sector`` lifetime.
//
// This is the GPU twin of the unified CPU symmetry backend that drives
// ``ed::symmetry::SectorOperator`` -- it makes a standalone
// ``SectorOperator`` backend-complete (CPU + GPU) versus the legacy
// ``SectorView``.
//
// Phase A of the operator-collapse GPU-parity work (Jun 2026).
// =============================================================================

#include <cstdint>

#include <ed/core/linear_operator.h>     // ed::LinearOperator::MatvecFn
#include <ed/core/streaming_symmetry.h>  // ::SymmetrySector
#include <ed/matvec/term_storage.h>      // ed::matvec::TermStorage
#include <ed/symmetry/rep_sector_data.h> // ed::symmetry::RepSectorData

namespace ed::symmetry {

/// Build a device-resident mirror of ``sector`` (orbit CSR + lookup +
/// term SoA from ``terms``) and return a complex matvec callable taking
/// DEVICE pointers. ``group_size`` is |G| (sets ``group_norm = 1/|G|``);
/// ``spin_l`` / ``n_sites`` are the lattice constants the term kernel
/// needs. ``n_up >= 0`` selects the dense rank-table lookup (Sz-conserving
/// sectors); ``n_up = -1`` (default) uses the legacy open-addressing hash,
/// which is correct for both full-Hilbert and fixed-Sz sectors.
///
/// On a non-CUDA build this throws ``std::logic_error``; callers reach it
/// only through ``select_backend``, which never picks the GPU lane unless
/// ``Geometry::supports_device_matvec`` is set (WITH_CUDA only).
ed::LinearOperator::MatvecFn
make_sector_matvec_gpu(const ::SymmetrySector&         sector,
                       double                          group_size,
                       double                          spin_l,
                       const ed::matvec::TermStorage&  terms,
                       int                             n_sites,
                       int                             n_up = -1);

/// Build a RESIDENT on-the-fly representative GPU matvec for one symmetry
/// sector ("On-the-fly representative SpMV" plan, Jun 2026). Consumes a
/// CSR-free ``RepSectorData`` (representatives + ``1/norm`` + the |G|
/// per-sector characters + the group permutations) and returns a complex
/// matvec callable taking DEVICE pointers. Unlike ``make_sector_matvec_gpu``
/// this allocates NO orbit CSR and NO O(full-Sz-dim) projection table: the
/// group action + projection are regenerated arithmetically on the device,
/// so per-SpMV traffic is just the in/out vectors (the genuine /|G| win for
/// the N=32 Sz+Symm mTPQ run).
///
/// Requires ``rep.usable()`` (fixed-Sz sector, ``n_up >= 0``). On a non-CUDA
/// build this throws ``std::logic_error``.
ed::LinearOperator::MatvecFn
make_sector_matvec_gpu_rep(const RepSectorData&            rep,
                           double                          spin_l,
                           const ed::matvec::TermStorage&  terms);

} // namespace ed::symmetry

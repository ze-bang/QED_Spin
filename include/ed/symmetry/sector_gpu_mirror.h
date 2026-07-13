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
// Stage 11c-2b (Jul 2026): the legacy orbit-CSR device mirror
// (``make_sector_matvec_gpu`` + GpuSectorMirror) was retired together with
// its ``ED_GPU_SYMMETRY_REP=0`` escape -- the on-the-fly representative
// mirror below is THE device representation for symmetry sectors.
// =============================================================================

#include <ed/core/linear_operator.h>     // ed::LinearOperator::MatvecFn
#include <ed/matvec/term_storage.h>      // ed::matvec::TermStorage
#include <ed/symmetry/rep_sector_data.h> // ed::symmetry::RepSectorData

namespace ed::symmetry {

/// Build a RESIDENT on-the-fly representative GPU matvec for one symmetry
/// sector ("On-the-fly representative SpMV" plan, Jun 2026). Consumes a
/// CSR-free ``RepSectorData`` (representatives + ``1/norm`` + the |G|
/// per-sector characters + the group permutations) and returns a complex
/// matvec callable taking DEVICE pointers. Allocates NO orbit CSR and NO
/// O(full-Sz-dim) projection table: the group action + projection are
/// regenerated arithmetically on the device, so per-SpMV traffic is just
/// the in/out vectors (the genuine /|G| win for the N=32 Sz+Symm mTPQ run).
///
/// Requires ``rep.usable()``; fixed-Sz sectors use the combinadic rank
/// reverse lookup, full-Hilbert (sym-only) sectors the identity rank.
/// On a non-CUDA build this throws ``std::logic_error``; callers reach it
/// only through ``select_backend``, which never picks the GPU lane unless
/// ``Geometry::supports_device_matvec`` is set (WITH_CUDA only).
ed::LinearOperator::MatvecFn
make_sector_matvec_gpu_rep(const RepSectorData&            rep,
                           double                          spin_l,
                           const ed::matvec::TermStorage&  terms);

/// HOST-pointer twin of ``make_sector_matvec_gpu_rep`` for callers whose
/// Krylov loop keeps its vectors in host RAM (the little-group engine's
/// CPU Lanczos): owns persistent device in/out buffers and stages one
/// H2D + D2H copy per apply. The staging cost is O(dim) against the
/// kernel's O(dim * terms * |G|) walk, so it is negligible for the large
/// sectors this exists for. Same non-CUDA / no-device failure contract
/// as the device-pointer factory.
ed::LinearOperator::MatvecFn
make_sector_matvec_gpu_rep_hostptr(const RepSectorData&            rep,
                                   double                          spin_l,
                                   const ed::matvec::TermStorage&  terms);

} // namespace ed::symmetry

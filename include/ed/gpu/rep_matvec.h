#pragma once
// =============================================================================
// include/ed/gpu/rep_matvec.h
//
// Pure declaration of the reusable per-sector GPU matvec entry point.
//
// This header carries NO CUDA includes, so CPU translation units (in
// ed_core / ed_solvers_cpu) can include it freely. The definition lives in
// ``src/gpu/rep_matvec.cu`` (compiled into
// ``ed_solvers_gpu`` when WITH_CUDA is ON); a throwing stub lives in the
// ``.cpp`` sibling for non-CUDA builds.
//
// The on-the-fly representative mirror below is the only device
// representation for symmetry sectors.
// =============================================================================

#include <ed/matvec/linear_operator.h>   // ed::LinearOperator::MatvecFn
#include <ed/ops/program.h>              // ed::ops::MaskedProgram
#include <ed/basis/rep_sector.h>         // ed::symmetry::RepSectorData

namespace ed::symmetry {

/// Build a RESIDENT on-the-fly representative GPU matvec for one symmetry
/// sector. Consumes a CSR-free ``RepSectorData`` (representatives +
/// ``1/norm`` + the |G| per-sector characters + the group permutations) and the
/// operator's row program (Operator::row_program: the walk of the adjoint, sector_rows.h)
/// and returns a complex matvec callable taking DEVICE pointers. Allocates NO
/// orbit CSR and NO O(full-Sz-dim) projection table: the group action +
/// projection are regenerated arithmetically on the device, so per-SpMV
/// traffic is just the in/out vectors (a genuine 1/|G| memory saving).
///
/// Requires ``rep.usable()``; fixed-Sz sectors use the combinadic rank
/// reverse lookup, full-Hilbert (sym-only) sectors the identity rank.
/// On a non-CUDA build this throws ``std::logic_error``; callers reach it
/// only through ``bind_cuda`` of an operator whose ``has_device_kernel()`` is true, which
/// needs WITH_CUDA.
ed::LinearOperator::MatvecFn
make_sector_matvec_gpu_rep(const RepSectorData&            rep,
                           const ed::ops::MaskedProgram&   rows);

/// O from sector `src` to sector `tgt` on the device: the rows of the target, each walk
/// target looked up in the source (``rows`` = compile_program({O^dagger}, tgt, src), the
/// host CrossSectorMatVec's program). Takes device pointers (in: src dim, out: tgt dim).
ed::LinearOperator::MatvecFn
make_cross_matvec_gpu_rep(const RepSectorData&            src,
                          const RepSectorData&            tgt,
                          const ed::ops::MaskedProgram&   rows);

/// The same sector matvec on k vectors at once (device pointers ins[i] -> outs[i]): one walk
/// over each row's terms and orbit lookups serves up to 8 vectors, and every output equals
/// the single-vector apply bit for bit.
ed::LinearOperator::MultiMatvecFn
make_sector_matvec_gpu_rep_multi(const RepSectorData&            rep,
                                 const ed::ops::MaskedProgram&   rows);

/// HOST-pointer twin of ``make_sector_matvec_gpu_rep`` for callers whose
/// Krylov loop keeps its vectors in host RAM (the little-group engine's
/// CPU Lanczos): owns persistent device in/out buffers and stages one
/// H2D + D2H copy per apply. The staging cost is O(dim) against the
/// kernel's O(dim * terms * |G|) walk, so it is negligible for the large
/// sectors this exists for. Same non-CUDA / no-device failure contract
/// as the device-pointer factory.
ed::LinearOperator::MatvecFn
make_sector_matvec_gpu_rep_hostptr(const RepSectorData&            rep,
                                   const ed::ops::MaskedProgram&   rows);

} // namespace ed::symmetry

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
// Two device representations of a symmetry sector's operator: the on-the-fly
// representative gather (no matrix), and the reduced CSR built on the device
// whenever it fits (build_sector_csr_gpu).
// =============================================================================

#include <ed/matvec/linear_operator.h>   // ed::LinearOperator::MatvecFn
#include <ed/matvec/reduced_csr.h>       // ed::matvec::ReducedSymmetryCsr
#include <ed/ops/program.h>              // ed::ops::MaskedProgram
#include <ed/basis/rep_sector.h>         // ed::symmetry::RepSectorData

#include <complex>
#include <cstdint>
#include <memory>

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

/// An operator's reduced CSR on one sector, built and kept on the device (P7.1). Opaque here;
/// freed with its last holder.
struct DeviceCsr;

/// What a device CSR holds and what its build took.
struct DeviceCsrInfo {
    std::uint64_t nnz     = 0;
    std::uint64_t bytes   = 0;
    double        build_s = 0.0;
    int           lanes   = 0;   ///< threads per row of its SpMV
};

/// The reduced CSR of the operator whose row program is ``rows`` on the 1-dim sector ``rep``,
/// built ON THE DEVICE from the walk the device gather runs: the entries of sector_rows.h
/// build_sector_csr (merged by column in emission order, exact zeros dropped), its values in a
/// dictionary up to kCsrDictMax distinct ones, else whole. Null when it would take more than
/// ``max_bytes`` of device memory, the device is out of memory, or the sector is not 1-dim.
std::shared_ptr<const DeviceCsr>
build_sector_csr_gpu(const RepSectorData&            rep,
                     const ed::ops::MaskedProgram&   rows,
                     std::uint64_t                   max_bytes);

/// A reduced CSR built on the host, uploaded as is (P7.5: the device kernel of a sector of an
/// irrep of dimension > 1, whose rows the host builds). Null when it holds more than `max_bytes`
/// or the device is out of memory.
std::shared_ptr<const DeviceCsr>
upload_csr_gpu(const ed::matvec::ReducedSymmetryCsr<std::complex<double>>& csr, std::uint64_t max_bytes);

/// Its applies on device pointers: one vector, and k at once (every output of the k-vector
/// apply equals the single apply bit for bit).
ed::LinearOperator::MatvecFn      csr_matvec_gpu(std::shared_ptr<const DeviceCsr> csr);
ed::LinearOperator::MultiMatvecFn csr_matvec_gpu_multi(std::shared_ptr<const DeviceCsr> csr);

DeviceCsrInfo device_csr_info(const DeviceCsr& csr);

/// Its entries copied to the host (tests).
ed::matvec::ReducedSymmetryCsr<std::complex<double>> download_csr(const DeviceCsr& csr);

} // namespace ed::symmetry

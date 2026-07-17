// =============================================================================
// src/matvec/cpu_backend_instantiations.cpp
//
// P6 of the operator-collapse refactor (Jun 2026): EXPLICIT INSTANTIATION
// translation unit for the host (MemSpace=Host) leg of the
// Operator<BasisPolicy, MemSpace> grid.
//
// ``CpuMatVecBackend<BasisPolicy, ...6 term-view bins>`` is a heavy template:
// it carries the matrix-free kernel (apply_terms<Policy,Scalar> for Complex
// AND double), the Eigen CSR assemble/cache machinery, and the real-input
// fast path. Before this TU every header that instantiated a backend
// (operator.h Full lane, fixed_sz_operator.h FixedSz lane, sector_operator.h
// + the streaming-symmetry unified .cpp Symmetry lane) re-instantiated the
// whole tree, multiplying compile time across the codebase.
//
// This TU instantiates each of the THREE host cells of the grid exactly once,
// over the single canonical term-view shape that every Operator uses (the six
// SoA record types from term_storage.h). Pairing these explicit definitions
// with ``extern template`` declarations in a later increment lets the rest of
// the build consume the prebuilt symbols instead of recompiling them; on its
// own this TU is a compile-coverage proof that all three host cells are
// coherent and instantiable as a standalone library object.
//
// The three host cells:
//   * cell 1H (Full)              -- FullBasisPolicy
//   * cell 2H (FixedSz)           -- FixedSzBasisPolicy
//   * cell 3H/4H (Symmetry / +Sz) -- SymmetryBasisPolicy (orbit-walk lane;
//                                    needs_orbit_walk==true compiles out the
//                                    CSR + real-fast-path branches)
//
// The CUDA host cells live in cuda_matvec_backend.cuh (compiled only under
// WITH_CUDA), and the DistributedHost lane is MpiMatVecImpl (header-only,
// WITH_MPI) -- neither is a CpuMatVecBackend specialization, so both are
// out of scope for this TU.
// =============================================================================

#include <ed/matvec/basis_policy.h>
#include <ed/matvec/matvec_backend.h>
#include <ed/matvec/symmetry_basis_policy.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/matvec/term_storage.h>

namespace ed::matvec {

// The canonical term-view shape: the six SoA record types every Operator
// passes to its backend factory (see Operator::DiagonalOneBody ... aliases
// in ed/core/operator.h, all of which resolve to these ed::matvec types).
template class CpuMatVecBackend<basis::FullBasisPolicy,
                                DiagOneBody, OffDiagOneBody, DiagTwoBody,
                                MixedTwoBody, OffDiagTwoBody, ThreeBodyTerm>;

template class CpuMatVecBackend<basis::FixedSzBasisPolicy,
                                DiagOneBody, OffDiagOneBody, DiagTwoBody,
                                MixedTwoBody, OffDiagTwoBody, ThreeBodyTerm>;


// cell 5H (RepSymmetry) -- on-the-fly representative SpMV (Jun 2026). The
// rep policy forces the complex matrix-free path (is_rep_symmetry==true
// compiles out the CSR + real-input fast-path branches) and dispatches the
// dedicated ``apply_terms_rep_symmetry`` kernel.
template class CpuMatVecBackend<basis::RepSymmetryBasisPolicy,
                                DiagOneBody, OffDiagOneBody, DiagTwoBody,
                                MixedTwoBody, OffDiagTwoBody, ThreeBodyTerm>;

}  // namespace ed::matvec

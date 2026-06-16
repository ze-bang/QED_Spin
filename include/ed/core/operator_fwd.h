#pragma once
// =============================================================================
// include/ed/core/operator_fwd.h
//
// Forward declarations for the collapsed operator aliases (operator-collapse
// Phase 4). Lets a header reference ``FixedSzOperator`` / ``SectorOperator``
// as INCOMPLETE types (via pointers / shared_ptr) without pulling the full
// ``subspace_operator.h``. Include the full header to actually use the type.
//
// Both aliases name instantiations of the single ``ed::SubspaceOperator``
// template; this header forward-declares that template and the policy structs
// so the aliases resolve. The default ``MemorySpace`` template argument is
// supplied by the full definition in ``subspace_operator.h``.
// =============================================================================

#include <ed/matvec/memory_space.h>

namespace ed::matvec::basis {
struct FixedSzBasisPolicy;
struct SymmetryBasisPolicy;
}  // namespace ed::matvec::basis

namespace ed {
template <class BasisPolicy, ed::matvec::MemorySpace MS>
class SubspaceOperator;
}  // namespace ed

using FixedSzOperator =
    ed::SubspaceOperator<ed::matvec::basis::FixedSzBasisPolicy,
                         ed::matvec::MemorySpace::Host>;

namespace ed::symmetry {
using SectorOperator =
    ed::SubspaceOperator<ed::matvec::basis::SymmetryBasisPolicy,
                         ed::matvec::MemorySpace::Host>;
}  // namespace ed::symmetry

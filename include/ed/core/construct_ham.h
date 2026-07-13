#pragma once
// =============================================================================
// include/ed/core/construct_ham.h  — BACKWARD-COMPATIBLE UMBRELLA HEADER
//
// This file was previously a 5422-line monolith.  Its contents have been
// split into focused single-responsibility headers; this file now simply
// includes them in dependency order so that every existing consumer that
// does `#include <ed/core/construct_ham.h>` continues to compile unchanged.
//
// New code should include only the specific header it needs:
//
//   #include <ed/core/basis_utils.h>           // popcount, generateFixedSzBasis, LinIndexTable
//   #include <ed/core/symmetry_metadata.h>     // SectorMetadata, SymmetryGroupInfo
//   #include <ed/core/operator.h>              // Operator (full Hilbert space)
//   #include <ed/core/fixed_sz_operator.h>     // FixedSzOperator
//   #include <ed/core/operator_builders.h>     // ed::ops::add_sum / add_single_site / ...
//   #include <ed/operators/operators.h>         // lightweight MatVec operator factories
// =============================================================================

#include <ed/core/basis_utils.h>
#include <ed/core/symmetry_metadata.h>
#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator_builders.h>

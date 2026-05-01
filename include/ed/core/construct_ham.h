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
//   #include <ed/core/operator_types.h>        // SingleSiteOperator, SumOperator, ...
//   #include <ed/core/fixed_sz_operator_types.h> // FixedSzSingleSiteOperator, ...
//   #include <ed/operators/operators.h>         // lightweight MatVec operator factories
// =============================================================================

#include <ed/core/basis_utils.h>
#include <ed/core/symmetry_metadata.h>
#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator_types.h>
#include <ed/core/fixed_sz_operator_types.h>

// ---------------------------------------------------------------------------
// Legacy result types kept here for backward compatibility.
// These are still referenced by src/solvers/cpu/dynamics.cpp,
// src/solvers/cpu/TPQ.cpp, and include/ed/solvers/dynamics.h.
// Do not remove until those callers are updated.
// ---------------------------------------------------------------------------

// χ(ω) = ∑_{n,m} (p_m - p_n) * |<n|A|m>|^2 / (ω - (E_n - E_m) + iη)
struct DynamicalSusceptibilityData {
    std::vector<double> frequencies;
    std::vector<std::complex<double>> chi;
};

// A(ω) = Σ_{n,m} |<n|O|m>|^2 δ(ω − (E_n − E_m)) · weight(m)
struct SpectralFunctionData {
    std::vector<double> frequencies;
    std::vector<std::complex<double>> spectral_function;
};

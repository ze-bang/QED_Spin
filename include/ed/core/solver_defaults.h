#pragma once
// =============================================================================
// include/ed/core/solver_defaults.h -- Stage 11a: solver defaults defined ONCE.
//
// These knobs were declared with matching literals in EDParameters, the
// per-method option structs (ftlm.h / ltlm.h), and inline fallbacks in
// orchestrator.cpp -- kept in sync BY COMMENT ("matches EDParameters::...").
// Matching-by-comment is the twin-drift class; matching-by-symbol is free.
// =============================================================================

#include <cstdint>

namespace ed::defaults {

/// FTLM / OFTLM Krylov subspace dimension.
inline constexpr std::uint64_t kFtlmKrylovDim = 100;

/// LTLM Krylov subspace dimension.
inline constexpr std::uint64_t kLtlmKrylovDim = 200;

/// Full reorthogonalization in the thermal Lanczos passes (CPU default).
inline constexpr bool kThermalFullReorth = true;

}  // namespace ed::defaults

#pragma once
// =============================================================================
// include/ed/orchestrator.h
//
// TRANSITIONAL (P2.4 C6 deletes it): one entry point over any `ed::LinearOperator`,
//
//     ed::solve -- ground-state eigenproblem (Lanczos / Krylov-Schur / full diag)
//                  on the Backend lane select_backend picks (CPU or CUDA),
//
// returning `GroundStateResult` with backend metadata and Krylov diagnostics.
// eigs runs it on device-placed blocks. Implementation: src/orchestrator/.
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ed/core/linear_operator.h>
#include <ed/core/results.h>
#include <ed/core/select_backend.h>

// The entry point lives under `ed::workflows::` (`ed::solve` is a top-level alias).
namespace ed::workflows {

enum class SolveMethod : std::uint8_t {
    Auto = 0,          ///< pick from dim + num_eigs
    Lanczos = 1,
    KrylovSchur = 3,
    FullDiag = 4,
};

struct SolveOptions {
    std::size_t num_eigs       = 1;
    std::size_t max_iter       = 0;        ///< 0 => auto
    double      tolerance      = 1e-10;
    bool        compute_vectors = false;
    SolveMethod method         = SolveMethod::Auto;
    BackendConstraints backend;
};

// ---------------------------------------------------------------------------
// Public entry points.
//
// `ed::solve` -- ground-state eigenproblem. Picks Lanczos / Krylov-Schur /
//                full-diag based on (num_eigs, geometry().global_dim, opts.method),
//                constructs the Backend via `select_backend(H.geometry(),
//                opts.backend)` and returns the BackendMetadata that fired.
// ---------------------------------------------------------------------------

GroundStateResult solve(const LinearOperator&  H,
                         SolveOptions           opts = {});

}  // namespace ed::workflows

namespace ed {
// Top-level alias for the entry point.
using ed::workflows::solve;

// Option structs and result enums get top-level aliases so callers can
// write `ed::SolveOptions` without reaching into `ed::workflows::`.
using ed::workflows::SolveOptions;
using ed::workflows::SolveMethod;
}  // namespace ed

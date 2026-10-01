#pragma once
// =============================================================================
// include/ed/core/results.h
//
// TRANSITIONAL (P2.4 C6 deletes it): the result of `ed::workflows::solve`, with
// the Backend identity carried in a `BackendMetadata` blob.
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ed {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// BackendMetadata --- carries the runtime identity of the lane that
// produced a result, so all orchestrators return the SAME struct
// regardless of which backend ran the kernel.
// ---------------------------------------------------------------------------
struct BackendMetadata {
    /// One of: "cpu", "gpu". Set by the orchestrator
    /// when constructing the result: where the kernel actually ran.
    std::string  lane         = "cpu";
    /// The result came from a dense diagonalisation, not a Krylov run.
    bool         dense        = false;
    double       wall_seconds = 0.0;
    /// Free-form key=value diagnostics (e.g. memory hwm, reorth count).
    std::vector<std::pair<std::string, std::string>> notes;
};

// ---------------------------------------------------------------------------
// KrylovDiagnostics --- the Lanczos / Krylov-Schur
// internals every orchestrator carries through. Set sparingly --- callers
// that only want eigenvalues need not inspect this.
// ---------------------------------------------------------------------------
struct KrylovDiagnostics {
    std::vector<double> alpha;
    std::vector<double> beta;
    /// Number of Lanczos / Krylov-Schur iterations actually performed.
    std::size_t         iters_done    = 0;
    /// L2 norm of the residual after the final step (`beta_last`).
    double              residual_norm = 0.0;
    /// Per-Ritz-value residual estimates `|beta_last * y[m-1, k]|`.
    std::vector<double> ritz_residuals;
    /// Did the kernel converge to the requested tolerance?
    bool                converged     = false;
};

// ---------------------------------------------------------------------------
// EigenvectorRef --- an opaque handle to the eigenvectors produced by
// a `solve` call: host vectors, or a flag saying they stayed in backend
// memory.
// ---------------------------------------------------------------------------
struct EigenvectorRef {
    /// Host-side storage, one vector per eigenvalue.
    std::vector<std::vector<Complex>>  host;
};

// ---------------------------------------------------------------------------
// GroundStateResult --- output of `ed::solve(H, opts)`.
// ---------------------------------------------------------------------------
struct GroundStateResult {
    std::vector<double>           eigenvalues;
    std::optional<EigenvectorRef> eigenvectors;
    KrylovDiagnostics             krylov;
    BackendMetadata               backend;
};

}  // namespace ed

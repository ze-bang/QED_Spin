#pragma once
// =============================================================================
// include/ed/core/results.h
//
// Result types of the orchestrators (`ed::solve`, `ed::thermal`): one shape
// per workflow, with the Backend identity carried in a `BackendMetadata`
// blob so downstream consumers can branch on lane without re-reading the
// function signature.
//
// Phase 3.3 of the Minimalist ED Collapse (May 2026).
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <ed/core/thermal_types.h>  // ThermodynamicData, FTLMResults

namespace ed {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// BackendMetadata --- carries the runtime identity of the lane that
// produced a result. Replaces the implicit knowledge previously
// embedded in the result type itself ("EDResults" => CPU,
// "DistributedLanczosGPUResult" => MPI+GPU). Lets all orchestrators
// return the SAME struct regardless of which backend ran the kernel.
// ---------------------------------------------------------------------------
struct BackendMetadata {
    /// One of: "cpu", "gpu". Set by the orchestrator
    /// when constructing the result.
    std::string  lane         = "cpu";
    std::size_t  cuda_devices = 0;
    double       wall_seconds = 0.0;
    /// Free-form key=value diagnostics (e.g. memory hwm, reorth count).
    std::vector<std::pair<std::string, std::string>> notes;
};

// ---------------------------------------------------------------------------
// KrylovDiagnostics --- the Lanczos / Krylov-Schur
// internals every orchestrator carries through. Replaces the bespoke
// `tridiag_alpha` / `tridiag_eigenvalues` fields the existing distributed
// result types each spelled differently. Set sparingly --- callers that
// only want eigenvalues need not inspect this.
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
    /// Boolean flag set when the kernel computed eigenvectors but
    /// returned them only on the originating backend's memory (caller
    /// can extract via the LinearOperator + Backend).
    bool                                on_backend = false;
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

// ---------------------------------------------------------------------------
// ThermalResult --- output of `ed::thermal(H, opts)`. Folds the FTLM /
// OFTLM / mTPQ family.
// ---------------------------------------------------------------------------
struct ThermalResult {
    /// Combined (across samples) thermodynamic functions.
    ThermodynamicData                thermo;
    /// Ground-state energy for diagnostic / shift purposes.
    double                           ground_state_energy = 0.0;
    /// Optional FTLM raw results (Ritz triples per sample). Empty
    /// for TPQ lanes.
    std::optional<FTLMResults>       ftlm;
    KrylovDiagnostics                krylov;
    BackendMetadata                  backend;
};

}  // namespace ed

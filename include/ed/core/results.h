#pragma once
// =============================================================================
// include/ed/core/results.h
//
// Unified result types for the Phase-4 orchestrators (`ed::solve`,
// `ed::thermal`, `ed::spectral`). Folds the four legacy per-deployment
// result structs --- `EDResults`, `DistributedLanczosResult`,
// `DistributedLanczosGPUResult`, `DistributedEigenpairsResult` --- into
// a single shape per workflow, with the Backend identity carried in a
// `BackendMetadata` blob so downstream consumers can branch on lane
// without re-reading the function signature.
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
    /// One of: "cpu", "mpi", "gpu", "mpi_gpu". Set by the orchestrator
    /// when constructing the result.
    std::string  lane         = "cpu";
    std::size_t  mpi_size     = 1;
    std::size_t  cuda_devices = 0;
    double       wall_seconds = 0.0;
    /// Free-form key=value diagnostics (e.g. memory hwm, reorth count).
    std::vector<std::pair<std::string, std::string>> notes;
};

// ---------------------------------------------------------------------------
// KrylovDiagnostics --- the Lanczos / Krylov-Schur / Block-Lanczos
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
// a `solve` call. The orchestrator may choose any of three storage
// strategies; the caller switches on the active member.
// ---------------------------------------------------------------------------
struct EigenvectorRef {
    /// Host-side storage, one vector per eigenvalue.
    std::vector<std::vector<Complex>>  host;
    /// HDF5 file path where eigenvectors were persisted (in lieu of
    /// host storage, for memory-bound runs).
    std::string                         hdf5_path;
    /// Boolean flag set when the kernel computed eigenvectors but
    /// returned them only on the originating backend's memory (caller
    /// can extract via the LinearOperator + Backend).
    bool                                on_backend = false;
};

// ---------------------------------------------------------------------------
// GroundStateResult --- output of `ed::solve(H, opts)`. Replaces
// `EDResults`, `DistributedLanczosResult`, `DistributedLanczosGPUResult`,
// `DistributedEigenpairsResult`.
// ---------------------------------------------------------------------------
struct GroundStateResult {
    std::vector<double>           eigenvalues;
    std::optional<EigenvectorRef> eigenvectors;
    KrylovDiagnostics             krylov;
    BackendMetadata               backend;
    /// Optional HDF5 result file path that mirrors the legacy
    /// `HDF5IO::saveDiagonalizationResults` output. Empty when
    /// eigenvectors weren't requested AND no auto-save was triggered.
    std::string                   hdf5_path;
};

// ---------------------------------------------------------------------------
// Per-sector thermodynamics entry. Mirrors the legacy
// `ThermalSectorEntry` (re-exported here so the orchestrator does not
// need to drag in the full operator header).
// ---------------------------------------------------------------------------
struct ThermalSectorEntry {
    int                 sz_index = 0;
    double              ground_state_energy = 0.0;
    ThermodynamicData   thermo;
    /// Free-form per-sector diagnostics.
    std::vector<std::pair<std::string, std::string>> notes;
};

// ---------------------------------------------------------------------------
// ThermalResult --- output of `ed::thermal(H, opts)`. Folds the FTLM /
// LTLM / mTPQ / cTPQ / KPM-DOS family.
// ---------------------------------------------------------------------------
struct ThermalResult {
    /// Combined (across sectors / samples) thermodynamic functions.
    ThermodynamicData                thermo;
    /// Per-sector breakdown when the workflow ran a multi-Sz loop.
    std::vector<ThermalSectorEntry>  per_sector;
    /// Ground-state energy for diagnostic / shift purposes.
    double                           ground_state_energy = 0.0;
    /// Optional FTLM raw results (Ritz triples per sample). Empty
    /// for TPQ / KPM-DOS lanes.
    std::optional<FTLMResults>       ftlm;
    KrylovDiagnostics                krylov;
    BackendMetadata                  backend;
    std::string                      hdf5_path;
};

// ---------------------------------------------------------------------------
// SpectralResult --- output of `ed::spectral(H, observables, opts)`.
// Carries the (omega, S(omega)) grid plus error bars.
// ---------------------------------------------------------------------------
struct SpectralResult {
    std::vector<double> omega;
    std::vector<double> S_real;
    std::vector<double> S_imag;
    /// Stochastic error bars (only populated for FTLM / sample-based lanes).
    std::vector<double> errors_real;
    std::vector<double> errors_imag;
    KrylovDiagnostics   krylov;
    BackendMetadata     backend;
    std::string         hdf5_path;
};

}  // namespace ed

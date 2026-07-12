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
    /// How many returned eigenvalues met the tolerance (<= eigenvalues.size()).
    std::size_t         n_converged   = 0;
    /// Convergence curve: residual of the worst target / first unconverged Ritz
    /// pair after each iteration (block Lanczos) / restart (block Krylov-Schur).
    std::vector<double> resid_history;
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
// SectorTag --- compact quantum-number label attached to per-sector
// output by the streaming-symmetry workflows. Lets downstream consumers
// (CLI table, Python, HDF5) print an eigenvalue / thermo block / S(omega)
// alongside the (sector_index, irrep, Sz, dim) it came from instead of
// the legacy "anonymous sorted doubles" payload.
//
// `quantum_numbers` mirrors `ed::SymmetrySector::quantum_numbers`
// (defined in `ed/core/streaming_symmetry.h`): when the
// automorphism_results/ directory provides per-sector momentum / point-
// group labels, those land in `quantum_numbers` element-wise. The
// orchestrator does not interpret the entries; it just carries them
// through so the caller's downstream code can index into the labels
// they care about.
// ---------------------------------------------------------------------------
struct SectorTag {
    /// Linear sector index in `[0, num_sectors)`. The order matches the
    /// `make_operator(streaming_symmetry=true)` loop and the per-sector
    /// HDF5 directory names (``sector_<sector_index>/``).
    std::size_t      sector_index = 0;
    /// Dimension of the per-sector basis (number of orbits in the
    /// symmetry block; equals `SectorView::dim()`).
    std::uint64_t    sector_dim   = 0;
    /// Per-sector quantum numbers (e.g. ``[k_x, k_y, ..., irrep_id]``).
    /// Empty when no automorphism metadata is attached.
    std::vector<int> quantum_numbers;
    /// Number of "up" spins for the fixed-Sz sub-axis. -1 means
    /// "fixed-Sz axis is off" (full Hilbert per irrep).
    int              n_up         = -1;
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

    // -----------------------------------------------------------------
    // Streaming-symmetry attribution (May 2026 SOTA upgrade).
    // -----------------------------------------------------------------
    //
    // When the result was produced by a streaming-symmetry workflow
    // (CLI ``run_streaming_symmetry_workflow`` or Python
    // ``_core.workflows_solve_streaming_symmetry_directory``), these
    // fields tag every eigenvalue with the sector it came from. They
    // are EMPTY for the orchestrator-only `ed::workflows::solve(*op,
    // opts)` lane (which operates on a single `LinearOperator`).
    //
    // Invariants when non-empty:
    //   * `sector_tags.size() == num_sectors_touched` (one entry per
    //     non-empty sector that contributed).
    //   * `eigenvalues_per_sector[k].size()` is the number of
    //     eigenvalues sector `k` contributed before the global merge.
    //   * `sector_index_of_eigenvalue[i]` indexes into `sector_tags`
    //     and tells you which sector each entry of `eigenvalues`
    //     originated from (parallel to `eigenvalues`).
    //
    // The legacy "sorted doubles only" payload remains in `eigenvalues`
    // unchanged; consumers that do not care about provenance can ignore
    // the new fields and the behaviour is identical to the pre-collapse
    // streaming kernel.
    std::vector<SectorTag>              sector_tags;
    std::vector<std::vector<double>>    eigenvalues_per_sector;
    std::vector<std::size_t>            sector_index_of_eigenvalue;
};

// ---------------------------------------------------------------------------
// Per-sector thermodynamics entry. Carries both the legacy Sz label
// (`sz_index` / `n_up`) and the SOTA streaming-symmetry attribution
// (`sector_index`, `quantum_numbers`, `sector_dim`) so that downstream
// consumers can identify which irrep / Sz / orbit-basis block produced
// the per-sector thermo block.
// ---------------------------------------------------------------------------
struct ThermalSectorEntry {
    /// Legacy Sz index (kept for compatibility with the pre-collapse
    /// `ThermalSectorEntry` from the Python facade).
    int                 sz_index = 0;
    double              ground_state_energy = 0.0;
    ThermodynamicData   thermo;
    /// Free-form per-sector diagnostics.
    std::vector<std::pair<std::string, std::string>> notes;

    // -----------------------------------------------------------------
    // Streaming-symmetry attribution (May 2026 SOTA upgrade).
    // -----------------------------------------------------------------
    SectorTag           tag;
};

// ---------------------------------------------------------------------------
// ThermalResult --- output of `ed::thermal(H, opts)`. Folds the FTLM /
// LTLM / mTPQ / KPM-DOS family.
// ---------------------------------------------------------------------------
/// One snapshotted TPQ state. Pillar 1 of the "Save and DSSF Upgrades"
/// plan (May 2026). The orchestrator's thermal finalizer iterates these
/// and lands each one in ``/tpq/samples/sample_<s>/state_beta_<b>`` of
/// the shared ``ed_results.h5`` file via ``HDF5IO::saveTPQState``.
struct TpqStateSnapshot {
    std::size_t            sample_index    = 0;
    /// Inverse temperature the caller requested via ``ThermalOptions
    /// ::probe_betas``. Kept so the snapshot's provenance survives a
    /// round-trip through HDF5 (the on-disk dataset name uses the
    /// effective beta, since that is what the state actually
    /// realises).
    double                 requested_beta  = 0.0;
    /// Closest kernel-step beta to ``requested_beta``. Used as the
    /// HDF5 dataset key.
    double                 effective_beta  = 0.0;
    /// Host-side TPQ state vector, length ``H.geometry().local_dim``.
    std::vector<Complex>   psi;
};

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

    // -----------------------------------------------------------------
    // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026): TPQ
    // trajectory + state-snapshot surface. Populated only by the mTPQ
    // branch of ``ed::workflows::thermal``; empty for
    // FTLM / LTLM / KPM-DOS.
    //
    // The trajectory fields are mirror-images of
    // ``MtpqResult::sample_*`` --
    // outer index = sample, inner index = kernel step. The orchestrator
    // copies them into ``R`` after the visit so the uniform finalizer
    // can persist them via ``HDF5IO::appendTPQThermodynamics``.
    // -----------------------------------------------------------------
    std::vector<std::vector<double>>    tpq_sample_betas;
    std::vector<std::vector<double>>    tpq_sample_energies;
    std::vector<std::vector<double>>    tpq_sample_variances;
    /// One entry per snapshot the kernel actually recorded.
    std::vector<TpqStateSnapshot>       tpq_state_snapshots;
};

// ---------------------------------------------------------------------------
// Per-sector contribution to the spectral function. Used when
// `ed::workflows::spectral` walks a set of (irrep_initial,
// irrep_final) sector pairs and accumulates the
// selection-rule-filtered pieces of ``S(Q, omega)``.
// ---------------------------------------------------------------------------
struct SpectralSectorEntry {
    /// Tag for the *initial* sector (where the ground state lives).
    SectorTag           initial;
    /// Tag for the *final* sector (where O|psi_0> lives).
    SectorTag           final_;
    /// This sector's contribution to S_real / S_imag on the shared
    /// omega grid. Empty when the (initial, final) pair was filtered
    /// out by the selection rule.
    std::vector<double> S_real;
    std::vector<double> S_imag;
    /// Static (equal-time) structure factor for this (initial, final)
    /// pair: ``||O_Q|psi_0>||^2 = sum_n |<n|O_Q|0>|^2``. Computed for
    /// free from the CF pivot norm by the cross-irrep spectral
    /// workflows; 0 when not populated.
    double              static_sf = 0.0;
    /// Free-form per-pair diagnostics (e.g. transition matrix element).
    std::vector<std::pair<std::string, std::string>> notes;
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

    // -----------------------------------------------------------------
    // Streaming-symmetry attribution (May 2026 SOTA upgrade).
    // -----------------------------------------------------------------
    //
    // When the result was produced by the streaming-symmetry spectral
    // workflow, `per_sector_pair` lists every (initial-sector,
    // final-sector) pair that survived the momentum / point-group
    // selection rule, together with that pair's contribution to the
    // merged S(omega) array. `selection_rule_label` describes the
    // physical filter that was applied (e.g. ``"k_final = k_initial +
    // Q"`` for spin spectral functions). Empty when no symmetry was
    // exploited.
    std::vector<SpectralSectorEntry> per_sector_pair;
    std::string                       selection_rule_label;
};

}  // namespace ed

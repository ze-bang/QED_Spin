#pragma once

// =============================================================================
// Lanczos Krylov-state checkpoint / restart                       (Phase 3a #1)
// =============================================================================
//
// At ~2 × 10⁸ sector states (N≈36 with full point-group + Sz symmetry, see
// SCALING.md) a Lanczos run is a multi-day job. Today the solver has no way
// to resume after a node failure or wall-clock kill: a SIGTERM at hour 40
// throws away every iteration. This header adds a small, side-channel
// checkpoint format that snapshots the exact Krylov state at iteration k so
// the *next* invocation can resume at iteration k+1.
//
// The default `lanczos()` solver checks three environment variables on entry:
//
//   ED_LANCZOS_CHECKPOINT_DIR
//       If set to a non-empty path, checkpoints are written into that
//       directory. Otherwise checkpointing is disabled (zero overhead).
//
//   ED_LANCZOS_CHECKPOINT_INTERVAL
//       Number of completed Lanczos iterations between checkpoint writes.
//       Default: 100. Lower for short jobs / fast crash recovery, higher
//       to reduce HDF5 I/O cost on long jobs (each write is ~ N * 16 B
//       just for v_prev + v_current; the ring buffer adds another ~20 N).
//
//   ED_LANCZOS_RESUME
//       If set to "1" (and ED_LANCZOS_CHECKPOINT_DIR points at a directory
//       that contains a previously-written checkpoint), the solver skips
//       its random-vector init and resumes from the snapshot.
//
// Atomicity: write_lanczos_checkpoint writes to "<dir>/lanczos_checkpoint
// .h5.tmp", flushes / closes, then std::filesystem::rename()s onto the
// canonical filename. A SIGKILL during the write therefore leaves the
// previous valid checkpoint in place; the worst-case loss is one
// checkpoint interval.
//
// What is NOT promised:
//   * Bit-for-bit reproducibility against an uninterrupted run. A resumed
//     run goes through the same algorithm (selective DGKS local re-orth)
//     starting from identical state, so the *eigenvalues* converge to the
//     same values within tolerance, but rounding of intermediate scalars
//     may differ if the user changes thread count / BLAS profile / OS
//     scheduler between runs. The unit test in
//     tests/unit/test_lanczos_checkpoint.cpp asserts eigenvalue
//     equivalence to ~1e-10, not exact bit equality.
//
//   * Cross-Hamiltonian sanity. The checkpoint stores last_w_norm =
//     beta[iteration]; on resume the solver computes one matvec against
//     v_current (which it had to do anyway) and the norm of the resulting
//     residual is compared to the stored value. If the user resumed
//     against a *different* Hamiltonian the norms diverge and we throw
//     loud and early instead of producing nonsense eigenvalues.
//
// Schema (HDF5, version 1):
//
//   /metadata/schema_version       : uint32 = 1
//   /metadata/solver               : string = "lanczos"
//   /metadata/N                    : uint64
//   /metadata/max_iter             : uint64    (best-effort: caller's max)
//   /metadata/exct                 : uint64
//   /metadata/tol                  : double
//   /metadata/iteration            : uint64    (number of completed iters)
//   /metadata/timestamp_iso        : string
//   /metadata/host                 : string
//   /metadata/complex_seed         : uint8
//   /metadata/last_w_norm          : double    (=beta[iteration]; H-fingerprint)
//   /tridiag/alpha                 : double[iteration]
//   /tridiag/beta                  : double[iteration+1]   (beta[0]=0)
//   /vectors/v_prev                : compound{r,i}[N]
//   /vectors/v_current             : compound{r,i}[N]
//   /ring/head                     : uint64
//   /ring/count                    : uint64
//   /ring/vectors                  : compound{r,i}[count, N]   (only if count>0)
//   /rng/state                     : compound{state_text}      (mt19937 state via operator<<)
//   /counters/total_reorth         : uint64
//   /counters/selective_reorth     : uint64
//   /convergence/prev_eigenvalues  : double[len]               (may be empty)
//   /convergence/converged_flag    : uint8
//
// The schema is intentionally flat so it can be inspected with
// `h5dump lanczos_checkpoint.h5` for debugging.
// =============================================================================

#include <complex>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace lanczos_io {

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// On-disk representation, mirroring the schema above. Members map 1:1 to the
// HDF5 datasets so the read/write loops are mechanical.
struct LanczosCheckpoint {
    // Identity / shape
    uint64_t N = 0;
    uint64_t max_iter = 0;
    uint64_t exct = 0;
    double   tol = 0.0;
    bool     complex_seed = false;

    // Iteration counter: number of *completed* Lanczos iterations.
    // After iteration j of the solver loop, `iteration = j + 1`.
    uint64_t iteration = 0;

    // Tridiagonal entries built so far. alpha.size() == iteration,
    // beta.size() == iteration + 1 (beta[0] is the leading 0 sentinel).
    std::vector<double> alpha;
    std::vector<double> beta;

    // The two adjacent Lanczos vectors needed to keep the three-term
    // recurrence going on resume.
    ComplexVector v_prev;
    ComplexVector v_current;

    // Ring buffer of recent vectors used by the DGKS-style local re-orth.
    // ring_vectors.size() == ring count; ordered oldest → newest in storage,
    // and on read we re-emit them via the (head, count) convention used by
    // the solver loop.
    uint64_t ring_head = 0;
    std::vector<ComplexVector> ring_vectors;

    // Serialized mt19937 state (via operator<<). Kept for forward-
    // compatibility with future RNG-using checkpointable solvers; the
    // default lanczos() body does not consume RNG inside the inner loop,
    // so on resume the absence of this field is tolerated.
    std::string rng_state_text;

    // Diagnostics
    uint64_t total_reorth_count = 0;
    uint64_t selective_reorth_count = 0;

    // Convergence-check state
    std::vector<double> prev_eigenvalues;
    bool eigenvalues_converged = false;

    // Hamiltonian fingerprint: last computed beta. On resume we recompute
    // ||H * v_current - alpha[k] * v_current - beta[k] * v_prev|| and
    // verify it matches; mismatch → throw (most likely the user resumed
    // against a different Hamiltonian).
    double last_w_norm = 0.0;
};

// ---------------------------------------------------------------------------
// Environment-variable knobs
// ---------------------------------------------------------------------------

// True iff ED_LANCZOS_CHECKPOINT_DIR is set to a non-empty value. The
// environment is re-read on every call (no static cache) so test code can
// flip the variable on and off between solver invocations.
bool checkpoint_enabled();

// True iff the user requested a resume (ED_LANCZOS_RESUME=1) AND a
// checkpoint file exists in the configured checkpoint directory.
bool checkpoint_resume_requested();

// Returns the configured checkpoint directory or empty string if
// ED_LANCZOS_CHECKPOINT_DIR is unset. Re-read each call.
std::string checkpoint_dir();

// Number of Lanczos iterations between checkpoint writes (default 100).
// Reads ED_LANCZOS_CHECKPOINT_INTERVAL each call.
uint64_t checkpoint_interval();

// Canonical checkpoint filename inside `dir`:
//   "<dir>/lanczos_checkpoint.h5"
std::string checkpoint_filename(const std::string& dir);

// ---------------------------------------------------------------------------
// I/O primitives
// ---------------------------------------------------------------------------

// Atomically write `cp` to "<dir>/lanczos_checkpoint.h5". Creates `dir` if
// necessary. Throws std::runtime_error on HDF5 / filesystem error.
//
// Implementation: writes to "<dir>/lanczos_checkpoint.h5.tmp", flushes,
// closes, then renames onto the canonical filename. A SIGKILL between
// write start and rename therefore leaves any previous valid checkpoint
// untouched.
void write_lanczos_checkpoint(const std::string& dir,
                              const LanczosCheckpoint& cp);

// Read and return the checkpoint stored in `dir`. Throws if the file does
// not exist, the schema_version is unsupported, or any required dataset
// is missing.
LanczosCheckpoint read_lanczos_checkpoint(const std::string& dir);

// ---------------------------------------------------------------------------
// mt19937 raw-state helpers. We round-trip via operator<< / operator>>
// (standardized in C++11) so the format is portable across compilers.
// ---------------------------------------------------------------------------
std::string capture_mt19937_state(const std::mt19937& gen);
void        restore_mt19937_state(std::mt19937& gen,
                                  const std::string& state_text);

}  // namespace lanczos_io

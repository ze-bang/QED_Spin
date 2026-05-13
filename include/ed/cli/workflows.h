#pragma once

// =============================================================================
// include/ed/cli/workflows.h
//
// Declarations for the CLI-driven workflow entry points used by
// `src/apps/ed_main.cpp`. These were originally free functions in
// ed_main.cpp; P1.11 (DSSF PR-B / audit §3.10) extracted them into their
// own translation unit (`src/cli/workflows.cpp`) so:
//
//   * `ed_main.cpp` shrinks to a thin argv → workflow dispatcher.
//   * The same workflow functions can be reused by future
//     `ED <subcommand>` entry points (P2.4) and by pybind11 bindings (P2.x)
//     without dragging the full `int main(int, char**)` machinery along.
//   * The future `ed::dssf::dssf_engine` (P2.2) has a clearly delimited
//     boundary it can sit behind: every call into the DSSF/SSSF pipeline
//     goes through one of the `compute_*_workflow` functions declared
//     below today, and through `ed::dssf::run(...)` tomorrow.
//
// This header does NOT introduce a `namespace ed::cli`. Keeping the
// declarations in the global namespace preserves source compatibility with
// the existing call sites in `ed_main.cpp` and makes the lift-and-shift
// bit-identical.
// =============================================================================

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <ed/core/ed_config.h>
#include <ed/core/ed_types.h>
#include <ed/core/ed_wrapper.h>      // EDResults
#include <ed/core/construct_ham.h>   // Operator

// -----------------------------------------------------------------------------
// String parsing helpers (CLI-side; small enough that exposing them to other
// TUs costs nothing and is needed by the legacy `run_dssf_mode` shim that
// still lives in ed_main.cpp).
// -----------------------------------------------------------------------------

/**
 * @brief Parse spin combinations from string format.
 *
 * Format: "op1,op2;op3,op4;..." where op is 0=Sp/Sx, 1=Sm/Sy, 2=Sz.
 */
std::vector<std::pair<int, int>>
parse_spin_combinations(const std::string& spin_combinations_str);

/**
 * @brief Parse momentum points from string format.
 *
 * Format: "Qx1,Qy1,Qz1;Qx2,Qy2,Qz2;..." (values are multiplied by π).
 */
std::vector<std::vector<double>>
parse_momentum_points(const std::string& momentum_str);

/**
 * @brief Parse polarization vector from string format.
 *
 * Format: "px,py,pz" (will be normalized).
 */
std::vector<double>
parse_polarization(const std::string& pol_str);

/**
 * @brief Construct operators based on configuration.
 *
 * Thin wrapper around `ed::dssf::build_observable_pairs` (P1.10). Kept so
 * the historical call sites in `ed_main.cpp` (and in the legacy
 * `run_dssf_mode` shim) keep compiling. New code should call
 * `ed::dssf::build_observable_pairs` directly.
 */
void construct_operators_from_config(
    const std::string& operator_type,
    const std::string& basis,
    const std::vector<std::pair<int, int>>& spin_combinations,
    const std::vector<std::vector<double>>& momentum_points,
    const std::vector<double>& polarization,
    double theta,
    uint64_t unit_cell_size,
    uint64_t num_sites,
    float spin_length,
    bool use_fixed_sz,
    int64_t n_up,
    const std::string& positions_file,
    std::vector<Operator>& obs_1_out,
    std::vector<Operator>& obs_2_out,
    std::vector<std::string>& names_out);

// -----------------------------------------------------------------------------
// Workflows (driven by EDConfig).
// -----------------------------------------------------------------------------

/// Standard exact diagonalization (no symmetries).
EDResults run_standard_workflow(const EDConfig& config);

/// Symmetry-exploiting diagonalization via the streaming-symmetry path.
EDResults run_streaming_symmetry_workflow(const EDConfig& config);

/// Disk-based streaming symmetry diagonalization (ultra-low-memory).
EDResults run_disk_streaming_workflow(const EDConfig& config);

/// Two-pass chunked symmetry diagonalization (low-memory basis build).
EDResults run_chunked_symmetry_workflow(const EDConfig& config);

/// Compute thermodynamics from a finite eigenvalue spectrum and persist
/// them to the run's HDF5 file.
void compute_thermodynamics(const std::vector<double>& eigenvalues,
                            const EDConfig& config);

/// Compute dynamical response (spectral functions): drives the DSSF
/// pipeline (Lanczos / FTLM / continued fraction) for the operators
/// specified in `config.dynamical`.
void compute_dynamical_response_workflow(const EDConfig& config);

/// Compute static response (thermal expectation values) for the operators
/// specified in `config.static_resp`.
void compute_static_response_workflow(const EDConfig& config);

/// Compute T=0 dynamical structure factor via continued fraction; the
/// optimal path for fixed-Sz 32-site ED.
void compute_ground_state_dssf_workflow(const EDConfig& config);

/// Compute KPM-based thermodynamics: Chebyshev moments of the spectral
/// density (Hutchinson-sampled), reconstructed thermal Z(β), E(β), C(β),
/// S(β), F(β), and an optional DOS grid. Operator-free; uses only the
/// Hamiltonian. Persists results under `/kpm_thermodynamics/...` in the
/// run's HDF5 file. Item #3 of the audit (expose KPM at the DSSF method
/// surface).
void compute_kpm_thermodynamics_workflow(const EDConfig& config);

/// Print the lowest `max_show` eigenvalues from the input vector.
void print_eigenvalue_summary(const std::vector<double>& eigenvalues,
                              uint64_t max_show = 10);

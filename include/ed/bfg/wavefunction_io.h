// =============================================================================
// include/ed/bfg/wavefunction_io.h
//
// HDF5 wavefunction loaders shared by the CPU and GPU BFG drivers and by the
// Python bindings. Before P2.1 (third slice) each driver carried its own copy:
//
//   * `src/apps/compute_bfg_order_parameters.cpp` -- the canonical, robust
//     loader that probes both `eigendata/eigenvector_*` and the legacy
//     top-level `eigenvector_*` paths, supports HDF5 compound types with
//     either `(real, imag)` or `(r, i)` field names, and falls back to a
//     raw-double layout. Also ships the TPQ batch loaders.
//   * `src/apps/compute_bfg_order_parameters_gpu.cu` -- a slimmer copy that
//     only knew about the legacy paths and the raw-double layout.
//
// Promoting the loaders to the `ed_bfg` static library lets the GPU driver,
// any future Python bindings, and the in-progress library split below the
// `compute_bfg_order_parameters.cpp` driver share the same authoritative
// implementation.
//
// Audit ref: P2.1 (BFG library extraction, third slice).
// =============================================================================

#pragma once

#include <complex>
#include <string>
#include <utility>
#include <vector>

namespace ed::bfg {

/**
 * Single TPQ snapshot loaded from an HDF5 results file.
 *
 * Sorted by ascending temperature (= descending beta) by `load_all_tpq_states`.
 */
struct TPQState {
    std::vector<std::complex<double>> psi;
    double temperature{0.0};
    double beta{0.0};
};

/**
 * Load a single wavefunction from an HDF5 file.
 *
 * Probes (in order) `eigendata/eigenvector_<idx>`, `eigendata/eigenvectors`,
 * the legacy top-level `eigenvector_<idx>` path, `eigenvectors`, `psi`,
 * `wavefunction`, and `ground_state`. Supports HDF5 compound complex types
 * with `(real, imag)` or `(r, i)` field names; falls back to raw doubles
 * (interpreted as `[re_0, im_0, re_1, im_1, ...]`) if no compound type is
 * found.
 *
 * @param filename          Path to the HDF5 results file.
 * @param eigenvector_idx   Which eigenvector to read (default 0).
 * @param verbose           If true, prints which dataset path was chosen and
 *                          the loaded amplitude count to stdout.
 *
 * @throws std::runtime_error on HDF5 errors or missing dataset.
 */
std::vector<std::complex<double>> load_wavefunction(const std::string& filename,
                                                   int eigenvector_idx = 0,
                                                   bool verbose = true);

/**
 * Load every TPQ snapshot from `tpq/samples/sample_<idx>/states/beta_*` in
 * an HDF5 file, sorted ascending in temperature.
 *
 * @param filename    Path to the HDF5 results file.
 * @param sample_idx  Which TPQ sample to load (default 0).
 * @param verbose     If true, prints sample count + temperature range.
 *
 * @throws std::runtime_error on HDF5 errors or missing TPQ states.
 */
std::vector<TPQState> load_all_tpq_states(const std::string& filename,
                                          int sample_idx = 0,
                                          bool verbose = true);

/**
 * Load the lowest-temperature (highest-beta) TPQ snapshot.
 *
 * Convenience wrapper around `load_all_tpq_states` for callers that only
 * need a single state. Returns the wavefunction and its temperature.
 *
 * @throws std::runtime_error on HDF5 errors or missing TPQ states.
 */
std::pair<std::vector<std::complex<double>>, double> load_tpq_state(
    const std::string& filename, int sample_idx = 0, bool verbose = true);

}  // namespace ed::bfg

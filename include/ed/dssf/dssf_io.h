// =============================================================================
// include/ed/dssf/dssf_io.h
//
// Unified `/dssf/...` HDF5 schema -- the canonical, library-level on-disk
// layout for every DSSF / SSSF / static-response result produced by
// `ed::dssf::run(...)` (P2.3 / DSSF PR-D, audit §3.10).
//
// Why a new schema?
// -----------------
// Today's outputs are split across THREE incompatible layouts that
// drifted independently:
//
//   * `compute_dynamical_response_workflow` -> `/dynamical/<op>/...`
//                                              (real/imag pair, attrs T,
//                                               total_samples)
//   * `compute_static_response_workflow`    -> `/correlations/<op>/...`
//                                              (1D temp grid, expectation,
//                                               variance, susceptibility)
//   * `TPQ_DSSF.cpp`                        -> `/dssf_results/...`
//                                              (positional + per-Q groups,
//                                               separate metadata blob)
//
// The unified schema collapses these onto a single root group `/dssf/`
// with a `schema_version` attribute so post-processing scripts can detect
// the layout once and dispatch on it forever after. The legacy writers
// remain in place (gated by `--dssf-legacy-output` for `TPQ_DSSF`) for
// one release; `TPQ_DSSF`-style outputs become read-only after P2.5 and
// are deleted in P2.14.
//
// Schema (root group `/dssf`):
//
//   /dssf/                              (group)
//     @schema_version                  uint32 = 1
//     @method                          string  ("dynamical_thermal" |
//                                                "static_thermal" |
//                                                "ground_state_dssf" |
//                                                "single_expectation")
//     @num_sites                       uint64
//     @spin_length                     double
//     @created_at                      string  (ISO-8601 UTC, optional)
//     /<op_name>/                      (group)  one per observable
//       @method                       string  (mirrors root @method)
//       @temperature                  double  (single-T view; for a
//                                              multi-T scan the writer
//                                              uses one group per T)
//       @total_samples                uint64
//       --- dynamical fields (only if method == DYNAMICAL_*) ---
//       frequencies                   double[]  (length F)
//       spectral_real                 double[]  (length F)
//       spectral_imag                 double[]  (length F)
//       error_real                    double[]  (length F)
//       error_imag                    double[]  (length F)
//       --- static fields (only if method == STATIC_THERMAL) ---
//       temperatures                  double[]  (length T)
//       expectation                   double[]  (length T)
//       expectation_error             double[]  (length T)
//       variance                      double[]  (optional, length T)
//       variance_error                double[]  (optional, length T)
//       susceptibility                double[]  (optional, length T)
//       susceptibility_error          double[]  (optional, length T)
//
// Slash-separated `op_name` continues to autocreate intermediate groups
// (legacy behaviour locked down by test_dssf_legacy_schema.cpp test #3).
// =============================================================================

#pragma once

#include <ed/dssf/dssf_engine.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ed::dssf {

/// Schema version stamped on `/dssf/@schema_version`. Bump when the
/// layout changes; readers should refuse to parse versions they don't
/// recognise so we get a hard, early failure instead of silent corruption.
constexpr std::uint32_t kSchemaVersion = 1u;

/**
 * One DSSF / SSSF result record. Population depends on `method`:
 *
 *   * DYNAMICAL_THERMAL / GROUND_STATE_DSSF: populate `frequencies`,
 *     `spectral_real`, `spectral_imag`, `error_real`, `error_imag`. The
 *     `temperatures` / `expectation` / ... fields stay empty.
 *
 *   * STATIC_THERMAL: populate `temperatures`, `expectation`,
 *     `expectation_error`, and (optionally) the `variance*` and
 *     `susceptibility*` arrays. The frequency-domain fields stay empty.
 *
 *   * SINGLE_EXPECTATION: populate `temperatures` (length 1, by
 *     convention) and `expectation` only.
 */
struct Record {
    DSSFMethod method{DSSFMethod::DYNAMICAL_THERMAL};
    std::string operator_name;
    double temperature{0.0};
    std::uint64_t total_samples{1};

    /// Dynamical fields (lengths must all match if non-empty).
    std::vector<double> frequencies;
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;
    std::vector<double> error_real;
    std::vector<double> error_imag;

    /// Static fields (lengths must all match if non-empty).
    std::vector<double> temperatures;
    std::vector<double> expectation;
    std::vector<double> expectation_error;
    std::vector<double> variance;
    std::vector<double> variance_error;
    std::vector<double> susceptibility;
    std::vector<double> susceptibility_error;
};

/**
 * Run-level metadata stamped on `/dssf/@*` attributes. Written once per
 * output file by `ensure_metadata`; subsequent calls (e.g. when adding
 * a second observable to the same file) only re-stamp if the values
 * change, so the file stays bit-stable for repeat runs.
 */
struct Metadata {
    DSSFMethod method{DSSFMethod::DYNAMICAL_THERMAL};
    std::uint64_t num_sites{0};
    double spin_length{0.5};
    std::string created_at;          ///< Optional ISO-8601 UTC string.
};

/**
 * Stamp `/dssf/@*` metadata on `filepath`. Creates the file (and the
 * `/dssf` root group) if it does not already exist.
 *
 * @throws std::runtime_error on any HDF5 error.
 */
void ensure_metadata(const std::string& filepath, const Metadata& meta);

/**
 * Write one observable record under `/dssf/<record.operator_name>`.
 * Creates intermediate groups for slash-separated `operator_name`.
 *
 * Pre-conditions:
 *   - `filepath` already contains a `/dssf` root with `@schema_version`
 *     stamped (i.e. `ensure_metadata` has been called).
 *   - For dynamical methods, the five frequency-domain arrays must
 *     have matching lengths.
 *   - For static methods, the populated time-domain arrays must have
 *     matching lengths.
 *
 * @throws std::invalid_argument on any of the above pre-conditions.
 * @throws std::runtime_error on any HDF5 error.
 */
void write_record(const std::string& filepath, const Record& record);

/**
 * Read back the record at `/dssf/<operator_name>` and the run-level
 * metadata attached to `/dssf`. Optional / unwritten arrays come back
 * as empty vectors.
 *
 * @throws std::invalid_argument if the file has no `/dssf/` group, an
 *         unrecognised `schema_version`, or no record at the given
 *         operator path.
 * @throws std::runtime_error on any HDF5 error.
 */
std::pair<Metadata, Record>
read_record(const std::string& filepath, const std::string& operator_name);

} // namespace ed::dssf

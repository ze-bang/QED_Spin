#pragma once

// =============================================================================
// ed_dispatch_symmetry.h — Phase 7.1 canonical 5-axis dispatcher
//
// This header is the single canonical entry point for the orthogonal
// SOLVER × FIXED_SZ × GPU × MPI × SYMMETRY axes from C++ callers. It exists
// because ed/core/ed_wrapper.h cannot itself depend on the streaming
// symmetry kernel (ed_wrapper_streaming.h includes ed_wrapper.h, not the
// other way round).
//
// Use this header instead of ed_wrapper.h whenever you might want symmetry
// projection. Callers that NEVER set use_symmetry can keep including
// ed_wrapper.h directly with no behaviour change.
//
// Contract:
//   * params.use_symmetry == false  → forwarded verbatim to
//     exact_diagonalization_from_directory(...) in ed_wrapper.h.
//   * params.use_symmetry == true   → routed to
//     exact_diagonalization_streaming_symmetry[_fixed_sz](...) in
//     ed_wrapper_streaming.h. The streaming kernel honours use_gpu
//     (per-sector GPU dispatch) and use_fixed_sz (fixed-Sz orbit basis)
//     orthogonally inside its own implementation.
//
// The deprecated explicit-block path (`exact_diagonalization_*_symmetrized`)
// and the chunked / disk-streaming variants are NOT reachable from this
// dispatcher. They remain available as expert escape hatches via the
// dedicated CLI flags (--chunked-symm, --disk-streaming) but are no
// longer selectable through the EDParameters flag axis.
// =============================================================================

#include <ed/core/ed_wrapper.h>            // exact_diagonalization_from_files / from_directory
#include <ed/core/ed_wrapper_streaming.h>  // streaming symmetry kernel (canonical)

#include <cstdint>
#include <string>

namespace ed_dispatch {

/**
 * @brief Canonical 5-axis ED dispatcher.
 *
 * Routes to either the standard full-Hilbert-space dispatcher or the
 * streaming symmetry kernel based on `params.use_symmetry`. All other
 * axes (use_fixed_sz, use_gpu, use_mpi) are passed through unchanged
 * and honoured inside the chosen kernel.
 *
 * @param directory Directory containing InterAll.dat / Trans.dat (and,
 *                  when use_symmetry=true, automorphism_results/ — the
 *                  streaming kernel will generate it on the fly if missing).
 * @param method    Diagonalization method (canonical or legacy).
 * @param params    Parameter bag, including the orthogonal axis flags.
 * @return EDResults with eigenvalues sorted ascending.
 */
inline EDResults exact_diagonalization_from_directory(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& counterterm_filename = "CounterTerm.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat"
) {
    if (!params.use_symmetry) {
        return ::exact_diagonalization_from_directory(
            directory, method, params, format,
            interaction_filename, single_site_filename,
            counterterm_filename, three_body_filename);
    }

    // ----- Symmetry path (canonical streaming kernel) -----
    if (params.use_fixed_sz) {
        std::int64_t n_up = (params.n_up >= 0)
                                ? params.n_up
                                : static_cast<std::int64_t>(params.num_sites / 2);
        return ::exact_diagonalization_streaming_symmetry_fixed_sz(
            directory, n_up, method, params,
            interaction_filename, single_site_filename,
            /*basis_cache_dir=*/"", /*precompute_basis_only=*/false);
    }
    return ::exact_diagonalization_streaming_symmetry(
        directory, method, params,
        interaction_filename, single_site_filename,
        /*basis_cache_dir=*/"", /*precompute_basis_only=*/false);
}

/**
 * @brief Canonical 5-axis ED dispatcher (file-path overload).
 *
 * Identical to the directory-based overload but accepts explicit file
 * paths. Derives the directory from `dirname(interaction_file)` so the
 * streaming symmetry kernel can locate / generate
 * `automorphism_results/` when use_symmetry=true.
 */
inline EDResults exact_diagonalization_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    const std::string& counterterm_file = "",
    const std::string& three_body_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD
) {
    if (!params.use_symmetry) {
        return ::exact_diagonalization_from_files(
            interaction_file, single_site_file, counterterm_file,
            three_body_file, method, params, format);
    }

    // ----- Symmetry path: derive the directory + basenames -----
    auto split_path = [](const std::string& path,
                         std::string& dir, std::string& base) {
        const auto pos = path.find_last_of('/');
        if (pos == std::string::npos) {
            dir = ".";
            base = path;
        } else {
            dir = path.substr(0, pos);
            base = path.substr(pos + 1);
        }
    };

    std::string directory, interaction_basename;
    split_path(interaction_file, directory, interaction_basename);

    std::string single_site_basename = "Trans.dat";
    if (!single_site_file.empty()) {
        std::string ignored_dir;
        split_path(single_site_file, ignored_dir, single_site_basename);
    }

    if (params.use_fixed_sz) {
        std::int64_t n_up = (params.n_up >= 0)
                                ? params.n_up
                                : static_cast<std::int64_t>(params.num_sites / 2);
        return ::exact_diagonalization_streaming_symmetry_fixed_sz(
            directory, n_up, method, params,
            interaction_basename, single_site_basename,
            /*basis_cache_dir=*/"", /*precompute_basis_only=*/false);
    }
    return ::exact_diagonalization_streaming_symmetry(
        directory, method, params,
        interaction_basename, single_site_basename,
        /*basis_cache_dir=*/"", /*precompute_basis_only=*/false);
}

}  // namespace ed_dispatch

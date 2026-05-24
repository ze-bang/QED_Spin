#pragma once

// =============================================================================
// DEPRECATED (Phase 5 — Minimalist ED Collapse, May 2026)
//   ed/core/dispatch.h has been SUPERSEDED by `ed/orchestrator.h`
//   (`ed::workflows::solve / thermal / spectral`). It remains in tree
//   as a forwarder during the migration window; new code must use the
//   `ed::workflows::*` entry points and `ed::make_operator(...)` factory.
// =============================================================================

// =============================================================================
// ed/core/dispatch.h — the single canonical ED entry point.
//
// This header is the public C++ surface for "give me eigenvalues / vectors
// from this Hamiltonian directory". It replaces, and supersedes:
//
//     ed/core/ed_wrapper.h             (full-Hilbert / fixed-Sz / GPU)
//     ed/core/ed_wrapper_streaming.h   (streaming spatial symmetry)
//     ed/core/ed_dispatch_symmetry.h   (one-shot symmetry dispatcher)
//
// All three legacy headers remain available as thin transparent forwarders
// so that out-of-tree callers keep compiling unchanged, but the matvec-
// unification rollout (Phase 6) moves every in-tree caller onto this
// single facade.
//
// Public interface
// ----------------
//
// ed::exact_diagonalization(directory, ...)
// ed::exact_diagonalization(interaction_file, ...)
//
// Both overloads dispatch on the orthogonal axes recorded in `params`:
//
//   * use_symmetry  → route through the streaming-symmetry kernel
//                     (sector-by-sector matvec); GPU-aware per sector.
//   * use_fixed_sz  → project to a fixed Sz sector (chosen by
//                     params.n_up, defaulting to num_sites / 2 when
//                     n_up < 0 -- typically the ground-state-containing
//                     sector for nearest-neighbor Heisenberg-like
//                     Hamiltonians without a Zeeman field).
//   * use_gpu       → GPU matvec for the full-Hilbert and fixed-Sz
//                     paths; per-sector GPU dispatch inside the
//                     streaming-symmetry kernel.
//   * use_mpi       → distributed CPU/GPU matvec (delegated to the
//                     ed/distributed/* library).
//
// These axes are orthogonal and combine freely; the legacy combinatorial
// `exact_diagonalization_*_fixed_sz_gpu_streaming_...` entry points are
// fully reachable through this one call. The auto_pilot::solve front-end
// detects symmetry / Sz conservation and sets these axes automatically.
//
// All bit-flip matvec eventually funnels through the matvec-unification
// MatVecOperator interface (ed/matvec/matvec.h); this file is the only
// public C++ surface that users should need to include.
// =============================================================================

#include <ed/core/ed_method_traits.h>      // canonicalize_method_and_flags
#include <ed/core/ed_wrapper.h>            // full-Hilbert + fixed-Sz + GPU kernels
#include <ed/core/ed_wrapper_streaming.h>  // streaming-symmetry kernel

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

namespace ed {

namespace detail {

// Phase 5 (matvec-unification): auto-detect spatial symmetry from the
// canonical "automorphism_results/sectors.json" file inside a Hamiltonian
// directory. Used by ed::exact_diagonalization to flip params.use_symmetry
// on for callers that have a symmetry directory but did not explicitly
// opt in (the user-visible "kicks in automatically if the Hamiltonian
// possesses it" behaviour from the audit).
//
// We only auto-promote when:
//   1. params.use_symmetry was left at its default (false), AND
//   2. `<directory>/automorphism_results/sectors.json` exists.
//
// Callers can defeat the auto-promotion by setting params.use_symmetry
// explicitly to false in a future EDParameters extension (currently the
// boolean is ambiguous between "explicit false" and "default false");
// we treat this as acceptable because the only failure mode is "we use
// symmetry projection when symmetry data is present", which is what the
// user asked for in the matvec-unification audit.
inline bool symmetry_data_present(const std::string& directory) {
    namespace fs = std::filesystem;
    const fs::path d(directory);
    if (!fs::exists(d) || !fs::is_directory(d)) return false;
    const fs::path ar = d / "automorphism_results";
    if (!fs::exists(ar) || !fs::is_directory(ar)) return false;

    // The directory layout written by the Python `automorphism` tool (and
    // by C++ `generate_automorphisms` in system_utils.h) is:
    //
    //   <dir>/automorphism_results/automorphisms.json     <- full list
    //   <dir>/automorphism_results/max_clique.json        <- chosen Abelian clique
    //   <dir>/automorphism_results/minimal_generators.json
    //   <dir>/automorphism_results/sector_metadata.json   <- irrep table
    //
    // The streaming kernel `generate_automorphisms` can regenerate the
    // last three from `automorphisms.json`, so the presence of EITHER
    // (a) the full automorphisms list, OR (b) a manually-curated clique
    // plus sector metadata is enough to take the symmetry route. We also
    // accept the legacy ``sectors.json`` / ``generators.json`` names that
    // some older pipelines emit.
    const char* candidates[] = {
        "automorphisms.json",
        "max_clique.json",
        "sector_metadata.json",
        "minimal_generators.json",
        "sectors.json",       // legacy
        "generators.json",    // legacy
    };
    for (const char* name : candidates) {
        const fs::path p = ar / name;
        if (fs::exists(p) && fs::is_regular_file(p)) return true;
    }
    return false;
}

}  // namespace detail

/**
 * @brief Canonical ED entry point (directory form).
 *
 * Dispatches on the orthogonal axes in `params` -- use_symmetry,
 * use_fixed_sz, use_gpu, use_mpi -- to the appropriate kernel. The
 * choice is deterministic from `params` alone; auto_pilot::solve is
 * the recommended front-end if you want axis selection driven by
 * Hamiltonian analysis (Sz conservation, Zeeman field detection,
 * symmetry-info presence).
 *
 * @param directory               Hamiltonian directory (InterAll.dat /
 *                                Trans.dat / CounterTerm.dat /
 *                                ThreeBodyG.dat and, when
 *                                params.use_symmetry == true,
 *                                automorphism_results/ -- generated on
 *                                the fly if missing).
 * @param method                  Diagonalization method.
 * @param params                  Parameter bag (axes, sizes, options).
 * @param format                  Hamiltonian file format.
 * @param interaction_filename    Interaction-term filename inside `directory`.
 * @param single_site_filename    Single-site (Trans / Zeeman) filename.
 * @param counterterm_filename    Counter-term filename.
 * @param three_body_filename     Three-body term filename.
 * @return EDResults              Eigenvalues sorted ascending + optional
 *                                eigenvectors / timing / sector breakdown.
 */
inline EDResults exact_diagonalization(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& counterterm_filename = "CounterTerm.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat")
{
    // Canonicalize the legacy compound enums (``_GPU`` / ``_FIXED_SZ`` /
    // ``_MPI`` / ``_CUDA``) up-front so every downstream path -- file,
    // streaming, fixed-Sz -- sees the same base method + orthogonal
    // flags. The full-Hilbert path repeats this internally; doing it
    // here (a) makes the streaming kernel honour ``params.use_gpu``
    // (it used to read ``is_gpu_method(method)`` off the enum suffix
    // and miss the modern ``LANCZOS + use_gpu=true`` case) and (b)
    // means callers passing a legacy enum together with already-set
    // flags don't lose information.
    EDParameters resolved = params;
    {
        const auto canon = ed::canonicalize_method_and_flags(
            method,
            resolved.use_fixed_sz, resolved.use_gpu, resolved.use_mpi);
        method                = canon.method;
        resolved.use_fixed_sz = canon.use_fixed_sz;
        resolved.use_gpu      = canon.use_gpu;
        resolved.use_mpi      = canon.use_mpi;
    }

    // Phase 5 (matvec-unification): auto-promote use_symmetry when the
    // caller has a Hamiltonian directory with automorphism_results/
    // present. This is the "kick in automatically if the Hamiltonian
    // possesses it" half of the audit -- the streaming-symmetry kernel
    // is strictly faster than the full-Hilbert kernel whenever a
    // non-trivial spatial group is present, so flipping the axis on
    // for the user is unambiguously correct when the data is there.
    if (!resolved.use_symmetry && detail::symmetry_data_present(directory)) {
        resolved.use_symmetry = true;
        std::cerr << "[ed::exact_diagonalization] auto-detected "
                  << "automorphism_results/ in '" << directory
                  << "' -- routing through the streaming-symmetry "
                  << "kernel. Pass params.use_symmetry=true to make "
                  << "this explicit, or remove the directory to "
                  << "force the full-Hilbert path.\n";
    }

    // ----- Non-symmetry path: full-Hilbert / fixed-Sz / GPU / MPI -----
    // ed_wrapper.h's exact_diagonalization_from_directory already routes
    // on use_fixed_sz, use_gpu, and use_mpi internally.
    if (!resolved.use_symmetry) {
        return ::exact_diagonalization_from_directory(
            directory, method, resolved, format,
            interaction_filename, single_site_filename,
            counterterm_filename, three_body_filename);
    }

    // ----- Spatial-symmetry path: streaming kernel (per-sector matvec) -----
    // The streaming kernel honours use_gpu (per-sector GPU dispatch) and
    // use_fixed_sz (fixed-Sz orbit basis) inside its implementation.
    if (resolved.use_fixed_sz) {
        const std::int64_t n_up = (resolved.n_up >= 0)
            ? resolved.n_up
            : static_cast<std::int64_t>(resolved.num_sites / 2);
        return ::exact_diagonalization_streaming_symmetry_fixed_sz(
            directory, n_up, method, resolved,
            interaction_filename, single_site_filename,
            resolved.basis_cache_dir, resolved.precompute_basis_only);
    }
    return ::exact_diagonalization_streaming_symmetry(
        directory, method, resolved,
        interaction_filename, single_site_filename,
        resolved.basis_cache_dir, resolved.precompute_basis_only);
}

/**
 * @brief Canonical ED entry point (file-path form).
 *
 * Same contract as the directory overload but takes explicit file paths.
 * The directory used to locate `automorphism_results/` (when
 * params.use_symmetry == true) is derived as `dirname(interaction_file)`.
 * Performs the same legacy-enum canonicalisation and symmetry
 * auto-detection as the directory overload so both entries behave
 * symmetrically.
 */
inline EDResults exact_diagonalization(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    const std::string& counterterm_file = "",
    const std::string& three_body_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD)
{
    EDParameters resolved = params;
    {
        const auto canon = ed::canonicalize_method_and_flags(
            method,
            resolved.use_fixed_sz, resolved.use_gpu, resolved.use_mpi);
        method                = canon.method;
        resolved.use_fixed_sz = canon.use_fixed_sz;
        resolved.use_gpu      = canon.use_gpu;
        resolved.use_mpi      = canon.use_mpi;
    }

    // ----- Derive directory + basenames -----
    const auto split_path = [](const std::string& path,
                               std::string& dir, std::string& base) {
        const auto pos = path.find_last_of('/');
        if (pos == std::string::npos) {
            dir  = ".";
            base = path;
        } else {
            dir  = path.substr(0, pos);
            base = path.substr(pos + 1);
        }
    };

    std::string directory, interaction_basename;
    split_path(interaction_file, directory, interaction_basename);

    // Symmetric with the directory overload: if the caller didn't opt out
    // of symmetry and ``automorphism_results/`` sits next to the
    // interaction file, route through the streaming kernel automatically.
    if (!resolved.use_symmetry && detail::symmetry_data_present(directory)) {
        resolved.use_symmetry = true;
        std::cerr << "[ed::exact_diagonalization] auto-detected "
                  << "automorphism_results/ in '" << directory
                  << "' -- routing through the streaming-symmetry "
                  << "kernel. Pass params.use_symmetry=true to make "
                  << "this explicit, or remove the directory to "
                  << "force the full-Hilbert path.\n";
    }

    if (!resolved.use_symmetry) {
        return ::exact_diagonalization_from_files(
            interaction_file, single_site_file, counterterm_file,
            three_body_file, method, resolved, format);
    }

    std::string single_site_basename = "Trans.dat";
    if (!single_site_file.empty()) {
        std::string ignored_dir;
        split_path(single_site_file, ignored_dir, single_site_basename);
    }

    if (resolved.use_fixed_sz) {
        const std::int64_t n_up = (resolved.n_up >= 0)
            ? resolved.n_up
            : static_cast<std::int64_t>(resolved.num_sites / 2);
        return ::exact_diagonalization_streaming_symmetry_fixed_sz(
            directory, n_up, method, resolved,
            interaction_basename, single_site_basename,
            resolved.basis_cache_dir, resolved.precompute_basis_only);
    }
    return ::exact_diagonalization_streaming_symmetry(
        directory, method, resolved,
        interaction_basename, single_site_basename,
        resolved.basis_cache_dir, resolved.precompute_basis_only);
}

}  // namespace ed

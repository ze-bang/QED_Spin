#pragma once

// =============================================================================
// ed_dispatch_symmetry.h — forwarder.
//
// This header was the Phase 7.1 symmetry-aware dispatcher. The
// matvec-unification rollout (Phase 6) collapsed it onto the single
// canonical entry point:
//
//     ed::exact_diagonalization(...)        in <ed/core/dispatch.h>
//
// Existing callers that include <ed/core/ed_dispatch_symmetry.h> and call
//
//     ed_dispatch::exact_diagonalization_from_directory(...)
//     ed_dispatch::exact_diagonalization_from_files(...)
//
// keep working unchanged via the shim namespace below. New code should
// prefer the canonical entry point in `ed::`.
// =============================================================================

#include <ed/core/dispatch.h>

namespace ed_dispatch {

inline EDResults exact_diagonalization_from_directory(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& counterterm_filename = "CounterTerm.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat")
{
    return ::ed::exact_diagonalization(
        directory, method, params, format,
        interaction_filename, single_site_filename,
        counterterm_filename, three_body_filename);
}

inline EDResults exact_diagonalization_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    const std::string& counterterm_file = "",
    const std::string& three_body_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD)
{
    return ::ed::exact_diagonalization(
        interaction_file, single_site_file, counterterm_file,
        three_body_file, method, params, format);
}

}  // namespace ed_dispatch

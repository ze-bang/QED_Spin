#pragma once
// =============================================================================
// include/ed/core/ed_wrapper.h
//
// Surface unification (May 2026): this header used to be the ~2.1k-line
// previous home of the deleted `ed::exact_diagonalization_*` family
// (`_core`, `_all_sz_sectors`, `_all_sz_sectors_gpu`, `_fixed_sz`,
// `_from_files`, `_from_directory`). The entire family was deleted
// alongside the pybind11 forwarders in lockstep with the collapse to
// `ed::make_operator(OperatorSpec)` + `ed::workflows::{solve, thermal,
// spectral}`.
//
// Two POD types survive because the orchestrator's adapters and the
// CLI's standard / streaming workflows still read them: `EDResults`
// (eigenvalue + thermo envelope) and `EDParameters` (the
// modernisation-audit-split parameter bag). Both live in dedicated
// small headers so callers can include only what they need:
//
//   * `EDResults`   -> `<ed/core/ed_legacy_types.h>`
//   * `EDParameters`-> `<ed/core/ed_parameters.h>`
//
// This shim file forwards to both so the historical include path keeps
// resolving. New code should include the focused headers directly.
//
// External migration: any caller reaching for one of the deleted
// `ed::exact_diagonalization_*` functions builds the equivalent shape
// with the unified factory + orchestrator:
//
//     ed::OperatorSpec spec;
//     spec.source             = ed::DirectoryPath{"/path/to/dir"};
//     spec.num_sites          = 16;
//     spec.fixed_sz           = std::nullopt;
//     spec.streaming_symmetry = false;
//     auto op = ed::make_operator(spec);
//     auto gs = ed::workflows::solve(*op, ed::SolveOptions{ .num_eigs = 5 });
//
// See `include/ed/core/make_operator.h` and `include/ed/orchestrator.h`
// for the full surface, and `CHANGELOG.md` for the per-wave port log.
// =============================================================================

#include <ed/core/ed_legacy_types.h>
#include <ed/core/ed_parameters.h>

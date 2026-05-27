#pragma once
// =============================================================================
// include/ed/api/feasibility.h
//
// `ed::estimate_resources(spec, opts)` and `ed::suggest_workflow(spec, intent)`
// -- thin C++ ports of the small heuristics in `python/qed/feasibility.py`.
//
// These do NOT replicate the full Python feasibility planner (no host
// probing, no GPU memory queries, no ranked-list alternatives). They
// return the same per-rank memory + wall-time order-of-magnitude
// estimate the Python planner produces, plus a one-line device
// recommendation, so that small C++ examples can warn the user before
// dispatching a too-big problem.
//
// Author: ed-collapse, Phase A of the "mirror examples" plan (May 2026).
// =============================================================================

#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <ed/api.h>
#include <ed/core/make_operator.h>
#include <ed/core/results.h>
#include <ed/core/sector_thermo.h>
#include <ed/orchestrator.h>

namespace ed {

// ---------------------------------------------------------------------------
// ResourceEstimate -- output of `estimate_resources`.
// ---------------------------------------------------------------------------
struct ResourceEstimate {
    /// Estimated basis dimension (Sz-projected or full Hilbert).
    std::uint64_t basis_dim   = 0;
    /// Per-rank memory in GiB.
    double        per_rank_gb = 0.0;
    /// Total (summed across ranks) memory in GiB.
    double        total_gb    = 0.0;
    /// Wall-time order-of-magnitude estimate (seconds).
    double        wall_seconds = 0.0;
    /// One-line device recommendation: "cpu" / "gpu" / "mpi" / "mpi_gpu".
    std::string   device_recommendation = "cpu";
    /// Free-form rationale entries (e.g. "vec_count=5", "dim crossed 2^14
    /// -> GPU recommended").
    std::vector<std::string> notes;
};

// ---------------------------------------------------------------------------
// Intent -- what the caller is about to do. Used by `suggest_workflow`.
// ---------------------------------------------------------------------------
enum class Intent : unsigned {
    GroundState,
    FiniteTemperature,
    Spectral,
};

// ---------------------------------------------------------------------------
// WorkflowSuggestion -- output of `suggest_workflow`.
// ---------------------------------------------------------------------------
struct WorkflowSuggestion {
    /// Recommended verb: "solve", "thermal", "spectral".
    std::string   verb;
    /// Recommended method token for that verb (e.g. "LANCZOS", "FTLM").
    std::string   method;
    /// Recommended device (forwarded from `estimate_resources`).
    std::string   device;
    /// Whether to enable streaming symmetry (recommended when the spec
    /// already carries `streaming_symmetry=true` OR when N >= 18 and the
    /// caller has a directory source).
    bool          use_symmetry = false;
    /// Whether to enable fixed-Sz projection (recommended when the spec
    /// already carries `fixed_sz` OR when the Hamiltonian conserves Sz).
    bool          use_fixed_sz = false;
    /// Free-form rationale.
    std::vector<std::string> notes;
};

// ---------------------------------------------------------------------------
// estimate_resources(spec, opts) -- per-rank + total memory + a wall-time
// floor for a given (solver, device) combination. Pure math; no system
// probing.
// ---------------------------------------------------------------------------
// estimate_resources / suggest_workflow take the spec by const-ref BUT
// inspect ONLY the cheap-to-read fields (num_sites, fixed_sz,
// streaming_symmetry). They do NOT call `ed::make_operator`, so they
// work on any spec -- including those carrying an `InMemoryOperator`
// (which would otherwise force a move because the spec is move-only).
[[nodiscard]] ResourceEstimate
estimate_resources(const OperatorSpec&         spec,
                   const ed::api::SolveOptions& opts = {});

// ---------------------------------------------------------------------------
// suggest_workflow(spec, intent) -- pick (verb, method, device,
// use_symmetry, use_fixed_sz) for the caller's intent. Uses the same
// thresholds Python's `qed.suggest_workflow` uses.
// ---------------------------------------------------------------------------
[[nodiscard]] WorkflowSuggestion
suggest_workflow(const OperatorSpec& spec, Intent intent = Intent::GroundState);

// ---------------------------------------------------------------------------
// thermal_auto(spec, opts) -- Sz-loop wrapper for thermal workflows.
//
// Mirrors the `use_sz_if_conserved=True` path in `qed.thermal`: detects
// Sz conservation from the spec (or from the Hamiltonian when supplied
// in-memory), enumerates `n_up in [sz_min, sz_max]` (defaulting to
// `[0, num_sites]`), dispatches `ed::api::thermal` per sector with
// `fixed_sz=n_up`, and recombines per-sector thermodynamics via
// `ed::core::combine_sector_thermodynamics`.
//
// When Sz is NOT conserved (or `use_sz_if_conserved=false`), falls
// back to a single full-Hilbert `ed::api::thermal` call.
// ---------------------------------------------------------------------------
// thermal_auto: per-sector Sz loop. Defined `inline` because the body
// calls `ed::api::thermal(spec, opts)` which instantiates
// `ed::make_operator(spec)`. Keeping the body header-side keeps the
// `WITH_MPI` distributed-constructor reference at the consumer's link
// step rather than at the api library's compile step.
//
// Each sector iteration constructs a fresh OperatorSpec by copying the
// cheap source-discriminator fields off the input spec
// (`FilePaths` / `DirectoryPath`); `InMemoryOperator` is not supported
// because the variant cannot be replicated across sectors.

namespace detail {

// Binomial C(n, k) helper used by `thermal_auto` to compute per-sector
// dimensions. Kept inline so the header is self-contained.
[[nodiscard]] inline std::uint64_t binom_u64_thermal_auto(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    long double r = 1.0L;
    for (int i = 0; i < k; ++i) {
        r *= static_cast<long double>(n - i);
        r /= static_cast<long double>(i + 1);
    }
    return static_cast<std::uint64_t>(r + 0.5L);
}

// Build a fresh OperatorSpec for one Sz sector. Copies the cheap-to-copy
// source variants and rejects `InMemoryOperator` (variant is move-only).
[[nodiscard]] inline ed::OperatorSpec
spec_for_sz_sector(const ed::OperatorSpec& base, int n_up) {
    ed::OperatorSpec out;
    std::visit([&out](auto&& src) {
        using T = std::decay_t<decltype(src)>;
        if constexpr (std::is_same_v<T, ed::FilePaths> ||
                       std::is_same_v<T, ed::DirectoryPath>) {
            out.source = src;
        } else {
            throw std::runtime_error(
                "ed::thermal_auto: per-sector Sz loop requires a "
                "FilePaths or DirectoryPath source (InMemoryOperator "
                "cannot be replicated across sectors). Build the "
                "spec via DirectoryPath / FilePaths if you need the "
                "Sz auto-loop.");
        }
    }, base.source);
    out.num_sites          = base.num_sites;
    out.spin_l             = base.spin_l;
    out.fixed_sz           = n_up;
    out.streaming_symmetry = base.streaming_symmetry;
    out.distributed        = base.distributed;
    out.sector_index       = base.sector_index;
#ifdef WITH_MPI
    out.comm               = base.comm;
#endif
    return out;
}

}  // namespace detail

[[nodiscard]] inline ThermalResult
thermal_auto(OperatorSpec            spec,
             ed::api::ThermalOptions opts,
             bool                    use_sz_if_conserved,
             std::optional<int>      sz_min,
             std::optional<int>      sz_max) {
    if (!use_sz_if_conserved || spec.fixed_sz.has_value()) {
        return ed::api::thermal(std::move(spec), std::move(opts));
    }

    const int N    = static_cast<int>(spec.num_sites);
    const int slo  = sz_min.value_or(0);
    const int shi  = sz_max.value_or(N);
    if (shi < slo) {
        throw std::invalid_argument(
            "ed::thermal_auto: sz_max < sz_min after defaulting.");
    }

    std::vector<ThermodynamicData>   per_sector;
    std::vector<std::uint64_t>       per_sector_dims;
    std::vector<ThermalSectorEntry>  sector_entries;
    double gs_energy = std::numeric_limits<double>::infinity();
    BackendMetadata backend_meta;
    KrylovDiagnostics last_krylov;

    for (int n_up = slo; n_up <= shi; ++n_up) {
        const auto sector_dim = detail::binom_u64_thermal_auto(N, n_up);
        if (sector_dim == 0) continue;

        try {
            auto sector_spec = detail::spec_for_sz_sector(spec, n_up);
            auto sector_result = ed::api::thermal(std::move(sector_spec), opts);
            per_sector.push_back(sector_result.thermo);
            per_sector_dims.push_back(sector_dim);

            ThermalSectorEntry entry;
            entry.sz_index            = n_up;
            entry.ground_state_energy = sector_result.ground_state_energy;
            entry.thermo              = sector_result.thermo;
            entry.tag.sector_dim      = sector_dim;
            entry.tag.n_up            = n_up;
            sector_entries.push_back(std::move(entry));

            if (sector_result.ground_state_energy < gs_energy) {
                gs_energy = sector_result.ground_state_energy;
            }
            backend_meta = sector_result.backend;
            last_krylov  = sector_result.krylov;
        } catch (const std::exception& e) {
            std::ostringstream note;
            note << "skipped n_up=" << n_up << " (" << e.what() << ")";
            backend_meta.notes.emplace_back("thermal_auto.skip", note.str());
        }
    }

    if (per_sector.empty()) {
        throw std::runtime_error(
            "ed::thermal_auto: every Sz sector in [sz_min, sz_max] failed "
            "or was empty.");
    }

    ThermalResult combined;
    combined.thermo              = ed::core::combine_sector_thermodynamics(
                                       per_sector, per_sector_dims);
    combined.per_sector          = std::move(sector_entries);
    combined.ground_state_energy = gs_energy;
    combined.backend             = std::move(backend_meta);
    combined.backend.notes.emplace_back("thermal_auto.sectors_used",
                                        std::to_string(per_sector.size()));
    combined.krylov              = std::move(last_krylov);
    return combined;
}

}  // namespace ed

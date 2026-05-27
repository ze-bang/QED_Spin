// =============================================================================
// src/api/feasibility.cpp
//
// Implementation of the thin C++ ports of Python's
// `qed.estimate_resources` / `qed.suggest_workflow` / `qed.thermal` (the
// Sz-loop wrapper).
// =============================================================================

#include <ed/api/feasibility.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

namespace ed {

namespace {

// Binomial coefficient C(n, k) -- used to compute fixed-Sz sector
// dimensions without overflowing for n <= 64.
[[nodiscard]] std::uint64_t binom_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    long double r = 1.0L;
    for (int i = 0; i < k; ++i) {
        r *= static_cast<long double>(n - i);
        r /= static_cast<long double>(i + 1);
    }
    return static_cast<std::uint64_t>(r + 0.5L);
}

[[nodiscard]] std::uint64_t basis_dim_for(const OperatorSpec& spec) {
    const int N = static_cast<int>(spec.num_sites);
    if (spec.fixed_sz.has_value()) {
        return binom_u64(N, *spec.fixed_sz);
    }
    if (N >= 0 && N < 63) return std::uint64_t{1} << N;
    return std::numeric_limits<std::uint64_t>::max();
}

// Resident vector count for the solver family (rough, matches Python's
// `_vector_count_for`). Used to derive the per-rank memory estimate.
[[nodiscard]] std::size_t resident_vectors_for(std::string_view method,
                                               std::size_t num_eigs,
                                               bool         compute_eigenvectors) {
    auto contains = [&](std::string_view needle) {
        return method.find(needle) != std::string_view::npos;
    };
    std::size_t v = 5;
    if (contains("FULL") || contains("full"))             v = 3;
    else if (contains("BLOCK")|| contains("block"))       v = 4 + 2 * std::max<std::size_t>(num_eigs, 1);
    else if (contains("KRYLOV")|| contains("krylov"))     v = 3 + std::max<std::size_t>(num_eigs, 1);
    else if (contains("LANCZOS")|| contains("lanczos"))   v = 5;
    else if (contains("FTLM")  || contains("LTLM"))       v = 5;
    else if (contains("TPQ"))                              v = 4;
    else if (contains("KPM"))                              v = 4;
    if (compute_eigenvectors) v += std::max<std::size_t>(num_eigs, 1);
    return v;
}

}  // namespace

ResourceEstimate
estimate_resources(const OperatorSpec&         spec,
                   const ed::api::SolveOptions& opts) {
    ResourceEstimate est;
    est.basis_dim = basis_dim_for(spec);

    const std::size_t vec_count = resident_vectors_for(
        opts.solver, opts.num_eigenvalues, opts.compute_eigenvectors);

    constexpr double bytes_per_complex = 16.0;
    constexpr double GiB               = 1024.0 * 1024.0 * 1024.0;
    est.per_rank_gb = bytes_per_complex *
                      static_cast<double>(est.basis_dim) *
                      static_cast<double>(vec_count) / GiB;
    est.total_gb    = est.per_rank_gb;

    // Time floor: ns_per_term * basis_dim * iter_count / 1e9. Choose
    // CPU=100 ns/element, GPU=5 ns/element matching Python's planner.
    const double ns_per_elem = (opts.device == "gpu" || opts.device == "mpi_gpu") ? 5.0 : 100.0;
    const std::size_t iter_count = opts.max_iterations.value_or(
        std::max<std::size_t>(150, 30 * opts.num_eigenvalues));
    est.wall_seconds = ns_per_elem *
                       static_cast<double>(est.basis_dim) *
                       static_cast<double>(iter_count) / 1e9;

    // Device recommendation: dim under 2^14 prefers CPU; under 2^22
    // prefers GPU; beyond that, mpi_gpu when available.
    if (est.basis_dim < (std::uint64_t{1} << 14)) {
        est.device_recommendation = "cpu";
    } else if (est.basis_dim < (std::uint64_t{1} << 22)) {
        est.device_recommendation = has_cuda_build() ? "gpu" : "cpu";
    } else if (has_nccl_build()) {
        est.device_recommendation = "mpi_gpu";
    } else if (has_mpi_build()) {
        est.device_recommendation = "mpi";
    } else if (has_cuda_build()) {
        est.device_recommendation = "gpu";
    } else {
        est.device_recommendation = "cpu";
    }

    std::ostringstream note;
    note << "basis_dim=" << est.basis_dim
         << ", vec_count=" << vec_count
         << ", iter_count=" << iter_count;
    est.notes.push_back(note.str());
    return est;
}

WorkflowSuggestion
suggest_workflow(const OperatorSpec& spec, Intent intent) {
    WorkflowSuggestion s;
    const std::uint64_t dim = basis_dim_for(spec);

    // Device pick: same thresholds as estimate_resources.
    ed::api::SolveOptions probe;
    auto est = estimate_resources(spec, probe);
    s.device = est.device_recommendation;

    s.use_symmetry = spec.streaming_symmetry;
    s.use_fixed_sz = spec.fixed_sz.has_value();

    switch (intent) {
    case Intent::GroundState:
        s.verb = "solve";
        if (dim <= (std::uint64_t{1} << 11)) {
            s.method = "FULL";
        } else {
            s.method = "LANCZOS";
        }
        break;
    case Intent::FiniteTemperature:
        s.verb = "thermal";
        s.method = "FTLM";
        break;
    case Intent::Spectral:
        s.verb = "spectral";
        s.method = "ground_state_cf";
        break;
    }

    std::ostringstream note;
    note << "intent="
         << (intent == Intent::GroundState ? "GroundState" :
             intent == Intent::FiniteTemperature ? "FiniteTemperature" :
             "Spectral")
         << ", dim=" << dim;
    s.notes.push_back(note.str());
    return s;
}

// Note: `ed::thermal_auto(...)` lives `inline` in
// `include/ed/api/feasibility.h` so its per-sector loop instantiates
// `ed::api::thermal(spec, opts)` (which itself instantiates
// `ed::make_operator`) at the consumer's translation unit rather than
// in `ed_solvers_cpu`. This avoids a back-edge link dependency to
// `ed_distributed` (which already publicly depends on
// `ed_solvers_cpu`).

}  // namespace ed

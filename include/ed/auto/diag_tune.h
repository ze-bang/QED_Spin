#pragma once
// =============================================================================
// include/ed/auto/diag_tune.h
//
// Header-only mirror of `qed.auto_tune.tune_diag` (see
// python/qed/auto_tune.py). Picks ARPACK ncv, FTLM/LTLM Krylov dim,
// mTPQ Taylor order + delta_beta, tolerance, max_iterations,
// max_subspace, block_size for the ED auto-pilot.
//
// Header-only so `ed::auto_pilot::solve(...)` can mutate the
// caller-built `EDParameters` without an extra translation unit.
// Numeric constants are copied verbatim from `qed/auto_tune.py`;
// keep them in sync.
// =============================================================================

#include <ed/core/ed_parameters.h>
#include <ed/core/ed_wrapper.h>   // is_cuda_compiled, is_scalapack_compiled
#include <ed/core/operator.h>
#include <ed/auto/dssf_tune.h>    // for ed::auto_pilot::dssf::estimate_bandwidth

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace ed::auto_pilot::diag {

// Reuse the level enum from the DSSF tuner — same conservative/balanced/
// aggressive semantics.
using ed::auto_pilot::dssf::TuneLevel;
using ed::auto_pilot::dssf::estimate_bandwidth;

namespace detail {

inline double tolerance_for(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 1e-8;
        case TuneLevel::Balanced:     return 1e-10;
        case TuneLevel::Aggressive:   return 1e-12;
    }
    return 1e-10;
}

// (max_iter_floor, max_iter_per_eig, max_sub_floor, max_sub_per_eig)
inline std::tuple<int,int,int,int> krylov_bounds_for(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return {150,  6, 60,  3};
        case TuneLevel::Balanced:     return {200,  8, 80,  4};
        case TuneLevel::Aggressive:   return {400, 16, 160, 8};
    }
    return {200, 8, 80, 4};
}

inline int arpack_ncv_factor(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 2;
        case TuneLevel::Balanced:     return 4;
        case TuneLevel::Aggressive:   return 6;
    }
    return 4;
}

inline int ftlm_krylov(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 80;
        case TuneLevel::Balanced:     return 100;
        case TuneLevel::Aggressive:   return 160;
    }
    return 100;
}

inline int ltlm_krylov(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 150;
        case TuneLevel::Balanced:     return 200;
        case TuneLevel::Aggressive:   return 320;
    }
    return 200;
}

inline int tpq_taylor_default(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 50;
        case TuneLevel::Balanced:     return 100;
        case TuneLevel::Aggressive:   return 200;
    }
    return 100;
}

inline double tpq_delta_beta_default(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 5e-2;
        case TuneLevel::Balanced:     return 1e-2;
        case TuneLevel::Aggressive:   return 2e-3;
    }
    return 1e-2;
}

inline std::pair<int,int> thermal_sample_bounds(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return {1, 4};
        case TuneLevel::Balanced:     return {1, 16};
        case TuneLevel::Aggressive:   return {4, 32};
    }
    return {1, 16};
}

} // namespace detail

inline double pick_tolerance(TuneLevel L = TuneLevel::Balanced) {
    return detail::tolerance_for(L);
}

inline std::uint64_t pick_max_iterations(std::uint64_t num_eigenvalues,
                                         std::uint64_t sector_dim,
                                         TuneLevel L = TuneLevel::Balanced) {
    auto [floor, per_eig, _ms_floor, _ms_per] = detail::krylov_bounds_for(L);
    (void)_ms_floor; (void)_ms_per;
    std::uint64_t target = std::max<std::uint64_t>(
        static_cast<std::uint64_t>(floor),
        per_eig * std::max<std::uint64_t>(1, num_eigenvalues) + 80);
    if (sector_dim > 1) target = std::min<std::uint64_t>(target, sector_dim - 1);
    return target;
}

inline std::uint64_t pick_max_subspace(std::uint64_t num_eigenvalues,
                                       std::uint64_t sector_dim,
                                       TuneLevel L = TuneLevel::Balanced) {
    auto [_floor, _per, sub_floor, sub_per] = detail::krylov_bounds_for(L);
    (void)_floor; (void)_per;
    std::uint64_t target = std::max<std::uint64_t>(
        static_cast<std::uint64_t>(sub_floor),
        sub_per * std::max<std::uint64_t>(1, num_eigenvalues) + 40);
    if (sector_dim > 1) target = std::min<std::uint64_t>(target, sector_dim - 1);
    return target;
}

inline std::int64_t pick_arpack_ncv(std::uint64_t num_eigenvalues,
                                    TuneLevel L = TuneLevel::Balanced) {
    const int factor = detail::arpack_ncv_factor(L);
    return static_cast<std::int64_t>(
        std::max<std::uint64_t>(2 * num_eigenvalues + 1,
                                static_cast<std::uint64_t>(factor) * num_eigenvalues));
}

inline std::uint64_t pick_ftlm_krylov_dim(TuneLevel L = TuneLevel::Balanced) {
    return static_cast<std::uint64_t>(detail::ftlm_krylov(L));
}

inline std::uint64_t pick_ltlm_krylov_dim(TuneLevel L = TuneLevel::Balanced) {
    return static_cast<std::uint64_t>(detail::ltlm_krylov(L));
}

inline double pick_tpq_delta_beta(double bandwidth,
                                  TuneLevel L = TuneLevel::Balanced) {
    const double base = detail::tpq_delta_beta_default(L);
    if (bandwidth <= 0) return base;
    return std::min(base, 0.5 / bandwidth);
}

inline std::uint64_t pick_tpq_taylor_order(double bandwidth, double delta_beta,
                                           TuneLevel L = TuneLevel::Balanced) {
    const std::uint64_t base = static_cast<std::uint64_t>(detail::tpq_taylor_default(L));
    if (bandwidth <= 0 || delta_beta <= 0) return base;
    const double arg = 0.5 * bandwidth * delta_beta;
    if (arg <= 1.0) return base;
    std::uint64_t p = base;
    const double log_arg = std::log(arg);
    for (std::uint64_t k = 0; k < 2 * base; ++k) {
        double log_fact = 0.0;
        for (std::uint64_t i = 1; i <= p; ++i) log_fact += std::log(static_cast<double>(i));
        if (static_cast<double>(p) * log_arg - log_fact < -27.6) return p;
        p += 10;
    }
    return p;
}

inline std::uint64_t pick_num_thermal_samples(std::uint64_t sector_dim,
                                              TuneLevel L = TuneLevel::Balanced) {
    auto [lo, hi] = detail::thermal_sample_bounds(L);
    if (sector_dim <= 1) return static_cast<std::uint64_t>(hi);
    const double target_d = std::ceil(64.0 / std::sqrt(static_cast<double>(sector_dim)));
    auto target = static_cast<std::uint64_t>(std::max(1.0, target_d));
    return std::max<std::uint64_t>(static_cast<std::uint64_t>(lo),
           std::min<std::uint64_t>(static_cast<std::uint64_t>(hi), target));
}

// ---------------------------------------------------------------------------
// Sentinel-based mutator. Only fields still at their EDParameters
// struct default get overwritten — so anything the caller already set
// passes through untouched. Mirrors `qed.workflow.diag` step 4.5.
// ---------------------------------------------------------------------------

struct AutoTuneOverrides {
    double bandwidth = std::numeric_limits<double>::quiet_NaN();
    TuneLevel level = TuneLevel::Balanced;
    bool verbose = true;
};

inline void apply_auto_tune(EDParameters& params,
                            std::uint64_t sector_dim,
                            std::uint64_t num_eigenvalues,
                            const ::Operator* op,
                            const AutoTuneOverrides& ov = {}) {
    const TuneLevel L = ov.level;

    // Bandwidth.
    double bw = ov.bandwidth;
    if (!std::isfinite(bw)) {
        bw = (op != nullptr) ? estimate_bandwidth(*op) : 4.0;
    }

    // Tolerance — sentinel = struct default 1e-10. We compare against
    // the canonical sentinel; if caller set anything else, we leave it.
    if (params.tolerance == 1e-10) {
        params.tolerance = pick_tolerance(L);
    }
    // max_iterations — sentinel = struct default 10000.
    if (params.max_iterations == 10000) {
        params.max_iterations = pick_max_iterations(num_eigenvalues, sector_dim, L);
    }
    // max_subspace — sentinel = struct default 100.
    if (params.max_subspace == 100) {
        params.max_subspace = pick_max_subspace(num_eigenvalues, sector_dim, L);
    }
    // arpack_ncv — sentinel = -1 (meaning "auto").
    if (params.arpack_ncv == -1) {
        params.arpack_ncv = pick_arpack_ncv(num_eigenvalues, L);
    }
    // ftlm_krylov_dim — sentinel = struct default 100.
    if (params.ftlm_krylov_dim == 100) {
        params.ftlm_krylov_dim = pick_ftlm_krylov_dim(L);
    }
    // ltlm_krylov_dim — sentinel = struct default 200.
    if (params.ltlm_krylov_dim == 200) {
        params.ltlm_krylov_dim = pick_ltlm_krylov_dim(L);
    }
    // ltlm_ground_krylov — sentinel = struct default 100. Reuse FTLM heuristic.
    if (params.ltlm_ground_krylov == 100) {
        params.ltlm_ground_krylov = pick_ftlm_krylov_dim(L);
    }
    // tpq_delta_beta — sentinel = struct default 1e-2.
    if (params.tpq_delta_beta == 1e-2) {
        params.tpq_delta_beta = pick_tpq_delta_beta(bw, L);
    }
    // tpq_taylor_order — sentinel = struct default 100.
    if (params.tpq_taylor_order == 100) {
        params.tpq_taylor_order = pick_tpq_taylor_order(bw, params.tpq_delta_beta, L);
    }

    if (ov.verbose) {
        std::cerr << "[ed::auto_pilot::diag] auto-tune: tol="
                  << params.tolerance
                  << " max_iter=" << params.max_iterations
                  << " max_sub=" << params.max_subspace
                  << " ftlm_M=" << params.ftlm_krylov_dim
                  << " ltlm_M=" << params.ltlm_krylov_dim
                  << " tpq_p=" << params.tpq_taylor_order
                  << " tpq_dbeta=" << params.tpq_delta_beta
                  << " arpack_ncv=" << params.arpack_ncv
                  << " bandwidth~=" << bw << "\n";
    }
}

} // namespace ed::auto_pilot::diag

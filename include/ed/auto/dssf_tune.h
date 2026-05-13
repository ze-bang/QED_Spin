#pragma once
// =============================================================================
// include/ed/auto/dssf_tune.h
//
// Header-only mirror of `qed.auto_tune` (python/qed/auto_tune.py) — the
// pure heuristic functions that pick η broadening / ω window / Krylov
// dim / # random vectors / KPM moments / device for the DSSF auto-pilot.
//
// Kept header-only so `ed::auto_pilot::dssf::compute(...)` can mutate
// the caller's `EDConfig` without pulling in extra translation units.
// The numeric constants are copied verbatim from `qed/auto_tune.py`;
// keep them in sync.
// =============================================================================

#include <ed/core/ed_config.h>
#include <ed/core/ed_wrapper.h>   // is_cuda_compiled, is_scalapack_compiled
#include <ed/core/operator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ed::auto_pilot::dssf {

// ---------------------------------------------------------------------------
// Aggressiveness levels — matches qed.auto_tune.Level.
// ---------------------------------------------------------------------------
enum class TuneLevel { Conservative, Balanced, Aggressive };

namespace detail {

inline double eta_grid_factor(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 5.0;
        case TuneLevel::Balanced:     return 3.0;
        case TuneLevel::Aggressive:   return 2.0;
    }
    return 3.0;
}

inline std::pair<int,int> krylov_bounds(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return {40, 60};
        case TuneLevel::Balanced:     return {80, 200};
        case TuneLevel::Aggressive:   return {120, 400};
    }
    return {80, 200};
}

inline std::pair<int,int> random_bounds(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return {4, 8};
        case TuneLevel::Balanced:     return {4, 32};
        case TuneLevel::Aggressive:   return {8, 64};
    }
    return {4, 32};
}

inline int kpm_moments_default(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 512;
        case TuneLevel::Balanced:     return 2048;
        case TuneLevel::Aggressive:   return 8192;
    }
    return 2048;
}

inline int num_omega_default(TuneLevel L) {
    switch (L) {
        case TuneLevel::Conservative: return 200;
        case TuneLevel::Balanced:     return 400;
        case TuneLevel::Aggressive:   return 800;
    }
    return 400;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Pure helpers (mirrors of qed.auto_tune.*).
// ---------------------------------------------------------------------------

inline double estimate_bandwidth(const ::Operator& op, double fallback = 4.0) {
    const auto n = static_cast<double>(op.getNumBits());
    double total = 0.0;
    bool seen = false;
    for (const auto& t : op.transform_data_) {
        total += std::abs(t.coefficient);
        seen = true;
    }
    for (const auto& t : op.three_body_data_) {
        total += std::abs(t.coefficient);
        seen = true;
    }
    if (!seen) return fallback * (n > 0 ? n : 1.0);
    return 2.0 * total;
}

inline std::pair<double,double>
pick_omega_window(double bandwidth, double margin = 0.1) {
    const double half = (1.0 + margin) * std::abs(bandwidth);
    return {-half, half};
}

inline int pick_num_omega_points(TuneLevel L = TuneLevel::Balanced) {
    return detail::num_omega_default(L);
}

inline double pick_eta(double bandwidth, int num_omega_points,
                       TuneLevel L = TuneLevel::Balanced,
                       double margin = 0.1) {
    if (num_omega_points <= 1) {
        return detail::eta_grid_factor(L) * 0.05 * std::abs(bandwidth);
    }
    auto [omin, omax] = pick_omega_window(bandwidth, margin);
    const double dw = (omax - omin) / static_cast<double>(num_omega_points - 1);
    return detail::eta_grid_factor(L) * dw;
}

inline int pick_krylov_dim(std::uint64_t sector_dim,
                           TuneLevel L = TuneLevel::Balanced) {
    auto [lo, hi] = detail::krylov_bounds(L);
    const int target = std::max(1, static_cast<int>(std::round(
        std::cbrt(static_cast<double>(sector_dim)))));
    return std::max(lo, std::min(hi, target));
}

inline int pick_num_random_vectors(std::uint64_t sector_dim,
                                   TuneLevel L = TuneLevel::Balanced) {
    auto [lo, hi] = detail::random_bounds(L);
    if (sector_dim <= 1) return hi;
    const int target = static_cast<int>(std::ceil(
        64.0 / std::sqrt(static_cast<double>(sector_dim))));
    return std::max(lo, std::min(hi, std::max(1, target)));
}

inline int pick_kpm_moments(TuneLevel L = TuneLevel::Balanced) {
    return detail::kpm_moments_default(L);
}

// ---------------------------------------------------------------------------
// Device picker — mirrors qed.auto_tune.pick_device.
// ---------------------------------------------------------------------------

enum class DSSFDevice { Auto, CPU, GPU, MPI, MPIGPU };

inline DSSFDevice pick_device(std::uint64_t sector_dim,
                              DSSFDevice user_request = DSSFDevice::Auto,
                              std::uint64_t gpu_dim_threshold = (1ULL << 17),
                              std::uint64_t mpi_dim_threshold = (1ULL << 22)) {
    if (user_request != DSSFDevice::Auto) return user_request;
    const bool use_gpu = is_cuda_compiled() && sector_dim >= gpu_dim_threshold;
    const bool use_mpi = is_scalapack_compiled()
                         && sector_dim >= mpi_dim_threshold;
    if (use_gpu && use_mpi) return DSSFDevice::MPIGPU;
    if (use_gpu)            return DSSFDevice::GPU;
    if (use_mpi)            return DSSFDevice::MPI;
    return DSSFDevice::CPU;
}

inline std::string to_string(DSSFDevice d) {
    switch (d) {
        case DSSFDevice::Auto:   return "auto";
        case DSSFDevice::CPU:    return "cpu";
        case DSSFDevice::GPU:    return "gpu";
        case DSSFDevice::MPI:    return "mpi";
        case DSSFDevice::MPIGPU: return "mpi_gpu";
    }
    return "auto";
}

// ---------------------------------------------------------------------------
// EDConfig auto-tuning entry point. Fills in missing knobs (broadening,
// omega window, krylov, num_random, KPM moments) on `cfg.dynamical`,
// `cfg.static_resp`, and the relevant per-method fields. The "missing"
// criterion is "field equals its struct default" — matches the EDConfig
// merge semantics in src/core/ed_config.cpp:392.
// ---------------------------------------------------------------------------

struct AutoTuneOverrides {
    // Any field left at its sentinel (NaN for double, 0 for integer) means
    // "let the auto-tuner pick". Pass an explicit value to override.
    double      bandwidth          = std::nan("");
    double      eta                = std::nan("");
    int         num_omega_points   = 0;
    int         krylov_dim         = 0;
    int         num_random_vectors = 0;
    int         kpm_moments        = 0;
    DSSFDevice  device             = DSSFDevice::Auto;
    TuneLevel   level              = TuneLevel::Balanced;
    bool        verbose            = true;
};

inline void apply_auto_tune(EDConfig& cfg,
                            std::uint64_t sector_dim,
                            const ::Operator* op,
                            const AutoTuneOverrides& ov = {}) {
    // Bandwidth.
    double W = std::isnan(ov.bandwidth)
        ? (op ? estimate_bandwidth(*op) : 4.0 * cfg.system.num_sites)
        : ov.bandwidth;

    // Number of omega points.
    int npts = ov.num_omega_points > 0
        ? ov.num_omega_points
        : pick_num_omega_points(ov.level);

    // Eta.
    double eta = std::isnan(ov.eta)
        ? pick_eta(W, npts, ov.level)
        : ov.eta;

    // Krylov / random vectors.
    int kdim = ov.krylov_dim > 0
        ? ov.krylov_dim
        : pick_krylov_dim(sector_dim, ov.level);
    int rvec = ov.num_random_vectors > 0
        ? ov.num_random_vectors
        : pick_num_random_vectors(sector_dim, ov.level);

    auto [omin, omax] = pick_omega_window(W);

    // Apply to dynamical block (only fields still at default).
    if (cfg.dynamical.broadening == 0.1)            cfg.dynamical.broadening    = eta;
    if (cfg.dynamical.num_omega_points == 1000)     cfg.dynamical.num_omega_points = static_cast<uint64_t>(npts);
    if (cfg.dynamical.omega_min == -5.0)            cfg.dynamical.omega_min     = omin;
    if (cfg.dynamical.omega_max ==  5.0)            cfg.dynamical.omega_max     = omax;
    if (cfg.dynamical.krylov_dim == 400)            cfg.dynamical.krylov_dim    = static_cast<uint64_t>(kdim);
    if (cfg.dynamical.num_random_states == 20)      cfg.dynamical.num_random_states = static_cast<uint64_t>(rvec);

    // Apply to static block.
    if (cfg.static_resp.krylov_dim == 400)          cfg.static_resp.krylov_dim    = static_cast<uint64_t>(kdim);
    if (cfg.static_resp.num_random_states == 20)    cfg.static_resp.num_random_states = static_cast<uint64_t>(rvec);

    // Device.
    DSSFDevice dev = pick_device(sector_dim, ov.device);
    if (dev == DSSFDevice::GPU || dev == DSSFDevice::MPIGPU) {
        cfg.dynamical.use_gpu  = true;
        cfg.static_resp.use_gpu = true;
        cfg.system.use_gpu      = true;
    }
    if (dev == DSSFDevice::MPI || dev == DSSFDevice::MPIGPU) {
        cfg.system.use_mpi = true;
    }

    if (ov.verbose) {
        std::cerr << "[ed::auto_pilot::dssf::apply_auto_tune] "
                  << "W=" << W
                  << " eta=" << eta
                  << " krylov=" << kdim
                  << " R=" << rvec
                  << " omega=[" << omin << "," << omax << "]x" << npts
                  << " device=" << to_string(dev)
                  << " level=" << static_cast<int>(ov.level) << "\n";
    }
}

} // namespace ed::auto_pilot::dssf

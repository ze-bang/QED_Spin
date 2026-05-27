// =============================================================================
// src/api/api_facade.cpp
//
// `ed::api::*` -- Python-named kwargs facade implementation.
// =============================================================================

#include <ed/api.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace ed::api {

namespace {

// Case-insensitive equality (ASCII).
[[nodiscard]] bool ieq(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ca = static_cast<unsigned char>(a[i]);
        const auto cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) return false;
    }
    return true;
}

// Strip surrounding whitespace.
[[nodiscard]] std::string_view trim(std::string_view s) {
    auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    while (!s.empty() && is_ws(s.front())) s.remove_prefix(1);
    while (!s.empty() && is_ws(s.back()))  s.remove_suffix(1);
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Method parsers.
// ---------------------------------------------------------------------------

std::optional<ed::workflows::SolveMethod>
parse_solve_method(std::string_view name) {
    const auto t = trim(name);
    if (t.empty() || ieq(t, "auto"))             return ed::workflows::SolveMethod::Auto;
    if (ieq(t, "LANCZOS")        || ieq(t, "Lanczos"))
        return ed::workflows::SolveMethod::Lanczos;
    if (ieq(t, "BLOCK_LANCZOS")  || ieq(t, "BlockLanczos") || ieq(t, "block_lanczos"))
        return ed::workflows::SolveMethod::BlockLanczos;
    if (ieq(t, "KRYLOV_SCHUR")   || ieq(t, "KrylovSchur")  || ieq(t, "krylov_schur"))
        return ed::workflows::SolveMethod::KrylovSchur;
    if (ieq(t, "FULL")           || ieq(t, "FullDiag")     || ieq(t, "full_diag")
        || ieq(t, "full"))
        return ed::workflows::SolveMethod::FullDiag;
    return std::nullopt;
}

std::optional<ed::workflows::ThermalOptions::Method>
parse_thermal_method(std::string_view name) {
    using M = ed::workflows::ThermalOptions::Method;
    const auto t = trim(name);
    if (ieq(t, "FTLM"))                                 return M::FTLM;
    if (ieq(t, "LTLM"))                                 return M::LTLM;
    if (ieq(t, "mTPQ") || ieq(t, "MTPQ") || ieq(t, "mtpq")) return M::mTPQ;
    if (ieq(t, "cTPQ") || ieq(t, "CTPQ") || ieq(t, "ctpq")) return M::cTPQ;
    if (ieq(t, "KPM_DOS") || ieq(t, "KpmDos") || ieq(t, "kpm_dos") || ieq(t, "KPMDOS"))
        return M::KpmDos;
    return std::nullopt;
}

std::optional<ed::workflows::SpectralOptions::Method>
parse_spectral_method(std::string_view name) {
    using M = ed::workflows::SpectralOptions::Method;
    const auto t = trim(name);
    if (t.empty() || ieq(t, "auto"))                       return std::nullopt;
    if (ieq(t, "ground_state_cf")   || ieq(t, "GroundStateCF")
        || ieq(t, "ground_state_dssf")|| ieq(t, "ground_state"))
        return M::GroundStateCF;
    if (ieq(t, "ftlm_dynamical")    || ieq(t, "FtlmDynamical")
        || ieq(t, "dynamical_thermal") || ieq(t, "dynamical"))
        return M::FtlmDynamical;
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// device_constraints.
// ---------------------------------------------------------------------------

ed::BackendConstraints
device_constraints(std::string_view device, std::uint64_t dim_hint) {
    const auto t = trim(device);
    ed::BackendConstraints c;

    // Default ("" / "auto"): allow everything, let the runtime pick. We
    // mirror Python's `workflow.py::_resolve_device("auto")` threshold by
    // disabling GPU when dim is small (and thus the CPU path is faster
    // due to launch overhead).
    if (t.empty() || ieq(t, "auto")) {
        c.allow_gpu     = true;
        c.allow_mpi     = true;
        c.allow_mpi_gpu = true;
        // Below 2^14 dim => keep CPU only (Python `workflow.py` threshold).
        if (dim_hint > 0 && dim_hint < (std::uint64_t(1) << 14)) {
            c.allow_gpu     = false;
            c.allow_mpi_gpu = false;
        }
        return c;
    }
    if (ieq(t, "cpu")) {
        c.allow_gpu     = false;
        c.allow_mpi     = false;
        c.allow_mpi_gpu = false;
        return c;
    }
    if (ieq(t, "gpu") || ieq(t, "cuda")) {
        c.allow_gpu     = true;
        c.allow_mpi     = false;
        c.allow_mpi_gpu = false;
        return c;
    }
    if (ieq(t, "mpi")) {
        c.allow_gpu     = false;
        c.allow_mpi     = true;
        c.allow_mpi_gpu = false;
        return c;
    }
    if (ieq(t, "mpi_gpu") || ieq(t, "mpi+gpu") || ieq(t, "mpicuda")) {
        c.allow_gpu     = true;
        c.allow_mpi     = true;
        c.allow_mpi_gpu = true;
        return c;
    }
    throw std::invalid_argument(
        "ed::api::device_constraints: unknown device token '" +
        std::string{t} +
        "'. Accepts 'auto' / 'cpu' / 'gpu' / 'mpi' / 'mpi_gpu'.");
}

// ---------------------------------------------------------------------------
// to_legacy: Python-named options -> C++-named options.
// ---------------------------------------------------------------------------

ed::workflows::SolveOptions
to_legacy(const SolveOptions& opts, std::uint64_t dim_hint) {
    ed::workflows::SolveOptions wf;
    wf.num_eigs        = opts.num_eigenvalues;
    wf.tolerance       = opts.tolerance;
    wf.compute_vectors = opts.compute_eigenvectors;
    wf.max_iter        = opts.max_iterations.value_or(0);
    if (opts.block_size) wf.block_size = *opts.block_size;
    wf.output_dir      = opts.output_dir;
    wf.selected_sectors = opts.selected_sectors;
    wf.use_fixed_sz    = opts.sz.has_value();
    wf.n_up            = opts.sz.value_or(-1);
    if (!opts.solver.empty()) {
        if (auto m = parse_solve_method(opts.solver); m.has_value()) {
            wf.method = *m;
        } else {
            throw std::invalid_argument(
                "ed::api::solve: unknown solver token '" + opts.solver +
                "'. Accepts 'LANCZOS', 'BLOCK_LANCZOS', 'KRYLOV_SCHUR', "
                "'FULL', 'auto'.");
        }
    }
    wf.backend = device_constraints(opts.device, dim_hint);
    return wf;
}

ed::workflows::ThermalOptions
to_legacy(const ThermalOptions& opts, std::uint64_t dim_hint) {
    ed::workflows::ThermalOptions wf;
    if (!opts.method.empty()) {
        if (auto m = parse_thermal_method(opts.method); m.has_value()) {
            wf.method = *m;
        } else {
            throw std::invalid_argument(
                "ed::api::thermal: unknown method token '" + opts.method +
                "'. Accepts 'FTLM', 'LTLM', 'mTPQ', 'cTPQ', 'KPM_DOS'.");
        }
    }
    wf.num_samples         = opts.num_samples;
    if (opts.krylov_dim) wf.krylov_dim = *opts.krylov_dim;
    wf.taylor_order        = opts.tpq_taylor_order;
    wf.delta_beta          = opts.tpq_delta_beta;
    wf.random_seed         = opts.random_seed;
    wf.output_dir          = opts.output_dir;
    wf.temp_min            = opts.T_min;
    wf.temp_max            = opts.T_max;
    wf.num_temp_bins       = opts.num_T;
    wf.selected_sectors    = opts.selected_sectors;
    wf.kpm_num_moments     = opts.kpm_num_moments;
    wf.kpm_num_random_vectors = opts.kpm_num_random_vectors;
    wf.backend = device_constraints(opts.device, dim_hint);
    return wf;
}

ed::workflows::SpectralOptions
to_legacy(const SpectralOptions& opts, std::uint64_t dim_hint) {
    ed::workflows::SpectralOptions wf;
    if (!opts.method.empty()) {
        if (auto m = parse_spectral_method(opts.method); m.has_value()) {
            wf.method = *m;
        }
        // Python "auto"/"" => leave as default GroundStateCF unless T set.
    }
    if (opts.method.empty()) {
        // Auto: if temperatures are present, switch to FtlmDynamical.
        if (!opts.temperatures.empty() || opts.T.has_value()) {
            wf.method = ed::workflows::SpectralOptions::Method::FtlmDynamical;
        }
    }
    if (opts.krylov_dim) wf.krylov_dim = *opts.krylov_dim;
    wf.broadening    = opts.eta;
    if (!opts.omega.empty()) {
        wf.omega_min = opts.omega.front();
        wf.omega_max = opts.omega.back();
        wf.num_omega = opts.omega.size();
    } else {
        wf.omega_min = opts.omega_min;
        wf.omega_max = opts.omega_max;
        wf.num_omega = opts.num_omega;
    }
    wf.energy_shift     = opts.energy_shift;
    wf.output_dir       = opts.output_dir;
    if (opts.num_random_vectors) wf.num_samples = *opts.num_random_vectors;
    wf.temperatures     = opts.temperatures;
    if (opts.T.has_value() && wf.temperatures.empty()) {
        wf.temperatures.push_back(*opts.T);
    }
    wf.observable_type  = opts.observable_type;
    wf.momentum_transfer = opts.momentum_transfer;
    wf.momentum_tolerance = opts.momentum_tolerance;
    wf.selected_sectors = opts.selected_sectors;
    wf.backend = device_constraints(opts.device, dim_hint);
    return wf;
}

// ---------------------------------------------------------------------------
// Verb facades.
//
// The OperatorSpec form builds the operator via `ed::make_operator` and
// routes to the same underlying `ed::workflows::*` call as the
// LinearOperator form, so the two paths are guaranteed bit-equal modulo
// the operator-construction side-effects (which are deterministic given
// the spec).
// ---------------------------------------------------------------------------

// Note: the OperatorSpec-form overloads live `inline` in `include/ed/api.h`
// to keep the `WITH_MPI` distributed-constructor back-edge at the
// consumer link site (where it already lives via the test / example
// helpers). Only the LinearOperator-form lives here, since it does NOT
// instantiate `ed::make_operator` and so does not need `ed_distributed`.

ed::GroundStateResult solve(const ed::LinearOperator& H, SolveOptions opts) {
    auto wf_opts = to_legacy(opts, H.geometry().global_dim);
    return ed::workflows::solve(H, std::move(wf_opts));
}

ed::ThermalResult thermal(const ed::LinearOperator& H, ThermalOptions opts) {
    auto wf_opts = to_legacy(opts, H.geometry().global_dim);
    return ed::workflows::thermal(H, std::move(wf_opts));
}

ed::SpectralResult spectral(const ed::LinearOperator& H,
                            const std::vector<const ed::LinearOperator*>& observables,
                            SpectralOptions opts) {
    auto wf_opts = to_legacy(opts, H.geometry().global_dim);
    return ed::workflows::spectral(H, observables, std::move(wf_opts));
}

}  // namespace ed::api

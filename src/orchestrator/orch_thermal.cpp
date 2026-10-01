// =============================================================================
// src/orchestrator/orch_thermal.cpp -- ed::workflows::thermal and its lanes
// (exact-small eigenspectrum fallback, mTPQ sampling, FTLM / OFTLM).
// Part of the workflow orchestrator; see orchestrator_internal.h for the
// file map.
// =============================================================================

#include "orchestrator_internal.h"

namespace ed::workflows {

namespace {

// ---------------------------------------------------------------------------
// Exact canonical thermodynamics from a complete eigenspectrum.
//
// Used as a small-sector fallback for mTPQ: when the Hilbert-space
// dimension is small (D <= ThermalOptions::dense_max_dim), the stochastic mTPQ
// estimator has large per-sample variance (no algorithmic bug — the inherent
// limitation is that TPQ needs D >> num_samples for good typicality). For
// the sz_spatial symmetry mode on N=8, the (n_up, k) sub-sectors have
// D ≈ 1–9, which causes dE failures of ~0.12 even with 20 samples.
//
// Math:
//   Z(β) = Σ_j exp(-β E_j)
//   F(T)  = −T ln Z(β)     ← includes the ln(D) baseline automatically
//   E(T)  = Σ_j E_j w_j,   w_j = exp(−β E_j)/Z
//   Cv(T) = β² (Σ_j E_j² w_j − E(T)²)
//   S(T)  = β (E − F)
//
// The absolute free energy (F carries ln(D) at high T) is exactly what
// combine_sector_thermodynamics expects for the per-sector Boltzmann weight.
// ---------------------------------------------------------------------------

// Forwards to the single canonical implementation (and its guards) in
// ed/symmetry/canonical_thermo.h.
static ThermodynamicData compute_canonical_thermo_from_eigs(
    const std::vector<double>& eigs,
    const std::vector<double>& temperatures)
{
    return ed::symmetry::canonical_thermo_from_eigs(eigs, temperatures);
}

}  // namespace

ThermalResult thermal(const LinearOperator& H, ThermalOptions opts) {
    require_hermitian_input(H, "ed::thermal");
    // mTPQ and FTLM dispatch through their `*_kernel<Backend>` templates
    // (CPU and CUDA); OFTLM is CPU-only. The variant visit at each lane
    // keeps the dispatch backend-agnostic.
    using Complex = std::complex<double>;
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(H.geometry().local_dim));
    ed::parallel::pin_omp_threads_once();

    if (!opts.observables.empty() && opts.method != ThermalOptions::Method::FTLM)
        throw std::invalid_argument("ed::thermal: observables need Method::FTLM");

    // Memory guard (thermal lane). The operator's basis is already built, so
    // the binding constraint is the kernel WORKING SET: FTLM keeps a
    // krylov_dim window of length-N vectors; TPQ a handful. Throw cleanly
    // before the Krylov-basis allocation rather than OOM-crash. H.global_dim()
    // is the per-call working dim (symmetry iterates sectors a level up, so this
    // is the sector dim there). FTLM and OFTLM keep one sample's Krylov basis
    // at a time; TPQ is O(1) state vectors.
    {
        const std::uint64_t D = H.global_dim();
        std::uint64_t vecs;
        constexpr std::uint64_t elem = 16ull;  // complex<double>
        switch (opts.method) {
            case ThermalOptions::Method::FTLM:
                vecs = std::max<std::size_t>(opts.krylov_dim, 4) + 4; break;
            case ThermalOptions::Method::OFTLM:
                // per-sample Krylov basis + the exact-eigenpair Lanczos basis
                vecs = std::max<std::size_t>(opts.krylov_dim, 4)
                     + 2 * opts.num_exact + 34; break;
            default:  // mTPQ: a handful of state vectors
                vecs = 8; break;
        }
        ed::core::guard_working_set(D * vecs * elem, "ed::thermal");
    }

    // When the caller does not supply an explicit ``opts.betas`` grid,
    // construct one from the temperature-scan knobs (``temp_min``,
    // ``temp_max``, ``num_temp_bins``) and mirror the resulting
    // temperature axis into ``R.thermo.temperatures`` so downstream
    // consumers can read the scan back without recomputing it from
    // ``opts.*``.
    if (opts.betas.empty() && opts.num_temp_bins > 0
        && opts.temp_min > 0.0 && opts.temp_max > opts.temp_min) {
        opts.betas.reserve(opts.num_temp_bins);
        const double t_lo = opts.temp_min;
        const double t_hi = opts.temp_max;
        const std::size_t n = opts.num_temp_bins;
        // Linear temperature axis (T = T_min + i*(T_max-T_min)/(n-1)),
        // descending in beta so the natural ascending-T print stays
        // ascending after the kernel.
        for (std::size_t i = 0; i < n; ++i) {
            const double T = (n == 1)
                ? t_lo
                : t_lo + (t_hi - t_lo) * static_cast<double>(i)
                          / static_cast<double>(n - 1);
            opts.betas.push_back(T > 0.0 ? 1.0 / T : 1.0 / 1e-300);
        }
    }

    ThermalResult R;
    if (!opts.betas.empty()) {
        R.thermo.temperatures.reserve(opts.betas.size());
        for (double b : opts.betas) {
            R.thermo.temperatures.push_back(b > 0.0 ? 1.0 / b : 0.0);
        }
    }
    const auto t0 = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // Small-sector exact-thermal fallback for EVERY sampling method.
    //
    // Every stochastic thermal method needs D >> num_samples for good
    // typicality. For small sectors (D <= opts.dense_max_dim), the per-sample
    // variance is too high for the dE tolerance even with 20+ samples (e.g.
    // within an n_up block, translation k-sectors have D ≈ 1–9 for N=8,
    // giving a statistical error of ~0.12 with 20 samples).
    //
    // Here the exact solve is both free and machine precise (at dim=64,
    // N=6 ring: 1.4e-15 exact vs 2.3e-02 FTLM sampling). The deliverable of
    // FTLM / OFTLM / mTPQ is identical -- canonical E(T)/C(T)/S(T) -- so
    // every sampling method takes the exact route.
    //
    // For any D <= opts.dense_max_dim, diagonalise exactly and compute the
    // canonical partition function directly. The resulting ThermodynamicData
    // uses the same absolute free-energy convention (F includes ln(D) at
    // high T) as the mTPQ canonical estimator, so it plugs in
    // correctly to combine_sector_thermodynamics for Sz/spatial recombination.
    // -----------------------------------------------------------------------
    const bool is_sampling_thermo_method =
        opts.method == ThermalOptions::Method::mTPQ  ||
        opts.method == ThermalOptions::Method::FTLM  ||
        opts.method == ThermalOptions::Method::OFTLM;
    // NOTE: this must NOT return early. Everything below the method dispatch
    // (free energy, timing, lane metadata) has to run for the exact result
    // exactly as it does for a sampled one. So: fill R.thermo, then flag the
    // dispatch chain to stand down.
    bool exact_thermo_done = false;
    bool host_only = false;   // the exact fallback and OFTLM run on the host whatever the backend
    const bool exact_small =
        is_sampling_thermo_method &&
        H.geometry().global_dim > 0 &&
        H.geometry().global_dim <= opts.dense_max_dim &&
        !R.thermo.temperatures.empty() &&
        // A seed transform restricts the stochastic trace to a
        // SUBSPACE (e.g. one spin tower). The exact fallback diagonalises
        // the whole block and would silently ignore the restriction --
        // stand down and let the sampling kernel honour the projection.
        // (Exact per-tower thermo needs the tower projection applied before
        // diagonalising, which this fallback does not do.)
        !opts.seed_transform && opts.observables.empty();
    // The exact fallback is a dense solve on the host by design; a device
    // requirement applies to the sampling kernels only.
    BackendConstraints constraints = opts.backend;
    if (exact_small) constraints.require_gpu = false;
    BackendVariant variant = select_backend(H.geometry(), constraints);
    if (exact_small) {
        const std::uint64_t D = H.geometry().global_dim;
        std::vector<double> eigs;
        full_diagonalization(H, D, D, eigs, /*compute_eigenvectors=*/false);
        if (!eigs.empty()) {
            R.thermo = compute_canonical_thermo_from_eigs(
                eigs, R.thermo.temperatures);
            R.ground_state_energy = eigs.front();
            R.backend.dense = true;
            host_only = true;
            exact_thermo_done = true;
        }
    }

    if (exact_thermo_done) {
        // The small-D exact fallback already filled R.thermo. Skip the
        // estimator, but fall through to the shared tail (free energy,
        // timing, lane metadata) like every other method.
    } else if (opts.method == ThermalOptions::Method::mTPQ) {
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            ed::thermal::MtpqOptions kopts;
            kopts.num_samples = opts.num_samples;
            kopts.random_seed = opts.random_seed;
            kopts.seed_transform = opts.seed_transform;
            if constexpr (!std::is_same_v<B, ed::matvec::CpuBackend>)
                kopts.batch_matvec = H.bind_cuda_multi();   // samples share each device H apply
            auto matvec = H.template bind<B>();

            // -------------------------------------------------------------
            // mTPQ. Every sample runs psi_k = (L - H)^k psi_0 / ||.|| and records E_k and the
            // growth factors; the canonical estimator (tpq_thermo.h) turns them into ln Z, E
            // and C at every requested beta at once. Recipe:
            //   1. spectral bounds (E_min, E_max) from a short Lanczos;
            //   2. L just above E_max (the series' terms peak near j* = beta (L - E_min), so
            //      L - E_max is pure cost);
            //   3. steps so the series converges at the coldest beta: j* + 8 sqrt(j*) terms;
            //      a target the trajectory cannot reach is refused, never clamped.
            // -------------------------------------------------------------
            // The coldest requested point: opts.betas comes from the caller or,
            // when empty, from the temp_min/temp_max grid built above.
            double beta_max = 0.0;
            for (double b : opts.betas) beta_max = std::max(beta_max, b);
            if (!(beta_max > 0.0)) beta_max = 100.0;

            double e_min_est = 0.0, e_max_est = 0.0;
            bool have_bounds = false;
            if (std::isfinite(opts.e_min_override)
                && std::isfinite(opts.e_max_override)
                && opts.e_max_override >= opts.e_min_override) {
                // Caller-supplied bounds (e.g. estimated once on the
                // largest sector and reused across the Sz loop).
                e_min_est = opts.e_min_override;
                e_max_est = opts.e_max_override;
                have_bounds = true;
            } else if constexpr (std::is_same_v<B, ed::matvec::CpuBackend>) {
                // CPU lane: run the shared Lanczos spectral-bound
                // estimator on a host-pointer wrapper of the matvec.
                std::function<void(const Complex*, Complex*, int)> legacy_H =
                    [&matvec](const std::complex<double>* in,
                              std::complex<double>* out, int n) {
                        matvec(in, out, static_cast<std::size_t>(n));
                    };
                const std::uint64_t bdim = H.geometry().local_dim;
                std::mt19937 gen(opts.random_seed
                                     ? static_cast<unsigned>(opts.random_seed)
                                     : 0x9E3779B9u);
                try {
                    const int kry = static_cast<int>(
                        std::min<std::uint64_t>(
                            60, std::max<std::uint64_t>(bdim, 1)));
                    ::estimate_spectral_bounds(
                        legacy_H, bdim, kry, /*tol=*/1e-10,
                        gen, e_min_est, e_max_est);
                    have_bounds = std::isfinite(e_min_est)
                                && std::isfinite(e_max_est)
                                && e_max_est >= e_min_est;
                } catch (...) {
                    have_bounds = false;
                }
            } else {
                // Device lanes: the same estimate from a short Lanczos run on this backend
                // (vectors stay device-resident); the extreme Ritz values of 60 steps.
                auto& be = *backend_uptr;
                const std::uint64_t bdim = H.geometry().local_dim;
                std::vector<Complex> seed_host(bdim);
                std::mt19937_64 gen(opts.random_seed ? opts.random_seed : 0x9E3779B97F4A7C15ULL);
                std::normal_distribution<double> nd(0.0, 1.0);
                for (auto& z : seed_host) z = Complex(nd(gen), nd(gen));
                auto seed = be.make_zero_vector(bdim);
                be.copy_from_host(seed_host.data(), seed.get(), bdim);
                ed::krylov::LanczosKernelOptions bo;
                bo.max_iter   = static_cast<std::size_t>(std::min<std::uint64_t>(60, std::max<std::uint64_t>(bdim, 1)));
                bo.reorth     = ed::krylov::ReorthPolicy::None;
                bo.keep_basis = false;
                bo.dim_cap    = bdim;
                try {
                    const auto lk = ed::krylov::lanczos_kernel(be, matvec, bdim, seed.get(), bo);
                    std::vector<double> d = lk.alpha, e;
                    for (std::size_t i = 1; i < lk.alpha.size(); ++i) e.push_back(lk.beta[i]);
                    e.resize(std::max<std::size_t>(d.size(), 1));
                    if (!d.empty() && LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(d.size()),
                                                    d.data(), e.data(), nullptr, 1) == 0) {
                        e_min_est = *std::min_element(d.begin(), d.end());
                        e_max_est = *std::max_element(d.begin(), d.end());
                        have_bounds = std::isfinite(e_min_est) && std::isfinite(e_max_est)
                                    && e_max_est >= e_min_est;
                    }
                } catch (...) {
                    have_bounds = false;
                }
            }

            if (!have_bounds)
                throw ed::ConvergenceError("mTPQ: the spectral bounds of the block could not be estimated");
            // L just above the spectrum: the series' terms peak near j* = beta (L - E_min), so a
            // larger L only costs steps. The margin covers the Lanczos estimate of E_max, a
            // lower bound on it.
            const double W = e_max_est - e_min_est;
            const double L = (opts.energy_shift > 0.0)
                ? opts.energy_shift
                : e_max_est + std::max({0.05 * W, 1e-6 * std::max(1.0, std::abs(e_max_est)), 1e-9});
            kopts.large_value = L;

            // Steps: sized so the canonical series converges at the coldest requested beta
            // (krylov_dim == 0), or the caller's count exactly.
            constexpr std::size_t MTPQ_HARD_CAP = 200000;
            const bool auto_steps = opts.krylov_dim == 0;
            std::size_t steps = auto_steps ? ed::thermal::mtpq_steps_for(beta_max, L, e_min_est)
                                           : opts.krylov_dim;
            if (steps > MTPQ_HARD_CAP)
                throw ed::ResourceLimit("mTPQ: T_min = " + std::to_string(1.0 / beta_max) + " needs "
                                        + std::to_string(steps) + " steps per sample (cap "
                                        + std::to_string(MTPQ_HARD_CAP) + "); ask for a warmer T_min");
            for (int attempt = 0;; ++attempt) {
                kopts.max_iter = std::max<std::size_t>(steps, 1);
                ed::thermal::MtpqResult kres = ed::thermal::mtpq_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
                R.ground_state_energy = kres.energies.empty()
                    ? 0.0 : *std::min_element(kres.energies.begin(), kres.energies.end());
                if (R.thermo.temperatures.empty()) break;
                auto mt = ed::thermal::mtpq_canonical_thermo(
                    kres.sample_energies, kres.sample_log_norms, L, R.thermo.temperatures,
                    static_cast<double>(H.geometry().global_dim));
                if (mt.unconverged.empty()) {
                    R.thermo = std::move(mt.thermo);
                    break;
                }
                // Too cold for the trajectory. An auto-sized run had underestimated the
                // spectral range: run once more with twice the steps. Never clamp.
                double T_reached = std::numeric_limits<double>::infinity();
                for (std::size_t t = 0; t < R.thermo.temperatures.size(); ++t)
                    if (std::find(mt.unconverged.begin(), mt.unconverged.end(), t) == mt.unconverged.end())
                        T_reached = std::min(T_reached, R.thermo.temperatures[t]);
                if (auto_steps && attempt == 0 && 2 * steps <= MTPQ_HARD_CAP) {
                    steps *= 2;
                    continue;
                }
                throw ed::ConvergenceError(
                    "mTPQ: " + std::to_string(steps) + " steps reach T = "
                    + (std::isfinite(T_reached) ? std::to_string(T_reached) : std::string("none of the targets"))
                    + " but the grid asks for T = " + std::to_string(1.0 / beta_max)
                    + (auto_steps ? std::string("") : std::string("; raise krylov or leave it unset")));
            }
        }, variant);
    } else if (opts.method == ThermalOptions::Method::FTLM) {
        // The FTLM kernel facade dispatches on Backend type internally
        // (see ftlm_kernel.h). Both CpuBackend and CudaBackend are
        // supported.
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            constexpr bool is_cpu =
                std::is_same_v<B, ed::matvec::CpuBackend>;
#ifdef WITH_CUDA
            constexpr bool is_cuda =
                std::is_same_v<B, ed::matvec::CudaBackend>;
#else
            constexpr bool is_cuda = false;
#endif
            if constexpr (!(is_cpu || is_cuda)) {
                throw std::runtime_error(
                    "ed::thermal: FTLM requires a CpuBackend or "
                    "CudaBackend; distributed backends are not yet "
                    "wired. Pin BackendConstraints to route through "
                    "the CPU/CUDA lanes.");
            } else {
                ed::thermal::FtlmOptions kopts;
                kopts.num_samples = opts.num_samples;
                kopts.krylov_dim  = opts.krylov_dim ? opts.krylov_dim : 100;
                kopts.betas       = opts.betas;
                kopts.random_seed = opts.random_seed;
                kopts.seed_transform = opts.seed_transform;
                for (const auto& O : opts.observables) {
                    if (!std::is_same_v<B, ed::matvec::CpuBackend> && !O->geometry().supports_device_matvec)
                        throw std::invalid_argument("ed::thermal: an observable has no device kernel for the selected GPU lane");
                    kopts.observables.push_back(O->template bind<B>());
                }
                if constexpr (!std::is_same_v<B, ed::matvec::CpuBackend>)
                    kopts.batch_matvec = H.bind_cuda_multi();   // samples share each device H apply
                auto matvec = H.template bind<B>();
                auto kres = ed::thermal::ftlm_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
                R.thermo.energy = std::move(kres.energy);
                R.thermo.specific_heat = std::move(kres.heat_capacity);
                R.thermo.entropy = std::move(kres.entropy);
                R.observables = std::move(kres.observables);
            }
            R.ground_state_energy = R.thermo.energy.empty()
                ? 0.0
                : *std::min_element(R.thermo.energy.begin(), R.thermo.energy.end());
        }, variant);
    } else if (opts.method == ThermalOptions::Method::OFTLM) {
        // OFTLM (Morita-Tohyama): FTLM + N_V exact low-lying states + random
        // vectors orthogonalized against them. CPU-only lane -- consumes the
        // host term-matvec directly (see oftlm_kernel.h).
        auto host_mv = H.bind_cpu();
        auto apply_H = [&host_mv](const Complex* in, Complex* out, int n) {
            host_mv(in, out, static_cast<std::size_t>(n));
        };
        ed::thermal::OftlmOptions kopts;
        kopts.num_samples = opts.num_samples;
        kopts.krylov_dim  = opts.krylov_dim ? opts.krylov_dim : 100;
        kopts.num_exact   = opts.num_exact;
        kopts.betas       = opts.betas;
        kopts.random_seed = opts.random_seed;
        auto kres = ed::thermal::oftlm_cpu(
            apply_H, H.geometry().global_dim, kopts);
        host_only = true;
        R.thermo.energy        = std::move(kres.energy);
        R.thermo.specific_heat = std::move(kres.heat_capacity);
        R.thermo.entropy       = std::move(kres.entropy);
        R.ground_state_energy = R.thermo.energy.empty()
            ? 0.0
            : *std::min_element(R.thermo.energy.begin(), R.thermo.energy.end());
    } else {
        throw std::runtime_error(
            "ed::thermal: unknown ThermalOptions::Method enumerator.");
    }

    // Populate ``R.thermo.free_energy = E - T * S`` post-hoc so downstream
    // consumers see the full thermodynamic quintet (T, E, Cv, S, F).
    // The FTLM / OFTLM kernel facades return E/Cv/S only and do not expose
    // ln Z; E - T S equals F = -T ln Z once normalised.
    if (R.thermo.free_energy.empty()
        && !R.thermo.energy.empty()
        && R.thermo.energy.size() == R.thermo.entropy.size()
        && R.thermo.energy.size() == R.thermo.temperatures.size()) {
        R.thermo.free_energy.reserve(R.thermo.energy.size());
        for (std::size_t i = 0; i < R.thermo.energy.size(); ++i) {
            R.thermo.free_energy.push_back(
                R.thermo.energy[i]
                - R.thermo.temperatures[i] * R.thermo.entropy[i]);
        }
    }

    // Pull the lane label from the actual ``BackendVariant``
    // ``select_backend`` returned, NOT the host operator's memory_space:
    // a host-resident operator that wires a GPU matvec through
    // ``bind_cuda()`` reports ``Host`` memory_space, but
    // ``select_backend`` picks ``CudaBackend`` when ``allow_gpu=true``
    // and ``supports_device_matvec=true`` -- except where the work ran on the
    // host anyway (the exact fallback, OFTLM).
    R.backend.lane = host_only ? "cpu" : ed::lane_label_from_variant(variant);
    const auto t1 = std::chrono::steady_clock::now();
    R.backend.wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    return R;
}

}  // namespace ed::workflows

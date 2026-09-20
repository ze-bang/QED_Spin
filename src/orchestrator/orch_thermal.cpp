// =============================================================================
// src/orchestrator/orch_thermal.cpp -- ed::workflows::thermal and its lanes
// (exact-small eigenspectrum fallback, mTPQ sampling, FTLM / LTLM / KpmDos,
// the all-Sz sweep).
// Part of the workflow orchestrator; see orchestrator_internal.h for the
// file map.
// =============================================================================

#include "orchestrator_internal.h"

namespace ed::workflows {

using namespace orch_detail;

namespace {

// ---------------------------------------------------------------------------
// Exact canonical thermodynamics from a complete eigenspectrum.
//
// Used as a small-sector fallback for mTPQ: when the Hilbert-space
// dimension is small (D <= SMALL_THERMAL_DIM), the stochastic mTPQ
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
constexpr std::uint64_t SMALL_THERMAL_DIM = 512;

// Audit 2026-07-31: forwards to the single canonical implementation in
// ed/symmetry/canonical_thermo.h (this used to be one of three
// byte-equivalent copies; the guards live there now).
static ThermodynamicData compute_canonical_thermo_from_eigs(
    const std::vector<double>& eigs,
    const std::vector<double>& temperatures)
{
    return ed::symmetry::canonical_thermo_from_eigs(eigs, temperatures);
}

}  // namespace

ThermalResult thermal(const LinearOperator& H, ThermalOptions opts) {
    require_hermitian_input(H, "ed::thermal");
    // All lanes are wired: mTPQ dispatches through the unified
    // `tpq_kernel` via the Phase 2.4 facades; FTLM / LTLM / KpmDos
    // dispatch through their own `*_kernel<Backend>` templates (CPU
    // implementations today, GPU when the kernels migrate). The variant
    // visit at each lane keeps the dispatch backend-agnostic.
    using Complex = std::complex<double>;
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(H.geometry().local_dim));
    ed::parallel::pin_omp_threads_once();

    BackendVariant variant;
#ifdef WITH_CUDA
    // fp32 single-GPU mTPQ fits one H100 (2 x complex<float> = 68.7 GB at
    // 2^32). The default gpu_mem_fits() estimate uses complex<double> x fudge 8
    // (= 549 GB) and would REJECT the GPU, falling to the CPU lane -- whose
    // 2^32 spectral-bound Lanczos auto-tune then OOM-kills the host. Force the
    // GPU lane here; the fp32 driver manages its own (fitting) device memory.
    if (opts.mtpq_fp32
        && opts.method == ThermalOptions::Method::mTPQ
        && H.supports_cuda_f32()
        && ed::have_cuda()) {
        variant = BackendVariant{std::make_unique<ed::matvec::CudaBackend>()};
    } else
#endif
    {
        variant = select_backend(H.geometry(), opts.backend);
    }

    // COMPLETION GUARANTEE (thermal lane). The operator's basis is already built,
    // so the binding constraint is the kernel WORKING SET: FTLM / LTLM keep a
    // krylov_dim window of length-N vectors; TPQ / KPM a handful. Plan it and
    // refuse cleanly if it would not fit, before allocating those vectors. The
    // small-sector exact fallback (D <= SMALL_THERMAL_DIM) is tiny and always
    // passes. allow_infeasible (force) opts out.
    // Leaf memory guard (planner feasibility pre-flight removed): throw cleanly
    // before the Krylov-basis allocation rather than OOM-crash. H.global_dim()
    // is the per-call working dim (symmetry iterates sectors a level up, so this
    // is the sector dim there). FTLM and LTLM both keep one sample's Krylov
    // basis at a time (LTLM routes through the FTLM kernel); TPQ/KPM are O(1)
    // state vectors.
    {
        const std::uint64_t D = H.global_dim();
        std::uint64_t vecs;
        std::uint64_t elem = 16ull;  // complex<double>
        bool fp32_lane = false;
#ifdef WITH_CUDA
        // fp32 single-GPU mTPQ lane: the lean driver keeps just two
        // complex<float> device vectors (psi + w), so the working set is
        // D * 2 * 8 bytes (68.7 GB at 2^32), a quarter of the double estimate
        // -- otherwise this guard's 8-vector complex<double> estimate (549 GB
        // at 2^32) would refuse the run before the driver ever allocates.
        fp32_lane = opts.mtpq_fp32
                 && opts.method == ThermalOptions::Method::mTPQ
                 && H.supports_cuda_f32();
#endif
        if (fp32_lane) {
            vecs = 2;
            elem = 8ull;  // complex<float>
        } else {
            switch (opts.method) {
                // LTLM thermodynamics dispatches through ftlm_kernel (Jul 2026,
                // 654ea06) -- it no longer stores a GS + excitation basis pair,
                // so it costs exactly what FTLM costs. The old 2*krylov estimate
                // outlived the kernel it modelled and over-charged LTLM ~2x
                // (400 vs 204 vectors at the kLtlmKrylovDim=200 default), which
                // can refuse a run that fits comfortably.
                case ThermalOptions::Method::LTLM:
                case ThermalOptions::Method::FTLM:
                    vecs = std::max<std::size_t>(opts.krylov_dim, 4) + 4; break;
                case ThermalOptions::Method::OFTLM:
                    // per-sample Krylov basis + the exact-eigenpair Lanczos basis
                    vecs = std::max<std::size_t>(opts.krylov_dim, 4)
                         + 2 * opts.num_exact + 34; break;
                default:  // mTPQ / KpmDos: a handful of state vectors
                    vecs = 8; break;
            }
        }
        ed::core::guard_working_set(D * vecs * elem, "ed::thermal");
    }

    // Surface unification follow-up (May 2026): when the caller does
    // not supply an explicit ``opts.betas`` grid, construct one from
    // the temperature-scan knobs (``temp_min``, ``temp_max``,
    // ``num_temp_bins``) and -- crucially -- mirror the resulting
    // temperature axis into ``R.thermo.temperatures`` so downstream
    // Python / CLI consumers can read the scan back without
    // recomputing it from ``opts.*``. Mirrors the legacy
    // ``finite_temperature_lanczos`` contract that every call site
    // relied on.
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
    // typicality. For small sectors (D <= SMALL_THERMAL_DIM), the per-sample
    // variance is too high for the dE tolerance even with 20+ samples. The
    // original symptom was the sz_spatial mTPQ failure: within an n_up block,
    // translation k-sectors have D ≈ 1–9 for N=8, giving a statistical error
    // of ~0.12 with 20 samples (vs the 0.08 tolerance).
    //
    // Jul 2026: the gate used to require mTPQ specifically, so FTLM/LTLM kept
    // sampling in a regime where the exact solve is both free and machine
    // precise -- measured at dim=64 (N=6 ring): mTPQ 1.4e-15 (this fallback)
    // vs FTLM/LTLM 2.3e-02 (sampling), i.e. 13 orders for microseconds of
    // eigensolve. The deliverable of FTLM / LTLM / OFTLM / mTPQ is identical
    // here -- canonical E(T)/C(T)/S(T) -- so all four take the exact route.
    // KpmDos is deliberately EXCLUDED: its deliverable includes the Chebyshev
    // density of states, which this path does not produce (same rationale as
    // the probe_betas carve-out below).
    //
    // For any D <= SMALL_THERMAL_DIM, diagonalise exactly and compute the
    // canonical partition function directly. The resulting ThermodynamicData
    // uses the same absolute free-energy convention (F includes ln(D) at
    // high T) as compute_tpq_thermo_from_trajectories, so it plugs in
    // correctly to combine_sector_thermodynamics for Sz/spatial recombination.
    // -----------------------------------------------------------------------
    const bool is_sampling_thermo_method =
        opts.method == ThermalOptions::Method::mTPQ  ||
        opts.method == ThermalOptions::Method::FTLM  ||
        opts.method == ThermalOptions::Method::LTLM  ||
        opts.method == ThermalOptions::Method::OFTLM;
    // NOTE: this must NOT return early. Everything below the method dispatch --
    // the universal-save persistence finalizer above all -- has to run for the
    // exact result exactly as it does for a sampled one, or the fallback
    // silently strips the caller's output_dir contract (R.hdf5_path empty ->
    // Python's sector_hdf5_paths empty). That is a bug this file has already
    // shipped once, on the solve verb (see apply_solve_save_finalizer's note).
    // So: fill R.thermo, then flag the dispatch chain to stand down.
    bool exact_thermo_done = false;
    if (is_sampling_thermo_method &&
        exact_small_thermal_enabled() &&
        H.geometry().global_dim > 0 &&
        H.geometry().global_dim <= SMALL_THERMAL_DIM &&
        !R.thermo.temperatures.empty() &&
        // Stage 12f: a seed transform restricts the stochastic trace to a
        // SUBSPACE (e.g. one spin tower). The exact fallback diagonalises
        // the whole block and would silently ignore the restriction --
        // stand down and let the sampling kernel honour the projection.
        // (Callers wanting exact per-tower thermo use the differencing
        // route in workflows_thermal_su2_tower instead.)
        !opts.seed_transform &&
        // When the caller requested TPQ state snapshots (probe_betas), the exact
        // fallback cannot produce them -- run the real TPQ trajectory instead
        // (accepting the small-sector variance the user implicitly opted into).
        opts.probe_betas.empty()) {
        const std::uint64_t D = H.geometry().global_dim;
        std::vector<double> eigs;
        full_diagonalization(H, D, D, eigs, /*dir=*/"", /*compute_eigenvectors=*/false);
        if (!eigs.empty()) {
            R.thermo = compute_canonical_thermo_from_eigs(
                eigs, R.thermo.temperatures);
            R.ground_state_energy = eigs.front();
            // The exact fallback ran on the selected backend lane; label it like
            // the normal return path.
            R.backend.lane = ed::lane_label_from_variant(variant);
            exact_thermo_done = true;
        }
    }

    if (exact_thermo_done) {
        // The small-D exact fallback already filled R.thermo. Skip the
        // estimator, but fall through to the shared tail (persistence
        // finalizer, timing, lane metadata) like every other method.
    } else if (opts.method == ThermalOptions::Method::mTPQ) {
        std::visit([&](auto& backend_uptr) {
            using BPtr = std::decay_t<decltype(backend_uptr)>;
            using B = typename BPtr::element_type;
            ed::thermal::MtpqOptions kopts;
            kopts.num_samples = opts.num_samples;
            kopts.random_seed = opts.random_seed;
            kopts.output_dir  = opts.output_dir;
            kopts.probe_betas = opts.probe_betas;
            kopts.seed_transform = opts.seed_transform;  // Stage 12f
            auto matvec = H.template bind<B>();

            // -------------------------------------------------------------
            // SOTA mTPQ auto-tune (June 2026).
            //
            // The microcanonical iteration |psi_{k+1}> = (L*I - H)|psi_k>
            // advances the effective inverse temperature by
            //   Delta_beta ~ 2 / (L - <H>),     beta_k = 2 k / (L - E_k).
            // So L sets BOTH (a) the high-temperature resolution (small
            // Delta_beta needs large L) and -- together with the step
            // count -- (b) the coldest temperature reached.
            //
            // The previous heuristic used ``log2(global_dim)`` as a
            // stand-in for the spectral ceiling E_max and coupled L to
            // ``temp_min`` via ``L = 2*max_iter/beta_target + proxy``.
            // That is doubly wrong: the dim proxy has nothing to do with
            // the actual band edge, and lowering ``temp_min`` to reach
            // colder T *shrank* L, coarsening Delta_beta and biasing the
            // specific heat / entropy (and risking L < E_max, which makes
            // (L - H) indefinite and corrupts the trajectory).
            //
            // SOTA recipe:
            //   1. Measure the true spectral bounds (E_min, E_max) with a
            //      short Lanczos (reuse the KPM estimator).
            //   2. Pick L from ONE resolution knob: Delta_beta_target.
            //        L_res  = <H>_inf + 2 / Delta_beta_target
            //        L_stab = E_max + buffer   (strictly above the ceiling)
            //        L      = max(L_res, L_stab)
            //   3. Size the iteration count INDEPENDENTLY so the trajectory
            //      still brackets beta_max = 1/temp_min:
            //        steps ~ beta_max * (L - E_min) / 2.
            // Resolution and cold-reach are thus decoupled; matvecs are
            // cheap so over-provisioning steps is affordable.
            // -------------------------------------------------------------
            const double dbeta_target = 0.02;  // internal quality knob
            const double beta_max = (opts.temp_min > 0.0)
                ? 1.0 / opts.temp_min : 100.0;

            double e_min_est = 0.0, e_max_est = 0.0;
            bool have_bounds = false;
            if (std::isfinite(opts.e_min_override)
                && std::isfinite(opts.e_max_override)
                && opts.e_max_override > opts.e_min_override) {
                // Caller-supplied bounds (e.g. estimated once on the
                // largest sector and reused across the Sz loop).
                e_min_est = opts.e_min_override;
                e_max_est = opts.e_max_override;
                have_bounds = true;
            } else if constexpr (std::is_same_v<B, ed::matvec::CpuBackend>) {
                // CPU lane: run the shared Lanczos spectral-bound
                // estimator on a host-pointer wrapper of the matvec.
                ed::kpm_dos::MatVec legacy_H =
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
                    // Extreme Ritz values converge first and are robust
                    // without reorthogonalization; skip it (the estimator
                    // does not retain the Krylov basis anyway) to avoid a
                    // spurious "reorth skipped" warning and wasted work.
                    ed::kpm_dos::estimate_spectral_bounds(
                        legacy_H, bdim, kry, /*full_reorth=*/false,
                        /*reorth_freq=*/0, /*tol=*/1e-10,
                        gen, e_min_est, e_max_est);
                    have_bounds = std::isfinite(e_min_est)
                                && std::isfinite(e_max_est)
                                && e_max_est > e_min_est;
                } catch (...) {
                    have_bounds = false;
                }
            }

            double L_auto;
            if (have_bounds) {
                const double W      = e_max_est - e_min_est;
                const double e_high = 0.5 * (e_min_est + e_max_est);
                const double L_res  = e_high + 2.0 / std::max(dbeta_target, 1e-6);
                const double L_stab = e_max_est + std::max(0.05 * W, 1e-9);
                L_auto = std::max(L_res, L_stab);
            } else {
                // Non-CPU backend or failed estimate: resolution-driven
                // floor plus a conservative dim-based pad (never below
                // the historical behaviour for high-T runs).
                const double bandwidth_proxy = std::log2(
                    static_cast<double>(std::max<std::uint64_t>(
                        H.geometry().global_dim, 2)));
                L_auto = 2.0 / std::max(dbeta_target, 1e-6) + bandwidth_proxy;
            }

            // Expert override (HPhi-style ``LargeValue``): a finite,
            // positive ``energy_shift`` pins L and skips the auto-tune.
            kopts.large_value = (opts.energy_shift > 0.0)
                ? opts.energy_shift : L_auto;

            // Iteration budget contract:
            //   * krylov_dim == 0  -> AUTO: size the step count so the
            //     trajectory brackets beta_max = 1/temp_min, using
            //     steps ~ beta_max*(L - E_min)/2 (capped for safety).
            //   * krylov_dim  > 0  -> RESPECT the caller's value exactly
            //     (expert override / explicit ``max_iterations=...``).
            // This keeps the default surface clean (no knob needed -- the
            // auto path guarantees the requested coldest T is reached)
            // while never silently overriding an explicit request.
            const double e_low = have_bounds ? e_min_est : 0.0;
            const std::size_t reach_iters = static_cast<std::size_t>(
                std::ceil(beta_max
                          * (kopts.large_value - e_low) / 2.0)) + 16;
            constexpr std::size_t MTPQ_HARD_CAP = 200000;
            kopts.max_iter = (opts.krylov_dim == 0)
                ? std::max<std::size_t>(std::min(reach_iters, MTPQ_HARD_CAP), 1)
                : opts.krylov_dim;

            ed::thermal::MtpqResult kres;
#ifdef WITH_CUDA
            // fp32 single-GPU lane: half-footprint state vectors let the full
            // 2^32 Hilbert space run mTPQ on one 80 GB H100. Reuses the L /
            // max_iter auto-tune computed above (kopts); the driver manages its
            // own device memory (bypasses the double CudaBackend vectors).
            if (ed::env::flag("ED_MTPQ_VERBOSE", false)) {
                std::fprintf(stderr,
                    "[mtpq-lane] mtpq_fp32=%d supports_cuda_f32=%d -> %s\n",
                    (int)opts.mtpq_fp32, (int)H.supports_cuda_f32(),
                    (opts.mtpq_fp32 && H.supports_cuda_f32()) ? "FP32-GPU"
                                                              : "double");
            }
            if (opts.mtpq_fp32 && H.supports_cuda_f32()) {
                kres = ed::thermal::mtpq_f32(H, kopts);
            } else
#endif
            {
                kres = ed::thermal::mtpq_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
            }
            R.ground_state_energy = kres.energies.empty()
                ? 0.0 : *std::min_element(kres.energies.begin(),
                                          kres.energies.end());
            // SOTA: aggregate per-sample (beta_k, E_k, var_k) trajectories
            // into ThermodynamicData on the requested temperature grid.
            // Closes the gap where mTPQ via qed.thermal raised
            // ``RuntimeError: solver returned no thermodynamic data``.
            if (!R.thermo.temperatures.empty()) {
                // Audit 2026-09: say so when the trajectory never reached the
                // coldest requested temperature -- the aggregator otherwise
                // extrapolates silently (measured: E(T=0.2) off by 12% at
                // N = 20 with a 100-step cap).
                double beta_reached = 0.0;
                for (const auto& tr : kres.sample_inv_temps)
                    for (double b : tr) beta_reached = std::max(beta_reached, b);
                double beta_wanted = 0.0;
                for (double T : R.thermo.temperatures)
                    if (T > 0.0) beta_wanted = std::max(beta_wanted, 1.0 / T);
                if (beta_wanted > 0.0 && beta_reached < 0.999 * beta_wanted) {
                    std::cerr << "[mTPQ] WARNING: the trajectory reached beta = "
                              << beta_reached << " but the temperature grid asks for beta = "
                              << beta_wanted << " (T_min = " << 1.0 / beta_wanted
                              << "); results below T = " << 1.0 / std::max(beta_reached, 1e-300)
                              << " are extrapolated. Raise max_iterations / leave krylov_dim "
                                 "unset so the step count is sized automatically." << std::endl;
                    R.backend.notes.emplace_back(
                        "mtpq_beta_reached", std::to_string(beta_reached) + " < wanted "
                        + std::to_string(beta_wanted));
                }
                ThermodynamicData td = ed::thermal::compute_tpq_thermo_from_trajectories(
                    kres.sample_inv_temps, kres.sample_energies,
                    kres.sample_variances, R.thermo.temperatures,
                    static_cast<double>(H.geometry().global_dim));
                if (!td.energy.empty()) {
                    // Mirror the LTLM/FTLM contract: the caller's T grid
                    // is authoritative -- overwrite R.thermo with the
                    // aggregator's output (which uses our T grid).
                    R.thermo = std::move(td);
                }
            }
            // Pillar 1 (May 2026): lift the per-sample TPQ trajectory
            // + state-vector snapshots into the outer ThermalResult so
            // the uniform finalizer (below) can persist them.
            R.tpq_sample_betas     = std::move(kres.sample_inv_temps);
            R.tpq_sample_energies  = std::move(kres.sample_energies);
            R.tpq_sample_variances = std::move(kres.sample_variances);
            for (std::size_t s = 0; s < kres.state_snapshots.size(); ++s) {
                for (std::size_t p = 0; p < kres.state_snapshots[s].size(); ++p) {
                    auto& psi = kres.state_snapshots[s][p];
                    if (psi.empty()) continue;
                    TpqStateSnapshot snap;
                    snap.sample_index   = s;
                    snap.requested_beta = (p < opts.probe_betas.size())
                                              ? opts.probe_betas[p]
                                              : 0.0;
                    snap.effective_beta = kres.state_snapshot_betas[s][p];
                    snap.psi            = std::move(psi);
                    R.tpq_state_snapshots.push_back(std::move(snap));
                }
            }
        }, variant);
    } else if (opts.method == ThermalOptions::Method::FTLM) {
        // Phase E of the "Close CPU/GPU Gaps" plan (May 2026): the
        // FTLM kernel facade now dispatches on Backend type internally
        // (see ftlm_kernel.h, mirroring LTLM at the block below).
        // Both CpuBackend and CudaBackend are supported; MpiBackend /
        // MpiCudaBackend are explicitly rejected by the kernel until
        // cross-rank Lanczos post-processing is wired.
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
                kopts.output_dir  = opts.output_dir;
                kopts.seed_transform = opts.seed_transform;  // Stage 12f
                auto matvec = H.template bind<B>();
                auto kres = ed::thermal::ftlm_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
                R.thermo.energy = std::move(kres.energy);
                R.thermo.specific_heat = std::move(kres.heat_capacity);
                R.thermo.entropy = std::move(kres.entropy);
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
        kopts.output_dir  = opts.output_dir;
        auto kres = ed::thermal::oftlm_cpu(
            apply_H, H.geometry().global_dim, kopts);
        R.thermo.energy        = std::move(kres.energy);
        R.thermo.specific_heat = std::move(kres.heat_capacity);
        R.thermo.entropy       = std::move(kres.entropy);
        R.ground_state_energy = R.thermo.energy.empty()
            ? 0.0
            : *std::min_element(R.thermo.energy.begin(), R.thermo.energy.end());
    } else if (opts.method == ThermalOptions::Method::LTLM) {
        // Phase E2 of the "Backend x Symmetries x Workflows" plan
        // (May 2026): the LTLM kernel now dispatches on Backend type
        // internally (see ltlm_kernel.h). Both CpuBackend and
        // CudaBackend are supported; MpiBackend / MpiCudaBackend are
        // explicitly rejected by the kernel until cross-rank Lanczos
        // post-processing is wired.
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
                    "ed::thermal: LTLM requires a CpuBackend or "
                    "CudaBackend; distributed backends are not yet "
                    "wired. Pin BackendConstraints to route through "
                    "the CPU/CUDA lanes.");
            } else {
                // Jul 2026: LTLM thermodynamics == FTLM. Both LTLM kernels
                // (the CPU low_temperature_lanczos and the backend
                // ltlm_kernel_via_backend) seeded a SECOND Lanczos from the
                // ground state and summed sum_n |<0|psi_n>|^2 e^{-bE_n} --
                // the GS-LOCAL density of states, i.e.
                // <0|He^{-bH}|0>/<0|e^{-bH}|0>, NOT the thermal trace. It
                // stayed pinned near E0 at every T (E(0.69)=-7.35 vs exact
                // -6.33). For a FUNCTION OF H (all thermodynamics here) the
                // LTLM symmetric estimator reduces EXACTLY to the FTLM
                // trace, so route through the verified FTLM kernel; the two
                // differ only for observables that do not commute with H,
                // which this thermodynamics path never computes.
                ed::thermal::FtlmOptions kopts;
                kopts.num_samples = opts.num_samples;
                kopts.krylov_dim  = opts.krylov_dim ? opts.krylov_dim : 100;
                kopts.betas       = opts.betas;
                kopts.random_seed = opts.random_seed;
                kopts.output_dir  = opts.output_dir;
                kopts.seed_transform = opts.seed_transform;  // Stage 12f
                auto matvec = H.template bind<B>();
                auto kres = ed::thermal::ftlm_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
                R.thermo.energy = std::move(kres.energy);
                R.thermo.specific_heat = std::move(kres.heat_capacity);
                R.thermo.entropy = std::move(kres.entropy);
                R.ground_state_energy = R.thermo.energy.empty() ? 0.0
                    : *std::min_element(R.thermo.energy.begin(),
                                        R.thermo.energy.end());
            }
        }, variant);
    } else if (opts.method == ThermalOptions::Method::KpmDos) {
        // Phase E1 of the "Backend x Symmetries x Workflows" plan
        // (May 2026): the KPM-DOS kernel now dispatches on Backend
        // type internally (see kpm_dos_kernel.h). Both CpuBackend and
        // CudaBackend are supported; MpiBackend / MpiCudaBackend are
        // explicitly rejected by the kernel until the cross-rank
        // Hutchinson reduction is wired.
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
                    "ed::thermal: KpmDos requires a CpuBackend or "
                    "CudaBackend; distributed backends are not yet "
                    "wired. Pin BackendConstraints to route through "
                    "the CPU/CUDA lanes.");
            } else {
                ed::thermal::KpmDosOptions kopts;
                kopts.betas = opts.betas;
                kopts.random_seed = opts.random_seed;
                // Wave B3 (May 2026): pass-through caller-supplied
                // spectral bounds. NaN means "estimate via Lanczos".
                kopts.e_min_override = opts.e_min_override;
                kopts.e_max_override = opts.e_max_override;
                // Closing-the-gap follow-up (May 2026): forward the
                // user's KPM knobs. ``0`` keeps the kernel default
                // so legacy CLI call sites with no knob set are
                // unaffected.
                if (opts.kpm_num_moments > 0) {
                    kopts.num_moments = opts.kpm_num_moments;
                }
                if (opts.kpm_num_random_vectors > 0) {
                    kopts.num_random_vectors = opts.kpm_num_random_vectors;
                }
                auto matvec = H.template bind<B>();
                auto kres = ed::thermal::kpm_dos_kernel<B>(
                    *backend_uptr, matvec, H.geometry().local_dim,
                    H.geometry().global_dim, kopts);
                R.thermo.energy = std::move(kres.energy);
                R.thermo.specific_heat = std::move(kres.specific_heat);
                R.thermo.entropy = std::move(kres.entropy);
                R.thermo.free_energy = std::move(kres.free_energy);
                R.ground_state_energy = kres.e_min_estimate;
                // Surface the raw DOS the method exists to produce (was
                // computed and discarded; only its thermodynamics escaped).
                R.dos_energies = std::move(kres.dos_grid_energies);
                R.dos_values   = std::move(kres.dos_grid_values);
            }
        }, variant);
    } else {
        throw std::runtime_error(
            "ed::thermal: unknown ThermalOptions::Method enumerator.");
    }

    // Surface unification follow-up (May 2026): populate
    // ``R.thermo.free_energy = E - T * S`` post-hoc so downstream
    // consumers see the full thermodynamic quintet (T, E, Cv, S, F).
    // The FTLM/LTLM/KpmDos kernel facades return E/Cv/S only; the
    // legacy ``finite_temperature_lanczos`` populated F from the
    // partition function (F = -T ln Z), which is mathematically
    // equivalent to E - T S once normalised. We use the latter form
    // here since the kernel does not expose ln Z.
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

    // -----------------------------------------------------------------
    // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026): uniform
    // thermal persistence finalizer. Mirrors the contract that lives in
    // ``ed::workflows::solve`` (l. 458-510): when ``opts.output_dir``
    // is set and the run is single-rank (the shared-file save is not
    // safe under MPI; per-rank files are written elsewhere), persist
    // the result to ``<output_dir>/ed_results.h5`` and surface the
    // resulting path via ``R.hdf5_path``.
    //
    // Method-conditional payload (user-confirmed policy):
    //   - mTPQ: the full per-sample (beta, E, var, step)
    //     trajectory (one row per kernel step, appended via
    //     ``HDF5IO::appendTPQThermodynamics``), plus state vectors at
    //     the betas closest to ``opts.probe_betas`` written via
    //     ``HDF5IO::saveTPQState``.
    //   - FTLM / LTLM / KPM_DOS: aggregated thermodynamic curves
    //     (``T, E, Cv, S, F``) only -- no state vectors.
    // -----------------------------------------------------------------
    // MPI-aware single-file emission (May 2026 follow-up): rank 0 owns
    // the aggregate thermo / TPQ-trajectory data (the kernels recombine
    // per-sample columns onto rank 0 before the orchestrator wraps the
    // result), so we run the finalizer there too. Slab-distributed TPQ
    // state vectors are *not* re-gathered here -- the per-rank
    // ``rank_<r>.h5`` files written by ``ed_distributed_main`` remain
    // the canonical location for those. The unified file therefore
    // ships:
    //   * mTPQ: per-sample trajectory rows (always on rank 0).
    //     The probe-beta state snapshots are written only when they are
    //     populated -- which is the serial case; in the distributed
    //     lane ``R.tpq_state_snapshots`` is empty on rank 0 and the
    //     loop is a no-op.
    //   * FTLM / LTLM / KPM_DOS: aggregated thermodynamic curves.
    if (!opts.output_dir.empty()
            && !HDF5IO::isDisabledOutputPath(opts.output_dir)
            && is_unified_writer(H.geometry())) {
        try {
            std::error_code ec;
            std::filesystem::create_directories(opts.output_dir, ec);
            const std::string h5_path =
                opts.output_dir + "/ed_results.h5";
            // Creates the file (or opens it) and lays down the
            // standard group skeleton (``/eigendata``, ``/tpq``,
            // ``/spectral_data`` ...).
            HDF5IO::createOrOpenFile(opts.output_dir);

            const bool is_tpq =
                (opts.method == ThermalOptions::Method::mTPQ
);
            if (is_tpq) {
                // (a) Per-sample trajectory rows.
                const std::size_t S =
                    std::min({R.tpq_sample_betas.size(),
                              R.tpq_sample_energies.size(),
                              R.tpq_sample_variances.size()});
                for (std::size_t s = 0; s < S; ++s) {
                    HDF5IO::ensureTPQSampleGroup(h5_path, s);
                    const auto& bs = R.tpq_sample_betas[s];
                    const auto& es = R.tpq_sample_energies[s];
                    const auto& vs = R.tpq_sample_variances[s];
                    const std::size_t K =
                        std::min({bs.size(), es.size(), vs.size()});
                    for (std::size_t k = 0; k < K; ++k) {
                        HDF5IO::TPQThermodynamicPoint pt;
                        pt.beta     = bs[k];
                        pt.energy   = es[k];
                        pt.variance = vs[k];
                        pt.doublon  = 0.0;
                        pt.step     = static_cast<std::uint64_t>(k);
                        HDF5IO::appendTPQThermodynamics(h5_path, s, pt);
                    }
                }
                // (b) State-vector snapshots at probe-betas.
                for (const auto& snap : R.tpq_state_snapshots) {
                    if (snap.psi.empty()) continue;
                    HDF5IO::ensureTPQSampleGroup(h5_path, snap.sample_index);
                    HDF5IO::saveTPQState(h5_path,
                                         snap.sample_index,
                                         snap.effective_beta,
                                         snap.psi,
                                         /*overwrite=*/true);
                }
            } else {
                // FTLM / LTLM / KPM-DOS: aggregated thermodynamic
                // curves. The kernel facades do not surface per-T
                // standard errors (those live on ``FTLMResults`` for
                // the legacy CLI path); the shared-file saver below
                // requires parallel arrays of equal length, so we
                // ship zero-valued error vectors of matching size.
                const std::size_t N = R.thermo.temperatures.size();
                const std::vector<double> zeros(N, 0.0);
                const char* label =
                    (opts.method == ThermalOptions::Method::FTLM)   ? "FTLM"
                  : (opts.method == ThermalOptions::Method::LTLM)   ? "LTLM"
                  : (opts.method == ThermalOptions::Method::KpmDos) ? "KPM_DOS"
                  : "thermal";
                HDF5IO::saveFTLMThermodynamics(
                    h5_path,
                    R.thermo.temperatures,
                    R.thermo.energy, zeros,
                    R.thermo.specific_heat, zeros,
                    R.thermo.entropy, zeros,
                    R.thermo.free_energy, zeros,
                    static_cast<std::uint64_t>(opts.num_samples),
                    std::string(label));
            }
            R.hdf5_path = h5_path;
        } catch (const std::exception& e) {
            // Non-fatal: skip persistence on I/O failure but surface
            // the cause via a backend note. The caller still gets the
            // in-memory ``R`` back; the empty ``R.hdf5_path`` flags
            // that no on-disk file was produced.
            std::cerr << "ed::thermal: persistence finalizer failed: "
                      << e.what() << std::endl;
        }
    }

    // Phase D (May 2026): truthful lane reporting -- pull the lane
    // label from the actual ``BackendVariant`` ``select_backend``
    // returned, NOT the host operator's memory_space. SectorView (and
    // every other host-resident operator that lazily wires a GPU
    // mirror through ``bind_cuda()``) reports ``Host`` memory_space
    // but ``select_backend`` picks ``CudaBackend`` when
    // ``allow_gpu=true`` and ``supports_device_matvec=true``. Reading
    // the variant directly is the only way the label can tell the
    // truth across all symmetry / non-symmetry workflows.
    R.backend.lane = ed::lane_label_from_variant(variant);
    const auto t1 = std::chrono::steady_clock::now();
    R.backend.wall_seconds =
        std::chrono::duration<double>(t1 - t0).count();
    return R;
}

}  // namespace ed::workflows

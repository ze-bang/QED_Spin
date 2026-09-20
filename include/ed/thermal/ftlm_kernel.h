#pragma once
// =============================================================================
// include/ed/thermal/ftlm_kernel.h
//
// FTLM (Finite-Temperature Lanczos Method) kernel ---
// ``template<Backend, MatvecFn>``. Same dual-backend pattern as
// ``ltlm_kernel<Backend>``:
//
//   * ``CpuBackend`` (WP10 C5) and ``CudaBackend`` (Phase E of the
//     "Close CPU/GPU Gaps" plan, May 2026): delegate to
//     ``detail::ftlm_kernel_via_backend``, a
//     fully Backend-templated body that reuses ``lanczos_kernel<Backend>``
//     (Phase 2 BLAS-1 facade) once per random sample. All BLAS-1 ops
//     run device-resident; cross-PCI traffic is limited to a host-seeded
//     random starting vector per sample and the small (M x M)
//     tridiagonal diagonalisation handled on the host with LAPACK.
//   * Distributed backends (MpiBackend / MpiCudaBackend): unsupported
//     today -- the body throws ``std::runtime_error`` so the caller can
//     pin ``BackendConstraints::allow_mpi = false`` and route through a
//     single-rank lane. Tracked under the MPI-parity follow-up.
//
// Algorithm:
//   * Draw ``R`` Gaussian random unit vectors ``|r>``.
//   * For each ``|r>`` run a length-``M`` Lanczos.
//   * Diagonalise the tridiagonal ``(alpha, beta)`` -> Ritz values + first-
//     component weights ``|<r | q_k>|^2``.
//   * Compute ``Z, <E>, Cv, S`` from the Ritz-value Lehmann representation
//     of the Krylov projection (``compute_ftlm_thermodynamics``).
//   * Average across samples using the Jensen-correct
//     ``average_ftlm_samples`` post-processor.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/parallel/thread_budget.h>  // auto_threads_for_dim + ThreadBudgetScope
#include <ed/solvers/ftlm.h>
#include <ed/solvers/lanczos.h>      // diagonalize_tridiagonal_ritz, generateGaussianRandomVector
#include <ed/thermal/sample_seed.h>

#ifdef WITH_CUDA
// Forward declaration so the ``if constexpr`` branch below can refer to
// CudaBackend without dragging the full ``cuda_backend.cuh`` into every
// consumer of this header. The orchestrator TU (which actually calls
// ``ftlm_kernel<CudaBackend>``) already pulls the full definition via
// ``select_backend.h``.
namespace ed { namespace matvec { class CudaBackend; } }
#endif

namespace ed::thermal {

using Complex = std::complex<double>;

struct FtlmOptions {
    std::size_t num_samples  = 40;
    std::size_t krylov_dim   = 100;
    std::vector<double> betas;           ///< inverse-temperature grid (positive)

    /// Optional exact temperature grid (WP10 C4). When non-empty it is
    /// used verbatim as the evaluation grid and reported as
    /// ``FtlmResult::temperatures``, bypassing the ``T = 1/beta``
    /// round trip, which can move a grid point by 1 ulp relative to a
    /// caller that built T directly (the legacy min/max/bins overload's
    /// ``exp`` grid). ``betas`` may then be left empty (it is filled with
    /// ``1/T``); if both are given they must have the same length.
    std::vector<double> temperatures;

    std::uint64_t random_seed = 0;       ///< 0 = nondeterministic (random_device)
    std::string output_dir;              ///< unused since WP10 C5 (no per-sample dumps)

    /// Knob parity with the legacy FTLMParameters (audit 2026-07-31):
    /// the CPU lane used to forward only krylov/samples/seed and let
    /// everything else silently take legacy defaults, which blocked the
    /// direct Python bindings from routing through this front door.
    /// Since WP10 C5 both lanes run the Backend-templated body, which
    /// honours ``full_reorthogonalization`` (FullCGS2 with a kept basis)
    /// and ignores the rest, exactly as the legacy CPU driver did for
    /// its results: ``max_iterations`` / ``tolerance`` /
    /// ``reorth_frequency`` were unused there since audit H5, and
    /// ``compute_error_bars`` only gated per-sample data and error bars
    /// that ``FtlmResult`` never carried. ``store_intermediate`` (the
    /// driver's per-sample HDF5 dump) is no longer honoured.
    std::uint64_t max_iterations           = 1000;
    double        tolerance                = 1e-10;
    /// Audit H5 (2026-09): stochastic-trace samples do not need a mutually
    /// orthogonal Krylov basis (ghost Ritz values only redistribute
    /// weight); the default is now local reorthogonalisation without a
    /// stored basis, which is what the backend (GPU) lane always did and
    /// removes the O(M^2 N) CGS2 traffic and the M x N basis of the CPU
    /// lane. Set true to restore the kept-basis FullCGS2 behaviour.
    bool          full_reorthogonalization = false;
    std::uint64_t reorth_frequency         = 10;
    bool          store_intermediate       = false;
    bool          compute_error_bars       = true;

    /// Stage 12f (SU(2) rollout): host-side transform applied to every
    /// Gaussian sample seed before it is normalised and staged (e.g. the
    /// Lowdin total-spin projection, so the stochastic trace runs over
    /// one spin tower). Must leave a normalisable vector; a zero result
    /// throws (the targeted subspace has no weight in this block).
    std::function<void(Complex*, std::size_t)> seed_transform;
};

struct FtlmResult {
    std::vector<double> betas;
    std::vector<double> temperatures;        ///< 1/betas, or opts.temperatures verbatim (grid order)
    std::vector<double> partition_function;
    std::vector<double> energy;
    std::vector<double> heat_capacity;
    std::vector<double> entropy;
    std::vector<double> free_energy;
    double ground_state_estimate =
        std::numeric_limits<double>::quiet_NaN();
};

namespace detail {

inline FtlmResult to_ftlm_result(const ::FTLMResults& legacy,
                                 const std::vector<double>& betas) {
    FtlmResult out;
    out.betas              = betas;
    out.temperatures       = legacy.thermo_data.temperatures;
    out.partition_function = legacy.thermo_data.Z_sample;
    out.energy             = legacy.thermo_data.energy;
    out.heat_capacity      = legacy.thermo_data.specific_heat;
    out.entropy            = legacy.thermo_data.entropy;
    out.free_energy        = legacy.thermo_data.free_energy;
    out.ground_state_estimate = legacy.ground_state_estimate;
    return out;
}

/// The (temperatures, betas) evaluation grid of ``opts``, index-aligned.
/// ``opts.temperatures`` wins when set (taken verbatim, betas = 1/T unless
/// the caller supplied them); otherwise T = 1/beta as before.
struct FtlmGrid {
    std::vector<double> temperatures;
    std::vector<double> betas;
};

inline FtlmGrid resolve_ftlm_grid(const FtlmOptions& opts,
                                  const char* who) {
    FtlmGrid g;
    if (!opts.temperatures.empty()) {
        if (!opts.betas.empty()
            && opts.betas.size() != opts.temperatures.size()) {
            throw std::invalid_argument(
                std::string(who) + ": opts.temperatures and opts.betas "
                "must have the same length when both are set.");
        }
        for (double t : opts.temperatures) {
            if (!(t > 0.0)) {
                throw std::invalid_argument(
                    std::string(who) + ": opts.temperatures must be "
                    "strictly positive.");
            }
        }
        g.temperatures = opts.temperatures;
        if (!opts.betas.empty()) {
            g.betas = opts.betas;
        } else {
            g.betas.reserve(g.temperatures.size());
            for (double t : g.temperatures) g.betas.push_back(1.0 / t);
        }
        return g;
    }
    if (opts.betas.empty()) {
        throw std::invalid_argument(
            std::string(who) + ": opts.betas (or opts.temperatures) must "
            "be non-empty (the temperature grid is required to evaluate "
            "Z, <E>, Cv, S).");
    }
    g.betas = opts.betas;
    g.temperatures.reserve(opts.betas.size());
    for (double b : opts.betas) {
        if (!(b > 0.0)) {
            throw std::invalid_argument(
                std::string(who) + ": opts.betas must be strictly "
                "positive.");
        }
        g.temperatures.push_back(1.0 / b);
    }
    return g;
}

/// Phase E of the "Close CPU/GPU Gaps" plan (May 2026): backend-
/// templated FTLM body. Used by every single-rank specialisation of
/// ``ftlm_kernel`` (``CpuBackend`` since WP10 C5, ``CudaBackend``;
/// future MPI lanes will land alongside cross-rank Lanczos
/// post-processing).
///
/// Mirrors the LTLM dual-backend pattern but is simpler:
/// FTLM only needs the first-component weights ``|<v0 | q_k>|^2``
/// (which the tridiagonal eigenvector solve already returns) so by
/// default we do NOT keep the Lanczos basis around (``keep_basis=false``,
/// LocalDGKS3). ``opts.full_reorthogonalization`` switches to FullCGS2
/// with a kept basis, the same kernel call the CPU driver's
/// ``build_lanczos_tridiagonal`` makes when full reorth is requested.
///
/// Parity with the retired Gen-1 CPU driver (WP10; deleted in C6):
///   * the whole call runs under ``ThreadBudgetScope(auto_threads_for_dim
///     (local_n))``. Nested inside the orchestrator's identical scope it
///     is a no-op: ``auto_threads_for_dim`` never exceeds the current
///     ``omp_get_max_threads()`` and the scope only touches the runtimes
///     when the requested count differs from the current one;
///   * a sample whose Lanczos / tridiagonal solve yields no Ritz values
///     is skipped (with a warning) instead of aborting the run; the
///     call throws only if every sample failed;
///   * ``ground_state_estimate`` is the minimum lowest Ritz value over
///     the valid samples;
///   * sample ``s`` starts from the driver's vector: engine
///     ``sample_engine(resolve_base_seed(seed), s)`` and
///     ``generateGaussianRandomVector`` (dznrm2 normalisation), pinned by
///     tests/unit/test_ftlm_sample_seed.cpp.
///
/// Algorithm per sample:
///   1. Host-side Gaussian seed ``v_0`` (see above), copy to backend
///      device-side scratch via ``backend.copy_from_host``.
///   2. ``lanczos_kernel<Backend>`` (krylov_dim, keep_basis=false) ->
///      tridiagonal ``(alpha, beta)``.
///   3. Host-side ``diagonalize_tridiagonal_ritz`` ->
///      ``ritz_values`` + first-component ``weights``.
///   4. Host-side ``compute_ftlm_thermodynamics`` -> per-sample
///      ``ThermodynamicData`` on the supplied temperature grid.
///
/// After the sample loop we hand the per-sample
/// ``ThermodynamicData`` vector to ``::average_ftlm_samples`` (the
/// existing host-side Jensen-correct averager) so the (CPU vs GPU)
/// lane produces identical output to within Lanczos noise.
template <typename Backend, typename MatvecFn>
FtlmResult ftlm_kernel_via_backend(const Backend& backend,
                                    MatvecFn&&     apply_H,
                                    std::size_t    local_n,
                                    std::uint64_t  global_n,
                                    const FtlmOptions& opts)
{
    if (local_n == 0) {
        throw std::invalid_argument("ftlm_kernel: local_n must be > 0");
    }
    if (opts.krylov_dim < 2) {
        throw std::invalid_argument(
            "ftlm_kernel: krylov_dim must be >= 2");
    }
    if (opts.num_samples == 0) {
        throw std::invalid_argument(
            "ftlm_kernel: num_samples must be > 0");
    }
    // Beta -> temperature for the host post-processors (or the caller's
    // exact ``opts.temperatures``). ``compute_ftlm_thermodynamics`` is
    // temperature-driven; the caller's ordering is preserved so the
    // returned curves are index-aligned with the grid.
    const FtlmGrid grid = resolve_ftlm_grid(opts, "ftlm_kernel");
    const std::vector<double>& temperatures = grid.temperatures;

    // Same dim-aware OMP+BLAS thread cap as the CPU driver. Harmless when
    // the orchestrator already applied it (see the doc comment above).
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(
            static_cast<std::uint64_t>(local_n)));

    // Seed contract shared with the CPU driver (WP10 C3): seed == 0 means
    // NONDETERMINISTIC ("use random_device"); explicit seeds are taken
    // verbatim, and every sample draws from its own ``sample_engine``.
    const std::uint64_t base_seed = resolve_base_seed(opts.random_seed);

    // generateGaussianRandomVector and the BLAS normalisation take int.
    if (local_n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "ftlm_kernel: local_n exceeds the int range of the host "
            "random-vector draw");
    }
    const int n_int = static_cast<int>(local_n);

    std::vector<::ThermodynamicData> per_sample;
    per_sample.reserve(opts.num_samples);
    double ground_state_estimate = std::numeric_limits<double>::infinity();

    for (std::size_t s = 0; s < opts.num_samples; ++s) {
        // ---- 1. Seed v_0 on the host, copy to backend ----
        // The CPU driver's draw verbatim (same engine, same Gaussian
        // stream, same dznrm2 + zscal normalisation), so both lanes start
        // every sample from bit-identical vectors.
        std::mt19937 rng = sample_engine(base_seed, static_cast<std::uint64_t>(s));
        ComplexVector v0_host = generateGaussianRandomVector(n_int, rng);
        // Stage 12f: subspace projection of the stochastic seed (e.g.
        // Lowdin total-spin), then renormalise the same way.
        if (opts.seed_transform) {
            opts.seed_transform(v0_host.data(), local_n);
            const double v0_nrm = cblas_dznrm2(n_int, v0_host.data(), 1);
            if (!(v0_nrm > 0.0)) {
                throw std::runtime_error(
                    "ftlm_kernel: zero-norm random start vector for sample "
                    + std::to_string(s)
                    + " (the seed transform annihilated it -- the "
                      "targeted subspace has no weight in this block)");
            }
            const Complex scale(1.0 / v0_nrm, 0.0);
            cblas_zscal(n_int, &scale, v0_host.data(), 1);
        }

        auto d_v0 = backend.make_zero_vector(local_n);
        backend.copy_from_host(v0_host.data(), d_v0.get(), local_n);

        // ---- 2. Lanczos: tridiagonal (basis kept only for full reorth) ----
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter = opts.krylov_dim;
        if (opts.full_reorthogonalization) {
            // Same kernel call as the CPU driver's
            // ``build_lanczos_tridiagonal`` with full_reorth = true.
            kopts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
            kopts.keep_basis = true;
        } else {
            // FTLM's first-component weights come from the tridiagonal
            // eigenvectors directly, so we only need a faithful (alpha,
            // beta) -- LocalDGKS3 is the cheap canonical reorth policy
            // matching the legacy CPU driver's
            // ``build_lanczos_tridiagonal`` body when full reorth is off.
            kopts.reorth     = ed::krylov::ReorthPolicy::LocalDGKS3;
            kopts.keep_basis = false;
        }

        std::vector<double> ritz_values;
        std::vector<double> weights;
        {
            auto k = ed::krylov::lanczos_kernel(
                backend,
                std::forward<MatvecFn>(apply_H),
                local_n, d_v0.get(), kopts);

            // ---- 3. Diagonalise tridiagonal on host -> ritz + weights ----
            // (the kept basis, if any, is released at the end of this
            // block, before the host-side post-processing).
            if (!k.alpha.empty()) {
                diagonalize_tridiagonal_ritz(
                    k.alpha, k.beta, ritz_values, weights);
            }
        }
        if (ritz_values.empty()) {
            // CPU-driver parity: a failed sample is dropped, not fatal.
            std::cerr << "  Warning: Tridiagonal diagonalization failed "
                         "(sample " << s << ")" << std::endl;
            continue;
        }
        ground_state_estimate =
            std::min(ground_state_estimate, ritz_values.front());

        // ---- 4. Host-side thermodynamics for this sample ----
        ::ThermodynamicData td = ::compute_ftlm_thermodynamics(
            ritz_values, weights, temperatures,
            static_cast<std::uint64_t>(global_n));
        per_sample.push_back(std::move(td));
    }

    if (per_sample.empty()) {
        throw std::runtime_error(
            "ftlm_kernel: every sample failed (no Ritz values from any "
            "of the " + std::to_string(opts.num_samples) + " samples)");
    }

    // ---- 5. Jensen-correct sample averaging ----
    ::FTLMResults legacy;
    legacy.ground_state_estimate = ground_state_estimate;
    ::average_ftlm_samples(per_sample, legacy);
    legacy.thermo_data.temperatures = temperatures;
    // Surface the raw Z_sample average too so ``to_ftlm_result`` can
    // forward it on the partition_function field. ``average_ftlm_samples``
    // already populates ``legacy.thermo_data.{energy, specific_heat,
    // entropy}``; we recompute Z_sample as the per-temperature mean
    // because the averager does not write it into ``legacy.thermo_data``.
    if (!per_sample.empty()
        && !per_sample.front().Z_sample.empty()) {
        legacy.thermo_data.Z_sample.assign(temperatures.size(), 0.0);
        for (const auto& td : per_sample) {
            for (std::size_t t = 0; t < temperatures.size(); ++t) {
                if (t < td.Z_sample.size()) {
                    legacy.thermo_data.Z_sample[t] += td.Z_sample[t];
                }
            }
        }
        const double inv_n =
            1.0 / static_cast<double>(per_sample.size());
        for (auto& z : legacy.thermo_data.Z_sample) z *= inv_n;
    }
    return to_ftlm_result(legacy, grid.betas);
}

}  // namespace detail

/// FTLM kernel facade.
///
/// ``CpuBackend`` and ``CudaBackend`` both delegate to
/// ``detail::ftlm_kernel_via_backend`` (WP10 C5; CUDA since Phase E of
/// the "Close CPU/GPU Gaps" plan, May 2026) -- a fully Backend-templated
/// body that reuses ``lanczos_kernel<Backend>`` once per random sample.
/// On CUDA all BLAS-1 ops run device-resident; the only cross-PCI
/// traffic is the host-seeded random starting vector per sample (a
/// single ``~N*16`` byte transfer at the top of each sample) and the
/// small ``(M x M)`` tridiagonal diagonalisation handled on the host
/// with LAPACK. On CPU the result equals what the deleted Gen-1 driver
/// returned for the same options; its opt-in sample-parallel loop,
/// per-sample HDF5 dumps (``store_intermediate``) and verbose sample
/// logging were not carried over.
template <typename Backend, typename MatvecFn>
FtlmResult ftlm_kernel(const Backend&  backend,
                       MatvecFn&&      apply_H,
                       std::size_t     local_n,
                       std::uint64_t   global_n,
                       const FtlmOptions& opts)
{
    // WP10 C5: one body for both single-rank lanes. The CPU lane used to
    // convert to FTLMParameters and call the Gen-1 driver (deleted in C6);
    // the Backend-templated body reproduced it sample for sample, so it is
    // used here too.
    constexpr bool single_rank =
#ifdef WITH_CUDA
        std::is_same_v<Backend, ed::matvec::CudaBackend> ||
#endif
        std::is_same_v<Backend, ed::matvec::CpuBackend>;
    if constexpr (single_rank) {
        return detail::ftlm_kernel_via_backend(
            backend,
            std::forward<MatvecFn>(apply_H),
            local_n, global_n, opts);
    } else {
        // MpiBackend / MpiCudaBackend: needs cross-rank reductions in
        // the Lanczos basis and in the per-temperature thermo averages
        // that the current Backend-templated body does not yet handle.
        // Tracked separately under the MPI-parity follow-up.
        throw std::runtime_error(
            "ftlm_kernel: distributed backends are not yet supported. "
            "Pin BackendConstraints::allow_mpi = false to route through "
            "the CPU/CUDA lanes.");
    }
}

}  // namespace ed::thermal

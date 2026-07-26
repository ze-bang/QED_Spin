#pragma once
// =============================================================================
// include/ed/thermal/ftlm_kernel.h
//
// FTLM (Finite-Temperature Lanczos Method) kernel ---
// ``template<Backend, MatvecFn>``. Same dual-backend pattern as
// ``ltlm_kernel<Backend>``:
//
//   * ``CpuBackend``: delegates to the legacy ``::finite_temperature_lanczos``
//     in ``src/solvers/cpu/ftlm.cpp`` so existing HDF5 output, console
//     logging, and OpenMP-over-samples behaviour are preserved.
//   * ``CudaBackend`` (Phase E of the "Close CPU/GPU Gaps" plan,
//     May 2026): delegates to ``detail::ftlm_kernel_via_backend``, a
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
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/lanczos.h>      // diagonalize_tridiagonal_ritz

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
    std::uint64_t random_seed = 0;
    std::string output_dir;

    /// Stage 12f (SU(2) rollout): host-side transform applied to every
    /// Gaussian sample seed before it is normalised and staged (e.g. the
    /// Lowdin total-spin projection, so the stochastic trace runs over
    /// one spin tower). Must leave a normalisable vector; a zero result
    /// throws (the targeted subspace has no weight in this block).
    std::function<void(Complex*, std::size_t)> seed_transform;
};

struct FtlmResult {
    std::vector<double> betas;
    std::vector<double> partition_function;
    std::vector<double> energy;
    std::vector<double> heat_capacity;
    std::vector<double> entropy;
};

namespace detail {

inline FtlmResult to_ftlm_result(const ::FTLMResults& legacy,
                                 const std::vector<double>& betas) {
    FtlmResult out;
    out.betas              = betas;
    out.partition_function = legacy.thermo_data.Z_sample;
    out.energy             = legacy.thermo_data.energy;
    out.heat_capacity      = legacy.thermo_data.specific_heat;
    out.entropy            = legacy.thermo_data.entropy;
    return out;
}

/// Phase E of the "Close CPU/GPU Gaps" plan (May 2026): backend-
/// templated FTLM body. Used by every non-Cpu Backend specialisation
/// of ``ftlm_kernel`` (today: ``CudaBackend``; future MPI lanes will
/// land alongside cross-rank Lanczos post-processing).
///
/// Mirrors the LTLM dual-backend pattern but is simpler:
/// FTLM only needs the first-component weights ``|<v0 | q_k>|^2``
/// (which the tridiagonal eigenvector solve already returns) so we do
/// NOT keep the Lanczos basis around (``keep_basis=false``). This
/// eliminates the per-sample device basis allocation that LTLM needs
/// for Ritz reconstruction.
///
/// Algorithm per sample:
///   1. Host-side Gaussian seed ``v_0`` (normalised), copy to backend
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
    if (opts.betas.empty()) {
        throw std::invalid_argument(
            "ftlm_kernel: opts.betas must be non-empty (the temperature "
            "grid is required to evaluate Z, <E>, Cv, S).");
    }

    // Convert beta -> temperature for the host post-processors. The
    // ``compute_ftlm_thermodynamics`` API is temperature-driven; we
    // preserve the caller's beta ordering so the returned curves are
    // index-aligned with ``opts.betas``.
    std::vector<double> temperatures;
    temperatures.reserve(opts.betas.size());
    for (double b : opts.betas) {
        if (!(b > 0.0)) {
            throw std::invalid_argument(
                "ftlm_kernel: opts.betas must be strictly positive.");
        }
        temperatures.push_back(1.0 / b);
    }

    const std::uint64_t base_seed = (opts.random_seed != 0)
        ? opts.random_seed
        : 0xFEEDFACEULL;

    std::vector<::ThermodynamicData> per_sample;
    per_sample.reserve(opts.num_samples);

    for (std::size_t s = 0; s < opts.num_samples; ++s) {
        // Per-sample RNG: salt the user seed with the sample index so
        // the CPU and GPU lanes draw the same sequence given the same
        // ``opts.random_seed``.
        std::mt19937_64 rng(base_seed + 0x9E3779B97F4A7C15ULL * s);
        std::normal_distribution<double> gauss(0.0, 1.0);

        // ---- 1. Seed v_0 on the host, normalise, copy to backend ----
        std::vector<Complex> v0_host(local_n);
        for (auto& c : v0_host) c = Complex(gauss(rng), gauss(rng));
        // Stage 12f: subspace projection of the stochastic seed (e.g.
        // Lowdin total-spin), BEFORE normalisation.
        if (opts.seed_transform) {
            opts.seed_transform(v0_host.data(), local_n);
        }
        double sum_sq = 0.0;
        for (auto c : v0_host) sum_sq += std::norm(c);
        const double v0_nrm = std::sqrt(sum_sq);
        if (!(v0_nrm > 0.0)) {
            throw std::runtime_error(
                "ftlm_kernel: zero-norm random start vector for sample "
                + std::to_string(s)
                + (opts.seed_transform
                       ? " (the seed transform annihilated it -- the "
                         "targeted subspace has no weight in this block)"
                       : ""));
        }
        const double inv = 1.0 / v0_nrm;
        for (auto& c : v0_host) c *= inv;

        auto d_v0 = backend.make_zero_vector(local_n);
        backend.copy_from_host(v0_host.data(), d_v0.get(), local_n);

        // ---- 2. Lanczos: tridiagonal only (no basis) ----
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter   = opts.krylov_dim;
        kopts.keep_basis = false;
        // FTLM's first-component weights come from the tridiagonal
        // eigenvectors directly, so we only need a faithful (alpha,
        // beta) -- LocalDGKS3 is the cheap canonical reorth policy
        // matching the legacy CPU driver's
        // ``build_lanczos_tridiagonal`` body when full reorth is off.
        kopts.reorth = ed::krylov::ReorthPolicy::LocalDGKS3;

        auto k = ed::krylov::lanczos_kernel(
            backend,
            std::forward<MatvecFn>(apply_H),
            local_n, d_v0.get(), kopts);

        if (k.alpha.empty()) {
            throw std::runtime_error(
                "ftlm_kernel: Lanczos produced no Ritz values for "
                "sample " + std::to_string(s));
        }

        // ---- 3. Diagonalise tridiagonal on host -> ritz + weights ----
        std::vector<double> ritz_values;
        std::vector<double> weights;
        diagonalize_tridiagonal_ritz(k.alpha, k.beta, ritz_values, weights);
        if (ritz_values.empty()) {
            throw std::runtime_error(
                "ftlm_kernel: tridiagonal diagonalisation failed for "
                "sample " + std::to_string(s));
        }

        // ---- 4. Host-side thermodynamics for this sample ----
        ::ThermodynamicData td = ::compute_ftlm_thermodynamics(
            ritz_values, weights, temperatures,
            static_cast<std::uint64_t>(global_n));
        per_sample.push_back(std::move(td));
    }

    // ---- 5. Jensen-correct sample averaging ----
    ::FTLMResults legacy;
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
    return to_ftlm_result(legacy, opts.betas);
}

}  // namespace detail

/// FTLM kernel facade.
///
/// Phase E of the "Close CPU/GPU Gaps" plan (May 2026): closes the
/// previous CPU-only ``static_assert`` gap. The body now dispatches
/// on the backend type:
///   * ``CpuBackend``: delegates to ``::finite_temperature_lanczos``
///     (the legacy CPU driver in ``src/solvers/cpu/ftlm.cpp``) so
///     existing HDF5 output, console logging, and OpenMP-over-samples
///     behaviour is preserved.
///   * ``CudaBackend``: delegates to ``detail::ftlm_kernel_via_backend``
///     -- a fully Backend-templated body that reuses
///     ``lanczos_kernel<CudaBackend>`` (Phase 2 BLAS-1 facade) once per
///     random sample. All BLAS-1 ops run device-resident; the only
///     cross-PCI traffic is the host-seeded random starting vector per
///     sample (a single ``~N*16`` byte transfer at the top of each
///     sample) and the small ``(M x M)`` tridiagonal diagonalisation
///     handled on the host with LAPACK.
template <typename Backend, typename MatvecFn>
FtlmResult ftlm_kernel(const Backend&  backend,
                       MatvecFn&&      apply_H,
                       std::size_t     local_n,
                       std::uint64_t   global_n,
                       const FtlmOptions& opts)
{
    if constexpr (std::is_same_v<Backend, ed::matvec::CpuBackend>) {
        // Stage 12f: the legacy CPU driver draws its own seeds and
        // cannot honour a subspace-restricting ``seed_transform`` --
        // route through the Backend-templated body (which applies the
        // transform per sample) instead of silently sampling the
        // whole block.
        if (opts.seed_transform) {
            return detail::ftlm_kernel_via_backend(
                backend, std::forward<MatvecFn>(apply_H),
                local_n, global_n, opts);
        }
        // CPU lane: legacy driver with full I/O behaviour.
        FTLMParameters params;
        params.krylov_dim   = static_cast<std::uint64_t>(opts.krylov_dim);
        params.num_samples  = static_cast<std::uint64_t>(opts.num_samples);
        params.random_seed  = opts.random_seed;

        // Evaluate the thermodynamics on the EXACT temperature grid the
        // caller supplied via ``opts.betas`` (T_k = 1/beta_k), preserving
        // its spacing. The previous code forwarded only (temp_min,
        // temp_max, num_bins) to the legacy driver, which rebuilt a
        // *log-spaced* grid internally -- so a linear ``opts.betas`` axis
        // produced energies sampled at the wrong temperatures while the
        // reported ``temperatures`` (derived from ``opts.betas``) stayed
        // linear. That mismatch is now closed: the CPU lane matches the
        // GPU lane, which already used 1/betas directly.
        std::vector<double> temperatures;
        temperatures.reserve(opts.betas.size());
        for (double b : opts.betas) {
            if (!(b > 0.0)) {
                throw std::invalid_argument(
                    "ftlm_kernel (CPU lane): opts.betas must be strictly "
                    "positive.");
            }
            temperatures.push_back(1.0 / b);
        }

        std::function<void(const Complex*, Complex*, int)> legacy_H =
            [&apply_H](const Complex* in, Complex* out, int n) {
                apply_H(in, out, static_cast<std::size_t>(n));
            };

        const auto legacy = ::finite_temperature_lanczos(
            legacy_H,
            static_cast<std::uint64_t>(local_n),
            params, temperatures, opts.output_dir);

        return detail::to_ftlm_result(legacy, opts.betas);
    }
#ifdef WITH_CUDA
    else if constexpr (std::is_same_v<Backend, ed::matvec::CudaBackend>) {
        return detail::ftlm_kernel_via_backend(
            backend,
            std::forward<MatvecFn>(apply_H),
            local_n, global_n, opts);
    }
#endif
    else {
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

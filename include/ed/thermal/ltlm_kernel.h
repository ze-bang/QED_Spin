#pragma once
// =============================================================================
// include/ed/thermal/ltlm_kernel.h
//
// LTLM (Low Temperature Lanczos Method) kernel —
// `template<Backend, MatvecFn>`. CPU body delegates to the existing
// `low_temperature_lanczos` in `src/solvers/cpu/ltlm.cpp`. GPU / MPI
// specialisations land alongside `CudaBackend` / `MpiBackend`.
//
// Algorithm:
//   * Find ground state |0>.
//   * Build a Krylov subspace from |0>.
//   * Evaluate Z, <O>, free energy with the Lehmann representation of
//     the projected H eigenpairs.
//
// LTLM is strictly more accurate than FTLM at low temperatures (it
// captures the ground state exactly) and is the natural finite-T
// driver below the FTLM/LTLM crossover.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/solvers/lanczos.h>      // diagonalize_tridiagonal_ritz
#include <ed/solvers/ltlm.h>

#ifdef WITH_CUDA
// Forward declaration so the `if constexpr` branch below can refer to
// CudaBackend without dragging the full `cuda_backend.cuh` into every
// consumer of this header. The orchestrator TU (which actually calls
// `ltlm_kernel<CudaBackend>`) already pulls the full definition via
// `select_backend.h`.
namespace ed { namespace matvec { class CudaBackend; } }
#endif

namespace ed::thermal {

using Complex = std::complex<double>;

struct LtlmOptions {
    std::size_t num_samples         = 1;
    std::size_t krylov_dim          = 200;
    std::size_t ground_state_krylov = 100;
    std::vector<double> betas;
    std::uint64_t random_seed = 0;
    std::string output_dir;
};

struct LtlmResult {
    std::vector<double> betas;
    std::vector<double> partition_function;
    std::vector<double> energy;
    std::vector<double> heat_capacity;
    std::vector<double> entropy;
    double ground_state_energy = 0.0;
};

namespace detail {

inline LtlmResult to_ltlm_result(const ::LTLMResults& legacy,
                                 const std::vector<double>& betas) {
    LtlmResult out;
    out.betas                = betas;
    out.partition_function   = legacy.thermo_data.Z_sample;
    out.energy               = legacy.thermo_data.energy;
    out.heat_capacity        = legacy.thermo_data.specific_heat;
    out.entropy              = legacy.thermo_data.entropy;
    out.ground_state_energy  = legacy.ground_state_energy;
    return out;
}

}  // namespace detail

namespace detail {

/// Templated LTLM body, runnable against any `Backend` that exposes
/// the standard BLAS-1 primitives. Used by the `CudaBackend`
/// specialisation of `ltlm_kernel`. The CPU lane keeps the legacy
/// ``::low_temperature_lanczos`` path to preserve its HDF5 output
/// and console logging behaviour.
///
/// Algorithm (mirrors src/solvers/cpu/ltlm.cpp::low_temperature_lanczos):
///   1. Seed a random unit vector v_0 on the host, copy to backend.
///   2. Run `lanczos_kernel<Backend>` (gs_krylov, keep_basis=true) ->
///      tridiagonal (alpha, beta) + Lanczos basis V.
///   3. Diagonalise the tridiagonal on the host -> Ritz values,
///      eigenvectors c.
///   4. Reconstruct |GS> = sum_j c[j,0] V_j via backend axpy_many.
///   5. Normalise |GS>.
///   6. Run `lanczos_kernel<Backend>` (krylov_dim, keep_basis=false)
///      from |GS> -> excitation tridiagonal.
///   7. Diagonalise -> excitation energies E_i and weights
///      w_i = |<GS | ψ_i>|^2.
///   8. Compute Z, E, Cv, S using the Lehmann representation, shifted
///      by ground-state energy for numerical stability.
template <typename Backend, typename MatvecFn>
LtlmResult ltlm_kernel_via_backend(const Backend& backend,
                                    MatvecFn&&     apply_H,
                                    std::size_t    local_n,
                                    const LtlmOptions& opts)
{
    if (local_n == 0) {
        throw std::invalid_argument("ltlm_kernel: local_n must be > 0");
    }
    if (opts.ground_state_krylov < 2) {
        throw std::invalid_argument(
            "ltlm_kernel: ground_state_krylov must be >= 2");
    }
    if (opts.krylov_dim < 2) {
        throw std::invalid_argument(
            "ltlm_kernel: krylov_dim must be >= 2");
    }

    const std::uint64_t seed = (opts.random_seed != 0)
        ? opts.random_seed
        : 0xC0FFEE2BULL;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // ---- 1. Seed v_0 on the host, normalise, copy to backend ----
    std::vector<Complex> v0_host(local_n);
    for (auto& c : v0_host) c = Complex(gauss(rng), gauss(rng));
    double sum_sq = 0.0;
    for (auto c : v0_host) sum_sq += std::norm(c);
    const double v0_nrm = std::sqrt(sum_sq);
    if (!(v0_nrm > 0.0)) {
        throw std::runtime_error(
            "ltlm_kernel: zero-norm random start vector");
    }
    for (auto& c : v0_host) c /= v0_nrm;

    auto d_v0 = backend.make_zero_vector(local_n);
    backend.copy_from_host(v0_host.data(), d_v0.get(), local_n);

    // ---- 2. First Lanczos: tridiagonal + basis for GS recovery ----
    ed::krylov::LanczosKernelOptions gs_opts;
    gs_opts.max_iter   = opts.ground_state_krylov;
    gs_opts.keep_basis = true;
    gs_opts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;

    auto gs_res = ed::krylov::lanczos_kernel(
        backend,
        std::forward<MatvecFn>(apply_H),
        local_n, d_v0.get(), gs_opts);

    if (gs_res.alpha.empty()) {
        throw std::runtime_error(
            "ltlm_kernel: ground-state Lanczos produced no Ritz values");
    }

    // ---- 3. Diagonalise tridiagonal on host ----
    std::vector<double> gs_ritz, gs_weights, gs_evecs;
    diagonalize_tridiagonal_ritz(
        gs_res.alpha, gs_res.beta, gs_ritz, gs_weights, &gs_evecs);
    if (gs_ritz.empty()) {
        throw std::runtime_error(
            "ltlm_kernel: ground-state tridiagonal diagonalisation failed");
    }
    const double ground_energy = gs_ritz[0];

    // ---- 4. Reconstruct |GS> on the backend ----
    const std::size_t m = gs_res.alpha.size();
    auto d_gs = backend.make_zero_vector(local_n);
    std::vector<const Complex*> basis_ptrs(m, nullptr);
    std::vector<Complex>        coeffs(m, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < m; ++j) {
        basis_ptrs[j] = gs_res.basis[j].get();
        // First eigenvector (ground state), column-major: evecs[j + 0*m].
        coeffs[j] = Complex(gs_evecs[j], 0.0);
    }
    backend.axpy_many(coeffs.data(), basis_ptrs.data(), m,
                      d_gs.get(), local_n);

    // ---- 5. Normalise |GS> ----
    const double gs_nrm = backend.nrm2(d_gs.get(), local_n);
    if (!(gs_nrm > 0.0)) {
        throw std::runtime_error(
            "ltlm_kernel: reconstructed ground state has zero norm");
    }
    backend.scale(Complex(1.0 / gs_nrm, 0.0), d_gs.get(), local_n);

    // ---- 6. Second Lanczos: excitation spectrum from |GS> ----
    ed::krylov::LanczosKernelOptions exc_opts;
    exc_opts.max_iter   = opts.krylov_dim;
    exc_opts.keep_basis = true;   // FullCGS2 reorth needs the basis.
    exc_opts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;

    auto exc_res = ed::krylov::lanczos_kernel(
        backend,
        std::forward<MatvecFn>(apply_H),
        local_n, d_gs.get(), exc_opts);

    // ---- 7. Diagonalise -> excitation energies + weights ----
    std::vector<double> exc_energies, exc_weights;
    diagonalize_tridiagonal_ritz(
        exc_res.alpha, exc_res.beta, exc_energies, exc_weights);
    if (exc_energies.empty()) {
        throw std::runtime_error(
            "ltlm_kernel: excitation tridiagonal diagonalisation failed");
    }

    // ---- 8. Compute Z, E, Cv, S ----
    //   Z(beta)  = sum_i w_i exp(-beta * (E_i - E_GS))
    //   <E>(beta) = (1/Z) sum_i w_i E_i exp(-beta * (E_i - E_GS))
    //   <E^2>     = (1/Z) sum_i w_i E_i^2 exp(...)
    //   Cv        = beta^2 (<E^2> - <E>^2)
    //   S         = beta (<E> - E_GS) + log(Z)
    //   F         = E_GS - T log(Z)
    // The shift by E_GS keeps the Boltzmann factor in [0, 1].
    LtlmResult out;
    out.betas               = opts.betas;
    out.ground_state_energy = ground_energy;
    out.partition_function.assign(opts.betas.size(), 0.0);
    out.energy.assign(opts.betas.size(),         0.0);
    out.heat_capacity.assign(opts.betas.size(),  0.0);
    out.entropy.assign(opts.betas.size(),        0.0);

    for (std::size_t t = 0; t < opts.betas.size(); ++t) {
        const double beta = opts.betas[t];
        double Z   = 0.0;
        double E1  = 0.0;
        double E2  = 0.0;
        for (std::size_t i = 0; i < exc_energies.size(); ++i) {
            const double shifted = exc_energies[i] - ground_energy;
            const double bf      = exc_weights[i] * std::exp(-beta * shifted);
            Z  += bf;
            E1 += bf * exc_energies[i];
            E2 += bf * exc_energies[i] * exc_energies[i];
        }
        if (Z > 1e-300) {
            const double E_mean   = E1 / Z;
            const double E_sq     = E2 / Z;
            out.partition_function[t] = Z;
            out.energy[t]             = E_mean;
            out.heat_capacity[t]      = beta * beta * (E_sq - E_mean * E_mean);
            out.entropy[t]            = beta * (E_mean - ground_energy)
                                        + std::log(Z);
        } else {
            out.partition_function[t] = 0.0;
            out.energy[t]             = ground_energy;
            out.heat_capacity[t]      = 0.0;
            out.entropy[t]            = 0.0;
        }
    }
    return out;
}

}  // namespace detail

/// LTLM kernel facade.
///
/// Phase E2 of the "Backend x Symmetries x Workflows" plan
/// (May 2026): closes the previous CPU-only `static_assert` gap. The
/// body now dispatches on the backend type:
///   * `CpuBackend`: delegates to ``::low_temperature_lanczos`` (the
///     legacy CPU driver in src/solvers/cpu/ltlm.cpp) so existing
///     HDF5 output and console logging is preserved.
///   * `CudaBackend`: delegates to ``detail::ltlm_kernel_via_backend``
///     -- a fully Backend-templated body that reuses
///     ``lanczos_kernel<CudaBackend>`` (Phase 2 BLAS-1 facade) twice:
///     once to find the ground state from a random start vector, and
///     once from the reconstructed ground state to build the
///     excitation spectrum. All BLAS-1 ops run device-resident; the
///     only cross-PCI traffic is the host-seeded random starting
///     vector (a single ~N*16 byte transfer at the top of the run)
///     and the small (M x M) tridiagonal diagonalisation handled on
///     the host with LAPACK.
template <typename Backend, typename MatvecFn>
LtlmResult ltlm_kernel(const Backend&  backend,
                       MatvecFn&&      apply_H,
                       std::size_t     local_n,
                       std::uint64_t   /*global_n*/,
                       const LtlmOptions& opts)
{
    if constexpr (std::is_same_v<Backend, ed::matvec::CpuBackend>) {
        // CPU lane: legacy driver with full I/O behaviour.
        LTLMParameters params;
        params.krylov_dim           = static_cast<std::uint64_t>(opts.krylov_dim);
        params.ground_state_krylov  = static_cast<std::uint64_t>(opts.ground_state_krylov);
        params.num_samples          = static_cast<std::uint64_t>(opts.num_samples);
        params.random_seed          = opts.random_seed;

        const double temp_min = opts.betas.empty()
            ? 0.01 : 1.0 / *std::max_element(opts.betas.begin(), opts.betas.end());
        const double temp_max = opts.betas.empty()
            ? 100.0 : 1.0 / *std::min_element(opts.betas.begin(), opts.betas.end());
        const std::uint64_t num_bins =
            opts.betas.empty() ? 32u
                                : static_cast<std::uint64_t>(opts.betas.size());

        std::function<void(const Complex*, Complex*, int)> legacy_H =
            [&apply_H](const Complex* in, Complex* out, int n) {
                apply_H(in, out, static_cast<std::size_t>(n));
            };

        const auto legacy = ::low_temperature_lanczos(
            legacy_H,
            static_cast<std::uint64_t>(local_n),
            params, temp_min, temp_max, num_bins, nullptr, opts.output_dir);

        return detail::to_ltlm_result(legacy, opts.betas);
    }
#ifdef WITH_CUDA
    else if constexpr (std::is_same_v<Backend, ed::matvec::CudaBackend>) {
        return detail::ltlm_kernel_via_backend(
            backend,
            std::forward<MatvecFn>(apply_H),
            local_n, opts);
    }
#endif
    else {
        // MpiBackend / MpiCudaBackend: needs cross-rank reductions in
        // the Lanczos basis + thermodynamic post-processing that the
        // current Backend-templated body does not yet handle.
        // Tracked separately under wave-b-thermal.
        throw std::runtime_error(
            "ltlm_kernel: distributed backends are not yet supported. "
            "Pin BackendConstraints to route through the CPU/CUDA lanes.");
    }
}

}  // namespace ed::thermal

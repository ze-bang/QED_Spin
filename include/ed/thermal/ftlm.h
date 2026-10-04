#pragma once
// =============================================================================
// include/ed/thermal/ftlm.h
//
// FTLM (Finite-Temperature Lanczos Method) kernel ---
// ``template<Backend, MatvecFn>``:
//
//   * one Backend-templated body for ``CpuBackend`` and ``CudaBackend`` that
//     reuses ``lanczos_kernel<Backend>``
//     (BLAS-1 facade) once per random sample. All BLAS-1 ops
//     run device-resident; cross-PCI traffic is limited to a host-seeded
//     random starting vector per sample and the small (M x M)
//     tridiagonal diagonalisation handled on the host with LAPACK.
//
// Algorithm:
//   * Draw ``R`` Gaussian random unit vectors ``|r>``.
//   * For each ``|r>`` run a length-``M`` Lanczos.
//   * Diagonalise the tridiagonal ``(alpha, beta)`` -> Ritz values + first-
//     component weights ``|<r | q_k>|^2``.
//   * Per sample, the moments Z, E1, E2 of the Ritz-value Lehmann representation of
//     the Krylov projection, about that sample's lowest Ritz value
//     (``detail::sample_moments``).
//   * Average the moments across samples about one common reference before taking
//     logarithms (Jensen: <ln Z> != ln <Z>), and return ln Z, E and the central
//     second moment V per beta (``detail::combine_samples``, ``Curves``).
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <ed/core/log.h>
#include <ed/krylov/lanczos.h>
#include <ed/krylov/tridiag.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/batcher.h>
#include <ed/matvec/cpu_backend.h>
#include <ed/parallel/thread_budget.h>  // auto_threads_for_dim + ThreadBudgetScope
#include <ed/thermal/curves.h>
#include <ed/thermal/sample_seed.h>

namespace ed::thermal {

using Complex = std::complex<double>;

struct FtlmOptions {
    std::size_t num_samples = 40;
    std::size_t krylov_dim = 100;
    /// A sample's Lanczos run stops at an invariant subspace, beta <= breakdown_tol (energy
    /// units; the engine passes 64 eps s_H, <ed/core/numerics.h>). 0: every step runs.
    double breakdown_tol = 0.0;
    std::vector<double> betas;           ///< inverse-temperature grid (positive), any order

    std::uint64_t random_seed = 0;       ///< 0 = nondeterministic (random_device)

    /// Stochastic-trace samples do not need a mutually orthogonal Krylov
    /// basis (ghost Ritz values only redistribute weight), so the default
    /// is local reorthogonalisation without a stored basis (no O(M^2 N)
    /// CGS2 traffic, no M x N basis). Set true for kept-basis FullCGS2.
    bool full_reorthogonalization = false;

    /// Host-side transform applied to every
    /// Gaussian sample seed before it is normalised and staged (e.g. the
    /// Lowdin total-spin projection, so the stochastic trace runs over
    /// one spin tower). Must leave a normalisable vector; a zero result
    /// throws (the targeted subspace has no weight in this block).
    std::function<void(Complex*, std::size_t)> seed_transform;

    /// Ritz pairs whose start-vector weight |<v0|psi_j>|^2 lies below this are dropped (0: none).
    /// A start inside an invariant subspace (seed_transform onto a spin tower) reaches the levels
    /// outside it only through roundoff, with roundoff weight: below the subspace's lowest level
    /// such a copy would still dominate Z at low enough T.
    double min_weight = 0.0;

    /// Static observables (`n_observables` > 0): <O>(T) = sum_r <phi_r|O|phi_r> / sum_r <phi_r|phi_r>
    /// with phi_r(beta) = sum_j e^{-beta (e_j - e_r) / 2} <psi_j|r> psi_j (the symmetric,
    /// low-temperature Lanczos form; e_r the sample's lowest weighted Ritz value). The Krylov basis is
    /// kept and fully reorthogonalised; the kernel forms phi for a few temperatures at a time, copies
    /// them to the host and calls `observe`, which returns out[i][o] = <phi_i|O_o|phi_i> for every
    /// vector i and observable o. It may be called from several threads at once (batched samples).
    std::size_t n_observables = 0;
    std::function<std::vector<std::vector<Complex>>(const std::vector<const Complex*>&)> observe;

    /// Device multi-vector H (LinearOperator::bind_cuda_multi). On a CUDA run it lets up to
    /// `batch_width` samples advance in lockstep, each H apply serving all of them in one
    /// launch; every sample computes exactly what it would alone.
    ed::LinearOperator::MultiMatvecFn batch_matvec;
    std::size_t batch_width = 8;
};

struct FtlmResult {
    /// ln Z, E, V and <O> per beta of FtlmOptions::betas (lnZ = ln n + ln <Z_r> - beta e_ref).
    Curves curves;
    /// The lowest Ritz value over the valid samples (an upper bound on the block's lowest level).
    double ground_state_estimate = std::numeric_limits<double>::quiet_NaN();
};

/// OFTLM, the orthogonalized FTLM (Morita & Tohyama, Phys. Rev. Research 2, 013205 (2020)): the
/// N_V lowest eigenstates are treated exactly and the random starts are orthogonalised against
/// them, which removes plain FTLM's low-temperature bias; N_V = 0 is FTLM. With the common shift
/// e_min = eps_0, D the block dimension and R the samples:
///
///   Z(beta) = sum_{i<N_V} e^{-beta (eps_i - e_min)}
///           + (D - N_V)/R sum_r sum_j |<r~|psi_j^r>|^2 e^{-beta (eps~_j^r - e_min)},
///
/// <E> Z and <E^2> Z likewise, ln Z = ln Z(beta) - beta e_min and V = <(H - <H>)^2>. The (D - N_V)
/// factor scales the random part to the complement, so Z is the full trace (no ln D term).
///
/// The estimator is unbiased for ANY orthonormal set of exact eigenvectors (the random part
/// covers the rest of the block), but only if each one is an eigenvector: an unconverged Ritz
/// pair (v, theta) puts e^{-beta theta} where <v|e^{-beta H}|v> belongs, which is larger by about
/// beta^2 ||H v - theta v||^2 / 2 (Jensen), so Z comes out low by an amount no number of samples
/// removes. The caller therefore supplies pairs it has certified by their residuals (the engine
/// uses its block eigensolver); the kernel does not compute them.
struct OftlmOptions {
    std::size_t num_samples = 20;   ///< R: random samples for the stochastic part
    std::size_t krylov_dim = 100;  ///< M: Lanczos steps per random sample
    /// The N_V exact states: eigenvalues and orthonormal eigenvectors (length N each) of the
    /// block, certified by their residuals. Empty is plain FTLM.
    std::vector<double> exact_values;
    std::vector<std::vector<std::complex<double>>> exact_vectors;
    /// Applied to every random start before it is orthogonalised to the exact states (a
    /// projection onto a spin tower); the random part's trace then runs over trace_dim states.
    std::function<void(std::complex<double>*, std::size_t)> seed_transform;
    std::uint64_t trace_dim = 0;   ///< 0: the block's N
    /// Random-part Ritz pairs with start weight below this are dropped (FtlmOptions::min_weight).
    double min_weight = 0.0;
    // scale-free: a default for C++ callers; the engine passes relative values (numerics.h)
    double breakdown_tol = 1e-10; ///< a random sample's run stops at beta <= this (energy units)
    std::vector<double> betas;         ///< inverse-temperature grid (strictly positive)
    std::uint64_t random_seed = 0;
    /// Static observables (FtlmOptions::observe): the exact states' own <psi_i|O|psi_i>, plus the
    /// random part's phi vectors, from a kept and fully reorthogonalised Krylov basis per sample;
    /// <O>(T) = [sum_i e^{-beta eps_i} <psi_i|O|psi_i> + (D - N_V)/R sum_r <phi_r|O|phi_r>] / Z.
    std::size_t n_observables = 0;
    std::function<std::vector<std::vector<std::complex<double>>>(const std::vector<const std::complex<double>*>&)>
        observe;
};

/// OFTLM on one block on backend `be` -- the host's or a device's (src/engine/oftlm.cpp). apply_H:
/// out = H in, length N, on `be`'s memory. The exact vectors go up once; every random start is
/// drawn on the host (seed_transform runs there), then orthogonalised and run on `be`.
Curves oftlm(const ed::matvec::Backend& be,
             const std::function<void(const std::complex<double>*, std::complex<double>*, std::size_t)>& apply_H,
             std::uint64_t N, const OftlmOptions& opts);

namespace detail {

/// One sample's Boltzmann moments about its lowest Ritz value e_min, per beta:
/// Z = sum_i w_i e^{-beta (e_i - e_min)}, E1 = sum_i (e_i - e_min) w_i e^{...}, E2 likewise with
/// (e_i - e_min)^2. Moments about e_min keep E2 free of cancellation.
struct SampleMoments {
    double e_min = 0.0;
    std::vector<double> Z, E1, E2;
};

[[nodiscard]] inline SampleMoments sample_moments(const std::vector<double>& ritz, const std::vector<double>& weights,
                                                  const std::vector<double>& betas) {
    SampleMoments m;
    // The reference is the lowest Ritz value that carries weight (one without weight -- a dropped
    // roundoff copy -- would underflow every term at low T).
    m.e_min = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < ritz.size(); ++i)
        if (weights[i] > 0.0) m.e_min = std::min(m.e_min, ritz[i]);
    if (!std::isfinite(m.e_min)) m.e_min = *std::min_element(ritz.begin(), ritz.end());
    m.Z.resize(betas.size());
    m.E1.resize(betas.size());
    m.E2.resize(betas.size());
    for (std::size_t t = 0; t < betas.size(); ++t) {
        const double beta = betas[t];
        double z = 0.0, e1 = 0.0, e2 = 0.0;
        for (std::size_t i = 0; i < ritz.size(); ++i) {
            const double x = ritz[i] - m.e_min;
            const double b = weights[i] * std::exp(-beta * x);
            z += b;
            e1 += x * b;
            e2 += x * x * b;
        }
        m.Z[t] = z;
        m.E1[t] = e1;
        m.E2[t] = e2;
    }
    return m;
}

/// The FTLM curves of an n-dimensional block from its samples' moments: each sample is rescaled
/// to the common reference e_ref (the lowest e_min) and the moments are averaged before any
/// logarithm, so lnZ = ln n + ln <Z> - beta e_ref, E = e_ref + <E1> / <Z> and
/// V = max(<E2> / <Z> - (<E1> / <Z>)^2, 0). Where <Z> <= 1e-300 the block is its ground
/// state: lnZ = -beta e_ref, E = e_ref, V = 0.
[[nodiscard]] inline Curves combine_samples(const std::vector<SampleMoments>& samples, std::uint64_t n,
                                            const std::vector<double>& betas) {
    Curves c;
    const std::size_t nT = betas.size(), R = samples.size();
    double e_ref = samples.front().e_min;
    for (const auto& s : samples) e_ref = std::min(e_ref, s.e_min);
    c.e_min = e_ref;
    const double ln_n = std::log(static_cast<double>(n));
    c.lnZ.resize(nT);
    c.E.resize(nT);
    c.V.resize(nT);
    for (std::size_t t = 0; t < nT; ++t) {
        const double beta = betas[t];
        double z = 0.0, e1 = 0.0, e2 = 0.0;
        for (const auto& s : samples) {
            // x - e_ref = (x - e_min_s) + d for the moments about each sample's e_min.
            const double d = s.e_min - e_ref, r = std::exp(-beta * d);
            z += s.Z[t] * r;
            e1 += (s.E1[t] + d * s.Z[t]) * r;
            e2 += (s.E2[t] + 2.0 * d * s.E1[t] + d * d * s.Z[t]) * r;
        }
        z /= static_cast<double>(R);
        e1 /= static_cast<double>(R);
        e2 /= static_cast<double>(R);
        if (z > 1e-300) {
            const double m1 = e1 / z, m2 = e2 / z;
            c.lnZ[t] = ln_n + std::log(z) - beta * e_ref;
            c.E[t] = e_ref + m1;
            c.V[t] = std::max(m2 - m1 * m1, 0.0);
        } else {
            c.lnZ[t] = -beta * e_ref;
            c.E[t] = e_ref;
            c.V[t] = 0.0;
        }
    }
    return c;
}

}  // namespace detail

/// FTLM on any Backend (CpuBackend, CudaBackend): one ``lanczos_kernel<Backend>`` per
/// random sample, BLAS-1 on the backend; the only cross-PCI traffic is the host-seeded
/// start vector per sample and the small (M x M) tridiagonal solved on the host.
///
/// FTLM only needs the first-component weights ``|<v0 | q_k>|^2``
/// (which the tridiagonal eigenvector solve already returns) so by
/// default we do NOT keep the Lanczos basis around (``keep_basis=false``,
/// LocalDGKS3). ``opts.full_reorthogonalization`` switches to FullCGS2
/// with a kept basis.
///
/// Contract:
///   * the whole call runs under ``ThreadBudgetScope(auto_threads_for_dim
///     (local_n))``. Nested inside the thermal verb's identical scope it
///     is a no-op: ``auto_threads_for_dim`` never exceeds the current
///     ``omp_get_max_threads()`` and the scope only touches the runtimes
///     when the requested count differs from the current one;
///   * a sample whose Lanczos run yields no Ritz values is skipped (with a
///     warning) instead of aborting the run; the call throws if every sample
///     failed, and a failed tridiagonal solve (a non-finite H) throws;
///   * ``ground_state_estimate`` is the minimum lowest Ritz value over
///     the valid samples;
///   * sample ``s`` starts from the vector drawn with engine
///     ``sample_engine(resolve_base_seed(seed), s)`` and
///     ``gaussian_vector`` (dznrm2 normalisation), pinned by
///     tests/unit/test_ftlm_sample_seed.cpp.
///
/// Algorithm per sample:
///   1. Host-side Gaussian seed ``v_0`` (see above), copy to backend
///      device-side scratch via ``backend.copy_from_host``.
///   2. ``lanczos_kernel<Backend>`` (krylov_dim, keep_basis=false) ->
///      tridiagonal ``(alpha, beta)``.
///   3. Host-side ``tridiag_eig`` ->
///      ``ritz_values`` + first-component ``weights``.
///   4. Host-side ``detail::sample_moments``: the sample's Boltzmann moments per beta.
///
/// After the sample loop ``detail::combine_samples`` averages the moments (Jensen-correct)
/// into the Curves, so the CPU and GPU lanes produce identical output to within Lanczos
/// noise.
template <typename Backend, typename MatvecFn>
FtlmResult ftlm_kernel(const Backend& backend, MatvecFn&& apply_H, std::size_t local_n, const FtlmOptions& opts) {
    if (local_n == 0) { throw std::invalid_argument("ftlm_kernel: local_n must be > 0"); }
    if (opts.krylov_dim < 2) { throw std::invalid_argument("ftlm_kernel: krylov_dim must be >= 2"); }
    if (opts.num_samples == 0) { throw std::invalid_argument("ftlm_kernel: num_samples must be > 0"); }
    // The curves are index-aligned with opts.betas, in the caller's order.
    const std::vector<double>& betas = opts.betas;
    if (betas.empty()) throw std::invalid_argument("ftlm_kernel: opts.betas must be non-empty");
    for (double b : betas)
        if (!(b > 0.0)) throw std::invalid_argument("ftlm_kernel: opts.betas must be strictly positive");

    // Dim-aware OMP+BLAS thread cap. Harmless when the thermal verb
    // already applied it (see the doc comment above).
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(static_cast<std::uint64_t>(local_n)));

    // Seed contract: seed == 0 means
    // NONDETERMINISTIC ("use random_device"); explicit seeds are taken
    // verbatim, and every sample draws from its own ``sample_engine``.
    const std::uint64_t base_seed = resolve_base_seed(opts.random_seed);


    // One sample: its Ritz data and thermodynamics, and each observable in its Ritz basis.
    // Samples are independent; they are combined below in sample order, however they were run.
    const std::size_t n_obs = opts.n_observables;
    if (n_obs > 0 && !opts.observe) throw std::invalid_argument("ftlm_kernel: n_observables > 0 without observe");
    struct Sample {
        bool ok = false;
        std::vector<double> ritz, weights, Y;                  // Y[i m + a]: Ritz vector i
        std::vector<double> oz;                               // oz[t] = <phi_t|phi_t>, about mom.e_min
        std::vector<std::vector<Complex>> obs;                // obs[t][o] = <phi_t|O_o|phi_t>, likewise
        detail::SampleMoments mom;
    };
    auto sample = [&](const auto& be, auto&& apply, std::size_t s) {
        Sample out;
        // ---- 1. Seed v_0 on the host, copy to backend ----
        // Drawn on the host for every backend (same engine, same Gaussian
        // stream, same dznrm2 + zscal normalisation), so both lanes start
        // every sample from bit-identical vectors.
        std::mt19937 rng = sample_engine(base_seed, static_cast<std::uint64_t>(s));
        std::vector<Complex> v0_host = gaussian_vector(local_n, rng);
        // Subspace projection of the stochastic seed (e.g.
        // Lowdin total-spin), then renormalise the same way.
        if (opts.seed_transform) {
            opts.seed_transform(v0_host.data(), local_n);
            if (!(normalize_host(v0_host.data(), local_n) > 0.0)) {
                throw std::runtime_error("ftlm_kernel: zero-norm random start vector for sample " + std::to_string(s)
                                         + " (the seed transform annihilated it -- the "
                                           "targeted subspace has no weight in this block)");
            }
        }

        auto d_v0 = be.make_zero_vector(local_n);
        be.copy_from_host(v0_host.data(), d_v0.get(), local_n);

        // ---- 2. Lanczos: tridiagonal (basis kept only for full reorth) ----
        ed::krylov::LanczosKernelOptions kopts;
        kopts.max_iter = opts.krylov_dim;
        if (opts.breakdown_tol > 0.0) kopts.breakdown_tol = opts.breakdown_tol;
        // Observables need the Krylov basis itself, orthonormal (the Ritz vectors are built from it).
        if (opts.full_reorthogonalization || n_obs > 0) {
            kopts.reorth = ed::krylov::ReorthPolicy::FullCGS2;
            kopts.keep_basis = true;
        } else {
            // FTLM's first-component weights come from the tridiagonal
            // eigenvectors directly, so we only need a faithful (alpha,
            // beta) -- LocalDGKS3 is the cheap canonical reorth policy.
            kopts.reorth = ed::krylov::ReorthPolicy::LocalDGKS3;
            kopts.keep_basis = false;
        }

        {
            auto k = ed::krylov::lanczos_kernel(be, apply, local_n, d_v0.get(), kopts);

            // ---- 3. Diagonalise tridiagonal on host -> ritz + weights ----
            // (the kept basis, if any, is released at the end of this
            // block, before the host-side post-processing).
            if (!k.alpha.empty()) {
                ed::krylov::TridiagEig t = ed::krylov::tridiag_eig(k.alpha, k.beta, k.alpha.size(), /*vectors=*/true);
                for (std::size_t j = 0; j < t.m; ++j)   // a dropped pair keeps its value, without weight
                    if (t.vectors[j * t.m] * t.vectors[j * t.m] < opts.min_weight) t.vectors[j * t.m] = 0.0;
                out.weights = t.weights();
                out.ritz = std::move(t.values);
                if (n_obs > 0) out.Y = std::move(t.vectors);
            }
            // ---- 4. Host-side Boltzmann moments of this sample ----
            if (!out.ritz.empty()) out.mom = detail::sample_moments(out.ritz, out.weights, betas);
            // Observables: phi_t = sum_a c_a v_a with c_a = sum_j g_j Y[j m + a], g_j =
            // e^{-beta_t (e_j - e_r) / 2} <psi_j|r> (<psi_j|r> = Y[j m], real), so that
            // <phi_t|O|phi_t> = sum_ij g_i g_j <psi_i|O|psi_j>, measured on the host by `observe` for
            // kPhiChunk temperatures at a time.
            const std::size_t m = out.ritz.size();
            if (n_obs > 0 && m > 0) {
                if (k.basis.size() < m) {
                    out.ritz.clear();
                } else {
                    constexpr std::size_t kPhiChunk = 4;
                    const std::size_t nT = betas.size(), width = std::min(kPhiChunk, nT);
                    std::vector<const Complex*> V(m);
                    for (std::size_t a = 0; a < m; ++a) V[a] = k.basis[a];
                    auto phi = be.make_zero_vector(local_n);
                    std::vector<std::vector<Complex>> host(width, std::vector<Complex>(local_n));
                    std::vector<Complex> c(m);
                    out.oz.assign(nT, 0.0);
                    out.obs.assign(nT, {});
                    const double e_r = out.mom.e_min;
                    for (std::size_t t0 = 0; t0 < nT; t0 += width) {
                        const std::size_t nc = std::min(width, nT - t0);
                        std::vector<const Complex*> ptrs(nc);
                        for (std::size_t i = 0; i < nc; ++i) {
                            const double beta = betas[t0 + i];
                            std::fill(c.begin(), c.end(), Complex(0, 0));
                            double z = 0.0;
                            for (std::size_t j = 0; j < m; ++j) {
                                const double g = std::exp(-0.5 * beta * (out.ritz[j] - e_r)) * out.Y[j * m];
                                z += g * g;
                                for (std::size_t a = 0; a < m; ++a) c[a] += g * out.Y[j * m + a];
                            }
                            out.oz[t0 + i] = z;
                            be.scale(Complex(0, 0), phi.get(), local_n);
                            be.axpy_many(c.data(), V.data(), m, phi.get(), local_n);
                            be.copy_to_host(phi.get(), host[i].data(), local_n);
                            ptrs[i] = host[i].data();
                        }
                        auto vals = opts.observe(ptrs);
                        if (vals.size() != nc) throw std::logic_error("ftlm_kernel: observe returned the wrong count");
                        for (std::size_t i = 0; i < nc; ++i) out.obs[t0 + i] = std::move(vals[i]);
                    }
                }
            }
        }
        if (out.ritz.empty()) {
            // A failed sample is dropped, not fatal.
            ED_LOG(Warn, "FTLM: sample %zu has no Ritz values; dropped", static_cast<std::size_t>(s));
            return out;
        }
        out.ok = true;
        return out;
    };

    std::vector<Sample> samples(opts.num_samples);
    bool batched = false;
#ifdef WITH_CUDA
    if constexpr (std::is_same_v<Backend, ed::matvec::CudaBackend>) {
        // Up to batch_width samples in lockstep, each on its own thread and backend, sharing
        // every H apply (one multi-vector launch). The observables are measured per sample.
        if (opts.batch_matvec && opts.batch_width > 1 && opts.num_samples > 1) {
            batched = true;
            for (std::size_t s0 = 0; s0 < opts.num_samples; s0 += opts.batch_width) {
                const std::size_t k = std::min(opts.batch_width, opts.num_samples - s0);
                ed::matvec::MatvecBatcher b;
                const auto H = b.wrap(opts.batch_matvec);
                b.run(k, [&](std::size_t i) {
                    const ed::matvec::CudaBackend be;
                    samples[s0 + i] = sample(be, H, s0 + i);
                });
            }
        }
    }
#endif
    if (!batched)
        for (std::size_t s = 0; s < opts.num_samples; ++s) samples[s] = sample(backend, apply_H, s);

    std::vector<detail::SampleMoments> moments;
    moments.reserve(opts.num_samples);
    double ground_state_estimate = std::numeric_limits<double>::infinity();
    // Observable sums over all samples, each sample's (about its own e_min) rescaled to the lowest:
    // the symmetric (low-temperature Lanczos) estimator sum_ij e^{-beta (e_i + e_j) / 2}
    // <r|psi_i><psi_i|O|psi_j><psi_j|r>, exact for the lowest state already at one sample, where the
    // one-sided FTLM form sum_i e^{-beta e_i} <r|psi_i><psi_i|O|r> fluctuates at low T.
    double obs_ref = std::numeric_limits<double>::infinity();
    if (n_obs > 0)
        for (const auto& smp : samples)
            if (smp.ok) obs_ref = std::min(obs_ref, smp.mom.e_min);
    std::vector<double> obs_z(n_obs > 0 ? betas.size() : 0, 0.0);
    std::vector<std::vector<Complex>> obs_num(n_obs, std::vector<Complex>(betas.size(), Complex(0, 0)));
    for (auto& smp : samples) {
        if (!smp.ok) continue;
        if (n_obs > 0)
            for (std::size_t t = 0; t < betas.size(); ++t) {
                const double f = std::exp(-betas[t] * (smp.mom.e_min - obs_ref));
                obs_z[t] += f * smp.oz[t];
                for (std::size_t o = 0; o < n_obs; ++o) obs_num[o][t] += f * smp.obs[t].at(o);
            }
        ground_state_estimate = std::min(ground_state_estimate, smp.mom.e_min);
        moments.push_back(std::move(smp.mom));
    }

    if (moments.empty()) {
        throw std::runtime_error("ftlm_kernel: every sample failed (no Ritz values from any "
                                 "of the "
                                 + std::to_string(opts.num_samples) + " samples)");
    }

    // ---- 5. Jensen-correct sample averaging ----
    FtlmResult out;
    out.curves = detail::combine_samples(moments, static_cast<std::uint64_t>(local_n), betas);
    out.ground_state_estimate = ground_state_estimate;
    for (auto& row : obs_num) {
        for (std::size_t t = 0; t < row.size(); ++t) row[t] /= obs_z[t];
        out.curves.O.push_back(std::move(row));
    }
    return out;
}

}  // namespace ed::thermal

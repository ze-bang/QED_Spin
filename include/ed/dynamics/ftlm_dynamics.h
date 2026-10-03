#pragma once
// =============================================================================
// include/ed/dynamics/ftlm_dynamics.h
//
// Finite-temperature Lanczos (Jaklic-Prelovsek) for a dynamical correlation <A^dag(t) B> between
// two sectors, on any Backend. A and B are rectangular: each maps the source sector (dim_src) to a
// target sector (dim_dst). Per source sector, with R Gaussian samples |r> drawn in the source
// basis, the Ritz states |psi_i> (energies E_i, first components c_i = <psi_i|r>) of a Lanczos run
// on H_src from |r>, and a second Lanczos run on H_dst from B|r> for the resolvent,
//
//   S(omega, T) = (dim_src / R) sum_r sum_i e^{-beta (E_i - E_min)} c_i
//                 sum_j <psi_i|A^dag|chi_j><chi_j|B|r> L_eta(omega - (lambda_j - E_i)),
//   Z(T)      = (dim_src / R) sum_r sum_i e^{-beta (E_i - E_min)} c_i^2,
//
// L_eta a unit Lorentzian. Both are returned UN-normalised (times dim_src, or trace_dim): the
// caller (the dynamics verb) sums S and Z over source sectors and divides.
//
// Every O(dim) step runs on the backend: both Lanczos runs keep their bases in backend memory,
// O is applied to each source Krylov vector there, and the overlaps W = (O V_H)^dag V_S come from
// one GEMM. Only the m_H x m_S overlap matrix and the tridiagonals reach the host. A = B is the
// autocorrelation (real); otherwise S is complex.
//
//   H_src, H_dst, A_apply, B_apply: (in, out, n) callables over backend pointers.
// =============================================================================

#include <ed/krylov/lanczos.h>
#include <ed/krylov/tridiag.h>
#include <ed/matvec/batcher.h>
#include <ed/thermal/sample_seed.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>

namespace ed::observables {

using Complex = std::complex<double>;

/// The smallest sector whose Lanczos runs reorthogonalise locally (ftlm_dynamics_kernel).
inline constexpr std::size_t kLocalReorthMinDim = std::size_t{1} << 16;

/// Parameters for the FTLM cross-irrep kernel.
struct FtlmCrossIrrepOptions {
    std::size_t krylov_dim = 200;
    /// Both Lanczos runs stop at an invariant subspace, beta <= breakdown_tol (energy units;
    /// the engine passes 64 eps s_H). 0: every step runs.
    double breakdown_tol = 0.0;
    std::size_t num_samples = 30;
    double broadening = 0.05;
    /// Base seed: sample s starts from gaussian_vector(dim_src, sample_engine(random_seed, s))
    /// (0 draws one).
    std::uint64_t random_seed = 0;
    /// Applied in place to each (host) random vector before use, e.g. a projection onto one
    /// spin tower; the kernel renormalises the result. The trace then runs over the image of
    /// the transform, whose dimension is `trace_dim` (0: the whole source sector).
    std::function<void(Complex*, std::size_t)> seed_transform;
    std::size_t trace_dim = 0;
    /// Source Ritz pairs whose start weight |<r|psi_i>|^2 lies below this are dropped (0: none): with
    /// the start projected onto a spin tower, the levels outside it carry only roundoff weight
    /// (ed::thermal::FtlmOptions::min_weight).
    double min_weight = 0.0;
    /// Device multi-vector source H (LinearOperator::bind_cuda_multi; each target carries its own,
    /// FtlmDynamicsTarget::batch): on a CUDA run up to `batch_width` samples advance in lockstep
    /// and share each H apply. A and B must then be safe to apply from several threads at once.
    std::function<void(const Complex* const*, Complex* const*, std::size_t, std::size_t)> batch_src;
    std::size_t batch_width = 8;
};

/// One target sector of a source: H there, A and B from the source as rows of it (callables over
/// backend pointers), and on a device the multi-vector H that lets samples advance in lockstep.
struct FtlmDynamicsTarget {
    std::size_t dim = 0;
    std::function<void(const Complex*, Complex*, std::size_t)> H, A, B;
    std::function<void(const Complex* const*, Complex* const*, std::size_t, std::size_t)> batch;
};

/// One source sector's UN-normalised FTLM accumulators (times dim_src, or trace_dim): Z per
/// temperature and S per target and temperature, about the common reference E_min (the lowest
/// weighted source Ritz value over the samples). The caller sums S and Z over source sectors.
struct FtlmDynamicsResult {
    std::map<double, double> Z;
    std::vector<std::map<double, std::vector<Complex>>> S;   ///< per target
    double E_min = 0.0;
    std::size_t samples_done = 0;
};

/// The estimator above for every target of one source at once: each sample's source Lanczos
/// (its Ritz data and Z) is formed once and serves every target -- the targets a probe reaches,
/// over every probe (audit P5-dynamics-06/08). No target: Z alone (a source every probe
/// annihilates still weighs in the partition function).
template <class Backend, class HSrc>
FtlmDynamicsResult ftlm_dynamics_kernel(Backend& be, HSrc&& H_src, std::size_t dim_src,
                                        const std::vector<FtlmDynamicsTarget>& targets,
                                        const std::vector<double>& temperatures, const std::vector<double>& omega,
                                        const FtlmCrossIrrepOptions& opts) {
    if (dim_src == 0) throw std::invalid_argument("ftlm_dynamics_kernel: empty source sector");
    for (const auto& t : targets)
        if (t.dim == 0) throw std::invalid_argument("ftlm_dynamics_kernel: empty target sector");
    if (temperatures.empty() || omega.empty() || opts.num_samples == 0)
        throw std::invalid_argument("ftlm_dynamics_kernel: no temperatures, frequencies or samples");
    constexpr double kInvPi = 0.3183098861837907;
    const std::size_t nW = omega.size(), nT = temperatures.size(), nt = targets.size();
    const double eta = opts.broadening;
    const std::uint64_t base_seed = ed::thermal::resolve_base_seed(opts.random_seed);

    // One sample, reduced against its own reference smin (its lowest weighted source Ritz value):
    // Z_s[T] and, per target, S_s[target][T][w]. Samples are combined below in sample order.
    struct Sample {
        bool ok = false;
        double smin = 0.0;
        std::vector<double> Z;                                   // [T]
        std::vector<std::vector<Complex>> S;                     // [target][T * nW + w]
    };
    auto sample = [&](auto& bk, auto&& Hs, auto&& Hds, std::size_t s) {
        Sample out;
        // Local DGKS3 reorthogonalisation in a large sector much larger than the Krylov depth: there the
        // O(m^2 n) full pass was most of a sample's work and buys nothing measurable (chain24, T = 1,
        // krylov 80: the weight moves by 8e-7 against a seed-to-seed spread of 4e-4; chain12 errors
        // against the dense Lehmann sum agree to 4 digits; jobs 62586014/62586016; audit P5-dynamics-05).
        // Small sectors keep full CGS2: it stops cleanly at an invariant subspace, and it keeps a host
        // and a device run of the same samples equal to roundoff -- without it their roundoff grows to
        // 1e-6..1e-5 (gate 39d45991, grid GPU-vs-CPU dynT cells), harmless next to sampling but it would
        // blind the device-consistency checks.
        auto lanczos = [&](auto&& H, const Complex* v0, std::size_t n) {
            ed::krylov::LanczosKernelOptions lo;
            lo.max_iter = std::min(n, opts.krylov_dim);
            lo.reorth = n > std::max<std::size_t>(4 * opts.krylov_dim, kLocalReorthMinDim)
                            ? ed::krylov::ReorthPolicy::LocalDGKS3
                            : ed::krylov::ReorthPolicy::FullCGS2;
            lo.keep_basis = true;
            if (opts.breakdown_tol > 0.0) lo.breakdown_tol = opts.breakdown_tol;
            auto mv = [&H](const Complex* in, Complex* o, std::size_t nn) { H(in, o, nn); };
            return ed::krylov::lanczos_kernel(bk, mv, n, v0, lo);
        };
        std::mt19937 gen = ed::thermal::sample_engine(base_seed, s);
        std::vector<Complex> r_host = ed::thermal::gaussian_vector(dim_src, gen);
        if (opts.seed_transform) {
            opts.seed_transform(r_host.data(), dim_src);
            double n2 = 0.0;
            for (const auto& z : r_host) n2 += std::norm(z);
            if (n2 < 1e-24) return out;                           // no weight in the image: redraw
            const double inv = 1.0 / std::sqrt(n2);
            for (auto& z : r_host) z *= inv;
        }
        auto r = bk.make_zero_vector(dim_src);
        bk.copy_from_host(r_host.data(), r.get(), dim_src);

        // ---- the source: Ritz values e_i, first components c_i, the reference ---------------
        auto kh = lanczos(Hs, r.get(), dim_src);
        if (kh.alpha.empty() || kh.basis.size() < kh.alpha.size()) return out;
        const std::size_t mH = kh.alpha.size();
        ed::krylov::TridiagEig th = ed::krylov::tridiag_eig(kh.alpha, kh.beta, mH, /*vectors=*/true);
        const std::vector<double> VH = std::move(th.vectors);   // VH[i * mH + a]
        const std::vector<double> ritz = std::move(th.values);
        std::vector<double> c(mH);
        for (std::size_t i = 0; i < mH; ++i) {
            c[i] = VH[i * mH];
            if (c[i] * c[i] < opts.min_weight) c[i] = 0.0;      // a roundoff copy: no weight
        }
        // The reference: the lowest source Ritz value with weight (a dropped copy would underflow the rest).
        out.smin = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < mH; ++i)
            if (c[i] != 0.0) out.smin = std::min(out.smin, ritz[i]);
        if (!std::isfinite(out.smin)) out.smin = *std::min_element(ritz.begin(), ritz.end());
        std::vector<double> wt(nT * mH);                         // e^{-beta (e_i - smin)}
        out.Z.assign(nT, 0.0);
        for (std::size_t it = 0; it < nT; ++it) {
            const double beta = 1.0 / temperatures[it];
            for (std::size_t i = 0; i < mH; ++i) {
                wt[it * mH + i] = std::exp(-beta * (ritz[i] - out.smin));
                out.Z[it] += c[i] * c[i] * wt[it * mH + i];
            }
        }
        out.ok = true;
        out.S.assign(nt, std::vector<Complex>(nT * nW, Complex(0, 0)));

        // ---- each target: the resolvent from B|r>, A on the source basis -------------------
        for (std::size_t tt = 0; tt < nt; ++tt) {
            const FtlmDynamicsTarget& tg = targets[tt];
            const std::size_t dim_dst = tg.dim;
            auto phi = bk.make_zero_vector(dim_dst);
            tg.B(r.get(), phi.get(), dim_dst);
            const double nphi = bk.nrm2(phi.get(), dim_dst);
            // scale-free: unit-vector norm
            if (nphi < 1e-14) continue;                               // B annihilates |r>: no spectral weight here
            bk.scale(Complex(1.0 / nphi, 0.0), phi.get(), dim_dst);
            auto ks = lanczos(Hds[tt], phi.get(), dim_dst);
            if (ks.alpha.empty() || ks.basis.size() < ks.alpha.size()) continue;
            const std::size_t mS = ks.alpha.size();
            ed::krylov::TridiagEig ts = ed::krylov::tridiag_eig(ks.alpha, ks.beta, mS, /*vectors=*/true);
            const std::vector<double> ritzS = std::move(ts.values);
            const std::vector<double> VS = std::move(ts.vectors);   // VS[j * mS + b]

            // W[a + b mH] = <A v_a | w_b>, one row a at a time: A v_a into one target-sized scratch
            // vector, then its overlaps with the target basis (dot_many gives <w_b | A v_a>).
            std::vector<Complex> W(mH * mS);
            {
                auto ov = bk.make_zero_vector(dim_dst);
                std::vector<const Complex*> wb(mS);
                for (std::size_t b = 0; b < mS; ++b) wb[b] = ks.basis[b];
                std::vector<Complex> row(mS);
                for (std::size_t a = 0; a < mH; ++a) {
                    bk.fill_zero(ov.get(), dim_dst);
                    tg.A(kh.basis[a], ov.get(), dim_dst);
                    bk.dot_many(wb.data(), mS, ov.get(), dim_dst, row.data());
                    for (std::size_t b = 0; b < mS; ++b) W[a + b * mH] = std::conj(row[b]);
                }
            }
            std::vector<Complex> Tm(mH * mS, Complex(0, 0));     // Tm[i mS + b] = sum_a VH[i,a] W[a,b]
            for (std::size_t i = 0; i < mH; ++i)
                for (std::size_t b = 0; b < mS; ++b) {
                    Complex acc(0, 0);
                    for (std::size_t a = 0; a < mH; ++a) acc += VH[i * mH + a] * W[a + b * mH];
                    Tm[i * mS + b] = acc;
                }
            // S_s[T][w] = sum_i e^{-beta (e_i - smin)} sum_j w_ij L_eta(w - (lambda_j - e_i)),
            // w_ij = c_i <A psi_i|chi_j> |B r| VS[j,0]: one source Ritz row s_i(w) at a time.
            auto& St = out.S[tt];
#pragma omp parallel
            {
                std::vector<Complex> si(nW), acc(nT * nW, Complex(0, 0));
#pragma omp for schedule(static)
                for (std::int64_t ii = 0; ii < static_cast<std::int64_t>(mH); ++ii) {
                    const std::size_t i = static_cast<std::size_t>(ii);
                    if (c[i] == 0.0) continue;
                    std::fill(si.begin(), si.end(), Complex(0, 0));
                    for (std::size_t j = 0; j < mS; ++j) {
                        Complex obar(0, 0);
                        for (std::size_t b = 0; b < mS; ++b) obar += Tm[i * mS + b] * VS[j * mS + b];
                        const Complex w_ij = c[i] * obar * (nphi * VS[j * mS]);
                        if (std::abs(w_ij) < 1e-300) continue;
                        const double E_ij = ritzS[j] - ritz[i];
                        for (std::size_t iw = 0; iw < nW; ++iw) {
                            const double d = omega[iw] - E_ij;
                            si[iw] += w_ij * ((eta * kInvPi) / (d * d + eta * eta));
                        }
                    }
                    for (std::size_t it = 0; it < nT; ++it) {
                        const double wi = wt[it * mH + i];
                        if (wi < 1e-300) continue;
                        for (std::size_t iw = 0; iw < nW; ++iw) acc[it * nW + iw] += wi * si[iw];
                    }
                }
#pragma omp critical(ftlm_dynamics_rows)
                for (std::size_t q = 0; q < nT * nW; ++q) St[q] += acc[q];
            }
        }
        return out;
    };

    std::vector<Sample> samples(opts.num_samples);
    bool batched = false;
#ifdef WITH_CUDA
    if constexpr (std::is_same_v<std::decay_t<Backend>, ed::matvec::CudaBackend>) {
        // Up to batch_width samples in lockstep, each on its own thread and backend, sharing every
        // source and target H apply (see MatvecBatcher: the waiting calls run grouped by operator);
        // A and B are applied per sample.
        const bool all_batch = std::all_of(targets.begin(), targets.end(), [](const auto& t) { return bool(t.batch); });
        if (opts.batch_src && all_batch && opts.batch_width > 1 && opts.num_samples > 1) {
            batched = true;
            for (std::size_t s0 = 0; s0 < opts.num_samples; s0 += opts.batch_width) {
                const std::size_t k = std::min(opts.batch_width, opts.num_samples - s0);
                ed::matvec::MatvecBatcher b;
                const auto Hs = b.wrap(opts.batch_src);
                std::vector<ed::matvec::MatvecBatcher::Single> Hds;
                for (const auto& t : targets) Hds.push_back(b.wrap(t.batch));
                b.run(k, [&](std::size_t i) {
                    ed::matvec::CudaBackend bk;
                    samples[s0 + i] = sample(bk, Hs, Hds, s0 + i);
                });
            }
        }
    }
#endif
    if (!batched) {
        std::vector<std::function<void(const Complex*, Complex*, std::size_t)>> Hds;
        for (const auto& t : targets) Hds.push_back(t.H);
        for (std::size_t s = 0; s < opts.num_samples; ++s) samples[s] = sample(be, H_src, Hds, s);
    }

    // ---- the samples about the common reference, in sample order ---------------------------
    FtlmDynamicsResult R;
    R.S.assign(nt, {});
    double E_min = std::numeric_limits<double>::infinity();
    for (const Sample& smp : samples)
        if (smp.ok) E_min = std::min(E_min, smp.smin);
    for (std::size_t it = 0; it < nT; ++it) {
        const double T = temperatures[it];
        R.Z[T] = 0.0;
        for (auto& St : R.S) St[T].assign(nW, Complex(0, 0));
    }
    for (const Sample& smp : samples) {
        if (!smp.ok) continue;
        for (std::size_t it = 0; it < nT; ++it) {
            const double T = temperatures[it];
            const double f = std::exp(-(1.0 / T) * (smp.smin - E_min));
            R.Z[T] += f * smp.Z[it];
            for (std::size_t tt = 0; tt < nt; ++tt) {
                auto& St = R.S[tt][T];
                for (std::size_t iw = 0; iw < nW; ++iw) St[iw] += f * smp.S[tt][it * nW + iw];
            }
        }
        R.samples_done++;
    }
    if (R.samples_done > 0) {
        const std::size_t tr = opts.trace_dim ? opts.trace_dim : dim_src;
        const double scale = static_cast<double>(tr) / static_cast<double>(R.samples_done);
        for (double T : temperatures) {
            R.Z[T] *= scale;
            for (auto& St : R.S)
                for (auto& v : St[T]) v *= scale;
        }
    }
    R.E_min = std::isfinite(E_min) ? E_min : 0.0;
    return R;
}

}  // namespace ed::observables

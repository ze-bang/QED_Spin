#pragma once
// =============================================================================
// include/ed/krylov/krylov_schur.h
//
// Backend-templated thick-restart Krylov-Schur (Stewart 2001; Wu & Simon 2000) for the lowest
// eigenpairs of a Hermitian operator: a contiguous basis grown by full CGS2 orthogonalisation,
// restarts that keep the lowest Ritz vectors (one GEMM), exact pairs of an invariant subspace kept
// and the search continued from a fresh deflated start, and a degeneracy probe for levels a Krylov
// space from one vector cannot hold twice. The same body serves every Backend (CPU / CUDA): the
// basis lives in backend memory and is touched only through the backend's BLAS (dot_many,
// axpy_many, gemm, copy, scale); the projected eigenproblem is solved on the host.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/krylov/lanczos.h>
#include <ed/krylov/subspace_policy.h>      // krylov_subspace_dim (shared sizing)
#include <ed/krylov/tridiag.h>
#include <ed/matvec/backend.h>

namespace ed::krylov {

using Complex = std::complex<double>;

struct KrylovSchurOptions {
    std::size_t num_eigs        = 1;
    std::size_t max_iter        = 100;
    // scale-free: a default for C++ callers; the engine passes relative values (numerics.h)
    double      tolerance       = 1e-10;
    bool        compute_vectors = false;
    /// Maximum restart cycles before we give up.
    std::size_t max_restarts    = 30;
    /// Breakdown threshold passed to the per-cycle `lanczos_kernel`.
    // scale-free: a default for C++ callers; the engine passes relative values (numerics.h)
    double      breakdown_tol   = 1e-13;
    /// Memory cap on the per-cycle Krylov subspace, in resident length-N
    /// vectors (the basis held during each restart cycle is the dominant cost).
    /// 0 = no cap. The block lanes set it from LanePolicy<B> (available RAM) so the
    /// footprint is PREDICTABLE: m_max <= max_subspace_vectors regardless of
    /// max_iter. See ed::krylov::krylov_subspace_dim.
    std::uint64_t max_subspace_vectors = 0;
    /// After convergence, look for a level the single-vector restarts skipped (a second copy
    /// of a degenerate eigenvalue) with one extra cycle from a fresh random start.
    bool        probe_degeneracy = true;
};

template <class Scalar>
struct KrylovSchurResultT {
    std::vector<double>                    eigenvalues;
    /// Locked Ritz vectors in backend memory, one per converged
    /// eigenvalue. Only populated when `opts.compute_vectors == true`.
    std::vector<typename ed::matvec::BasicBackend<Scalar>::UniqueVec> eigenvectors;
    std::size_t                            iters_done = 0;
    std::size_t                            restarts   = 0;
    bool                                   converged  = false;
    /// The locked vectors span the whole space: every eigenvalue was found (fewer than
    /// num_eigs when the space is smaller).
    bool                                   exhausted  = false;
};
using KrylovSchurResult = KrylovSchurResultT<Complex>;

/// The kernel's fresh starts by default: unit-variance Gaussian entries, one normal stream across
/// every start of a run.
struct GaussianStart {
    std::normal_distribution<double> nd{0.0, 1.0};
    template <class Scalar>
    void operator()(std::mt19937_64& gen, std::vector<Scalar>& v) {
        for (auto& z : v) z = gaussian_entry<Scalar>(nd, gen);
    }
};

/// Thick-restart Krylov-Schur for the lowest num_eigs eigenpairs of the Hermitian `matvec`, from
/// `seed_local` (backend memory, dimension `local_n`). `fresh(gen, host)` fills a host vector with
/// each fresh start (after an invariant subspace, and the probe's): a caller whose eigenpairs live
/// in an invariant subspace it can draw from (a spin tower) passes its own.
///
/// A cycle grows a contiguous basis V to m vectors, each fully orthogonalised (CGS2) against V and
/// against the pairs already found; the projected matrix is tridiagonal, with an arrowhead after a
/// restart. Its lowest Ritz pairs converge when |beta_m s_m,i| < tolerance. A restart keeps the p
/// lowest Ritz vectors (one GEMM, V <- V S_p) and the residual direction, so each cycle adds m - p
/// matvecs instead of rebuilding from one vector (Stewart 2001; Wu & Simon 2000). An exact invariant
/// subspace (breakdown) yields exact pairs: they are kept, and the search goes on from a fresh start
/// deflated against them; two fresh starts inside the found span mean the space is exhausted.
/// Then the degeneracy probe: a Krylov space from one vector holds one vector per eigenspace, so a
/// fresh start deflated against the found pairs looks for a level the search skipped (a second
/// copy of a degenerate eigenvalue).
template <typename Backend, typename MatvecFn, typename Fresh = GaussianStart>
KrylovSchurResultT<typename Backend::scalar_type>
krylov_schur_kernel(Backend&                             be,
                    MatvecFn&&                           matvec,
                    std::size_t                          local_n,
                    const typename Backend::scalar_type* seed_local,
                    const KrylovSchurOptions&            opts,
                    Fresh                                fresh = {})
{
    using Scalar = typename Backend::scalar_type;
    using UniqueVec = typename ed::matvec::BasicBackend<Scalar>::UniqueVec;

    if (opts.max_iter == 0) {
        throw std::invalid_argument("krylov_schur_kernel: max_iter == 0");
    }
    KrylovSchurResultT<Scalar> R;
    const std::size_t n = local_n;
    if (n == 0) { R.converged = true; R.exhausted = true; return R; }

    const std::size_t k = std::max<std::size_t>(1, opts.num_eigs);
    // The cycle: m basis vectors (within the memory cap and the space), p of them kept per restart.
    const std::size_t m_cap = ed::krylov::krylov_subspace_dim(k, opts.max_iter, static_cast<std::uint64_t>(n),
                                                              opts.max_subspace_vectors);
    const std::size_t p_want = k + std::max<std::size_t>(k / 2, 8);
    const std::size_t m = std::max<std::size_t>(1, std::min<std::size_t>({m_cap, 2 * p_want + 20, n}));
    const std::size_t p_keep = std::min(p_want, m > 1 ? m - 1 : std::size_t{1});

    // n x cols, contiguous, each column first-touched on its own: a column is spread over the NUMA
    // domains by the row partition every kernel uses, as a separate vector would be (one fill of
    // the whole block would put each column on a single domain, and every gather from it there).
    auto columns = [&](std::size_t cols) {
        UniqueVec b{be.allocate(n * cols), typename ed::matvec::BasicBackend<Scalar>::Deleter{&be}};
        for (std::size_t j = 0; j < cols; ++j) be.fill_zero(b.get() + j * n, n);
        return b;
    };
    auto V = columns(m + 1);   // columns 0..m
    auto col = [&](std::size_t j) { return V.get() + j * n; };
    auto w = be.make_zero_vector(n);
    std::vector<UniqueVec> found_vecs;
    std::vector<double>    found_vals;

    // Twice: x -= sum_c <c, x> c over the first `j` columns of V and the found vectors. Returns the
    // coefficients on the columns.
    auto orthogonalize = [&](Scalar* x, std::size_t j) {
        std::vector<const Scalar*> b;
        b.reserve(j + found_vecs.size());
        for (std::size_t i = 0; i < j; ++i) b.push_back(col(i));
        for (const auto& f : found_vecs) b.push_back(f.get());
        std::vector<Scalar> c(b.size(), Scalar(0)), c2(b.size());
        for (int pass = 0; pass < 2 && !b.empty(); ++pass) {
            be.dot_many(b.data(), b.size(), x, n, c2.data());
            for (std::size_t i = 0; i < b.size(); ++i) { c[i] += c2[i]; c2[i] = -c2[i]; }
            be.axpy_many(c2.data(), b.data(), b.size(), x, n);
        }
        c.resize(j);
        return c;
    };
    // out[:, 0..cols) = V[:, 0..mm) S[:, 0..cols), S the projected eigenvectors (column-major, mm x mm).
    auto rotate = [&](const std::vector<double>& S, std::size_t mm, std::size_t cols, Scalar* out) {
        std::vector<Scalar> Sh(mm * cols);
        for (std::size_t c = 0; c < cols; ++c)
            for (std::size_t r = 0; r < mm; ++r) Sh[c * mm + r] = Scalar(S[c * mm + r]);
        auto Sd = be.make_zero_vector(mm * cols);
        be.copy_from_host(Sh.data(), Sd.get(), mm * cols);
        be.gemm('N', 'N', n, cols, mm, Scalar(1), V.get(), n, Sd.get(), mm, Scalar(0), out, n);
    };

    // One search for the lowest `want` pairs from the seed in column 0, at most `budget` cycles:
    // appends what converged to the found pairs. 0: converged (or an invariant subspace was
    // exhausted); 1: out of cycles; 2: the seed lies in the found span; 3: the first cycle's lowest
    // Ritz value is at or above `give_up` (the probe's "nothing skipped below").
    enum Outcome { kConverged = 0, kBudget = 1, kNullSeed = 2, kNothingBelow = 3 };
    auto search = [&](std::size_t want, std::size_t budget, double give_up) -> Outcome {
        orthogonalize(col(0), 0);
        const double nrm = be.nrm2(col(0), n);
        // scale-free: unit-vector norm
        if (nrm < 1e-13) return kNullSeed;
        be.scale(Scalar(1.0 / nrm), col(0), n);
        std::size_t p = 0;
        std::vector<double> theta, arrow;   // the kept Ritz values and their couplings to column p
        for (std::size_t cycle = 0; cycle < budget; ++cycle) {
            std::vector<double> T(m * m, 0.0);
            for (std::size_t i = 0; i < p; ++i) {
                T[i + i * m] = theta[i];
                T[i + p * m] = T[p + i * m] = arrow[i];
            }
            std::size_t mm = m;
            double beta_last = 0.0;
            for (std::size_t j = p; j < m; ++j) {
                matvec(col(j), w.get(), n);
                ++R.iters_done;
                const auto c = orthogonalize(w.get(), j + 1);
                T[j + j * m] = std::real(c[j]);
                const double beta = be.nrm2(w.get(), n);
                if (beta <= opts.breakdown_tol) {   // an invariant subspace: its Ritz pairs are exact
                    mm = j + 1;
                    beta_last = 0.0;
                    break;
                }
                if (j + 1 < m) T[j + (j + 1) * m] = T[(j + 1) + j * m] = beta;
                be.copy(w.get(), col(j + 1), n);
                be.scale(Scalar(1.0 / beta), col(j + 1), n);
                beta_last = beta;
            }
            ++R.restarts;
            std::vector<double> Tm(mm * mm);
            for (std::size_t c = 0; c < mm; ++c)
                for (std::size_t r = 0; r < mm; ++r) Tm[c * mm + r] = T[c * m + r];
            const TridiagEig e = symmetric_eig(std::move(Tm), mm);
            if (cycle == 0 && e.values[0] >= give_up) return kNothingBelow;
            const std::size_t top = std::min(want, mm);
            std::size_t conv = 0;
            while (conv < top && std::abs(beta_last) * std::abs(e.vectors[conv * mm + (mm - 1)]) < opts.tolerance)
                ++conv;
            const bool done = conv == top;     // an invariant subspace (beta 0) converges all of them
            if (done || cycle + 1 == budget) {
                if (conv > 0) {
                    auto X = columns(conv);
                    rotate(e.vectors, mm, conv, X.get());
                    for (std::size_t i = 0; i < conv; ++i) {
                        auto x = be.make_zero_vector(n);
                        be.copy(X.get() + i * n, x.get(), n);
                        found_vals.push_back(e.values[i]);
                        found_vecs.emplace_back(std::move(x));
                    }
                }
                return done ? kConverged : kBudget;
            }
            // Thick restart: the p lowest Ritz vectors, then the residual direction.
            p = std::min(p_keep, mm - 1);
            {
                auto U = columns(p);
                rotate(e.vectors, mm, p, U.get());
                be.copy(U.get(), col(0), n * p);
            }
            be.copy(col(mm), col(p), n);
            theta.assign(e.values.begin(), e.values.begin() + static_cast<std::ptrdiff_t>(p));
            arrow.resize(p);
            for (std::size_t i = 0; i < p; ++i) arrow[i] = beta_last * e.vectors[i * mm + (mm - 1)];
        }
        return kBudget;
    };

    // Fresh starts (`fresh`: Gaussian by default): the way on past an exact invariant subspace (the
    // Ising ring: a few distinct levels, each thousands of times), and the probe's.
    std::mt19937_64 fresh_gen(0xF8E5A7C3ULL);
    std::vector<Scalar> host(n);
    auto fresh_seed = [&](std::mt19937_64& gen) {
        fresh(gen, host);
        be.copy_from_host(host.data(), col(0), n);
    };

    be.copy(seed_local, col(0), n);
    bool exhausted = false;
    std::size_t budget = opts.max_restarts;
    int null_starts = 0;
    bool stalled = false;
    while (found_vals.size() < k && R.restarts < budget) {
        const Outcome o = search(k - found_vals.size(), budget - R.restarts,
                                 std::numeric_limits<double>::infinity());   // never gives up
        if (o == kNullSeed) {
            if (++null_starts >= 2) { exhausted = true; break; }
            fresh_seed(fresh_gen);
            continue;
        }
        null_starts = 0;
        if (o == kBudget) { stalled = true; break; }
        if (found_vals.size() < k) fresh_seed(fresh_gen);
    }
    bool converged = !stalled && found_vals.size() >= k;

    // Degeneracy probe: a fresh start deflated against the found pairs has a generic component on
    // a skipped copy; a Ritz value of it below the highest found level is an upper bound on a level
    // that was skipped. Take it, keep the k lowest, and look again.
    if (!found_vals.empty() && !exhausted && opts.probe_degeneracy) {
        std::mt19937_64 probe_gen(0xDE6E4E7AULL);
        for (std::size_t round = 0; round < k; ++round) {
            const double top = *std::max_element(found_vals.begin(), found_vals.end());
            // scale-free: relative to |top|, floored by the (relative) lock tolerance
            const double gap = std::max(10.0 * opts.tolerance, 1e-8 * std::abs(top));
            fresh_seed(probe_gen);
            const std::size_t before = found_vals.size();
            const Outcome o = search(1, std::max<std::size_t>(opts.max_restarts, 1), top - gap);
            if (o == kNothingBelow || o == kNullSeed) break;
            if (found_vals.size() == before) { converged = false; break; }   // skipped, not recovered
            if (found_vals.back() >= top - gap) {                            // converged above: nothing skipped
                found_vals.pop_back();
                found_vecs.pop_back();
                break;
            }
            while (found_vals.size() > k) {
                const auto hi = std::max_element(found_vals.begin(), found_vals.end()) - found_vals.begin();
                found_vals.erase(found_vals.begin() + hi);
                found_vecs.erase(found_vecs.begin() + hi);
            }
        }
    }

    // Ascending.
    std::vector<std::size_t> order(found_vals.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return found_vals[a] < found_vals[b]; });
    for (std::size_t i : order) R.eigenvalues.push_back(found_vals[i]);
    if (opts.compute_vectors)
        for (std::size_t i : order) R.eigenvectors.emplace_back(std::move(found_vecs[i]));
    R.converged = converged;
    R.exhausted = exhausted;
    return R;
}

}  // namespace ed::krylov

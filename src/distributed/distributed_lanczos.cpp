// =============================================================================
// src/distributed/distributed_lanczos.cpp
//
// Phase 3b #2 implementation. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_lanczos_kernel.h>
#include <ed/parallel/thread_budget.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace ed::distributed {

namespace {

using Complex = std::complex<double>;

// Not constexpr: OpenMPI predefined handles cast through (void*) at runtime.
const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

// ----- rank-local BLAS-level helpers ----------------------------------------
// We keep these inline / non-vectorised for simplicity and correctness; the
// existing `cblas_*` calls expect raw `double*` pointers via ed/core/blas
// and the bit-for-bit reproducibility properties of MPI_SUM forbid us from
// using Kahan-style local reductions anyway. For the bounded-N test regime
// the inner loop is tiny.
inline double local_norm_sq(const Complex* x, std::uint64_t n) {
    double s = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) {
        s += std::norm(x[i]);
    }
    return s;
}

inline Complex local_zdotc(const Complex* x, const Complex* y, std::uint64_t n) {
    Complex s(0.0, 0.0);
    for (std::uint64_t i = 0; i < n; ++i) {
        s += std::conj(x[i]) * y[i];
    }
    return s;
}

inline void local_axpy(Complex a, const Complex* x, Complex* y, std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

inline void local_scal(double a, Complex* x, std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) {
        x[i] *= a;
    }
}

// ----- distributed BLAS-level helpers ---------------------------------------
inline double dist_norm(const Complex* x_local, std::uint64_t n_local,
                        MPI_Comm comm) {
    double local = local_norm_sq(x_local, n_local);
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(global);
}

inline Complex dist_zdotc(const Complex* x_local, const Complex* y_local,
                          std::uint64_t n_local, MPI_Comm comm) {
    Complex local = local_zdotc(x_local, y_local, n_local);
    double buf_in[2]  = {local.real(), local.imag()};
    double buf_out[2] = {0.0, 0.0};
    MPI_Allreduce(buf_in, buf_out, 2, MPI_DOUBLE, MPI_SUM, comm);
    return Complex(buf_out[0], buf_out[1]);
}

// -----------------------------------------------------------------------------
// Phase 8 #6: batched <basis[k]|w> for all k in one MPI_Allreduce.
//
// Classical-Gram-Schmidt reorth is implemented as
//
//     for k: c_k = <V_k | w>;  w -= c_k * V_k
//
// The Allreduce is the expensive part: with MGS we pay ``m`` of them per
// Lanczos iteration (one per basis vector), so over a 200-iter Lanczos that
// is ~20 000 round-trips through the network stack -- and they dominate the
// runtime once N gets large enough that the local SpMV is bandwidth-bound.
//
// CGS2 batches all m dots into a single Allreduce per pass; running two
// passes (the "twice-is-enough" trick) restores the numerical stability we
// lost when going from MGS, at the cost of 2 batched Allreduces total.
//
// Returns the *batched* coefficients on every rank in `c_out` (size = m).
// Callers should follow up with a local axpy loop using these coefficients.
// -----------------------------------------------------------------------------
inline void dist_zdotc_batched(
    const std::vector<std::vector<Complex>>& basis,
    const Complex* w_local,
    std::uint64_t n_local,
    MPI_Comm comm,
    std::vector<Complex>& c_out) {

    const std::size_t m = basis.size();
    c_out.assign(m, Complex(0.0, 0.0));
    if (m == 0) return;

    // Local dots. Use a flat real-pair buffer so we can hand it straight to
    // MPI_Allreduce -- we are not allowed to reduce on std::complex<double>
    // directly because MPI's predefined datatypes are language-level types
    // and the SUM op on MPI_C_DOUBLE_COMPLEX is technically undefined in
    // some MPI 3.x stacks. Two MPI_DOUBLE entries per coefficient is the
    // portable shape.
    std::vector<double> local_buf(2 * m, 0.0);
    for (std::size_t k = 0; k < m; ++k) {
        const Complex c = local_zdotc(basis[k].data(), w_local, n_local);
        local_buf[2 * k]     = c.real();
        local_buf[2 * k + 1] = c.imag();
    }
    std::vector<double> global_buf(2 * m, 0.0);
    MPI_Allreduce(local_buf.data(), global_buf.data(),
                  static_cast<int>(2 * m),
                  MPI_DOUBLE, MPI_SUM, comm);
    for (std::size_t k = 0; k < m; ++k) {
        c_out[k] = Complex(global_buf[2 * k], global_buf[2 * k + 1]);
    }
}

// Generate the global initial vector on rank 0 (deterministic from `seed`),
// scatter slabs to every rank.  Replicates the L2-normalised "random unit
// vector" that lanczos() uses for its v0.
void scatter_initial_vector(const DistributedOperator& op,
                            unsigned long seed,
                            std::vector<Complex>& v_local) {
    const int rank = op.rank();
    const int size = op.comm_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t local_n    = op.local_size();

    v_local.assign(local_n, Complex(0.0, 0.0));

    std::vector<int> sendcounts(size), displs(size);
    int run = 0;
    for (int r = 0; r < size; ++r) {
        std::uint64_t off, n;
        DistributedOperator::balanced_slab(global_dim, r, size, off, n);
        // MPI_Scatterv counts are int; same overflow concern as in
        // distributed_operator.cpp build_comm_pattern_. For the bounded
        // test scope we are well below INT_MAX.
        sendcounts[r] = static_cast<int>(n);
        displs[r]     = run;
        run          += sendcounts[r];
    }

    if (rank == 0) {
        std::vector<Complex> v_global(static_cast<std::size_t>(global_dim));
        std::mt19937_64 gen(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            const double a = nd(gen);
            const double b = nd(gen);
            v_global[i] = Complex(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : v_global) z *= inv;

        MPI_Scatterv(v_global.data(), sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    } else {
        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    }

    // Re-normalise locally + globally for numerical hygiene; no-op if the
    // global vector was already exactly normalised, except for floating
    // point noise from the scatter.
    const double n2 = dist_norm(v_local.data(), local_n, op.comm());
    if (n2 > 0.0) local_scal(1.0 / n2, v_local.data(), local_n);
}

// Solve the (m x m) symmetric tridiagonal eigenproblem with diagonal alpha
// and off-diagonal beta[1..m-1]. Returns sorted ascending eigenvalues.
// Done locally on every rank (cheap; m is at most max_iter ~ a few hundred).
std::vector<double> solve_tridiag(const std::vector<double>& alpha,
                                  const std::vector<double>& beta,
                                  std::size_t m) {
    if (m == 0) return {};
    Eigen::MatrixXd T(m, m);
    T.setZero();
    for (std::size_t i = 0; i < m; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < m) {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(T, Eigen::EigenvaluesOnly);
    std::vector<double> evals(m);
    for (std::size_t i = 0; i < m; ++i) evals[i] = es.eigenvalues()(i);
    std::sort(evals.begin(), evals.end());
    return evals;
}

// Same as solve_tridiag but also returns the squared first-row component of
// each eigenvector |<e_0|psi_k>|^2 -- the FTLM weight. Eigenvalues here are
// returned in Eigen's natural ascending order (not re-sorted, since that
// would shuffle the weights).
void solve_tridiag_with_weights(const std::vector<double>& alpha,
                                const std::vector<double>& beta,
                                std::size_t m,
                                std::vector<double>& evals,
                                std::vector<double>& weights) {
    evals.clear();
    weights.clear();
    if (m == 0) return;
    Eigen::MatrixXd T(m, m);
    T.setZero();
    for (std::size_t i = 0; i < m; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < m) {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    evals.resize(m);
    weights.resize(m);
    const auto& V = es.eigenvectors();  // m x m, columns = eigenvectors
    for (std::size_t k = 0; k < m; ++k) {
        evals[k] = es.eigenvalues()(k);
        const double v0k = V(0, k);
        weights[k] = v0k * v0k;
    }
}

// Same as solve_tridiag_with_weights but also exports the FULL (m x m)
// eigenvector matrix in column-major flat layout (column k starts at
// offset k * m). Caller uses this with the rank-local Krylov basis to
// reconstruct distributed Ritz vectors via psi_k_local = V_local @ U[:,k].
void solve_tridiag_with_eigenvectors(const std::vector<double>& alpha,
                                     const std::vector<double>& beta,
                                     std::size_t m,
                                     std::vector<double>& evals,
                                     std::vector<double>& weights,
                                     std::vector<double>& evecs_colmajor) {
    evals.clear();
    weights.clear();
    evecs_colmajor.clear();
    if (m == 0) return;
    Eigen::MatrixXd T(m, m);
    T.setZero();
    for (std::size_t i = 0; i < m; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < m) {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    evals.resize(m);
    weights.resize(m);
    evecs_colmajor.resize(m * m);
    const auto& V = es.eigenvectors();  // m x m, columns = eigenvectors
    for (std::size_t k = 0; k < m; ++k) {
        evals[k] = es.eigenvalues()(k);
        const double v0k = V(0, k);
        weights[k] = v0k * v0k;
        for (std::size_t j = 0; j < m; ++j) {
            evecs_colmajor[k * m + j] = V(j, k);
        }
    }
}

}  // namespace

DistributedLanczosResult distributed_lanczos(
    const DistributedOperator& op,
    const DistributedLanczosOptions& options) {

    const int rank = op.rank();
    const std::uint64_t local_n = op.local_size();
    const std::uint64_t max_iter = options.max_iter;
    const std::uint64_t exct = std::max<std::uint64_t>(1, options.exct);
    const double tol = options.tol;

    if (max_iter == 0) {
        throw std::invalid_argument("distributed_lanczos: max_iter == 0");
    }

    // Phase 8 #3: dim-aware OMP+BLAS thread cap on the rank-local hot path.
    // We size the budget against the *rank-local* slab dimension, not the
    // global one: each rank only ever touches its own ``local_n`` slice in
    // local_axpy / local_zdotc / local_norm_sq, and it is that loop that
    // pays the OpenBLAS / OMP startup cost. Capping to a sane thread count
    // here is the same trick that gave Phase 6.1 single-rank Lanczos a
    // 2.5x speedup at N=16; on MPI runs it also keeps us from
    // oversubscribing the node when (n_ranks * omp_max_threads) far
    // exceeds the physical core count. Honours ``ED_AUTO_THREADS=0`` for
    // users who prefer to do their own pinning via mpiexec --bind-to.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(local_n));

    // compute_eigenvectors implies we MUST keep the basis around AND it
    // must be numerically orthonormal -- otherwise V_local @ U[:,k] is
    // not a useful Ritz vector. Force full_reorth = true in that case.
    const bool keep_basis = options.full_reorth || options.compute_eigenvectors;

    // Initial vector
    std::vector<Complex> v_curr;
    scatter_initial_vector(op, options.seed, v_curr);

    // Storage for Krylov basis vectors -- each rank holds its slab of every
    // V_j. For full_reorth (or compute_eigenvectors) we keep all of them;
    // otherwise we only keep V_{j-1} and V_j for the three-term recurrence.
    std::vector<std::vector<Complex>> basis;
    if (keep_basis) {
        basis.reserve(max_iter);
        basis.push_back(v_curr);
    }

    std::vector<Complex> v_prev(local_n, Complex(0.0, 0.0));
    std::vector<Complex> w(local_n, Complex(0.0, 0.0));

    std::vector<double> alpha;
    std::vector<double> beta;
    alpha.reserve(max_iter);
    beta.reserve(max_iter + 1);
    beta.push_back(0.0);  // beta[0] is unused (starts the recurrence)

    double prev_smallest = std::numeric_limits<double>::infinity();
    int iters_done = 0;

    for (std::uint64_t j = 0; j < max_iter; ++j) {
        // w = H * v_curr
        op.apply(v_curr.data(), w.data());

        // alpha_j = <v_curr | w>
        Complex alpha_c = dist_zdotc(v_curr.data(), w.data(), local_n, op.comm());
        alpha.push_back(alpha_c.real());

        // w -= alpha_j * v_curr   +   beta_j * v_prev
        local_axpy(Complex(-alpha.back(), 0.0), v_curr.data(), w.data(), local_n);
        if (j > 0) {
            local_axpy(Complex(-beta.back(), 0.0), v_prev.data(), w.data(),
                       local_n);
        }

        // Optional full re-orthogonalization. Triggered by either
        // full_reorth = true OR compute_eigenvectors = true.
        //
        // Phase 8 #6: replace the ``m`` serial MPI_Allreduce calls of MGS
        // with two CGS passes (CGS2), each batching all ``m`` dots into one
        // Allreduce. For full-reorth Lanczos this is the dominant runtime
        // cost at scale (m^2 small Allreduces dominate the local SpMV).
        // Two passes give MGS-equivalent numerical stability ("twice is
        // enough"); we still keep MGS as a fallback for tiny basis sizes
        // where the dominant cost is the local axpy, not the Allreduce.
        if (keep_basis && !basis.empty()) {
            // Heuristic threshold: below 8 prior vectors the batched-dot's
            // 2*m-double Allreduce is barely cheaper than 1-coeff
            // Allreduces and the CGS2 second pass is pure overhead. Above
            // that, CGS2 wins decisively.
            constexpr std::size_t kCgs2Threshold = 8;
            if (basis.size() < kCgs2Threshold) {
                for (std::size_t k = 0; k < basis.size(); ++k) {
                    Complex c = dist_zdotc(basis[k].data(), w.data(), local_n,
                                           op.comm());
                    local_axpy(-c, basis[k].data(), w.data(), local_n);
                }
            } else {
                std::vector<Complex> coeffs;
                for (int pass = 0; pass < 2; ++pass) {
                    dist_zdotc_batched(basis, w.data(), local_n, op.comm(),
                                       coeffs);
                    for (std::size_t k = 0; k < basis.size(); ++k) {
                        if (coeffs[k] != Complex(0.0, 0.0)) {
                            local_axpy(-coeffs[k], basis[k].data(), w.data(),
                                       local_n);
                        }
                    }
                }
            }
        }

        // beta_{j+1} = ||w||_2
        const double b = dist_norm(w.data(), local_n, op.comm());
        beta.push_back(b);

        ++iters_done;

        if (options.verbose && rank == 0) {
            std::cout << "  [dist-lanczos] j=" << j
                      << " alpha=" << alpha.back()
                      << " beta_{j+1}=" << b << std::endl;
        }

        // Convergence / breakdown checks (eat exit even if j+1 < exct so we
        // don't run a useless extra iteration with beta=0).
        if (b < 1e-14) {
            // Krylov subspace exhausted (invariant subspace).
            break;
        }

        if ((j + 1) % 5 == 0 || j + 1 == max_iter) {
            std::vector<double> ev = solve_tridiag(alpha, beta, alpha.size());
            const double smallest = ev.front();
            if (alpha.size() >= exct + 1) {
                if (std::abs(smallest - prev_smallest) < tol) {
                    // Converged.
                    ++j;  // for the loop increment we didn't get
                    (void)j;
                    iters_done = static_cast<int>(alpha.size());
                    if (options.verbose && rank == 0) {
                        std::cout << "  [dist-lanczos] converged at iter "
                                  << alpha.size() << " (smallest="
                                  << smallest << ")" << std::endl;
                    }
                    break;
                }
            }
            prev_smallest = smallest;
        }

        // Advance: v_prev <- v_curr, v_curr <- w / b
        v_prev.swap(v_curr);
        local_scal(1.0 / b, w.data(), local_n);
        v_curr.swap(w);
        // w now holds the previous v_curr -- the next iteration overwrites
        // it via op.apply, so its contents do not matter.

        if (keep_basis) {
            basis.push_back(v_curr);
        }
    }

    // Solve final tridiagonal eigenproblem.
    DistributedLanczosResult result;
    result.iterations = iters_done;

    if (options.compute_eigenvectors) {
        std::vector<double> evals_unsorted, weights_unsorted, evecs_cm;
        solve_tridiag_with_eigenvectors(alpha, beta, alpha.size(),
                                        evals_unsorted, weights_unsorted,
                                        evecs_cm);
        result.tridiag_eigenvalues   = evals_unsorted;
        result.tridiag_weights       = weights_unsorted;
        result.tridiag_eigenvectors  = std::move(evecs_cm);
        // Drop the trailing element of `basis` (which is V_{m}, *one past*
        // the m vectors V_0..V_{m-1} that the eigensolve actually used).
        // After `iters_done` iterations alpha has m entries, basis has
        // `keep_basis ? m : 0` entries pre-loop plus one push per iter
        // -> m + 1 entries when keep_basis. Drop the last so basis size
        // matches the tridiag dimension exactly.
        if (basis.size() > static_cast<std::size_t>(iters_done)) {
            basis.resize(static_cast<std::size_t>(iters_done));
        }
        result.krylov_basis_local = std::move(basis);

        std::vector<double> evals_sorted = evals_unsorted;
        std::sort(evals_sorted.begin(), evals_sorted.end());
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  evals_sorted.size());
        result.eigenvalues.assign(evals_sorted.begin(),
                                  evals_sorted.begin() + n_keep);
    } else if (options.compute_weights) {
        std::vector<double> evals_unsorted, weights_unsorted;
        solve_tridiag_with_weights(alpha, beta, alpha.size(),
                                    evals_unsorted, weights_unsorted);
        result.tridiag_eigenvalues = evals_unsorted;
        result.tridiag_weights     = weights_unsorted;
        std::vector<double> evals_sorted = evals_unsorted;
        std::sort(evals_sorted.begin(), evals_sorted.end());
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  evals_sorted.size());
        result.eigenvalues.assign(evals_sorted.begin(),
                                  evals_sorted.begin() + n_keep);
    } else {
        std::vector<double> all_ev =
            solve_tridiag(alpha, beta, alpha.size());
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  all_ev.size());
        result.eigenvalues.assign(all_ev.begin(),
                                   all_ev.begin() + n_keep);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Phase 3b #6: rank-local Ritz vector reconstruction
// ---------------------------------------------------------------------------
void reconstruct_local_eigenvector(
    const DistributedLanczosResult& result,
    std::size_t k,
    std::vector<std::complex<double>>& psi_k_local) {

    if (result.krylov_basis_local.empty() ||
        result.tridiag_eigenvectors.empty()) {
        throw std::invalid_argument(
            "reconstruct_local_eigenvector: result lacks krylov_basis_local "
            "or tridiag_eigenvectors -- did you set "
            "compute_eigenvectors=true on DistributedLanczosOptions?");
    }
    const std::size_t m = result.tridiag_eigenvalues.size();
    if (m == 0 || result.krylov_basis_local.size() != m) {
        throw std::invalid_argument(
            "reconstruct_local_eigenvector: basis size ("
            + std::to_string(result.krylov_basis_local.size())
            + ") does not match tridiag dim (" + std::to_string(m) + ")");
    }
    if (k >= m) {
        throw std::out_of_range(
            "reconstruct_local_eigenvector: k=" + std::to_string(k)
            + " out of range, m=" + std::to_string(m));
    }
    const std::size_t local_n = result.krylov_basis_local[0].size();
    psi_k_local.assign(local_n, std::complex<double>(0.0, 0.0));
    const double* U_col = &result.tridiag_eigenvectors[k * m];
    for (std::size_t j = 0; j < m; ++j) {
        const double c_j = U_col[j];
        if (c_j == 0.0) continue;
        const auto& V_j = result.krylov_basis_local[j];
        for (std::size_t i = 0; i < local_n; ++i) {
            psi_k_local[i] += c_j * V_j[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3b #6: convenience wrapper
// ---------------------------------------------------------------------------
DistributedEigenpairsResult distributed_lanczos_eigenvectors(
    const DistributedOperator& op,
    const DistributedLanczosOptions& options) {

    DistributedLanczosOptions opts = options;
    opts.compute_eigenvectors = true;
    opts.compute_weights      = true;
    opts.full_reorth          = true;

    DistributedLanczosResult lres = distributed_lanczos(op, opts);

    DistributedEigenpairsResult out;
    out.iterations  = lres.iterations;

    // tridiag_eigenvalues / tridiag_eigenvectors are ordered by Eigen's
    // SelfAdjointEigenSolver, which returns ascending eigenvalues. The
    // sorted-prefix in lres.eigenvalues therefore corresponds 1-to-1 to
    // columns 0..n_keep-1 of the U matrix.
    const std::size_t n_keep = lres.eigenvalues.size();
    out.eigenvalues = lres.eigenvalues;
    out.eigenvectors_local.assign(n_keep, {});
    for (std::size_t k = 0; k < n_keep; ++k) {
        reconstruct_local_eigenvector(lres, k, out.eigenvectors_local[k]);
    }
    return out;
}

// =============================================================================
// Phase 3b #7 stage 3: distributed Lanczos on the symmetry-projected operator.
//
// Builds a rank-major-scattered initial vector that matches the
// `DistributedSymmetryOperator`'s LPT-permuted slab geometry, then dispatches
// to the templated Lanczos kernel.
// =============================================================================
DistributedLanczosResult distributed_lanczos_symmetry(
    const DistributedSymmetryOperator& op,
    const DistributedLanczosOptions& options) {

    const int rank = op.rank();
    const int size = op.comm_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t local_n    = op.local_size();

    // ---------------- Initial vector scatter ---------------------------------
    // Strategy: rank 0 generates a deterministic L2-normalised global random
    // vector in NATURAL orbit ordering, scatters it via per-rank packed buffers
    // permuted into rank-major order using `partition.rank_orbits`, and every
    // rank receives its slab directly into rank-major layout. This matches
    // what `DistributedSymmetryOperator::apply` expects.
    const auto& partition = op.partition();
    std::vector<int> sendcounts(size, 0), displs(size, 0);
    {
        int run = 0;
        for (int r = 0; r < size; ++r) {
            sendcounts[r] = static_cast<int>(partition.rank_orbits[r].size());
            displs[r] = run;
            run += sendcounts[r];
        }
    }

    std::vector<Complex> v_local(local_n, Complex(0.0, 0.0));

    if (rank == 0) {
        std::vector<Complex> v_natural(static_cast<std::size_t>(global_dim));
        std::mt19937_64 gen(options.seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            const double a = nd(gen);
            const double b = nd(gen);
            v_natural[i] = Complex(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : v_natural) z *= inv;

        // Permute into rank-major packed buffer: slot (rank_offsets[r] + k)
        // holds amplitude of orbit `partition.rank_orbits[r][k]`.
        std::vector<Complex> v_rankmajor(
            static_cast<std::size_t>(global_dim));
        for (int r = 0; r < size; ++r) {
            for (std::size_t k = 0; k < partition.rank_orbits[r].size(); ++k) {
                const std::size_t orbit_id = partition.rank_orbits[r][k];
                const std::size_t global_pos = partition.rank_offsets[r] + k;
                v_rankmajor[global_pos] = v_natural[orbit_id];
            }
        }

        MPI_Scatterv(v_rankmajor.data(), sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    } else {
        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    }

    // ---------------- Run kernel --------------------------------------------
    return kernel::distributed_lanczos_kernel(op, std::move(v_local), options);
}

}  // namespace ed::distributed

#endif  // WITH_MPI

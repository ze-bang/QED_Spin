// =============================================================================
// src/distributed/distributed_lanczos.cpp
//
// Phase 3b #2 implementation. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_lanczos.h>

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

    // Initial vector
    std::vector<Complex> v_curr;
    scatter_initial_vector(op, options.seed, v_curr);

    // Storage for Krylov basis vectors -- each rank holds its slab of every
    // V_j. For options.full_reorth = true we keep all of them; otherwise we
    // only keep V_{j-1} and V_j for the three-term recurrence.
    std::vector<std::vector<Complex>> basis;
    if (options.full_reorth) {
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

        // Optional full re-orthogonalization (MGS over all prior basis vectors)
        if (options.full_reorth && !basis.empty()) {
            for (std::size_t k = 0; k < basis.size(); ++k) {
                Complex c = dist_zdotc(basis[k].data(), w.data(), local_n,
                                       op.comm());
                local_axpy(-c, basis[k].data(), w.data(), local_n);
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

        if (options.full_reorth) {
            basis.push_back(v_curr);
        }
    }

    // Solve final tridiagonal eigenproblem.
    std::vector<double> all_ev =
        solve_tridiag(alpha, beta, alpha.size());

    DistributedLanczosResult result;
    result.iterations = iters_done;
    const std::size_t n_keep =
        std::min<std::size_t>(static_cast<std::size_t>(exct), all_ev.size());
    result.eigenvalues.assign(all_ev.begin(), all_ev.begin() + n_keep);
    return result;
}

}  // namespace ed::distributed

#endif  // WITH_MPI

// =============================================================================
// src/distributed/distributed_lanczos.cpp
//
// Phase 3b #2 implementation. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_lanczos_kernel.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

// `local_zdotc`, `local_axpy`, `dist_zdotc`, `dist_zdotc_batched`, and the
// three `solve_tridiag*` helpers used to live here too. They were the
// rank-local hot path of the inline Lanczos body that this TU was
// implementing pre-May-2026. Both call sites — the row-slab body that
// used to be inline below, and the symmetry-projected
// `distributed_lanczos_symmetry` further down — now hand off to
// `kernel::distributed_lanczos_kernel` (in
// `include/ed/distributed/distributed_lanczos_kernel.h`), which carries
// the same primitives in its own anonymous namespace AND delegates the
// algorithm body to `ed::krylov::lanczos_kernel<MpiBackend>` for the
// recurrence / CGS2 reorth / breakdown / convergence steps. None of the
// reorth or convergence primitives are needed here anymore.

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

}  // namespace

// =============================================================================
// `distributed_lanczos(DistributedOperator&, options)` -- row-slab path.
//
// As of May 2026 (day 8 of the Krylov-kernel unification rollout) this is a
// thin front-end onto the templated `kernel::distributed_lanczos_kernel<OpT>`
// in `include/ed/distributed/distributed_lanczos_kernel.h`, which in turn
// delegates to `ed::krylov::lanczos_kernel<MpiBackend>`. The previous
// inline ~200-LOC body of this function (deterministic v0 scatter, three-
// term recurrence, batched-CGS2 reorth, breakdown, relative-Δλ early-exit,
// tridiagonal solve, Ritz-vector storage) was a near-line-for-line
// duplicate of the templated kernel's body; the templated kernel handles
// every one of those concerns. Only the geometry-specific v0 scatter is
// kept here, because the row-slab partition uses
// `DistributedOperator::balanced_slab(...)` and the symmetry-projected
// path uses `OrbitPartition::rank_orbits[r]` (different scatter shape).
// =============================================================================
DistributedLanczosResult distributed_lanczos(
    const DistributedOperator& op,
    const DistributedLanczosOptions& options) {

    if (options.max_iter == 0) {
        throw std::invalid_argument("distributed_lanczos: max_iter == 0");
    }

    // Initial vector: row-slab scatter geometry. Pre-kernel hand-off.
    std::vector<Complex> v_local;
    scatter_initial_vector(op, options.seed, v_local);

    return kernel::distributed_lanczos_kernel(op, std::move(v_local), options);
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

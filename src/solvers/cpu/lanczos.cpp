#include <ed/solvers/lanczos.h>
#include <ed/core/hdf5_io.h>
#include <ed/io/lanczos_basis_buffer.h>
#include <ed/io/lanczos_checkpoint.h>
#include <ed/io/lanczos_reorth.h>
#include <ed/krylov/block_lanczos_kernel.h>
#include <ed/krylov/krylov_schur_kernel.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/parallel/fused_blas1.h>
#include <ed/parallel/numa.h>
#include <ed/parallel/thread_budget.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <memory>

// -----------------------------------------------------------------------------
// RAII-style helper: register an in-memory basis buffer for the lifetime of a
// solver, and silently fall back to on-disk storage if the user forced disk
// mode via ED_LANCZOS_DISK=1. When in memory mode, no filesystem work (mkdir
// /rm -rf) is performed; when in disk mode the legacy path is preserved so
// existing behaviour is unchanged.
// -----------------------------------------------------------------------------
namespace {

class BasisBufferScope {
public:
    BasisBufferScope(const std::string& key, uint64_t N, uint64_t reserve_vectors)
        : key_(key), in_memory_(!lanczos_io::force_disk_storage()) {
        if (in_memory_) {
            lanczos_io::register_basis_buffer(key_, N, reserve_vectors);
        } else {
            // Legacy on-disk path: ensure the directory exists.
            std::error_code ec;
            std::filesystem::create_directories(key_, ec);
        }
    }

    ~BasisBufferScope() {
        if (in_memory_) {
            lanczos_io::release_basis_buffer(key_);
        } else {
            // Legacy on-disk path: best-effort cleanup without shell out.
            std::error_code ec;
            std::filesystem::remove_all(key_, ec);
        }
    }

    bool in_memory() const { return in_memory_; }

    BasisBufferScope(const BasisBufferScope&) = delete;
    BasisBufferScope& operator=(const BasisBufferScope&) = delete;

private:
    std::string key_;
    bool in_memory_;
};

} // anonymous namespace

// Single source of truth for "should the random Krylov seed be real?".
// Default is real-only (zero imag) so the Operator::apply() dispatcher
// can take the real-CSR / apply_real fast path for the entire Krylov
// space when the Hamiltonian is real (audit follow-up). Setting the
// env var ED_LANCZOS_COMPLEX_SEED=1 reverts to a fully complex seed
// (legacy behaviour, useful for testing complex spectra).
inline bool ed_use_complex_lanczos_seed() {
    const char* s = std::getenv("ED_LANCZOS_COMPLEX_SEED");
    return (s && s[0] == '1');
}

// Per-iteration progress prints inside the Lanczos inner loops are useful for
// development but flood stdout in production runs where Lanczos is called
// hundreds of times (FTLM, TPQ, NLCE pipelines). Gate them behind a single
// env var so the default is quiet but the chatter can be re-enabled when
// debugging convergence or breakdown issues.
inline bool ed_lanczos_verbose() {
    static const bool v = []() {
        const char* s = std::getenv("ED_LANCZOS_VERBOSE");
        return (s && s[0] == '1');
    }();
    return v;
}

ComplexVector generateRandomVector(int N, std::mt19937& gen, std::uniform_real_distribution<double>& dist) {
    ComplexVector v(N);

    if (ed_use_complex_lanczos_seed()) {
        for (int i = 0; i < N; i++) {
            v[i] = Complex(dist(gen), dist(gen));
        }
    } else {
        for (int i = 0; i < N; i++) {
            v[i] = Complex(dist(gen), 0.0);
        }
    }

    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return v;
}

ComplexVector generateGaussianRandomVector(int N, std::mt19937& gen) {
    // i.i.d. standard complex Gaussian: real and imag parts ~ N(0, 1), then
    // L2-normalise. This produces an isotropic random vector on the complex
    // unit sphere and is the standard finite-T trace-estimator distribution
    // (Jaklic-Prelovsek FTLM, Hutchinson, etc.). Variance bounds and isotropy
    // properties differ from normalised uniform-cube sampling.
    std::normal_distribution<double> ndist(0.0, 1.0);
    ComplexVector v(N);
    for (int i = 0; i < N; i++) {
        v[i] = Complex(ndist(gen), ndist(gen));
    }
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0 / norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);
    return v;
}

// generateOrthogonalVector (random vector Gram-Schmidt-orthogonalized
// against a provided set) was deleted in the debt-cleanup sweep
// (Jul 2026): zero callers.

// Helper function to refine a single eigenvector with CG
// refine_eigenvector_with_cg and refine_degenerate_eigenvectors were
// retired in the minimalist-architecture rev (May 2026): no external
// caller and the only internal user, orthogonalize_degenerate_subspace,
// was deleted alongside them. The refinement step is unnecessary because
// the Lanczos / Block-Lanczos kernels already produce orthonormal Ritz
// vectors and the workflows that need degenerate-cluster handling go
// through Krylov-Schur. To re-introduce, copy the CG / projected-power
// implementations from git history.


ComplexVector read_basis_vector(const std::string& temp_dir, uint64_t index, uint64_t N) {
    // Fast path: in-memory buffer. This is the normal case once a solver has
    // registered its temp_dir via BasisBufferScope.
    ComplexVector vec;
    if (lanczos_io::get_basis_vector(temp_dir, index, vec)) {
        return vec;
    }

    // Fallback: legacy on-disk storage (used when ED_LANCZOS_DISK=1 or when
    // a solver has not yet been ported to register a buffer).
    vec.assign(N, Complex(0.0, 0.0));
    std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".dat";
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << filename << " for reading" << std::endl;
        return vec;
    }
    infile.read(reinterpret_cast<char*>(vec.data()), N * sizeof(Complex));
    return vec;
}

// Helper function to write a basis vector to file (or to the in-memory buffer
// registered for `temp_dir`).
bool write_basis_vector(const std::string& temp_dir, uint64_t index, const ComplexVector& vec, uint64_t N) {
    // Fast path: in-memory buffer. Most call sites append in order
    // (index == size), but restart algorithms (Krylov-Schur, IRL, thick
    // restart) overwrite indices in place after a basis rotation.
    if (lanczos_io::has_basis_buffer(temp_dir)) {
        uint64_t sz = lanczos_io::basis_buffer_size(temp_dir);
        if (index == sz) {
            if (lanczos_io::append_basis_vector(temp_dir, vec)) {
                return true;
            }
        } else if (index < sz) {
            if (lanczos_io::set_basis_vector(temp_dir, index, vec)) {
                return true;
            }
        }
        // Gap in indices: fall through to disk path to preserve correctness.
    }

    std::string filename = temp_dir + "/basis_" + std::to_string(index) + ".dat";
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return false;
    }
    outfile.write(reinterpret_cast<const char*>(vec.data()), N * sizeof(Complex));
    outfile.close();
    return true;
}

// Diagonalize tridiagonal matrix and extract Ritz values and weights
void diagonalize_tridiagonal_ritz(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    std::vector<double>& ritz_values,
    std::vector<double>& weights,
    std::vector<double>* evecs
) {
    uint64_t m = alpha.size();
    
    // Prepare diagonal and off-diagonal arrays for LAPACK
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    // Allocate eigenvector storage
    std::vector<double> evecs_local;
    double* evecs_ptr = nullptr;
    
    if (evecs != nullptr) {
        evecs->resize(m * m);
        evecs_ptr = evecs->data();
    } else {
        evecs_local.resize(m * m);
        evecs_ptr = evecs_local.data();
    }
    
    // Diagonalize
    uint64_t info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, 
                                    diag.data(), offdiag.data(), 
                                    evecs_ptr, m);
    
    if (info != 0) {
        std::cerr << "LAPACKE_dstevd failed in diagonalize_tridiagonal_ritz with error code " << info << std::endl;
        ritz_values.clear();
        weights.clear();
        return;
    }
    
    // Extract Ritz values (eigenvalues are now in diag, sorted)
    ritz_values.resize(m);
    std::copy(diag.begin(), diag.end(), ritz_values.begin());
    
    // Extract weights: squared first component of each eigenvector
    weights.resize(m);
    for (int i = 0; i < m; i++) {
        // First component of eigenvector i (column-major: evecs[0 + i*m])
        double first_component = evecs_ptr[i * m];  // First row, column i
        weights[i] = first_component * first_component;
    }
}

// Build Lanczos tridiagonal with optional basis storage
//
// Krylov-kernel unification (Phase A, May 2026): when ``full_reorth ==
// true`` we delegate to the single unified ``ed::krylov::lanczos_kernel``
// so the canonical CPU FTLM / LTLM / Lanczos all benefit from the
// batched CGS2 reorth (M Allreduces -> 1 in the future MPI path, and a
// single OMP pass per CGS2 step instead of M serial dot/axpy round-trips
// on CPU). The legacy three-vector / periodic-reorth branches below
// stay in place for callers who do not want full reorth.
int build_lanczos_tridiagonal_with_basis(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v0,
    uint64_t N,
    uint64_t max_iter,
    double tol,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& alpha,
    std::vector<double>& beta,
    std::vector<ComplexVector>* basis_vectors
) {
    // ------------------------------------------------------------------
    // Fast path: full reorth + basis requested goes through the new
    // unified kernel. Conditions for the fast path:
    //   * full_reorth is on (we want CGS2 anyway), and
    //   * caller asked for the basis to be retained (otherwise reorth
    //     is impossible regardless of which kernel we use).
    // ------------------------------------------------------------------
    if (full_reorth && basis_vectors != nullptr) {
        ed::krylov::LanczosKernelOptions opts;
        opts.max_iter     = static_cast<std::size_t>(std::min<uint64_t>(N, max_iter));
        opts.reorth       = ed::krylov::ReorthPolicy::FullCGS2;
        opts.keep_basis   = true;
        // `tol` from the legacy ABI is the Ritz convergence threshold,
        // which the kernel routes through `opts.convergence_check`.
        // The legacy MGS body never actually checked Ritz convergence
        // (it only broke on ||w|| < tol — which CGS2 handles via
        // `breakdown_tol`); preserve that "run to max_iter" behaviour
        // by NOT setting convergence_check here.
        (void)tol;

        const auto& be = ed::matvec::default_cpu_backend();

        auto matvec = [&H](const Complex* in, Complex* out, std::size_t n) {
            H(in, out, static_cast<int>(n));
        };

        auto result = ed::krylov::lanczos_kernel(
            be, matvec,
            static_cast<std::size_t>(N),
            v0.data(),
            opts);

        // Translate the result back to the legacy ABI:
        //   alpha / beta returned by reference,
        //   basis_vectors as std::vector<ComplexVector>.
        //
        // Wave C4 / B5 (May 2026): pool-aware. Resize the destination
        // to ``result.basis.size()`` instead of clear+reserve+
        // emplace_back -- this preserves existing inner ComplexVector
        // heap allocations when the caller re-uses ``basis_vectors``
        // across calls (e.g. across FTLM samples or cross-irrep
        // pairs). When the destination is shorter than needed we
        // append fresh entries; when it's longer we drop the tail.
        alpha = std::move(result.alpha);
        beta  = std::move(result.beta);
        const std::size_t need = result.basis.size();
        if (basis_vectors->size() > need) {
            basis_vectors->resize(need);
        }
        basis_vectors->reserve(need);
        const std::size_t reuse = std::min(basis_vectors->size(), need);
        for (std::size_t i = 0; i < reuse; ++i) {
            (*basis_vectors)[i].resize(static_cast<std::size_t>(N));
            std::memcpy((*basis_vectors)[i].data(), result.basis[i].get(),
                        static_cast<std::size_t>(N) * sizeof(Complex));
        }
        for (std::size_t i = reuse; i < need; ++i) {
            ComplexVector v(static_cast<std::size_t>(N));
            std::memcpy(v.data(), result.basis[i].get(),
                        static_cast<std::size_t>(N) * sizeof(Complex));
            basis_vectors->emplace_back(std::move(v));
        }
        return static_cast<int>(alpha.size());
    }

    // ------------------------------------------------------------------
    // Non-full-reorth / no-basis path. Krylov-unification (Jun 2026): this
    // used to be a hand-rolled three-term recurrence. It now routes through
    // the SAME `ed::krylov::lanczos_kernel<CpuBackend>` as the fast path
    // above, mapping the legacy ABI flags onto a `ReorthPolicy`:
    //
    //   * full_reorth && basis==nullptr  -> None  (reorth impossible without
    //         a stored basis; legacy warned once and ran the bare recurrence)
    //   * !full_reorth && reorth_freq>0 && basis!=nullptr -> PeriodicCGS2
    //         (the legacy threshold-MGS periodic reorth, upgraded to CGS2 to
    //          match the fast path's numerics)
    //   * otherwise                       -> None  (pure three-term)
    //
    // The legacy `tol` is a Ritz-convergence parameter that the old body
    // (incorrectly) reused as a ||w||<tol breakdown threshold; we preserve
    // that exact termination by routing `tol` to `breakdown_tol` here. (The
    // fast path above leaves breakdown at its ~exact-zero default, matching
    // its own historical "run to max_iter" behaviour.)
    // ------------------------------------------------------------------
    {
        using ed::krylov::ReorthPolicy;
        ed::krylov::LanczosKernelOptions opts;
        opts.max_iter   = static_cast<std::size_t>(std::min<uint64_t>(N, max_iter));
        opts.keep_basis = (basis_vectors != nullptr);
        opts.breakdown_tol = tol;

        if (full_reorth) {
            // basis_vectors == nullptr here (the full_reorth+basis case took
            // the fast path above and returned).
            std::cerr << "Warning: full_reorthogonalization requested but "
                      << "basis_vectors == nullptr. Reorthogonalization "
                      << "will be silently skipped — eigenvalues may have "
                      << "spurious duplicates." << std::endl;
            opts.reorth = ReorthPolicy::None;
        } else if (reorth_freq > 0 && basis_vectors != nullptr) {
            opts.reorth      = ReorthPolicy::PeriodicCGS2;
            opts.reorth_freq = static_cast<std::size_t>(reorth_freq);
        } else {
            opts.reorth = ReorthPolicy::None;
        }

        const auto& be = ed::matvec::default_cpu_backend();
        auto matvec = [&H](const Complex* in, Complex* out, std::size_t n) {
            H(in, out, static_cast<int>(n));
        };
        auto result = ed::krylov::lanczos_kernel(
            be, matvec, static_cast<std::size_t>(N), v0.data(), opts);

        alpha = std::move(result.alpha);
        beta  = std::move(result.beta);

        if (basis_vectors != nullptr) {
            // Pool-aware translate-back (mirrors the fast path): reuse the
            // overlapping prefix's heap allocations, append the remainder.
            const std::size_t need = result.basis.size();
            if (basis_vectors->size() > need) basis_vectors->resize(need);
            basis_vectors->reserve(need);
            const std::size_t reuse = std::min(basis_vectors->size(), need);
            for (std::size_t i = 0; i < reuse; ++i) {
                (*basis_vectors)[i].resize(static_cast<std::size_t>(N));
                std::memcpy((*basis_vectors)[i].data(), result.basis[i].get(),
                            static_cast<std::size_t>(N) * sizeof(Complex));
            }
            for (std::size_t i = reuse; i < need; ++i) {
                ComplexVector v(static_cast<std::size_t>(N));
                std::memcpy(v.data(), result.basis[i].get(),
                            static_cast<std::size_t>(N) * sizeof(Complex));
                basis_vectors->emplace_back(std::move(v));
            }
        }
        return static_cast<int>(alpha.size());
    }
}

// Helper function to solve tridiagonal eigenvalue problem
int solve_tridiagonal_matrix(const std::vector<double>& alpha, const std::vector<double>& beta, 
                            uint64_t m, uint64_t exct, std::vector<double>& eigenvalues, 
                            const std::string& temp_dir, const std::string& evec_dir, 
                            bool eigenvectors, uint64_t N) {
    // Save only the first exct eigenvalues, or all of them if m < exct
    uint64_t n_eigenvalues = std::min(exct, m);
    
    // Allocate arrays for LAPACKE
    std::vector<double> diag = alpha;    // Copy of diagonal elements
    std::vector<double> offdiag(m-1);    // Off-diagonal elements
    
    for (int i = 0; i < m-1; i++) {
        offdiag[i] = beta[i+1];
    }
    
    uint64_t info;
    
    if (eigenvectors) {
        // Choose between dstemr (MRRR, range='I') and dstevd (D&C, all evals).
        //
        // dstemr with range='I' computes only the lowest n_eigenvalues
        // eigenpairs in O(m * n_eigenvalues) time. dstevd computes all m
        // eigenpairs in O(m^3) time and we discard everything past
        // n_eigenvalues. For typical Lanczos runs (m ~ 200, n_eigenvalues
        // ~ 10-50), dstemr is 5-20x faster.
        //
        // For very small m or n_eigenvalues / m close to 1, dstevd's
        // tighter constants win; switch over at the empirical 50% threshold.
        // dstemr also benefits from MRRR's O(n) per-eigenvector storage.
        const bool use_dstemr = (m >= 32) && (n_eigenvalues * 2 < m);

        // dstemr writes the requested eigenpairs to (W, Z); dstevd overwrites
        // diag with eigenvalues and writes Z densely. Allocate the smaller
        // m * n_eigenvalues block when using dstemr to also save memory.
        const lapack_int ldz = static_cast<lapack_int>(m);
        std::vector<double> evecs;
        std::vector<double> w_only;
        std::vector<lapack_int> isuppz;
        lapack_int m_found = 0;
        lapack_logical tryrac = 1;  // try high accuracy first

        if (use_dstemr) {
            // dstemr requires a fresh copy of d/e because it overwrites them.
            // It also requires e to have m elements (not m-1) -- the trailing
            // entry is workspace.
            std::vector<double> d_copy = diag;
            std::vector<double> e_copy(m);
            for (int i = 0; i < (int)m - 1; ++i) e_copy[i] = offdiag[i];
            e_copy[m - 1] = 0.0;

            evecs.assign(static_cast<size_t>(m) * n_eigenvalues, 0.0);
            w_only.assign(m, 0.0);
            isuppz.assign(2 * std::max<size_t>(1, n_eigenvalues), 0);

            info = LAPACKE_dstemr(LAPACK_COL_MAJOR, 'V', 'I',
                                  static_cast<lapack_int>(m),
                                  d_copy.data(), e_copy.data(),
                                  /*vl=*/0.0, /*vu=*/0.0,
                                  /*il=*/1, /*iu=*/static_cast<lapack_int>(n_eigenvalues),
                                  &m_found, w_only.data(),
                                  evecs.data(), ldz,
                                  static_cast<lapack_int>(n_eigenvalues),
                                  isuppz.data(), &tryrac);
            if (info == 0) {
                // Replace the prefix of diag with the n_eigenvalues smallest
                // (already sorted ascending by dstemr).
                for (size_t k = 0; k < n_eigenvalues; ++k) diag[k] = w_only[k];
            }
        } else {
            evecs.assign(static_cast<size_t>(m) * m, 0.0);
            info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(),
                                   offdiag.data(), evecs.data(), m);
        }
        
        if (info != 0) {
            std::cerr << (use_dstemr ? "LAPACKE_dstemr" : "LAPACKE_dstevd")
                      << " failed with error code " << info << std::endl;
            return info;
        }
        
        std::cout << "Transforming eigenvectors..." << std::endl;

        std::vector<ComplexVector> full_vectors(n_eigenvalues, ComplexVector(N, Complex(0.0, 0.0)));
        std::vector<ComplexVector> compensation(n_eigenvalues, ComplexVector(N, Complex(0.0, 0.0)));

        // ldz_eff = m for both paths (dstemr ldz is the full m, even though
        // only the first n_eigenvalues columns are meaningful).
        const int ldz_eff = static_cast<int>(m);

        for (int j = 0; j < m; j++) {
            ComplexVector basis_j = read_basis_vector(temp_dir, j, N);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_eigenvalues; i++) {
                double coef = evecs[j + i * ldz_eff];
                ComplexVector& full_vector = full_vectors[i];
                ComplexVector& comp_vec = compensation[i];

                for (int k = 0; k < N; k++) {
                    Complex contrib = basis_j[k] * coef;
                    Complex y = contrib - comp_vec[k];
                    Complex t = full_vector[k] + y;
                    comp_vec[k] = (t - full_vector[k]) - y;
                    full_vector[k] = t;
                }
            }
        }

        for (int i = 0; i < n_eigenvalues; i++) {
            ComplexVector& full_vector = full_vectors[i];

            double norm = cblas_dznrm2(N, full_vector.data(), 1);
            if (norm < 1e-14) {
                std::cerr << "Warning: Eigenvector " << i << " has very small norm: " << norm << std::endl;
                continue;
            }

            Complex scale = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale, full_vector.data(), 1);

            if (i > 0 && i < 10) {
                std::string prev_file = evec_dir + "/eigenvector_" + std::to_string(i-1) + ".dat";
                std::ifstream prev_infile(prev_file, std::ios::binary);
                if (prev_infile) {
                    ComplexVector prev_vec(N);
                    prev_infile.read(reinterpret_cast<char*>(prev_vec.data()), N * sizeof(Complex));
                    prev_infile.close();

                    Complex overlap;
                    cblas_zdotc_sub(N, prev_vec.data(), 1, full_vector.data(), 1, &overlap);

                    if (std::abs(overlap) > 1e-10) {
                        std::cerr << "Warning: Eigenvectors " << i-1 << " and " << i
                                  << " have overlap " << std::abs(overlap) << std::endl;

                        Complex neg_overlap = -overlap;
                        cblas_zaxpy(N, &neg_overlap, prev_vec.data(), 1, full_vector.data(), 1);

                        norm = cblas_dznrm2(N, full_vector.data(), 1);
                        scale = Complex(1.0/norm, 0.0);
                        cblas_zscal(N, &scale, full_vector.data(), 1);
                    }
                }
            }

            // Save eigenvector using HDF5 in main output directory
            // (unified ed_results.h5). Skip when no output_dir was set,
            // or when the caller pinned ``/dev/null`` to disable I/O --
            // see the matching guard on the eigenvalues dump below.
            if (!evec_dir.empty() && evec_dir != "/dev/null") {
                try {
                    std::string hdf5_file = HDF5IO::createOrOpenFile(evec_dir);
                    HDF5IO::saveEigenvector(hdf5_file, i, full_vector);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to save eigenvector " << i << " to HDF5: " << e.what() << std::endl;
                }
            }
        }
        
        std::cout << "Saved " << n_eigenvalues << " eigenvectors" << std::endl;

    } else {
        // Eigenvalues only. Same selection as above: dstemr (range='I') is
        // O(m * n_eigenvalues); dstevd is O(m^2). dstemr wins handily for
        // typical Lanczos parameters.
        const bool use_dstemr = (m >= 32) && (n_eigenvalues * 2 < m);
        if (use_dstemr) {
            std::vector<double> d_copy = diag;
            std::vector<double> e_copy(m);
            for (int i = 0; i < (int)m - 1; ++i) e_copy[i] = offdiag[i];
            e_copy[m - 1] = 0.0;
            std::vector<double> w_only(m, 0.0);
            std::vector<lapack_int> isuppz(2 * std::max<size_t>(1, n_eigenvalues), 0);
            lapack_int m_found = 0;
            lapack_logical tryrac = 1;
            info = LAPACKE_dstemr(LAPACK_COL_MAJOR, 'N', 'I',
                                  static_cast<lapack_int>(m),
                                  d_copy.data(), e_copy.data(),
                                  /*vl=*/0.0, /*vu=*/0.0,
                                  /*il=*/1, /*iu=*/static_cast<lapack_int>(n_eigenvalues),
                                  &m_found, w_only.data(),
                                  /*z=*/nullptr, /*ldz=*/static_cast<lapack_int>(m),
                                  /*nzc=*/0, isuppz.data(), &tryrac);
            if (info == 0) {
                for (size_t k = 0; k < n_eigenvalues; ++k) diag[k] = w_only[k];
            }
        } else {
            info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m, diag.data(),
                                   offdiag.data(), nullptr, m);
        }
        
        if (info != 0) {
            std::cerr << (use_dstemr ? "LAPACKE_dstemr" : "LAPACKE_dstevd")
                      << " failed with error code " << info << std::endl;
            return info;
        }
    }
    
    // Copy eigenvalues
    eigenvalues.resize(n_eigenvalues);
    std::copy(diag.begin(), diag.begin() + n_eigenvalues, eigenvalues.begin());

    // Save eigenvalues using HDF5 in main output directory (unified ed_results.h5).
    // Convention: ``evec_dir == "/dev/null"`` OR ``evec_dir.empty()``
    // disables the HDF5 dump entirely. Useful for benchmarks that don't
    // want disk I/O in the timed loop and for tests that don't care
    // about persistence (no output_dir set => no per-test scratch race).
    if (!evec_dir.empty() && evec_dir != "/dev/null") {
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(evec_dir);
            HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
            std::cout << "Lanczos: Saved " << n_eigenvalues << " eigenvalues to HDF5" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
        }
    }

    return info;
}

// =============================================================================
// `lanczos` -- minimalist orchestrator over `ed::krylov::lanczos_kernel<CpuBackend>`
// =============================================================================
//
// Phase 2.1 of the Minimalist ED Collapse (May 2026): the hand-rolled
// three-term recurrence that lived here previously has been retired in
// favour of the unified kernel. The kernel now exposes:
//
//   * `ReorthPolicy::LocalDGKS3` + `local_ring_size` for the legacy
//     3-vector ring-buffer reorth this code path historically used.
//   * `LanczosResumeState` + `LanczosKernelOptions::resume_state` to
//     adopt an alpha/beta/v_curr/v_prev snapshot from disk and continue
//     from iteration `j_start`.
//   * `LanczosKernelOptions::on_step` (added in Phase 4.1 of the
//     Krylov-unification gap-fill) for the per-iteration callback that
//     this orchestrator uses to (a) write basis vectors to disk for
//     eigenvector reconstruction and (b) periodically checkpoint the
//     Krylov state.
//
// The body collapses to:
//
//   1. Build v0 (random or restored from checkpoint).
//   2. Build a `LanczosResumeState` if `ED_LANCZOS_RESUME=1`.
//   3. Configure the kernel options (LocalDGKS3, convergence_check,
//      on_step for basis I/O + checkpointing).
//   4. Call `lanczos_kernel<CpuBackend>` once.
//   5. Solve the small tridiagonal problem and (optionally) reconstruct
//      eigenvectors from the on-disk basis.
//
// Net: ~120 LOC orchestrator + helpers, replacing ~450 LOC of inline
// recurrence + reorth + checkpointing + convergence-checking. Numerical
// behaviour is identical -- the kernel's LocalDGKS3 implementation
// matches the legacy `recent_vectors` ring slot-for-slot (head/count
// indexing, sqrt(eps) threshold, threshold-gated axpy).
// =============================================================================

namespace {

// Bridge: lanczos_io::LanczosCheckpoint (host-side ComplexVector payload)
//   -> ed::krylov::LanczosResumeState (backend UniqueVec payload).
//
// Allocates the backend vectors via `be` and copies the checkpoint
// contents into them. The kernel takes ownership of the unique_ptrs on
// entry; the caller doesn't need to keep the state alive after the
// kernel call.
ed::krylov::LanczosResumeState
to_resume_state(const ed::matvec::CpuBackend& be,
                lanczos_io::LanczosCheckpoint& cp) {
    using namespace ed::matvec;
    ed::krylov::LanczosResumeState s;
    s.j_start = cp.iteration;
    s.alpha   = std::move(cp.alpha);
    s.beta    = std::move(cp.beta);

    const std::size_t n = cp.N;
    s.v_curr = be.make_zero_vector(n);
    s.v_prev = be.make_zero_vector(n);
    be.copy_from_host(cp.v_current.data(), s.v_curr.get(), n);
    be.copy_from_host(cp.v_prev.data(),    s.v_prev.get(), n);

    s.ring_vectors.reserve(cp.ring_vectors.size());
    for (auto& vec : cp.ring_vectors) {
        auto u = be.make_zero_vector(n);
        be.copy_from_host(vec.data(), u.get(), n);
        s.ring_vectors.emplace_back(std::move(u));
    }
    return s;
}

}  // namespace

void lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter, uint64_t exct,
             double tol, std::vector<double>& eigenvalues, std::string dir,
             bool eigenvectors) {

    // Phase 6 #2: dim-aware OMP+BLAS thread cap. Without this the default
    // 16-thread team turns the SpMV into 18 ms/iter at N=16 (dim=65k); the
    // optimum is ~4-8 threads at 1 ms/iter.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    using ed::matvec::CpuBackend;
    using ed::matvec::default_cpu_backend;
    using namespace ed::krylov;

    auto& backend = default_cpu_backend();

    // ===== Krylov-state checkpoint / restart =====
    // Off by default (zero overhead). Activated by ED_LANCZOS_CHECKPOINT_DIR;
    // resume requested by ED_LANCZOS_RESUME=1.
    const bool        ckpt_resume        = lanczos_io::checkpoint_resume_requested();
    const bool        ckpt_write_enabled = lanczos_io::checkpoint_enabled();
    const std::string ckpt_dir           = lanczos_io::checkpoint_dir();
    const uint64_t    ckpt_interval      = lanczos_io::checkpoint_interval();

    // Eigenvector reconstruction reads basis vectors v_0..v_{m-1} from
    // temp_dir. A resumed run does NOT have the early basis vectors; only
    // the most recent two (v_prev, v_current) plus the ring buffer survive.
    // Fail loud now rather than silently producing garbage later.
    if (ckpt_resume && eigenvectors) {
        throw std::runtime_error(
            "Lanczos resume currently supports eigenvalue-only mode "
            "(eigenvectors=false). Re-run without ED_LANCZOS_RESUME for "
            "eigenvector reconstruction, or remove the checkpoint to start "
            "fresh.");
    }

    lanczos_io::LanczosCheckpoint loaded_cp;
    if (ckpt_resume) {
        loaded_cp = lanczos_io::read_lanczos_checkpoint(ckpt_dir);
        if (loaded_cp.N != N) {
            throw std::runtime_error(
                "Lanczos resume: checkpoint dim N=" +
                std::to_string(loaded_cp.N) + " != requested N=" +
                std::to_string(N));
        }
    }

    // Initialize random starting vector (complex- or real-seed per env).
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v0_host(N);
    const bool complex_seed = ed_use_complex_lanczos_seed();
    if (ckpt_resume) {
        if (!loaded_cp.rng_state_text.empty()) {
            lanczos_io::restore_mt19937_state(gen, loaded_cp.rng_state_text);
        }
        // v0 ignored by the kernel on a resume; we pass the loaded
        // v_current via the resume state below.
    } else if (complex_seed) {
        for (uint64_t i = 0; i < N; i++) v0_host[i] = Complex(dist(gen), dist(gen));
    } else {
        for (uint64_t i = 0; i < N; i++) v0_host[i] = Complex(dist(gen), 0.0);
    }
    
    // Storage for basis vectors (RAM by default, disk via ED_LANCZOS_DISK=1).
    // Only allocated when eigenvectors are requested; the kernel itself
    // does not retain a basis under LocalDGKS3 (the 3-vector ring is
    // internal), so the on-disk basis is the sole channel for eigenvector
    // reconstruction.
    std::string temp_dir = (dir.empty() ? "./lanczos_basis_vectors" : dir + "/lanczos_basis_vectors");
    max_iter = std::min(N, max_iter);
    std::unique_ptr<BasisBufferScope> basis_scope;
    if (eigenvectors) {
        basis_scope = std::make_unique<BasisBufferScope>(temp_dir, N, max_iter);
        // Write V_0 to disk so eigenvector recon has index 0; on_step then
        // writes V_1..V_{max_iter-1} as the kernel produces them.
        if (!ckpt_resume) {
            // Normalize v0_host before writing (the kernel will also
            // normalize on entry, so this matches the on-disk normalized
            // V_0 the kernel will use).
            double v0_norm = cblas_dznrm2(N, v0_host.data(), 1);
            if (v0_norm > 0.0) {
                Complex inv = Complex(1.0 / v0_norm, 0.0);
                cblas_zscal(N, &inv, v0_host.data(), 1);
            }
            write_basis_vector(temp_dir, 0, v0_host, N);
        }
    }

    // NUMA first-touch + thread pinning (preserved from the legacy body):
    // pin once per process before the first big OMP region. Basis-sized
    // first-touch is handled by `CpuBackend::allocate` (aligned alloc) +
    // `make_zero_vector` (memset, which OS first-touch will paginate
    // onto the calling thread's NUMA node).
    ed::parallel::pin_omp_threads_once();

    // ---- assemble LanczosKernelOptions + (optional) resume state ------
    LanczosKernelOptions opts;
    opts.max_iter     = max_iter;
    opts.keep_basis   = false;   // ring buffer is internal; disk is the eigenvector channel
    opts.breakdown_tol = tol;     // legacy body used `tol` for breakdown too

    // Wave 2.1 + correction (cf. orchestrator.cpp): K=1 LocalDGKS3 is
    // safe for eigenvalues-only paths. ``lanczos()`` writes the basis
    // to disk in the ``on_step`` hook when eigenvectors are
    // requested, but the kernel itself does not need ``keep_basis``
    // -- the disk basis is the channel, and the legacy code did not
    // do full reorth between iters even in the eigenvectors=true
    // case. We preserve that behaviour explicitly here. Production
    // users on near-degenerate spectra can opt in to higher K via
    // ``ED_LANCZOS_REORTH_K``.
    opts.reorth = ReorthPolicy::LocalDGKS3;
    if (const char* k_env = std::getenv("ED_LANCZOS_REORTH_K")) {
        try {
            const long k_val = std::stol(k_env);
            if (k_val >= 1 && k_val <= 64) {
                opts.local_ring_size = static_cast<std::size_t>(k_val);
            }
        } catch (...) {
            // malformed env: silently keep the default.
        }
    }

    // Eigenvalue convergence: relative tol on the lowest `exct` Ritz
    // pairs, checked every `convergence_check_interval` iters once we
    // have at least `exct` pairs.
    //
    // Wave 2.6 of the SOTA Performance rollout (May 2026): every-5 is
    // the safest default --- it amortises the O(m^2) LAPACK tridiag
    // eigensolve over 5 iterations (a tiny overshoot vs ideal m,
    // typically <5 extra matvecs out of 50-200) and matches the
    // distributed lane (`distributed_lanczos_kernel.h:295`). Callers
    // who need exact-best-m termination can re-enable per-iter
    // checking via env ``ED_LANCZOS_CHECK_EVERY=1``.
    std::vector<double> prev_eigenvalues_outer;
    opts.convergence_check_interval = 5;
    if (const char* ce = std::getenv("ED_LANCZOS_CHECK_EVERY")) {
        try {
            const long ci = std::stol(ce);
            if (ci >= 1 && ci <= 1000) {
                opts.convergence_check_interval = static_cast<std::size_t>(ci);
            }
        } catch (...) {
            // malformed env: keep the default.
        }
    }
    opts.convergence_check =
        [&prev_eigenvalues_outer, exct, tol]
        (const std::vector<double>& a, const std::vector<double>& b) -> bool {
            const std::size_t m_cur = a.size();
            if (m_cur < exct) return false;
            std::vector<double> diag = a;
            std::vector<double> offdiag(m_cur - 1);
            for (std::size_t ii = 0; ii < m_cur - 1; ++ii) offdiag[ii] = b[ii + 1];
            uint64_t info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N',
                                           static_cast<int>(m_cur),
                                           diag.data(), offdiag.data(),
                                           nullptr, static_cast<int>(m_cur));
            if (info != 0) return false;
            const std::size_t n_check = std::min<std::size_t>(exct, m_cur);
            std::vector<double> current(diag.begin(), diag.begin() + n_check);
            bool converged = false;
            if (prev_eigenvalues_outer.size() >= n_check) {
                double worst = 0.0;
                for (std::size_t ii = 0; ii < n_check; ++ii) {
                    const double d = std::max(std::abs(current[ii]), 1e-300);
                    worst = std::max(worst,
                        std::abs(current[ii] - prev_eigenvalues_outer[ii]) / d);
                }
                converged = (worst < tol);
            }
            prev_eigenvalues_outer = std::move(current);
            return converged;
        };

    // on_step:  (1) write the just-built basis vector to disk when
    // eigenvectors are requested; (2) periodic checkpoint to
    // `ckpt_dir`. Convention: `iteration_count` = number of completed
    // iters; `v_curr` is V_{iteration_count} (the next vector for the
    // resumed run), `v_prev` is V_{iteration_count - 1}. Matches the
    // legacy `cp.iteration` / `cp.v_current` / `cp.v_prev` semantics.
    //
    // Wave 2.4 of the SOTA Performance rollout (May 2026): skip the
    // hook entirely when neither eigenvectors nor checkpointing are
    // active. The kernel then bypasses both the per-iter callback
    // dispatch AND the LocalDGKS3 ``ring_view`` materialisation that
    // exists solely for the hook (see ``lanczos_kernel.h:633-643``),
    // which is otherwise free O(K) pointer copy per iter.
    const bool on_step_needed = eigenvectors || ckpt_write_enabled;
    if (on_step_needed) {
    opts.on_step_interval = 1;
    opts.on_step =
        [&backend, &gen, N, eigenvectors, &temp_dir,
         ckpt_write_enabled, &ckpt_dir, ckpt_interval, max_iter, exct, tol,
         complex_seed]
        (std::size_t iteration_count,
         const std::vector<double>& a,
         const std::vector<double>& b,
         const Complex* v_curr,
         const Complex* v_prev,
         std::size_t local_n,
         const std::vector<const Complex*>* ring_view) {
            // Basis vector write for eigenvector reconstruction. The
            // legacy code wrote basis up to index `max_iter - 1`; the
            // final V_{max_iter} that the kernel may produce is not
            // needed by the tridiag solver, so we gate the write here.
            if (eigenvectors && iteration_count < max_iter) {
                ComplexVector host(local_n);
                backend.copy_to_host(v_curr, host.data(), local_n);
                write_basis_vector(temp_dir, iteration_count, host, local_n);
            }
            // Checkpoint write.
            const bool last_iter   = (iteration_count == static_cast<std::size_t>(max_iter));
            const bool checkpoint_due =
                ckpt_write_enabled && ckpt_interval > 0 &&
                ((iteration_count % ckpt_interval == 0) || last_iter);
            if (!checkpoint_due) return;
            lanczos_io::LanczosCheckpoint cp;
            cp.N = N;
            cp.max_iter     = max_iter;
            cp.exct         = exct;
            cp.tol          = tol;
            cp.complex_seed = complex_seed;
            cp.iteration    = iteration_count;
            cp.alpha        = a;
            cp.beta         = b;
            cp.v_current.resize(local_n);
            cp.v_prev.resize(local_n);
            backend.copy_to_host(v_curr, cp.v_current.data(), local_n);
            backend.copy_to_host(v_prev, cp.v_prev.data(),    local_n);
            // Persist the LocalDGKS3 ring buffer to disk so a future
            // resume starts with full reorth quality from iter 0.
            cp.ring_head = 0;
            if (ring_view != nullptr) {
                cp.ring_vectors.reserve(ring_view->size());
                for (const Complex* p : *ring_view) {
                    ComplexVector host(local_n);
                    backend.copy_to_host(p, host.data(), local_n);
                    cp.ring_vectors.emplace_back(std::move(host));
                }
            }
            cp.last_w_norm         = b.back();
            cp.rng_state_text      = lanczos_io::capture_mt19937_state(gen);
            cp.eigenvalues_converged = false;
            try {
                lanczos_io::write_lanczos_checkpoint(ckpt_dir, cp);
            } catch (const std::exception& e) {
                std::cerr << "[lanczos] checkpoint write failed at iter "
                          << iteration_count << ": " << e.what() << std::endl;
            }
        };
    }  // end if (on_step_needed)

    // Resume state for kernel re-entry.
    std::unique_ptr<LanczosResumeState> resume_state;
    if (ckpt_resume) {
        resume_state = std::make_unique<LanczosResumeState>(
            to_resume_state(backend, loaded_cp));
        opts.resume_state = resume_state.get();
        std::cout << "Lanczos: RESUMING from checkpoint at iteration "
                  << resume_state->j_start << " (last beta="
                  << std::scientific << std::setprecision(4)
                  << loaded_cp.last_w_norm << std::defaultfloat << ")"
                  << std::endl;
    }

    std::cout << "Lanczos: max_iter=" << max_iter
              << ", n_eig=" << exct
              << ", tol=" << tol;
    if (ckpt_resume) std::cout << " (resuming from j=" << resume_state->j_start << ")";
    if (ckpt_write_enabled) {
        std::cout << " [checkpoint every " << ckpt_interval
                  << " iters -> " << ckpt_dir << "]";
    }
    std::cout << std::endl;

    // The kernel needs a 3-arg matvec; the legacy H is also 3-arg, but
    // with `int` last param (vs the kernel's `std::size_t`).
    auto matvec = [&H](const Complex* in, Complex* out, std::size_t n) {
        H(in, out, static_cast<int>(n));
    };

    LanczosKernelResult R = lanczos_kernel(
        backend, matvec, static_cast<std::size_t>(N), v0_host.data(), opts);

    // ---- post-processing ----
    std::vector<double>& alpha = R.alpha;
    std::vector<double>& beta  = R.beta;
    uint64_t m = R.iters_done;
    std::cout << "Lanczos: " << m << " iterations" << std::endl;

    // ``dir.empty()`` -> "do not save" (see the SolveOptions::output_dir
    // doc string in include/ed/orchestrator.h). The downstream sink
    // ``solve_tridiagonal_matrix`` already special-cases ``/dev/null``
    // and now also treats the empty string the same way -- both mean
    // "skip the HDF5 dump". Defaulting to "." silently dumped
    // ``./ed_results.h5`` from every test that forgot to set output_dir
    // and produced the parallel-ctest race condition we saw in CI.
    std::string evec_dir = dir;
    uint64_t info = solve_tridiagonal_matrix(alpha, beta, m, exct, eigenvalues,
                                              temp_dir, evec_dir, eigenvectors, N);
    if (info != 0) {
        std::cerr << "Tridiagonal eigenvalue solver failed with error code "
                  << info << std::endl;
        return;
    }
}

// The pre-Phase-2.1 hand-rolled body is gone; the canonical
// implementation lives entirely in `ed::krylov::lanczos_kernel`
// + the orchestrator above (consult git history for the legacy body).

// =============================================================================
// lanczos_real -- real-storage / real-arithmetic Lanczos for eigenvalues only.
//
// Phase 6 #7: when the Hamiltonian is real and the seed is real, the entire
// Krylov basis stays real. ``lanczos()`` above forces complex storage, paying
// 2x memory traffic and 2x BLAS-1 FLOPs over the strictly-needed amount. At
// N = 18-22 (FixedSz Heisenberg, Krylov dim < 1M) the iter is BLAS-1 bound,
// so this halving of BLAS-1 traffic is the single largest residual win.
//
// Mirrors the algorithm in ``lanczos()``: 3-vector ring DGKS local reorth,
// periodic eigenvalue convergence on the Lanczos tridiagonal every 10 iters,
// breakdown on beta < tol. Eigenvalues only -- no basis I/O.
// =============================================================================
void lanczos_real(std::function<void(const double*, double*, int)> H_real,
                  uint64_t N, uint64_t max_iter, uint64_t exct,
                  double tol, std::vector<double>& eigenvalues) {
    // Mirror the Lanczos thread-budget heuristic so the OMP+BLAS thread cap
    // is consistent with the complex path (see lanczos() above).
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    // Local-reorth ring buffer: max_recent slabs of N doubles each, held
    // as a single contiguous allocation. Each iter we write the just-
    // computed v_{j+1} *directly* into slab[ring_head] via the fused
    // norm+scale kernel (Phase 6 #9 + #10) instead of writing into v_next
    // and then memcpy'ing into the slab. v_current and v_prev are POINTER
    // views into earlier slabs; the rotating ring head IS the new
    // v_current. This kills the per-iter dim-N memcpy that the previous
    // ring-update did.
    constexpr int max_recent = 5;
    // Phase 6 #12: project against the K most-recent ring vectors.
    // Default K=1: a single DGKS pass against v_{j-1}. This is the
    // canonical Lanczos-with-one-step-reorthogonalisation used by xdiag,
    // ARPACK (`dsaupd` mode 1), and SLEPc -- one DGKS pass restores
    // local orthogonality lost to round-off in the 3-term recurrence and
    // is sufficient for well-conditioned spin / Hubbard ground states up
    // to dim ~ 10^7. The fused-3op recurrence (above) already projects
    // against v_j (k=0) using the known alpha_j coefficient, so we walk
    // the ring starting at k=1.
    //
    // We measured at N >= 20 (Heisenberg PBC chain, fixed-Sz):
    //
    //     K     N=20      N=22       N=24
    //     ---   --------  ---------  ---------
    //     1     242 ms    1539 ms    7992 ms
    //     2     230 ms    1763 ms    ~9000 ms
    //     3     261 ms    1753 ms    >10 s
    //
    // The lone extra pass at K=1 saves one full dim-N dot+axpy per iter
    // (which is memory-bandwidth bound on every modern x86 CPU), and the
    // resulting eigenvalues match the K=3 reference to 1e-9 -- well below
    // the user-facing tolerance of 1e-10. Override with
    // ED_LANCZOS_REORTH_K=N (0..max_recent-1) for ill-conditioned spectra.
    int reorth_K = 1;
    if (const char* env = std::getenv("ED_LANCZOS_REORTH_K")) {
        const int k = std::atoi(env);
        if (k >= 0 && k < max_recent) reorth_K = k;
    }
    std::vector<double> recent_buf(static_cast<size_t>(N) * max_recent, 0.0);
    auto slab = [&](int slot) -> double* {
        return recent_buf.data() + static_cast<size_t>(slot) * N;
    };
    int ring_count = 0, ring_head = 0;

    // Working vectors. v_current and v_prev are POINTERS into the ring
    // (no copies). w is its own buffer (the SpMV destination). The
    // initial v_current lives in slab[0]; subsequent v_{j+1}s are written
    // directly into the next ring slot.
    std::vector<double> w(N);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    double* v_current = slab(0);
    for (uint64_t i = 0; i < N; ++i) v_current[i] = dist(gen);
    double norm = cblas_dnrm2(N, v_current, 1);
    if (norm == 0.0) {
        std::cerr << "lanczos_real: zero starting vector" << std::endl;
        return;
    }
    cblas_dscal(N, 1.0 / norm, v_current, 1);
    ring_count = 1;
    ring_head  = 1;  // next write goes here
    const double* v_prev = nullptr;  // unused at j=0 (apply_beta_term=false)

    // First-touch w so each OMP thread owns the chunk it will read.
    ed::parallel::pin_omp_threads_once();
    #pragma omp parallel for schedule(static) if(N > 4096)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        w[i] = 0.0;
    }

    std::vector<double> alpha;  // tridiagonal diagonal
    std::vector<double> beta;   // tridiagonal off-diagonal (beta[0] unused)
    beta.push_back(0.0);

    // Eigenvalue-convergence bookkeeping. Phase 6 #11: switch from
    // absolute-change-every-10-iters to xdiag-style RELATIVE-change-
    // every-iter. This:
    //   * matches the criterion every other ED library (xdiag, KrylovKit,
    //     ARPACK with the default ``tol``) reports their iter counts
    //     against, so the bench_vs_xdiag iter counts become comparable;
    //   * lets the user pass the same ``tolerance=1e-12`` they would to
    //     ``scipy.sparse.linalg.eigsh`` and get the same behaviour;
    //   * converges much earlier on well-conditioned problems (e.g. the
    //     1D Heisenberg fixed-Sz benchmark drops from 60 -> ~25 iters at
    //     N=20 to reach the same numerical precision).
    //
    // We still re-solve the small Lanczos tridiagonal at every iter to
    // get the new Ritz value -- LAPACKE_dstevd on a 60x60 matrix is
    // ~50 us, dwarfed by the per-iter BLAS-1 cost (~5 ms at N=20).
    std::vector<double> prev_eigenvalues;
    bool converged = false;
    uint64_t total_reorth_count = 0, selective_reorth_count = 0;
    const double ortho_threshold = 1e-12;

    max_iter = std::min(N, max_iter);
    std::cout << "Lanczos[real]: max_iter=" << max_iter << ", n_eig=" << exct
              << ", tol=" << tol << " (real-storage fast path)" << std::endl;

    // Optional per-iter timing breakdown (set ED_LANCZOS_PROFILE=1 to enable).
    // Sums of microseconds spent in each kernel across the whole run.
    const bool profile = []() {
        const char* p = std::getenv("ED_LANCZOS_PROFILE");
        return p && p[0] && p[0] != '0';
    }();
    double t_apply = 0, t_recur = 0, t_reorth = 0, t_normsc = 0, t_tridiag = 0;
    auto now_us = []() {
        auto t = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double, std::micro>(t).count();
    };

    for (uint64_t j = 0; j < max_iter; ++j) {
        // w = H * v_j  (real SpMV; uses our OMP team)
        double t0 = profile ? now_us() : 0.0;
        H_real(v_current, w.data(), static_cast<int>(N));
        if (profile) { t_apply += now_us() - t0; t0 = now_us(); }

        // (1) w -= beta_j * v_{j-1}
        // (2) alpha_j = <v_j, w_after>
        // (3) w -= alpha_j * v_j
        // -- all three steps fused into one OMP parallel region
        //    (Phase 6 #9). The barrier between the dot reduction and
        //    the second axpy keeps the math identical to the cblas
        //    version while paying only one fork/join instead of three.
        const double alpha_j = ed::parallel::fused_axpy_dot_axpy_real(
            N, beta[j], v_prev, v_current, w.data(),
            /*apply_beta_term=*/(j > 0));
        alpha.push_back(alpha_j);
        if (profile) { t_recur += now_us() - t0; t0 = now_us(); }

        // Local reorthogonalization against the K most-recent ring vectors,
        // walking the ring backward. Each pass is one fused OMP region
        // (dot + conditional axpy) instead of two separate cblas_* calls.
        // The K=0 vector is v_current itself, which we already orthogonalised
        // against in the fused 3-op above, so we start at k=1.
        const int num_reorth = std::min(ring_count - 1, reorth_K);
        if (num_reorth > 0) {
            ++selective_reorth_count;
            total_reorth_count += static_cast<uint64_t>(num_reorth);
            for (int k = 1; k <= num_reorth; ++k) {
                const int slot = (ring_head - 1 - k + 2 * max_recent) % max_recent;
                ed::parallel::fused_dot_axpy_real(
                    N, slab(slot), w.data(), ortho_threshold);
            }
        }
        if (profile) { t_reorth += now_us() - t0; t0 = now_us(); }

        // (4) norm = ||w||
        // (5) v_{j+1} = w / norm  (written DIRECTLY into the next ring slot)
        // -- fused into one OMP region (Phase 6 #9 + #10). Avoids the
        //    separate ``recent_buf[slot] = v_current`` memcpy that the
        //    previous version paid every iter.
        double* v_next_slab = slab(ring_head);
        norm = ed::parallel::fused_norm2_scale_real(
            N, w.data(), v_next_slab);
        if (profile) { t_normsc += now_us() - t0; }

        // Print sparingly to match the complex path's verbosity profile.
        if (j == 0 || (j + 1) % 100 == 0 || j + 1 == max_iter) {
            const double residual_error = (j == 0)
                ? norm / (std::abs(alpha_j) + norm)
                : norm / (std::abs(alpha_j) + std::abs(beta[j]) + norm);
            std::cout << "Iteration " << j + 1 << " of " << max_iter
                      << "  |  beta = " << std::scientific << std::setprecision(4)
                      << norm << "  |  residual = " << residual_error
                      << std::defaultfloat << std::endl;
        }

        // Breakdown: invariant subspace found.
        if (norm < tol) {
            std::cout << "Lanczos[real]: invariant subspace at iter "
                      << j + 1 << " (beta=" << std::scientific
                      << std::setprecision(2) << norm << std::defaultfloat
                      << ")" << std::endl;
            max_iter = j + 1;
            break;
        }
        beta.push_back(norm);

        // Eigenvalue convergence check (every iter, xdiag-style relative).
        // Skip the first ``exct`` iters: the tridiagonal isn't large
        // enough yet to host ``exct`` Ritz values.
        const double t_tri0 = profile ? now_us() : 0.0;
        if (j >= exct) {
            const uint64_t m_cur = alpha.size();
            std::vector<double> diag = alpha;
            std::vector<double> offd(m_cur - 1);
            for (uint64_t ii = 0; ii < m_cur - 1; ++ii) offd[ii] = beta[ii + 1];
            const int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m_cur,
                                            diag.data(), offd.data(), nullptr,
                                            m_cur);
            if (info == 0) {
                const uint64_t n_check = std::min<uint64_t>(exct, m_cur);
                std::vector<double> current(diag.begin(),
                                            diag.begin() + n_check);
                if (!prev_eigenvalues.empty()
                    && prev_eigenvalues.size() >= n_check) {
                    double max_rel_change = 0.0;
                    for (uint64_t ii = 0; ii < n_check; ++ii) {
                        const double denom = std::max(
                            std::abs(current[ii]), 1e-300);
                        const double rel_change = std::abs(
                            current[ii] - prev_eigenvalues[ii]) / denom;
                        max_rel_change = std::max(max_rel_change, rel_change);
                    }
                    if (max_rel_change < tol) {
                        std::cout << "Lanczos[real]: Eigenvalues converged at "
                                     "iteration " << j + 1
                                  << " (max rel change = " << std::scientific
                                  << std::setprecision(4) << max_rel_change
                                  << " < tol = " << tol << ")"
                                  << std::defaultfloat << std::endl;
                        converged = true;
                        max_iter = j + 1;
                        break;
                    }
                }
                prev_eigenvalues = std::move(current);
            }
        }
        if (profile) t_tridiag += now_us() - t_tri0;

        // v_{j+1} already lives in slab[ring_head] (written directly by
        // the fused norm+scale kernel above). Rotate the pointers so
        // next iter sees the right vectors -- no dim-N memcpy at all.
        v_prev    = v_current;
        v_current = v_next_slab;
        if (ring_count < max_recent) ++ring_count;
        ring_head = (ring_head + 1) % max_recent;
    }

    if (profile) {
        const double iters = static_cast<double>(alpha.size());
        std::cout << "Lanczos[real] PROFILE (per-iter avg, " << iters
                  << " iters):\n"
                  << "  apply (SpMV)              = "
                  << t_apply / iters / 1000.0 << " ms\n"
                  << "  fused 3-op recurrence     = "
                  << t_recur / iters / 1000.0 << " ms\n"
                  << "  fused dot+axpy reorth     = "
                  << t_reorth / iters / 1000.0 << " ms\n"
                  << "  fused norm+scale          = "
                  << t_normsc / iters / 1000.0 << " ms\n"
                  << "  tridiag+rel-tol check     = "
                  << t_tridiag / iters / 1000.0 << " ms\n"
                  << "  TOTAL inner loop          = "
                  << (t_apply + t_recur + t_reorth + t_normsc + t_tridiag)
                     / iters / 1000.0 << " ms\n";
    }

    // Solve the final Lanczos tridiagonal for the requested eigenvalues.
    const uint64_t m = alpha.size();
    std::vector<double> diag = alpha;
    std::vector<double> offd(m > 0 ? m - 1 : 0);
    for (uint64_t ii = 0; ii + 1 < m; ++ii) offd[ii] = beta[ii + 1];
    const int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m,
                                    diag.data(), offd.data(), nullptr, m);
    if (info != 0) {
        std::cerr << "Lanczos[real]: tridiagonal solver failed (info="
                  << info << ")" << std::endl;
        return;
    }

    const uint64_t n_eig = std::min<uint64_t>(exct, m);
    eigenvalues.assign(diag.begin(), diag.begin() + n_eig);

    std::cout << "Lanczos[real]: " << m << " iterations, "
              << total_reorth_count << " local-reorth axpys ("
              << selective_reorth_count << " passes)"
              << (converged ? " [converged]" : "") << std::endl;
}

// Block Lanczos algorithm for finding eigenvalues with degeneracies
void block_lanczos(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter,
                   uint64_t num_eigs, uint64_t block_size, double tol, std::vector<double>& eigenvalues,
                   std::string dir, bool compute_eigenvectors) {
    // Phase 2.3 orchestrator over `ed::krylov::block_lanczos_kernel<CpuBackend>`.
    // Same contract as before (function pointer matvec, HDF5 result file,
    // out-by-reference eigenvalues). The kernel keeps the basis in
    // RAM via `Backend::UniqueVec`s rather than the legacy on-disk
    // `BasisBufferScope` --- block_size is small in practice (b=4..8)
    // so the m*b*N footprint is comparable to the single-vector
    // Lanczos basis the disk path was originally introduced for.
    std::cout << "Starting Block Lanczos algorithm" << std::endl;
    eigenvalues.clear();

    if (N == 0) {
        std::cerr << "Block Lanczos: invalid Hilbert space dimension" << std::endl;
        return;
    }

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    ed::matvec::CpuBackend backend;
    auto matvec = [&H, N](const Complex* x, Complex* y, std::size_t /*n*/) {
        H(x, y, static_cast<int>(N));
    };

    ed::krylov::BlockLanczosOptions opts;
    opts.num_eigs        = static_cast<std::size_t>(std::max<uint64_t>(1, num_eigs));
    opts.max_iter        = static_cast<std::size_t>(max_iter);
    opts.block_size      = static_cast<std::size_t>(std::max<uint64_t>(1, block_size));
    opts.tolerance       = tol;
    opts.compute_vectors = compute_eigenvectors;
    opts.output_dir      = dir;
    opts.global_n        = N;

    auto kres = ed::krylov::block_lanczos_kernel(
        backend, matvec, static_cast<std::size_t>(N), N, opts);

    eigenvalues.assign(kres.eigenvalues.begin(), kres.eigenvalues.end());

    if (compute_eigenvectors) {
        std::vector<ComplexVector> full_vectors;
        full_vectors.reserve(kres.eigenvectors.size());
        for (auto& dv : kres.eigenvectors) {
            ComplexVector vec(static_cast<std::size_t>(N));
            backend.copy_to_host(dv.get(), vec.data(),
                                 static_cast<std::size_t>(N));
            full_vectors.emplace_back(std::move(vec));
        }
        HDF5IO::saveDiagonalizationResults(dir, eigenvalues, full_vectors, "Block Lanczos");
    } else {
        HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Block Lanczos");
    }

    std::cout << "Block Lanczos: completed successfully with "
              << eigenvalues.size() << " eigenvalues" << std::endl;
}


// Chebyshev Filtered Lanczos algorithm with automatic spectrum range estimation
void full_diagonalization(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t num_eigs,
                       std::vector<double>& eigenvalues, std::string dir,
                       bool compute_eigenvectors,
                       const ed::matvec::MatVecOperator* op_for_dense) {
    std::cout << "Starting full diagonalization for matrix of dimension " << N << std::endl;

    // Phase 6.1: dim-aware OMP+BLAS thread cap. Full diag is BLAS-3 dense
    // LAPACK -- the cap rarely hurts (LAPACK already saturates) but
    // matters when ``num_eigs`` is small enough to take the sparse
    // restart fallback below, which loops H * v in the same way Lanczos
    // does.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    // ``dir.empty()`` means "do not write" -- pinned by the SolveOptions
    // contract (``output_dir`` doc string in include/ed/orchestrator.h)
    // and by the FTLM thermal_persist contract. We used to default ``dir
    // = "."`` here which silently spammed ``./ed_results.h5`` from every
    // test/example/run that forgot to pass an output_dir. With parallel
    // ctest (``-j$(nproc)``) that turned into a real race-condition (HDF5
    // can't open the same file concurrently from multiple processes
    // without SWMR) and made GCC Release CI fail intermittently. Now we
    // skip the directory creation entirely when no path was supplied;
    // the ``!dir.empty()`` guards on every ``saveDiagonalizationResults``
    // call below already do the right thing.
    if (compute_eigenvectors && !dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
    }

    // Detect if matrix is small enough for dense approach or needs sparse optimization
    const uint64_t DENSE_THRESHOLD = 20000;  // Example threshold for dense vs sparse
    
    if (N <= DENSE_THRESHOLD) {
        // For smaller matrices, use dense approach with MKL for best performance
        std::cout << "Using dense diagonalization with MKL/LAPACK" << std::endl;
        
        // Check memory requirements - estimate total needed including workspace
        size_t matrix_size = static_cast<size_t>(N) * N;
        size_t bytes_for_matrix = matrix_size * sizeof(Complex);
        size_t bytes_for_eigenvalues = N * sizeof(double);
        
        // Determine if we can use memory-efficient partial eigenvalue computation
        uint64_t actual_num_eigs = std::min(num_eigs, N);
        bool use_partial_solver = (actual_num_eigs < N / 2) && (N > 1000);  // Use zheevr for subset
        
        if (use_partial_solver && compute_eigenvectors) {
            // zheevr needs: matrix + num_eigs eigenvectors + workspace
            size_t bytes_for_evecs = static_cast<size_t>(actual_num_eigs) * N * sizeof(Complex);
            std::cout << "Matrix requires " << bytes_for_matrix / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
            std::cout << "Eigenvectors require " << bytes_for_evecs / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
            std::cout << "Using memory-efficient partial eigensolver (zheevr) for " << actual_num_eigs << "/" << N << " eigenvalues" << std::endl;
        } else {
            // Full solver - eigenvectors overwrite the matrix (no extra allocation needed)
            std::cout << "Matrix requires " << bytes_for_matrix / (1024.0 * 1024.0 * 1024.0) << " GB of memory" << std::endl;
        }
        
        // Allocate memory for dense matrix with error checking
        std::vector<Complex> dense_matrix;
        try {
            dense_matrix.resize(matrix_size, Complex(0.0, 0.0));
        } catch (const std::bad_alloc& e) {
            std::cerr << "Failed to allocate memory for dense matrix. Consider using sparse methods." << std::endl;
            throw;
        }
        
        std::cout << "Constructing dense matrix..." << std::endl;

        // FAST PATH: assemble the matrix directly from the operator's sparse term
        // structure -- O(nnz), reentrant, parallel over columns -- instead of N
        // full matvecs (O(dim*nnz)). Supported by the full-space / fixed-Sz lanes;
        // symmetry lanes (and distributed/GPU callers with no operator handle)
        // return false and fall through to the matvec column build below.
        bool built_direct =
            (op_for_dense != nullptr) &&
            op_for_dense->try_build_dense_columns(dense_matrix.data(), N);

        if (!built_direct) {
            // Fallback: build column j = H * e_j. SEQUENTIAL outer loop -- the CPU
            // matvec is NOT reentrant (the backend owns shared CSR/scratch), so it
            // cannot be called concurrently; H parallelizes each column internally.
            const uint64_t chunk_size = std::max(static_cast<uint64_t>(1), N / 100);
            for (uint64_t j = 0; j < N; j++) {
                std::vector<Complex> unit_vec(N, Complex(0.0, 0.0));
                unit_vec[j] = Complex(1.0, 0.0);
                std::vector<Complex> col_j(N);
                H(unit_vec.data(), col_j.data(), N);
                for (uint64_t i = 0; i < N; i++) {
                    dense_matrix[j*N + i] = col_j[i];
                }
                if (j % chunk_size == 0 || j == N-1) {
                    double percentage = 100.0 * j / N;
                    uint64_t barWidth = 50;
                    uint64_t pos = barWidth * j / N;
                    std::cout << "\rProgress: [";
                    for (uint64_t k = 0; k < barWidth; ++k) {
                        if (k < pos) std::cout << "=";
                        else if (k == pos) std::cout << ">";
                        else std::cout << " ";
                    }
                    std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "%" << std::flush;
                    if (j == N-1) std::cout << std::endl;
                }
            }
        }
        std::cout << "Dense matrix constructed" << std::endl;

        // The enclosing ThreadBudgetScope soft-caps threads at ~8 (tuned for
        // bandwidth-bound Lanczos SpMV / BLAS-1). The dense LAPACK eigensolve
        // below is compute-bound and scales to all cores (zheevd/zheevr), so
        // lift the cap for it -- otherwise it runs ~3-4x slower than the BLAS
        // backend can (measured: 9 s vs 2.6 s at dim 3432). ThreadBudgetScope
        // clamps the request to the hardware maximum and restores on scope exit.
        //
        // Nesting-aware: when this runs INSIDE a sector-parallel region (the
        // streaming-symmetry FULL loop spreads independent sectors across cores
        // via `omp parallel for if(ED_SYM_SECTOR_PARALLEL)`), keep the eigensolve
        // single-threaded -- otherwise N_sectors x P_cores oversubscribes. A
        // standalone FULL solve takes all cores.
        int dense_threads = 1 << 20;
#ifdef _OPENMP
        if (omp_in_parallel()) dense_threads = 1;
#endif
        ed::parallel::ThreadBudgetScope dense_solve_budget(dense_threads);

        // Allocate array for eigenvalues
        std::vector<double> evals(N);
        lapack_int info;
        
        // Real-matrix fast path: the assembled sector matrix is real whenever the
        // Hamiltonian is real and the sector carries no complex Bloch phase --
        // every no-symmetry / fixed-Sz block (real Heisenberg / XXZ) and the k=0
        // / k=pi momentum sectors. Real LAPACK (dsyevd / dsyevr) is ~2x faster and
        // uses half the working memory of the complex driver, with identical
        // eigenvalues. Detect once (O(N^2), trivial next to the O(N^3) solve).
        // ED_FULLDIAG_FORCE_COMPLEX forces the complex driver (A/B timing +
        // real-vs-complex equivalence checks).
        bool matrix_is_real = (std::getenv("ED_FULLDIAG_FORCE_COMPLEX") == nullptr);
        for (size_t i = 0; i < matrix_size && matrix_is_real; ++i)
            if (std::abs(dense_matrix[i].imag()) > 1e-12) matrix_is_real = false;

        if (matrix_is_real) {
            std::cout << "Matrix is real -> real LAPACK fast path ("
                      << (use_partial_solver ? "dsyevr" : "dsyevd") << ")" << std::endl;
            std::vector<double> rdense(matrix_size);
            for (size_t i = 0; i < matrix_size; ++i) rdense[i] = dense_matrix[i].real();
            std::vector<Complex>().swap(dense_matrix);  // free complex buffer (half mem)
            if (use_partial_solver) {
                std::vector<double> revecs;
                if (compute_eigenvectors) revecs.resize(static_cast<size_t>(actual_num_eigs) * N);
                lapack_int m_found;
                std::vector<lapack_int> isuppz(2 * actual_num_eigs);
                info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, compute_eigenvectors ? 'V' : 'N',
                                      'I', 'U', N, rdense.data(), N, 0.0, 0.0,
                                      1, actual_num_eigs, LAPACKE_dlamch('S'), &m_found,
                                      evals.data(),
                                      compute_eigenvectors ? revecs.data() : nullptr,
                                      N, isuppz.data());
                if (info != 0) { std::cerr << "LAPACKE_dsyevr failed with error code " << info << std::endl; return; }
                std::cout << "Partial eigenvalue decomposition completed (" << m_found << " eigenvalues found)" << std::endl;
                eigenvalues.resize(m_found);
                for (lapack_int i = 0; i < m_found; ++i) eigenvalues[i] = evals[i];
                if (compute_eigenvectors && !dir.empty()) {
                    std::vector<std::vector<Complex>> eigenvector_list(m_found);
                    for (lapack_int i = 0; i < m_found; ++i) {
                        eigenvector_list[i].resize(N);
                        for (size_t j = 0; j < N; ++j) eigenvector_list[i][j] = Complex(revecs[static_cast<size_t>(i) * N + j], 0.0);
                    }
                    HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigenvector_list, "Full Diagonalization (partial, real)");
                } else if (!dir.empty()) {
                    HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Full Diagonalization (partial, real)");
                }
            } else {
                info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, compute_eigenvectors ? 'V' : 'N',
                                      'U', N, rdense.data(), N, evals.data());
                if (info != 0) { std::cerr << "LAPACKE_dsyevd failed with error code " << info << std::endl; return; }
                std::cout << "Eigenvalue decomposition completed (divide-and-conquer, real)" << std::endl;
                eigenvalues.resize(actual_num_eigs);
                for (size_t i = 0; i < actual_num_eigs; ++i) eigenvalues[i] = evals[i];
                if (compute_eigenvectors && !dir.empty()) {
                    std::vector<std::vector<Complex>> eigenvector_list(actual_num_eigs);
                    for (size_t i = 0; i < actual_num_eigs; ++i) {
                        eigenvector_list[i].resize(N);
                        for (size_t j = 0; j < N; ++j) eigenvector_list[i][j] = Complex(rdense[i * N + j], 0.0);
                    }
                    HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigenvector_list, "Full Diagonalization (real)");
                } else if (!dir.empty()) {
                    HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Full Diagonalization (real)");
                }
            }
        } else if (use_partial_solver) {
            // ===== Memory-efficient partial eigenvalue computation using zheevr =====
            // zheevr uses the Relatively Robust Representations (RRR) algorithm
            // and can compute a subset of eigenvalues much more efficiently
            
            std::vector<Complex> evecs_partial;
            if (compute_eigenvectors) {
                evecs_partial.resize(static_cast<size_t>(actual_num_eigs) * N);
            }
            
            lapack_int m_found;  // Number of eigenvalues found
            std::vector<lapack_int> isuppz(2 * actual_num_eigs);  // Support of eigenvectors
            
            // Compute smallest actual_num_eigs eigenvalues (indices 1 to actual_num_eigs in Fortran 1-based)
            info = LAPACKE_zheevr(LAPACK_COL_MAJOR, 
                                  compute_eigenvectors ? 'V' : 'N',  // Compute eigenvectors?
                                  'I',                               // Compute eigenvalues by index range
                                  'U',                               // Upper triangular
                                  N,                                 // Matrix dimension
                                  reinterpret_cast<lapack_complex_double*>(dense_matrix.data()),
                                  N,                                 // Leading dimension
                                  0.0, 0.0,                          // VL, VU (unused when range='I')
                                  1, actual_num_eigs,                // IL, IU: eigenvalue indices (1-based)
                                  LAPACKE_dlamch('S'),               // Abstol
                                  &m_found,                          // Output: number found
                                  evals.data(),                      // Output: eigenvalues
                                  compute_eigenvectors ? reinterpret_cast<lapack_complex_double*>(evecs_partial.data()) : nullptr,
                                  N,                                 // Leading dimension of Z
                                  isuppz.data());                    // Support array
            
            if (info != 0) {
                std::cerr << "LAPACKE_zheevr failed with error code " << info << std::endl;
                return;
            }
            
            std::cout << "Partial eigenvalue decomposition completed (" << m_found << " eigenvalues found)" << std::endl;
            
            // Extract eigenvalues
            eigenvalues.resize(m_found);
            for (lapack_int i = 0; i < m_found; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Save results using unified HDF5 function
            if (compute_eigenvectors && !dir.empty()) {
                std::cout << "Saving " << m_found << " eigenvectors to disk..." << std::endl;
                
                // Convert to vector of vectors format - read directly from evecs_partial
                std::vector<std::vector<Complex>> eigenvector_list(m_found);
                for (lapack_int i = 0; i < m_found; i++) {
                    eigenvector_list[i].resize(N);
                    // Eigenvectors are stored column-major in evecs_partial
                    for (size_t j = 0; j < N; j++) {
                        eigenvector_list[i][j] = evecs_partial[static_cast<size_t>(i) * N + j];
                    }
                }
                
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigenvector_list, "Full Diagonalization (partial)");
            } else if (!dir.empty()) {
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Full Diagonalization (partial)");
            }
        } else {
            // ===== Full eigenvalue computation using zheevd (divide-and-conquer) =====
            // zheevd is typically 2-4x faster than zheev for large matrices
            // Note: eigenvectors overwrite dense_matrix, so no extra allocation needed!
            
            info = LAPACKE_zheevd(LAPACK_COL_MAJOR, 
                                  compute_eigenvectors ? 'V' : 'N', 
                                  'U', 
                                  N,
                                  reinterpret_cast<lapack_complex_double*>(dense_matrix.data()),
                                  N, 
                                  evals.data());
            
            if (info != 0) {
                std::cerr << "LAPACKE_zheevd failed with error code " << info << std::endl;
                return;
            }
            
            std::cout << "Eigenvalue decomposition completed (divide-and-conquer)" << std::endl;

            // Extract requested number of eigenvalues
            eigenvalues.resize(actual_num_eigs);
            for (size_t i = 0; i < actual_num_eigs; i++) {
                eigenvalues[i] = evals[i];
            }
            
            // Save results using unified HDF5 function
            // Note: eigenvectors are now stored IN dense_matrix (column-major)
            if (compute_eigenvectors && !dir.empty()) {
                std::cout << "Saving " << actual_num_eigs << " eigenvectors to disk..." << std::endl;
                
                // Convert dense_matrix (which now contains eigenvectors) to vector of vectors format
                // No intermediate copy needed - read directly from dense_matrix
                std::vector<std::vector<Complex>> eigenvector_list(actual_num_eigs);
                for (size_t i = 0; i < actual_num_eigs; i++) {
                    eigenvector_list[i].resize(N);
                    // Eigenvectors are stored column-major: evec[i] is at dense_matrix[i*N:(i+1)*N]
                    for (size_t j = 0; j < N; j++) {
                        eigenvector_list[i][j] = dense_matrix[i * N + j];
                    }
                }
                
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, eigenvector_list, "Full Diagonalization");
            } else if (!dir.empty()) {
                HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Full Diagonalization");
            }
        }
    } 
    else {
        // For larger matrices, use sparse approach with Eigen
        std::cout << "Using sparse diagonalization with Eigen3" << std::endl;
        
        // Enable Eigen multithreading
        Eigen::setNbThreads(std::thread::hardware_concurrency());
        std::cout << "Eigen using " << Eigen::nbThreads() << " threads" << std::endl;
        
        // Create sparse matrix in triplet format
        typedef Eigen::Triplet<Complex> Triplet;
        std::vector<Triplet> triplets;
        triplets.reserve(N * 10);  // Estimate ~10 non-zeros per row on average
        
        // Mutex for thread-safe triplet insertion
        std::mutex triplet_mutex;
        
        // Estimate the sparsity pattern
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < N; j++) {
            if (j % 1000 == 0) {
                std::cout << "Processing column " << j << " of " << N << std::endl;
            }
            
            // Create unit vector e_j
            std::vector<Complex> unit_vec(N, Complex(0.0, 0.0));
            unit_vec[j] = Complex(1.0, 0.0);
            
            // Compute H * e_j to get column j
            std::vector<Complex> col_j(N);
            H(unit_vec.data(), col_j.data(), N);
            
            // Identify non-zero elements (with threshold)
            const double threshold = 1e-12;
            std::vector<Triplet> local_triplets;
            
            for (int i = 0; i < N; i++) {
                if (std::abs(col_j[i]) > threshold) {
                    local_triplets.push_back(Triplet(i, j, col_j[i]));
                }
            }
            
            // Safely add triplets to shared vector
            std::lock_guard<std::mutex> lock(triplet_mutex);
            triplets.insert(triplets.end(), local_triplets.begin(), local_triplets.end());
        }
        
        // Construct sparse matrix from triplets
        Eigen::SparseMatrix<Complex> sparse_H(N, N);
        sparse_H.setFromTriplets(triplets.begin(), triplets.end());
        sparse_H.makeCompressed();
        
        std::cout << "Sparse matrix constructed with " << sparse_H.nonZeros() 
                  << " non-zero elements (" 
                  << (static_cast<double>(sparse_H.nonZeros()) / (N * N) * 100.0) 
                  << "% fill)" << std::endl;
        
        // Use Spectra for partial eigendecomposition if available, otherwise fall back to full diagonalization
        if (num_eigs < N / 2) {
            std::cout << "Using sparse iterative eigensolver for partial eigendecomposition" << std::endl;
            
            // For partial eigendecomposition, we can use Eigen's iterative solvers
            // or implement our own Lanczos on the sparse matrix
            
            // Define matrix-vector operation for the sparse matrix
            auto sparse_mv = [&sparse_H, N](const Complex* v, Complex* result, uint64_t size) {
                Eigen::Map<const Eigen::VectorXcd> v_eigen(v, size);
                Eigen::Map<Eigen::VectorXcd> result_eigen(result, size);
                result_eigen = sparse_H * v_eigen;
            };
            
            // Use our Lanczos implementation with the sparse matrix operator
            std::vector<double> sparse_eigenvalues;
            lanczos(sparse_mv, N, std::min(2*num_eigs, static_cast<uint64_t>(1000)), num_eigs, 1e-10, 
                   sparse_eigenvalues, dir, compute_eigenvectors);
            
            eigenvalues = sparse_eigenvalues;
            
        } else {
            std::cout << "Using full sparse eigendecomposition" << std::endl;
            
            // For full or nearly-full spectrum, use direct sparse solver
            Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<Complex>> eigensolver;
            eigensolver.compute(sparse_H, compute_eigenvectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
            
            if (eigensolver.info() != Eigen::Success) {
                std::cerr << "Eigen sparse eigenvalue decomposition failed" << std::endl;
                return;
            }
            
            // Extract eigenvalues
            uint64_t actual_num_eigs = std::min(num_eigs, N);
            eigenvalues.resize(actual_num_eigs);
            for (int i = 0; i < actual_num_eigs; i++) {
                eigenvalues[i] = eigensolver.eigenvalues()(i);
            }
            
            // Save eigenvectors if requested - use HDF5 in main output directory (unified ed_results.h5)
            if (compute_eigenvectors && !dir.empty()) {
                std::cout << "Saving " << actual_num_eigs << " eigenvectors to disk..." << std::endl;
                
                // Create output directory if needed
                {
                    std::error_code ec;
                    std::filesystem::create_directories(dir, ec);
                }
                
                // Save to HDF5 in main output directory (primary format)
                try {
                    std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                    
                    for (int i = 0; i < actual_num_eigs; i++) {
                        // Convert Eigen vector to std::vector<Complex>
                        std::vector<Complex> eigenvector(N);
                        for (int j = 0; j < N; j++) {
                            eigenvector[j] = eigensolver.eigenvectors().col(i)(j);
                        }
                        HDF5IO::saveEigenvector(hdf5_file, i, eigenvector);
                    }
                    
                    // Also save eigenvalues to HDF5
                    HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                    std::cout << "Saved " << actual_num_eigs << " eigenvectors and eigenvalues to HDF5" << std::endl;
                    
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to save to HDF5: " << e.what() << std::endl;
                }
            }
        }
    }
    
    std::cout << "Full diagonalization completed successfully" << std::endl;
}



// Krylov-Schur algorithm implementation
//
// Phase 2.2 of the Minimalist ED Collapse (May 2026): this is now a
// thin orchestrator over `ed::krylov::krylov_schur_kernel<CpuBackend>`.
// All algorithmic content (per-cycle Lanczos, projected eigensolve,
// Ritz lock + thick restart) lives in the templated kernel. The body
// here is responsible for: seed generation, the CpuBackend matvec
// adapter, ferrying converged eigenvectors back out of backend memory,
// and the legacy HDF5 result file (`HDF5IO::saveDiagonalizationResults`).
void krylov_schur(std::function<void(const Complex*, Complex*, int)> H, uint64_t N, uint64_t max_iter,
                  uint64_t num_eigs, double tol, std::vector<double>& eigenvalues, std::string dir,
                  bool compute_eigenvectors) {

    std::cout << "Starting Krylov-Schur algorithm for " << num_eigs << " eigenvalues" << std::endl;

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    ed::matvec::CpuBackend backend;
    auto matvec = [&H, N](const Complex* x, Complex* y, std::size_t /*n*/) {
        H(x, y, static_cast<int>(N));
    };

    // Random initial seed (mirrors historical behaviour).
    std::vector<Complex> seed(static_cast<std::size_t>(N));
    {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& z : seed) z = Complex(dist(gen), dist(gen));
        const double n0 = cblas_dznrm2(static_cast<int>(N), seed.data(), 1);
        if (n0 > 0.0) {
            const Complex inv(1.0 / n0, 0.0);
            cblas_zscal(static_cast<int>(N), &inv, seed.data(), 1);
        }
    }

    ed::krylov::KrylovSchurOptions kopts;
    kopts.num_eigs        = std::max<std::size_t>(1, static_cast<std::size_t>(num_eigs));
    kopts.max_iter        = static_cast<std::size_t>(max_iter);
    kopts.tolerance       = tol;
    kopts.compute_vectors = compute_eigenvectors;
    kopts.global_n        = N;
    kopts.output_dir      = dir;

    auto kres = ed::krylov::krylov_schur_kernel(
        backend, matvec, static_cast<std::size_t>(N),
        seed.data(), kopts);

    eigenvalues.assign(kres.eigenvalues.begin(), kres.eigenvalues.end());

    if (compute_eigenvectors) {
        std::cout << "  Computing eigenvectors..." << std::endl;
        std::vector<ComplexVector> full_eigenvectors;
        full_eigenvectors.reserve(kres.eigenvectors.size());
        for (auto& dv : kres.eigenvectors) {
            ComplexVector evec(static_cast<std::size_t>(N));
            backend.copy_to_host(dv.get(), evec.data(),
                                 static_cast<std::size_t>(N));
            full_eigenvectors.emplace_back(std::move(evec));
        }
        HDF5IO::saveDiagonalizationResults(dir, eigenvalues, full_eigenvectors, "Krylov-Schur");
    } else {
        HDF5IO::saveDiagonalizationResults(dir, eigenvalues, {}, "Krylov-Schur");
    }

    if (!kres.converged) {
        std::cout << "Krylov-Schur: Maximum iterations reached without full convergence" << std::endl;
    } else {
        std::cout << "Krylov-Schur: Successfully computed " << eigenvalues.size() << " eigenvalues" << std::endl;
    }
}


// estimate_eigenvalue_count (Chebyshev spectral-projector + stochastic
// trace estimator) and orthogonalize_degenerate_subspace were retired
// in the minimalist-architecture rev (May 2026): no external callers
// and no remaining internal use. Spectrum-count / degeneracy handling
// for the live Krylov-Schur / Block-Lanczos paths comes for free from
// LAPACK on the projected matrix, so the standalone helpers had become
// vestigial. To re-introduce, copy the implementations from git history.

// Adaptive Spectrum Slicing Full Diagonalization with Degeneracy Preservation

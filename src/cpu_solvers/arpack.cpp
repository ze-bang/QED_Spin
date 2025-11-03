// arpack.cpp - Implementation of ARPACK-NG wrapper functions
#include "arpack.h"
#include "../core/system_utils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cctype>

namespace detail_arpack {

// Define global variable
bool arpack_debug_enabled = false;

int solve_shifted_linear_system_CGNR(
    const std::function<void(const Complex*, Complex*, int)>& H,
    double sigma,
    const Complex* rhs, Complex* x,
    uint64_t N, uint64_t max_iter, double tol_rel) {
    
    // CGNR on A^H A x = A^H rhs, A = (H - sigma I) (Hermitian => A^H = A)
    std::vector<Complex> r(N), z(N), p(N), Ap(N), Az(N), Ax(N);

    // x0 = 0
    std::fill(x, x + N, Complex(0.0, 0.0));

    // r0 = rhs - A x0 = rhs
    std::copy(rhs, rhs + N, r.begin());

    // z0 = A^H r0 = A r0
    apply_shifted_H(H, r.data(), z.data(), N, sigma);

    // p = z
    std::copy(z.begin(), z.end(), p.begin());

    // norm_rhs
    double norm_rhs = cblas_dznrm2(N, r.data(), 1);
    if (norm_rhs == 0.0) return 0;

    Complex z_dot_z;
    cblas_zdotc_sub(N, z.data(), 1, z.data(), 1, &z_dot_z);

    uint64_t it = 0;
    for (it = 0; it < max_iter; ++it) {
        // Ap = A p
        apply_shifted_H(H, p.data(), Ap.data(), N, sigma);

        // alpha = (z,z) / (Ap,Ap)
        Complex Ap_dot_Ap;
        cblas_zdotc_sub(N, Ap.data(), 1, Ap.data(), 1, &Ap_dot_Ap);
        if (std::abs(Ap_dot_Ap) == 0.0) break;
        Complex alpha = z_dot_z / Ap_dot_Ap;

        // x = x + alpha p
        cblas_zaxpy(N, &alpha, p.data(), 1, x, 1);

        // r = r - alpha Ap
        Complex neg_alpha = -alpha;
        cblas_zaxpy(N, &neg_alpha, Ap.data(), 1, r.data(), 1);

        // check residual
        double nr = cblas_dznrm2(N, r.data(), 1);
        if (nr / norm_rhs <= tol_rel) break;

        // z_new = A^H r = A r
        apply_shifted_H(H, r.data(), Az.data(), N, sigma);

        Complex z_new_dot_z_new;
        cblas_zdotc_sub(N, Az.data(), 1, Az.data(), 1, &z_new_dot_z_new);

        Complex beta = z_new_dot_z_new / z_dot_z;
        z_dot_z = z_new_dot_z_new;

        // p = z_new + beta p
        for (int i = 0; i < N; ++i) {
            p[i] = Az[i] + beta * p[i];
        }
        // z = z_new
        z.swap(Az);
    }
    return it;
}

int solve_shifted_linear_system_Hermitian_CG_or_CGNR(
    const std::function<void(const Complex*, Complex*, int)>& H,
    double sigma,
    const Complex* rhs,
    Complex* x,
    uint64_t N,
    uint64_t max_iter,
    double tol_rel,
    const std::function<void(const Complex*, Complex*, int)>* M_prec) {
    
    // Try (preconditioned) CG first assuming A = (H - sigma I) is HPD; if breakdown, fallback to a robust method.
    std::vector<Complex> r(N), p(N), Ap(N), Ax(N), z(N), delta(N);

    // x0 = 0
    std::fill(x, x + N, Complex(0.0, 0.0));

    // r0 = b - A x0 = b
    std::copy(rhs, rhs + N, r.begin());

    // Preconditioned residual z0 = M^{-1} r0 if provided; else z0 = r0
    if (M_prec && *M_prec) {
        (*M_prec)(r.data(), z.data(), N);
    } else {
        std::copy(r.begin(), r.end(), z.begin());
    }

    // p0 = z0 (PCG)
    std::copy(z.begin(), z.end(), p.begin());

    // norms
    double norm_b = cblas_dznrm2(N, r.data(), 1);
    if (norm_b == 0.0) return 0;

    Complex rsold_c;
    cblas_zdotc_sub(N, r.data(), 1, z.data(), 1, &rsold_c); // r^H z
    double rsold = rsold_c.real();

    uint64_t it = 0;
    bool fell_back = false;
    for (it = 0; it < max_iter; ++it) {
        // Ap = A p
        apply_shifted_H(H, p.data(), Ap.data(), N, sigma);

        // pAp = Re(p^H Ap) (should be > 0 for HPD)
        Complex pAp_c;
        cblas_zdotc_sub(N, p.data(), 1, Ap.data(), 1, &pAp_c);
        double pAp = pAp_c.real();
        if (pAp <= 0.0 || std::isnan(pAp)) { fell_back = true; break; }

        double alpha = rsold / (pAp + 1e-300);
        // x = x + alpha p
        Complex alpha_c(alpha, 0.0);
        cblas_zaxpy(N, &alpha_c, p.data(), 1, x, 1);
        // r = r - alpha Ap
        Complex neg_alpha_c(-alpha, 0.0);
        cblas_zaxpy(N, &neg_alpha_c, Ap.data(), 1, r.data(), 1);

        double nr = cblas_dznrm2(N, r.data(), 1);
        if (nr / norm_b <= tol_rel) {
            return it + 1;
        }

        // z = M^{-1} r if preconditioner provided, else r
        if (M_prec && *M_prec) {
            (*M_prec)(r.data(), z.data(), N);
        } else {
            std::copy(r.begin(), r.end(), z.begin());
        }

        Complex rsnew_c;
        cblas_zdotc_sub(N, r.data(), 1, z.data(), 1, &rsnew_c); // r^H z
        double rsnew = rsnew_c.real();
        double beta = rsnew / (rsold + 1e-300);
        rsold = rsnew;

        // p = z + beta p
        for (int i = 0; i < N; ++i) {
            p[i] = z[i] + Complex(beta, 0.0) * p[i];
        }
    }

    if (!fell_back) return it; // used up iterations

    // Fallback 1: MINRES-like via Conjugate Residuals (robust for Hermitian indefinite)
    auto solve_MINRES_CR = [&](const Complex* b, Complex* y, uint64_t max_it, double tol_rel_inner) -> uint64_t {
        std::vector<Complex> rr(N), pp(N), App(N), Ar(N);
        // y0 = 0
        std::fill(y, y + N, Complex(0.0, 0.0));
        // rr0 = b - A y0 = b
        std::copy(b, b + N, rr.begin());
        double nb = cblas_dznrm2(N, rr.data(), 1);
        if (nb == 0.0) return uint64_t(0);
        // p0 = rr0; App0 = A p0
        std::copy(rr.begin(), rr.end(), pp.begin());
        apply_shifted_H(H, pp.data(), App.data(), N, sigma);

        uint64_t k = 0;
        for (k = 0; k < max_it; ++k) {
            // alpha = (r^H A p) / (A p^H A p)
            Complex rHAp;
            cblas_zdotc_sub(N, rr.data(), 1, App.data(), 1, &rHAp);
            Complex ApHAp;
            cblas_zdotc_sub(N, App.data(), 1, App.data(), 1, &ApHAp);
            if (std::abs(ApHAp) == 0.0) break;
            Complex alpha = rHAp / ApHAp;

            // y = y + alpha p
            cblas_zaxpy(N, &alpha, pp.data(), 1, y, 1);
            // rr = rr - alpha A p
            Complex neg_alpha = -alpha;
            cblas_zaxpy(N, &neg_alpha, App.data(), 1, rr.data(), 1);

            double nrr = cblas_dznrm2(N, rr.data(), 1);
            if (nrr / nb <= tol_rel_inner) break;

            // Ar = A rr
            apply_shifted_H(H, rr.data(), Ar.data(), N, sigma);

            // beta = (r^H A r) / (A p^H A p)
            Complex rHAr;
            cblas_zdotc_sub(N, rr.data(), 1, Ar.data(), 1, &rHAr);
            Complex beta_c = rHAr / ApHAp;

            // p = rr + beta p
            for (int i = 0; i < N; ++i) {
                pp[i] = rr[i] + beta_c * pp[i];
            }
            // A p = A r + beta A p
            for (int i = 0; i < N; ++i) {
                App[i] = Ar[i] + beta_c * App[i];
            }
        }
        return k;
    };

    // Compute correction d to current x using CR, then x += d
    apply_shifted_H(H, x, Ax.data(), N, sigma);
    for (int i = 0; i < N; ++i) delta[i] = Complex(0.0, 0.0);
    std::vector<Complex> rhs_corr(N);
    for (int i = 0; i < N; ++i) rhs_corr[i] = rhs[i] - Ax[i];

    uint64_t it_cr = solve_MINRES_CR(rhs_corr.data(), delta.data(), std::max(uint64_t(1), max_iter - it), tol_rel);
    Complex one(1.0, 0.0);
    cblas_zaxpy(N, &one, delta.data(), 1, x, 1);

    if (it_cr > 0) return it + it_cr;

    // Fallback 2: CGNR as last resort
    uint64_t it2 = solve_shifted_linear_system_CGNR(H, sigma, rhs_corr.data(), delta.data(), N, std::max(uint64_t(1), max_iter - it), tol_rel);
    cblas_zaxpy(N, &one, delta.data(), 1, x, 1);
    return it + it2;
}

int arpack_core(const std::function<void(const Complex*, Complex*, int)>& H,
                uint64_t N, uint64_t max_iter, uint64_t nev, double tol,
                const std::string& which,
                bool shift_invert, double sigma,
                std::vector<double>& evals_out,
                std::vector<Complex>& evecs_out,
                bool want_evecs,
                const std::function<void(const Complex*, Complex*, int)>* M_prec,
                int64_t explicit_ncv,
                bool use_initial_resid,
                const std::vector<Complex>* initial_resid,
                uint64_t inner_max_override,
                double inner_tol_override) {
    
    uint64_t ncv = (explicit_ncv > 0) ? std::min(N, static_cast<uint64_t>(explicit_ncv))
                                 : std::min(N, std::max(uint64_t(4 * nev + 20), uint64_t(30)));
    uint64_t ldv = N;
    
    // ARPACK requires int parameters
    int ido_int = 0;
    int N_int = static_cast<int>(N);
    int nev_int = static_cast<int>(nev);
    int ncv_int = static_cast<int>(ncv);
    int ldv_int = static_cast<int>(ldv);
    
    char bmat = 'I';
    // Sanitize WHICH to valid ARPACK options for complex: {LM, SM, LR, SR, LI, SI}
    std::string W = which;
    std::transform(W.begin(), W.end(), W.begin(), [](unsigned char c){ return std::toupper(c); });
    // For Hermitian matrices: algebraic ordering = real part ordering
    if (W == "SA") W = "SR"; // Smallest Algebraic -> Smallest Real
    if (W == "LA") W = "LR"; // Largest Algebraic -> Largest Real
    if (W == "BE") W = "LR"; // Both Ends not supported in complex interface
    if (!(W == "LM" || W == "SM" || W == "LR" || W == "SR" || W == "LI" || W == "SI")) {
        W = "SM";
    }
    char which_c[3] = {0,0,0};
    which_c[0] = W[0];
    which_c[1] = W[1];
    double tol_a = tol > 0 ? tol : 1e-10;
    uint64_t info = 0;
    std::vector<Complex> resid(N, Complex(0.0, 0.0));
    if (use_initial_resid && initial_resid && (int)initial_resid->size() == N) {
        std::copy(initial_resid->begin(), initial_resid->end(), resid.begin());
        double nrm0 = cblas_dznrm2(N, resid.data(), 1);
        if (nrm0 > 0) {
            Complex sc(1.0 / nrm0, 0.0);
            cblas_zscal(N, &sc, resid.data(), 1);
        } else {
            for (int i = 0; i < N; ++i) {
                resid[i] = Complex((i % 7) * 0.01 + 1e-6, 0.0);
            }
        }
        info = 1;
    }
    std::vector<Complex> V(static_cast<size_t>(ldv) * ncv);
    std::vector<Complex> workd(3 * N);
    uint64_t lworkl = 3 * ncv * ncv + 5 * ncv;
    int lworkl_int = static_cast<int>(lworkl);
    std::vector<Complex> workl(lworkl);
    std::vector<double> rwork(ncv, 0.0);
    int iparam[11] = {0};
    int ipntr[14] = {0};
    int info_int = info;

    iparam[0] = 1;
    iparam[2] = static_cast<int>(std::max(uint64_t(1), max_iter));
    iparam[6] = shift_invert ? 3 : 1;

    uint64_t debug_iter = 0;
    do {
        znaupd_(&ido_int, &bmat, &N_int, which_c, &nev_int, &tol_a, resid.data(),
                &ncv_int, V.data(), &ldv_int, iparam, ipntr, workd.data(),
                workl.data(), &lworkl_int, rwork.data(), &info_int,
                (ftnlen)1, (ftnlen)2);

        if (arpack_debug_enabled) {
            std::cerr << "[ARPACK DEBUG] iter=" << debug_iter++
                      << " ido=" << ido_int
                      << " info=" << info_int
                      << " nev=" << nev
                      << " ncv=" << ncv
                      << " iparam[2](max_iter)=" << iparam[2]
                      << " iparam[4](nconv)=" << iparam[4]
                      << " iparam[8](num_OPx)=" << iparam[8]
                      << " which=" << which_c
                      << std::endl;
        }
        if (ido_int == -1 || ido_int == 1) {
            Complex* x = &workd[ipntr[0] - 1];
            Complex* y = &workd[ipntr[1] - 1];
            if (!shift_invert) {
                apply_H(H, x, y, N);
            } else {
                uint64_t inner_max = (inner_max_override > 0) ? inner_max_override
                                : std::max(uint64_t(50), std::min(N, uint64_t(200)));
                double inner_tol = (inner_tol_override > 0) ? inner_tol_override
                                   : std::max(1e-14, tol * 1e-2);
                solve_shifted_linear_system_Hermitian_CG_or_CGNR(H, sigma, x, y, N, inner_max, inner_tol, M_prec);
            }
        } else if (ido_int == 2) {
            Complex* x = &workd[ipntr[0] - 1];
            Complex* y = &workd[ipntr[1] - 1];
            std::copy(x, x + N, y);
        }
    } while (ido_int != 99);

    if (arpack_debug_enabled) {
        std::cerr << "[ARPACK DEBUG] znaupd exit info=" << info_int
                  << " nconv=" << iparam[4]
                  << " iterations=" << iparam[2]
                  << " num_OPx=" << iparam[8]
                  << std::endl;
    }

    uint64_t nconv = iparam[4];
    if (info_int != 0 && info_int != 1) {
        std::cerr << "ARPACK znaupd failed with info = " << info_int << std::endl;
        return info_int;
    }

    int rvec = want_evecs ? 1 : 0;
    char howmny = 'A';
    std::vector<int> select(ncv, 0);
    std::vector<Complex> D(nev + 1);  // ARPACK may write nev+1 values in some cases
    std::vector<Complex> Z; Z.resize(static_cast<size_t>(N) * (nev + 1));
    Complex SIGMA(sigma, 0.0);
    std::vector<Complex> workev(2 * ncv);

    int ldz = static_cast<int>(N);
    int info2 = 0;

    zneupd_(&rvec, &howmny, select.data(),
            D.data(), Z.data(), &ldz, &SIGMA, workev.data(),
            &bmat, &N_int, which_c, &nev_int, &tol_a, resid.data(),
            &ncv_int, V.data(), &ldv_int, iparam, ipntr, workd.data(),
            workl.data(), &lworkl_int, rwork.data(), &info2,
            (ftnlen)1, (ftnlen)1, (ftnlen)2);

    if (info2 != 0) {
        std::cerr << "ARPACK zneupd failed with info = " << info2 << std::endl;
        if (arpack_debug_enabled) {
            std::cerr << "[ARPACK DEBUG] zneupd failure diagnostics:\n"
                      << "  info2=" << info2 << " ( -14 often means: not enough Ritz values satisfied convergence or workspace corrupted )\n"
                      << "  N=" << N << " nev=" << nev << " ncv=" << ncv << " nconv=" << nconv << " (require ncv >> nev)\n"
                      << "  which=" << which_c << " tol=" << tol_a << " shift_invert=" << shift_invert << " sigma=" << sigma << "\n"
                      << "  Suggest: increase max_iter, increase ncv, relax tol, or change WHICH." << std::endl;
        }
        return info2;
    }

    uint64_t have = std::min(nconv, nev);
    if (have <= 0) {
        std::cerr << "No converged Ritz pairs (nconv=" << nconv << ")" << std::endl;
        return info;
    }

    evals_out.resize(have);
    for (int i = 0; i < have; ++i) {
        evals_out[i] = std::real(D[i]);
    }

    std::vector<int> perm(have);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, uint64_t b) { return evals_out[a] < evals_out[b]; });

    std::vector<double> evals_sorted(have);
    for (int i = 0; i < have; ++i) evals_sorted[i] = evals_out[perm[i]];
    evals_out.swap(evals_sorted);

    if (want_evecs) {
        evecs_out.resize(static_cast<size_t>(N) * have);
        for (int k = 0; k < have; ++k) {
            uint64_t src = perm[k];
            for (int i = 0; i < N; ++i) {
                evecs_out[static_cast<size_t>(k) * N + i] = Z[static_cast<size_t>(src) * N + i];
            }
            double nrm = cblas_dznrm2(N, &evecs_out[static_cast<size_t>(k) * N], 1);
            if (nrm > 0) {
                Complex sc(1.0 / nrm, 0.0);
                cblas_zscal(N, &sc, &evecs_out[static_cast<size_t>(k) * N], 1);
                // Phase normalization: make largest component real and positive
                uint64_t imax = 0; double amax = 0.0;
                for (int i = 0; i < N; ++i) {
                    double a = std::abs(evecs_out[static_cast<size_t>(k) * N + i]);
                    if (a > amax) { amax = a; imax = i; }
                }
                if (amax > 1e-14) {
                    Complex v = evecs_out[static_cast<size_t>(k) * N + imax];
                    Complex phase = v / std::abs(v);  // unit phase factor
                    Complex phase_conj = std::conj(phase);
                    cblas_zscal(N, &phase_conj, &evecs_out[static_cast<size_t>(k) * N], 1);
                }
            }
        }
    }

    return (have > 0) ? 0 : info;
}

int arpack_core_advanced(const std::function<void(const Complex*, Complex*, int)>& H,
                         uint64_t N,
                         const ArpackAdvancedOptions& opts,
                         std::vector<double>& evals_out,
                         std::vector<Complex>& evecs_out,
                         bool want_evecs) {
    
    uint64_t base_ncv = (opts.ncv > 0) ? opts.ncv
                    : std::min(N, std::max(uint64_t((opts.which == "SM" || opts.which == "SR" ? 6 : 4) * opts.nev + 40), uint64_t(30)));
    double target_tol = opts.tol;
    double first_tol = (opts.two_phase_refine && opts.relaxed_tol > target_tol) ? opts.relaxed_tol : target_tol;

    bool tried_shift_invert = opts.shift_invert;
    bool use_shift_invert = opts.shift_invert;
    double sigma = opts.shift_invert ? opts.sigma : 0.0;

    uint64_t attempts = opts.max_restarts + 1;
    uint64_t attempt_idx = 0;
    uint64_t last_info = 0;
    std::vector<double> best_evals;
    std::vector<Complex> best_evecs;
    uint64_t best_converged = 0;

    for (attempt_idx = 0; attempt_idx < attempts; ++attempt_idx) {
        double tol_this = (attempt_idx == 0) ? first_tol : target_tol;
        uint64_t ncv_attempt = base_ncv;
        if (attempt_idx > 0 && opts.auto_enlarge_ncv) {
            double growth = std::pow(opts.ncv_growth, attempt_idx);
            ncv_attempt = std::min(N, static_cast<uint64_t>(std::ceil(base_ncv * growth)));
            ncv_attempt = std::max(ncv_attempt, std::min(N, opts.nev + 10));
        }
        // Ensure ncv > nev and ncv <= N
        ncv_attempt = std::max(ncv_attempt, opts.nev + 2);
        ncv_attempt = std::min(ncv_attempt, N);
        uint64_t inner_max = opts.inner_max_iter;
        double inner_tol = -1.0;
        if (use_shift_invert) {
            if (opts.adaptive_inner_tol) {
                // Clamp ratio to prevent extreme values
                double ratio = std::max(0.1, std::min(10.0, tol_this / target_tol));
                inner_tol = std::max(opts.inner_tol_min, ratio * opts.inner_tol_factor * tol_this);
            } else {
                inner_tol = std::max(opts.inner_tol_min, opts.inner_tol_factor * tol_this);
            }
        }

        if (opts.verbose) {
            std::cerr << "[ARPACK ADV] attempt=" << attempt_idx
                      << " use_shift_invert=" << use_shift_invert
                      << " sigma=" << sigma
                      << " ncv=" << ncv_attempt
                      << " tol=" << tol_this
                      << (inner_tol>0?(" inner_tol="+std::to_string(inner_tol)) : "")
                      << std::endl;
        }

        std::vector<double> evals_cur;
        std::vector<Complex> evecs_cur;
        uint64_t info = arpack_core(H, N, opts.max_iter, opts.nev, tol_this, opts.which,
                               use_shift_invert, sigma, evals_cur, evecs_cur, want_evecs,
                               (opts.M_prec ? &opts.M_prec : nullptr),
                               ncv_attempt,
                               opts.use_initial_resid, opts.initial_resid,
                               use_shift_invert ? inner_max : -1,
                               use_shift_invert ? inner_tol : -1.0);
        last_info = info;

        uint64_t converged = static_cast<int>(evals_cur.size());
        if (converged > best_converged) {
            best_converged = converged;
            best_evals.swap(evals_cur);
            if (want_evecs) best_evecs.swap(evecs_cur);
        }

        if (info == 0 && converged >= opts.nev) {
            evals_out = std::move(best_evals);
            if (want_evecs) evecs_out = std::move(best_evecs);
            if (opts.verbose) std::cerr << "[ARPACK ADV] success on attempt " << attempt_idx << std::endl;
            return 0;
        }

        if (attempt_idx + 1 < attempts) continue;

        if (!use_shift_invert && opts.auto_switch_to_shift_invert && !tried_shift_invert) {
            if (opts.verbose) std::cerr << "[ARPACK ADV] switching to shift-invert fallback..." << std::endl;
            tried_shift_invert = true;
            use_shift_invert = true;
            sigma = opts.switch_sigma;
            attempts += 1;
            continue;
        }
    }

    if (best_converged > 0) {
        evals_out = std::move(best_evals);
        if (want_evecs) evecs_out = std::move(best_evecs);
    }
    return last_info == 0 ? (best_converged >= opts.nev ? 0 : 1) : last_info;
}

void save_eigs_to_dir(const std::vector<double>& evals,
                      const std::vector<Complex>* evecs,
                      uint64_t N, uint64_t nev, const std::string& dir) {
    if (dir.empty()) return;
    std::string evec_dir = dir + "/eigenvectors";
    std::string cmd = "mkdir -p " + evec_dir;
    safe_system_call(cmd);

    {
        std::string fbin = evec_dir + "/eigenvalues.dat";
        std::ofstream ofs(fbin, std::ios::binary);
        if (ofs) {
            size_t n = evals.size();
            ofs.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
            ofs.write(reinterpret_cast<const char*>(evals.data()), sizeof(double) * n);
        }
    }
    {
        std::string ftxt = evec_dir + "/eigenvalues.txt";
        std::ofstream ofs(ftxt);
        if (ofs) {
            ofs << std::scientific << std::setprecision(15);
            for (double v : evals) ofs << v << "\n";
        }
    }

    if (evecs) {
        for (int k = 0; k < nev; ++k) {
            std::string fbin = evec_dir + "/evec_" + std::to_string(k) + ".dat";
            std::ofstream ofs(fbin, std::ios::binary);
            if (ofs) {
                const Complex* col = &(*evecs)[static_cast<size_t>(k) * N];
                ofs.write(reinterpret_cast<const char*>(col), sizeof(Complex) * N);
            }
        }
    }
}

} // namespace detail_arpack

// Public API implementations
void arpack_eigs(std::function<void(const Complex*, Complex*, int)> H, uint64_t N,
                 uint64_t max_iter, uint64_t exct, double tol,
                 std::vector<double>& eigenvalues,
                 std::string dir, bool eigenvectors,
                 const std::string& which) {
    std::vector<Complex> evecs;
    uint64_t info = detail_arpack::arpack_core(H, N, max_iter, exct, tol, which, false, 0.0,
                                          eigenvalues, evecs, eigenvectors);
    if (info != 0) {
        std::cerr << "arpack_eigs failed with info=" << info << std::endl;
        return;
    }
    if (eigenvectors) {
        detail_arpack::save_eigs_to_dir(eigenvalues, &evecs, N, static_cast<int>(eigenvalues.size()), dir);
    } else {
        detail_arpack::save_eigs_to_dir(eigenvalues, nullptr, N, 0, dir);
    }
}

void arpack_ground_state(std::function<void(const Complex*, Complex*, int)> H, uint64_t N,
                         uint64_t max_iter, uint64_t exct, double tol,
                         std::vector<double>& eigenvalues,
                         std::string dir, bool eigenvectors) {
    // For Hermitian matrices, "SR" finds algebraically smallest (most negative) eigenvalues
    // "SM" would find eigenvalues closest to zero, which is wrong for ground state
    arpack_eigs(H, N, max_iter, exct, tol, eigenvalues, dir, eigenvectors, "SR");
}

void arpack_largest(std::function<void(const Complex*, Complex*, int)> H, uint64_t N,
                    uint64_t max_iter, uint64_t exct, double tol,
                    std::vector<double>& eigenvalues,
                    std::string dir, bool eigenvectors) {
    // For Hermitian matrices, "LR" finds algebraically largest (most positive) eigenvalues
    arpack_eigs(H, N, max_iter, exct, tol, eigenvalues, dir, eigenvectors, "LR");
}

void arpack_shift_invert(std::function<void(const Complex*, Complex*, int)> H, uint64_t N,
                         uint64_t max_iter, uint64_t num_eigs, double sigma, double tol,
                         std::vector<double>& eigenvalues,
                         std::string dir, bool compute_eigenvectors) {
    std::vector<Complex> evecs;
    uint64_t info = detail_arpack::arpack_core(H, N, max_iter, num_eigs, tol, "LM", true, sigma,
                                          eigenvalues, evecs, compute_eigenvectors);
    if (info != 0) {
        std::cerr << "arpack_shift_invert failed with info=" << info << std::endl;
        return;
    }
    if (compute_eigenvectors) {
        detail_arpack::save_eigs_to_dir(eigenvalues, &evecs, N, static_cast<int>(eigenvalues.size()), dir);
    } else {
        detail_arpack::save_eigs_to_dir(eigenvalues, nullptr, N, 0, dir);
    }
}

void arpack_shift_invert_prec(std::function<void(const Complex*, Complex*, int)> H, uint64_t N,
                              uint64_t max_iter, uint64_t num_eigs, double sigma, double tol,
                              std::function<void(const Complex*, Complex*, int)> M_prec,
                              std::vector<double>& eigenvalues,
                              std::string dir, bool compute_eigenvectors) {
    std::vector<Complex> evecs;
    uint64_t info = detail_arpack::arpack_core(H, N, max_iter, num_eigs, tol, "LM", true, sigma,
                                          eigenvalues, evecs, compute_eigenvectors, &M_prec);
    if (info != 0) {
        std::cerr << "arpack_shift_invert_prec failed with info=" << info << std::endl;
        return;
    }
    if (compute_eigenvectors) {
        detail_arpack::save_eigs_to_dir(eigenvalues, &evecs, N, static_cast<int>(eigenvalues.size()), dir);
    } else {
        detail_arpack::save_eigs_to_dir(eigenvalues, nullptr, N, 0, dir);
    }
}

int arpack_eigs_advanced(std::function<void(const Complex*, Complex*, int)> H, uint64_t N,
                         const detail_arpack::ArpackAdvancedOptions& opts,
                         std::vector<double>& eigenvalues,
                         std::string dir, bool eigenvectors,
                         std::vector<Complex>* out_evecs) {
    std::vector<Complex> evecs_local;
    std::vector<Complex>& evecs_ref = out_evecs ? *out_evecs : evecs_local;
    uint64_t info = detail_arpack::arpack_core_advanced(H, N, opts, eigenvalues, evecs_ref, eigenvectors);
    if (info != 0 && opts.verbose) {
        std::cerr << "arpack_eigs_advanced: final info=" << info
                  << " eigenvalues_found=" << eigenvalues.size() << std::endl;
    }
    if ((eigenvectors || out_evecs) && !evecs_ref.empty()) {
        detail_arpack::save_eigs_to_dir(eigenvalues, &evecs_ref, N, static_cast<int>(eigenvalues.size()), dir);
    } else {
        detail_arpack::save_eigs_to_dir(eigenvalues, nullptr, N, 0, dir);
    }
    return info;
}

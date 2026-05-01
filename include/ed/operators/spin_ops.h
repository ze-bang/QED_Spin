// =============================================================================
// include/ed/operators/spin_ops.h
//
// Factory functions for common spin-1/2 operators in the FULL computational
// (Fock) basis of N spins, dimension D = 2^N.
//
// Bit / spin convention (matches construct_ham.h and ftlm.cpp):
//   bit i = 0  -> spin UP   (+1/2)
//   bit i = 1  -> spin DOWN (-1/2)
//   state s is an integer in [0, 2^N), bits numbered from 0 (site 0) to N-1.
//
// All factories return a MatVec (= std::function<void(const Complex*,Complex*,int)>)
// that can be passed directly to ed::ftlm::jp::compute_dynamical_correlation,
// ed::ltlm::compute_ltlm_dynamical_correlation, or any other kernel in this
// library.
//
// Operators
// ---------
//   Sz_i           diagonal:  +1/2 for UP, -1/2 for DOWN
//   Sp_i = S+_i    raising:   DOWN -> UP  (bit 1 -> 0)
//   Sm_i = S-_i    lowering:  UP -> DOWN  (bit 0 -> 1)
//
// Fourier-space operators (momentum-space):
//   Sz_q(q)  = (1/sqrt(N)) sum_j exp(i q r_j) Sz_j
//   Sp_q(q)  = (1/sqrt(N)) sum_j exp(i q r_j) Sp_j
//   Sm_q(q)  = (1/sqrt(N)) sum_j exp(i q r_j) Sm_j
//
//   where r_j = j (lattice spacing a=1) and q is a continuous real number
//   (typically 2 pi k / N for k = 0, 1, ..., N-1 for a 1-D chain).
//
// Notes
// -----
// * These operators act on the full 2^N Hilbert space.  If you are working
//   inside a fixed-Sz sector with a sector-compressed basis, you need a
//   sector-aware MatVec instead (see compute_dynamical_correlation_cross_sector).
//
// * Sp_i and Sm_i do NOT conserve Sz, so when used with JP / LTLM in the
//   full basis, the sampled initial vectors span all Sz sectors and the
//   resulting spectral function contains contributions from all sectors.
//   This is correct and matches the textbook Lehmann sum.
//
// * The Fourier operators Sp_q / Sm_q produce the transverse DSF:
//     S^{+-}(q, omega, T) = <Sp_q^dagger Sm_q>(omega, T)
//   which is computed by passing O1 = Sm_q, O2 = Sm_q to the correlator
//   and using the convention S^{+-} = <S^+ S^->.  Alternatively, pass
//   O1 = O2 = Sp_q (self-correlation) to get |<m|Sp_q|n>|^2.
// =============================================================================

#pragma once

#include <complex>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include <ed/core/matvec_types.h>

namespace ed::ops {

using Complex = ed::types::Complex;
using MatVec  = ed::types::MatVec;

// ---------------------------------------------------------------------------
// Site-local operators
// ---------------------------------------------------------------------------

/// S^z_i:  |s> -> (+1/2 if bit i = 0, -1/2 if bit i = 1) |s>
inline MatVec make_Sz_site(std::uint64_t site, std::uint64_t N) {
    if (site >= N)
        throw std::out_of_range("make_Sz_site: site >= N");
    const std::uint64_t mask = 1ULL << site;
    const std::uint64_t dim  = 1ULL << N;
    return [mask, dim](const Complex* in, Complex* out, int /*n*/) {
        for (std::uint64_t s = 0; s < dim; ++s) {
            const bool down = ((s & mask) != 0);
            out[s] = (down ? -0.5 : +0.5) * in[s];
        }
    };
}

/// S^+_i:  |...DOWN_i...> -> |...UP_i...>   (bit i: 1 -> 0)
///         |...UP_i...>   -> 0
inline MatVec make_Sp_site(std::uint64_t site, std::uint64_t N) {
    if (site >= N)
        throw std::out_of_range("make_Sp_site: site >= N");
    const std::uint64_t mask = 1ULL << site;
    const std::uint64_t dim  = 1ULL << N;
    return [mask, dim](const Complex* in, Complex* out, int /*n*/) {
        for (std::uint64_t s = 0; s < dim; ++s) out[s] = Complex(0.0, 0.0);
        for (std::uint64_t s = 0; s < dim; ++s) {
            if ((s & mask) != 0) {           // bit set = DOWN
                const std::uint64_t t = s ^ mask;   // clear bit = UP
                out[t] += in[s];
            }
        }
    };
}

/// S^-_i:  |...UP_i...>   -> |...DOWN_i...>   (bit i: 0 -> 1)
///         |...DOWN_i...> -> 0
inline MatVec make_Sm_site(std::uint64_t site, std::uint64_t N) {
    if (site >= N)
        throw std::out_of_range("make_Sm_site: site >= N");
    const std::uint64_t mask = 1ULL << site;
    const std::uint64_t dim  = 1ULL << N;
    return [mask, dim](const Complex* in, Complex* out, int /*n*/) {
        for (std::uint64_t s = 0; s < dim; ++s) out[s] = Complex(0.0, 0.0);
        for (std::uint64_t s = 0; s < dim; ++s) {
            if ((s & mask) == 0) {           // bit clear = UP
                const std::uint64_t t = s | mask;   // set bit = DOWN
                out[t] += in[s];
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Fourier-space operators  (1-D chain, r_j = j, lattice spacing a=1)
// ---------------------------------------------------------------------------

/// S^z_q = (1/sqrt(N)) sum_j exp(i*q*j) S^z_j
///
/// Diagonal operator; result[s] = (1/sqrt(N)) sum_j exp(i q j) sz_j(s) * in[s]
inline MatVec make_Sz_q(double q, std::uint64_t N) {
    const std::uint64_t dim = 1ULL << N;
    const double inv_sqrtN  = 1.0 / std::sqrt(static_cast<double>(N));

    // Precompute site weights.
    std::vector<Complex> wj(N);
    for (std::uint64_t j = 0; j < N; ++j)
        wj[j] = inv_sqrtN * Complex(std::cos(q * j), std::sin(q * j));

    // For each basis state s precompute the diagonal matrix element
    //   coeff[s] = sum_j wj[j] * sz_j(s)
    std::vector<Complex> coeff(dim, Complex(0.0, 0.0));
    for (std::uint64_t s = 0; s < dim; ++s) {
        Complex c(0.0, 0.0);
        for (std::uint64_t j = 0; j < N; ++j) {
            const bool down = ((s >> j) & 1ULL) != 0;
            c += wj[j] * (down ? -0.5 : +0.5);
        }
        coeff[s] = c;
    }

    return [coeff, dim](const Complex* in, Complex* out, int /*n*/) {
        for (std::uint64_t s = 0; s < dim; ++s)
            out[s] = coeff[s] * in[s];
    };
}

/// S^+_q = (1/sqrt(N)) sum_j exp(i*q*j) S^+_j
///
/// Off-diagonal; for each DOWN site j: |s> -> exp(iq j)/sqrt(N) |s with bit j cleared>
inline MatVec make_Sp_q(double q, std::uint64_t N) {
    const std::uint64_t dim = 1ULL << N;
    const double inv_sqrtN  = 1.0 / std::sqrt(static_cast<double>(N));

    std::vector<Complex> wj(N);
    for (std::uint64_t j = 0; j < N; ++j)
        wj[j] = inv_sqrtN * Complex(std::cos(q * j), std::sin(q * j));

    return [wj, dim, N](const Complex* in, Complex* out, int /*n*/) {
        for (std::uint64_t s = 0; s < dim; ++s) out[s] = Complex(0.0, 0.0);
        for (std::uint64_t s = 0; s < dim; ++s) {
            if (in[s] == Complex(0.0, 0.0)) continue;
            for (std::uint64_t j = 0; j < N; ++j) {
                if ((s >> j) & 1ULL) {               // bit set = DOWN
                    const std::uint64_t t = s ^ (1ULL << j);  // clear bit
                    out[t] += wj[j] * in[s];
                }
            }
        }
    };
}

/// S^-_q = (1/sqrt(N)) sum_j exp(i*q*j) S^-_j
///
/// Hermitian conjugate of S^+_{-q}.
inline MatVec make_Sm_q(double q, std::uint64_t N) {
    const std::uint64_t dim = 1ULL << N;
    const double inv_sqrtN  = 1.0 / std::sqrt(static_cast<double>(N));

    std::vector<Complex> wj(N);
    for (std::uint64_t j = 0; j < N; ++j)
        wj[j] = inv_sqrtN * Complex(std::cos(q * j), std::sin(q * j));

    return [wj, dim, N](const Complex* in, Complex* out, int /*n*/) {
        for (std::uint64_t s = 0; s < dim; ++s) out[s] = Complex(0.0, 0.0);
        for (std::uint64_t s = 0; s < dim; ++s) {
            if (in[s] == Complex(0.0, 0.0)) continue;
            for (std::uint64_t j = 0; j < N; ++j) {
                if (!((s >> j) & 1ULL)) {             // bit clear = UP
                    const std::uint64_t t = s | (1ULL << j);  // set bit
                    out[t] += wj[j] * in[s];
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Convenience: sum of Sz over all sites (total Sz / particle-number proxy)
// ---------------------------------------------------------------------------

/// Total S^z = sum_i S^z_i  (diagonal, no prefactor)
inline MatVec make_Sz_total(std::uint64_t N) {
    const std::uint64_t dim = 1ULL << N;
    return [dim, N](const Complex* in, Complex* out, int /*n*/) {
        for (std::uint64_t s = 0; s < dim; ++s) {
            int n_up = 0;
            for (std::uint64_t j = 0; j < N; ++j)
                if (!((s >> j) & 1ULL)) ++n_up;   // bit 0 = UP
            const double sz = n_up - 0.5 * static_cast<double>(N);
            out[s] = sz * in[s];
        }
    };
}

} // namespace ed::ops

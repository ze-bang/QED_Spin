#pragma once
// =============================================================================
// include/ed/symmetry/time_reversal.h
//
// Time reversal as sector PAIRING metadata -- deliberately NOT a projector
// (an antiunitary element cannot enter P = (1/|G|) sum chi*(g) g).
//
// For a Hamiltonian whose computational-basis matrix elements are REAL
// (every term coefficient real -- the isReal() fast-path condition, i.e.
// H is time-reversal even up to the spin basis convention), complex
// conjugation K satisfies K H K = H and maps the momentum sector with
// characters chi_k to the sector with chi_k* = chi_{-k}. Hence
//
//     spec(H | k)  ==  spec(H | -k),    dim(k) == dim(-k),
//     Z_k(beta)    ==  Z_{-k}(beta).
//
// The sector walk therefore solves ONE member of each conjugate pair
// {k, -k} and copies the result to the partner; self-conjugate sectors
// (real characters: k = 0, k = pi, parity irreps) have real blocks when H is
// real.
//
// Kramers bookkeeping (T^2 = -1 degeneracy tags for odd spin-1/2 counts)
// is not implemented: it would only affect degeneracy-aware convergence
// heuristics, not any produced number.
// =============================================================================

#include <ed/config/env_registry.h>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <ed/matvec/term_storage.h>

namespace ed::symmetry {

/// True iff every term coefficient is real (imag <= tol): the condition
/// under which complex conjugation commutes with H in the computational
/// basis.
[[nodiscard]] inline bool
hamiltonian_is_real(const ed::matvec::TermStorage& t,
                    double tol = 1e-14) noexcept {
    auto ok = [tol](const std::complex<double>& c) {
        return std::abs(c.imag()) <= tol;
    };
    for (const auto& x : t.diag_one_body)    if (!ok(x.coefficient)) return false;
    for (const auto& x : t.offdiag_one_body) if (!ok(x.coefficient)) return false;
    for (const auto& x : t.diag_two_body)    if (!ok(x.coefficient)) return false;
    for (const auto& x : t.mixed_two_body)   if (!ok(x.coefficient)) return false;
    for (const auto& x : t.offdiag_two_body) if (!ok(x.coefficient)) return false;
    for (const auto& x : t.three_body)       if (!ok(x.coefficient)) return false;
    return true;
}

}  // namespace ed::symmetry

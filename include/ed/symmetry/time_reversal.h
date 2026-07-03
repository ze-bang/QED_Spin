#pragma once
// =============================================================================
// include/ed/symmetry/time_reversal.h
//
// Stage 6 of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md): time reversal as sector
// PAIRING metadata -- deliberately NOT a projector (an antiunitary
// element cannot enter P = (1/|G|) sum chi*(g) g).
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
// The sector loop therefore solves ONE member of each conjugate pair
// {k, -k} and copies the result to the partner; self-conjugate sectors
// (real characters: k = 0, k = pi, parity irreps) additionally take the
// existing real-Hermitian fast path (`is_real_hermitian()` already
// consults the per-sector character -- that half of Stage 6 was wired by
// the rep-lazy machinery).
//
// Kramers bookkeeping (T^2 = -1 degeneracy tags for odd spin-1/2 counts)
// is deferred: it only affects degeneracy-aware convergence heuristics,
// not any produced number.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <ed/core/symmetry_metadata.h>  // SymmetryGroupInfo
#include <ed/matvec/term_storage.h>

namespace ed::symmetry {

/// Env gate for the Stage-6 pairing (default ON; ED_SYM_TIME_REVERSAL=0
/// disables). Read per call so tests can toggle from Python.
[[nodiscard]] inline bool time_reversal_pairing_enabled() noexcept {
    const char* v = std::getenv("ED_SYM_TIME_REVERSAL");
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
}

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

/// Conjugate-sector pairing: partner[s] = index of the sector whose
/// per-generator phase factors are the complex conjugates of sector s's
/// (i.e. the -k sector), or -1 when no match is found (malformed
/// metadata -- callers must then skip pairing for s). Self-conjugate
/// sectors (real characters) map to themselves.
[[nodiscard]] inline std::vector<std::int32_t>
conjugate_sector_pairing(const SymmetryGroupInfo& info,
                         double tol = 1e-10) {
    const std::size_t n = info.sectors.size();
    std::vector<std::int32_t> partner(n, -1);
    for (std::size_t s = 0; s < n; ++s) {
        const auto& pf = info.sectors[s].phase_factors;
        for (std::size_t t = 0; t < n; ++t) {
            const auto& qf = info.sectors[t].phase_factors;
            if (qf.size() != pf.size()) continue;
            bool match = true;
            for (std::size_t g = 0; g < pf.size(); ++g) {
                if (std::abs(qf[g] - std::conj(pf[g])) > tol) {
                    match = false;
                    break;
                }
            }
            if (match) {
                partner[s] = static_cast<std::int32_t>(t);
                break;
            }
        }
    }
    return partner;
}

}  // namespace ed::symmetry

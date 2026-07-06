#pragma once
// =============================================================================
// include/ed/symmetry/observable_character.h
//
// Stage 8d (SymmetryEngine v2): term-level classifiers for PROBE observables.
//
// The projected spectral lanes (Sz-parity halves, full-space prod-sigma^x
// flip sectors) need to know how a probe observable O moves between
// synthetic sectors:
//
//   * ``spin_flip_character``: X O X with X = prod_i sigma^x_i equals
//     +O (flip-EVEN, stays in the same (k, s) flip slot), -O (flip-ODD,
//     connects (k, +) <-> (k+Q, -)), or neither (the projected flip lane
//     must be declined for this probe). Note X S^z X = -S^z: a
//     momentum-resolved S^z_Q probe is flip-ODD; S^x_Q = (S^+ + S^-)/2
//     is flip-EVEN; a lone S^+_Q has no definite character.
//
//   * ``delta_n_up_parity``: whether every term changes the set-bit count
//     by the same amount mod 2 (0 = parity-even probe, stays in the same
//     Sz-parity half; 1 = parity-odd, crosses halves; -1 = mixed, decline).
//     This is the mod-2 shadow of the exact selection rule
//     ``_infer_delta_n_up`` uses on the U(1) lane, and accepts probes
//     (e.g. S^x = (S^+ + S^-)/2) that have no single exact delta.
//
// Both walk the same 6-field ``Operator::TransformData`` rows the DSSF
// bindings receive, mirroring ``hamiltonian_is_spin_flip_symmetric``
// (spin_flip.h) which answers the same question for H's TermStorage.
// =============================================================================

#include <ed/core/operator.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace ed::symmetry {

namespace detail {

// Canonical (sorted-key, coefficient-accumulated) term multiset. ``flip``
// applies the global spin flip: S+ <-> S- (op code 0 <-> 1), S^z -> -S^z
// (coefficient negation). Factors on DISTINCT sites commute, so two-body
// keys are ordered by site; same-site pairs keep their order (S+S- on one
// site does not commute with its swap).
using TermKey = std::tuple<int, std::uint64_t, int, std::uint64_t>;

inline std::map<TermKey, std::complex<double>>
canonical_probe_terms(const std::vector<Operator::TransformData>& ts,
                      bool flip)
{
    std::map<TermKey, std::complex<double>> m;
    for (const auto& t : ts) {
        std::complex<double> c = t.coefficient;
        int o1 = static_cast<int>(t.op_type);
        int o2 = t.is_two_body ? static_cast<int>(t.op_type_2) : -1;
        if (flip) {
            auto f = [&c](int& o) {
                if (o == 2) c = -c;      // X Sz X = -Sz
                else        o ^= 1;      // X S+- X = S-+
            };
            f(o1);
            if (t.is_two_body) f(o2);
        }
        std::uint64_t s1 = t.site_index;
        std::uint64_t s2 = t.is_two_body ? t.site_index_2 : 0;
        if (t.is_two_body && s2 < s1) {  // distinct sites commute
            std::swap(o1, o2);
            std::swap(s1, s2);
        }
        m[{o1, s1, o2, s2}] += c;
    }
    for (auto it = m.begin(); it != m.end();) {
        it = (std::abs(it->second) < 1e-14) ? m.erase(it) : std::next(it);
    }
    return m;
}

}  // namespace detail

/// +1: X O X == +O (flip-even).  -1: X O X == -O (flip-odd).
///  0: neither -- the probe has no definite flip character and the
///     flip-projected spectral lane cannot route it.
[[nodiscard]] inline int
spin_flip_character(const std::vector<Operator::TransformData>& ts)
{
    const auto a = detail::canonical_probe_terms(ts, /*flip=*/false);
    const auto b = detail::canonical_probe_terms(ts, /*flip=*/true);
    if (a.empty() || a.size() != b.size()) return 0;
    bool even = true, odd = true;
    for (const auto& [k, v] : a) {
        const auto it = b.find(k);
        if (it == b.end()) return 0;
        const double scale = 1.0 + std::abs(v);
        if (std::abs(it->second - v) > 1e-12 * scale) even = false;
        if (std::abs(it->second + v) > 1e-12 * scale) odd  = false;
        if (!even && !odd) return 0;
    }
    return even ? +1 : (odd ? -1 : 0);
}

/// 0: every term preserves set-bit parity (probe stays in its Sz-parity
///    half). 1: every term changes it (probe crosses halves). -1: mixed.
[[nodiscard]] inline int
delta_n_up_parity(const std::vector<Operator::TransformData>& ts)
{
    int par = -2;
    for (const auto& t : ts) {
        int d = (t.op_type == 2) ? 0 : 1;   // any S+- toggles one bit
        if (t.is_two_body) d += (t.op_type_2 == 2) ? 0 : 1;
        d &= 1;
        if (par == -2)     par = d;
        else if (par != d) return -1;
    }
    return (par < 0) ? 0 : par;
}

/// True when ``b == conj(a)`` as term multisets: identical operator
/// structure, elementwise-conjugated coefficients. This is the exact
/// gate for the multi-Q spectral sweep's time-reversal panel copy: for
/// a REAL H the eigenbasis can be chosen real, so
///
///   <n| conj(O) |0> = conj(<n| O |0>)   =>   S_{conj(O)} == S_O
///
/// for EVERY probe channel (including Sz-changing S^+- probes), and the
/// physical -Q probe of a channel is precisely the coefficient
/// conjugation (e^{iQ.r} -> e^{-iQ.r}).
[[nodiscard]] inline bool
transforms_are_conjugate(const std::vector<Operator::TransformData>& a,
                         const std::vector<Operator::TransformData>& b)
{
    std::vector<Operator::TransformData> ca = a;
    for (auto& t : ca) t.coefficient = std::conj(t.coefficient);
    const auto ma = detail::canonical_probe_terms(ca, /*flip=*/false);
    const auto mb = detail::canonical_probe_terms(b,  /*flip=*/false);
    if (ma.empty() || ma.size() != mb.size()) return false;
    for (const auto& [k, v] : ma) {
        const auto it = mb.find(k);
        if (it == mb.end()) return false;
        if (std::abs(it->second - v) > 1e-12 * (1.0 + std::abs(v)))
            return false;
    }
    return true;
}

}  // namespace ed::symmetry

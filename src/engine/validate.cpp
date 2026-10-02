// =============================================================================
// src/engine/validate.cpp -- see validate.h.
// =============================================================================
#include "validate.h"

#include <ed/core/errors.h>
#include <ed/ops/invariance.h>

#include <cmath>
#include <cstdio>
#include <string>

namespace ed::sectors::detail {

namespace {

[[noreturn]] void refuse(const char* verb, const std::string& what) {
    throw ed::InvalidRequest(std::string(verb) + ": " + what);
}

std::string num(double x) {
    char b[32];
    std::snprintf(b, sizeof b, "%.6g", x);
    return b;
}

}  // namespace

void validate_hamiltonian(const ::Operator& H, const char* verb) {
    const auto n = H.getNumBits();
    if (n < 1 || n > 63) refuse(verb, "H acts on " + std::to_string(n) + " sites; 1..63 are supported");
    if (!ed::ops::hermitian(H.canonical()))
        refuse(verb, "H is not Hermitian (H - H^dagger exceeds 1e-10 of its largest coefficient); its "
                     "eigenvalues need not be real");
}

void validate_spec(const Spec& s, int n_sites, const char* verb) {
    if (s.n_up < -1 || s.n_up > n_sites)
        refuse(verb, "the Sz sector n_up = " + std::to_string(s.n_up) + " is outside 0.." + std::to_string(n_sites)
                     + " (the number of up spins)");
    if (s.sz_parity < -1 || s.sz_parity > 1)
        refuse(verb, "sz_parity = " + std::to_string(s.sz_parity) + " is not 0 (even), 1 (odd) or -1 (both)");
    if (s.spin_flip < -1 || s.spin_flip > 1)
        refuse(verb, "spin_flip = " + std::to_string(s.spin_flip) + " is not -1 (auto), 0 (off) or 1 (require)");
    if (s.time_reversal < -1 || s.time_reversal > 1)
        refuse(verb, "time_reversal = " + std::to_string(s.time_reversal) + " is not -1, 0 or 1");
    if (s.two_S < -1 || s.two_S > n_sites)
        refuse(verb, "total spin 2S = " + std::to_string(s.two_S) + " is outside 0.." + std::to_string(n_sites));
    if (s.two_S >= 0 && (n_sites - s.two_S) % 2 != 0)
        refuse(verb, "total spin S = " + std::to_string(s.two_S) + "/2 does not exist for N = "
                     + std::to_string(n_sites));
}

void validate_observable(const ::Operator* O, int n_sites, const char* verb, std::size_t index) {
    if (!O) refuse(verb, "observable " + std::to_string(index) + " is None");
    if (static_cast<int>(O->getNumBits()) != n_sites)
        refuse(verb, "observable " + std::to_string(index) + " acts on " + std::to_string(O->getNumBits())
                     + " sites, H on " + std::to_string(n_sites));
}

void validate_eigs_options(const EigsOptions& o) {
    if (o.k < 1) refuse("eigs", "k must be >= 1, got " + std::to_string(o.k));
    if (o.dense_max_dim < -1) refuse("eigs", "dense_max_dim must be >= 0 (or -1: automatic)");
    if (o.per_block < 0) refuse("eigs", "per_block must be >= 0");
    if (!(o.prune_margin >= 0.0) || !std::isfinite(o.prune_margin))
        refuse("eigs", "prune_margin must be finite and >= 0, got " + num(o.prune_margin));
    if (!(o.window >= 0.0) || !std::isfinite(o.window))
        refuse("eigs", "window must be finite and >= 0, got " + num(o.window));
}

void validate_thermal_spec(const ThermalSpec& t, int n_sites) {
    if (t.temperatures.empty()) refuse("thermal", "no temperatures");
    for (double T : t.temperatures)
        if (!std::isfinite(T) || !(T > 0.0)) refuse("thermal", "temperatures must be finite and > 0, got " + num(T));
    if (t.method != ThermalSpec::Method::Exact) {
        if (t.samples < 1) refuse("thermal", "samples must be >= 1");
        if (t.method == ThermalSpec::Method::FTLM && t.krylov < 1) refuse("thermal", "krylov must be >= 1");
    }
    for (std::size_t i = 0; i < t.observables.size(); ++i)
        validate_observable(t.observables[i], n_sites, "thermal", i);
}

void validate_dynamics_spec(const DynamicsSpec& d) {
    if (d.omega.empty()) refuse("dynamics", "empty frequency grid");
    for (double w : d.omega)
        if (!std::isfinite(w)) refuse("dynamics", "the frequency grid has a non-finite entry " + num(w));
    if (!std::isfinite(d.eta) || !(d.eta > 0.0))
        refuse("dynamics", "the Lorentzian width eta must be finite and > 0, got " + num(d.eta));
    for (double T : d.temperatures)
        if (!std::isfinite(T) || !(T > 0.0)) refuse("dynamics", "temperatures must be finite and > 0, got " + num(T));
    if (d.krylov < 1) refuse("dynamics", "krylov must be >= 1");
    if (!d.temperatures.empty() && d.samples < 1) refuse("dynamics", "samples must be >= 1");
    if (!std::isfinite(d.degeneracy_tol) || !(d.degeneracy_tol >= 0.0))
        refuse("dynamics", "degeneracy_tol must be finite and >= 0, got " + num(d.degeneracy_tol));
    if (d.dense_max_dim < -1) refuse("dynamics", "dense_max_dim must be >= 0 (or -1: automatic)");
}

}  // namespace ed::sectors::detail

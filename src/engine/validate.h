// =============================================================================
// src/engine/validate.h -- the checks every ed::sectors entry point runs before any work: a
// request that cannot be answered raises ed::InvalidRequest naming the bad input (a
// non-Hermitian H, an Sz label outside 0..N, an observable on other sites or missing, a
// temperature or Lorentzian width that is not finite and positive, ...).
// =============================================================================
#pragma once

#include <ed/sectors/dynamics.h>
#include <ed/sectors/sectors.h>
#include <ed/sectors/thermal.h>

#include <cstddef>

namespace ed::sectors::detail {

/// Every set registered environment variable parses as its kind (ed::env::malformed).
void validate_environment(const char* verb);

/// H acts on 1..63 sites and is Hermitian relative to its largest coefficient (and the
/// environment is valid: every verb calls this first).
void validate_hamiltonian(const ::Operator& H, const char* verb);

/// The Spec's labels name sectors of n_sites sites: n_up in -1..N, sz_parity and the toggles
/// in their ranges, 2S in -1..N with N - 2S even.
void validate_spec(const Spec& s, int n_sites, const char* verb);

/// An observable is present and acts on the n_sites sites of H.
void validate_observable(const ::Operator* O, int n_sites, const char* verb, std::size_t index);

void validate_eigs_options(const EigsOptions& o);
void validate_thermal_spec(const ThermalSpec& t, int n_sites);
void validate_dynamics_spec(const DynamicsSpec& d);

}  // namespace ed::sectors::detail

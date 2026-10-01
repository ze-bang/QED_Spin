// canonical_thermo.h - exact canonical thermodynamics from a full eigenvalue list.
//
// A generic "thermodynamics from a full spectrum" helper with no dependence on
// the symmetry-adapted-basis solve machinery; used by the little-group solver.
// Header-only inline.
#pragma once

#include <ed/core/thermal_types.h>   // ThermodynamicData

#include <algorithm>
#include <cmath>
#include <vector>

namespace ed::symmetry {

// Exact canonical thermodynamics from a full eigenvalue list (multiplicities
// already folded in). Z(β)=Σ e^{-βE}; reference-shifted by E0 for stability.
// This is the single implementation; other callers forward here. A
// non-positive temperature (or Z <= 0) leaves that grid point at zero
// instead of dividing by zero.
inline ThermodynamicData
canonical_thermo_from_eigs(const std::vector<double>& eigs,
                           const std::vector<double>& T) {
    ThermodynamicData td;
    td.temperatures = T;
    if (eigs.empty() || T.empty()) return td;
    const std::size_t nT = T.size();
    td.energy.assign(nT, 0.0);
    td.specific_heat.assign(nT, 0.0);
    td.free_energy.assign(nT, 0.0);
    td.entropy.assign(nT, 0.0);
    const double E0 = *std::min_element(eigs.begin(), eigs.end());
    td.e_min = E0;
    for (std::size_t i = 0; i < nT; ++i) {
        const double t = T[i];
        if (!(t > 0.0)) continue;
        const double beta = 1.0 / t;
        double Z = 0.0, E = 0.0;
        for (double e : eigs) {
            const double w = std::exp(-beta * (e - E0));
            Z += w; E += w * (e - E0);
        }
        if (!(Z > 0.0)) continue;
        const double Eavg = E0 + E / Z;
        // Second pass: the central moment (raw <E^2> - <E>^2 cancels to rounding noise once
        // C T^2 drops below ulp(E^2)).
        double V = 0.0;
        for (double e : eigs) {
            const double d = e - Eavg;
            V += std::exp(-beta * (e - E0)) * d * d;
        }
        const double Cv    = beta * beta * V / Z;
        const double F     = E0 - t * std::log(Z);   // -T ln Z_full
        const double S     = (Eavg - F) / t;
        td.energy[i]        = Eavg;
        td.specific_heat[i] = Cv;
        td.free_energy[i]   = F;
        td.entropy[i]       = S;
    }
    return td;
}

}  // namespace ed::symmetry

// canonical_thermo.h - exact canonical thermodynamics from a full eigenvalue list.
//
// Extracted from the retired symmetry_adapted.{h,cpp} (Consolidation Family 6):
// this reduction is a generic "thermodynamics from a full spectrum" helper with
// no dependence on the symmetry-adapted-basis solve machinery, and is used by
// the production little-group solver. Kept header-only inline.
#pragma once

#include <ed/core/thermal_types.h>   // ThermodynamicData

#include <algorithm>
#include <cmath>
#include <vector>

namespace ed::symmetry {

// Exact canonical thermodynamics from a full eigenvalue list (multiplicities
// already folded in). Z(β)=Σ e^{-βE}; reference-shifted by E0 for stability.
inline ThermodynamicData
canonical_thermo_from_eigs(const std::vector<double>& eigs,
                           const std::vector<double>& T) {
    ThermodynamicData td;
    td.temperatures = T;
    if (eigs.empty()) return td;
    const double E0 = *std::min_element(eigs.begin(), eigs.end());
    td.e_min = E0;
    for (double t : T) {
        const double beta = 1.0 / t;
        double Z = 0.0, E = 0.0, E2 = 0.0;
        for (double e : eigs) {
            const double w = std::exp(-beta * (e - E0));
            Z += w; E += w * e; E2 += w * e * e;
        }
        const double Eavg  = E / Z;
        const double E2avg = E2 / Z;
        const double Cv    = beta * beta * (E2avg - Eavg * Eavg);
        const double F     = E0 - t * std::log(Z);   // -T ln Z_full
        const double S     = (Eavg - F) / t;
        td.energy.push_back(Eavg);
        td.specific_heat.push_back(Cv);
        td.free_energy.push_back(F);
        td.entropy.push_back(S);
    }
    return td;
}

}  // namespace ed::symmetry

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
//
// Audit 2026-07-31: this is now the SINGLE implementation -- the
// orchestrator's file-local compute_canonical_thermo_from_eigs and the
// SU(2) tower binding's su2_exact_thermo_from_eigs were byte-equivalent
// twins and forward here. Their T<=0 / Z<=0 guards were folded in
// (a non-positive temperature leaves that grid point at zero instead of
// dividing by zero).
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
        double Z = 0.0, E = 0.0, E2 = 0.0;
        for (double e : eigs) {
            const double w = std::exp(-beta * (e - E0));
            Z += w; E += w * e; E2 += w * e * e;
        }
        if (!(Z > 0.0)) continue;
        const double Eavg  = E / Z;
        const double E2avg = E2 / Z;
        const double Cv    = beta * beta * (E2avg - Eavg * Eavg);
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

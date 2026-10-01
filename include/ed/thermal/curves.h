#pragma once
// =============================================================================
// include/ed/thermal/curves.h
//
// What every thermal method returns for one block: per inverse temperature, in the
// caller's order,
//
//   lnZ   ln of the block's partition function,
//   E     the canonical energy <H>,
//   V     the central second moment <(H - <H>)^2> (formed about a reference energy,
//         so it does not cancel at low T as a raw <H^2> - <H>^2 would),
//   O     <O> per static observable.
//
// The sector combination consumes exactly these (blocks add in log space; the variance
// of the mixture is the law of total variance). exact_curves is the one exact formula.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace ed::thermal {

struct Curves {
    std::vector<double> lnZ, E, V;
    std::vector<std::vector<std::complex<double>>> O;   ///< O[o][beta index]
};

/// The exact curves of a block from its eigenvalues; with `diag_obs` (diag_obs[o][n] =
/// <n|O_o|n> for each eigenvalue n) also <O>. Boltzmann weights are taken about the lowest
/// eigenvalue, and V in a second pass about <E>, free of cancellation.
[[nodiscard]] inline Curves exact_curves(const std::vector<double>& eigenvalues,
                                         const std::vector<double>& betas,
                                         const std::vector<std::vector<std::complex<double>>>* diag_obs = nullptr) {
    Curves c;
    const std::vector<double>& ev = eigenvalues;
    const double e0 = *std::min_element(ev.begin(), ev.end());
    if (diag_obs) c.O.assign(diag_obs->size(), {});
    for (double bt : betas) {
        double z = 0.0, e = 0.0;
        std::vector<std::complex<double>> o(diag_obs ? diag_obs->size() : 0, std::complex<double>(0, 0));
        for (std::size_t n = 0; n < ev.size(); ++n) {
            const double x = ev[n], w = std::exp(-bt * (x - e0));
            z += w; e += w * (x - e0);
            for (std::size_t k = 0; k < o.size(); ++k) o[k] += w * (*diag_obs)[k][n];
        }
        c.lnZ.push_back(std::log(z) - bt * e0);
        c.E.push_back(e0 + e / z);
        double v = 0.0;                    // second pass: the central moment, free of cancellation
        for (std::size_t n = 0; n < ev.size(); ++n) {
            const double dx = ev[n] - e0 - e / z;
            v += std::exp(-bt * (ev[n] - e0)) * dx * dx;
        }
        c.V.push_back(v / z);
        for (std::size_t k = 0; k < o.size(); ++k) c.O[k].push_back(o[k] / z);
    }
    return c;
}

}  // namespace ed::thermal

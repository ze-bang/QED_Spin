#pragma once
// =============================================================================
// include/ed/core/operator_types_detail.h
//
// Private implementation helpers shared by the concrete operator types in
// ``operator_types.h`` and ``fixed_sz_operator_types.h``. These factor out
// the substantial duplication that the position-dependent operator family
// had accumulated over several refactors:
//
//   * Pyrochlore sublattice basis (z_mu, repeated in 4 constructors).
//   * Reading positions.dat (was *inconsistent* between the full and
//     fixed-Sz versions; the full version's "x y z per line" matches what
//     ``ed::input::write_positions_file`` actually writes -- the fixed-Sz
//     version's 6-column reader was a silent format mismatch that
//     produced all-zero positions when fed canonical files).
//   * Computing exp(i Q . R_i) / sqrt(N) phase factors.
//   * Computing transverse (sublattice-weighted) phase factors.
//   * Dispatching a Cartesian-basis (Sx / Sy / Sz) single-site term to
//     the canonical AoS term setter on Operator.
//
// All helpers live in ``ed::core::detail``. Pure functions with no class
// affinity; consumed by every operator type that needs them.
// =============================================================================

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/core/operator.h>  // Operator::Complex, addOneBodyTerm

namespace ed::core::detail {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Pyrochlore sublattice basis (z_mu, mu = 0..3): the local easy-axis unit
// vector at each of the four sublattices. Used by Transverse* and
// TransverseExperimental* operators to weight site contributions.
// ---------------------------------------------------------------------------
inline constexpr double k_inv_sqrt3 = 0.5773502691896258;  // 1 / sqrt(3)

inline constexpr std::array<std::array<double, 3>, 4> kPyrochloreSublatticeBasis = {{
    {-k_inv_sqrt3, -k_inv_sqrt3, -k_inv_sqrt3},
    {-k_inv_sqrt3,  k_inv_sqrt3,  k_inv_sqrt3},
    { k_inv_sqrt3, -k_inv_sqrt3,  k_inv_sqrt3},
    { k_inv_sqrt3,  k_inv_sqrt3, -k_inv_sqrt3},
}};

// ---------------------------------------------------------------------------
// read_positions_file
//
// Parses the canonical positions.dat format produced by
// ``ed::input::write_positions_file``: one line per site, "x y z" in
// scientific notation. Lines starting with '#' are treated as comments.
//
// Returns a vector of size ``expected_sites``; entries that aren't
// present in the file are left at (0,0,0).
//
// History: ``operator_types.h``'s ``BasePositionOperator::readPositionsFromFile``
// already parsed the canonical format. ``fixed_sz_operator_types.h``'s
// version expected "site_id matrix_idx sublattice x y z" (a legacy
// 6-column format), which silently produced all-zero positions when fed
// the canonical files written by HamiltonianBuilder. Unifying on the
// 3-column reader fixes that latent bug and removes a 30-line
// duplicate.
// ---------------------------------------------------------------------------
inline std::vector<std::array<double, 3>>
read_positions_file(const std::string& filename, std::uint64_t expected_sites)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(
            "ed::core::detail::read_positions_file: could not open " + filename);
    }

    std::vector<std::array<double, 3>> positions(
        expected_sites, std::array<double, 3>{0.0, 0.0, 0.0});

    std::string line;
    std::uint64_t site_id = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double x, y, z;
        if (iss >> x >> y >> z) {
            if (site_id < expected_sites) {
                positions[site_id] = {x, y, z};
            }
            ++site_id;
        }
    }
    return positions;
}

// ---------------------------------------------------------------------------
// compute_phase_factors
//
// Returns the per-site complex phase phi_i = norm * exp(i Q . R_i).
// Used by Sum* and Experimental* operators.
// ---------------------------------------------------------------------------
inline std::vector<Complex>
compute_phase_factors(const std::vector<double>& Q,
                      const std::vector<std::array<double, 3>>& positions,
                      double norm)
{
    std::vector<Complex> phases(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        double dot = 0.0;
        for (std::size_t d = 0; d < 3; ++d) dot += Q[d] * positions[i][d];
        phases[i] = norm * std::exp(Complex(0.0, dot));
    }
    return phases;
}

// ---------------------------------------------------------------------------
// compute_transverse_phase_factors
//
// Per-site phi_i = (1 / sqrt(N)) * (v . z_{sublattice(i)}) * exp(i Q . R_i)
// where sublattice(i) = i % 4 and z_mu is the pyrochlore basis above.
// Used by Transverse* and TransverseExperimental* operators.
// ---------------------------------------------------------------------------
inline std::vector<Complex>
compute_transverse_phase_factors(const std::vector<double>& Q,
                                 const std::vector<double>& v,
                                 const std::vector<std::array<double, 3>>& positions)
{
    const std::size_t N = positions.size();
    const double norm = 1.0 / std::sqrt(static_cast<double>(N));
    std::vector<Complex> phases(N);
    for (std::size_t i = 0; i < N; ++i) {
        double Q_dot_R = 0.0;
        for (std::size_t d = 0; d < 3; ++d) Q_dot_R += Q[d] * positions[i][d];

        const auto& z = kPyrochloreSublatticeBasis[i % 4];
        double v_dot_z = 0.0;
        for (std::size_t d = 0; d < 3; ++d) v_dot_z += v[d] * z[d];

        phases[i] = norm * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
    }
    return phases;
}

// ---------------------------------------------------------------------------
// add_cartesian_site_term
//
// Append a single-site Cartesian-basis term ``phase * S^alpha_site`` to
// the operator, routed through the canonical typed AoS API.
//
//   alpha = 0  (Sx)  ->  S+/2 + S-/2     two terms emitted
//   alpha = 1  (Sy)  ->  -iS+/2 + iS-/2  two terms emitted
//   alpha = 2  (Sz)  ->  Sz              one term emitted
//
// Throws std::invalid_argument on any other ``alpha``. The Sz coupling
// does NOT include the spin_l factor -- the matvec kernel multiplies by
// spin_l inside the term loop.
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_cartesian_site_term(OperatorT& op,
                                    std::uint64_t alpha,
                                    std::uint64_t site,
                                    Complex phase)
{
    constexpr std::uint8_t kOpSPlus  = 0;
    constexpr std::uint8_t kOpSMinus = 1;
    constexpr std::uint8_t kOpSz     = 2;

    switch (alpha) {
    case 0:  // Sx = (S+ + S-) / 2
        op.addOneBodyTerm(kOpSPlus,  site, phase * 0.5);
        op.addOneBodyTerm(kOpSMinus, site, phase * 0.5);
        break;
    case 1:  // Sy = (S+ - S-) / (2i) = -i(S+ - S-)/2
        op.addOneBodyTerm(kOpSPlus,  site, phase * Complex(0.0, -0.5));
        op.addOneBodyTerm(kOpSMinus, site, phase * Complex(0.0,  0.5));
        break;
    case 2:  // Sz
        op.addOneBodyTerm(kOpSz, site, phase);
        break;
    default:
        throw std::invalid_argument(
            "ed::core::detail::add_cartesian_site_term: alpha must be 0 (Sx), "
            "1 (Sy), or 2 (Sz); got " + std::to_string(alpha));
    }
}

// ---------------------------------------------------------------------------
// add_experimental_site_term
//
// Append the per-site contribution of an "experimental" operator,
//   phase_i * ( cos(theta) Sz_i  +  sin(theta) Sx_i ),
// expanded into the canonical S+/S-/Sz term AoS. Sx is rewritten as
// (S+ + S-)/2 (see ``add_cartesian_site_term``); the resulting three
// terms per site are emitted in (Sz, S+, S-) order to match the legacy
// classifier output for stable diff-based testing.
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_experimental_site_term(OperatorT& op,
                                       std::uint64_t site,
                                       Complex phase,
                                       double cos_theta,
                                       double sin_theta)
{
    constexpr std::uint8_t kOpSPlus  = 0;
    constexpr std::uint8_t kOpSMinus = 1;
    constexpr std::uint8_t kOpSz     = 2;

    op.addOneBodyTerm(kOpSz, site, phase * cos_theta);
    const Complex sin_half = phase * (0.5 * sin_theta);
    op.addOneBodyTerm(kOpSPlus,  site, sin_half);
    op.addOneBodyTerm(kOpSMinus, site, sin_half);
}

}  // namespace ed::core::detail

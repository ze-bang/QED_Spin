#pragma once
// =============================================================================
// include/ed/core/fixed_sz_operator_types.h
//
// Concrete fixed-Sz operator subclasses, parallelling the full-basis
// types in ``operator_types.h`` but inheriting from ``FixedSzOperator``.
//
// All position-dependent helpers (positions.dat parsing, phase factors,
// pyrochlore sublattice basis, Cartesian Sx/Sy expansion) live in the
// shared ``operator_types_detail.h`` -- the fixed-Sz variants and their
// full-basis counterparts now share ONE implementation, eliminating
// the previous ~250-line duplication and a latent format-mismatch bug
// in the old fixed-Sz position-file parser (it read a 6-column legacy
// format while ``ed::input::write_positions_file`` writes the canonical
// 3-column format).
//
// Term mutation is routed through the canonical typed AoS setters
// (``addOneBodyTerm`` / ``addTwoBodyTerm``) -- no ``addTransform``
// (std::function) closures anywhere in this header. This makes the
// fixed-Sz position operators trivially parallelisable through the
// matvec backend just like their full-basis counterparts.
//
// Class hierarchy (all derive from FixedSzOperator):
//   FixedSzSingleSiteOperator
//   FixedSzDoubleSiteOperator
//   FixedSzBasePositionOperator
//     FixedSzSumOperator
//     FixedSzSumOperatorXYZ
//     FixedSzSublatticeOperator
//     FixedSzTransverseOperator
//     FixedSzTransverseOperatorXYZ
//     FixedSzExperimentalOperator
//     FixedSzTransverseExperimentalOperator
//
// Note: Sx and Sy terms do not conserve total Sz. Applying them within
// a fixed-Sz sector yields zero (the matvec kernel drops out-of-sector
// scatter via the basis-policy lookup). Constructors that accept
// Cartesian ``op`` (0=Sx, 1=Sy, 2=Sz) accept all three values without
// throwing; only the Sz contribution survives in the sector, mirroring
// the full-basis behaviour.
// =============================================================================

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator_types_detail.h>

// =============================================================================
// FixedSzSingleSiteOperator
// =============================================================================

/**
 * Single-site fixed-Sz operator. ``op`` selects:
 *   0=S+, 1=S-, 2=Sz, 3=Sx, 4=Sy
 *
 * S+, S-, Sx, Sy all flip a bit -> they scatter out of the sector and
 * vanish under the matvec backend's ``index_of()`` lookup. Sz survives.
 * This matches the legacy semantics, just routed through the canonical
 * AoS API instead of an addTransform closure.
 */
class FixedSzSingleSiteOperator : public FixedSzOperator {
public:
    FixedSzSingleSiteOperator(uint64_t num_site, float spin_l, int64_t n_up,
                              uint64_t op, uint64_t site_j)
        : FixedSzOperator(num_site, spin_l, n_up)
    {
        if (op > 4) {
            throw std::invalid_argument(
                "FixedSzSingleSiteOperator: op must be 0 (S+), 1 (S-), 2 (Sz), "
                "3 (Sx), or 4 (Sy)");
        }
        if (site_j >= num_site) {
            throw std::invalid_argument(
                "FixedSzSingleSiteOperator: site index >= num_site");
        }
        if (op <= 2) {
            addOneBodyTerm(static_cast<uint8_t>(op), site_j,
                           Complex(1.0, 0.0));
        } else {
            ed::core::detail::add_cartesian_site_term(
                *this, op - 3, site_j, Complex(1.0, 0.0));
        }
    }
};

// =============================================================================
// FixedSzDoubleSiteOperator
// =============================================================================

/**
 * Two-site coupling op_i(site_i) * op_j(site_j) with unit weight, in a
 * fixed-Sz sector. ``op_i``, ``op_j`` in {0=S+, 1=S-, 2=Sz}.
 */
class FixedSzDoubleSiteOperator : public FixedSzOperator {
public:
    FixedSzDoubleSiteOperator(uint64_t num_site, float spin_l, int64_t n_up,
                              uint64_t op_i, uint64_t site_i,
                              uint64_t op_j, uint64_t site_j)
        : FixedSzOperator(num_site, spin_l, n_up)
    {
        if (op_i > 2 || op_j > 2) {
            throw std::invalid_argument(
                "FixedSzDoubleSiteOperator: op must be 0 (S+), 1 (S-), or 2 (Sz)");
        }
        if (site_i >= num_site || site_j >= num_site) {
            throw std::invalid_argument(
                "FixedSzDoubleSiteOperator: site index >= num_site");
        }
        addTwoBodyTerm(static_cast<uint8_t>(op_i), site_i,
                       static_cast<uint8_t>(op_j), site_j,
                       Complex(1.0, 0.0));
    }
};

// =============================================================================
// FixedSzBasePositionOperator
// =============================================================================

/**
 * Tag base for position-dependent fixed-Sz operators. Owns no extra
 * state; geometry helpers live in ``ed::core::detail``.
 */
class FixedSzBasePositionOperator : public FixedSzOperator {
public:
    FixedSzBasePositionOperator(uint64_t num_site, float spin_l, int64_t n_up)
        : FixedSzOperator(num_site, spin_l, n_up) {}
};

// =============================================================================
// FixedSzSumOperator / FixedSzSumOperatorXYZ
// =============================================================================

class FixedSzSumOperator : public FixedSzBasePositionOperator {
public:
    FixedSzSumOperator(uint64_t num_site, float spin_l, int64_t n_up,
                       uint64_t op,
                       const std::vector<double>& Q_vector,
                       const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up)
    {
        const auto positions = ed::core::detail::read_positions_file(
            positions_file, num_site);
        const auto phases = ed::core::detail::compute_phase_factors(
            Q_vector, positions, 1.0 / std::sqrt(static_cast<double>(num_site)));

        for (uint64_t site = 0; site < num_site; ++site) {
            addOneBodyTerm(static_cast<uint8_t>(op), site, phases[site]);
        }
    }
};

class FixedSzSumOperatorXYZ : public FixedSzBasePositionOperator {
public:
    FixedSzSumOperatorXYZ(uint64_t num_site, float spin_l, int64_t n_up,
                          uint64_t op,
                          const std::vector<double>& Q_vector,
                          const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up)
    {
        const auto positions = ed::core::detail::read_positions_file(
            positions_file, num_site);
        const auto phases = ed::core::detail::compute_phase_factors(
            Q_vector, positions, 1.0 / std::sqrt(static_cast<double>(num_site)));

        for (uint64_t site = 0; site < num_site; ++site) {
            ed::core::detail::add_cartesian_site_term(
                *this, op, site, phases[site]);
        }
    }
};

// =============================================================================
// FixedSzSublatticeOperator
// =============================================================================

class FixedSzSublatticeOperator : public FixedSzBasePositionOperator {
public:
    FixedSzSublatticeOperator(uint64_t sublattice_idx, uint64_t unit_cell_size,
                              uint64_t num_site, float spin_l, int64_t n_up,
                              uint64_t op,
                              const std::vector<double>& Q_vector,
                              const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up)
    {
        const auto positions = ed::core::detail::read_positions_file(
            positions_file, num_site);
        const auto phases = ed::core::detail::compute_phase_factors(
            Q_vector, positions, 1.0 / std::sqrt(static_cast<double>(num_site)));

        for (uint64_t site = sublattice_idx; site < num_site;
             site += unit_cell_size)
        {
            addOneBodyTerm(static_cast<uint8_t>(op), site, phases[site]);
        }
    }
};

// =============================================================================
// FixedSzTransverseOperator / FixedSzTransverseOperatorXYZ
// =============================================================================

class FixedSzTransverseOperator : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseOperator(uint64_t num_site, float spin_l, int64_t n_up,
                              uint64_t op,
                              const std::vector<double>& Q_vector,
                              const std::vector<double>& v,
                              const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up)
    {
        const auto positions = ed::core::detail::read_positions_file(
            positions_file, num_site);
        const auto phases = ed::core::detail::compute_transverse_phase_factors(
            Q_vector, v, positions);

        for (uint64_t site = 0; site < num_site; ++site) {
            addOneBodyTerm(static_cast<uint8_t>(op), site, phases[site]);
        }
    }
};

class FixedSzTransverseOperatorXYZ : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseOperatorXYZ(uint64_t num_site, float spin_l, int64_t n_up,
                                 uint64_t op,
                                 const std::vector<double>& Q_vector,
                                 const std::vector<double>& v,
                                 const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up)
    {
        const auto positions = ed::core::detail::read_positions_file(
            positions_file, num_site);
        const auto phases = ed::core::detail::compute_transverse_phase_factors(
            Q_vector, v, positions);

        for (uint64_t site = 0; site < num_site; ++site) {
            ed::core::detail::add_cartesian_site_term(
                *this, op, site, phases[site]);
        }
    }
};

// =============================================================================
// FixedSzExperimentalOperator / FixedSzTransverseExperimentalOperator
// =============================================================================

class FixedSzExperimentalOperator : public FixedSzBasePositionOperator {
public:
    FixedSzExperimentalOperator(uint64_t num_site, float spin_l, int64_t n_up,
                                double theta,
                                const std::vector<double>& Q_vector,
                                const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up)
    {
        const auto positions = ed::core::detail::read_positions_file(
            positions_file, num_site);
        const auto phases = ed::core::detail::compute_phase_factors(
            Q_vector, positions, 1.0 / std::sqrt(static_cast<double>(num_site)));

        const double cos_theta = std::cos(theta);
        const double sin_theta = std::sin(theta);
        for (uint64_t site = 0; site < num_site; ++site) {
            ed::core::detail::add_experimental_site_term(
                *this, site, phases[site], cos_theta, sin_theta);
        }
    }
};

class FixedSzTransverseExperimentalOperator : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseExperimentalOperator(uint64_t num_site, float spin_l,
                                          int64_t n_up, double theta,
                                          const std::vector<double>& Q_vector,
                                          const std::vector<double>& v,
                                          const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up)
    {
        const auto positions = ed::core::detail::read_positions_file(
            positions_file, num_site);
        const auto phases = ed::core::detail::compute_transverse_phase_factors(
            Q_vector, v, positions);

        const double cos_theta = std::cos(theta);
        const double sin_theta = std::sin(theta);
        for (uint64_t site = 0; site < num_site; ++site) {
            ed::core::detail::add_experimental_site_term(
                *this, site, phases[site], cos_theta, sin_theta);
        }
    }
};

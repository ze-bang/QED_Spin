#pragma once
// =============================================================================
// include/ed/core/operator_types.h
//
// Concrete full-basis operator subclasses, all routed through the
// canonical typed-AoS term API (``addOneBodyTerm`` / ``addTwoBodyTerm``).
// No legacy ``addTransform`` (std::function) closures, no per-class
// duplication of position-file parsing or sublattice geometry -- the
// shared infrastructure lives in ``operator_types_detail.h``.
//
// Class hierarchy
// ---------------
//   Operator
//     SingleSiteOperator         : single-site (S+, S-, Sz, Sx, Sy)
//     DoubleSiteOperator         : single two-body coupling
//     BasePositionOperator       : geometry helper base (positions + phases)
//       SumOperator              : S^alpha = sum_i S^alpha_i e^{iQ.R_i} / sqrt(N)
//       SumOperatorXYZ           : same with Cartesian alpha (Sx/Sy/Sz)
//       SublatticeOperator       : restricted to one sublattice
//       TransverseOperator       : v.z_sublattice weighted single-op
//       TransverseOperatorXYZ    : same with Cartesian alpha
//       ExperimentalOperator     : cos(theta) Sz + sin(theta) Sx per site
//       TransverseExperimentalOperator : same with transverse weights
//
// Op-type convention (CRITICAL -- two conventions co-exist by design):
//
//   * SingleSiteOperator         : op in {0=S+, 1=S-, 2=Sz, 3=Sx, 4=Sy}
//   * SumOperator                : op in {0=S+, 1=S-, 2=Sz}     (basis-aligned)
//   * SumOperatorXYZ / TransverseOperatorXYZ : op in {0=Sx, 1=Sy, 2=Sz}
//                                                                 (Cartesian)
//
// The Cartesian convention is the standard one for spin-correlation
// observables; the basis-aligned convention matches the file format the
// codebase reads/writes for Hamiltonian terms.
// =============================================================================

#include <ed/core/operator.h>
#include <ed/core/operator_types_detail.h>

// =============================================================================
// SingleSiteOperator
// =============================================================================

/**
 * Single-site operator. ``op`` selects from {S+, S-, Sz, Sx, Sy}.
 *
 * History: previously threw on op=3,4 (Sx,Sy) although TPQ.cpp's
 * ``createSxOperators`` / ``createSyOperators`` construct it that way.
 * The Sx/Sy branches now expand into the canonical S+, S- terms via
 * ``add_cartesian_site_term`` -- the latent throw is closed.
 */
class SingleSiteOperator : public Operator {
public:
    SingleSiteOperator(uint64_t num_site, float spin_l,
                       uint64_t op, uint64_t site_j)
        : Operator(num_site, spin_l)
    {
        if (op > 4) {
            throw std::invalid_argument(
                "SingleSiteOperator: op must be 0 (S+), 1 (S-), 2 (Sz), "
                "3 (Sx), or 4 (Sy)");
        }
        if (site_j >= num_site) {
            throw std::invalid_argument(
                "SingleSiteOperator: site index >= num_site");
        }
        if (op <= 2) {
            addOneBodyTerm(static_cast<uint8_t>(op), site_j,
                           Complex(1.0, 0.0));
        } else {
            // op = 3 -> Sx (Cartesian alpha = 0); op = 4 -> Sy (Cartesian alpha = 1).
            ed::core::detail::add_cartesian_site_term(
                *this, op - 3, site_j, Complex(1.0, 0.0));
        }
    }
};

// =============================================================================
// DoubleSiteOperator
// =============================================================================

/**
 * Two-site coupling op_i(site_i) * op_j(site_j) with unit weight.
 * ``op_i``, ``op_j`` in {0=S+, 1=S-, 2=Sz}.
 */
class DoubleSiteOperator : public Operator {
public:
    DoubleSiteOperator(uint64_t num_site, float spin_l,
                       uint64_t op_i, uint64_t site_i,
                       uint64_t op_j, uint64_t site_j)
        : Operator(num_site, spin_l)
    {
        if (op_i > 2 || op_j > 2) {
            throw std::invalid_argument(
                "DoubleSiteOperator: op must be 0 (S+), 1 (S-), or 2 (Sz)");
        }
        if (site_i >= num_site || site_j >= num_site) {
            throw std::invalid_argument(
                "DoubleSiteOperator: site index >= num_site");
        }
        addTwoBodyTerm(static_cast<uint8_t>(op_i), site_i,
                       static_cast<uint8_t>(op_j), site_j,
                       Complex(1.0, 0.0));
    }
};

// =============================================================================
// BasePositionOperator
// =============================================================================

/**
 * Tag base for position-dependent operators. Owns no extra state; the
 * geometry helpers live in ``ed::core::detail``. Kept as a separate type
 * so callers can constrain templates to "any position-dependent
 * operator" without enumerating every subclass.
 */
class BasePositionOperator : public Operator {
public:
    BasePositionOperator(uint64_t num_site, float spin_l)
        : Operator(num_site, spin_l) {}
};

// =============================================================================
// SumOperator
// =============================================================================

/**
 * S^alpha_Q = (1 / sqrt(N)) sum_i S^alpha_i e^{iQ.R_i}, basis-aligned.
 * ``op`` in {0=S+, 1=S-, 2=Sz}.
 */
class SumOperator : public BasePositionOperator {
public:
    SumOperator(uint64_t num_site, float spin_l, uint64_t op,
                const std::vector<double>& Q_vector,
                const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l)
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

// =============================================================================
// SumOperatorXYZ
// =============================================================================

/**
 * S^alpha_Q with Cartesian alpha. ``op`` in {0=Sx, 1=Sy, 2=Sz}.
 */
class SumOperatorXYZ : public BasePositionOperator {
public:
    SumOperatorXYZ(uint64_t num_site, float spin_l, uint64_t op,
                   const std::vector<double>& Q_vector,
                   const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l)
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
// SublatticeOperator
// =============================================================================

/**
 * SumOperator restricted to sites ``sublattice_idx + k * unit_cell_size``.
 */
class SublatticeOperator : public BasePositionOperator {
public:
    SublatticeOperator(uint64_t sublattice_idx, uint64_t unit_cell_size,
                       uint64_t num_site, float spin_l, uint64_t op,
                       const std::vector<double>& Q_vector,
                       const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l)
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
// TransverseOperator
// =============================================================================

/**
 * Transverse Sum, basis-aligned. Per-site weight is
 * ``(1/sqrt(N)) (v . z_{sublattice(i)}) e^{iQ.R_i}`` with
 * z_mu the pyrochlore basis (``ed::core::detail::kPyrochloreSublatticeBasis``).
 */
class TransverseOperator : public BasePositionOperator {
public:
    TransverseOperator(uint64_t num_site, float spin_l, uint64_t op,
                       const std::vector<double>& Q_vector,
                       const std::vector<double>& v,
                       const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l)
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

// =============================================================================
// TransverseOperatorXYZ
// =============================================================================

/**
 * Transverse Sum with Cartesian ``op`` in {0=Sx, 1=Sy, 2=Sz}.
 */
class TransverseOperatorXYZ : public BasePositionOperator {
public:
    TransverseOperatorXYZ(uint64_t num_site, float spin_l, uint64_t op,
                          const std::vector<double>& Q_vector,
                          const std::vector<double>& v,
                          const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l)
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
// ExperimentalOperator
// =============================================================================

/**
 * Sum_i phase_i * (cos(theta) Sz_i + sin(theta) Sx_i).
 *
 * Note: ``cos(theta) Sz`` carries no extra spin_l factor (the matvec
 * kernel applies spin_l inside the Sz term loop); ``sin(theta) Sx``
 * expands to ``sin(theta)/2 * (S+ + S-)`` via the Cartesian helper.
 */
class ExperimentalOperator : public BasePositionOperator {
public:
    ExperimentalOperator(uint64_t num_site, float spin_l, double theta,
                         const std::vector<double>& Q_vector,
                         const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l)
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

// =============================================================================
// TransverseExperimentalOperator
// =============================================================================

/**
 * Like ``ExperimentalOperator`` but with the transverse
 * sublattice-weighted phase factor.
 */
class TransverseExperimentalOperator : public BasePositionOperator {
public:
    TransverseExperimentalOperator(uint64_t num_site, float spin_l, double theta,
                                   const std::vector<double>& Q_vector,
                                   const std::vector<double>& v,
                                   const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l)
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

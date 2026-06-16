#pragma once
// =============================================================================
// include/ed/core/operator_builders.h
//
// Free term-builder functions that replace the concrete operator subclass zoo
// (the former ``operator_types.h`` / ``fixed_sz_operator_types.h``). Each builder
// takes any Operator-like object (``Operator`` for the full Hilbert space or
// ``FixedSzOperator`` for a fixed-Sz sector) by reference and appends terms via
// the canonical typed AoS API (``addOneBodyTerm`` / ``addTwoBodyTerm``),
// reusing the shared geometry / phase math in ``operator_types_detail.h``.
//
// Op-type conventions (unchanged from the legacy zoo):
//   * basis-aligned single-op : op in {0=S+, 1=S-, 2=Sz}     (matches file fmt)
//   * Cartesian single-op     : op in {0=Sx, 1=Sy, 2=Sz}     (use_xyz = true)
//   * add_single_site         : op in {0=S+, 1=S-, 2=Sz, 3=Sx, 4=Sy}
//
// Sx / Sy expand into canonical S+/S- terms via the Cartesian helper, so they
// are perfectly valid in a fixed-Sz sector (the spin-flip terms scatter out of
// the sector and vanish under the basis-policy ``index_of`` lookup).
// =============================================================================

#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator_types_detail.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::ops {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// add_single_site : op in {0=S+, 1=S-, 2=Sz, 3=Sx, 4=Sy}
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_single_site(OperatorT& op, std::uint64_t which, std::uint64_t site) {
    const std::uint64_t num_site = op.getNumBits();
    if (which > 4) {
        throw std::invalid_argument(
            "ed::ops::add_single_site: op must be 0 (S+), 1 (S-), 2 (Sz), "
            "3 (Sx), or 4 (Sy)");
    }
    if (site >= num_site) {
        throw std::invalid_argument(
            "ed::ops::add_single_site: site index >= num_site");
    }
    if (which <= 2) {
        op.addOneBodyTerm(static_cast<std::uint8_t>(which), site, Complex(1.0, 0.0));
    } else {
        // which = 3 -> Sx (Cartesian alpha = 0); which = 4 -> Sy (alpha = 1).
        ed::core::detail::add_cartesian_site_term(op, which - 3, site,
                                                  Complex(1.0, 0.0));
    }
}

// ---------------------------------------------------------------------------
// add_double_site : op_i / op_j in {0=S+, 1=S-, 2=Sz}, unit weight
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_double_site(OperatorT& op,
                            std::uint64_t op_i, std::uint64_t site_i,
                            std::uint64_t op_j, std::uint64_t site_j) {
    const std::uint64_t num_site = op.getNumBits();
    if (op_i > 2 || op_j > 2) {
        throw std::invalid_argument(
            "ed::ops::add_double_site: op must be 0 (S+), 1 (S-), or 2 (Sz)");
    }
    if (site_i >= num_site || site_j >= num_site) {
        throw std::invalid_argument(
            "ed::ops::add_double_site: site index >= num_site");
    }
    op.addTwoBodyTerm(static_cast<std::uint8_t>(op_i), site_i,
                      static_cast<std::uint8_t>(op_j), site_j,
                      Complex(1.0, 0.0));
}

// ---------------------------------------------------------------------------
// add_sum : S^a_Q = (1/sqrt(N)) sum_i S^a_i e^{iQ.R_i}
//   use_xyz = false -> basis-aligned op in {0=S+, 1=S-, 2=Sz}
//   use_xyz = true  -> Cartesian   op in {0=Sx, 1=Sy, 2=Sz}
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_sum(OperatorT& op, std::uint64_t which,
                    const std::vector<double>& Q,
                    const std::string& positions_file,
                    bool use_xyz) {
    const std::uint64_t N = op.getNumBits();
    const auto positions = ed::core::detail::read_positions_file(positions_file, N);
    const auto phases = ed::core::detail::compute_phase_factors(
        Q, positions, 1.0 / std::sqrt(static_cast<double>(N)));
    for (std::uint64_t site = 0; site < N; ++site) {
        if (use_xyz) {
            ed::core::detail::add_cartesian_site_term(op, which, site, phases[site]);
        } else {
            op.addOneBodyTerm(static_cast<std::uint8_t>(which), site, phases[site]);
        }
    }
}

// ---------------------------------------------------------------------------
// add_sublattice : add_sum restricted to sublattice_idx + k * unit_cell_size
//   (basis-aligned op only, matching the legacy SublatticeOperator)
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_sublattice(OperatorT& op,
                           std::uint64_t sublattice_idx,
                           std::uint64_t unit_cell_size,
                           std::uint64_t which,
                           const std::vector<double>& Q,
                           const std::string& positions_file) {
    const std::uint64_t N = op.getNumBits();
    const auto positions = ed::core::detail::read_positions_file(positions_file, N);
    const auto phases = ed::core::detail::compute_phase_factors(
        Q, positions, 1.0 / std::sqrt(static_cast<double>(N)));
    for (std::uint64_t site = sublattice_idx; site < N; site += unit_cell_size) {
        op.addOneBodyTerm(static_cast<std::uint8_t>(which), site, phases[site]);
    }
}

// ---------------------------------------------------------------------------
// add_transverse : transverse (sublattice-weighted) sum
//   use_xyz selects Cartesian vs basis-aligned op (same convention as add_sum)
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_transverse(OperatorT& op, std::uint64_t which,
                           const std::vector<double>& Q,
                           const std::vector<double>& v,
                           const std::string& positions_file,
                           bool use_xyz) {
    const std::uint64_t N = op.getNumBits();
    const auto positions = ed::core::detail::read_positions_file(positions_file, N);
    const auto phases = ed::core::detail::compute_transverse_phase_factors(
        Q, v, positions);
    for (std::uint64_t site = 0; site < N; ++site) {
        if (use_xyz) {
            ed::core::detail::add_cartesian_site_term(op, which, site, phases[site]);
        } else {
            op.addOneBodyTerm(static_cast<std::uint8_t>(which), site, phases[site]);
        }
    }
}

// ---------------------------------------------------------------------------
// add_experimental : sum_i phase_i * (cos(theta) Sz_i + sin(theta) Sx_i)
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_experimental(OperatorT& op, double theta,
                             const std::vector<double>& Q,
                             const std::string& positions_file) {
    const std::uint64_t N = op.getNumBits();
    const auto positions = ed::core::detail::read_positions_file(positions_file, N);
    const auto phases = ed::core::detail::compute_phase_factors(
        Q, positions, 1.0 / std::sqrt(static_cast<double>(N)));
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    for (std::uint64_t site = 0; site < N; ++site) {
        ed::core::detail::add_experimental_site_term(
            op, site, phases[site], cos_theta, sin_theta);
    }
}

// ---------------------------------------------------------------------------
// add_transverse_experimental : add_experimental with transverse phases
// ---------------------------------------------------------------------------
template <class OperatorT>
inline void add_transverse_experimental(OperatorT& op, double theta,
                                        const std::vector<double>& Q,
                                        const std::vector<double>& v,
                                        const std::string& positions_file) {
    const std::uint64_t N = op.getNumBits();
    const auto positions = ed::core::detail::read_positions_file(positions_file, N);
    const auto phases = ed::core::detail::compute_transverse_phase_factors(
        Q, v, positions);
    const double cos_theta = std::cos(theta);
    const double sin_theta = std::sin(theta);
    for (std::uint64_t site = 0; site < N; ++site) {
        ed::core::detail::add_experimental_site_term(
            op, site, phases[site], cos_theta, sin_theta);
    }
}

// ---------------------------------------------------------------------------
// Convenience value factory for the common single-site full-basis observable.
// ---------------------------------------------------------------------------
inline Operator make_single_site(std::uint64_t num_site, float spin_l,
                                 std::uint64_t which, std::uint64_t site) {
    Operator op(num_site, spin_l);
    add_single_site(op, which, site);
    return op;
}

}  // namespace ed::ops

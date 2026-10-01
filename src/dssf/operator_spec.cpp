// =============================================================================
// src/dssf/operator_spec.cpp
//
// Implementation of `ed::dssf::build_observables`. See the header for the
// design rationale.
//
// Argument validation throws std::invalid_argument; private helpers carry
// the cross-product / normalization arithmetic.
// =============================================================================

#include <ed/dssf/operator_spec.h>

#include <ed/core/operator_builders.h>

#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ed::dssf {
namespace {

// Build a full-Hilbert observable value-type Operator via ``build``.
template <typename BuildFn>
Operator make_full(const OperatorSpec& spec, BuildFn&& build) {
    Operator op(spec.num_sites, 0.5f);
    build(op);
    return op;
}

constexpr double kZeroTol = 1e-10;

std::array<double, 3> cross_product(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

std::array<double, 3> normalize(const std::array<double, 3>& v) {
    const double norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (norm < kZeroTol) {
        return {0.0, 0.0, 0.0};
    }
    return {v[0] / norm, v[1] / norm, v[2] / norm};
}

const char* component_name(int op, bool use_xyz_basis) {
    if (use_xyz_basis) {
        switch (op) {
            case 0: return "Sx";
            case 1: return "Sy";
            case 2: return "Sz";
            default: return "Unknown";
        }
    }
    switch (op) {
        case 2: return "Sz";
        case 0: return "Sp";
        case 1: return "Sm";
        default: return "Unknown";
    }
}

// "<prefix>_q_Qx<Q0>_Qy<Q1>_Qz<Q2>": the stable stem of every observable name.
std::string q_name(const std::string& prefix, const std::vector<double>& Q) {
    std::stringstream ss;
    ss << prefix << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2];
    return ss.str();
}

void emit(Observables& out, Operator op, std::string name) {
    out.operators.push_back(std::move(op));
    out.names.push_back(std::move(name));
}

} // namespace

std::pair<std::array<double, 3>, std::array<double, 3>>
compute_transverse_bases(const std::vector<double>& Q,
                         const std::vector<double>& polarization) {
    if (Q.size() != 3) {
        throw std::invalid_argument(
            "ed::dssf::compute_transverse_bases: Q must be a 3-vector");
    }
    if (polarization.size() != 3) {
        throw std::invalid_argument(
            "ed::dssf::compute_transverse_bases: polarization must be a 3-vector");
    }
    const std::array<double, 3> pol_array = {
        polarization[0], polarization[1], polarization[2]};
    const auto cross = cross_product(Q, polarization);
    const double cross_norm = std::sqrt(cross[0] * cross[0] +
                                        cross[1] * cross[1] +
                                        cross[2] * cross[2]);
    std::array<double, 3> transverse_basis_2;
    if (cross_norm < kZeroTol) {
        if (std::abs(pol_array[0]) > 0.5) {
            transverse_basis_2 = normalize(cross_product({0.0, 1.0, 0.0}, polarization));
        } else {
            transverse_basis_2 = normalize(cross_product({1.0, 0.0, 0.0}, polarization));
        }
    } else {
        transverse_basis_2 = normalize(cross);
    }
    return {pol_array, transverse_basis_2};
}

Observables build_observables(const OperatorSpec& spec) {
    const std::string& type = spec.operator_type;
    const bool experimental = type == "experimental" || type == "transverse_experimental";
    if (type != "sum" && type != "transverse" && type != "sublattice" && !experimental) {
        throw std::invalid_argument(
            "ed::dssf::build_observables: unknown operator_type '" + type + "'");
    }
    if (spec.components.empty() && !experimental) {
        throw std::invalid_argument(
            "ed::dssf::build_observables: components is empty");
    }
    if (spec.momentum_points.empty()) {
        throw std::invalid_argument(
            "ed::dssf::build_observables: momentum_points is empty");
    }
    if (spec.polarization.size() != 3) {
        throw std::invalid_argument(
            "ed::dssf::build_observables: polarization must be a 3-vector");
    }
    if (spec.num_sites == 0) {
        throw std::invalid_argument(
            "ed::dssf::build_observables: num_sites must be > 0");
    }

    const bool use_xyz_basis = (spec.basis == "xyz");
    const auto& pf = spec.positions_file;
    Observables out;

    for (const auto& Q : spec.momentum_points) {
        std::vector<double> e1, e2;
        if (type == "transverse" || type == "transverse_experimental") {
            const auto [b1, b2] = compute_transverse_bases(Q, spec.polarization);
            e1.assign(b1.begin(), b1.end());
            e2.assign(b2.begin(), b2.end());
        }

        if (type == "experimental") {
            std::stringstream theta;
            theta << "_theta" << spec.theta;
            emit(out, make_full(spec, [&](auto& op) { ed::ops::add_experimental(op, spec.theta, Q, pf); }),
                 q_name("Experimental", Q) + theta.str());
            continue;
        }
        if (type == "transverse_experimental") {
            std::stringstream theta;
            theta << "_theta" << spec.theta;
            const std::string stem = q_name("TransverseExperimental", Q) + theta.str();
            for (const auto* e : {&e1, &e2}) {
                emit(out, make_full(spec, [&](auto& op) {
                         ed::ops::add_transverse_experimental(op, spec.theta, Q, *e, pf);
                     }),
                     stem + (e == &e1 ? "_NSF" : "_SF"));
            }
            continue;
        }

        for (const int c : spec.components) {
            const auto which = static_cast<std::uint64_t>(c);
            const std::string stem = q_name(component_name(c, use_xyz_basis), Q);
            if (type == "sum") {
                emit(out, make_full(spec, [&](auto& op) {
                         ed::ops::add_sum(op, which, Q, pf, use_xyz_basis);
                     }),
                     stem);
            } else if (type == "transverse") {
                for (const auto* e : {&e1, &e2}) {
                    emit(out, make_full(spec, [&](auto& op) {
                             ed::ops::add_transverse(op, which, Q, *e, pf, use_xyz_basis);
                         }),
                         stem + (e == &e1 ? "_NSF" : "_SF"));
                }
            } else {   // sublattice
                auto one = [&](std::uint64_t sub) {
                    emit(out, make_full(spec, [&](auto& op) {
                             ed::ops::add_sublattice(op, sub, spec.unit_cell_size, which, Q, pf);
                         }),
                         stem + "_sub" + std::to_string(sub));
                };
                if (spec.sublattice) {
                    one(*spec.sublattice);
                } else {
                    for (std::uint64_t sub = 0; sub < spec.unit_cell_size; ++sub) one(sub);
                }
            }
        }
    }
    return out;
}

} // namespace ed::dssf

// =============================================================================
// src/dssf/operator_spec.cpp
//
// Implementation of `ed::dssf::build_observable_pairs`. See the header
// for the full design rationale (P1.10 / DSSF PR-A).
//
// The body is a direct lift-and-shift of the legacy
// `construct_operators_from_config` from src/apps/ed_main.cpp, with the
// following minimal changes:
//   * lives in a namespace
//   * returns a struct rather than three out-parameters
//   * argument validation throws std::invalid_argument instead of writing
//     stderr warnings (the CLI layer is now responsible for translating
//     them into user-friendly messages)
//   * private helpers for the cross-product / normalization arithmetic so
//     ed_main.cpp doesn't need to keep its own copies
// =============================================================================

#include <ed/dssf/operator_spec.h>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ed::dssf {
namespace {

// Audit #2 helper: emit a sliced Operator (legacy obs_1/obs_2 entry) AND a
// type-preserving shared_ptr<FixedSzOperator> entry (obs_1_fs/obs_2_fs).
// Use a templated lambda factory pattern via this small helper so callers
// keep the constructor-arg list local. The shared_ptr keeps the fully
// derived deleter (make_shared<Derived>) so non-virtual ~Operator is OK.
template <typename FsT, typename... Args>
void push_fs_pair_a(ObservablePairs& out, Args&&... args) {
    auto p = std::make_shared<FsT>(std::forward<Args>(args)...);
    out.obs_1.push_back(Operator(*p));
    out.obs_1_fs.push_back(std::static_pointer_cast<FixedSzOperator>(p));
}
template <typename FsT, typename... Args>
void push_fs_pair_b(ObservablePairs& out, Args&&... args) {
    auto p = std::make_shared<FsT>(std::forward<Args>(args)...);
    out.obs_2.push_back(Operator(*p));
    out.obs_2_fs.push_back(std::static_pointer_cast<FixedSzOperator>(p));
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

const char* spin_combination_name(int op, bool use_xyz_basis) {
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

void append_sum_pair(const OperatorSpec& spec,
                     const std::vector<double>& Q,
                     int op_type_1,
                     int op_type_2,
                     bool use_xyz_basis,
                     ObservablePairs& out) {
    const auto& pf = spec.positions_file;
    const auto N = spec.num_sites;
    const auto S = spec.spin_length;
    const bool emit_b = !spec.single_obs_only;

    if (spec.use_fixed_sz) {
        if (use_xyz_basis) {
            push_fs_pair_a<FixedSzSumOperatorXYZ>(out, N, S, spec.n_up, op_type_1, Q, pf);
            if (emit_b) {
                push_fs_pair_b<FixedSzSumOperatorXYZ>(out, N, S, spec.n_up, op_type_2, Q, pf);
            }
        } else {
            push_fs_pair_a<FixedSzSumOperator>(out, N, S, spec.n_up, op_type_1, Q, pf);
            if (emit_b) {
                push_fs_pair_b<FixedSzSumOperator>(out, N, S, spec.n_up, op_type_2, Q, pf);
            }
        }
    } else {
        if (use_xyz_basis) {
            out.obs_1.push_back(Operator(SumOperatorXYZ(N, S, op_type_1, Q, pf)));
            if (emit_b) {
                out.obs_2.push_back(Operator(SumOperatorXYZ(N, S, op_type_2, Q, pf)));
            }
        } else {
            out.obs_1.push_back(Operator(SumOperator(N, S, op_type_1, Q, pf)));
            if (emit_b) {
                out.obs_2.push_back(Operator(SumOperator(N, S, op_type_2, Q, pf)));
            }
        }
    }
}

void append_transverse_pair(const OperatorSpec& spec,
                            const std::vector<double>& Q,
                            int op_type_1,
                            int op_type_2,
                            bool use_xyz_basis,
                            const std::vector<double>& e1_vec,
                            const std::vector<double>& e2_vec,
                            ObservablePairs& out) {
    const auto& pf = spec.positions_file;
    const auto N = spec.num_sites;
    const auto S = spec.spin_length;

    if (spec.use_fixed_sz) {
        if (use_xyz_basis) {
            push_fs_pair_a<FixedSzTransverseOperatorXYZ>(out, N, S, spec.n_up, op_type_1, Q, e1_vec, pf);
            push_fs_pair_b<FixedSzTransverseOperatorXYZ>(out, N, S, spec.n_up, op_type_2, Q, e1_vec, pf);
            push_fs_pair_a<FixedSzTransverseOperatorXYZ>(out, N, S, spec.n_up, op_type_1, Q, e2_vec, pf);
            push_fs_pair_b<FixedSzTransverseOperatorXYZ>(out, N, S, spec.n_up, op_type_2, Q, e2_vec, pf);
        } else {
            push_fs_pair_a<FixedSzTransverseOperator>(out, N, S, spec.n_up, op_type_1, Q, e1_vec, pf);
            push_fs_pair_b<FixedSzTransverseOperator>(out, N, S, spec.n_up, op_type_2, Q, e1_vec, pf);
            push_fs_pair_a<FixedSzTransverseOperator>(out, N, S, spec.n_up, op_type_1, Q, e2_vec, pf);
            push_fs_pair_b<FixedSzTransverseOperator>(out, N, S, spec.n_up, op_type_2, Q, e2_vec, pf);
        }
    } else {
        if (use_xyz_basis) {
            out.obs_1.push_back(Operator(TransverseOperatorXYZ(N, S, op_type_1, Q, e1_vec, pf)));
            out.obs_2.push_back(Operator(TransverseOperatorXYZ(N, S, op_type_2, Q, e1_vec, pf)));
            out.obs_1.push_back(Operator(TransverseOperatorXYZ(N, S, op_type_1, Q, e2_vec, pf)));
            out.obs_2.push_back(Operator(TransverseOperatorXYZ(N, S, op_type_2, Q, e2_vec, pf)));
        } else {
            out.obs_1.push_back(Operator(TransverseOperator(N, S, op_type_1, Q, e1_vec, pf)));
            out.obs_2.push_back(Operator(TransverseOperator(N, S, op_type_2, Q, e1_vec, pf)));
            out.obs_1.push_back(Operator(TransverseOperator(N, S, op_type_1, Q, e2_vec, pf)));
            out.obs_2.push_back(Operator(TransverseOperator(N, S, op_type_2, Q, e2_vec, pf)));
        }
    }
}

void append_sublattice_pair(const OperatorSpec& spec,
                            const std::vector<double>& Q,
                            int op_type_1,
                            int op_type_2,
                            std::uint64_t sub_i,
                            std::uint64_t sub_j,
                            ObservablePairs& out) {
    const auto& pf = spec.positions_file;
    const auto N = spec.num_sites;
    const auto S = spec.spin_length;
    const auto U = spec.unit_cell_size;
    const bool emit_b = !spec.single_obs_only;
    if (spec.use_fixed_sz) {
        push_fs_pair_a<FixedSzSublatticeOperator>(out, sub_i, U, N, S, spec.n_up, op_type_1, Q, pf);
        if (emit_b) {
            push_fs_pair_b<FixedSzSublatticeOperator>(out, sub_j, U, N, S, spec.n_up, op_type_2, Q, pf);
        }
    } else {
        out.obs_1.push_back(Operator(SublatticeOperator(sub_i, U, N, S, op_type_1, Q, pf)));
        if (emit_b) {
            out.obs_2.push_back(Operator(SublatticeOperator(sub_j, U, N, S, op_type_2, Q, pf)));
        }
    }
}

void append_experimental_pair(const OperatorSpec& spec,
                              const std::vector<double>& Q,
                              ObservablePairs& out) {
    const auto& pf = spec.positions_file;
    const auto N = spec.num_sites;
    const auto S = spec.spin_length;
    const bool emit_b = !spec.single_obs_only;
    if (spec.use_fixed_sz) {
        push_fs_pair_a<FixedSzExperimentalOperator>(out, N, S, spec.n_up, spec.theta, Q, pf);
        if (emit_b) {
            push_fs_pair_b<FixedSzExperimentalOperator>(out, N, S, spec.n_up, spec.theta, Q, pf);
        }
    } else {
        out.obs_1.push_back(Operator(ExperimentalOperator(N, S, spec.theta, Q, pf)));
        if (emit_b) {
            out.obs_2.push_back(Operator(ExperimentalOperator(N, S, spec.theta, Q, pf)));
        }
    }
}

void append_transverse_experimental_pair(const OperatorSpec& spec,
                                         const std::vector<double>& Q,
                                         const std::vector<double>& e1_vec,
                                         const std::vector<double>& e2_vec,
                                         ObservablePairs& out) {
    const auto& pf = spec.positions_file;
    const auto N = spec.num_sites;
    const auto S = spec.spin_length;
    if (spec.use_fixed_sz) {
        push_fs_pair_a<FixedSzTransverseExperimentalOperator>(out, N, S, spec.n_up, spec.theta, Q, e1_vec, pf);
        push_fs_pair_b<FixedSzTransverseExperimentalOperator>(out, N, S, spec.n_up, spec.theta, Q, e1_vec, pf);
        push_fs_pair_a<FixedSzTransverseExperimentalOperator>(out, N, S, spec.n_up, spec.theta, Q, e2_vec, pf);
        push_fs_pair_b<FixedSzTransverseExperimentalOperator>(out, N, S, spec.n_up, spec.theta, Q, e2_vec, pf);
    } else {
        out.obs_1.push_back(Operator(TransverseExperimentalOperator(N, S, spec.theta, Q, e1_vec, pf)));
        out.obs_2.push_back(Operator(TransverseExperimentalOperator(N, S, spec.theta, Q, e1_vec, pf)));
        out.obs_1.push_back(Operator(TransverseExperimentalOperator(N, S, spec.theta, Q, e2_vec, pf)));
        out.obs_2.push_back(Operator(TransverseExperimentalOperator(N, S, spec.theta, Q, e2_vec, pf)));
    }
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

ObservablePairs build_observable_pairs(const OperatorSpec& spec) {
    if (spec.spin_combinations.empty()) {
        throw std::invalid_argument(
            "ed::dssf::build_observable_pairs: spin_combinations is empty");
    }
    if (spec.momentum_points.empty()) {
        throw std::invalid_argument(
            "ed::dssf::build_observable_pairs: momentum_points is empty");
    }
    if (spec.polarization.size() != 3) {
        throw std::invalid_argument(
            "ed::dssf::build_observable_pairs: polarization must be a 3-vector");
    }
    if (spec.num_sites == 0) {
        throw std::invalid_argument(
            "ed::dssf::build_observable_pairs: num_sites must be > 0");
    }

    const bool use_xyz_basis = (spec.basis == "xyz");
    ObservablePairs out;

    for (const auto& Q : spec.momentum_points) {
        std::array<double, 3> tb1{}, tb2{};
        const bool needs_transverse =
            (spec.operator_type == "transverse" ||
             spec.operator_type == "transverse_experimental");
        if (needs_transverse) {
            std::tie(tb1, tb2) = compute_transverse_bases(Q, spec.polarization);
        }
        const std::vector<double> e1_vec = {tb1[0], tb1[1], tb1[2]};
        const std::vector<double> e2_vec = {tb2[0], tb2[1], tb2[2]};

        for (const auto& combo : spec.spin_combinations) {
            const int op_type_1 = combo.first;
            const int op_type_2 = combo.second;

            // Convert operator indices for ladder basis: the pair-product
            // path maps the first slot 0->1 (Sp -> Sm) so that ⟨S-(Q)†
            // S+(Q)⟩ is built from `(Sm, Sp)`. The single_expectation
            // workflow skips this swap so that ⟨ψ|S+|ψ⟩ stays labelled
            // "Sp" rather than getting silently relabelled "Sm".
            int first = op_type_1;
            const int second = op_type_2;
            if (!use_xyz_basis && !spec.single_obs_only) {
                first = first == 2 ? 2 : 1 - first;
            }

            const std::string base_name =
                spec.single_obs_only
                    ? std::string(spin_combination_name(first, use_xyz_basis))
                    : std::string(spin_combination_name(first, use_xyz_basis)) +
                          std::string(spin_combination_name(second, use_xyz_basis));

            if (spec.operator_type == "sum") {
                std::stringstream name_ss;
                name_ss << base_name
                        << "_q_Qx" << Q[0]
                        << "_Qy"   << Q[1]
                        << "_Qz"   << Q[2];
                append_sum_pair(spec, Q, op_type_1, op_type_2,
                                use_xyz_basis, out);
                out.names.push_back(name_ss.str());

            } else if (spec.operator_type == "transverse") {
                std::stringstream name_sf, name_nsf;
                name_sf  << base_name
                         << "_q_Qx" << Q[0]
                         << "_Qy"   << Q[1]
                         << "_Qz"   << Q[2] << "_NSF";
                name_nsf << base_name
                         << "_q_Qx" << Q[0]
                         << "_Qy"   << Q[1]
                         << "_Qz"   << Q[2] << "_SF";
                append_transverse_pair(spec, Q, op_type_1, op_type_2,
                                       use_xyz_basis, e1_vec, e2_vec, out);
                out.names.push_back(name_sf.str());
                out.names.push_back(name_nsf.str());

            } else if (spec.operator_type == "sublattice") {
                auto emit_sublattice = [&](std::uint64_t sub_i, std::uint64_t sub_j) {
                    std::stringstream name_ss;
                    name_ss << base_name
                            << "_q_Qx" << Q[0]
                            << "_Qy"   << Q[1]
                            << "_Qz"   << Q[2]
                            << "_sub" << sub_i;
                    if (!spec.single_obs_only) {
                        name_ss << "_sub" << sub_j;
                    }
                    append_sublattice_pair(spec, Q, op_type_1, op_type_2,
                                           sub_i, sub_j, out);
                    out.names.push_back(name_ss.str());
                };

                if (spec.sublattice_filter) {
                    const auto [sub_i, sub_j] = *spec.sublattice_filter;
                    emit_sublattice(sub_i, sub_j);
                } else {
                    for (std::uint64_t sub_i = 0; sub_i < spec.unit_cell_size; ++sub_i) {
                        for (std::uint64_t sub_j = sub_i; sub_j < spec.unit_cell_size; ++sub_j) {
                            emit_sublattice(sub_i, sub_j);
                        }
                    }
                }

            } else if (spec.operator_type == "experimental") {
                if (combo.first == spec.spin_combinations[0].first &&
                    combo.second == spec.spin_combinations[0].second) {
                    std::stringstream name_ss;
                    name_ss << "Experimental_q_Qx" << Q[0]
                            << "_Qy"               << Q[1]
                            << "_Qz"               << Q[2]
                            << "_theta"            << spec.theta;
                    append_experimental_pair(spec, Q, out);
                    out.names.push_back(name_ss.str());
                }

            } else if (spec.operator_type == "transverse_experimental") {
                if (combo.first == spec.spin_combinations[0].first &&
                    combo.second == spec.spin_combinations[0].second) {
                    std::stringstream name_sf, name_nsf;
                    name_sf  << "TransverseExperimental_q_Qx" << Q[0]
                             << "_Qy"                          << Q[1]
                             << "_Qz"                          << Q[2]
                             << "_theta"                       << spec.theta
                             << "_NSF";
                    name_nsf << "TransverseExperimental_q_Qx" << Q[0]
                             << "_Qy"                          << Q[1]
                             << "_Qz"                          << Q[2]
                             << "_theta"                       << spec.theta
                             << "_SF";
                    append_transverse_experimental_pair(
                        spec, Q, e1_vec, e2_vec, out);
                    out.names.push_back(name_sf.str());
                    out.names.push_back(name_nsf.str());
                }

            } else {
                throw std::invalid_argument(
                    "ed::dssf::build_observable_pairs: unknown operator_type '" +
                    spec.operator_type + "'");
            }
        }
    }
    return out;
}

} // namespace ed::dssf

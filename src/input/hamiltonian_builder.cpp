// =============================================================================
// src/input/hamiltonian_builder.cpp
//
// Implementation of `ed::input::HamiltonianBuilder`. The builder
// accumulates one-/two-/three-body terms in the canonical (S+, S-, Sz)
// basis used by `class Operator` (see `include/ed/core/construct_ham.h`)
// and emits them either as an in-process Operator or as the legacy
// directory-of-text-files format consumed by `./ED <dir>`.
// =============================================================================

#include <ed/input/hamiltonian_builder.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include <ed/core/construct_ham.h>
#include <ed/input/file_io.h>

namespace ed::input {

namespace {

inline std::array<std::complex<double>, 3> non_kramer_factor_row(int sub_a) {
    using C = std::complex<double>;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const C gamma = std::exp(C(0.0, 2.0 * pi / 3.0));
    const C g2 = gamma * gamma;
    // Row of the per-(a,b) phase matrix for non-Kramers pyrochlore. Returns
    // {f(a,1), f(a,2), f(a,3)} with f(a,a) = 0 implied (filled below).
    switch (sub_a) {
        case 0: return {C(1, 0), gamma, g2};
        case 1: return {C(1, 0), g2, gamma};
        case 2: return {gamma, g2, C(1, 0)};
        case 3: return {g2, gamma, C(1, 0)};
        default: return {C(0, 0), C(0, 0), C(0, 0)};
    }
}

inline std::complex<double> non_kramer_factor(int sub_a, int sub_b) {
    if (sub_a == sub_b) return {0.0, 0.0};
    auto row = non_kramer_factor_row(sub_a);
    int j = sub_b > sub_a ? sub_b - 1 : sub_b;
    return row[j];
}

inline std::string op_to_string(Op op) {
    switch (op) {
        case Op::Sp: return "S+";
        case Op::Sm: return "S-";
        case Op::Sz: return "Sz";
    }
    return "??";
}

}  // namespace

HamiltonianBuilder::HamiltonianBuilder(std::size_t num_sites, double spin)
    : num_sites_(num_sites), spin_(spin) {
    if (num_sites == 0) {
        throw std::invalid_argument("HamiltonianBuilder: num_sites must be > 0");
    }
    if (num_sites >= 64) {
        throw std::invalid_argument(
            "HamiltonianBuilder: num_sites >= 64 is not supported by the "
            "underlying matrix-free Operator (1ULL << num_sites overflow)");
    }
}

HamiltonianBuilder& HamiltonianBuilder::add_one_body(Op op,
                                                    std::size_t site,
                                                    Complex coeff) {
    if (site >= num_sites_) {
        throw std::out_of_range(
            "HamiltonianBuilder::add_one_body: site index >= num_sites");
    }
    one_body_.push_back(OneBodyTerm{op, site, coeff});
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::add_two_body(
    Op op_i, std::size_t site_i,
    Op op_j, std::size_t site_j,
    Complex coeff) {
    if (site_i >= num_sites_ || site_j >= num_sites_) {
        throw std::out_of_range(
            "HamiltonianBuilder::add_two_body: site index >= num_sites");
    }
    two_body_.push_back(TwoBodyTerm{op_i, site_i, op_j, site_j, coeff});
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::add_three_body(
    Op op_i, std::size_t site_i,
    Op op_j, std::size_t site_j,
    Op op_k, std::size_t site_k,
    Complex coeff) {
    if (site_i >= num_sites_ || site_j >= num_sites_ || site_k >= num_sites_) {
        throw std::out_of_range(
            "HamiltonianBuilder::add_three_body: site index >= num_sites");
    }
    three_body_.push_back(
        ThreeBodyTerm{op_i, site_i, op_j, site_j, op_k, site_k, coeff});
    return *this;
}

// ---------------------------------------------------------------------------
// High-level shortcuts
// ---------------------------------------------------------------------------

HamiltonianBuilder& HamiltonianBuilder::heisenberg(
    const std::vector<std::pair<std::size_t, std::size_t>>& bonds, double J) {
    return xxz(bonds, J, J);
}

HamiltonianBuilder& HamiltonianBuilder::xxz(
    const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
    double Jxy, double Jz) {
    const Complex half_jxy(0.5 * Jxy, 0.0);
    const Complex jz(Jz, 0.0);
    for (auto [i, j] : bonds) {
        if (i == j) continue;
        add_two_body(Op::Sp, i, Op::Sm, j, half_jxy);
        add_two_body(Op::Sm, i, Op::Sp, j, half_jxy);
        if (Jz != 0.0) add_two_body(Op::Sz, i, Op::Sz, j, jz);
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::xyz(
    const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
    double Jxx, double Jyy, double Jzz) {
    // Sx Sx + Sy Sy = (Jxx+Jyy)/4 (S+S- + S-S+) + (Jxx-Jyy)/4 (S+S+ + S-S-)
    const Complex pm_pm((Jxx - Jyy) / 4.0, 0.0);
    const Complex pm_mp((Jxx + Jyy) / 4.0, 0.0);
    const Complex zz(Jzz, 0.0);
    for (auto [i, j] : bonds) {
        if (i == j) continue;
        if (Jxx != Jyy) {
            add_two_body(Op::Sp, i, Op::Sp, j, pm_pm);
            add_two_body(Op::Sm, i, Op::Sm, j, pm_pm);
        }
        if (Jxx + Jyy != 0.0) {
            add_two_body(Op::Sp, i, Op::Sm, j, pm_mp);
            add_two_body(Op::Sm, i, Op::Sp, j, pm_mp);
        }
        if (Jzz != 0.0) add_two_body(Op::Sz, i, Op::Sz, j, zz);
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::ising(
    const std::vector<std::pair<std::size_t, std::size_t>>& bonds, double J) {
    const Complex jz(J, 0.0);
    for (auto [i, j] : bonds) {
        if (i == j) continue;
        add_two_body(Op::Sz, i, Op::Sz, j, jz);
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::transverse_field_ising(
    const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
    double J, double h) {
    const Complex mj(-J, 0.0);
    for (auto [i, j] : bonds) {
        if (i == j) continue;
        add_two_body(Op::Sz, i, Op::Sz, j, mj);
    }
    if (h != 0.0) {
        const Complex half_mh(-0.5 * h, 0.0);
        for (std::size_t i = 0; i < num_sites_; ++i) {
            add_one_body(Op::Sp, i, half_mh);
            add_one_body(Op::Sm, i, half_mh);
        }
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::kitaev(
    const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
    const std::vector<int>& bond_axis,
    double K) {
    if (bonds.size() != bond_axis.size()) {
        throw std::invalid_argument(
            "HamiltonianBuilder::kitaev: bonds and bond_axis must have the "
            "same length");
    }
    const Complex k(K, 0.0);
    const Complex k_quarter(K / 4.0, 0.0);
    const Complex k_quarter_neg(-K / 4.0, 0.0);
    for (std::size_t b = 0; b < bonds.size(); ++b) {
        auto [i, j] = bonds[b];
        if (i == j) continue;
        switch (bond_axis[b]) {
            case 0: {  // x bond: K Sx Sx = K/4 (S+S+ + S+S- + S-S+ + S-S-)
                add_two_body(Op::Sp, i, Op::Sp, j, k_quarter);
                add_two_body(Op::Sp, i, Op::Sm, j, k_quarter);
                add_two_body(Op::Sm, i, Op::Sp, j, k_quarter);
                add_two_body(Op::Sm, i, Op::Sm, j, k_quarter);
                break;
            }
            case 1: {  // y bond: K Sy Sy = -K/4 (S+S+ - S+S- - S-S+ + S-S-)
                add_two_body(Op::Sp, i, Op::Sp, j, k_quarter_neg);
                add_two_body(Op::Sp, i, Op::Sm, j, k_quarter);
                add_two_body(Op::Sm, i, Op::Sp, j, k_quarter);
                add_two_body(Op::Sm, i, Op::Sm, j, k_quarter_neg);
                break;
            }
            case 2: {  // z bond
                add_two_body(Op::Sz, i, Op::Sz, j, k);
                break;
            }
            default:
                throw std::invalid_argument(
                    "HamiltonianBuilder::kitaev: bond_axis must be 0(x), 1(y), or 2(z)");
        }
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::dm(
    const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
    const std::vector<std::array<double, 3>>& D_per_bond) {
    if (bonds.size() != D_per_bond.size()) {
        throw std::invalid_argument(
            "HamiltonianBuilder::dm: bonds and D_per_bond must have the "
            "same length");
    }
    // S_i x S_j has components
    //   (S_i x S_j)_x = Sy_i Sz_j - Sz_i Sy_j
    //   (S_i x S_j)_y = Sz_i Sx_j - Sx_i Sz_j
    //   (S_i x S_j)_z = Sx_i Sy_j - Sy_i Sx_j
    // In the (S+, S-, Sz) basis:
    //   Sx = (S+ + S-) / 2
    //   Sy = (S+ - S-) / (2i)
    using C = Complex;
    for (std::size_t b = 0; b < bonds.size(); ++b) {
        auto [i, j] = bonds[b];
        if (i == j) continue;
        const auto& D = D_per_bond[b];
        const double Dx = D[0], Dy = D[1], Dz = D[2];
        if (Dx == 0.0 && Dy == 0.0 && Dz == 0.0) continue;

        // D . (S_i x S_j) expanded in the +/-/Sz basis
        //   = (Dx/2)(Sy_i Sz_j - Sz_i Sy_j)*2  (oops - direct expansion below)
        // We expand Sa_i Sb_j and group into (op_i, op_j, coeff):
        //   Sy_i Sz_j = (1/(2i))(S+_i Sz_j - S-_i Sz_j)
        //   Sz_i Sy_j = (1/(2i))(Sz_i S+_j - Sz_i S-_j)
        //   Sz_i Sx_j = (1/2 )(Sz_i S+_j + Sz_i S-_j)
        //   Sx_i Sz_j = (1/2 )(S+_i Sz_j + S-_i Sz_j)
        //   Sx_i Sy_j = (1/(4i))((S+_i + S-_i)(S+_j - S-_j))
        //   Sy_i Sx_j = (1/(4i))((S+_i - S-_i)(S+_j + S-_j))
        const C inv_2i(0.0, -0.5);
        const C inv_4i(0.0, -0.25);

        // Dx (Sy_i Sz_j - Sz_i Sy_j)
        if (Dx != 0.0) {
            add_two_body(Op::Sp, i, Op::Sz, j, Dx * inv_2i);
            add_two_body(Op::Sm, i, Op::Sz, j, -Dx * inv_2i);
            add_two_body(Op::Sz, i, Op::Sp, j, -Dx * inv_2i);
            add_two_body(Op::Sz, i, Op::Sm, j, Dx * inv_2i);
        }
        // Dy (Sz_i Sx_j - Sx_i Sz_j)
        if (Dy != 0.0) {
            add_two_body(Op::Sz, i, Op::Sp, j, C(Dy * 0.5, 0.0));
            add_two_body(Op::Sz, i, Op::Sm, j, C(Dy * 0.5, 0.0));
            add_two_body(Op::Sp, i, Op::Sz, j, C(-Dy * 0.5, 0.0));
            add_two_body(Op::Sm, i, Op::Sz, j, C(-Dy * 0.5, 0.0));
        }
        // Dz (Sx_i Sy_j - Sy_i Sx_j)
        if (Dz != 0.0) {
            // Sx_i Sy_j = inv_4i (S+_i + S-_i)(S+_j - S-_j)
            add_two_body(Op::Sp, i, Op::Sp, j, Dz * inv_4i);
            add_two_body(Op::Sp, i, Op::Sm, j, -Dz * inv_4i);
            add_two_body(Op::Sm, i, Op::Sp, j, Dz * inv_4i);
            add_two_body(Op::Sm, i, Op::Sm, j, -Dz * inv_4i);
            // -Sy_i Sx_j
            add_two_body(Op::Sp, i, Op::Sp, j, -Dz * inv_4i);
            add_two_body(Op::Sp, i, Op::Sm, j, -Dz * inv_4i);
            add_two_body(Op::Sm, i, Op::Sp, j, Dz * inv_4i);
            add_two_body(Op::Sm, i, Op::Sm, j, Dz * inv_4i);
        }
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::zeeman(std::array<double, 3> h_vec) {
    const Complex hx(h_vec[0], 0.0);
    const Complex hy(h_vec[1], 0.0);
    const Complex hz(h_vec[2], 0.0);
    // -h . S = -hx Sx - hy Sy - hz Sz
    //        = -(hx/2)(S+ + S-) + (hy/(2i))(S+ - S-) - hz Sz
    for (std::size_t i = 0; i < num_sites_; ++i) {
        if (h_vec[0] != 0.0) {
            add_one_body(Op::Sp, i, Complex(-h_vec[0] / 2.0, 0.0));
            add_one_body(Op::Sm, i, Complex(-h_vec[0] / 2.0, 0.0));
        }
        if (h_vec[1] != 0.0) {
            add_one_body(Op::Sp, i, Complex(0.0, h_vec[1] / 2.0));
            add_one_body(Op::Sm, i, Complex(0.0, -h_vec[1] / 2.0));
        }
        if (h_vec[2] != 0.0) add_one_body(Op::Sz, i, Complex(-h_vec[2], 0.0));
    }
    (void)hx;
    (void)hy;
    (void)hz;
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::zeeman_per_site(
    const std::vector<std::array<double, 3>>& h_per_site) {
    if (h_per_site.size() != num_sites_) {
        throw std::invalid_argument(
            "HamiltonianBuilder::zeeman_per_site: h_per_site.size() must equal num_sites");
    }
    for (std::size_t i = 0; i < num_sites_; ++i) {
        const auto& h = h_per_site[i];
        if (h[0] != 0.0) {
            add_one_body(Op::Sp, i, Complex(-h[0] / 2.0, 0.0));
            add_one_body(Op::Sm, i, Complex(-h[0] / 2.0, 0.0));
        }
        if (h[1] != 0.0) {
            add_one_body(Op::Sp, i, Complex(0.0, h[1] / 2.0));
            add_one_body(Op::Sm, i, Complex(0.0, -h[1] / 2.0));
        }
        if (h[2] != 0.0) add_one_body(Op::Sz, i, Complex(-h[2], 0.0));
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::on_site_field(double h_z) {
    if (h_z == 0.0) return *this;
    for (std::size_t i = 0; i < num_sites_; ++i) {
        add_one_body(Op::Sz, i, Complex(h_z, 0.0));
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::ring_exchange(
    const std::vector<std::array<std::size_t, 4>>& plaquettes,
    double K) {
    if (K == 0.0) return *this;
    // Spin-1/2 ring exchange P_p + P_p^{-1} expanded as
    // 4(S1.S2)(S3.S4) + 4(S1.S4)(S2.S3) - 4(S1.S3)(S2.S4) + 1/4
    // ⇒ four-body terms. We're storing only 1-/2-/3-body terms in the
    // canonical Operator format; each four-body S_a S_b S_c S_d is a sum
    // of two-body * two-body terms when the operators commute on disjoint
    // sites, but the C++ Operator does not natively support 4-body. We
    // therefore expand each ring as **three triples of two-body terms**
    // glued via the equivalent identity:
    //   4 (S1.S2)(S3.S4) = sum over (a,b in xyz) Sa_1 Sa_2 Sb_3 Sb_4
    // which is *not* representable as a single 2-body Operator term.
    //
    // For the C++ matrix-free Operator we instead route ring-exchange
    // through `three_body_data_`, which already supports the full
    // S^{op_i}_i S^{op_j}_j S^{op_k}_k tri-product, by **fusing** the
    // 4th leg via `Sz_a Sz_b = 1/4` for spin-1/2 anti-aligned pairs. This
    // is mathematically equivalent for the spin-1/2 ring on a square
    // plaquette but is **lossy** for spin > 1/2; we therefore restrict
    // ring_exchange to the spin-1/2 case.
    if (std::abs(spin_ - 0.5) > 1e-12) {
        throw std::runtime_error(
            "HamiltonianBuilder::ring_exchange: only spin-1/2 is supported");
    }
    // Concrete 4-body spin-1/2 ring exchange Hamiltonian (Misguich/Lhuillier):
    //   K (P_p + P_p^{-1}) = K/2 [ Sz_1 Sz_3 + Sz_2 Sz_4 + 4(Sz_1 Sz_2 + ...) - ...
    // The exact decomposition is non-trivial; for the v1 release we expose
    // the *primitive* triples via add_three_body and let the user supply
    // them. To keep the API surface complete we implement the simplest
    // safe case: three-leg ring exchange (S_i . S_j)(S_k . S_l) appears
    // only at >=4-body rank, so we **reject** plaquettes here with a
    // clear error message that points to add_three_body for tri-spin
    // chiral terms or to a dedicated four-body extension that remains
    // future work.
    if (!plaquettes.empty()) {
        throw std::runtime_error(
            "HamiltonianBuilder::ring_exchange: 4-body ring exchange is not "
            "yet representable in the canonical Operator (which is 1/2/3-body). "
            "Use add_three_body(...) for tri-spin chiral terms or open a "
            "follow-up issue for native 4-body support.");
    }
    return *this;
}

HamiltonianBuilder& HamiltonianBuilder::pyrochlore_non_kramers(
    const Lattice& lat,
    double Jxx,
    double Jyy,
    double Jzz,
    bool include_isotropic) {
    if (lat.num_sites != num_sites_) {
        throw std::invalid_argument(
            "pyrochlore_non_kramers: lattice/builder num_sites mismatch");
    }
    if (include_isotropic) xxz(lat.nn_pairs(), (Jxx + Jyy) / 2.0, Jzz);
    // Add the J_pmpm phase-twisted S+S+/S-S- terms (sublattice-dependent).
    const double Jpmpm = (Jxx - Jyy) / 4.0;
    if (Jpmpm == 0.0) return *this;
    for (auto [i, j] : lat.nn_pairs()) {
        const auto factor = non_kramer_factor(lat.sublattice[i],
                                              lat.sublattice[j]);
        const Complex c_pp = Complex(Jpmpm, 0.0) * factor;
        const Complex c_mm = std::conj(c_pp);
        if (std::abs(c_pp) > 0.0) add_two_body(Op::Sm, i, Op::Sm, j, c_pp);
        if (std::abs(c_mm) > 0.0) add_two_body(Op::Sp, i, Op::Sp, j, c_mm);
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Output paths
// ---------------------------------------------------------------------------

void HamiltonianBuilder::emit_into(Operator& op) const {
    if (op.getNumBits() != num_sites_) {
        throw std::invalid_argument(
            "HamiltonianBuilder::emit_into: Operator num_bits != builder num_sites");
    }
    for (const auto& t : one_body_) {
        Operator::TransformData td;
        td.op_type = op_to_int(t.op);
        td.site_index = t.site;
        td.coefficient = t.coeff;
        td.is_two_body = false;
        op.transform_data_.push_back(td);
    }
    for (const auto& t : two_body_) {
        Operator::TransformData td;
        td.op_type = op_to_int(t.op_i);
        td.site_index = t.site_i;
        td.op_type_2 = op_to_int(t.op_j);
        td.site_index_2 = t.site_j;
        td.coefficient = t.coeff;
        td.is_two_body = true;
        op.transform_data_.push_back(td);
    }
    for (const auto& t : three_body_) {
        Operator::ThreeBodyTransformData td;
        td.op_type_1 = op_to_int(t.op_i);
        td.site_index_1 = t.site_i;
        td.op_type_2 = op_to_int(t.op_j);
        td.site_index_2 = t.site_j;
        td.op_type_3 = op_to_int(t.op_k);
        td.site_index_3 = t.site_k;
        td.coefficient = t.coeff;
        op.three_body_data_.push_back(td);
    }
    op.invalidateMatrixCaches();
}

std::shared_ptr<Operator> HamiltonianBuilder::to_operator() const {
    auto op = std::make_shared<Operator>(static_cast<uint64_t>(num_sites_),
                                         static_cast<float>(spin_));
    emit_into(*op);
    return op;
}

void HamiltonianBuilder::write_directory(
    const std::string& output_dir,
    const Lattice* lat,
    const FileOptions& opts) const {
    namespace fs = std::filesystem;
    fs::create_directories(output_dir);
    const std::string trans_path = output_dir + "/" + opts.trans_filename;
    const std::string inter_path = output_dir + "/" + opts.inter_all_filename;
    write_trans_file(trans_path, one_body_, opts.tol);
    write_inter_all_file(inter_path, two_body_, opts.tol);
    if (!three_body_.empty()) {
        write_three_body_file(output_dir + "/" + opts.three_body_filename,
                              three_body_, opts.tol);
    }
    if (lat && opts.write_positions) {
        write_positions_file(output_dir + "/" + opts.positions_filename,
                             lat->positions);
    }
    // Observable files
    for (Op op : opts.one_body_obs) {
        const std::string name =
            "one_body_correlations" + op_to_string(op) + ".dat";
        write_one_body_correlation_file(output_dir + "/" + name, op,
                                        num_sites_);
    }
    for (auto [oa, ob] : opts.two_body_obs) {
        const std::string name = "two_body_correlations" +
                                 op_to_string(oa) + op_to_string(ob) + ".dat";
        write_two_body_correlation_file(output_dir + "/" + name, oa, ob,
                                        num_sites_);
    }
    if (lat && opts.write_lattice_metadata) {
        std::ofstream meta(output_dir + "/lattice.json");
        meta << "{\n  \"num_sites\": " << num_sites_ << ",\n";
        meta << "  \"label\": \"" << lat->label << "\",\n";
        meta << "  \"pbc\": " << (lat->pbc ? "true" : "false") << ",\n";
        meta << "  \"sublattice\": [";
        for (std::size_t i = 0; i < lat->sublattice.size(); ++i) {
            meta << lat->sublattice[i];
            if (i + 1 < lat->sublattice.size()) meta << ", ";
        }
        meta << "]\n}\n";
    }
}

double HamiltonianBuilder::l1_norm() const noexcept {
    double s = 0.0;
    for (const auto& t : one_body_) s += std::abs(t.coeff);
    for (const auto& t : two_body_) s += std::abs(t.coeff);
    for (const auto& t : three_body_) s += std::abs(t.coeff);
    return s;
}

HamiltonianBuilder& HamiltonianBuilder::clear() {
    one_body_.clear();
    two_body_.clear();
    three_body_.clear();
    return *this;
}

}  // namespace ed::input

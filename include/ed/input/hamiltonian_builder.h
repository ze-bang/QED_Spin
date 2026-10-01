// =============================================================================
// include/ed/input/hamiltonian_builder.h
//
// `HamiltonianBuilder`: composable, fluent C++ API that accumulates
// one-/two-/three-body spin terms and emits them in-process as an
// `ed::Operator` (no file I/O).
//
// Term shortcuts cover Heisenberg / XXZ / XYZ / Kitaev / Ising-with-
// transverse-field / Zeeman / DM / single-site / ring exchange, plus
// arbitrary low-level add_one/two/three_body, so a textbook Hamiltonian
// does not need raw per-bond loops.
// =============================================================================

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ed/input/lattice.h>
#include <ed/input/types.h>

// Forward-declare to avoid pulling the heavy construct_ham.h header in.
class Operator;

namespace ed::input {


// =============================================================================
// HamiltonianBuilder
// =============================================================================

class HamiltonianBuilder {
public:
    HamiltonianBuilder(std::size_t num_sites, double spin = 0.5);

    // ------------------------------------------------------------------
    // Low-level term insertion (purely additive)
    // ------------------------------------------------------------------

    HamiltonianBuilder& add_one_body(Op op, std::size_t site, Complex coeff);

    HamiltonianBuilder& add_two_body(
        Op op_i, std::size_t site_i,
        Op op_j, std::size_t site_j,
        Complex coeff);

    HamiltonianBuilder& add_three_body(
        Op op_i, std::size_t site_i,
        Op op_j, std::size_t site_j,
        Op op_k, std::size_t site_k,
        Complex coeff);

    // ------------------------------------------------------------------
    // High-level Hamiltonian shortcuts
    // ------------------------------------------------------------------

    // Isotropic Heisenberg on a list of bonds:
    //   H += J * sum_<ij> [Sx_i Sx_j + Sy_i Sy_j + Sz_i Sz_j]
    // emitted in the (S+, S-, Sz) basis as
    //   J * (0.5 S+_i S-_j + 0.5 S-_i S+_j + Sz_i Sz_j).
    HamiltonianBuilder& heisenberg(
        const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
        double J);

    // Anisotropic XXZ on a list of bonds:
    //   H += sum_<ij> [Jxy * (Sx_i Sx_j + Sy_i Sy_j) + Jz * Sz_i Sz_j]
    HamiltonianBuilder& xxz(
        const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
        double Jxy,
        double Jz);

    // Fully anisotropic XYZ on a list of bonds:
    //   H += sum_<ij> [Jxx Sx_i Sx_j + Jyy Sy_i Sy_j + Jzz Sz_i Sz_j]
    HamiltonianBuilder& xyz(
        const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
        double Jxx,
        double Jyy,
        double Jzz);

    // Pure Ising:
    //   H += J * sum_<ij> Sz_i Sz_j
    HamiltonianBuilder& ising(
        const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
        double J);

    // Transverse-field Ising:
    //   H += -J sum_<ij> Sz_i Sz_j - h sum_i Sx_i
    HamiltonianBuilder& transverse_field_ising(
        const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
        double J,
        double h);

    // Kitaev: per-bond axis (0=x, 1=y, 2=z) selects which `Sa Sa` term
    // is added with coupling `K`. `bonds` and `bond_axis` must have the
    // same length.
    HamiltonianBuilder& kitaev(
        const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
        const std::vector<int>& bond_axis,
        double K);

    // Dzyaloshinskii-Moriya:
    //   H += sum_<ij> D_ij . (S_i x S_j)
    // `D_per_bond` carries the 3-vector D_ij for each bond (same length).
    HamiltonianBuilder& dm(
        const std::vector<std::pair<std::size_t, std::size_t>>& bonds,
        const std::vector<std::array<double, 3>>& D_per_bond);

    // Uniform Zeeman field:
    //   H += -h_vec . sum_i S_i
    HamiltonianBuilder& zeeman(std::array<double, 3> h_vec);

    // Site-resolved Zeeman field (e.g. local rotated frames on the
    // pyrochlore lattice):
    //   H += -sum_i h_per_site[i] . S_i
    HamiltonianBuilder& zeeman_per_site(
        const std::vector<std::array<double, 3>>& h_per_site);

    // Single-ion uniaxial-z field on every site:
    //   H += sum_i h * Sz_i
    // (Equivalent to `zeeman({0,0,-h})` up to sign convention; provided
    // as a common shortcut.)
    HamiltonianBuilder& on_site_field(double h_z);

    // 4-site ring exchange:
    //   H += K * sum_p (P_p + P_p^-1)
    // where P_p cyclically permutes the four spins around plaquette p.
    // For spin-1/2 we expand into the standard 4-spin form
    // P + P^{-1} = 4 (S_1.S_2)(S_3.S_4) + 4 (S_1.S_4)(S_2.S_3) - 4 (S_1.S_3)(S_2.S_4) + 1/4
    // emitted as four-body terms; the constant is absorbed into the
    // diagonal Sz_i Sz_i counter (skipped here -- it is a global energy
    // shift). Intended for pyrochlore / kagome ring-exchange Hamiltonians;
    // use with care because four-body terms blow up the operator size.
    HamiltonianBuilder& ring_exchange(
        const std::vector<std::array<std::size_t, 4>>& plaquettes,
        double K);

    // Pyrochlore non-Kramers Jpmpm phase. Adds the standard XXZ
    // Heisenberg (via xxz above) plus the J_pmpm phase-twisted
    // S+S+ / S-S- terms.
    //
    //   J_pm   = -(Jxx + Jyy)/4
    //   J_pmpm = (Jxx - Jyy)/4
    // For each bond (i, j), with sublattice indices `sub[i]`, `sub[j]`
    // (drawn from `Lattice.sublattice`), the term added is
    //   J_pmpm * gamma(sub[i], sub[j]) * S-_i S-_j   (and h.c.)
    // The labels must be the pyrochlore's (one per site, in 0..3, different
    // across every bond), and include_isotropic = false needs Jzz = 0 (Jzz
    // enters only the XXZ part); anything else is ed::InvalidRequest.
    HamiltonianBuilder& pyrochlore_non_kramers(
        const Lattice& lat,
        double Jxx,
        double Jyy,
        double Jzz,
        bool include_isotropic = true);

    // ------------------------------------------------------------------
    // Output paths
    // ------------------------------------------------------------------

    // Materialise into an in-memory `ed::Operator`. **No file I/O.**
    std::shared_ptr<Operator> to_operator() const;

    // Overload that writes the terms directly into an existing Operator
    // (used by the pybind11 bindings to wire into `qed.Operator`).
    void emit_into(Operator& op) const;


    // ------------------------------------------------------------------
    // Inspection
    // ------------------------------------------------------------------

    const std::vector<OneBodyTerm>& one_body_terms() const noexcept { return one_body_; }
    const std::vector<TwoBodyTerm>& two_body_terms() const noexcept { return two_body_; }
    const std::vector<ThreeBodyTerm>& three_body_terms() const noexcept { return three_body_; }

    std::size_t num_sites() const noexcept { return num_sites_; }
    double spin() const noexcept { return spin_; }

    // Sum of |coefficient| across all term lists (rough size metric).
    double l1_norm() const noexcept;

    // Drop accumulated terms (returns *this for chaining).
    HamiltonianBuilder& clear();

private:
    std::size_t num_sites_;
    double spin_;
    std::vector<OneBodyTerm> one_body_;
    std::vector<TwoBodyTerm> two_body_;
    std::vector<ThreeBodyTerm> three_body_;
};

}  // namespace ed::input

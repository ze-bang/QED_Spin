// =============================================================================
// include/ed/input/types.h
//
// Common types for the ed_input library: the spin operator enum, and the bond,
// plaquette and position records of the lattice geometry. The Hamiltonian
// builder is Python (python/qed/_builder.py); it reads Op through the bindings
// (qed.input.Op).
//
// Op speaks the (S+, S-, Sz) codes of the C++ `Operator`
// (`include/ed/ops/operator.h`) records, so a builder record maps onto an
// Operator record with no basis change.
// =============================================================================

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ed::input {

using Complex = std::complex<double>;

// Spin operator codes that match the C++ Operator's TransformData::op_type
// (0 = S+, 1 = S-, 2 = Sz), as a strong enum (bound as qed.input.Op).
enum class Op : std::uint8_t {
    Sp = 0,
    Sm = 1,
    Sz = 2,
};

inline constexpr std::uint8_t op_to_int(Op op) noexcept { return static_cast<std::uint8_t>(op); }

// A bond from site i to site j, kept in that orientation: it matters for
// antisymmetric couplings such as Dzyaloshinskii-Moriya. The optional
// `bond_type` tag carries lattice metadata (e.g. Kitaev x/y/z bond colour).
struct Bond {
    std::size_t i;
    std::size_t j;
    int bond_type = 0;

    Bond() = default;
    Bond(std::size_t i_, std::size_t j_, int t = 0) : i(i_), j(j_), bond_type(t) {}
};

inline bool operator==(const Bond& a, const Bond& b) noexcept {
    return a.i == b.i && a.j == b.j && a.bond_type == b.bond_type;
}

// Plaquette / ring (4 sites for ring-exchange terms).
struct Plaquette {
    std::array<std::size_t, 4> sites;
    int plaquette_type = 0;
};

// 3D Cartesian site position.
using Position = std::array<double, 3>;

}  // namespace ed::input

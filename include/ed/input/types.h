// =============================================================================
// include/ed/input/types.h
//
// Common types for the ed_input library: spin operator enum, term records,
// and small typedefs shared between lattice geometry, the Hamiltonian
// builder, and the file writers.
//
// All terms speak the canonical (S+, S-, Sz) basis used by the C++
// `Operator` (`include/ed/core/construct_ham.h`). This is the same basis
// used in `Trans.dat` / `InterAll.dat` / `ThreeBodyG.dat`, so writing
// these files (or building a programmatic `Operator`) is a one-step
// translation with no extra basis change.
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
// (0 = S+, 1 = S-, 2 = Sz). Storing them as a strong enum lets the
// templated DSL refuse mistakes at compile time without losing the wire
// format guarantee.
enum class Op : std::uint8_t {
    Sp = 0,
    Sm = 1,
    Sz = 2,
};

inline constexpr std::uint8_t op_to_int(Op op) noexcept {
    return static_cast<std::uint8_t>(op);
}

// A bond in a graph; canonical orientation is (i < j). The optional
// `bond_type` tag carries lattice metadata (e.g. Kitaev x/y/z bond colour,
// nearest-vs-next-nearest, sublattice pair label).
struct Bond {
    std::size_t i;
    std::size_t j;
    int bond_type = 0;

    Bond() = default;
    Bond(std::size_t i_, std::size_t j_, int t = 0) : i(i_), j(j_), bond_type(t) {
        if (i > j) {
            std::swap(i, j);
        }
    }
};

inline bool operator==(const Bond& a, const Bond& b) noexcept {
    return a.i == b.i && a.j == b.j && a.bond_type == b.bond_type;
}

// Plaquette / ring (4 sites for ring-exchange terms).
struct Plaquette {
    std::array<std::size_t, 4> sites;
    int plaquette_type = 0;
};

// Building blocks that the Hamiltonian builder accumulates.
struct OneBodyTerm {
    Op op;
    std::size_t site;
    Complex coeff;
};

struct TwoBodyTerm {
    Op op_i;
    std::size_t site_i;
    Op op_j;
    std::size_t site_j;
    Complex coeff;
};

struct ThreeBodyTerm {
    Op op_i;
    std::size_t site_i;
    Op op_j;
    std::size_t site_j;
    Op op_k;
    std::size_t site_k;
    Complex coeff;
};

// 3D Cartesian site position.
using Position = std::array<double, 3>;

}  // namespace ed::input

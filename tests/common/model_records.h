// =============================================================================
// tests/common/model_records.h -- the records the Hamiltonian builder (now Python,
// python/qed/_builder.py) writes for a few models, for the C++ tests that want those exact
// records (the dm_z case, for example, relies on its cancelling S+S+ / S-S- pairs). Each writer
// appends to an Operator the records the builder method of the same name emits, in its order.
// =============================================================================
#pragma once

#include <ed/ops/operator.h>

#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace ed_tests::records {

using Cx = std::complex<double>;
using Bonds = std::vector<std::pair<std::size_t, std::size_t>>;
inline constexpr std::uint8_t Sp = 0, Sm = 1, Sz = 2;

/// Jxy (Sx Sx + Sy Sy) + Jz Sz Sz per bond.
inline void xxz(Operator& H, const Bonds& bonds, double Jxy, double Jz) {
    for (const auto& [i, j] : bonds) {
        if (i == j) continue;
        H.addTwoBodyTerm(Sp, i, Sm, j, Cx(0.5 * Jxy, 0.0));
        H.addTwoBodyTerm(Sm, i, Sp, j, Cx(0.5 * Jxy, 0.0));
        if (Jz != 0.0) H.addTwoBodyTerm(Sz, i, Sz, j, Cx(Jz, 0.0));
    }
}

inline void heisenberg(Operator& H, const Bonds& bonds, double J) { xxz(H, bonds, J, J); }

/// Jxx Sx Sx + Jyy Sy Sy + Jzz Sz Sz per bond.
inline void xyz(Operator& H, const Bonds& bonds, double Jxx, double Jyy, double Jzz) {
    for (const auto& [i, j] : bonds) {
        if (i == j) continue;
        if (Jxx != Jyy) {
            H.addTwoBodyTerm(Sp, i, Sp, j, Cx((Jxx - Jyy) / 4.0, 0.0));
            H.addTwoBodyTerm(Sm, i, Sm, j, Cx((Jxx - Jyy) / 4.0, 0.0));
        }
        if (Jxx + Jyy != 0.0) {
            H.addTwoBodyTerm(Sp, i, Sm, j, Cx((Jxx + Jyy) / 4.0, 0.0));
            H.addTwoBodyTerm(Sm, i, Sp, j, Cx((Jxx + Jyy) / 4.0, 0.0));
        }
        if (Jzz != 0.0) H.addTwoBodyTerm(Sz, i, Sz, j, Cx(Jzz, 0.0));
    }
}

/// -h . S on every site.
inline void zeeman(Operator& H, double hx, double hy, double hz) {
    for (std::uint64_t i = 0; i < H.getNumBits(); ++i) {
        if (hx != 0.0) {
            H.addOneBodyTerm(Sp, i, Cx(-hx / 2.0, 0.0));
            H.addOneBodyTerm(Sm, i, Cx(-hx / 2.0, 0.0));
        }
        if (hy != 0.0) {
            H.addOneBodyTerm(Sp, i, Cx(0.0, hy / 2.0));
            H.addOneBodyTerm(Sm, i, Cx(0.0, -hy / 2.0));
        }
        if (hz != 0.0) H.addOneBodyTerm(Sz, i, Cx(-hz, 0.0));
    }
}

/// -J Sz Sz per bond - h Sx on every site.
inline void transverse_field_ising(Operator& H, const Bonds& bonds, double J, double h) {
    for (const auto& [i, j] : bonds)
        if (i != j) H.addTwoBodyTerm(Sz, i, Sz, j, Cx(-J, 0.0));
    if (h != 0.0)
        for (std::uint64_t i = 0; i < H.getNumBits(); ++i) {
            H.addOneBodyTerm(Sp, i, Cx(-0.5 * h, 0.0));
            H.addOneBodyTerm(Sm, i, Cx(-0.5 * h, 0.0));
        }
}

/// D . (S_i x S_j) per bond, one D for every bond (the builder's dm with a uniform vector).
inline void dm(Operator& H, const Bonds& bonds, double Dx, double Dy, double Dz) {
    const Cx i2(0.0, -0.5), i4(0.0, -0.25);
    for (const auto& [i, j] : bonds) {
        if (i == j || (Dx == 0.0 && Dy == 0.0 && Dz == 0.0)) continue;
        if (Dx != 0.0) {
            H.addTwoBodyTerm(Sp, i, Sz, j, Dx * i2);
            H.addTwoBodyTerm(Sm, i, Sz, j, -Dx * i2);
            H.addTwoBodyTerm(Sz, i, Sp, j, -Dx * i2);
            H.addTwoBodyTerm(Sz, i, Sm, j, Dx * i2);
        }
        if (Dy != 0.0) {
            H.addTwoBodyTerm(Sz, i, Sp, j, Cx(Dy * 0.5, 0.0));
            H.addTwoBodyTerm(Sz, i, Sm, j, Cx(Dy * 0.5, 0.0));
            H.addTwoBodyTerm(Sp, i, Sz, j, Cx(-Dy * 0.5, 0.0));
            H.addTwoBodyTerm(Sm, i, Sz, j, Cx(-Dy * 0.5, 0.0));
        }
        if (Dz != 0.0) {   // Sx_i Sy_j - Sy_i Sx_j, with the S+S+ / S-S- records that cancel
            H.addTwoBodyTerm(Sp, i, Sp, j, Dz * i4);
            H.addTwoBodyTerm(Sp, i, Sm, j, -Dz * i4);
            H.addTwoBodyTerm(Sm, i, Sp, j, Dz * i4);
            H.addTwoBodyTerm(Sm, i, Sm, j, -Dz * i4);
            H.addTwoBodyTerm(Sp, i, Sp, j, -Dz * i4);
            H.addTwoBodyTerm(Sp, i, Sm, j, -Dz * i4);
            H.addTwoBodyTerm(Sm, i, Sp, j, Dz * i4);
            H.addTwoBodyTerm(Sm, i, Sm, j, Dz * i4);
        }
    }
}

/// K S^a_i S^a_j with a = axis[b] (0: x, 1: y, 2: z) per bond.
inline void kitaev(Operator& H, const Bonds& bonds, const std::vector<int>& axis, double K) {
    for (std::size_t b = 0; b < bonds.size(); ++b) {
        const auto [i, j] = bonds[b];
        if (i == j) continue;
        const Cx q(K / 4.0, 0.0), mq(-K / 4.0, 0.0);
        if (axis[b] == 0) {
            H.addTwoBodyTerm(Sp, i, Sp, j, q); H.addTwoBodyTerm(Sp, i, Sm, j, q);
            H.addTwoBodyTerm(Sm, i, Sp, j, q); H.addTwoBodyTerm(Sm, i, Sm, j, q);
        } else if (axis[b] == 1) {
            H.addTwoBodyTerm(Sp, i, Sp, j, mq); H.addTwoBodyTerm(Sp, i, Sm, j, q);
            H.addTwoBodyTerm(Sm, i, Sp, j, q); H.addTwoBodyTerm(Sm, i, Sm, j, mq);
        } else {
            H.addTwoBodyTerm(Sz, i, Sz, j, Cx(K, 0.0));
        }
    }
}

}  // namespace ed_tests::records

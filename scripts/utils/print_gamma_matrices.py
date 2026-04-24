#!/usr/bin/env python3
"""Print the canonical three-spin phase tables used by helper_pyrochlore_super.py.

Reference: Kadowaki, Wakita, Fak, Ollivier, Ohira-Kawamura,
Phys. Rev. B 105, 014439 (2022) [arXiv:2109.08799], Tables III, IV, V.

The non-Kramers pyrochlore three-spin Hamiltonian is

    H_3s = sum_{i=1}^{3} J_{3s,i}
           sum_{<r, r', r''>^(i)}
              [ exp(i phi^(i)_{r,r',r''}) sigma^+_r sigma^z_{r'} sigma^z_{r''} + h.c. ]

where the geometric class i is

    Type 1 (collinear): sub_{r'} == sub_{r''}, with r' and r'' on opposite tetras at r,
    Type 2 (same-tetra): r' and r'' both NN of r in the same tetrahedron,
    Type 3 (cross): r' and r'' on different tetras of r, and sub_{r'} != sub_{r''}.

The phases phi^(i) are integer multiples of 2pi/3 fixed by C_3 / inversion symmetry.
We print them in 0-indexed convention (sub_idx = nu - 1).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from edlib.helper_pyrochlore_super import PHI_COLLINEAR_2PI3, PHI_PAIR_2PI3  # noqa: E402

omega = np.exp(1j * 2 * np.pi / 3)


def fmt_phase(p):
    if p == 0:
        return "  1   "
    if p == 1:
        return "  ω   "
    if p == -1 or p == 2:
        return "  ω²  "
    return f"{omega**p:+.3f}"


def print_collinear():
    print("=" * 70)
    print("Type 1 (collinear): J_{3s,1} sigma^+_r sigma^z_{r'} sigma^z_{r''} + h.c.")
    print("Phase phi(sub_j, sub') in units of 2pi/3 (sub' = sub'')")
    print("=" * 70)
    print()
    header = "sub_j |    sub'=0   |    sub'=1   |    sub'=2   |    sub'=3   "
    print(header)
    print("-" * len(header))
    for sub_j in range(4):
        cells = []
        for mu in range(4):
            if mu == sub_j:
                cells.append("     -      ")
                continue
            p = PHI_COLLINEAR_2PI3[sub_j][mu]
            cells.append(f"  p={p:+d} ({fmt_phase(p).strip()})")
        print(f"  {sub_j}   | " + " | ".join(c.ljust(11) for c in cells))
    print()


def print_pair():
    print("=" * 70)
    print("Type 2 (same-tetra): J_{3s,2} sigma^+_r sigma^z_{r'} sigma^z_{r''} + h.c.")
    print("Type 3 (cross):      J_{3s,3} sigma^+_r sigma^z_{r'} sigma^z_{r''} + h.c.")
    print("Phase phi(sub_j, {sub', sub''}) in units of 2pi/3 (same map for Type 2 & 3;")
    print("only the coupling J_{3s,2} vs J_{3s,3} differs).")
    print("=" * 70)
    print()
    for sub_j in range(4):
        print(f"sub_j = {sub_j}:")
        for pair_set, p in PHI_PAIR_2PI3[sub_j].items():
            a, b = sorted(pair_set)
            print(f"   {{sub' = {a}, sub'' = {b}}}: p = {p:+d}  -> exp(i p 2pi/3) = {fmt_phase(p).strip()}")
        print()


def main():
    print()
    print("Canonical non-Kramers pyrochlore three-spin phase tables")
    print("Kadowaki et al., Phys. Rev. B 105, 014439 (2022), Tables III/IV/V.")
    print(f"omega = exp(i 2pi/3) = {omega:+.6f}")
    print(f"omega^2 = {omega**2:+.6f}")
    print()
    print_collinear()
    print_pair()


if __name__ == "__main__":
    main()

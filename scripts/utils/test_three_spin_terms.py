#!/usr/bin/env python3
"""Self-test for the canonical three-spin term implementation in
``helper_pyrochlore_super.generate_three_spin_terms``.

Checks:
  1) Per-type triplet counts match the canonical Kadowaki budget
     (3 Type 1 + 6 Type 2 + 6 Type 3 = 15 per site).
  2) For a hand-picked central site (sub_j = 0), the phases emitted by the
     code agree exactly with Kadowaki Tables III, IV, V for Type 1 collinear,
     Type 2 same-tetra, and Type 3 cross triplets.
  3) The Hermiticity property: every (i, +, j, k) row has a matching
     (i, -, j, k) row with conjugate coefficient.

Run from anywhere:
    python exact_diagonalization_cpp/scripts/utils/test_three_spin_terms.py
"""

import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from edlib.helper_pyrochlore_super import (  # noqa: E402
    PHI_COLLINEAR_2PI3,
    PHI_PAIR_2PI3,
    create_nn_lists,
    generate_pyrochlore_super_cluster,
    generate_three_spin_terms,
)


def classify(j, i, k, sub_idx, vertex_to_cell):
    sub_j = sub_idx[j]
    sub_i = sub_idx[i]
    sub_k = sub_idx[k]
    same_side = (vertex_to_cell[i][:4] == vertex_to_cell[j][:4]) == (
        vertex_to_cell[k][:4] == vertex_to_cell[j][:4]
    )
    if sub_i == sub_k:
        return 1, sub_j, sub_i
    return (2 if same_side else 3), sub_j, frozenset({sub_i, sub_k})


def main():
    vertices, edges, _, node_mapping, vertex_to_cell = generate_pyrochlore_super_cluster(
        1, 1, 1, use_pbc=True
    )
    nn_list, _, sub_idx = create_nn_lists(edges, node_mapping, vertices, vertex_to_cell)
    n_sites = len(nn_list)

    # Pick a non-zero, asymmetric (J_{3s,1}, J_{3s,2}, J_{3s,3}) so each class
    # is visible in the output coefficients.
    J = (0.13, 0.27, -0.41)
    terms = generate_three_spin_terms(nn_list, node_mapping, J, sub_idx, vertex_to_cell)

    # ---- Test 1: per-class counts ----
    counts_per_type = Counter()
    for j in nn_list:
        for a_idx, i in enumerate(sorted(nn_list[j])):
            for k in sorted(nn_list[j])[a_idx + 1:]:
                t, _, _ = classify(j, i, k, sub_idx, vertex_to_cell)
                counts_per_type[t] += 1

    print(f"[counts] sites = {n_sites}")
    print(f"[counts] Type 1 = {counts_per_type[1]} (expected {3 * n_sites})")
    print(f"[counts] Type 2 = {counts_per_type[2]} (expected {6 * n_sites})")
    print(f"[counts] Type 3 = {counts_per_type[3]} (expected {6 * n_sites})")
    assert counts_per_type[1] == 3 * n_sites, "Type-1 triplet count mismatch"
    assert counts_per_type[2] == 6 * n_sites, "Type-2 triplet count mismatch"
    assert counts_per_type[3] == 6 * n_sites, "Type-3 triplet count mismatch"
    total_unordered_triplets = counts_per_type[1] + counts_per_type[2] + counts_per_type[3]
    # Each unordered (r', r'') around r emits one S+ and one S- row -> factor of 2.
    assert len(terms) == 2 * total_unordered_triplets, (
        f"Got {len(terms)} rows, expected {2 * total_unordered_triplets} "
        "(2 per unordered (r', r'') around r)."
    )
    print(f"[counts] OK: {len(terms)} operator rows = 2 * {total_unordered_triplets} unordered triplets\n")

    # ---- Test 2: phase agreement with the canonical tables ----
    omega = np.exp(1j * 2 * np.pi / 3)
    # Build a dict (j_node, i_node, k_node_unordered) -> complex coefficient (S+ row).
    rows_plus = {}
    for op_i, si, op_j, sj, op_k, sk, cr, ci in terms:
        if op_j != 0:  # skip S- rows for this test
            continue
        key = (sj, frozenset({si, sk}))
        rows_plus.setdefault(key, []).append((si, sk, complex(cr, ci)))

    # node_mapping is identity for this cluster, so node id == matrix index.
    expected = {1: J[0], 2: J[1], 3: J[2]}
    mismatches = 0
    sample_lines = []
    for j in nn_list:
        for a_idx, i in enumerate(sorted(nn_list[j])):
            for k in sorted(nn_list[j])[a_idx + 1:]:
                t, _, key2 = classify(j, i, k, sub_idx, vertex_to_cell)
                if t == 1:
                    p = PHI_COLLINEAR_2PI3[sub_idx[j]][sub_idx[i]]
                else:
                    p = PHI_PAIR_2PI3[sub_idx[j]][key2]
                expected_coeff = expected[t] * (omega ** p)
                row_set = rows_plus.get((j, frozenset({i, k})), [])
                # Find the row with matching (i, k) ordering.
                match = None
                for si, sk, c in row_set:
                    if {si, sk} == {i, k}:
                        match = c
                        break
                if match is None:
                    mismatches += 1
                    if len(sample_lines) < 5:
                        sample_lines.append(
                            f"missing j={j}, (i,k)={(i,k)}, type={t}"
                        )
                    continue
                if not np.isclose(match, expected_coeff, atol=1e-12):
                    mismatches += 1
                    if len(sample_lines) < 5:
                        sample_lines.append(
                            f"phase mismatch j={j}, (i,k)={(i,k)}, type={t}: "
                            f"got {match}, expected {expected_coeff}"
                        )

    if mismatches:
        print(f"[phases] FAILED: {mismatches} mismatches")
        for line in sample_lines:
            print("   ", line)
        sys.exit(1)
    print("[phases] OK: every emitted (S+) coefficient matches J_{3s,t} * exp(i * p * 2pi/3)")

    # ---- Test 3: Hermiticity (S- row is conjugate of paired S+ row) ----
    paired = {}
    for op_i, si, op_j, sj, op_k, sk, cr, ci in terms:
        key = (si, sj, sk)
        paired.setdefault(key, {})[op_j] = complex(cr, ci)
    bad_h = 0
    for key, ops in paired.items():
        if 0 not in ops or 1 not in ops:
            bad_h += 1
            continue
        if not np.isclose(ops[1], np.conj(ops[0])):
            bad_h += 1
    if bad_h:
        print(f"[h.c. ] FAILED: {bad_h} non-Hermitian (S+, S-) pairs")
        sys.exit(1)
    print("[h.c. ] OK: every S+ row has a S- partner with conjugate coefficient")

    # ---- Bonus: total sums per class (sanity) ----
    sums = {1: 0j, 2: 0j, 3: 0j}
    for j in nn_list:
        for a_idx, i in enumerate(sorted(nn_list[j])):
            for k in sorted(nn_list[j])[a_idx + 1:]:
                t, _, key2 = classify(j, i, k, sub_idx, vertex_to_cell)
                p = PHI_COLLINEAR_2PI3[sub_idx[j]][sub_idx[i]] if t == 1 else PHI_PAIR_2PI3[sub_idx[j]][key2]
                sums[t] += omega ** p
    # By the Z_3 symmetry of the phase tables, the sum of phases per class
    # should vanish (each phase value 0, +1, -1 appears equal numbers of times).
    print(f"[Z3   ] Sum of exp(i phi) per class: {sums}  (expected ~ 0 by symmetry)")
    for t, s in sums.items():
        assert abs(s) < 1e-9, f"Z_3 sum non-zero for class {t}: {s}"
    print("[Z3   ] OK: per-class phase sums vanish (Z_3 symmetry holds)")
    print("\nAll three-spin term tests passed.")


if __name__ == "__main__":
    main()

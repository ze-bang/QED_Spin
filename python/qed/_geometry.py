"""Cluster geometry: the momenta a periodic cluster allows, and the conventions that tie them to the
engine's momentum labels.

Convention (the whole package): a translation T moves every spin by its displacement d_T (for the
engine's permutation p, site i of the image carries the spin of site p[i], so d_T = r_i - r_{p[i]}).
A state of crystal momentum q has U_T|psi> = exp(-i q.d_T)|psi>, i.e. the label theta_T = q.d_T / 2 pi
(mod 1) that ``result.momentum(i, [T])`` reports. A Fourier component O_q = N^-1/2 sum_r e^{-i q.r} O_r
satisfies U_T O_q U_T^dag = e^{+i q.d_T} O_q: it maps momentum sector k to k - q.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Optional, Sequence

import numpy as np

from .errors import InvalidRequest

_TOL = 1e-8


def _rows(vectors) -> np.ndarray:
    """The nonzero rows of a list of 3-vectors, as a (d, 3) array."""
    V = np.asarray(vectors, float).reshape(-1, 3)
    return V[np.linalg.norm(V, axis=1) > _TOL]


def _dual(D: np.ndarray) -> np.ndarray:
    """Rows g_m in the span of the rows d_m of D with g_m . d_n = 2 pi delta_mn."""
    return 2.0 * np.pi * np.linalg.solve(D @ D.T, D)


def _hnf_basis(relations: list) -> np.ndarray:
    """A basis (rows) of the integer lattice spanned by `relations` (vectors in Z^d), by integer row
    reduction (d <= 3)."""
    R = [list(map(int, r)) for r in relations if any(r)]
    if not R:
        raise InvalidRequest("cluster_momenta: the translations generate an infinite group (no relation)")
    d = len(R[0])
    basis = []
    for col in range(d):
        piv = [r for r in R if r[col] != 0]
        R = [r for r in R if r[col] == 0]
        while piv:
            # Euclid on this column: reduce every row by the one with the smallest entry; rows
            # that reach 0 here move on to the next columns.
            piv.sort(key=lambda r: abs(r[col]))
            p, rest = piv[0], []
            for r in piv[1:]:
                q = r[col] // p[col]
                r2 = [a - q * b for a, b in zip(r, p)]
                (rest if r2[col] != 0 else R).append(r2)
            if not rest:
                basis.append(p)
                break
            piv = [p] + rest
        R = [r for r in R if any(r)]
    if len(basis) != d:
        raise InvalidRequest("cluster_momenta: the translations do not close in every direction they span")
    return np.array(basis, float)


def displacement(perm: Sequence[int], positions) -> np.ndarray:
    """The displacement d_T a translation moves the spins by: the most common r_i - r_{p[i]} over the
    sites (a site whose image wraps around the cluster differs from it by a supercell vector)."""
    r = np.asarray(positions, float).reshape(-1, 3)
    p = np.asarray(perm, int)
    if len(p) != len(r):
        raise InvalidRequest(f"displacement: a permutation of {len(p)} sites for {len(r)} positions")
    votes = Counter(tuple(np.round(r[i] - r[p[i]], 6)) for i in range(len(r)))
    return np.array(votes.most_common(1)[0][0], float)


def cluster_momenta(
    lattice=None,
    *,
    supercell=None,
    primitive=None,
    translations: Optional[Sequence[Sequence[int]]] = None,
    positions=None,
    fold: str = "ws",
) -> np.ndarray:
    """The momenta q (an (n_cells, 3) array, Cartesian, in the units of the positions) a periodic
    cluster allows: q . s in 2 pi Z for every supercell vector s, one per class modulo the primitive
    reciprocal lattice -- one per momentum sector of the translation group.

    Give either a ``qed.input.Lattice`` from a periodic generator (it carries ``supercell`` and
    ``lattice_vectors``), ``supercell`` and ``primitive`` vectors directly, or ``translations``
    (generator permutations, the engine's convention) with ``positions``: the displacements of the
    generators span the primitive lattice and the group's relations give the supercell. The positions
    must be those of a compact cluster: a translation's displacement is the one most sites show.
    ``fold="ws"`` returns each q at its shortest image (the Wigner-Seitz cell), ``"cell"`` in the
    parallelepiped of the primitive reciprocal vectors."""
    if fold not in ("ws", "cell"):
        raise InvalidRequest(f"cluster_momenta: fold must be 'ws' or 'cell', got {fold!r}")
    if translations is not None:
        if positions is None:
            if lattice is None:
                raise InvalidRequest("cluster_momenta: translations need positions (or a lattice)")
            positions = lattice.positions
        gens = [list(map(int, t)) for t in translations]
        D = np.array([displacement(t, positions) for t in gens])
        keep = np.linalg.norm(D, axis=1) > _TOL
        if not keep.all():
            raise InvalidRequest("cluster_momenta: a translation that moves no spin")
        # Integer relations: products of generators that act as the identity.
        n = len(gens[0])
        ident = tuple(range(n))
        seen = {ident: (0,) * len(gens)}
        frontier = [ident]
        relations = []
        while frontier:
            nxt = []
            for g in frontier:
                for m, t in enumerate(gens):
                    h = tuple(g[t[i]] for i in range(n))
                    coord = tuple(c + (1 if k == m else 0) for k, c in enumerate(seen[g]))
                    if h in seen:
                        rel = tuple(a - b for a, b in zip(coord, seen[h]))
                        if any(rel):
                            relations.append(rel)
                    else:
                        seen[h] = coord
                        nxt.append(h)
            frontier = nxt
        Rb = _hnf_basis(relations)
        L = Rb @ D            # supercell vectors in Cartesian coordinates
        A = D                 # the primitive lattice: the generators' displacements
        # Generators may be dependent (e.g. T1, T2, T1 T2): reduce A to a basis of its span.
        if np.linalg.matrix_rank(A, tol=1e-6) < len(A):
            raise InvalidRequest("cluster_momenta: pass independent translation generators")
    else:
        if lattice is not None:
            supercell = lattice.supercell if supercell is None else supercell
            primitive = lattice.lattice_vectors if primitive is None else primitive
        if supercell is None or primitive is None:
            raise InvalidRequest(
                "cluster_momenta: give a periodic lattice, supercell and primitive vectors, or translations"
            )
        L = _rows(supercell)
        A = _rows(primitive)[: len(L)]
        if len(L) == 0:
            raise InvalidRequest("cluster_momenta: the lattice has no supercell (open boundaries, or built from "
                                 "an adjacency list without one)")
    d = len(L)
    if len(A) != d:
        raise InvalidRequest("cluster_momenta: as many primitive vectors as supercell vectors are needed")
    M = L @ np.linalg.pinv(A)                      # the supercell in primitive coordinates
    if not np.allclose(M, np.round(M), atol=1e-6):
        raise InvalidRequest(
            "cluster_momenta: the supercell vectors are not integer combinations of the primitive ones"
        )
    M = np.round(M)
    n_cells = int(round(abs(np.linalg.det(M))))
    G = _dual(A)                                   # primitive reciprocal vectors
    Minv_T = np.linalg.inv(M).T
    # q = n . (dual of the supercell) for integer n; in primitive-reciprocal coordinates x = n M^-T,
    # one class per x mod 1 (n in [0, n_cells)^d reaches every class: n_cells times a class is 0).
    xs = set()
    for nvec in itertools.product(range(n_cells), repeat=d):
        x = np.mod(np.asarray(nvec, float) @ Minv_T, 1.0)
        xs.add(tuple(np.round(x, 9) % 1.0))
    if len(xs) != n_cells:
        raise InvalidRequest(f"cluster_momenta: found {len(xs)} momenta for {n_cells} cells")
    X = np.array(sorted(xs))
    Q = X @ G
    if fold == "ws":
        shifts = np.array(list(itertools.product((-1, 0, 1), repeat=d)), float) @ G

        # the shortest image; ties go to the image with the larger components (a fixed choice)
        def shortest(q):
            return min((q + s for s in shifts), key=lambda v: (round(float(v @ v), 9), tuple(np.round(-v, 9))))

        Q = np.array([shortest(q) for q in Q])
    return Q


def momentum_label(q, displacement_vector) -> float:
    """theta_T = q . d_T / 2 pi mod 1: the label ``result.momentum(i, [T])`` reports for a state of
    momentum q, T the translation with displacement d_T."""
    return float(np.mod(np.dot(np.asarray(q, float), np.asarray(displacement_vector, float)) / (2.0 * np.pi), 1.0))

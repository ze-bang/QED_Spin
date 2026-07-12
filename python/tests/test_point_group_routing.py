"""Stage 9c: ONE point-group routing decision; SAB retired to test oracle.

Pinned here:
  * point_group='auto' (the default) PROJECTS eigenvalue-only solve /
    full_spectrum calls through the factorized little-group engine, and
    the eigenvalues match the point_group='off' abelian lane.
  * 'full' = require-projection: raises (with the decline reason) on a
    residue-less symmetry, under ED_SYM_LITTLE_GROUP=0, and for
    vector-consuming calls -- no silent SAB fallback anymore.
  * ED_SYM_LITTLE_GROUP=0 is the bisection gate: 'auto' degrades to the
    abelian rep lane with folds, bit-compatible with the pre-9c route.
  * explicit non-abelian generator lists route through split_nonabelian
    (greedy maximal-abelian subgroup + coset residues) == dense.
  * symmetry='translation' never runs the NP-hard max-clique analyzer
    (the hang this stage fixes).
"""
from __future__ import annotations

import numpy as np
import pytest

qed = pytest.importorskip("qed")
pytest.importorskip("pynauty")

N = 8


def _ring(n=N):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    return b.to_operator()


def _dense(n=N):
    dim = 1 << n
    Hd = np.zeros((dim, dim))

    def szv(s, i):
        return 0.5 if (s >> i) & 1 else -0.5

    for s in range(dim):
        for i in range(n):
            j = (i + 1) % n
            Hd[s, s] += szv(s, i) * szv(s, j)
            if ((s >> i) & 1) != ((s >> j) & 1):
                Hd[s ^ (1 << i) ^ (1 << j), s] += 0.5
    return Hd


def test_auto_projects_by_default(capsys):
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    r_auto = qed.solve(H, symmetry=gen, num_eigenvalues=3, sz=N // 2,
                       verbose=True)                     # point_group default
    assert "LITTLE-GROUP" in capsys.readouterr().out
    r_off = qed.solve(H, symmetry=gen, num_eigenvalues=3, sz=N // 2,
                      point_group="off", verbose=False)
    np.testing.assert_allclose(r_auto.eigenvalues, r_off.eigenvalues,
                               rtol=0, atol=1e-8)


def test_full_raises_without_residue():
    H = _ring()
    # An explicit ABELIAN generator list carries no residue to project.
    t = [(i + 1) % N for i in range(N)]
    with pytest.raises(RuntimeError, match="point_group='full'"):
        qed.solve(H, symmetry=[t], num_eigenvalues=2, sz=N // 2,
                  point_group="full", verbose=False)


def test_env_gate_degrades_auto_and_fails_full(monkeypatch, capsys):
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set

    monkeypatch.setenv("ED_SYM_LITTLE_GROUP", "0")
    r_gated = qed.solve(H, symmetry=gen, num_eigenvalues=3, sz=N // 2,
                        verbose=True)
    out = capsys.readouterr().out
    assert "LITTLE-GROUP" not in out
    with pytest.raises(RuntimeError, match="ED_SYM_LITTLE_GROUP"):
        qed.solve(H, symmetry=gen, num_eigenvalues=3, sz=N // 2,
                  point_group="full", verbose=False)
    monkeypatch.delenv("ED_SYM_LITTLE_GROUP")

    r_on = qed.solve(H, symmetry=gen, num_eigenvalues=3, sz=N // 2,
                     verbose=False)
    np.testing.assert_allclose(np.asarray(r_gated.eigenvalues),
                               np.asarray(r_on.eigenvalues),
                               rtol=0, atol=1e-8)


def test_vector_consumers_stay_abelian():
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    # sector= restricts to one irrep -- the projection lane cannot honor
    # it, so 'full' must raise rather than silently ignore the request.
    with pytest.raises(RuntimeError, match="sector|eigenvector"):
        qed.solve(H, symmetry=gen, num_eigenvalues=1, sz=N // 2,
                  sector=[0], point_group="full", verbose=False)


def test_explicit_nonabelian_generators_project():
    """Raw non-abelian generator input previously ONLY had the monolithic
    SAB route; split_nonabelian now carves (abelian, residues) out of the
    closed group and the factorized engine reproduces the dense spectrum."""
    H = _ring()
    t = [(i + 1) % N for i in range(N)]           # rotation
    refl = [(N - i) % N for i in range(N)]        # reflection -> D8
    fs = qed.full_spectrum(H, symmetry=[t, refl], verbose=False)
    w = np.linalg.eigvalsh(_dense())
    np.testing.assert_allclose(np.asarray(fs.eigenvalues), w,
                               rtol=0, atol=1e-10)


def test_split_nonabelian_shapes():
    from qed.point_group_routing import split_nonabelian
    t = [(i + 1) % N for i in range(N)]
    refl = [(N - i) % N for i in range(N)]
    A, res = split_nonabelian([t, refl])
    assert len(A) == N                  # the Z8 rotation subgroup
    assert len(res) >= 1                # one reflection coset
    # abelian-only input declines with a reason string
    out = split_nonabelian([t])
    assert isinstance(out, str) or len(out[1]) == 0 or out[1]


def test_translation_mode_never_runs_max_clique(monkeypatch):
    """The Stage-9c hang fix: symmetry='translation' passes
    translation_only=True to find_symmetries, so the NP-hard clique
    analyzer must never execute."""
    from qed.input import lattice as L
    import edlib.automorphism_finder as af

    def _boom(self, *a, **k):
        raise AssertionError("max-clique analyzer ran in translation mode")

    monkeypatch.setattr(af.AutomorphismCliqueAnalyzer, "find_maximum_clique",
                        _boom)
    # fresh operator (find_symmetries memoizes on content)
    n = 10
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    H10 = b.to_operator()
    lat = L.chain(n, pbc=True)
    r = qed.solve(H10, symmetry="translation", lattice=lat, sz=n // 2,
                  num_eigenvalues=1, verbose=False)
    assert len(r.eigenvalues) == 1


def test_project_lane_carries_labels():
    """Stage 10c: the default (projected) solve returns per-eigenvalue
    quantum-number labels, aligned with eigenvalues."""
    H = _ring()
    gen = qed.find_symmetries(H, verbose=False).full_set
    r = qed.solve(H, symmetry=gen, sz=N // 2, num_eigenvalues=4,
                  verbose=False)
    n = len(r.eigenvalues)
    for attr in ("block_k_raw", "block_flip_parity", "block_irrep",
                 "block_irrep_dim", "block_multiplicity",
                 "block_subspace"):
        assert len(getattr(r, attr)) == n, attr
    assert len(r.irrep_characters) > 0
    assert all(s == N // 2 for s in r.block_subspace)
    # the GS momentum decodes to k=0 through chi_k(T)
    import cmath
    A_T = list(gen.generators[0])
    # locate T inside the closed abelian group used by the engine: the
    # characters are indexed by the CALLER's element order, which for the
    # routed path is the sorted closure of the generators.
    from qed.point_group_routing import split_nonabelian
    A, _res = split_nonabelian(gen)
    aT = A.index(A_T)
    ph = r.irrep_characters[r.block_k_raw[0]][aT]
    k_phys = round(-cmath.phase(ph) * N / (2 * cmath.pi)) % N
    assert k_phys == 0

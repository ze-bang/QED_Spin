#!/usr/bin/env python3
"""examples/15_python_unified_interface.py

End-to-end walkthrough of the unified Python interface introduced by the
May-2026 surface unification: ONE namespace, THREE verbs.

This is the Python mirror of ``examples/00_unified_interface.cpp``. The
canonical Python shape is plain kwargs::

    H = ...                                               # build an Operator
    res = qed.solve(H, num_eigenvalues=1, solver='LANCZOS')
    # res.eigenvalues, res.thermo_data, ...

Three workflow verbs exist:

* :func:`qed.solve(H, **kwargs) <qed.solve>`               -- ground state
* :func:`qed.thermal(H, **kwargs) <qed.thermal>`           -- finite-T
* :func:`qed.spectral(H, observables, **kwargs) <qed.spectral>`
                                                            -- dynamical

Each one accepts a ``qed.Operator`` (or ``qed.FixedSzOperator``) plus
plain keyword arguments. Backend (CPU / GPU / MPI / MPI+GPU) is
auto-selected via ``select_backend`` inside the C++ orchestrator.

Use cases covered below:

  [1] ``qed.solve``    Lanczos ground state with auto-Sz projection
  [2] ``qed.solve``    on a pre-built ``FixedSzOperator`` (Sz=N/2 sector)
  [3] ``qed.solve``    across every method (LANCZOS / FULL /
                       KRYLOV_SCHUR / BLOCK_LANCZOS)
  [4] ``qed.solve``    with ``compute_eigenvectors=True``
  [5] ``qed.thermal``  finite-T thermodynamics via FTLM
  [6] ``qed.spectral`` continued-fraction S(omega) on a custom observable
  [7] ``device=cpu``   pinning the backend lane

Run::

    python3 examples/15_python_unified_interface.py
    python3 examples/15_python_unified_interface.py 8   # N=8 ring
"""
from __future__ import annotations

import math
import sys

import numpy as np

import qed


def build_heisenberg_chain(n_sites: int, j_coupling: float = 1.0) -> "qed.Operator":
    """Programmatic periodic Heisenberg chain via the fluent builder."""
    bonds = [(i, (i + 1) % n_sites) for i in range(n_sites)]
    return (qed.input.HamiltonianBuilder(n_sites)
                     .heisenberg(bonds, J=j_coupling)
                     .to_operator())


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print("=" * 60)
    print(f"  Unified Python interface: end-to-end walkthrough  (N={N})")
    print("=" * 60)
    print()

    H = build_heisenberg_chain(N)

    # ------------------------------------------------------------------
    # [1] Ground-state Lanczos with auto-Sz on by default.
    # ------------------------------------------------------------------
    print("[1] qed.solve  --  Lanczos ground state (auto-Sz on)")
    res = qed.solve(H,
                    num_eigenvalues=2,
                    solver="LANCZOS",
                    tolerance=1e-12,
                    max_iterations=200,
                    verbose=False, plan=False)
    print(f"    E0 = {res.eigenvalues[0]:.10f}")
    print(f"    E1 = {res.eigenvalues[1]:.10f}")
    print()

    # ------------------------------------------------------------------
    # [2] Pre-projected FixedSzOperator at half-filling. Equivalent to
    #     [1] but constructs the FixedSzOperator explicitly first.
    # ------------------------------------------------------------------
    print(f"[2] qed.solve  --  pre-projected FixedSzOperator(n_up={N//2})")
    H_sz = H.make_fixed_sz(N // 2)
    res2 = qed.solve(H_sz, num_eigenvalues=1, solver="LANCZOS",
                     tolerance=1e-12, verbose=False, plan=False)
    print(f"    dim   = {H_sz.dimension}  (full would be {2**N})")
    print(f"    E0    = {res2.eigenvalues[0]:.10f}")
    print()

    # ------------------------------------------------------------------
    # [3] Method comparison. With auto_sz=False to keep the full
    #     Hilbert space; works for small N only.
    # ------------------------------------------------------------------
    print("[3] qed.solve  --  method comparison (LANCZOS / FULL / KRYLOV_SCHUR)")
    for m in ("LANCZOS", "FULL", "KRYLOV_SCHUR"):
        try:
            r = qed.solve(H, num_eigenvalues=1, solver=m,
                          tolerance=1e-10, verbose=False, plan=False)
            print(f"    {m:14s} E0 = {r.eigenvalues[0]:.10f}")
        except Exception as exc:  # pragma: no cover - skip unavailable
            print(f"    {m:14s} skipped: {exc}")
    print()

    # ------------------------------------------------------------------
    # [4] compute_eigenvectors=True. Eigenvectors land in
    #     ``result.eigenvectors_path`` (HDF5) or directly on the
    #     ``EigenvectorRef`` if the orchestrator path is taken.
    # ------------------------------------------------------------------
    print("[4] qed.solve  --  compute_eigenvectors=True")
    res4 = qed.solve(H,
                     num_eigenvalues=1,
                     solver="LANCZOS",
                     tolerance=1e-10,
                     compute_eigenvectors=True,
                     output_dir="",  # in-memory only; no HDF5 sink
                     verbose=False, plan=False)
    print(f"    E0 = {res4.eigenvalues[0]:.10f}")
    print(f"    eigenvectors_computed = {res4.eigenvectors_computed}")
    print()

    # ------------------------------------------------------------------
    # [5] qed.thermal  FTLM in a small temperature window.
    # ------------------------------------------------------------------
    print("[5] qed.thermal  --  FTLM finite-T thermodynamics")
    try:
        tr = qed.thermal(H,
                         method="FTLM",
                         T_min=0.1, T_max=2.0, num_T=8,
                         num_samples=4, krylov_dim=30,
                         tolerance=1e-8,
                         verbose=False)
        print(f"    T points         = {len(tr.temperatures)}")
        print(f"    ground state E0  = {tr.ground_state_energy:.10f}")
        print(f"    used_sz_decomp   = {tr.used_sz_decomposition}")
    except Exception as exc:  # pragma: no cover
        print(f"    failed: {exc}")
    print()

    # ------------------------------------------------------------------
    # [6] qed.spectral  continued-fraction S(omega). We use SZ_0
    #     (single-site Sz at site 0) as a toy observable.
    # ------------------------------------------------------------------
    print("[6] qed.spectral  --  ground-state continued-fraction S(omega)")
    Sz0 = qed.Operator(N, 0.5)
    Sz0.add_one_body(qed.OP_SZ, 0, 1.0)
    try:
        sp = qed.spectral(H, [Sz0],
                          method="ground_state_cf",
                          omega=np.linspace(-4.0, 4.0, 32),
                          eta=0.1,
                          krylov_dim=80,
                          verbose=False)
        print(f"    omega points = {len(sp.omega)}")
        print(f"    omega range  = [{sp.omega[0]:.3f}, {sp.omega[-1]:.3f}]")
        print(f"    S_real[0]    = {sp.S_real[0]:.6f}")
    except Exception as exc:  # pragma: no cover
        print(f"    failed: {exc}")
    print()

    # ------------------------------------------------------------------
    # [7] Pin the backend to CPU explicitly.
    # ------------------------------------------------------------------
    print("[7] qed.solve  --  device='cpu' pin")
    res7 = qed.solve(H,
                     num_eigenvalues=1,
                     solver="LANCZOS",
                     device="cpu",
                     tolerance=1e-12,
                     verbose=False, plan=False)
    print(f"    E0 (CPU lane) = {res7.eigenvalues[0]:.10f}")
    print()

    print("All blocks complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

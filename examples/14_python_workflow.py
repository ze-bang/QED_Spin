#!/usr/bin/env python3
"""examples/14_python_workflow.py

End-to-end demo of the Phase-9 stress-free workflow:

    1. Build a Hamiltonian via the fluent C++-backed builder.
    2. Discover symmetries (and U(1) Sz status) on the in-memory operator.
    3. Diagonalise four ways with the unified ``qed.solve`` entry point:
         a) full Hilbert space, default everything
         b) projected onto the Sz=N/2 sector
         c) projected onto the largest commuting automorphism subgroup
         d) projected onto BOTH the symmetry group and the Sz=N/2 sector
    4. Pick a SUBGROUP of the discovered ``full_set`` and diagonalise on
       that smaller projection (cheaper sectors, fewer of them).
    5. Show how to discover non-default simulation parameters via
       ``qed.list_diag_parameters`` and pass them through ``extra_params``.

All four paths must agree on the ground-state energy. For the periodic
6-site spin-1/2 Heisenberg ring with J=1 the Bethe-ansatz reference is
``E0 = -2.802775637731995``.

Run::

    python3 examples/14_python_workflow.py
    python3 examples/14_python_workflow.py 8        # N=8 ring
"""
from __future__ import annotations

import sys

import qed as qed


def main() -> int:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    # ------------------------------------------------------------------
    # 1. Build the Hamiltonian.
    # ------------------------------------------------------------------
    bonds = [(i, (i + 1) % N) for i in range(N)]
    H = (qed.input.HamiltonianBuilder(N)
                  .heisenberg(bonds, J=1.0)
                  .to_operator())
    print(f"Built periodic Heisenberg ring  N={N}  dim={H.dimension}\n")

    # ------------------------------------------------------------------
    # 2. Discover symmetries (also reports U(1) Sz status + sectors).
    # ------------------------------------------------------------------
    report = qed.find_symmetries(H, verbose=False)
    print(report.summary())
    print()

    # ------------------------------------------------------------------
    # 3. Four diagonalisations, smart-defaulted.
    # ------------------------------------------------------------------
    print("--- (a) full Hilbert space ---")
    res_full = qed.solve(H, num_eigenvalues=4, verbose=False)
    print(f"  E0 = {res_full.eigenvalues[0]:.10f}")

    if report.has_u1_sz:
        print(f"\n--- (b) fixed Sz sector, n_up={N // 2} ---")
        res_sz = qed.solve(H, num_eigenvalues=4, sz=N // 2, verbose=False)
        print(f"  E0 = {res_sz.eigenvalues[0]:.10f}")

    if report.full_set is not None:
        print(f"\n--- (c) symmetry projection ({report.full_set.name}, "
              f"|G|={report.full_set.group_size}) ---")
        res_sym = qed.solve(H, num_eigenvalues=4,
                           symmetry=report.full_set, verbose=False)
        print(f"  E0 = {res_sym.eigenvalues[0]:.10f}")

        if report.has_u1_sz:
            print(f"\n--- (d) symmetry + fixed Sz, n_up={N // 2} ---")
            res_both = qed.solve(H, num_eigenvalues=4,
                                symmetry=report.full_set,
                                sz=N // 2, verbose=False)
            print(f"  E0 = {res_both.eigenvalues[0]:.10f}")

        # -------------------------------------------------------------
        # 4. Pick a subset of the discovered generators.
        # -------------------------------------------------------------
        if len(report.full_set.generators) > 1:
            print("\n--- (e) one-generator subgroup of full_set ---")
            sub = report.full_set[0]
            print(f"  picked: {sub}  (description: {sub.description})")
            res_sub = qed.solve(H, num_eigenvalues=4,
                               symmetry=sub, verbose=False)
            print(f"  E0 = {res_sub.eigenvalues[0]:.10f}")
            print("  -> equivalent: report.full_set.subgroup([0])")
            print("  -> equivalent: report.get('full_automorphism[0]')")

    # ------------------------------------------------------------------
    # 5. Discover and override niche simulation parameters.
    # ------------------------------------------------------------------
    print("\n--- (f) overriding ARPACK parameters via extra_params ---")
    print("  Run qed.list_diag_parameters('arpack') to see every ARPACK")
    print("  knob; here we just override a couple for the demo.")
    res_arp = qed.solve(
        H, num_eigenvalues=2,
        solver="ARPACK_ADVANCED", device="cpu",
        verbose=False,
        extra_params={
            "arpack_which": "SA",      # smallest algebraic eigenvalue
            "arpack_max_restarts": 5,
        },
    )
    print(f"  E0 = {sorted(res_arp.eigenvalues)[0]:.10f}")

    # ------------------------------------------------------------------
    # 6. Thermal trajectory via mTPQ (case-insensitive solver name).
    #     The same qed.solve(...) entry point handles thermal solvers;
    #     just switch the solver kwarg. mTPQ writes raw imaginary-time
    #     trajectories to output_dir and post-processes to a unified
    #     thermodynamic curve.
    # ------------------------------------------------------------------
    import tempfile
    print("\n--- (g) thermal trajectory via mTPQ ---")
    print("  Same qed.solve() call; case-insensitive 'mtpq' resolves to")
    print("  DiagonalizationMethod.mTPQ. Trajectories land in output_dir.")
    with tempfile.TemporaryDirectory(prefix="qed_demo_thermal_") as tmp:
        res_thermal = qed.solve(
            H,
            solver="mtpq",
            sz=N // 2 if report.has_u1_sz else None,
            num_samples=2,
            target_beta=10.0,
            num_temp_points=20,
            output_dir=tmp,
            verbose=False,
        )
        n_traj = len(list(res_thermal.eigenvalues))
        artefacts = sorted(__import__("os").listdir(tmp))
        print(f"  trajectory length = {n_traj}")
        print(f"  artefacts written = {artefacts}")

    print("\n--- (h) device introspection ---")
    print("  qed.solver_device_support() answers 'can THIS build run")
    print("  solver X on device Y?' (cpu / gpu / mpi / mpi_gpu).")
    print()
    qed.solver_device_support(solver="lanczos")
    print()
    print("  For the matrix-vector calls themselves:")
    print("    qed.solve(H, device='cpu')      -- in-process CPU")
    print("    qed.solve(H, device='gpu')      -- in-process single GPU")
    print("                                       (auto temp-dir routing)")
    print("    qed.mpi.run_distributed('lanczos', n_ranks=N, ...)")
    print("                                   -- multi-rank CPU MPI")
    print("    qed.mpi.run_distributed('lanczos', n_ranks=N,")
    print("                            use_gpu=True, ...)")
    print("                                   -- multi-GPU + NCCL")
    print("    qed.mpi.run_distributed('tpq', n_ranks=N, ...)")
    print("                                   -- distributed canonical TPQ")

    print("\n--- (i) pre-flight planner: 'will this fit on this host?' ---")
    print("  qed.solve always asks the planner before dispatching; here")
    print("  we call it directly so we can show the verdict.")
    rep = qed.estimate_resources(H, solver="LANCZOS", device="cpu",
                                  num_eigenvalues=2)
    for line in rep.summary().splitlines():
        print("  " + line)

    print()
    print("  For a much larger problem the planner refuses the dispatch")
    print("  with ranked, copy-pasteable suggestions. dry_run=True prints")
    print("  the verdict and exits without running the kernel.")
    big = (qed.input.HamiltonianBuilder(28)
              .heisenberg([(i, (i + 1) % 28) for i in range(28)], J=1.0)
              .to_operator())
    print()
    qed.solve(big, solver="FULL", dry_run=True, verbose=False)

    print()
    print("  Goal-oriented planner: rank candidate workflows for a goal")
    print("  ('ground_state' / 'low_lying' / 'thermal' / 'spectral').")
    sug = qed.suggest_workflow(H, intent="ground_state", num_eigenvalues=4)
    print()
    for line in sug.summary(top=4).splitlines():
        print("  " + line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

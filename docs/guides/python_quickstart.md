# Python quickstart (`qed`)

The Python facade lives under the `qed` package. It is a `pybind11`
layer over the same `ed_solvers_*` static libraries that drive the
`ED` CLI, so results are bit-identical. The public surface is three
verbs that mirror the C++ orchestrator:

* `qed.solve(H, **kw)` — ground state, eigenvalues, a few low-lying
  states.
* `qed.thermal(H, **kw)` — finite-temperature thermodynamics (FTLM,
  LTLM, mTPQ, cTPQ, KPM-DOS).
* `qed.spectral(H_or_dir, **kw)` — static and dynamical structure
  factors (`S(Q)`, `S(Q,T)`, `S(Q,ω)`, `S(Q,ω,T)`).

Hamiltonians are built via the `qed.input` lattice + Hamiltonian
builder (the in-process mirror of the C++ `ed::input` library), via
the legacy `Operator` / `FixedSzOperator` constructors with `add_*`
terms, or by pointing the three verbs at a directory of HPhi-format
files.

This page covers the everyday entry points. For the full advanced
catalogue (device pinning, MPI launcher, GPU per-sector dispatch,
in-process symmetry round-trip, every `EDParameters` knob via
`extra_params=`) jump to [`python_advanced.md`](python_advanced.md);
for the one-call reference see [`one_call_api.md`](one_call_api.md).

## Heisenberg chain in 10 lines

```python
import qed

N = 12
H = (qed.input.HamiltonianBuilder(N)
        .heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
        .to_operator())

res = qed.solve(H, num_eigenvalues=3)
print("E0 =", res.eigenvalues[0])
# E0 = -5.387390917445587   (Bethe ansatz at N=12, J=1)
```

The same three verbs cover finite-T thermodynamics and spectral
functions:

```python
# Finite-temperature thermodynamics via FTLM
thermo = qed.thermal(H, method="FTLM",
                     T_min=0.05, T_max=10.0, num_T=200,
                     num_samples=8)

# T = 0 dynamical S(Q, ω) on a Heisenberg directory
import numpy as np
spec = qed.spectral("runs/heisenberg12",
                    omega=np.linspace(-2, 6, 200), eta=0.05)
```

## What gets decided for you

Every kwarg is optional. Anything left unspecified is auto-tuned from
the operator size, the build flags (`qed.has_cuda_build()`,
`qed.has_mpi_build()`), and the requested `num_eigenvalues` / `T` /
`omega`. Typical decisions:

* **Solver** — `FULL` for tiny dim, `LANCZOS` for ≤ 5 eigs,
  `KRYLOV_SCHUR` for ≤ 20, `BLOCK_LANCZOS` for more.
* **Device** — `cpu` / `gpu` / `mpi` / `mpi_gpu`, picked by
  Hilbert / sector dim and the available backends.
* **Symmetry** — pass `sz=` to project to a fixed-Sz sector
  (`FixedSzSubspace`), pass `symmetry=` (a `GeneratorSet` or the dict
  from `qed.symmetry.group_from_generators`) to add a `SpatialProjector`
  to the chain. The two axes are orthogonal — see
  [Symmetries](#symmetries-orthogonal-composition) below.
* **Broadening** (`eta`), Krylov dim (`krylov_dim`), number of random
  vectors (`num_samples`) — auto-tuned by `qed.auto_tune.*`.

Override any of them via the explicit kwarg of the same name. There is no
pre-flight planner; instead, a memory guard checks the dominant allocation
against available RAM and raises a clean error (rather than OOM-killing the
host) if it won't fit. Bypass with `ED_MEM_GUARD_OFF=1`.

## Symmetries — orthogonal composition

`sz=` and `symmetry=` are the two orthogonal axes of the symmetry
sector decomposition introduced in May 2026
(see [`docs/architecture/SYMMETRY.md`](../architecture/SYMMETRY.md) §6):

```python
# Mode              Subspace            ProjectorChain
qed.solve(H)                            # (FullSpace, [])
qed.solve(H, sz=N//2)                   # (FixedSz,   [])
qed.solve(H, symmetry=gens)             # (FullSpace, [Spatial])
qed.solve(H, sz=N//2, symmetry=gens)    # (FixedSz,   [Spatial])
```

The same kwargs work on `qed.thermal` and `qed.spectral`. The
shortest path is `symmetry="auto"` -- the automorphism search runs
internally and the maximal commuting generator set is used:

```python
res = qed.solve(H, num_eigenvalues=4, sz=N // 2, symmetry="auto")
```

Two more discrete symmetries auto-compose on top: the global spin
flip (transport n_up -> N - n_up + the (k, +/-) projection at half
filling) and time reversal (solve k, copy the spectrum to -k). Each
has a four-state toggle -- `spin_flip=` / `time_reversal=` in
{"auto", "on", "off", "require"} -- where `"on"` REPORTS: it confirms
the detection, or warns and continues without the symmetry when your
Hamiltonian lacks it (a Zeeman field breaks the flip, complex
couplings break TR):

```python
res = qed.solve(H, sz=N // 2, symmetry="auto",
                spin_flip="on", time_reversal="on")
print(qed._core.detect_hamiltonian_symmetries(H))
# {'spin_flip': True, 'time_reversal': True, 'u1': True, 'sz_parity': True}
```

The remaining axes, each keyable:

```python
qed.solve(H, symmetry=gen, sz="even")              # Sz-parity half: the Z2
                                                   # remnant when S+S+ terms
                                                   # break U(1) (auto-detected)
qed.solve(H, symmetry=gen, point_group="auto")     # star folding (solve one
                                                   # momentum per point-group
                                                   # star, copy the spectrum)
qed.solve(H, symmetry=gen, point_group="full")     # TRUE non-abelian: d>=2
                                                   # irrep blocks ~ dim/|G|
qed.solve(H, symmetry="translation", lattice=lat)  # T projector + point
                                                   # group as stars
```

To inspect / pick groups explicitly:

```python
report = qed.find_symmetries(H, verbose=False)
print(report.summary())              # U(1) Sz status + generator candidates

res = qed.solve(H, num_eigenvalues=4,
                sz=N // 2,
                symmetry=report.full_set)
```

Knob-complete walkthroughs: [`examples/tour/`](../../examples/tour/).

## Hamiltonian builder DSL

`qed.input.HamiltonianBuilder` wraps the C++ `Operator` API in a
fluent QuSpin-style builder. Common Hamiltonians get one-liner
shortcuts (`heisenberg`, `transverse_field_ising`, `xxz`, `kitaev`,
`zeeman_per_site`, …); arbitrary terms always go through
`.add_two_body(op_i, site_i, op_j, site_j, coeff)`.

```python
import qed

lat = qed.input.lattice.chain(12, pbc=True)
H = (qed.input.HamiltonianBuilder(lat.num_sites)
        .heisenberg(lat.nn_pairs(), J=1.0)
        .zeeman_per_site([0.1] * lat.num_sites)
        .to_operator())

print("Ground-state energy:", qed.solve(H).eigenvalues[0])
```

The builder also writes the directory format the `ED` CLI consumes:

```python
(qed.input.HamiltonianBuilder(lat.num_sites)
        .heisenberg(lat.nn_pairs(), J=1.0)
        .write_directory("./chain12", lattice=lat))
```

## Spectral functions — `qed.spectral`

The `(T, omega)` tuple selects the kernel automatically:

| `T` given? | `omega` given? | Method picked         |
|------------|----------------|-----------------------|
| no         | no             | `single_expectation`  |
| no         | yes            | `ground_state_dssf`   |
| yes        | no             | `static_thermal`      |
| yes        | yes            | `dynamical_thermal`   |

```python
import numpy as np

# Static structure factor at one T:
qed.spectral("runs/heisenberg6", T=0.5)

# T = 0 dynamical S(Q, ω):
qed.spectral("runs/heisenberg6", omega=np.linspace(-2, 2, 200))

# Full S(Q, ω, T):
qed.spectral("runs/heisenberg6",
             T=[0.1, 0.3, 1.0],
             omega=np.linspace(-2, 2, 200))
```

Pass `symmetry=True` (or a dict with `observable=` /
`momentum_transfer=` / `delta_n_up=`) to engage the streaming-symmetry
spectral binding — see [`one_call_api.md`](one_call_api.md) §3 for the
full SOTA cross-irrep and finite-T cross-irrep recipes.

## MPI distributed runs

```python
if qed.has_mpi_build():
    qed.mpi.run_distributed("./chain24",
                            method="lanczos",
                            n_ranks=8)
```

The helper builds the right `mpiexec` argv for
`ed_distributed_main`. For GPU + MPI add `use_gpu=True` (uses NCCL).

## Next steps

* [`one_call_api.md`](one_call_api.md) — full reference for
  `qed.solve` / `qed.thermal` / `qed.spectral`.
* [`workflow.md`](workflow.md) — end-to-end recipes (symmetry
  discovery, half-filling sweep, mTPQ trajectory, memory guard).
* [`python_advanced.md`](python_advanced.md) — every advanced pattern
  (device pinning, in-process symmetry round-trip, every
  `EDParameters` knob).
* [`python_api_coverage.md`](python_api_coverage.md) — C++ / Python /
  CLI capability matrix.

# QED_Spin: exact diagonalization of spin-1/2 Hamiltonians

QED_Spin computes the lowest levels and eigenvectors, full spectra, thermodynamics and
dynamical correlations of spin-1/2 Hamiltonians with arbitrary terms: one-, two-, three-body
and longer products of spin operators. Every calculation is split into the symmetry sectors
the Hamiltonian has. The engine is C++17 with OpenMP and, optionally, CUDA on one NVIDIA GPU.
The Python package `qed` sits on top of it.

```python
import cmath, math
import numpy as np
import qed

N = 20
H = qed.input.HamiltonianBuilder(N).heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0).to_operator()

r = qed.eigs(H, 4)          # lowest 4 states; symmetries found automatically (spatial ones need pynauty)
print(r.energies)           # repeated by multiplicity
for L in r.levels:          # one entry per block eigenvalue
    print(L.energy, L.multiplicity, L.n_up)

th = qed.thermal(H, np.linspace(0.1, 4, 40), method="ftlm")   # E, C, entropy, F, lnZ, M, chi

# S^z at q = pi, built from single-site products
Sz_pi = sum((qed.Operator.product(N, "z", [j], cmath.exp(-1j * math.pi * j) / math.sqrt(N))
             for j in range(N)), qed.Operator(N))
w = np.linspace(0, 4, 400)
S0 = qed.dynamics(H, Sz_pi, w, eta=0.05)            # T = 0: continued fraction
S1 = qed.dynamics(H, Sz_pi, w, eta=0.05, T=[0.5])   # T > 0: finite-temperature Lanczos
```

## What it computes

| Call | Result |
|---|---|
| `qed.eigs(H, k=1, *, sym=None, vectors=False, dense_max_dim=None, allow_partial=False, device="cpu", prune=True, window=0.0)` | `EigResult`. `energies` holds the lowest `k` energies, each repeated by its multiplicity (with `window > 0`, also every block's lowest level within `window` above the k-th). `levels` has one entry per block eigenvalue with its quantum numbers. Methods: `vectors(basis="full", n_up=None)`, `expect(ops)` and `matrix_element(O, i, j)` (these need `vectors=True`); `momentum(i, translations)`, `irrep_characters(i)`, `save(path)` |
| `qed.load_eigs(path)` | the `EigResult` that `save` wrote to an `.npz` file. Vectors, `expect` and `matrix_element` work without H. A file of format 1 (written by qed < 0.6, when a set bit meant spin down) is refused |
| `qed.spectrum(H, *, sym=None, device="cpu")` | `SpectrumResult`: every eigenvalue, from a dense diagonalisation of each block |
| `qed.thermal(H, T, *, method="ftlm", sym=None, samples=40, krylov=None, steps=None, exact_states=0, seed=0, device="cpu", observables=None, dense_max_dim=None)` | `ThermalResult`: `E`, `C`, `entropy`, `F` and `lnZ` per temperature. `M` and `chi` are filled when H conserves Sz and `sym` decomposes by it. `O` holds ⟨O⟩(T) for each of `observables` |
| `qed.dynamics(H, O, omega, B=None, *, eta=0.05, T=None, sym=None, krylov=200, samples=40, seed=0, degeneracy_tol=1e-8, device="cpu", dense_max_dim=None, prune=True)` | `DynamicsResult`: $S_{AB}(\omega)$ at T = 0 or at the temperatures `T` |
| `qed.expect(H, ops, k=1, *, sym=None, device="cpu", **eigs_kwargs)` | `ExpectResult`: ⟨O⟩ in each of the lowest levels, averaged over the level's symmetry multiplet |

- **`thermal` methods.**
  - `"exact"` takes every block's full spectrum.
  - `"ftlm"` is finite-temperature Lanczos with Lanczos depth `krylov` (default 100). With
    `exact_states > 0` (OFTLM), that many lowest states of each block are handled exactly, as
    certified eigenpairs that make up whole degenerate levels.
  - `"mtpq"` is microcanonical thermal pure quantum states with the canonical estimator.
    `steps` defaults to enough for the coldest `T`.

  The sampled methods draw `samples` random vectors per block. `seed=0` draws a seed. Blocks
  up to `dense_max_dim` states (default 512; 0 always samples) are diagonalised exactly.
  `observables` works with `"exact"` and `"ftlm"`, and only with `exact_states` = 0 (`"mtpq"`,
  or `exact_states > 0` with any method, raises `InvalidRequest`). Only `"ftlm"` reads
  `exact_states`.
- **`dynamics`.** It computes
  $S_{AB}(\omega) = \sum_m p_m \langle m|A^\dagger \delta(\omega - H + E_m) B|m\rangle$,
  broadened by a Lorentzian of width `eta`.
  - `O` is the probe A, or a list of probes.
  - `B=None` gives each probe's autocorrelation, which is real.
  - A `qed.Operator` for `B` gives every ⟨O_i† B⟩. A list as long as `O` gives the pairs
    ⟨O_i† B_i⟩. `"all"` gives the full matrix ⟨O_i† O_j⟩.
  - `S` is indexed as `S[..., i, :]`. The probe axes come first: `[len(O)]`, or
    `[len(O), len(O)]` for `B="all"`, and none for a single `O`. Then comes one row per
    temperature (one row at T = 0), then ω.
  - With `T=None`, the result averages over the degenerate ground manifold: every level
    within `degeneracy_tol` · s_H of E0, where s_H is the sum of |c| over H's terms. ω is
    measured from E0.
  - With `T=[...]`, `dynamics` runs finite-temperature Lanczos with `samples` vectors per
    source sector, and ω is the transferred energy.
  - `sym.select(sz=..., momentum=...)` restricts the source states. Point-group selections
    (`k0`, `irrep`, `irrep_character`) raise `qed.errors.Unsupported`.
- **Operators may break every symmetry of H**, and may change Sz.
  - Observables (`expect`, thermal `observables`) enter each block averaged over the
    symmetries the block resolves, which leaves traces and multiplet averages unchanged.
  - Probes (`dynamics`, `matrix_element`) connect the sectors they map between.
- **What the results report.** The results of `eigs`, `spectrum`, `thermal` and `dynamics`
  report `placement`: how many solves ran as Krylov or dense solves on the device or the
  host (`device_krylov`, `device_dense`, `host_krylov`, `host_dense`). Every result carries
  `diagnostics`, a list of `(code, message)` pairs for fallbacks the run took.
  `EigResult.block_stats` has one dict per solved block:
  - its labels, dimension and `kind` (`group` or `plain`);
  - the lane that applied H: `dense`, `csr`, `csr-real`, `walk`, `gpu-gather`,
    `device-csr` or `device-gather`;
  - phase timings, `nnz`, `csr_bytes` and the number of applies.
- **Levels** (`qed._core.sectors.Level`) carry these labels:
  - `energy`, `multiplicity`, `n_up` and `sz_parity`;
  - `momentum` (the abelian characters) and `irrep_characters`;
  - `flip_parity`, `fold` (`"K"`, `"theta"` or None), `block_dim`, and the engine's indices
    `k0`, `k_raw` and `irrep`.

## Operators

`qed.Operator(N)` is a spin-1/2 operator on N < 64 sites. It has an exact algebra:

- `qed.Operator.product(N, ops, sites, coeff=1)` is
  `coeff * O_0(sites[0]) O_1(sites[1]) ...`, where the last factor acts first. Each
  character of `ops` is one of `+ - z x y u d I`: S⁺, S⁻, Sᶻ, Sˣ, Sʸ, the projectors |↑⟩⟨↑|
  and |↓⟩⟨↓|, and the identity. A product may have any number of factors, and sites may
  repeat: same-site products reduce by the spin-1/2 algebra (S⁺S⁺ = 0).
- The algebra operations are `A + B`, `A - B`, `-A`, `c * A`, `A / c` and `A @ B` (B acts
  first).
- The methods are `A.adjoint()`, `A.copy()`, `A.equals(B, rtol=1e-10)`,
  `A.is_hermitian(rtol=1e-10)` and `A.terms()`. `terms()` returns the canonical terms as
  `(coeff, ops, sites)`; they are the same however the operator was written.
  `A.image(perm, flip=False)` returns U O U†, followed by the global spin flip when `flip`
  is true. `A.apply(v)` applies the operator in the full 2^N basis.
- `add_one_body(op_type, site, coeff)`, `add_two_body(...)` and `add_three_body(...)` append
  records. Each `op_type` is one of `qed.OP_SPLUS`, `qed.OP_SMINUS`, `qed.OP_SZ`.

Terms on four or more sites work in the Hamiltonian on every lane, CPU and GPU, and in every
observable (`expect`, thermal `observables`, `dynamics`, `matrix_element`). Examples are ring
exchange, (S_i·S_j)(S_k·S_l), or any longer product. Symmetry detection and observables read
the canonical terms, so results depend on the operator, not on how it was written.

```python
P, N = qed.Operator.product, 12
dot = lambda i, j: P(N, "+-", [i, j], 0.5) + P(N, "-+", [i, j], 0.5) + P(N, "zz", [i, j])
H = sum((dot(i, (i + 1) % N) for i in range(N)), qed.Operator(N))
H = H + 0.2 * sum((dot(i, (i + 1) % N) @ dot((i + 2) % N, (i + 3) % N) for i in range(N)), qed.Operator(N))
assert H.is_hermitian()
```

`qed.input.HamiltonianBuilder(N)` builds common models from bonds. Its methods are:

- `heisenberg`, `xxz`, `xyz`, `ising` and `transverse_field_ising`;
- `kitaev` and `dm`;
- the fields `zeeman`, `zeeman_per_site` and `on_site_field`;
- the four-site terms `ring_exchange` and `ss_ss`;
- `pyrochlore_non_kramers`.

`qed.input.lattice` generates the bonds: `chain`, `square`, `triangular`, `honeycomb`, `kagome`,
`pyrochlore`, `from_neighbor_lists` and `from_cluster_file`. `qed.dssf` builds the
momentum-resolved spin operators of structure factors. See [`docs/operators.md`](docs/operators.md).

## Symmetries

Every verb takes `sym=`, a `qed.Symmetry`. The default `Symmetry.auto()` uses everything H
has except total spin:

- **Sz.** H splits into U(1) sectors, or into Sz-parity halves when only parity is
  conserved. `Symmetry(sz=n)` selects the sector with n up spins (Sz = n − N/2). `sz` also
  takes `"even"`, `"odd"` or `"off"`.
- **Spatial.** Site permutations that commute with H, either found by `qed.find_symmetries`
  (graph automorphisms; needs `pynauty`) or given as `spatial=[perm, ...]` or as
  `qed.Symmetries(abelian, residues)`. The largest normal abelian subgroup gives the momenta.
  The coset representatives (the point group) give the little groups and their irreps,
  projective ones included.
- **Spin flip and time reversal.** The global spin flip is used when H has it. Time reversal
  is K (complex conjugation) for a real H. Otherwise it is Θ = Π_i (iσʸ_i) K, which pairs
  (Sz, k) with (−Sz, −k).
- **Total spin.** `Symmetry(total_spin=S)` restricts the calculation to spin S. H must be
  SU(2) symmetric, optionally in a uniform field along z.

`Symmetry.auto().select(sz=, momentum={T: theta}, irrep_character={R: chi})` restricts a verb
to some sectors (`dynamics` takes only `sz` and `momentum`). A selection that matches no block
raises `qed.errors.EmptySelection`. Each level reports its momentum, `result.momentum(i, translations)`: the
fractions θ with T|ψ⟩ = e^{−2πiθ}|ψ⟩. It also reports its little-group characters,
`result.irrep_characters(i)`.

For large groups, orbit representatives are found through a block system of the lattice. The
`ED_SYM_SUBLATTICE` variable controls this:

- unset: on for N ≥ 24 sites and at least 16 distinct site permutations for a verb that runs on
  the device (`device="gpu"`, or `"auto"` with a visible GPU) and for `eigs`, `spectrum` and exact
  `thermal`; at least 64 for host sampled `thermal` and host `dynamics`, whose thousands of host
  sparse applies lose cache locality on chains (whose bonds join neighbouring bits);
- `1`: whenever a block system exists;
- `0`: never.

This changes which member represents each orbit, not the physics. The representative basis
differs by phases, so seeded sampled results move at sampling level. A saved result records the
rule it was computed with. See [`docs/symmetry.md`](docs/symmetry.md).

## Devices

Every verb takes `device="cpu" | "gpu" | "auto"`; `expect` passes it on to `eigs`.

- **`"cpu"`** runs on the host (OpenMP) and never initialises CUDA.
- **`"gpu"` is strict.**
  - It needs a CUDA build and a visible device; otherwise it raises
    `qed.errors.DeviceUnavailable`.
  - Every Krylov solve and every dense batch runs on the device. A block that cannot run
    there raises `DeviceUnsupported`, naming the block, or `ResourceLimit` when its working
    set does not fit in device memory; a dense block too large for the device, or whose
    device solve fails even alone, also raises `ResourceLimit`. Nothing falls back to the host.
  - Some dense solves stay on the host, and `placement` counts them:
    - `eigs` blocks up to `dense_max_dim`, of at most 32 states, or whose dimension is at most
      twice the number of levels the block owes;
    - sampled `thermal` blocks up to `dense_max_dim`;
    - exact `thermal` with `observables`.
- **`"auto"`** places each block by the table in `include/ed/core/device.h`. A block goes to
  the device at these sizes (Krylov blocks also need a device kernel and a working set that
  fits in free device memory):
  - eigs, FTLM/mTPQ and OFTLM: from 2^14 states;
  - T = 0 continued fractions: from 2^14 states;
  - T > 0 dynamics sources: from 2^16 states;
  - dense spectra: from 1024 states.

  Every other block runs on the host, and so does a dense block too large for the device or a
  dense batch whose device solve fails.

These parts run on the device:

- **Krylov lanes.** The `eigs` k = 1 scan, thick-restart Krylov-Schur and certified vectors;
  FTLM, OFTLM and mTPQ; the T = 0 continued fractions and T > 0 FTLM dynamics.
  - FTLM and mTPQ advance as many of a block's random vectors in lockstep as fit in device
    memory, so those samples share each H apply (on an allocation failure they retry with half
    as many).
- **H applies.**
  - A sector of a one-dimensional irrep builds its reduced CSR on the device once and applies
    it with a deterministic SpMV; an operator applied only once or twice keeps the gather. The
    budget is `ED_GPU_CSR_BUDGET_GIB`: unset, half the free device memory at the bind; `0`, no
    device CSR.
  - A sector of an irrep of dimension > 1 uploads the reduced CSR built on the host. `"gpu"`
    refuses such a sector (`DeviceUnsupported`) only when that CSR does not fit the block's CSR
    budget or the device CSR budget, and raises `ResourceLimit` when it cannot be uploaded
    after the block was placed.
  - Without a device CSR, a sector of a one-dimensional irrep is applied by a matrix-free
    gather kernel. `block_stats` reports the lane as `device-csr` or `device-gather`.
  - Under `"auto"` and `"gpu"`, an operator on a sector of a one-dimensional irrep that is
    applied on the host and whose reduced CSR was not built (under `"auto"` a block kept on the
    host; under `"gpu"` an operator a host path applies, such as S² or an observable) may still
    be applied with the device gather on host vectors from 2^20 representatives (lane
    `gpu-gather`; a Krylov solve on it counts as `host_krylov`). `ED_SYM_LG_GPU=0` vetoes this,
    `=1` drops the floor.
- **Dense spectra** (`spectrum`, exact `thermal`). They use cuSOLVER's 64-bit `syevd`, and real
  blocks run in real arithmetic. Batches hold up to `ED_GPU_DENSE_BATCH_GIB` (default 2 GiB,
  at most a quarter of the free device memory and of the RAM). A block larger than a batch
  runs alone when it and its workspace fit in half the free device memory; otherwise `"auto"`
  solves it on the host and `"gpu"` raises `ResourceLimit`. Under `"gpu"` a batch whose device
  solve fails is retried in halves on the device.

`qed.debug_env("ED_GPU")` lists the device variables with their values and defaults.

## Build

```bash
scripts/build.sh --variant cpu            # -> build/cpu   (OpenMP)
scripts/build.sh --variant cuda           # -> build/cuda  (CUDA; sm_90 unless CMAKE_CUDA_ARCHITECTURES or CUDAARCHS is set)
export PYTHONPATH=$PWD/python QED_CORE_DIR=$PWD/build/cuda/python/qed
```

`scripts/build.sh` options:

- `--cluster NAME` sources `scripts/clusters/NAME.env` first: `local` (default) or `alliance`.
- `--tests` builds the C++ unit tests; `--no-python` skips `qed._core`.
- `--target T` builds one target; `--clean` removes the build directory first.
- `--jobs N` sets the parallel jobs; `--name S` builds into `build/<variant>-S`.
- `--arch A` sets `-march` (default `native`).
- Arguments after `--` go to CMake, for example
  `scripts/build.sh --variant cuda -- -DCMAKE_CUDA_ARCHITECTURES=80`.

The extension is never written into the source tree. From a checkout, `QED_CORE_DIR` names
the build to load.

`pip install .` builds and installs a CPU-only package, because `pyproject.toml` sets
`WITH_CUDA=OFF`. An installed package needs no `QED_CORE_DIR`. `pip install ".[symmetry]"`
adds `pynauty`.

Requirements:

- CMake ≥ 3.18 and a C++17 compiler with OpenMP.
- BLAS, LAPACK and LAPACKE, chosen with `-DBLAS_PROFILE=AUTO|FLEXIBLAS|OPENBLAS|MKL`.
- Eigen 3.
- Python ≥ 3.9 with NumPy ≥ 1.21.
- pybind11 ≥ 2.10. CMake fetches v2.13.6 when it finds none; `pip install .` uses
  pybind11 ≥ 2.13 and scikit-build-core ≥ 0.10.
- For the CUDA variant: the CUDA toolkit (cudart, cuBLAS, cuRAND, cuSOLVER) and compute
  capability ≥ 6.0.

Optional: `pynauty` for `Symmetry(spatial="auto")` and `find_symmetries`. Without it, `auto`
warns and runs without spatial symmetry. The unit tests use Catch2 v3, which CMake fetches
when it is absent.

## Conventions

- Bit i of a basis state is site i, and a set bit is spin up. `n_up` counts up spins.
- `Symmetry(sz=n)` selects Sz = n − N/2.
- A site permutation `p` acts on states as: bit i of U|s⟩ = bit p[i] of |s⟩.
- The library prints nothing. Messages go to `logging.getLogger("qed")`;
  `qed.set_log_level(level, stream=None)` or `QED_LOG_LEVEL` sets the level (default `"warn"`).
  Without a stream, warn- and error-level records also arrive as `qed.errors.QEDWarning`.
- Refusals raise the classes in `qed.errors`. Each one also derives from the matching builtin
  (`ValueError`, `NotImplementedError`, `RuntimeError`, `MemoryError`). An argument of the wrong
  Python type raises `TypeError`, and an index out of range (an `Operator` site, a level of
  `matrix_element`) `IndexError`.
- Environment variables are read through one registry (`include/ed/core/config.h`);
  `qed.debug_env(prefix)` lists them. An `ED_*` name the registry does not declare warns at
  import (`ED_BUILD_*`, `ED_TEST_*` and `ED_BENCH_*` excepted; `QED_*` names are not scanned).
  A registered variable whose value does not parse warns at import, and every verb then
  refuses to run. With `ED_ENV_STRICT=1` both raise at import. A flag is off for `false`,
  `off`, `no` (any case) or an integer equal to zero, and on for `true`, `on`, `yes` or a
  nonzero integer.

## Verification

- `tests/unit/` holds the C++ unit tests (Catch2, run with `ctest`).
- `tests/python/` holds the API tests (pytest).
- `tests/python/grid/` checks every task × symmetry content × backend cell against a dense
  reference built without the library. It also checks properties of each answer:
  multiplicities, residuals, labels, scale covariance, sum rules and detailed balance.
- `tests/python/golden/` holds recorded results on CPU and GPU.
- `tests/python/regress/` replays every audit repro, so a fixed bug stays fixed.
- `tests/python/fuzz/` compares random small models with a dense reference.
- `bench/` holds timed cases with recorded baselines; `bench/submit.sh` runs them as separate
  jobs.
- `scripts/gate/` runs the unit tests, the API tests, the grid, the golden records, the
  regression repros, the fuzzer, the examples and the source lints as SLURM arrays on CPU and
  GPU nodes (the benchmarks are not part of it). `scripts/gate/sanitize.sbatch` runs ASan/UBSan
  and compute-sanitizer.
- GitHub CI (`.github/workflows/ci.yml`) runs the unit tests (GCC Release, and Clang with
  ASan/UBSan), the wheel with pytest, the examples and the CPU grid, the pre-commit hooks
  (clang-format, ruff) and clang-tidy. It also compiles the CUDA build without running it. `.github/workflows/docs.yml` builds the documentation site.

## Layout

```
python/qed/          the package: verbs (_verbs/), Symmetry, discovery, input (lattices,
                     HamiltonianBuilder), dssf, symmetry, errors, logging
python/qed/_bindings the pybind11 module qed._core: Operator, lattices, the sector verbs
include/ed/          the engine's installed headers: sectors, basis, ops, matvec, krylov,
                     thermal, dynamics, gpu, core, input, parallel
src/                 engine sources: engine/ (sector drivers), basis/, ops/, gpu/, input/, parallel/
tests/unit/          C++ unit tests
tests/python/        API tests; grid/, golden/, regress/, fuzz/; support/ (model zoo, dense oracle)
examples/            one runnable script per family of verbs
bench/               benchmark cases, baselines and results
scripts/             build.sh, env.sh, clusters/, gate/, golden/, the check_*.sh lints
docs/                the documentation site (Sphinx + Doxygen)
cmake/               build modules: BLAS, dependencies, pybind11, Catch2
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md): how a call reaches the kernels.
- [`docs/symmetry.md`](docs/symmetry.md): symmetries, sectors and labels.
- [`docs/operators.md`](docs/operators.md): operators and their algebra.
- [`docs/dynamics.md`](docs/dynamics.md): dynamical correlations.
- API reference: [`docs/api/python.rst`](docs/api/python.rst) and [`docs/api/cpp.rst`](docs/api/cpp.rst).
  The site is built with `-DED_BUILD_DOCS=ON` and `cmake --build <build> --target sphinx`.
- [`examples/`](examples/README.md): one runnable script per family of verbs.
- [`CHANGELOG.md`](CHANGELOG.md): the history. [`CONTRIBUTING.md`](CONTRIBUTING.md): the workflow.

MIT license; to cite, see [`CITATION.cff`](CITATION.cff).

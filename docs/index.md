# QED_Spin

QED_Spin does exact diagonalization of spin-1/2 Hamiltonians on the CPU (OpenMP) and on one
NVIDIA GPU (CUDA). It computes:

- lowest levels and eigenvectors;
- full spectra;
- thermodynamics: exact, FTLM, OFTLM and mTPQ;
- observables in one pass: expectation values, pair correlations and structure factors, transitions
  between levels, in levels or at temperatures, on operator families and momentum axes;
- dynamical correlations S_AB(ω) at zero and finite temperature.

Each calculation is split into the symmetry sectors of the Hamiltonian: Sz or its parity,
lattice momenta, point-group irreps, spin flip and time reversal by default, and total spin on
request. Hamiltonians and
observables are arbitrary sums of products of S⁺, S⁻ and Sᶻ, each product on any number of
sites.

```{toctree}
:maxdepth: 2

architecture
symmetry
operators
observables
dynamics
qfi
api/python
api/cpp
```

| page | contents |
|---|---|
| {doc}`Architecture <architecture>` | the path from a Python call to the kernels, and the backends |
| {doc}`Symmetry <symmetry>` | `qed.Symmetry`, sectors, level labels, selections, representatives |
| {doc}`Operators <operators>` | `qed.Operator` and its algebra, `HamiltonianBuilder`, lattices, `qed.dssf` |
| {doc}`Observables <observables>` | `qed.measure`: expectation values, correlations, transitions and dynamics on families, momenta and temperatures |
| {doc}`Dynamics <dynamics>` | `qed.dynamics`: probes, cross-correlations, T = 0 and T > 0 |
| {doc}`Python API <api/python>` | every public name of `qed` and its submodules |
| {doc}`C++ API <api/cpp>` | the installed headers: `ed::sectors`, operators, placement, backends, kernels |

The repository [README](https://github.com/ze-bang/QED_Spin#readme) has the quick start, the
build and the device rules. `examples/` has one runnable script per family of verbs. The
history is in [CHANGELOG.md](https://github.com/ze-bang/QED_Spin/blob/main/CHANGELOG.md), and
the development workflow in
[CONTRIBUTING.md](https://github.com/ze-bang/QED_Spin/blob/main/CONTRIBUTING.md).

The site is built from this directory with `-DED_BUILD_DOCS=ON` and
`cmake --build <build> --target sphinx`. Doxygen reads `include/` and `src/`, and Sphinx
renders the pages (MyST for Markdown, Breathe for the C++ reference).

## Indices and tables

- {ref}`genindex`
- {ref}`search`

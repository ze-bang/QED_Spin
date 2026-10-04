# Observables

Every observable computation is **a quantity, in a state, on an index axis**. The three are
independent: any quantity works in any state and on any axis, and one call can mix several
quantities.

| quantity | request | levels / ground manifold | temperatures `T=[...]` |
|---|---|---|---|
| one-point ⟨O⟩ | `qed.Expect(ops)` | ⟨ψ\|O\|ψ⟩, multiplet-averaged | Tr(e^{-H/T} O)/Z |
| equal-time pair ⟨A†B⟩ | `qed.Correlations(A, B=None)` | ⟨ψ\|A_a† B_b\|ψ⟩ | Tr(e^{-H/T} A_a† B_b)/Z |
| transitions ⟨m\|O\|n⟩ | `qed.Transitions(A, B=None)` | line strengths, pair matrices | (levels only) |
| dynamic pair S_AB(ω) | `qed.Dynamics(A, omega, B=None)` | Lehmann sum from the ground manifold | thermal Lehmann sum |

The **index axis** is whatever the operands are:

- one `qed.Operator`, or a sequence of them: axis `(len(ops),)`;
- a `qed.Family`: its shape, e.g. `(3, N)` for `qed.Family.spins(lattice)`, `(n_bonds,)` for
  `qed.Family.bonds(pairs, f)`;
- a `qed.MomentumFamily`, `family.fourier(q)`: the family's last axis replaced by momenta.

The operands are built with the operator algebra (`qed.Operator.product`, `+`, `-`, `*`, `@`,
`adjoint`), so a family can hold any operator: spins, bond energies S_i·S_j, chiralities, dimer
or plaquette terms.

## One pass for many quantities

`qed.measure(H, requests, k=1, T=None, ...)` answers every request from one eigensolve (or one
thermal pass) and one sweep of each block's vectors:

```python
lat = qed.input.lattice.triangular(3, 4, True)
spins = qed.Family.spins(lat)                      # S_i^a, shape (3, 12)
m = qed.measure(H, [qed.Expect(spins), qed.Correlations(spins)], states="ground")
mean, corr = m                                     # ExpectResult, CorrelationResult
S = corr.fourier("cluster")                        # S^ab(q) at the cluster momenta
```

`qed.expect`, `qed.correlations` and `qed.transitions` are the one-request forms.
`qed.thermal(..., requests=[...])` measures in the thermodynamics pass, and
`qed.dynamics` takes families and momentum families directly.

**Why one pass is cheap.** Every operator is averaged over the symmetries of the level (or
block) it is measured in, and averaged operators that are equal up to a factor are evaluated
once. Pairs related by a symmetry, (i, j) and (gi, gj), average to the same operator, so the full
N² correlation matrix of a translation-invariant state costs about N operators. All of them go
through one sweep of the level's basis. At finite temperature the same sweep runs over:

- **exact**: every eigenvector of each block;
- **FTLM**: each sample's φ(T) = Σ_j e^{-E_j/2T} ⟨ψ_j|r⟩ ψ_j, from its Krylov basis (the
  symmetric, low-temperature estimator), a few temperatures at a time;
- **OFTLM**: the certified exact states, plus the samples' φ(T) in the complement;
- **mTPQ**: each step's state and its successor, combined by the canonical series of Sugiura and
  Shimizu. That series is exact in expectation for an operator that commutes with H, and the
  standard approximation otherwise (accurate where it is sharply peaked: large systems).

## States

- `states="levels"` (default): one row per returned level, averaged over its symmetry
  multiplet, so the value does not depend on which partner the solver returned.
  `multiplicities[i] * values[i]` is the level's share of Tr(P_E O).
- `states="ground"`: one row, the levels within `degeneracy_tol` of the lowest, weighted by
  multiplicity. Partners in other blocks are included only if the solve returned them: raise
  `k` or pass `window=`.
- `T=[...]`: one row per temperature. `method`, `samples`, `krylov`, `steps`, `exact_states`,
  `seed` and `dense_max_dim` are those of `qed.thermal`. The thermodynamics come along in
  `MeasureResult.thermal`.
- `qed.eigs(H, per_block=m)` returns the lowest `m` levels of every symmetry block: the excited
  states of every sector, for `states="levels"` measurements and for transitions.

## Momentum convention

One convention everywhere: in families, in `qed.dssf`, in dynamics probes and in the momentum
labels.

$$O_q = N^{-1/2} \sum_r e^{-iq\cdot r} O_r, \qquad
S(q) = \langle O_q^\dagger O_q\rangle = N^{-1}\sum_{ab} e^{+iq\cdot(r_a-r_b)}\langle O_a^\dagger O_b\rangle .$$

A state of crystal momentum q carries the label θ_T = q·d_T/2π that `result.momentum(i, [T])`
reports, where d_T = r_i − r_{p[i]} is the translation's displacement. O_q maps momentum k to
k − q.

Helpers in `qed.input`:

- `cluster_momenta(lattice)` gives the momenta the periodic cluster allows. It also takes
  supercell and primitive vectors, or translation permutations with positions.
- `high_symmetry_points(lattice)` and `momentum_path(["G", "K", "M", "G"], n, lattice=...)` give
  paths through the zone. `correlations(...).fourier(q)` evaluates S(q) at any q, not only the
  cluster's.

For spin families, `StructureFactor.perp()` is the neutron projector
Σ_ab (δ_ab − q_a q_b/q²) S^ab(q), and `trace()` is Σ_a S^aa(q).

## Transitions

`qed.transitions(A, initial, final=None, B=None, pairs=False, raw=False)` takes levels from one
or two `eigs` results over the same spatial symmetry, each as an `EigResult` or
`(EigResult, indices)`. It returns the invariants of the multiplets:

- `strength[i, j, *A_index]` = d_i⁻¹ Σ_{n'∈i, m'∈j} |⟨m'|A_a|n'⟩|², the pole weight of A at
  ω = E_j − E_i in the T = 0 dynamics from level i;
- with pairs, `T[i, j, *A_index, *B_index]` = d_i⁻¹ Σ ⟨n'|A_a†|m'⟩⟨m'|B_b|n'⟩. Summed over a
  complete set of final levels it gives the equal-time ⟨A_a† B_b⟩.

Selection rules are exact zeros, not computed: momentum (O_q takes k to k − q) and the S^z change.
`raw=True` keeps the member-resolved amplitudes (`amplitudes(i, j)`). They depend on the members
chosen: the solver's vector first, then its symmetry images.

## Dynamics on families

`qed.dynamics(H, family.fourier("cluster"), omega)` gives S(q, ω) at every cluster momentum,
with the result's `q` and `index` set. With `B="all"`, the cross pairs that share a B share its
Lanczos runs at T = 0: one run per B and target block projects every A, so P operators cost P
runs, not P². `qed.Dynamics(A, omega, ...)` is the request form inside `qed.measure`; it runs its
own dynamics pass at the measurement's temperatures.

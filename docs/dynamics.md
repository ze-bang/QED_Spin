# Dynamics

`qed.dynamics` computes dynamical correlation functions -- autocorrelations such as the
dynamical structure factor $S^{zz}(q, \omega)$, and cross-correlations $S^{ab}(q, \omega)$ --
at zero temperature from the ground manifold by continued fractions, and at finite temperature
by the finite-temperature Lanczos method (FTLM), over the symmetry sectors of $H$. Operators,
probes built with `qed.Operator` and `qed.dssf`, and the conventions (a set bit is spin up,
$S^z = \pm 1/2$) are on the [Operators](operators.md) page; `qed.Symmetry` and its options
are on the [Symmetry](symmetry.md) page.

## Definition

For probes $A$ and $B$,

$$
S_{AB}(\omega) = \sum_m p_m \, \langle m | A^\dagger \, \delta_\eta(\omega - H + E_m) \, B | m \rangle
= \sum_{m,n} p_m \, \langle m|A^\dagger|n\rangle \langle n|B|m\rangle \, L_\eta\big(\omega - (E_n - E_m)\big),
$$

where $|n\rangle$, $E_n$ are the eigenpairs of $H$ and the delta function is broadened into the
unit-area Lorentzian

$$
L_\eta(x) = \frac{1}{\pi} \frac{\eta}{x^2 + \eta^2}.
$$

- **T = 0**: $p_m = 1/g$ on the $g$ states of the ground manifold, 0 elsewhere.
- **T > 0**: $p_m = e^{-E_m/T}/Z$ ($k_B = 1$, $T$ in the units of $H$).

Equivalently $S_{AB}(\omega) = \frac{1}{2\pi}\int dt\, e^{i\omega t - \eta|t|} \langle A^\dagger(t) B\rangle$.
The autocorrelation ($B = A = O$) is
$S_O(\omega) = \sum_{m,n} p_m |\langle n|O|m\rangle|^2 L_\eta(\omega - E_n + E_m) \ge 0$; with
$O = S^a_q$ it is the dynamical structure factor. $S_{BA}(\omega) = S_{AB}(\omega)^*$.

## Calling it

```python
qed.dynamics(H, O, omega, B=None, *, eta=0.05, T=None, sym=None, krylov=200, samples=40,
             seed=0, degeneracy_tol=1e-8, device="cpu", dense_max_dim=None, prune=True)
    -> qed.DynamicsResult
```

| argument | default | meaning |
|---|---|---|
| `H` | | the Hamiltonian, a Hermitian `qed.Operator` |
| `O` | | the probe $A$: a `qed.Operator` or a non-empty sequence of them |
| `omega` | | the frequencies: any non-empty sequence of finite numbers, in any order |
| `B` | `None` | `None`, a `qed.Operator`, a sequence as long as `O`, or `"all"` (see Probes) |
| `eta` | `0.05` | the Lorentzian half-width $\eta > 0$, in the units of $H$ |
| `T` | `None` | `None`: zero temperature. A sequence of finite positive temperatures (a scalar is read as one temperature) |
| `sym` | `None` | a `qed.Symmetry`; `None` is `qed.Symmetry.auto()` |
| `krylov` | `200` | Lanczos depth: of each continued fraction (T = 0), of the source and target runs (T > 0); >= 1 |
| `samples` | `40` | T > 0: random vectors per source sector; >= 1 |
| `seed` | `0` | T > 0: base seed; 0 draws a fresh one |
| `degeneracy_tol` | `1e-8` | T = 0: the ground-manifold window, relative to $s_H$; >= 0 |
| `device` | `"cpu"` | `"cpu"`, `"gpu"` or `"auto"` |
| `dense_max_dim` | `None` | T = 0: the dense crossover of the ground-manifold eigensolve, as in `qed.eigs`; >= 0 |
| `prune` | `True` | T = 0: the ground-manifold eigensolve skips blocks whose short Lanczos estimate lies far above $E_0$ |

Arguments that cannot be answered raise `qed.errors.InvalidRequest` (a `ValueError`) before
any work: a non-Hermitian $H$, a probe that is not a `qed.Operator`, acts on another number of
sites or has a non-finite coefficient, `B` of the wrong kind or length, an empty or non-finite
`omega`, an `eta` that is not finite and positive, `T=[]` (use `T=None`), a temperature that
is not finite and positive, `krylov=0`, `samples=0` at T > 0, a `degeneracy_tol` that is
negative or not finite, `dense_max_dim < 0`, a `device` other than `"cpu"`, `"gpu"`, `"auto"`,
and `spin_flip="require"` / `time_reversal="require"` for an $H$ without that symmetry.
`krylov`, `samples` and `seed` are unsigned in the engine: a negative value is a `TypeError`.

## Probes and the shape of `S`

One call evaluates one or more probes, each a pair $(A, B)$:

| `O` | `B` | probes | `S.shape` | dtype |
|---|---|---|---|---|
| operator | `None` | $S_{O}$ | `(rows, len(omega))` | real |
| operator | operator | $S_{O,B}$ | `(rows, len(omega))` | complex |
| sequence of $P$ | `None` | $S_{O_i}$ | `(P, rows, len(omega))` | real |
| sequence of $P$ | operator | $S_{O_i,B}$ | `(P, rows, len(omega))` | complex |
| sequence of $P$ | sequence of $P$ | $S_{O_i,B_i}$ | `(P, rows, len(omega))` | complex |
| sequence of $P$ | `"all"` | $S_{O_i,O_j}$ | `(P, P, rows, len(omega))` | complex |

`rows` is 1 at T = 0 and `len(T)` at T > 0: `S[..., i, :]` is the spectrum at `T[i]`, in the
order given; a temperature listed twice gets the same row twice.

- `B=None` gives each probe's autocorrelation as a **real** array. It is real at T = 0; at
  T > 0 the imaginary part a finite sample gives the estimate is dropped.
- Any other `B` gives a **complex** array. A cross spectrum is complex in general; it is real
  for a real $H$ with real probes (up to sampling noise at T > 0). A probe
  paired with the very same object (the diagonal of `"all"`) is computed as an
  autocorrelation; an equal copy is computed as a cross pair.
- Under `"all"`, $S_{O_j,O_i} = S_{O_i,O_j}^*$ holds at T = 0 once the Krylov runs have
  converged (the two come from runs started from different vectors; to roundoff when `krylov`
  covers every target sector), and on average over samples at T > 0.
- At T = 0 the cross pairs that share their B share its Lanczos runs: one run from B|ψ⟩ per
  target block projects every A (up to 32 at a time), so `"all"` over P operators costs P runs
  per block, not P².
- The probes of one call share the ground manifold (T = 0) and the source sectors with each
  sample's source Lanczos run (T > 0). At T > 0 each probe sums over the source sectors its
  own symmetries relate (see Symmetry), so a probe's result does not depend on the other
  probes of the call: a sequence gives what separate calls give.
- Probes may change $S^z$ ($S^\pm$), mix $S^z$ changes ($S^x$, $S^y$) and need not share any
  symmetry of $H$. Their terms decide exactly which target sectors they reach.

Cross spectra satisfy the polarisation identity
$S_{AB} = \frac14 \sum_{k=0}^{3} i^{-k} S_{X_k}$ with $X_k = A + i^k B$, which relates them to
autocorrelations of single operators.

## Frequencies and the result

`omega` is the energy transferred to the system: a peak at $\omega$ is a transition from
$|m\rangle$ to a state $\omega$ higher.

- **T = 0**: $\omega$ is measured from the ground-state energy $E_0$, so the poles sit at
  $E_n - E_0$. They are $\ge 0$ -- with only Lorentzian tails at $\omega < 0$ -- unless a
  restriction of `sym` (an explicit `sz`, an $S^z$-parity half, `total_spin` or a momentum
  selection) puts $E_0$ above states the probe reaches.
- **T > 0**: $\omega = E_n - E_m$ for every pair; weight appears at both signs of $\omega$.

$\omega$, $\eta$ and $T$ are in the units of $H$, and $S$ in inverse units of $H$ times the
units of $A^\dagger B$. Every tolerance is relative to the scale of $H$, so for $s > 0$
`qed.dynamics(s * H, O, s * omega, eta=s * eta)` (with `T=s * T` and the same fixed `seed` at
finite temperature) returns `S / s`.

`qed.DynamicsResult` holds

| field | |
|---|---|
| `omega` | the frequencies, as given |
| `T` | the temperatures, as given; empty at T = 0 |
| `S` | the spectra (shapes above) |
| `e0` | T = 0: the lowest energy in the sectors `sym` allows. T > 0: 0.0 (not computed) |
| `ground_manifold` | T = 0: $g$, the number of states averaged over (every member of every degenerate level). T > 0: 0 |
| `device_blocks` | continued fractions (T = 0) or FTLM source sectors (T > 0) run on a GPU |
| `symmetry` | the `qed.Symmetry` used |
| `diagnostics` | `(code, message)` pairs: those of resolving `sym`, and `("restricted_ensemble", ...)` (below) |
| `placement` | solves per lane: `device_krylov`, `device_dense`, `host_krylov`, `host_dense` (ground-manifold eigensolves, continued fractions, FTLM sources) |

## Zero temperature

`T=None` (the default).

**The ground manifold.** $E_0$ is the lowest energy in the sectors `sym` allows. The manifold
is every level within `degeneracy_tol` $\times\, s_H$ of $E_0$, where $s_H$ is the sum of
$|c|$ over the canonical terms of $H$ written with the site factors $S^\pm$ and
$\sigma^z = 2S^z$ (an upper bound of $\lVert H\rVert$):

```python
s_H = sum(abs(c) / 2 ** ops.count("z") for c, ops, _ in H.terms())
```

A pruned $k = 1$ eigensolve with that window finds the blocks at $E_0$; each is then solved
alone, deeper until a level above the window appears, so degeneracies inside a block are
caught. `dense_max_dim` and `prune` apply to these solves as in `qed.eigs`. A block that cannot
certify its levels raises `qed.errors.ConvergenceError` (a `RuntimeError`). Each level is expanded into all its symmetry
partners (other momenta of its star, point-group partners, its spin-flip and time-reversal
partners), and under `Symmetry(total_spin=S)` with an SU(2)-symmetric $H$ into all $2S + 1$
members of each multiplet (the solve returns the $S^z = S$ member; the others follow by
$S^-_\mathrm{tot}$). The result averages these $g$ states with equal weight 1/g. A small
`degeneracy_tol` near 0 can drop a partner solved in another block, whose energy agrees with
$E_0$ only to roundoff; a larger one averages over a quasi-degenerate manifold.

**Continued fractions.** For each state $|a\rangle$ of the manifold, each probe and each target
sector, $|\phi\rangle = B|a\rangle$ restricted to that sector starts a Lanczos run of `krylov`
steps (at least 2; no reorthogonalisation; it stops early at an invariant subspace, when
$\beta$ falls below $64\,\epsilon\, s_H$, $\epsilon$ the double-precision machine epsilon).
Under a point group or the spin flip the run is per one-dimensional irrep block of the target
(see Symmetry). An autocorrelation is the continued fraction

$$
S(\omega) = -\frac{1}{\pi} \mathrm{Im}\, \langle\phi| (\omega + i\eta - H + E_0)^{-1} |\phi\rangle ,
$$

a cross pair its pole form $\sum_k \langle a|A^\dagger|\theta_k\rangle\langle\theta_k|B|a\rangle L_\eta(\omega - \theta_k + E_0)$
over the Ritz pairs $(\theta_k, |\theta_k\rangle)$ of the run from $B|a\rangle$, with $A|a\rangle$
projected on each Krylov vector as it forms. The contributions are summed over target sectors
and divided by $g$.

A depth-$M$ continued fraction reproduces the spectral moments
$\langle\phi|(H - E_0)^k|\phi\rangle$ for $k \le 2M - 1$ (in exact arithmetic). The zeroth and
first moments are exact at any `krylov`, for a cross pair too: the total weight is
$\lVert B|a\rangle \rVert^2$ (an autocorrelation) or $\langle a|A^\dagger B|a\rangle$, and the
depth is at least 2. The line shape converges with `krylov`;
smaller `eta` resolves more poles and needs a deeper run. Check convergence by repeating with a
larger `krylov`. With `krylov` at least the dimension of every target sector the result is the
exact broadened Lehmann sum.

The working set holds one target sector at a time.

## Finite temperature

`T=[T1, T2, ...]`. Every temperature of a call comes from the same samples.

**Source sectors.** The initial states are the momentum sectors of every $S^z$ subspace (or
$S^z$-parity half, or the whole space) that `sym` allows. Each source sector $s$ of dimension
$D_s$ is sampled -- however small; there is no exact path at T > 0 -- with `samples` Gaussian
random vectors $|r\rangle$. For each one, a Lanczos run on $H_s$ from $|r\rangle$ (depth
$\min(\texttt{krylov}, D_s)$) gives Ritz pairs $(E_i, |\psi_i\rangle)$ with
$c_i = \langle\psi_i|r\rangle$, and for each target sector $t$ a probe reaches, a second run on
$H_t$ from $B|r\rangle$ (depth $\min(\texttt{krylov}, D_t)$) gives $(\lambda_j, |\chi_j\rangle)$.
With $R$ = `samples`,

$$
S_s(\omega) = \frac{D_s}{R} \sum_r \sum_i e^{-E_i/T} c_i \sum_j \langle\psi_i|A^\dagger|\chi_j\rangle \langle\chi_j|B|r\rangle \, L_\eta\big(\omega - (\lambda_j - E_i)\big),
\qquad
Z_s = \frac{D_s}{R} \sum_r \sum_i e^{-E_i/T} c_i^2 ,
$$

and $S(\omega) = \sum_s S_s(\omega) / \sum_s Z_s$, with $Z$ summed over every source sector,
including those the probes annihilate (Jaklic-Prelovsek). Both Lanczos runs keep their basis
and reorthogonalise it fully (CGS2), or locally (DGKS3) in sectors larger than both $2^{16}$
states and $4 \times$ `krylov`.

**Sampling.**

- The estimate has sampling noise; it shrinks with `samples` and with the sector dimensions.
  Estimate it by repeating with other seeds. Where only a few states carry the Boltzmann
  weight the noise is largest; for the ground state use `T=None`.
- `seed=0` draws a fresh seed per call (results differ run to run). A fixed `seed` seeds each
  source sector from `seed` and the sector's identity ($n_\uparrow$, parity, momentum), so the
  samples do not depend on the order in which sectors are scheduled.
- `samples` is per source sector: a call runs `samples` times the number of source sectors
  it runs.

**Restricted ensembles.** With an explicit `sz=n`, `sz="even"` / `"odd"`,
`sym.select(momentum=...)` or `total_spin=S`, the average runs over that part of the space only
(its canonical ensemble), and `diagnostics` carries
`("restricted_ensemble", "dynamics: the averages run over ... only: ...")`. Under
`total_spin=S` with an SU(2)-symmetric $H$ the sources are the spin-$S$ tower's members at every
$S^z = S, \ldots, -S$: each random vector is projected onto the tower, and the tower's dimension
replaces $D_s$. A momentum selection that matches no source sector raises
`qed.errors.EmptySelection`.

**Memory.** A sample holds about $(\texttt{krylov} + 5) \times 16$ bytes $\times (D_s + D_t)$ on
the host. Sources below $2^{16}$ states run concurrently, one thread each, as many as the RAM
holds; larger ones run one at a time with every thread, and one whose working set exceeds 90%
of the available RAM raises `qed.errors.ResourceLimit` (`ED_MEM_GUARD_OFF=1` lifts the
check). The call holds one source subspace and its targets at a time.

## Symmetry

Probes are generally not invariant under the point group, the spin flip or time reversal (a
Fourier mode $O_q$ maps to $O_{Rq}$), so `qed.dynamics` works in the momentum sectors of the
abelian part of the spatial group and uses the rest only where it is exact for the probe:

| `sym` | T = 0 | T > 0 |
|---|---|---|
| `spatial` (abelian part) | momentum sectors of sources and targets | momentum sectors of sources and targets |
| `spatial` (point group) | the ground-manifold eigensolve runs on the irrep blocks\*; each target momentum sector is split into its one-dimensional irrep blocks, with one continued fraction per block, and the part of $B\lvert a\rangle$ in irreps of dimension > 1 runs on the whole momentum sector | for each residue $R$ with $RAR^\dagger = c_A A$, $RBR^\dagger = c_B B$ and $c_A^* c_B = 1$, the momenta $R$ relates are sampled once, weighted by the size of their orbit |
| `sz="auto"` | $S^z$ (or parity) sectors; probes may change $S^z$ | the same; every $S^z$ sector is a source |
| `sz=n`, `"even"`, `"odd"` | the ground state of that part | a restricted ensemble |
| `spin_flip` (`"auto"`, `"require"`) | folds the ground-manifold solve\*, and, when $H$ is flip symmetric, splits the targets that are their own flip image ($S^z = 0$, a parity half at even $N$, the full space) into flip-even and flip-odd blocks | when $FAF = c_A A$ and $FBF = c_B B$ ($F = \prod_i \sigma^x_i$), the sources with $n_\uparrow > N/2$ are not run: each one below $N/2$ stands for its mirror, its $Z$ counted twice and its $S$ with the factor $1 + c_A^* c_B$ |
| `time_reversal` | folds the ground-manifold solve\* only (never the targets) | not used |
| `total_spin=S` | the lowest spin-$S$ multiplet, all its members | the spin-$S$ tower (restricted ensemble) |
| `select(momentum=...)` | the ground state of those momentum sectors (solved on momentum sectors) | restricts the sources (restricted ensemble); no point-group folding |
| `select(k0=...)`, `select(irrep=...)`, `select(irrep_character=...)` | `qed.errors.Unsupported` | `qed.errors.Unsupported` |

\* Under `device="gpu"` or a momentum selection the ground manifold is solved on the momentum
sectors instead, without the point group, the flip or time reversal (without a selection it
is the same manifold).

`"require"` for `spin_flip` or `time_reversal` checks that $H$ has the symmetry (either complex
conjugation $K$ or $\Theta$ for time reversal), and raises `InvalidRequest` otherwise. At
T > 0 the point group folds sources only without `total_spin` and without a momentum
selection; the spin flip only without `total_spin` and when every source $S^z$ sector's mirror
is a source too. With `qed.Symmetry.none()` everything runs on the full $2^N$ space.

## Sum rules and checks

Integrating the Lorentzians over all $\omega$ gives the frequency moments of the probe pair in
the initial ensemble:

$$
\int S_{AB}(\omega)\, d\omega = \langle A^\dagger B\rangle,
\qquad
\int \omega\, S_{AB}(\omega)\, d\omega = \langle A^\dagger [H, B]\rangle ,
$$

with $\langle\cdot\rangle$ the ground-manifold average at T = 0 (where $H|a\rangle = E_0|a\rangle$
makes the second identity hold with $\omega$ measured from $E_0$) and the thermal average at
T > 0. Both right-hand sides are operators the algebra builds, and `qed.expect` /
`qed.thermal(observables=...)` evaluate them:

```python
import numpy as np
import qed

trapezoid = getattr(np, "trapezoid", None) or np.trapz
# dense near the spectrum (spacing eta / 10), sparse in the tails
omega = np.concatenate([np.linspace(-200.0, -5.0, 400, endpoint=False),
                        np.linspace(-5.0, 10.0, 3001),
                        np.linspace(10.0, 200.0, 400)[1:]])
m0 = O.adjoint() @ O
m1 = O.adjoint() @ (H @ O - O @ H)

r = qed.dynamics(H, O, omega, eta=0.05)                       # T = 0
g = qed.expect(H, [m0, m1], 1)                                # the ground level(s)
ref = (g.multiplicities[:, None] * g.values).sum(axis=0) / g.multiplicities.sum()
print(trapezoid(r.S[0], omega), ref[0].real)
print(trapezoid(omega * r.S[0], omega), ref[1].real)

r = qed.dynamics(H, O, omega, eta=0.05, T=[1.0], seed=1)      # T = 1
th = qed.thermal(H, [1.0], method="exact", observables=[m0, m1])   # small N; else "ftlm"
print(trapezoid(r.S[0], omega), th.O[0, 0].real)
```

(`qed.expect(..., 1)` returns the lowest level and every copy of it found in other blocks;
`multiplicities * values` summed over them is the manifold average when they make up the
ground manifold. Pass the same `sym` to the reference as to `dynamics`: a restricted ensemble
has its own averages.)

What the checks test, and what limits them:

- **Truncation of the grid.** The Lorentzian tails decay as $1/\omega^2$: a pole at $E$ keeps
  the fraction $[\arctan((b - E)/\eta) - \arctan((a - E)/\eta)]/\pi$ of its weight inside
  $[a, b]$, about $1 - \frac{\eta}{\pi}\big(\frac{1}{E - a} + \frac{1}{b - E}\big)$. The first moment
  picks up $\frac{\eta}{2\pi} \ln\frac{(b - E)^2 + \eta^2}{(a - E)^2 + \eta^2}$ per unit weight,
  which a window placed symmetrically about the spectrum keeps small. The grid spacing must
  resolve $\eta$.
- **T = 0.** The continued fraction keeps both moments exactly at any `krylov`, so these
  checks test the operators, the ground manifold and the grid, not the Krylov convergence of
  the line shape; for that, compare runs at two values of `krylov`.
- **T > 0.** The integral is FTLM's sampled zeroth moment, so it agrees with the exact thermal
  average within the sampling noise (a few percent at a few tens of samples on small sectors).
- **Detailed balance**, at T > 0: $S_O(-\omega) = e^{-\omega/T} S_{O^\dagger}(\omega)$ holds for
  the Lehmann sum; the broadened spectrum obeys it where $\eta \ll T$, and the FTLM estimate
  within its noise. Evaluate `O` and `O.adjoint()` in one call (`[O, O.adjoint()]`).
- **Small systems.** For $N \lesssim 12$ the exact Lehmann sum is a reference: build dense
  matrices with `Operator.apply` (see [Operators](operators.md)), diagonalise $H$, and sum
  $p_m |\langle n|O|m\rangle|^2 L_\eta(\omega - E_n + E_m)$.

`examples/03_dynamics.py` prints the weight over $[0, 4]$ only. That window misses the
Lorentzian tails below $\omega = 0$ at T = 0 and the weight at $\omega < 0$ at T = 1, so
neither number is a complete zeroth moment.

## Devices

| `device` | T = 0 | T > 0 |
|---|---|---|
| `"cpu"` | everything on the host; CUDA is never initialised | the same |
| `"auto"` | the ground-manifold eigensolve as `qed.eigs(device="auto")`; continued fractions on the device for target blocks of at least $2^{14}$ states whose Lanczos vectors fit in free device memory | on the device: source sectors of at least $2^{16}$ states whose two Krylov bases fit in free device memory |
| `"gpu"` | the ground manifold is solved on momentum sectors on the device (small blocks may still be solved densely on the host); every continued fraction on the device | every source sector on the device |

- `"gpu"` is strict: without a CUDA build or a visible device it raises
  `qed.errors.DeviceUnavailable` before any work. The blocks dynamics runs on (momentum
  sectors and one-dimensional irrep blocks) have device kernels; a continued fraction or a
  source sector whose device working set exceeds the free device memory raises
  `qed.errors.ResourceLimit`.
- On the device a source sector runs up to 8 of its samples in lockstep, sharing each $H$
  apply, as many as fit in 90% of the free device memory; device sources run one at a time.
- An operator applied many times (a target $H$ in a continued fraction or FTLM run) is applied
  through a reduced CSR, built on the device within `ED_GPU_CSR_BUDGET_GIB` (unset: half the
  free device memory; 0: the walk); an operator applied only once or twice keeps the walk.
  On the host a probe between two sectors walks its first apply and from the second uses a
  merged CSR when its bound fits `ED_XSEC_CSR_BUDGET_GIB` (default 4, in GiB, shared among the
  sectors building one at the same time); over budget it is walked on every apply. A T = 0
  continued fraction applies the probe once, so it never builds that CSR.
- `device_blocks` and `placement` report where the solves ran. `ED_SYM_PROFILE=1` logs the
  wall time of each phase (target sectors, ground manifold, continued fraction, compile and
  apply $O$, the FTLM kernel) at the `info` level.

## Choosing the knobs

| knob | effect | guidance |
|---|---|---|
| `eta` | line width; peak height $1/(\pi\eta)$ per unit weight | finite-size poles are discrete: choose $\eta$ above their spacing for a smooth curve, below it to resolve them; resolve $\eta$ on the grid |
| `krylov` | T = 0: continued-fraction depth; T > 0: depth of both Lanczos runs | increase until the line shape stops changing; smaller $\eta$ needs more |
| `samples` | T > 0 statistical error | increase until results from different seeds agree |
| `seed` | T > 0 reproducibility | fix it for reproducible runs |
| `degeneracy_tol` | T = 0 ground-manifold window ($\times s_H$) | the default keeps exact degeneracies; raise it to average a quasi-degenerate manifold |
| `dense_max_dim` | T = 0: blocks up to this dimension are diagonalised densely in the ground-manifold solve | `None` is the automatic crossover of `qed.eigs` |
| `prune` | T = 0: `False` solves every block in the ground-manifold solve | the answer is the same; `False` costs more |

`dense_max_dim`, `prune` and `degeneracy_tol` are not used at T > 0; `samples` and `seed` are
not used at T = 0.

## Worked examples

**`examples/03_dynamics.py`**: $S^{zz}(q, \omega)$ of the 16-site Heisenberg ring at T = 0 and
T = 1. The probe is built record by record,

```python
def sz_q(q):
    o = qed.Operator(N)
    for j in range(N):
        o.add_one_body(qed.OP_SZ, j, cmath.exp(-1j * q * j) / math.sqrt(N))
    return o
```

$S^z_q = N^{-1/2}\sum_j e^{-iqj} S^z_j$, at $q = \pi/2$ and $\pi$. `qed.dynamics(H, sz_q(q), omega,
eta=0.05)` uses the default `Symmetry.auto()`: the singlet ground state is solved on its
folded block, and $S^z_q$ carries it into the momentum sector shifted by $q$, at the same
$S^z$. `S[0]` is the T = 0 row. The T = 1 call (`T=[1.0], samples=20, seed=1`) samples the
momentum sectors of the $S^z$ sectors and runs once each set of sources that a symmetry of
$H$ relates where it maps the probe to a multiple of itself with $c_A^* c_B = 1$: the spin
flip ($F S^z_q F = -S^z_q$) relates $S^z$ and $-S^z$, and at $q = \pi$, where the reflection
maps $S^z_q$ to itself, it relates $k$ and $-k$; at $q = \pi/2$ the reflection maps $S^z_q$ to
$S^z_{-q}$ and is not used. `S[0]` is the row of T = 1. $S^\pm_q$ work
the same way and reach the neighbouring $S^z$ sectors.

**`examples/06_cross_dynamics.py`**: the transverse matrix $S^{ab}(\pi, \omega)$, $a, b \in \{x, y\}$,
of a 14-site Heisenberg ring in a field $h = 1$ along $z$, at T = 0. The probes are built with
`Operator.product` (`"x"`, `"y"`, `"+"`, `"-"` letters) and summed with the algebra.
`qed.dynamics(H, [s_q("x", q), s_q("y", q)], omega, B="all", eta=0.05)` returns `S` of shape
`(2, 2, 1, 301)`; `r.S[:, :, 0, :]` is the matrix $[S^{xx}, S^{xy}; S^{yx}, S^{yy}]$ from one
ground-state solve. The field leaves the rotations about $z$, so $S^{xx} = S^{yy}$ and
$S^{xy} = -S^{yx}$; with $S^{yx} = (S^{xy})^*$ this makes $S^{xy}$ purely imaginary. Because
$S^x = (S^+ + S^-)/2$, $S^y = (S^+ - S^-)/(2i)$, and $S^+_q|0\rangle$ and $S^-_q|0\rangle$ lie in
different $S^z$ sectors, $2(S^{xx} + iS^{xy})$ and $2(S^{xx} - iS^{xy})$ equal the
autocorrelations of $S^+_q$ and $S^-_q$, which the script computes separately and compares.

## Errors

| error | when |
|---|---|
| `qed.errors.InvalidRequest` (`ValueError`) | the validation above |
| `qed.errors.EmptySelection` | a selection that matches no sector |
| `qed.errors.Unsupported` (`NotImplementedError`) | `select(k0=...)`, `select(irrep=...)`, `select(irrep_character=...)` |
| `qed.errors.DeviceUnavailable` (`RuntimeError`) | `device="gpu"` without a CUDA build or a visible device |
| `qed.errors.ResourceLimit` (`MemoryError`) | a working set that does not fit |
| `qed.errors.ConvergenceError` (`RuntimeError`) | a ground-manifold block that could not certify its levels |
| `TypeError` | a negative `krylov`, `samples` or `seed` |

## Implementation map

| | |
|---|---|
| `python/qed/_verbs/dynamics.py` | the verb: argument checks, probe pairs, result shapes, temperature order |
| `include/ed/sectors/dynamics.h` | `ed::sectors::dynamics(H, s, probes, d)`: `DynamicsSpec`, `Probe{A, B}` (`B` null: `A`'s autocorrelation), `DynamicsCurves` with `S[probe][row][omega]` |
| `src/engine/dynamics.cpp` | the ground manifold, target blocks, source folding, placement |
| `include/ed/dynamics/cf.h` | `continued_fraction`, `cf_spectral_from_vector`, `cross_spectral_many` (one Lanczos run from B psi projects every A) |
| `include/ed/dynamics/ftlm_dynamics.h` | `ftlm_dynamics_kernel`: one source sector against all its targets |

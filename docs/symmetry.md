# Symmetry

Every verb (`qed.eigs`, `qed.spectrum`, `qed.thermal`, `qed.dynamics`, `qed.expect`) takes a
`sym=` argument, a `qed.Symmetry`. It names which symmetries of H the calculation may use; the
verb resolves it against H and runs block by block. `sym=None` is `qed.Symmetry.auto()`: every
symmetry H has. The symmetries change how the work is split, never the answer: spectra,
thermodynamics, S(omega) and expectation values are those of H on the space the request names.

```python
import numpy as np
import qed

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
b.heisenberg([(i, (i + 2) % N) for i in range(N)], J=0.35)
H = b.to_operator()

r = qed.eigs(H, 4)                                   # Symmetry.auto()
ref = qed.eigs(H, 4, sym=qed.Symmetry.none())        # one block of 2^N states
assert np.allclose(r.energies, ref.energies)
```

The sections below follow the decomposition: the conventions, the `Symmetry` object and how it
resolves, Sz, spatial symmetry, the spin flip, the antiunitary symmetries, total spin, sector
selection, what each verb does with each symmetry, and the rule that picks orbit
representatives.

## Conventions

**Basis states.** A basis state of N sites is an integer s; bit i is site i, and a **set bit is
spin up**. `n_up` counts up spins, and $S^z = n_\mathrm{up} - N/2$. `Symmetry(sz=n)` selects the
sector with n up spins, i.e. $S^z = n - N/2$. Full-basis vectors (`EigResult.vectors()`) are
indexed by s; Sz-sector vectors (`vectors(basis="sz", n_up=n)`) list the $\binom{N}{n}$ states of
popcount n in ascending integer order.

**Sz parity.** `sz="even"` / `"odd"` name the half of the space whose number of up spins has that
parity.

**Site permutations.** A permutation p is a list of N ints. It acts on basis states as

$$\text{bit } i \text{ of } U_p|s\rangle \;=\; \text{bit } p[i] \text{ of } |s\rangle ,$$

so $U_g U_h = U_{g'}$ with $g'[i] = h[g[i]]$. `qed.symmetry.compose(a, b)[i] = a[b[i]]` (b
first) is the composition of the lists, and on states
$U_{\mathrm{compose}(a,b)} = U_b U_a$. `qed.symmetry.translation(n_sites, shift=1)` returns
$T[i] = (i - \mathrm{shift}) \bmod N$: $U_T$ carries the spin of site i to site i + shift.

**Symmetry verdicts.** Every test of whether H has a symmetry (a site permutation, the spin flip,
complex conjugation, time reversal, U(1), Sz parity, SU(2)) compares H's canonical terms
(`Operator.terms()`), with one relative tolerance, $10^{-10}$ times H's largest coefficient. The
verdict does not depend on how H was written or on its overall scale.

## `qed.Symmetry`

`qed.Symmetry` is a frozen dataclass:

| argument | default | meaning |
|---|---|---|
| `spatial` | `"auto"` | `"auto"`: the automorphisms of H's interaction graph that commute with H (`qed.find_symmetries`). A list of site permutations: the group they generate. A `qed.Symmetries`: an explicit split into abelian part and residues. `None`: no spatial symmetry. |
| `sz` | `"auto"` | `"auto"`: decompose by Sz, or by Sz parity when H conserves only that. An `int` n: the one sector with n up spins. `"even"` / `"odd"`: the sectors (or the half) whose up-spin count has that parity. `"off"`: no Sz decomposition. |
| `spin_flip` | `"auto"` | the global flip $F = \prod_i \sigma^x_i$: `"auto"` (use it when H has it), `"off"`, `"require"` (raise `InvalidRequest` when H lacks it). |
| `time_reversal` | `"auto"` | the antiunitary map K or $\Theta$ (see [Time reversal](#time-reversal-k-and-theta)): `"auto"`, `"off"`, `"require"` (raise when H has neither). |
| `point_group` | `True` | `False` keeps only the abelian part of the spatial group. |
| `total_spin` | `None` | a non-negative multiple of 1/2: restrict to total spin S (see [Total spin](#total-spin)). |

The remaining fields (`only_k0`, `only_irrep`, `only_momentum`, `only_irrep_character`) are set by
[`select()`](#selecting-sectors-select), not by hand. The toggles are case-insensitive.

- `Symmetry.auto()` is `Symmetry()`.
- `Symmetry.none()` is `Symmetry(spatial=None, sz="off", spin_flip="off", time_reversal="off")`:
  one block, the whole $2^N$ space.

### `groups()` and `resolve()`

`Symmetry.groups(H, diagnostics=None)` returns `(A, residues)`: `A` the closed abelian group (the
momenta), identity first, normal in the spatial group; `residues` one representative per coset of
A other than A itself (empty under `point_group=False`). Each element is a `list[int]`.
`diagnostics`, when given, receives a `(code, message)` pair for each fallback taken. A string
`spatial` other than `"auto"` raises `InvalidRequest`.

`Symmetry.resolve(H, diagnostics=None)` returns the engine's `qed._core.sectors.Spec`:

| `Spec` field | from | meaning |
|---|---|---|
| `abelian`, `residues` | `groups(H)` | as above |
| `n_up` | `sz=int` | the one Sz sector; -1: all |
| `sz_parity` | `sz="even"/"odd"` | 0 / 1; -1: both |
| `use_sz` | `sz="off"` | `False`: no Sz decomposition |
| `spin_flip`, `time_reversal` | toggles | -1 auto, 0 off, 1 require |
| `two_S` | `total_spin` | 2S; -1: no restriction |
| `only_k0`, `only_irrep`, `only_momentum`, `only_irrep_chars` | `select()` | the selection |

The verbs call `resolve`; a result's `diagnostics` start with the pairs it produced:

| code | when |
|---|---|
| `auto_spatial_skipped` | `spatial="auto"` but pynauty is not installed: a `RuntimeWarning`, and the run continues without spatial symmetry |
| `aut_capped` | H's interaction graph has more than 4096 automorphisms: no spatial symmetry is used |
| `co_group_capped` | the co-group exceeded 128 elements and the group was cut to a subgroup |

```python
sym = qed.Symmetry.auto()
notes = []
A, residues = sym.groups(H, notes)      # J1-J2 ring: the 12 translations, 1 residue (D_12)
spec = sym.resolve(H)
print(len(spec.abelian), len(spec.residues), spec.n_up, spec.two_S, notes)
```

## Sz

What H conserves along z is read from its canonical terms: every term keeps the up-spin count
(U(1)), changes it by even amounts only (Sz parity: e.g. $S^+_iS^+_j$ terms), or neither.

| `sz` | U(1) H | parity-only H | neither |
|---|---|---|---|
| `"auto"` | every Sz sector | the two parity halves | the full space |
| `n` (int) | sector n | `InvalidRequest` | `InvalidRequest` |
| `"even"` / `"odd"` | the sectors with n of that parity | that half | `InvalidRequest` |
| `"off"` | the full space | the full space | the full space |

`sz` must be a non-negative `int` at most N (a `bool` is refused) or one of the strings.
`sz="off"` on a U(1) H gives correct but larger blocks. With the spin flip, sectors are paired
(see [Spin flip](#spin-flip)); a level reports the sector its block was solved in (`Level.n_up`,
or `Level.sz_parity` for a parity half, with `n_up == -1`).

```python
even = qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz="even")).energies
parts = [qed.spectrum(H, sym=qed.Symmetry(spatial=None, sz=n)).energies
         for n in range(0, N + 1, 2)]
assert np.allclose(np.sort(even), np.sort(np.concatenate(parts)))
```

## Spatial symmetry

### Finding the group: `qed.find_symmetries`

`qed.find_symmetries(operator, *, verbose=True, clique_budget=None)` returns a `qed.Symmetries`.
H's canonical terms become a coloured graph: vertex colours from the one-body terms, a colour per
interacting pair from its couplings, a vertex per triple of sites carrying three-body terms. A
term on more than three sites does not enter the graph. pynauty enumerates the graph's
automorphisms; each is then checked exactly against every term of H (`_core.check_generators_commute`),
which removes, for example, the elements that reverse a DM vector or a scalar-chirality triangle.
The survivors are split as described below.

- A graph with more than 4096 automorphisms (a field-only, empty or all-to-all H has N!) is
  counted but not enumerated: the result carries no spatial symmetry and an `aut_capped`
  diagnostic. Pass generators of a subgroup instead.
- The search is memoised on H's term content (32 entries), so a sweep calling it repeatedly on the
  same H pays once.
- `verbose`: cost notes at Info (else Debug). `clique_budget` is deprecated and ignored.
- pynauty is needed only here.

`qed.Symmetries(abelian, residues=[], diagnostics=[])`: `abelian` lists the closed abelian group
(from `find_symmetries`: sorted, identity first) or generators of it; `residues` one
representative per coset; `diagnostics` the `(code, message)` pairs. `describe()` returns the
order of the group, the size of the abelian part, the number of residues, and one line per
diagnostic.

```python
found = qed.find_symmetries(H, verbose=False)
print(found.describe())
```

### Giving the group: permutations or a `Symmetries`

**A list.** `Symmetry(spatial=[p1, p2, ...])` takes any generating set. Each entry must be a
permutation of the N sites (else `InvalidRequest`). The group is closed (at most 4096 elements,
else `InvalidRequest`) and split automatically. The engine then checks that every element used
commutes with H: a permutation that is not a symmetry raises `InvalidRequest` rather than give a
wrong spectrum.

**A `Symmetries`.** `Symmetry(spatial=qed.Symmetries(abelian=gens, residues=perms))` is an
explicit split, checked but not re-chosen: the generators of the abelian part must commute (their
closure is capped at 4096 elements), and each residue must normalise the abelian part
($p A p^{-1} = A$), else `InvalidRequest`. A residue in A, or in the coset of an earlier residue,
is dropped. With `point_group=False` the residues are not used and not checked. Any object with an
`abelian` attribute (and optionally `residues`; numpy arrays included) is accepted.

```python
t = qed.symmetry.translation(N, 1)          # t[i] = (i - 1) mod N
m = qed.symmetry.reflection_1d(N)           # m[i] = N - 1 - i
d12 = qed.Symmetry(spatial=[t, m])          # closed to D_12, split automatically
same = qed.Symmetry(spatial=qed.Symmetries(abelian=[t], residues=[m]))
k_only = qed.Symmetry(spatial=[t], point_group=False)
```

`qed.symmetry` also provides `identity(n_sites)`, `compose(a, b)`, `power(g, k)` ($k \geq 0$),
`order(g)`, `site_swap(n_sites, a, b)`, `generate_group(generators)` (sorted, identity first),
`close_group(gens, cap=4096)` (sorted tuples; `None` without generators or above the cap) and
`split_nonabelian(symmetry_or_gens)` (`(A, residues)`, or a `str` saying why there is nothing to
split).

### The split: abelian part and residues

The engine needs the abelian part A to be **normal** in the group it is given: a residue p then
maps each momentum sector of A onto another ($\chi_k \mapsto \chi_k(p^{-1} \cdot p)$), which is
what makes stars, little groups and multiplets. For a closed group G the split is:

1. A = the largest normal abelian subgroup found by growing unions of conjugacy classes whose
   elements commute pairwise. Ties go to more fixed-point-free elements (translations), then to
   higher element orders. A may be the identity alone (S_5 on five sites has no non-trivial
   normal abelian subgroup); every other element is then a residue.
2. `residues` = one representative (the first in sorted order) per coset of A other than A.
3. When the co-group |G/A| exceeds 128, G is cut down to the largest subgroup found among the
   normalisers $N_G(A')$ of maximal abelian subgroups $A'$ (co-group at most 128), or to an
   abelian part alone, with a `co_group_capped` diagnostic.

On a lattice with periodic boundaries A is normally the translation group. A cluster whose graph
has more automorphisms than its lattice (accidental symmetry) can give a different A: the 3x3
triangular torus with nearest-neighbour bonds is the complete tripartite graph $K_{3,3,3}$ with
1296 automorphisms, and `"auto"` takes $A = \mathbb{Z}_3^3$ (27 elements) with 47 residues. Its
momenta are then not lattice momenta. To label by lattice momenta, pass the lattice's own
generators as a list, and check with `groups(H)` that A is the translation group:

```python
L = 3
idx = lambda x, y: (x % L) + L * (y % L)
xy = [(x, y) for y in range(L) for x in range(L)]
T1 = [idx(x + 1, y) for x, y in xy]
T2 = [idx(x, y + 1) for x, y in xy]
C6 = [idx(-y, x + y) for x, y in xy]                # (x, y) -> (-y, x + y)
lattice_sym = qed.Symmetry(spatial=[T1, T2, C6])
# with H the 3x3 triangular torus: A is {T1^a T2^b} (9 elements), 5 residues
```

### Momenta, stars and little groups

Inside one Sz sector (or parity half, or the full space) the engine decomposes by:

- **Momenta.** The characters $\chi_k$ of A, built exactly (integer phases modulo the group's
  exponent). A momentum sector holds the states with $U_a|\psi\rangle = \chi_k(a)|\psi\rangle$
  for every $a \in A$.
- **Stars.** The momenta related by the residues (and by time reversal, which pairs k with -k)
  form a star. One momentum per star is solved; the others are isospectral and counted in the
  multiplicity (`Level.star_size`).
- **Little groups.** The residues that fix the star representative k form the little co-group
  $P_k$ (identity included); $G_k = A \cdot P_k$ is the little group. With the spin flip engaged
  it is extended by $\{1, F\}$.

Each star is one block per irrep of its little co-group (per flip parity when the flip is
engaged: a star never mixes $F = +1$ with $F = -1$). Every irrep is a **group sector**: a
representative basis of the whole little group $G_k$, about $d\,D/|G_k|$ states for an irrep of
dimension d, D the dimension of the subspace ($\binom{N}{n_\mathrm{up}}$ in one Sz sector). The
momentum sector itself is never built for such a star;
its dimension (from Burnside's count) checks that the irrep sectors tile it. A star whose little
co-group is trivial is one **plain** block, its momentum sector. `block_stats[i]["kind"]` reads
`"group"` or `"plain"`.

### Irreps of any dimension and projective factor systems

For coset representatives $p_e$ of $P_k$, the product of two is a translation times a third:
$U_{p_e}U_{p_f} = U_{a_{ef}}U_{p_g}$. The irreps of $G_k$ that restrict to $\chi_k$ on A are the
$\omega$-projective irreps of $P_k$ with factor system $\omega(e,f) = \chi_k(a_{ef})$:

$$D(a\,p_e) = \chi_k(a)\,D(e), \qquad D(e)\,D(f) = \omega(e,f)\,D(ef).$$

When $\omega \equiv 1$ these are the ordinary irreps of $P_k$. Otherwise the $\omega$-projective
irreps are used (`ed::symmetry::decompose_projective_irreps`); a coboundary (coset
representatives that do not close into a subgroup) and a genuinely projective factor system are
handled alike. Irreps of dimension $d > 1$ are blocks of their
own: each level of such a block counts d times. The sector kernels carry irreps up to dimension 8
(`ed::matvec::kMaxIrrepDim`). With `ED_SYM_PROFILE=1` (set before `import qed`, it also makes the
import-time log level `"info"` on stderr), or at log level `"debug"`, the engine logs each
group-sector star with `|G_k0|`, its block count, the momentum sector's dimension, and
"projective factor system" when $\omega \not\equiv 1$.

```python
# 4x4 square torus, J1-J2: C4v at Gamma has the two-dimensional irrep E
L = 4
idx = lambda x, y: (x % L) + L * (y % L)
xy = [(x, y) for y in range(L) for x in range(L)]
b = qed.input.HamiltonianBuilder(L * L)
b.heisenberg([(idx(x, y), idx(x + 1, y)) for x, y in xy]
             + [(idx(x, y), idx(x, y + 1)) for x, y in xy], J=1.0)
b.heisenberg([(idx(x, y), idx(x + 1, y + 1)) for x, y in xy]
             + [(idx(x, y), idx(x + 1, y - 1)) for x, y in xy], J=0.3)
Hsq = b.to_operator()
Tx = [idx(x + 1, y) for x, y in xy]
Ty = [idx(x, y + 1) for x, y in xy]
C4 = [idx(-y, x) for x, y in xy]
mirror = [idx(y, x) for x, y in xy]
sym = qed.Symmetry(spatial=[Tx, Ty, C4, mirror], sz=8, spin_flip="off", time_reversal="off")
A, residues = sym.groups(Hsq)                        # 16 translations, 7 residues
gamma = sym.select(momentum={tuple(Tx): 0, tuple(Ty): 0})
r = qed.eigs(Hsq, 12, sym=gamma, prune=False)
print([(lv.energy, lv.irrep_dim, lv.multiplicity) for lv in r.levels])   # irrep_dim 2: E
```

### Labels on levels

`EigResult.levels` and `SpectrumResult.levels` hold one `Level` per block eigenvalue
(`energies` repeats each by its multiplicity). A level carries its block's quantum numbers:

| `Level` attribute | meaning |
|---|---|
| `energy`, `multiplicity` | the eigenvalue and how many states it stands for |
| `n_up`, `sz_parity` | the Sz sector or parity half solved (-1: none) |
| `k0`, `k_raw` | engine-internal star / abelian-irrep indices (`k0` includes the flip parity); **not** the momentum |
| `flip_parity` | 0: $F = +1$, 1: $F = -1$, -1: flip not engaged in this block |
| `irrep`, `irrep_dim` | engine-internal little-co-group irrep index (-1: plain block) and its dimension d |
| `star_size` | momenta in the star |
| `mirror` | 2: the block's image in Sz sector $N - n_\mathrm{up}$ (or in the other parity half) is folded in |
| `tr_folded`, `fold` | the antiunitary pairing (see [Time reversal](#time-reversal-k-and-theta)) |
| `block_dim` | the block's dimension |
| `momentum` | $\chi_k(a)$ for every a of the abelian group, in the order `groups(H)[0]` lists it |
| `irrep_characters` | `(residue index, chi)` pairs over the little co-group, -1 the identity |
| `vector` | index of the level's vector (-1: none) |

The multiplicity is `star_size` x `irrep_dim` (x 2 for a folded $\sigma/\sigma^*$ pair) x
`mirror` x (2S + 1 for a whole spin multiplet under `total_spin`).

Two methods turn the raw labels into physical ones (on `EigResult`, `SpectrumResult`, and on a
result restored by `qed.load_eigs`; for `ExpectResult` use its `eigs` field). The index i is into
`levels`, not `energies`.

- `result.momentum(i, translations)` returns, for each translation T (any element of the abelian
  group, else `InvalidRequest`), the fraction $\theta \in [0, 1)$ of a full turn with

  $$U_T|\psi\rangle = e^{-2\pi i\theta}|\psi\rangle ,$$

  as a `fractions.Fraction`. For `T = qed.symmetry.translation(N, 1)` a plane wave
  $\sum_r e^{ikr}|r\rangle$ ($|r\rangle$ one flipped spin at site r) has $\theta = k/2\pi$; for
  `T = [(i + 1) % N for i in range(N)]` (its inverse) $\theta = -k/2\pi \bmod 1$. The momentum reported is the star representative's:
  the other members of the star are isospectral and included in the multiplicity.
- `result.irrep_characters(i)` returns `{R: chi}` over the level's little co-group, R a coset
  representative from `groups(H)[1]` as a tuple, the identity included (its character is the
  irrep dimension). A star with a trivial co-group reports the trivial irrep `{identity: 1}`. A
  level without labels (a file saved before labels existed) gives `{}`.

The character of a residue is that of the listed permutation R: the engine builds the irreps of
$G_k$ as $D(a\,p_e) = \chi_k(a)\,D(e)$, so another representative of the same coset,
$R' = $ `compose(a, R)` with $a \in A$, has character $\chi_k(a)\,\chi(R)$. At k = 0 the two
agree; at the zone boundary they can differ in sign. For a one-dimensional irrep with a real
character, $\chi(R)$ is the eigenvalue of $U_R$ on the level's states.

`qed.symmetry.momentum_of(level, spec, translations)` and
`qed.symmetry.irrep_characters_of(level, spec, n_sites)` are the same functions on a raw `Level`
and a `Spec`.

```python
from fractions import Fraction

sym = qed.Symmetry(spatial=[t, m], sz=N // 2, time_reversal="off")
A, residues = sym.groups(H)
R = tuple(residues[0])
s = qed.spectrum(H, sym=sym)                  # levels ascending
for i, lv in enumerate(s.levels[:6]):
    print(f"E={lv.energy:.8f} x{lv.multiplicity} k={s.momentum(i, [t])[0]} "
          f"chars={s.irrep_characters(i)}")
```

## Spin flip

$F = \prod_i \sigma^x_i$ maps $n_\mathrm{up} \to N - n_\mathrm{up}$. With `spin_flip="auto"` and a
flip-symmetric H:

- **Inside a subspace F maps to itself** ($n_\mathrm{up} = N/2$, a parity half with N even, the
  full space) the abelian group is extended to $A \times \{1, F\}$: every block has a flip parity
  (`Level.flip_parity` 0 for $F = +1$, 1 for $F = -1$).
- **Between sectors** ($n_\mathrm{up} \neq N/2$), when both members of a pair are requested
  (`sz="auto"`, or `"even"` / `"odd"` with N even), F pairs sector n with $N - n$: only the member
  with $S^z \geq 0$ is solved, and its levels carry `mirror == 2` (the multiplicity is doubled).
  An explicit `sz=n` solves sector n alone, unpaired. For a parity-only H with N odd, F exchanges
  the two halves: the even half is solved, `mirror == 2`.

`spin_flip="require"` raises `InvalidRequest` when H is not flip-symmetric (a Zeeman term along z
breaks it). With `total_spin` or an explicit `sz` other than N/2, a flip-symmetric H is accepted
and the flip is used where it applies. `"off"` keeps the sectors apart. The flip commutes with
every site permutation, so it never changes a momentum label.

```python
flip = qed.Symmetry(spatial=None, sz=N // 2)
r = qed.spectrum(H, sym=flip)
print({L.flip_parity for L in r.levels})               # {0, 1}
```

## Time reversal: K and Theta

`time_reversal` uses one antiunitary map, chosen from H:

- **K**, complex conjugation in the $S^z$ basis, $K\,c|s\rangle = c^*|s\rangle$, when H is real
  in that basis (`K H K = H`).
- Otherwise **$\Theta = \prod_i (i\sigma^y_i)\,K$**, time reversal ($S^a \to -S^a$ for every a),
  when H has it:
  $\Theta\,c|s\rangle = (-1)^{n_\downarrow(s)}\,c^*\,|\bar s\rangle$, $\bar s$ the state with
  every spin reversed. $\Theta$ maps $S^z \to -S^z$, and $\Theta^2 = (-1)^N$.

What each map folds:

| map | folds | result |
|---|---|---|
| K | k with -k into one star (counted in `star_size`; `tr_folded` where no residue already relates them); inside a real momentum sector, an irrep $\sigma$ with $\sigma^*$ (same dimension) into one block (not under an `irrep` selection) | up to 2x fewer blocks; `Level.fold == "K"` |
| $\Theta$ | inside a subspace it maps to itself ($S^z = 0$, a parity half with N even, the full space): the same folds as K; not beside the flip half ($\Theta F \Theta^{-1} = (-1)^N F$) | `Level.fold == "theta"` |
| $\Theta$ | between Sz sectors, when the flip does not pair them: $(S^z, k)$ with $(-S^z, -k)$ | `mirror == 2`, `Level.fold == "theta"` |

For odd N, $\Theta^2 = -1$ (Kramers): with Sz conserved and every Sz sector requested
(`sz="auto"`), every sector is paired with its $-S^z$ partner (by $\Theta$, or by the flip when H
has it) and every level's multiplicity is even. An explicit `sz=n` solves one member of each pair.

$\Theta$ is not used under a [selection](#selecting-sectors-select): it would add sectors the
selection did not name. K is not used under an `irrep_character` selection: it would fold
$\sigma^*$ into $\sigma$'s block.

`time_reversal="require"` raises `InvalidRequest` when H is invariant under neither K nor
$\Theta$ (e.g. a scalar chirality $S_i\cdot(S_j\times S_k)$, odd under $\Theta$ and not real); it
asserts the symmetry and does not force a fold where none applies.

**On results.** `EigResult.time_reversal` and `SpectrumResult.time_reversal` name the map that
folded any level: `"K"`, `"theta"` or `None`. `Level.fold` names each level's pairing (`"K"`,
`"theta"`, or `None`), `Level.tr_folded` whether its block's states come with antiunitary images
the block does not hold. `vectors()` adds the images to the multiplet, and `expect` and thermal
`observables` average $\langle O\rangle$ with the image's, $\langle A v|O|A v\rangle =
\overline{\langle v|A^{-1} O A|v\rangle}$, so a time-reversal-odd operator averages correctly.
Saved results carry `level_fold`.

```python
n = 7
b = qed.input.HamiltonianBuilder(n)
bonds = [(i, (i + 1) % n) for i in range(n)]
b.heisenberg(bonds, J=1.0)
b.dm(bonds, [[0.0, 0.0, 0.3]] * n)          # D_z: not real, but Theta-invariant
Hdm = b.to_operator()
tn = qed.symmetry.translation(n, 1)
on = qed.spectrum(Hdm, sym=qed.Symmetry(spatial=[tn], point_group=False))
off = qed.spectrum(Hdm, sym=qed.Symmetry(spatial=[tn], point_group=False, time_reversal="off"))
assert (on.time_reversal, off.time_reversal) == ("theta", None)
assert np.allclose(on.energies, off.energies) and len(on.levels) < len(off.levels)
assert all(L.multiplicity % 2 == 0 for L in on.levels)       # odd N: Kramers pairs
```

## Total spin

`total_spin=S` restricts to total spin S. H must be SU(2) symmetric, up to a uniform field along
z ($H - h S^z_\mathrm{tot}$ SU(2) invariant for some h, including 0).

- **Without the field** each multiplet is solved once, at its $S^z = +S$ member
  ($n_\mathrm{up} = (N + 2S)/2$), and each level counts 2S + 1 times. `sz` may name only that
  sector.
- **With the field** each $S^z$ member is a level of its own, solved in its own Sz sector
  (multiplicity 1 from the multiplet), and `sz` / `"even"` / `"odd"` pick members.

Refused, with `InvalidRequest`: a `total_spin` that is not a non-negative multiple of 1/2; an S
that does not exist for N (2S > N, or N - 2S odd); an H that is not SU(2) symmetric up to a
uniform z field; an `sz` that disagrees with the $S^z = S$ sector (no field), or an `sz` / parity
that names no member $S^z \in \{-S, \dots, S\}$ (field). A selection whose blocks hold no state of
spin S raises `EmptySelection`.

The spin-flip, point-group and momentum structure is kept. No verb projects H: `eigs` starts its
Krylov lanes from random valence-bond states of spin S in the block, certifies each eigenpair by
$\|(S^2 - S(S+1))\psi\|$, and falls back to a penalty $H + \mu f(S^2)$ when an off-tower level
displaced a tower level; `spectrum` and exact `thermal` diagonalise H on the tower's states
($Q^\dagger H Q$); FTLM, OFTLM and finite-T `dynamics` start from $P_S$ of a Gaussian vector; mTPQ
re-projects its iterate. Tower dimensions come from Burnside's count.

Operators under `total_spin` (`expect`, thermal `observables`): without the field an O that is
not SU(2) invariant enters through its SU(2)-scalar part (its average over all spin rotations;
$S^z_iS^z_j$ enters as $S_i\cdot S_j/3$), whose expectation is the multiplet average; this is
computed exactly on terms of up to five sites (more raise `Unsupported`). With the field, O enters
as it is. `EigResult.vectors(basis="sz", n_up=n)` returns the members at any $S^z$ in
$-S \dots S$.

```python
s1 = qed.Symmetry(spatial=None, total_spin=1)
r = qed.eigs(H, 6, sym=s1)                     # S = 1 levels, each x3
assert all(L.multiplicity % 3 == 0 for L in r.levels)

bf = qed.input.HamiltonianBuilder(N)
bf.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
bf.zeeman((0.0, 0.0, 0.2))                     # -h S^z_tot: SU(2) up to a uniform field
Hh = bf.to_operator()
split = qed.spectrum(Hh, sym=s1)               # every member a level of its own
top = qed.spectrum(Hh, sym=qed.Symmetry(spatial=None, total_spin=1, sz=N // 2 + 1))   # S^z = +1
```

## Selecting sectors: `select()`

`Symmetry.select(*, sz=None, k0=None, irrep=None, momentum=None, irrep_character=None)` returns
the same symmetry restricted to some sectors. The group, folds and labels are unchanged.

| argument | keeps |
|---|---|
| `sz` | replaces `sz` (an int, `"even"`, `"odd"`, ...) |
| `momentum` | `{T: theta}`: the stars holding a momentum with $U_T\psi = e^{-2\pi i\theta}\psi$ for each given T (an element of the abelian group as a tuple; theta a fraction of a full turn, as `momentum()` reports it). A list of such dicts keeps any of them. |
| `irrep_character` | `{R: chi}`: the little-co-group irreps with character chi on each given R (a coset representative from `groups(H)[1]`; the identity names the irrep dimension). A list keeps any of them. Blocks whose little co-group lacks some R are dropped. |
| `k0` | the engine's star indices, as on `Level.k0` |
| `irrep` | the engine's irrep indices, as on `Level.irrep`; names group-sector blocks only (a plain block is never returned) |

- A level answers for its whole star, so a selected momentum also counts its star partners, and
  `momentum()` reports the star representative. For one momentum exactly, use
  `point_group=False` and `time_reversal="off"` (K pairs k with -k).
- A star with a trivial co-group carries the trivial irrep (character 1 on the identity);
  `irrep_character={identity: d}` therefore partitions the spectrum by irrep dimension.
- `momentum` and `irrep_character` are resolved by `resolve()`: a T that is not in the abelian
  group, or an R that is not a listed coset representative, raises `InvalidRequest`.
- Under `irrep_character` time reversal is not folded, so each irrep is its own block. $\Theta$ is
  not used under any selection.

A selection (`momentum`, `irrep_character`, `k0`, `irrep`) that matches no block raises
`qed.errors.EmptySelection` (a subclass of `InvalidRequest` and `ValueError`) in `eigs`,
`spectrum`, `thermal` and `expect`, and a `momentum` selection does so in `dynamics` (which refuses
the other three with `Unsupported`), instead of answering for an empty space: a momentum no star has, a residue that does not fix the
selected momentum, an irrep dimension no co-group has. Under `total_spin` it also raises when the
selected blocks hold no spin-S state. `thermal` and finite-T `dynamics` under any restriction
(one Sz sector or parity, a selection, a total spin) add a `("restricted_ensemble", ...)`
diagnostic: the result is the canonical ensemble of that part.

```python
from fractions import Fraction

k_sym = qed.Symmetry(spatial=[t], point_group=False, sz=N // 2, time_reversal="off")
full = qed.spectrum(H, sym=k_sym)
parts = [qed.spectrum(H, sym=k_sym.select(momentum={tuple(t): Fraction(j, N)})).energies
         for j in range(N)]
assert np.allclose(np.sort(np.concatenate(parts)), np.sort(full.energies))

try:
    qed.spectrum(H, sym=k_sym.select(momentum={tuple(t): 0.3}))       # 0.3 * 12 is no integer
except qed.errors.EmptySelection as e:
    print(e)

pg = qed.Symmetry(spatial=[t, m], sz=N // 2, time_reversal="off")
R = tuple(pg.groups(H)[1][0])
gamma = pg.select(momentum={tuple(t): 0})
even = qed.spectrum(H, sym=gamma.select(irrep_character={R: 1.0}))
odd = qed.spectrum(H, sym=gamma.select(irrep_character={R: -1.0}))
```

## What each verb does with each symmetry

| verb | momenta, point group | Sz, flip, time reversal | total_spin | selections |
|---|---|---|---|---|
| `eigs` | blocks per star and irrep | subspaces, flip parity or mirror, K / $\Theta$ folds | yes | all |
| `spectrum` | as `eigs`; every block dense | as `eigs` | yes ($Q^\dagger H Q$) | all |
| `thermal` | as `eigs`; blocks combined with their multiplicities | as `eigs`; `M`, `chi` from the Sz decomposition | yes, every method | all; `restricted_ensemble` |
| `expect` | through `eigs(..., vectors=True)` | as `eigs` | yes (SU(2)-scalar part) | all |
| `dynamics` | see below | see below | yes | `sz`, `momentum`; `k0`, `irrep`, `irrep_character` raise `Unsupported` |

**`eigs`, `spectrum`, `thermal`, `expect`** use every symmetry resolved. Operators passed to
`expect`, `EigResult.expect` and thermal `observables` may break any symmetry: each block uses O
averaged over the symmetries it resolves (the group, the flip where it folds or projects, the
antiunitary image where it folds), which has the same expectation in the level's multiplet and
the same thermal trace. `EigResult.vectors()` completes each degenerate multiplet by applying
the residues, the antiunitary map, the flip or $\Theta$ mirror and, under `total_spin`, total
$S^-$, returning orthonormal eigenvectors; full-basis vectors are limited to N <= 34.
`EigResult.matrix_element(O, i, j)` works between the vectors the solver returned for two levels,
for any O. `ThermalResult.M` and `chi` are `None` without an Sz decomposition
(`Symmetry.none()`, `sz="off"`, or an H that does not conserve Sz).

**`dynamics`.** A probe O is in general not invariant under the point group, the flip or time
reversal (a Fourier mode $O_Q$ maps to $O_{pQ}$), so the probes act between momentum sectors of
A, and every other symmetry is used where it is exact:

- *T = 0.* The ground manifold (every level within `degeneracy_tol` times $s_H$ of $E_0$, $s_H$
  the sum of |c| over H's terms) is solved on the folded blocks (point group, flip, time
  reversal) and each level is expanded into its whole multiplet in momentum sectors. Under a
  `momentum` selection or `device="gpu"` it is solved on momentum sectors instead. In each target
  momentum sector O reaches, a continued fraction runs on each one-dimensional irrep block of the
  little co-group (and of the flip parity where the target subspace is its own flip image), and
  the part of $O|\psi\rangle$ in irreps of dimension > 1 on the whole momentum sector; time
  reversal is not folded there.
- *T > 0.* The sources are momentum sectors (finite-temperature Lanczos, `samples` per sector).
  Sources related by a symmetry U of H contribute alike: when $UAU^\dagger = c_A A$ and
  $UBU^\dagger = c_B B$, the source $Uk$ gives the same Z and $\bar c_A c_B$ times the S of k.
  Each probe folds by the symmetries it follows: the flip (H flip-symmetric, `spin_flip` not
  `"off"`), whose mirror sector $N - n$ is then not run for that probe; a residue whose factor
  $\bar c_A c_B$ is 1 (one momentum per orbit, weighted by the orbit size). A probe's result does
  not depend on the other probes of the call. No folding under `total_spin`; no point-group
  folding under a `momentum` selection.
- `spin_flip="require"` / `time_reversal="require"` check that H has the symmetry (K or
  $\Theta$).
- Under `total_spin` without a field, the other members of each ground multiplet follow by total
  $S^-$; at T > 0 the tower members at every $S^z$ are sources.

**Devices.** The symmetry decomposition is the same on every device. A sector of a
one-dimensional irrep builds its reduced CSR on the device within `ED_GPU_CSR_BUDGET_GIB` (unset:
half the free device memory; 0, or over the budget: the device walk). Under `device="gpu"`
(strict) the sectors of irreps of dimension > 1 run on the device through their reduced CSR,
built on the host and uploaded; a block whose CSR does not fit both its CSR budget and
`ED_GPU_CSR_BUDGET_GIB` raises `qed.errors.DeviceUnsupported`, naming the block. Under `"gpu"`,
`eigs` still solves densely on the host a block of at most 32 states or with $2k \geq$ its
dimension, and every block within its dense crossover (`dense_max_dim`). Dense spectra of many blocks are solved on the
device in batches of up to 2 GiB of matrices (`ED_GPU_DENSE_BATCH_GIB`; at most a quarter of the
free device memory and of the RAM).

## Representatives: plain order and sublattice key order

A symmetry sector stores one representative state per orbit of its group (the site permutations
of the abelian group or of a little group, with their flip images when the flip is engaged).
Which member represents an orbit is a rule:

- **Plain order**: the member least as an integer.
- **Sublattice key order**: when every element of the group maps the blocks of a block system
  onto blocks (the sublattices of a superlattice, under the lattice's translations and point
  group), the sites are renumbered in a key order where block j holds key bits
  $[N - (j+1)L,\ N - jL)$, its sites ascending; the representative is the member least in that
  key. The leading L key bits of an image then come from one block of the state through a table,
  and only the elements that reach the least leading block need a full image (Wietek and
  Läuchli, Phys. Rev. E 98, 033309 (2018); `include/ed/basis/sublattice_code.h`). Block systems
  of 2 to 8 blocks of equal size, at most 16 sites each, are used (the one with the largest
  blocks).

`ED_SYM_SUBLATTICE` chooses:

| value | rule |
|---|---|
| unset | key order for groups on $N \geq 24$ sites that have a block system and at least 16 distinct site permutations for a verb that runs on the device (`device="gpu"`, or `"auto"` with a visible GPU) and for `eigs`, `spectrum` and exact `thermal`; at least 64 for host sampled `thermal` and host `dynamics`; plain order otherwise |
| `1` | key order whenever a block system exists (groups of 2 to 65535 elements, $4 \leq N \leq 64$) |
| `0` | plain order always |

The rule is decided per group (the abelian group of a momentum sector, the little group of a
group sector), from its distinct site permutations (a group with or without its flip half gets
the same key order) and, when the variable is unset, the calling verb (its device, and whether it
applies H on the host thousands of times per block), at the moment the group is compiled. One run can therefore hold sectors of both
kinds; `qed.debug_env("ED_SYM_SUBLATTICE")` shows the variable's value.

**Why it never changes physics.** The orbits, their norms and the block each orbit belongs to do
not depend on which member represents it. Replacing a representative r by $g r$ changes that
orbit's symmetrised basis states by a phase (one-dimensional irreps) or a unitary among its
states (irreps of dimension > 1): the block's matrix changes by a unitary change of basis.
Eigenvalues, thermodynamics, S(omega), expectation values and full-basis eigenvectors (up to a
phase, and the choice of basis inside a degenerate level) are the same under either rule. The
tests (`tests/python/test_sublattice.py`, forced on against forced off, on the host and the
device) compare the `eigs` energies, the Rayleigh quotients of the returned full-basis vectors,
`spectrum`, exact `thermal` E and C and `expect` values to $10^{-10}$ absolute, and T = 0
`dynamics` to $10^{-8}$. Sampled methods (FTLM, OFTLM, mTPQ, finite-T `dynamics`) draw
their random vectors in the representative basis, so a run at a fixed `seed` draws different
vectors under the two rules: results move at sampling-noise level.

**A result keeps its rule.** Every sector records the key order it was computed with. A result
held in memory keeps using it whatever `ED_SYM_SUBLATTICE` says later. `EigResult.save` writes it
per stored basis (`basis<i>_sublattice`: the key order's fingerprint, 0 for plain order);
`qed.load_eigs` rebuilds that key order from the stored group regardless of the environment,
checks a sample of the stored representatives against it, and refuses a file whose key order this
build does not reproduce. A file without the field predates the key order and holds plain-order
representatives.

**Cost.** Key-order representatives make the canonicalisation cheaper and a host sparse apply
slower (rows sorted by state spread neighbours over the key order). Measured
(`sublattice_code.h`): the CSR build 5x, the orbit table 2.5x and the device gather 1.8x faster;
a host apply +11% at the 432-element tri36 group and +50% at the 30 translations of a 30-site
chain momentum sector (chain28 FTLM +18% wall), nothing measurable on the device. A verb that
applies H a few times per block gains even on a chain (chain30 `eigs` 1.33 -> 1.21 s; tri30
`eigs` 74.8 -> 54.9 s). So the default engages the key order from 16 permutations on the device
and for `eigs`, `spectrum` and exact `thermal`, and from 64 for host sampling and host `dynamics`
(2D space groups; chains, whose bonds join neighbouring bits, stay plain there).

```bash
ED_SYM_SUBLATTICE=0 python run.py      # plain order everywhere
ED_SYM_SUBLATTICE=1 python run.py      # key order wherever the group has a block system
```

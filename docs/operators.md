# Operators

Every Hamiltonian, observable and dynamics probe in `qed` is a `qed.Operator`: a sum of
products of spin-1/2 operators on `num_sites` sites, with complex coefficients. This page
covers how to build one (records, products and the algebra, the `HamiltonianBuilder`
shortcuts and lattices, the `qed.dssf` probe builders) and what the verbs do with an operator
that is not the Hamiltonian. [Dynamics](dynamics.md) covers `qed.dynamics` itself, and
[Symmetry](symmetry.md) the `qed.Symmetry` options referred to here.

## Conventions

- Sites are `0 .. N-1`, `N = num_sites`, with `1 <= N <= 63`.
- A basis state is an integer `s`; bit `i` is site `i`, and **a set bit is spin up**. `n_up`
  counts up spins, and `qed.Symmetry(sz=n)` selects $S^z_\mathrm{tot} = n - N/2$.
- The site operators are spin-1/2 operators, not Pauli matrices: $S^z = \pm 1/2$,
  $S^\pm = S^x \pm i S^y$, $S^x = (S^+ + S^-)/2$, $S^y = (S^+ - S^-)/(2i)$.
- A product is written left to right and applied right to left: in
  $c\, O_0(i_0)\, O_1(i_1) \cdots O_{k-1}(i_{k-1})$ the last factor acts first.
- Energies, frequencies and temperatures ($k_B = 1$) are in the units of $H$.

## `qed.Operator`

### Construction and records

```python
O = qed.Operator(num_sites)          # the zero operator on num_sites sites
O.num_sites                          # N
O.dimension                          # 2**N
```

`num_sites >= 64` raises `qed.errors.Unsupported`. An operator on 0 sites can be constructed but not
used: its canonical form, and every verb, need `1 <= num_sites <= 63`.

Terms are appended in place, one record per call, with integer op codes `qed.OP_SPLUS = 0`,
`qed.OP_SMINUS = 1`, `qed.OP_SZ = 2`:

| method | appends |
|---|---|
| `add_one_body(op_type, site, coeff)` | $c\, O(\mathrm{site})$ |
| `add_two_body(op_type_1, site_1, op_type_2, site_2, coeff)` | $c\, O_1(\mathrm{site}_1)\, O_2(\mathrm{site}_2)$, $O_2$ acting first |
| `add_three_body(op_type_1, site_1, op_type_2, site_2, op_type_3, site_3, coeff)` | $c\, O_1 O_2 O_3$ on three sites |

An op code outside 0..2 raises `qed.errors.InvalidRequest` (a `ValueError`); a site
`>= num_sites` raises `IndexError`. Sites may repeat and the product is then reduced exactly:
`add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 0, 1.0)` is $\lvert\uparrow\rangle\langle\uparrow\rvert_0 = 1/2 + S^z_0$.
In a three-body record with a repeated site the first factor acts first (the opposite of
`add_two_body`); on three distinct sites the order is immaterial. Write same-site products with
`Operator.product`, where the order is unambiguous.

A call appends without merging and costs O(1). What the operator *is* -- for every verb,
every symmetry check and `equals` -- is its canonical form (below), built on first use and
rebuilt after the records change.

### Products: `Operator.product`

```python
qed.Operator.product(num_sites, ops, sites, coeff=1)
```

returns $c\, O_0(\mathrm{sites}[0])\, O_1(\mathrm{sites}[1]) \cdots$, the last factor acting
first, with one character of `ops` per entry of `sites`:

| letter | operator |
|---|---|
| `+` | $S^+$ |
| `-` | $S^-$ |
| `z` | $S^z$ |
| `x` | $S^x = (S^+ + S^-)/2$ |
| `y` | $S^y = (S^+ - S^-)/(2i)$ |
| `u` | $\lvert\uparrow\rangle\langle\uparrow\rvert = 1/2 + S^z$ |
| `d` | $\lvert\downarrow\rangle\langle\downarrow\rvert = 1/2 - S^z$ |
| `I` | identity |

Sites may repeat; the spin-1/2 algebra is applied exactly:

```python
P, N = qed.Operator.product, 4
P(N, "++", [2, 2]).equals(qed.Operator(N))              # S+ S+ = 0
P(N, "zz", [3, 3], 4.0).equals(P(N, "I", [0]))           # (S^z)^2 = 1/4
P(N, "+-", [1, 1]).equals(P(N, "u", [1]))                # S+ S- = |up><up|
P(N, "xy", [0, 0]).equals(P(N, "z", [0], 0.5j))          # S^x S^y = (i/2) S^z
```

`ops` and `sites` of different lengths, a site outside `0 .. num_sites-1` and an unknown letter
raise `InvalidRequest`. The identity is `P(N, "I", [i])` for any valid `i` (or `P(N, "", [])`).

### The algebra

| expression | result |
|---|---|
| `A + B`, `A - B`, `-A` | sum, difference, negation |
| `c * A`, `A * c` | scalar multiple; `c` a Python `int`, `float` or `complex` |
| `A / c` | `A` times `1/c`; `c == 0` raises `qed.errors.InvalidRequest` (a `ValueError`) |
| `A @ B` | operator product, `B` acting first |
| `A.adjoint()` | $A^\dagger$ |
| `A.copy()`, `copy.copy(A)`, `copy.deepcopy(A)` | an independent copy |

Every result is a new `Operator`, computed exactly on the canonical terms of the operands (no
matrix is formed). The operands must act on the same number of sites (`InvalidRequest`
otherwise). Notes:

- `*` takes a scalar only; `A * B` between two operators is a `TypeError`. The product is
  `A @ B`, a commutator `A @ B - B @ A`.
- `==` is object identity. Compare operators with `equals`.
- Python's `sum` starts from the integer 0; pass a zero operator as the start:
  `sum(terms, qed.Operator(N))`.
- A sum costs of order the number of terms of its operands and `A @ B` of order the product
  of their numbers of terms, so building an operator of $T$ terms one `+` at a time costs of
  order $T^2$. For a Hamiltonian of many terms, append
  records in place (`add_*_body`, `HamiltonianBuilder`) and use the algebra for the pieces that
  need it.
- After an operation the records hold the canonical terms: a term on $k \le 3$ sites as one
  $k$-body record, the identity as the record $4c\, S^z_0 S^z_0$, and a term on four or more
  sites outside the records (below).

### Comparing, reading and transforming

| method | returns |
|---|---|
| `A.equals(other, rtol=1e-10)` | `True` when the canonical terms agree, each within `rtol` times the largest coefficient of either. Different `num_sites`: `False`. |
| `A.is_hermitian(rtol=1e-10)` | `A.equals(A.adjoint())` in the same sense; `False` when a coefficient is NaN or infinite |
| `A.terms()` | the canonical terms as a list of `(coeff, ops, sites)` |
| `A.image(perm, flip=False)` | $U A U^\dagger$ for the site permutation `perm`, then the global spin flip $\prod_i \sigma^x_i$ when `flip` |
| `A.apply(vec)` | $A\lvert v\rangle$ on the full $2^N$ space |

**`terms()`** is the unique expansion of the operator: `ops` is a string over `+ - z` on
ascending `sites` (a tuple), and the identity has `ops == ""`. The operator equals
`sum(Operator.product(N, ops, list(sites), coeff) for coeff, ops, sites in A.terms())`
however it was written: the Cartesian and the ladder form of $\vec S_0 \cdot \vec S_1$ give
the same terms, `x`, `y`, `u`, `d` and `I` are re-expressed over `+ - z`, and records that
cancel exactly leave nothing. Coefficients are Python `complex` and refer to $S^z$ (not
$\sigma^z$): `P(N, "zz", [0, 1]).terms()` is `[((1+0j), "zz", (0, 1))]`.

**`image(perm, flip)`**: site `i` of the image carries what site `perm[i]` carried, so
`perm[i] = (i + 1) % N` moves an operator on site 1 to site 0. `perm` must be a permutation of
`0 .. N-1` (`InvalidRequest` otherwise). An operator is invariant under a permutation iff it is
invariant under the inverse, so symmetry checks do not depend on the direction:

```python
T = [(i + 1) % N for i in range(N)]
H.image(T).equals(H)                         # translation invariant
H.image(list(range(N)), flip=True).equals(H) # spin-flip invariant
Sz = sum((qed.Operator.product(N, "z", [i]) for i in range(N)), qed.Operator(N))
C = H @ Sz - Sz @ H                          # [H, S^z_tot], term by term
max((abs(c) for c, _, _ in C.terms()), default=0.0)
```

`examples/05_operator_algebra.py` builds a J1-J2 chain
with a four-spin term this way, checks its symmetries and measures $\langle D^2\rangle$ for the
dimer order parameter $D = \sum_i (-1)^i\, \vec S_i \cdot \vec S_{i+1} / N$.

**`apply(vec)`** takes a 1-D array of length $2^N$ (converted to complex128) and returns
$A\lvert v\rangle$; another length or dimension raises `InvalidRequest`. It walks the
canonical terms, or a full-space CSR assembled at the first apply after the records change:
by default when $2^N \le 2^{20}$ (`ED_CSR_DIM_MAX` sets the bound; `ED_CSR_FORCE=1` always
assembles it, `=0` never). It is for small $N$ and for
checks (a dense matrix is `np.column_stack([A.apply(e) for e in np.eye(2**N, dtype=complex)])`).

**Record readers.** `iter_one_body_terms()`, `iter_two_body_terms()`,
`iter_three_body_terms()` and `transform_tuples()` (one- and two-body records only) list the
records as stored -- unmerged, in insertion order. No record holds a term on four or more
sites, so they raise `qed.errors.Unsupported` for an operator that has one. `terms()` is the
representation to read.

### Terms on four or more sites

A canonical term on four or more sites (from `Operator.product` with four distinct sites, a
product such as `bond(0, 1) @ bond(2, 3)`, or the builder's `ring_exchange` and `ss_ss`) is
stored as a canonical term, not as a record; `terms()` lists it, and the record readers refuse
the operator (`Unsupported`).
Such terms work

- in $H$, in every verb and on every lane, CPU and GPU;
- in `Operator.apply`;
- in every observable: `qed.expect`, `EigResult.expect`, `EigResult.matrix_element`,
  `qed.thermal(observables=...)` and the probes of `qed.dynamics`.

Two limits:

- Under `Symmetry(total_spin=S)` with an SU(2)-symmetric $H$, `expect` and thermal
  `observables` replace an observable that is not SU(2) invariant by its SU(2)-scalar part
  (below). That part is computed exactly on terms of at most 5 sites; such an observable with
  any longer term raises `qed.errors.Unsupported` there, even when that term is itself SU(2)
  invariant. An SU(2)-invariant observable is used as it is, whatever its terms.
- Symmetry discovery (`Symmetry(spatial="auto")`, `qed.find_symmetries`) builds its
  interaction graph from the terms on at most three sites and checks every candidate
  permutation against all terms. When those terms leave more than 4096 graph automorphisms
  (an $H$ made only of four-site terms, for instance) no spatial symmetry is used, with an
  `("aut_capped", ...)` diagnostic; pass `spatial=` explicitly.

## `qed.input.HamiltonianBuilder`

A fluent list of records, materialised as a `qed.Operator`:

```python
lat = qed.input.lattice.chain(8, pbc=True)
H = (qed.input.HamiltonianBuilder(lat.num_sites)
     .heisenberg(lat.nn_pairs(), J=1.0)
     .zeeman((0.0, 0.0, 0.2))
     .to_operator())
```

`HamiltonianBuilder(num_sites)` needs `1 <= num_sites <= 63` (`qed.errors.InvalidRequest`, a
`ValueError`). Every term method and `clear()` return the builder. The refusals below are
`InvalidRequest` unless they say otherwise. Arguments:

- `bonds`: an iterable of `(i, j)` pairs (tuples or lists), as `Lattice.nn_pairs()` returns
  them. `qed.input.Bond` objects are not accepted; pass `(b.i, b.j)`. A bond with `i == j` is
  skipped.
- Sites are non-negative integers (`TypeError` for a negative or non-integer site);
  a site `>= num_sites` raises `InvalidRequest`.
- Every bond method checks the whole call before it adds anything: a refused call leaves the
  builder unchanged.

### Methods

| method | adds |
|---|---|
| `add_one_body(op, site, coeff)` | $c\, \mathrm{op}(\mathrm{site})$; `op` a `qed.input.Op` (`Op.Sp`, `Op.Sm`, `Op.Sz`), `TypeError` otherwise |
| `add_two_body(op_i, site_i, op_j, site_j, coeff)` | $c\, \mathrm{op}_i(\mathrm{site}_i)\, \mathrm{op}_j(\mathrm{site}_j)$, the right factor acting first |
| `add_three_body(op_i, site_i, op_j, site_j, op_k, site_k, coeff)` | $c\, \mathrm{op}_i\, \mathrm{op}_j\, \mathrm{op}_k$ |
| `heisenberg(bonds, J=1.0)` | $J\, \vec S_i \cdot \vec S_j$ |
| `xxz(bonds, Jxy, Jz)` | $J_{xy}(S^x_i S^x_j + S^y_i S^y_j) + J_z S^z_i S^z_j$ |
| `xyz(bonds, Jxx, Jyy, Jzz)` | $J_{xx} S^x_i S^x_j + J_{yy} S^y_i S^y_j + J_{zz} S^z_i S^z_j$ |
| `ising(bonds, J=1.0)` | $J\, S^z_i S^z_j$ |
| `transverse_field_ising(bonds, J, h)` | $-J \sum_{\mathrm{bonds}} S^z_i S^z_j - h \sum_{i=0}^{N-1} S^x_i$ (the field on every site) |
| `kitaev(bonds, bond_axis, K=1.0)` | $K\, S^a_i S^a_j$ with $a$ = `bond_axis[b]` in {0: x, 1: y, 2: z} for bond `b` |
| `dm(bonds, D_per_bond)` | $\vec D_b \cdot (\vec S_i \times \vec S_j)$ for bond `b = (i, j)` as oriented |
| `zeeman(h)` | $-\vec h \cdot \vec S_i$ on every site; `h` a tuple `(hx, hy, hz)` |
| `zeeman_per_site(h_per_site)` | $-\vec h_i \cdot \vec S_i$, one 3-vector per site |
| `on_site_field(h_z)` | $+h_z S^z_i$ on every site (the opposite sign to `zeeman`) |
| `ring_exchange(plaquettes, K=1.0)` | $K(P + P^{-1})$ per plaquette |
| `ss_ss(pairs, K=1.0)` | $\tfrac{K}{2}\{\vec S_i \cdot \vec S_j, \vec S_k \cdot \vec S_l\}$ per pair of bonds |
| `pyrochlore_non_kramers(lattice, Jxx, Jyy, Jzz, include_isotropic=True)` | see below |

Details and refusals:

- `xxz` adds no $S^zS^z$ record when `Jz == 0`; `xyz` adds only the records with nonzero
  coefficients. `heisenberg(bonds, J)` is `xxz(bonds, J, J)`.
- `kitaev`: `bonds` and `bond_axis` of different lengths, or an axis outside {0, 1, 2} on any
  bond (a skipped self-bond included), raise `InvalidRequest`; an axis that is not an integer
  raises `TypeError`. On `lattice.honeycomb` the colours are the bond types:
  `b.kitaev(lat.nn_pairs(), [bd.bond_type for bd in lat.nn_bonds], K)`.
- `dm`: one 3-vector per bond (`InvalidRequest` otherwise); the orientation of each bond matters,
  and `Lattice.nn_pairs()` keeps the orientation the generator chose (below).
- `zeeman` raises `TypeError` unless `h` is a tuple and `InvalidRequest` unless it has 3 entries;
  `zeeman_per_site` needs exactly `num_sites` vectors.
- `ring_exchange(plaquettes, K)`: each plaquette is four distinct sites `(a, b, c, d)`
  (`InvalidRequest` for another length or a repeated site). $P$ is the cyclic exchange
  $a \to b \to c \to d \to a$ of the four spins -- site $b$ takes the spin of site $a$, and so
  on -- equal to $P_{ab} P_{bc} P_{cd}$ with $P_{ij} = 1/2 + 2\, \vec S_i \cdot \vec S_j$,
  constants included. It is stored as its $16 + 16$ matrix elements and becomes canonical
  terms on up to four sites. With `K == 0` it adds nothing and checks only that each
  plaquette has four non-negative integer sites.
- `ss_ss(pairs, K)`: each entry is `((i, j), (k, l))`; a bond on one site raises `InvalidRequest`.
  For disjoint bonds the term is the product $(\vec S_i \cdot \vec S_j)(\vec S_k \cdot \vec S_l)$;
  for bonds that share a site it is the Hermitian part, the anticommutator over 2. The same
  bond twice gives $(\vec S_i \cdot \vec S_j)^2$, reduced exactly.
- `pyrochlore_non_kramers(lattice, Jxx, Jyy, Jzz, include_isotropic=True)` reads the
  nearest-neighbour bonds and sublattice labels (0..3) of `lattice`, whose `num_sites` must
  equal the builder's (`InvalidRequest`). It adds `xxz(nn, (Jxx + Jyy)/2, Jzz)` when
  `include_isotropic`, plus
  $J_{\pm\pm} \sum_{\langle ij\rangle} (\gamma_{ij} S^-_i S^-_j + \gamma_{ij}^* S^+_i S^+_j)$ with
  $J_{\pm\pm} = (J_{xx} - J_{yy})/4$, $\gamma_{01} = \gamma_{23} = 1$,
  $\gamma_{02} = \gamma_{13} = \zeta$, $\gamma_{03} = \gamma_{12} = \zeta^2$,
  $\zeta = e^{2\pi i/3}$ ($\gamma$ symmetric). Labels outside 0..3, a bond inside one
  sublattice, or `include_isotropic=False` with `Jzz != 0` raise `InvalidRequest`.

### Output

| member | |
|---|---|
| `to_operator()` | a new `qed.Operator` with the accumulated terms |
| `emit_into(operator)` | appends them to an existing `Operator` in place (`InvalidRequest` if `num_sites` differ); returns `None` |
| `num_sites` | the number of sites |
| `len(b)` | the number of records (`heisenberg` with `J != 0`: 3 per bond; `ring_exchange`: 32 per plaquette; `ss_ss`: 9 per disjoint pair, 18 per pair sharing a site) |
| `l1_norm` | $\sum \lvert c\rvert$ over the records as stored (not over the canonical terms) |
| `clear()` | removes every record |

One-, two- and three-body records go in through `add_*_body`, in that order, then the
four-site records of `ring_exchange` and `ss_ss` through `Operator.product`. The order affects
only the record list; the operator is its canonical form.

### Lattices: `qed.input.lattice`

| generator | |
|---|---|
| `chain(length, pbc=False)` | 1D chain along $\hat x$ |
| `square(Lx, Ly, pbc=False)` | square lattice |
| `triangular(Lx, Ly, pbc=False)` | $a_1 = (1, 0, 0)$, $a_2 = (1/2, \sqrt3/2, 0)$ |
| `honeycomb(Lx, Ly, pbc=False)` | 2-site basis (A, B); `Bond.bond_type` in {0, 1, 2} = Kitaev x, y, z |
| `kagome(Lx, Ly, pbc=False)` | 3-site basis A = (0, 0), B = (1/2, 0), C = (1/4, $\sqrt3/4$) |
| `pyrochlore(Lx, Ly, Lz, pbc=False)` | 4-site basis, $4 L_x L_y L_z$ sites |
| `from_neighbor_lists(positions, nn_pairs, sublattice=[])` | from explicit positions and edges |
| `from_cluster_file(path)` | a `cluster.txt`-style file |

A `qed.input.Lattice` has `num_sites`, `positions` (one `[x, y, z]` per site), `sublattice`, `nn_bonds`,
`nnn_bonds`, `nnnn_bonds` (lists of `qed.input.Bond` with `i`, `j`, `bond_type`),
`lattice_vectors`, `pbc`, `label`, and the methods `nn_pairs()`, `nnn_pairs()`,
`nnnn_pairs()` (lists of `(i, j)`) and `all_sites()`.

- Sites are numbered cell by cell, the first direction fastest (pyrochlore: the last), and by
  basis site within a cell.
- Nearest-neighbour bonds are listed once each, oriented as generated: the chain $i \to i+1$
  (the wrap bond $N-1 \to 0$ included), every kagome triangle counter-clockwise, every
  honeycomb bond from A to B, every pyrochlore bond from the lower sublattice to the higher.
  A uniform DM vector over `nn_pairs()` is then translation invariant, except along a
  periodic length of 2, where a pair's two bonds are one bond.
- `nnn_pairs()` and `nnnn_pairs()` are the second and third distance shells (minimum image
  on a periodic lattice), `i < j`. A lattice built by `from_neighbor_lists` knows no shells,
  and asking for one raises `InvalidRequest`.
- `from_neighbor_lists` takes each position as an `(x, y, z)` 3-vector (`TypeError` otherwise)
  and raises `InvalidRequest` for an edge endpoint `>= len(positions)`, an edge `(i, i)`, or a
  non-empty `sublattice` of another length.
- Lattices with a basis need at least 2 cells along a periodic direction.
- `from_cluster_file`: a `positions` block of lines `x y`, `x y z` or `id x y z`, and an
  `edges` (or `bonds`) block of lines `i j`; headers are case-blind and may end in `:`; a
  block may state its length; `#` starts a comment line; anything else raises
  `InvalidRequest` naming the line. A file that cannot be opened, or that lists no positions,
  also raises `InvalidRequest` (without a line number).

## `qed.dssf`: momentum-resolved spin operators

`qed.dssf.build_observables(spec)` returns a `qed.dssf.Observables`: `operators` (a list of
`qed.Operator`), `names` (a list of `str`, same order) and `len()`. Feed the operators to
`qed.dynamics` as probes; `S[i]` of the result belongs to `names[i]`:

```python
spec = qed.dssf.OperatorSpec()
spec.operator_type = "sum"
spec.basis = "ladder"
spec.components = [2]                                  # S^z
spec.momentum_points = [[math.pi / 2, 0.0, 0.0], [math.pi, 0.0, 0.0]]
spec.num_sites = 16
spec.positions_file = "positions.dat"                  # one "x y z" or "id x y z" line per site
obs = qed.dssf.build_observables(spec)
r = qed.dynamics(H, obs.operators, omega)              # r.S[i]: obs.names[i]
```

`OperatorSpec` fields (each type-checked when set: a value of the wrong type, or a negative
`num_sites`, `unit_cell_size` or `sublattice`, raises `TypeError`; the ranges below are checked by
`build_observables`):

| field | default | meaning |
|---|---|---|
| `operator_type` | `"sum"` | `"sum"`, `"transverse"`, `"sublattice"`, `"experimental"`, `"transverse_experimental"` |
| `basis` | `"ladder"` | `"ladder"`: components 0, 1, 2 are $S^+, S^-, S^z$; `"xyz"`: $S^x, S^y, S^z$ (any other string reads as `"ladder"`) |
| `components` | `[]` | component indices in 0..2, one operator each (any integer is accepted when set) |
| `momentum_points` | `[]` | 3-vectors $Q$ in absolute units (the units of the positions) |
| `polarization` | `[1, 0, 0]` | the transverse direction $e_1$ |
| `theta` | `0.0` | the angle of the experimental types |
| `unit_cell_size` | `4` | $U$ for `"sublattice"` |
| `num_sites` | `0` | $N$ (must be > 0) |
| `positions_file` | `""` | path (a `str`) to the site positions |
| `sublattice` | `None` | one sublattice for `"sublattice"`; `None`: all |

With positions $R_i$ and $\varphi_i = e^{iQ\cdot R_i}/\sqrt N$:

| `operator_type` | operator(s) per $Q$ | name |
|---|---|---|
| `"sum"` | $\sum_i \varphi_i S^a_i$ per component $a$ | `{C}_q_Qx{Qx}_Qy{Qy}_Qz{Qz}` |
| `"transverse"` | $\sum_i \varphi_i (e\cdot z_{i \bmod 4}) S^a_i$ for $e = e_1$ and $e = e_2$, per component | `..._NSF` ($e_1$), `..._SF` ($e_2$) |
| `"sublattice"` | $\sum_{i = s, s+U, s+2U, \ldots} \varphi_i S^a_i$ per sublattice $s$ and component | `..._sub{s}` |
| `"experimental"` | $\sum_i \varphi_i (\cos\theta\, S^z_i + \sin\theta\, S^x_i)$ | `Experimental_q_..._theta{theta}` |
| `"transverse_experimental"` | the same with the transverse weights | `TransverseExperimental_q_..._theta{theta}_NSF` / `_SF` |

`{C}` is `Sp`, `Sm`, `Sz` (ladder) or `Sx`, `Sy`, `Sz` (xyz); numbers are formatted with `g`.
$z_\mu$ are the pyrochlore local axes, $z_0 = -(1,1,1)/\sqrt3$, $z_1 = (-1,1,1)/\sqrt3$,
$z_2 = (1,-1,1)/\sqrt3$, $z_3 = (1,1,-1)/\sqrt3$. `compute_transverse_bases(Q, polarization)`
returns `(e1, e2)`: $e_1$ is the polarization as given (not normalised), $e_2$ the unit vector
along $Q \times e_1$ (along $\hat y \times e_1$ or $\hat x \times e_1$ when $Q \parallel e_1$).
`components` and `basis` are not used by the experimental types. Operators are ordered by $Q$,
then component, then $e_1$, $e_2$ or sublattice.

The positions file has one line per site, `x y z` or `id x y z` with the id the site's index
counting from 0 (blank lines, `#` lines and lines whose first field is not a number skipped),
and must list exactly `num_sites` sites. `np.savetxt(path, np.array(lat.positions))` writes one
from a `Lattice`. `build_observables` raises `TypeError` unless `spec` is an `OperatorSpec`, and
`qed.errors.InvalidRequest` (a `ValueError`) for an unknown type, empty `components` (except the
experimental types) or `momentum_points`, a $Q$ or polarization that is not a 3-vector,
`num_sites == 0`, a component outside 0..2, `unit_cell_size == 0` or
`sublattice >= unit_cell_size` for `"sublattice"`, a positions file that cannot be opened, and a
positions file with the wrong number of sites, a line of another column count, an id that is not
the site's index, or a coordinate that is not a number.

The phase is $e^{-iQ\cdot R_i}/\sqrt{N}$, the package's convention (as in `qed.Family.fourier` and the
examples' $S^a_q = N^{-1/2}\sum_j e^{-iqj} S^a_j$). Before 0.7.0 dssf used $e^{+iQ\cdot R_i}$: an old
$Q$ is the new $-Q$.

## Operators as observables

Any `qed.Operator` on the same number of sites as $H$ can be measured. Observables need not be
Hermitian (results are complex) and need not commute with $H$ or share any of its symmetries.
An observable that is `None`, acts on another number of sites or has a non-finite coefficient
raises `InvalidRequest`. $H$ itself must be Hermitian (to 1e-10 of its largest coefficient).

| call | returns |
|---|---|
| `qed.measure(H, requests, k=1, *, states="levels", degeneracy_tol=1e-8, sym=None, device="cpu", **eigs_kwargs)` | `MeasureResult`: one answer per request (`qed.Expect(ops)`, `qed.Correlations(A, B=None)`), all from one eigensolve and one sweep of each level's basis; `H` may be an `EigResult` with vectors |
| `qed.expect(H, ops, k=1, *, states="levels", ...)` | `ExpectResult`: `energies`, `multiplicities`, `values` (complex, `[rows, *index]`), `levels`, `eigs`, `diagnostics`, `rows`, `index`; `ground()` |
| `qed.correlations(H, A, B=None, k=1, *, states="levels", ...)` | `CorrelationResult`: `C[rows, *A_index, *B_index]` $=\langle A_a^\dagger B_b\rangle$, the one-point values `mean_a`, `mean_b`; `ground()`, `connected()`, `fourier(q)` (the structure factor, `StructureFactor.S[rows, *A_lead, *B_lead, q]`) |
| `EigResult.expect(ops)`, `EigResult.correlations(A, B=None)` | the same from an existing `qed.eigs(..., vectors=True)` result, also after `qed.load_eigs` |
| `qed.transitions(A, initial, final=None, *, B=None, pairs=False, raw=False)` | `TransitionResult` between the levels `initial` and `final` (each an `EigResult` or `(EigResult, indices)`): `omega[i, j]`, `strength[i, j, *A_index]` $= d_i^{-1}\sum_{n'\in i,\,m'\in j} \lvert\langle m'\rvert A_a \lvert n'\rangle\rvert^2$, with pairs `T[i, j, *A_index, *B_index]`; `ground()`, and `amplitudes(i, j)` with `raw=True`. Also the `qed.Transitions(A, ...)` request of `qed.measure` |
| `EigResult.matrix_element(O, i, j)` | $\langle v_i \rvert O \lvert v_j\rangle$ between the vectors of `levels[i]` and `levels[j]` |
| `qed.thermal(H, T, observables=[...], method=...)` | `ThermalResult.O`: $\langle O\rangle(T) = \mathrm{Tr}(e^{-H/T} O)/Z$, complex, `[len(observables), len(T)]` |
| `qed.dynamics(H, O, omega, B=None, ...)` | correlation spectra; see [Dynamics](dynamics.md) |

`expect` passes `eigs_kwargs` (`dense_max_dim`, `allow_partial`, `prune`, `window`) to
`qed.eigs` and solves with vectors. Thermal observables need `method="exact"` or `"ftlm"`, and
`exact_states=0` with either (`InvalidRequest` otherwise); `"exact"` evaluates them through each
block's eigenvectors on the host, `"ftlm"` with the symmetric estimator
$\sum_r \sum_{ij} e^{-(E_i + E_j)/2T} \langle r|\psi_i\rangle \langle\psi_i|O|\psi_j\rangle \langle\psi_j|r\rangle / Z$
on each sample's Krylov basis.

`ops`, `A` and `B` are a `qed.Operator`, a sequence of them, a `qed.Family` (an index axis: its
shape, e.g. `(3, N)` for `qed.Family.spins(lattice)`) or a `qed.MomentumFamily` (`family.fourier(q)`:
a momentum axis on the family's last index, $O_q = N^{-1/2}\sum_r e^{-iq\cdot r} O_r$). Pairs are
formed exactly with the operator algebra, and pairs related by a symmetry are evaluated once:

```python
lat = qed.input.lattice.chain(N, True)
spins = qed.Family.spins(lat)                              # S_i^a, shape (3, N)
c = qed.correlations(H, spins)                             # C[row, a, i, b, j] = <S_i^a S_j^b>
S = c.fourier("cluster")                                   # S^ab(q) at the cluster's momenta
m = qed.measure(H, [qed.Expect(spins), qed.Correlations(spins)], states="ground")   # one pass
```

### What "O need not share any symmetry of H" means

**`expect` and thermal observables.** Each level, and each block of a thermal trace, lives in
one symmetry sector. Within it the engine replaces $O$ by its average over the symmetries the
calculation uses -- the spatial group of `sym` (momenta and point group), the spin flip where
the block folds or projects by it, and the time-reversal partner where the level is paired by
it -- and drops the terms that cannot connect the sector to itself (terms that change $S^z$ in
an $S^z$ sector, or change it by an odd amount in an $S^z$-parity half). The averaged operator
has the same trace against any function of $H$ over an ensemble those symmetries preserve, so:

- A thermal average $\langle O\rangle(T)$ is that of $O$ itself, exactly, whatever $O$ is
  (`examples/02_thermal.py` measures one bond's
  $\langle \vec S_0\cdot\vec S_1\rangle(T)$).
- `values[i, a]` is the average of $\langle\psi|O_a|\psi\rangle$ over the members of level
  `i`'s symmetry multiplet: the quantity that does not depend on which partner the solver
  returned. For a non-degenerate level it is $\langle\psi|O|\psi\rangle$, and
  `multiplicities[i] * values[i, a]` is the level's contribution to $\mathrm{Tr}(P_E O_a)$.

In practice:

- A local operator measures its symmetry-averaged value: $\langle S^z_0 S^z_1\rangle$ in a
  level of a translation-invariant chain is the per-bond correlation. An operator that changes
  sign under an element of the group used (a staggered magnetisation under a one-site
  translation, a bond-nematic difference under a 90-degree rotation) has expectation zero in
  every level -- also in a degenerate multiplet whose individual vectors would not give zero.
- An operator that changes $S^z$ (`P(N, "+", [0])`) has zero expectation in an $S^z$ sector.
- A level paired by spin flip (the $\pm S^z$ members, folded by `spin_flip="auto"`) averages
  over both members: $\langle S^z_\mathrm{tot}\rangle = 0$ there. A level paired by time
  reversal (momenta $k$ and $-k$) averages $O$ with its time-reversed image: a time-reversal-odd
  observable (a spin current, a scalar chirality) averages to zero.
- Under `Symmetry(total_spin=S)` with an SU(2)-symmetric $H$ (no field), a level stands for
  its $2S+1$ members and $O$ enters through its SU(2)-scalar part, its average over all spin
  rotations: $S^z_i S^z_j$ enters as $\vec S_i \cdot \vec S_j / 3$. This part is exact for
  terms on at most 5 sites; an $O$ that is not SU(2) invariant as a whole and has a term on more
  than 5 sites raises `Unsupported` (even when that term is itself invariant).
  In a uniform field along $z$ every member is a level of its own and $O$ enters as it is.
- To measure one partner, narrow the symmetry so that it is a level of its own (an explicit
  `sz=n`, `spin_flip="off"`, `time_reversal="off"`, `point_group=False`), or use
  `EigResult.matrix_element(O, i, i)`, which evaluates $O$ as it is between the vectors the
  solver returned (any $O$: it may change $S^z$ and break every symmetry), or the full-space
  vectors of `EigResult.vectors()` with `O.apply` for small $N$.

**`dynamics`.** Probes are not averaged. `qed.dynamics` works in the momentum sectors of the
abelian part of the spatial group and carries $B\lvert m\rangle$ from its source sector into every
target sector (momentum, $S^z$ or parity) that a term of $B$ reaches; which targets are reached
follows from the operator's terms exactly. A Fourier mode $S^a_q$ connects each momentum to
the one shifted by $q$ only; $S^\pm$ change the $S^z$ sector; a single-site operator reaches
every momentum.
The point group and the spin flip are used where they are exact for the probe; see
[Dynamics](dynamics.md) (section "Symmetry").

## Implementation map

| | |
|---|---|
| `python/qed/_bindings/core.cpp` | the `qed.Operator` binding: records, algebra, `terms`, `image`, `apply` |
| `include/ed/ops/operator.h` | `Operator`: records, four-site terms, `canonical()`, the full-space apply |
| `include/ed/ops/algebra.h`, `src/ops/algebra.cpp` | `MaskedOperator`: canonical terms, `product`, sums, products, adjoints, images |
| `include/ed/ops/invariance.h`, `src/ops/invariance.cpp` | records to canonical terms and back (`masked`, `to_operator`, `product_terms`), the symmetry verdicts, group averages, the SU(2)-scalar part |
| `include/ed/ops/program.h` | operators compiled into programs over symmetry sectors (matrix elements between sectors) |
| `python/qed/_builder.py` | `HamiltonianBuilder` |
| `include/ed/input/lattice.h`, `src/input/lattice.cpp`, `python/qed/_bindings/input.cpp` | the lattice generators |
| `python/qed/dssf.py` | `OperatorSpec`, `build_observables`, `compute_transverse_bases` |
| `src/engine/expect.cpp`, `src/engine/walk.h` (`Averager`) | how `expect` and thermal observables average $O$ |

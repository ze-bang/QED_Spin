# Architecture

QED_Spin has one path from a Python call to the kernels. Every verb walks the same symmetry
blocks, places each block by the same rule (`ed::place`) and applies every sector operator through
the same row walk.

```text
python/qed/_verbs        eigs · spectrum · thermal · dynamics · expect     argument checks, results
        │                Symmetry.resolve(H) -> _core.sectors.Spec        (find_symmetries for spatial="auto")
        ▼
_core.sectors.<verb>     python/qed/_bindings/sectors.cpp                  the GIL is released
        │                (expect: EigsResult.expect on the eigs result)
        ▼
ed::sectors::<verb>      src/engine/{eigs,thermal,dynamics,expect}.cpp     H, Spec, options, environment validated
        │  subspaces(H, s)            Sz sectors, parity halves or the full space; flip / Theta mirrors
        │  detail::walk(...)          one momentum star at a time: build_star_blocks -> blocks
        │  detail::block_operator()   the block's LinearOperator (and its spin tower)
        │  ed::place(device, req)     the block's Lane
        ▼
kernels                  dense (LAPACK, cuSOLVER) · Lanczos scan · ground-state vector · Krylov-Schur ·
        │                FTLM · OFTLM · mTPQ · continued fraction · FTLM dynamics
        │                include/ed/{krylov,thermal,dynamics}, templated on the backend
        ▼
backends, operators      CpuBackend, BasicCpuBackend<double>, CudaBackend;
                         RepSectorMatVec: reduced CSR or row walk on the host, device CSR or device gather
```

Conventions: bit i of a basis state is site i, a set bit is spin up, `n_up` counts up spins and
`Symmetry(sz=n)` selects Sz = n - N/2. A permutation T acts as bit i of T|s> = bit T[i] of |s>.
Every threshold that judges an energy is relative to s_H, the sum of |c| over H's canonical terms
(`<ed/core/numerics.h>`), so a rescaled H takes the same decisions.

## From a call to a Spec

Each verb checks its arguments, resolves `device=` (`_verbs/_device.py`) and calls
`Symmetry.resolve(H)`, which fills a `_core.sectors.Spec`:

| Spec field | from | meaning |
|---|---|---|
| `abelian` | `spatial` | a closed abelian group of site permutations, normal in the group it spans with the residues (the whole spatial group unless the co-group was capped); its characters are the momenta |
| `residues` | `spatial`, `point_group` | one representative per coset of `abelian` (the point group) |
| `n_up`, `sz_parity`, `use_sz` | `sz` | one Sz sector, one half by the parity of the up-spin count, or `sz="off"` |
| `spin_flip`, `time_reversal` | same names | -1 auto, 0 off, 1 require |
| `two_S` | `total_spin` | 2S, or -1 |
| `only_k0`, `only_irrep`, `only_momentum`, `only_irrep_chars` | `Symmetry.select` | block selections |

`spatial="auto"` runs `find_symmetries(H)` (pynauty) on an interaction graph built from H's
canonical terms and splits the automorphisms that commute with H into the largest normal abelian
subgroup and its coset representatives. A graph with more than 4096 automorphisms is not
enumerated (no spatial symmetry, with a diagnostic); a group whose co-group exceeds 128 elements
is cut down to a subgroup whose co-group fits. A permutation list is closed (at most 4096
elements) and split the same way; a `qed.Symmetries` is an explicit split, checked, not re-chosen
(its residues must form a group with the abelian part and give at most 128 cosets).
The engine then validates H (1 to 63 sites, finite, Hermitian), the Spec (permutations, `abelian`
normalised by every residue, every permutation commuting with H), the options and the environment
(a malformed `ED_*` value refuses every verb) before any work.

## Symmetry sectors

### Subspaces

`subspaces(H, s)` (`src/engine/eigs.cpp`) reads what H conserves along z from its canonical terms:

| H conserves | subspaces |
|---|---|
| Sz | every Sz sector, or the one named. With the spin flip (unless `spin_flip="off"`) -- else, for an H that is not real, time reversal Theta (unless `time_reversal="off"` or a `select` is active) -- n_up and N - n_up are solved once, at Sz >= 0 (`mirror = 2`); with `sz="even"`/`"odd"` only for N even |
| Sz parity only | both parity halves, or the one named; for odd N the flip (Theta) exchanges them and one is solved |
| neither, or `sz="off"` | the full 2^N space |
| SU(2), `total_spin=S` | the Sz = S sector; a level stands for its multiplet (`members = 2S + 1`) |
| SU(2) in a uniform field along z, `total_spin=S` | the sectors Sz = -S..S, every member a level of its own |

### Stars and blocks

Per subspace, `make_engine_context` (`src/engine/context.cpp`) builds the abelian irreps in closed
form (trivial character first); extends A by the global spin flip, A' = A x Z2, where H is
flip-invariant and the subspace is its own image (n_up = N/2, a parity half with N even, the full
space); chooses the antiunitary fold of stars -- K for a real H, else Theta where the subspace is
closed under it and the flip is not engaged; and maps each residue's conjugation onto the momenta.
`star_partition` unites momenta under the residues and the fold, and `walk()` (`src/engine/walk.h`)
builds one star at a time and hands its blocks to the verb. Under `select(irrep_character=...)`
nothing is folded by time reversal, and Theta folds nothing under a momentum, star or irrep
selection (`select(sz=...)` only names an Sz sector: at Sz = 0 Theta still folds k with -k).

`build_star_blocks` (`src/engine/stars.cpp`) takes the momentum sector's dimension from Burnside's
count, (1/|A'|) sum_g conj(chi(g)) Tr U_g, without building the sector, and then:

- **nontrivial little co-group** P_k0 (the residues fixing k0, one per coset of A): the full
  little group G_k0 = A.P_k0 (x flip). Its table p_e p_f = a_ef p_g carries the factor system
  omega(e, f) = chi_k0(a_ef); the irreps of G_k0 that restrict to chi_k0 on A are the
  omega-projective irreps of P_k0 (ordinary for omega = 1), D(a p_e) = chi_k0(a) D(e). Every irrep
  is a **group sector** in the representative basis of G_k0, about d C(N, n_up) / |G_k0| states;
  the sectors must tile the momentum sector (sum of d times their dimensions), else the build throws.
- **trivial co-group**: one plain block, the momentum sector itself, built from the subspace's
  orbit table under A (acquired by the first star that needs it).

Time reversal folds sigma and sigma* of a real momentum into one block of doubled multiplicity; a
star that only time reversal closes (k with -k) is marked `tr_folded`. A level's multiplicity is the
star size |star| x d (x 2 when folded) x `mirror` x `members`. Levels carry `n_up`,
`flip_parity`, `k0` and `irrep` (engine indices), `momentum` (chi_k on `abelian`), `irrep_characters` (chi_sigma on the
residues, -1 the identity) and `fold` (`"K"`, `"theta"` or None).

### Orbit tables and representatives

An `OrbitTable` (`<ed/basis/orbit_table.h>`) is the irrep-independent result of one fused scan of a
subspace (fixed Sz in Gosper order, a parity half, or the full space) under a `CompiledGroup`
(permutation then XOR mask, applied through byte tables): the representatives in ascending state
order and, per representative, a uint16 id of its stabiliser in a deduplicated list. The scan runs
in 64 dynamic chunks per thread merged in state order, so the table is the same at any thread
count. A sector of characters chi keeps the representatives of nonzero norm,
norm^2 = |sum_{h in Stab} chi(h)|^2 / |Stab| (`filter_reps`). Tables are shared within the process
through a registry keyed by a content hash (group, subspace, table version, sublattice code),
spot-checked on every hit and holding at most 8 tables.

The representative of an orbit is its least member in an order. Without a code the order is the
integer one, and canonicalising a state takes |G| images. With a **sublattice code**
(`<ed/basis/sublattice_code.h>`) the group permutes the blocks of a block system (2 to 8 equal
blocks of at most 16 sites; the system with the largest blocks is taken), and the order is a key
order in which each block is a contiguous
range of bits (block j holds key bits [N - (j+1)L, N - jL)). The leading L key bits of an image
come from one block of the state through a table, so the least leading block is the least of m
table entries, and only the elements that reach it -- listed per entry, usually one or a few --
need a full image (Wietek and Lauchli, Phys. Rev. E 98, 033309 (2018)).

| `ED_SYM_SUBLATTICE` | code |
|---|---|
| unset | N >= 24 sites and >= 16 distinct site permutations for a verb on the device and for eigs, spectrum and exact thermal; >= 64 for host sampled thermal and host dynamics |
| `1` | whenever a block system exists |
| `0` | never |

The key order is a function of the group's distinct site permutations and that variable, read
when a `CompiledGroup` is built (a group with or without its spin-flip half gets the same order).
The orbit table records the code (`OrbitTable::slc`), every sector built from the table carries it
(`RepSectorData::slc`), and the host policy and the device mirror read it there, so a result keeps
the rule it was computed with. `EigResult.save` stores each basis's code
fingerprint (`sublattice`); `load_eigs` rebuilds the code from the stored group, refuses a
fingerprint it cannot reproduce and checks a sample of the stored representatives; a file without
the field is in the plain order. The physics does not depend on the rule; the basis differs by
phases, so seeded sampled results move at sampling level.

### Sectors of an irrep of dimension d > 1

Representative r holds rank(r) <= d states, the partner-0 states
`|r; a> = sqrt(d/|G|) sum_j C_r[j][a] Ptilde_0j |r>` with `Ptilde_ij = sum_g conj(D(g)_ij) U(g)`.
M_r = sum_{s in Stab(r)} conj(D(s)) is |Stab| times a projector; C_r, shared by its stabiliser
class, holds its eigenvectors of eigenvalue |Stab| scaled by 1/sqrt(|Stab|), so
C_r^dag M_r C_r = I (`group_sector_irrep_from_table`). An operator entry between representatives
is the rank(r) x rank(c) block conj(h) C_r^dag A(t) C_c, A(t) = sum_{g: g t = rep_c} D(g)^T.
The row walk's blocks hold irreps up to dimension 8 (`kMaxIrrepDim`, `<ed/matvec/sector_rows.h>`):
the sector build refuses a larger one (`Unsupported`), and `load_eigs` a file that holds one
(`InvalidRequest`); d = 1 is the case C = 1/norm.

### State lookups

A connected state t is canonicalised -- its representative and the character sum over the elements
that reach it, in one running-minimum pass (`index_and_projection`) -- then looked up:

| lookup | used by | cost |
|---|---|---|
| shared rank table (combinadic rank -> representative of the subspace's table) and a per-sector int32 remap | fixed-Sz momentum sectors whose C(N, n_up) int32 entries fit `ED_SYM_REP_RANKTABLE_BUDGET_GIB` | O(1) |
| rank buckets: the representatives whose key (the rank at fixed Sz, else the state) falls in each run of 2^shift keys; about one per representative, at most 2^24 (64 MB) | every other sector | one bucket |
| binary search over the sorted representatives | sectors with neither | O(log dim) |

The device mirror (`src/gpu/rep_matvec.cu`) holds the shared rank table once per subspace, or state
buckets of its own.

### Applying an operator: the row walk and the reduced CSR

An operator is its canonical terms (`ed::ops::MaskedOperator`), compiled into the row program of
its adjoint (`MaskedProgram`: terms grouped by flip mask, then by the required value on the flipped
bits). Row r of O in a sector is the **row walk** of that program from the representative s_r
(`<ed/ops/row_walk.h>`, `<ed/matvec/sector_rows.h>`): each group acting on s_r gives a target t and
h = <t|O^dag|s_r>, and O[r][j(t)] += inv_norm[r] conj(h proj(t)) with (j, proj) from
`index_and_projection`; the diagonal needs no lookup. The same walk serves H, observables, S^2 and
operators between two sectors (`CrossSectorMatVec`: rows of the target, looked up in the source).

`RepSectorMatVec` (`src/engine/internal.h`) decides on its first apply how the block applies H.
`EigResult.block_stats` (one dict per solved `eigs` block: `k0`, `irrep`, `flip_parity`, `n_up`,
`dim`, `kind` (`"group"` or `"plain"`), `lane`, `context_orbit_s`, `star_orbit_s`,
`star_build_s`, `build_s`, `nnz`, `csr_bytes`, `applies`, `apply_s`, `other_s`, `solve_s`)
reports it in `lane` (a host block solved without an apply of H, a dense solve, reports `"dense"`):

| lane | representation | when |
|---|---|---|
| `csr` | the reduced CSR, built once, then an SpMV | default (`ED_SYM_REDUCED_CSR` unset or true); dim < 2^32; estimate within the block's CSR budget |
| `csr-real` | the CSR's real dictionary on real vectors | a host `eigs` lane (or its prune estimate) of a block whose CSR keeps a dictionary with every imaginary part within 32 eps of its largest value, unless `ED_SYM_REAL=0`; FTLM, OFTLM, mTPQ and dynamics stay complex |
| `walk` | the row walk on every apply, O(#reps) memory | the CSR declined |
| `gpu-gather` | the device gather on host vectors | an operator applied on the host under `device="auto"` or `"gpu"` (a host Krylov lane under `"auto"`; under `"gpu"` an operator a host path applies, such as S^2 or an observable), 1-dim sector, CSR declined, >= 2^20 representatives (`ED_SYM_LG_GPU`) |
| `device-csr`, `device-gather` | the device kernels below | the solve runs on the device |
| `dense` | the matrix, written from the CSR | dense solves |

The reduced CSR keeps 4-byte columns and its values as ids into a dictionary of distinct values --
uint8 up to 256, uint16 up to 65536 (`kCsrDictMax`) -- or, past that, whole: 5-6 bytes an entry
against 20, the same bits. The build is one parallel pass over chunks of rows with per-chunk
dictionaries merged in chunk order, its arrays first-touched in the SpMV's static partition, and
entries merged by column in emission order, so it repeats bit for bit. The budget check charges 7
bytes an entry and one entry per term and row (d per term for an irrep of dimension d); when that
bound does not fit, the mean length of 4096 sampled rows decides. A full-value CSR is built only
within the budget, against its exact size. An operator applied once or twice (S^2 certifying a few
vectors) walks before it builds (`defer_csr`); a cross-sector operator walks its first apply and
builds from its second, within `ED_XSEC_CSR_BUDGET_GIB`.

## Tasks

Every verb walks subspaces x stars x blocks; `dynamics` also builds bare momentum sectors
(`little_group_k_sectors_stream`). What runs per block, and the `ed::Task` it is
placed as:

| verb | per block | task |
|---|---|---|
| `eigs` | dense at dim <= 2 or dim <= the crossover (`dense_max_dim`; `None`: min(4 max(40k, 400), 8192, the largest dense solve within half the free RAM), k the levels the block owes: 1600 for k <= 10), else Krylov: for one level the Paige-gated Lanczos scan, or with `vectors=True` the certified ground-state vector (residual <= 1e-9 s_H); for k levels thick-restart Krylov-Schur with locking (p = k + max(k/2, 8) kept, cycles of up to 2p + 20 vectors). A block owes ceil(k / its multiplicity) levels. **Pruning**: a block above the crossover gets a 40-step estimate theta - abs(r) and is solved, in order of increasing estimate, unless it exceeds the k-th level found so far by more than max(0.02 max(abs(E_k), 0.05 s_H), `window`). A block that stopped short of certifying levels that may lie in the window makes it incomplete, which raises unless `allow_partial` | `Eigs` |
| `spectrum` | the whole block in a dense batch (below) | `DenseBatch` |
| `thermal`, `"exact"` | a dense batch; with `observables`, a host eigensolve with vectors and the diagonal of each averaged O | `DenseBatch`, host |
| `thermal`, `"ftlm"` | dense on the host at dim <= `dense_max_dim` (default 512; the same fallback serves OFTLM and mTPQ); else `samples` Lanczos runs of depth `krylov` with local DGKS3 reorthogonalisation, or with observables a kept, fully reorthogonalised basis and the symmetric estimator | `Sampled` |
| `thermal`, `exact_states=n` (OFTLM) | the n lowest states as certified eigenpairs of the block eigensolver, completed to whole levels (up to 16 more pairs), and random starts orthogonalised against them | `Oftlm` |
| `thermal`, `"mtpq"` | canonical mTPQ: spectral bounds from a 60-step Lanczos, then `steps` iterations of (L - H) per sample | `Sampled` |
| `dynamics`, `T=None` | the ground manifold: a k = 1 `eigs` with a window of `degeneracy_tol` s_H on the caller's folded Spec, each block at E0 solved deeper, each level expanded into its multiplet in momentum sectors (`members_of`). Per target subspace, one momentum sector at a time: B\|a> by `CrossSectorMatVec`, projected onto the 1-dim irrep blocks of that momentum (one continued fraction each), the part in irreps of dimension > 1 on the momentum sector; a cross pair projects A\|a> on the fraction's Krylov vectors. Under `device="gpu"` or a momentum selection the ground manifold is solved on bare momentum sectors | `DynamicsCf` |
| `dynamics`, `T=[...]` | per source momentum sector (every source counts in Z): one FTLM-dynamics run against each target sector a probe reaches, its samples seeded by (n_up, parity, momentum); each probe counts once every source its own symmetries relate (the spin flip, momenta in one residue orbit) | `DynamicsFtlm` |
| `expect` | O averaged over the group (and the flip where the level folds or projects by it), its terms that leave the level's subspace dropped (Sz-changing at fixed Sz, parity-changing in a parity half): one `rep_matrix_elements` sweep per (basis, flip, kept terms) over all levels, one apply per level for d > 1; a folded level averages with its image | host |
| `matrix_element` | the lambda-projected program between two sectors of one group, else an explicit orbit walk | host |
| `vectors()` | each level expanded into the Sz sector or 2^N and closed under the residues, its fold, the flip and total S- | host |

Under `total_spin` the solvers keep the bare H (`detail::block_operator` in `src/engine/walk.h`,
the tower in `src/engine/tower.cpp`): Krylov lanes start from random valence-bond states of spin
S, every returned pair is certified through S^2 (on the host applied as S- S+ + Sz(Sz + 1)
through the sector one up spin higher, on a spin-flip sector through the sector without the flip
where that fits; otherwise, and on the device, as the S^2 operator itself), and
H + mu f(S^2) is the fallback when an off-tower level took a tower level's place; dense paths diagonalise
Q^dag H Q on the tower; sampled methods start from P_S of a Gaussian; tower dimensions come from
Burnside's count.

Blocks combine in log space: ln Z = logsumexp_b (ln w_b + ln Z_b), C from the law of total
variance, M and chi from the blocks' Sz. Host blocks below 2^16 states (`kHostPoolMaxDim`: deferred
thermal blocks, small dynamics sources) run concurrently, one thread each with serial BLAS. Dense
batches (`DenseBatch`, `walk.h`) queue host blocks within 16 GiB or a quarter of the free RAM and
solve them concurrently, one serial LAPACK call per thread, largest first (a block larger than that
budget is solved alone, on the threaded LAPACK); device blocks are written from their CSR into
batches of at most `ED_GPU_DENSE_BATCH_GIB` of matrices and solved on a pool of up to 8 streams by
cuSOLVER's 64-bit syevd, a real block in real arithmetic. A block larger than a batch goes up alone when it
and its workspace fit half the free device memory. Under `"auto"` a block that does not fit, and a
batch whose device solve fails, go to the host; under `"gpu"` the block raises `ResourceLimit`, and
the failed batch is retried in halves on the device until a single block that still fails raises
`ResourceLimit`.

## Device placement

`ed::place(Device, BlockRequest)` (`<ed/core/select_backend.h>`) is the one device decision. It
returns the block's `ed::Lane` -- `HostDense`, `HostKrylov`, `DeviceDense`, `DeviceKrylov` --
which results count in `placement` (`host_dense`, `host_krylov`, `device_dense`, `device_krylov`).
In order:

1. `DenseBatch`: the host under `"cpu"`; without a device, the host under `"auto"` and an error
   under `"gpu"`; with one, the device under `"gpu"`, and under `"auto"` from 1024 states
   (`kDeviceDenseMinDim`).
2. A block the verb solves densely: `HostDense` under every device.
3. `"cpu"`: `HostKrylov`, before any probe; CUDA is never initialised.
4. `"auto"`: the device when the block has a device kernel, dim >= its task's floor, a device is
   visible and, where the row checks fit (not under `ED_MEM_GUARD_OFF`), its device working set
   fits the free device memory; else the host.
5. `"gpu"`: the device, or an error naming the block.

An `Eigs` block of at most 32 states (`kDeviceDenseMaxDim`) or with 2 want >= dim (want: the levels
the block owes) is solved densely on the host under `"auto"` and `"gpu"`. The `"auto"` table (`auto_row`, `<ed/core/device.h>`):

| task | floor | fit checked |
|---|---|---|
| `Eigs`, `Sampled`, `Oftlm` | 2^14 | yes |
| `DenseBatch` | 1024 (rule 1) | the batch sizes itself |
| `DynamicsCf` | 2^14 | yes (Lanczos vectors) |
| `DynamicsFtlm` | 2^16 | yes (both bases) |

Device kernels (`<ed/gpu/rep_matvec.h>`):

| operator | device apply |
|---|---|
| 1-dim sector (momentum sector, 1-dim group sector) | **device CSR**: its reduced CSR built on the device at the first bind -- one warp merges a row in registers, values in a dictionary -- and applied with 2-32 threads a row and a fixed shuffle tree; within `ED_GPU_CSR_BUDGET_GIB`, for programs of at most 256 flip groups, not for deferred operators. Otherwise the **device gather** (`walk_gather`): the warp's rows step through their own acting groups and canonicalise together, so no lane idles on a group that does not act |
| sector of an irrep of dimension > 1 | its host reduced CSR, uploaded (`upload_csr_gpu`); it has a device kernel only when that CSR fits the block's CSR budget and the device CSR budget |
| between two sectors (dynamics at T > 0) | the device walk (`make_cross_matvec_gpu_rep`) |

A k-vector apply gives each output bit for bit as a single apply does, so FTLM, mTPQ and FTLM
dynamics advance up to 8 samples in lockstep on the device (`MatvecBatcher`), as many as fit 90% of
the free device memory (OFTLM runs its samples one at a time); thermal FTLM and mTPQ retry with half
the width on an allocation failure. The device blocks of a thread share one
`CudaBackend` (`thread_cuda_backend()`), which reads the Krylov basis in place.

`device="gpu"` refuses:

| condition | error |
|---|---|
| no CUDA build, no visible device, or device memory that cannot be queried | `DeviceUnavailable` |
| a block without a device kernel: a sector of an irrep of dimension > 1 whose CSR does not fit; the message names star, irrep, n_up and dimension | `DeviceUnsupported` |
| a thermal observable without a device kernel; a spin-tower penalty re-solve whose penalty operator has no device kernel | `DeviceUnsupported` |
| an `eigs`, sampled `thermal`, OFTLM or `dynamics` block whose device working set exceeds the free device memory | `ResourceLimit` |
| a d > 1 CSR that cannot be uploaded after the block was placed | `ResourceLimit` |
| a dense-batch block too large for the device, or whose device solve fails alone (a failed batch is retried in halves first) | `ResourceLimit` |

It still runs on the host: `eigs` blocks within the dense crossover, of at most 32 states or with
2 want >= dim; sampled `thermal` blocks within `dense_max_dim`; exact `thermal` with observables;
`expect`, `matrix_element` and `vectors()`.

## Memory budgets

`<ed/core/footprint.h>` is the one working-set estimate: per solver path, the host and device bytes
of the vectors it holds at its peak (16 bytes an entry, 8 on a real lane) or of its dense
matrices. Operators are budgeted separately, by the variables below.

| path | counts |
|---|---|
| `FtlmSample`, `FtlmSampleKept` | per sample: its seed and three working vectors; with observables also the kept basis of `krylov` vectors |
| `Mtpq` | the bounds Lanczos (5 vectors), then psi and H psi per sample |
| `KrylovSchur` | the m + 1 basis columns, the p kept Ritz vectors, the k pairs found, a working vector |
| `GsKeptBasis`, `GsTwoPass` | the kept Krylov vectors (none for `GsTwoPass`), the seed, three recurrence vectors, the Ritz vector |
| `DenseValues`, `DenseVectors` | 24, 32 bytes per matrix entry |
| `Multiplet` | the k vectors, the expanded state, a seed and the state table |
| `DynamicsFtlm` | per sample both Krylov bases with their seeds and working vectors |

Its readers: the RAM guard (`guard_working_set`: `ResourceLimit` above 90% of the RAM the job may
still allocate), `place()`'s device fit, the Krylov-Schur cycle cap (90% of the lane's memory), the
automatic dense crossover, the lockstep sample width, the dense batches, the multiplet, and the CSR
budget of a sampled `thermal` block (an `eigs` block reserves 16 bytes x dim x 6 vectors for one
level, x (3k + 30) for k). Free RAM is the smaller of `MemAvailable` and the headroom under the job's
cgroup-v2 limit (inactive page cache counted free); free device memory includes what the process's
memory pool holds unused.

| variable | default | bounds |
|---|---|---|
| `ED_SYM_SECTOR_CSR_BUDGET_GIB` | 0.55 x free RAM less the block solver's working set (8 GiB if RAM cannot be read; unlimited under `ED_MEM_GUARD_OFF`) | the reduced CSRs of one block (H, then S^2, then observables: `CsrBudget`), divided among concurrent sector builders; set, an absolute cap (0 admits none) |
| `ED_XSEC_CSR_BUDGET_GIB` | 4.0 | a cross-sector CSR (dynamics, the S^2 ladder), divided among concurrent builders |
| `ED_SYM_REP_RANKTABLE_BUDGET_GIB` | 0.5 | the shared rank table, C(N, n_up) int32; 0 builds none |
| `ED_GPU_CSR_BUDGET_GIB` | half the free device memory at the bind | an operator's device CSR, built there or uploaded; 0: the device gather for a 1-dim sector, and no device kernel for a sector of an irrep of dimension > 1 |
| `ED_GPU_DENSE_BATCH_GIB` | 2 | matrices per device dense batch, also at most a quarter of the free device memory and of the free RAM |
| `ED_GPU_SYM_CACHE_GIB` | 25% (rank tables, at most 24 GiB) and 15% (sector mirrors, at most 16 GiB) of the device's total memory | each device cache that pins recent symmetry tables, read once per process; 0 pins none, a positive budget keeps at least the newest entry |
| `ED_MEM_GUARD_OFF` | off | on: the RAM guard and the RAM- and device-fit caps above stand down (the defaults of `ED_GPU_CSR_BUDGET_GIB` and `ED_GPU_SYM_CACHE_GIB` stay) |

Fixed caps: the orbit-table registry (8 tables; past the newest, a quarter of the free RAM), the
host dense queue (16 GiB or a quarter of the free RAM), the stars `eigs` keeps for its pruning
survivors (a quarter of the free RAM). Lane switches: `ED_SYM_REDUCED_CSR` (0: always walk, read
once per process), `ED_SYM_REAL`, `ED_SYM_SUBLATTICE`, `ED_SYM_LG_GPU` (0 vetoes the host-vector
device gather, 1 drops its 2^20 floor) and `ED_SYM_PROFILE` (phase timers at Info). Every variable
is declared once, in `include/ed/core/config.h`: `qed.debug_env()` lists them with their live
values and `qed.env_snapshot()` the ones set; an unknown `ED_*` name warns at import (raises under
`ED_ENV_STRICT=1`).

## Verification

| layer | where | checks |
|---|---|---|
| lints | `scripts/check_env_registry.sh`, `check_no_print.sh`, `check_tolerance_literals.sh`, `check_int_narrowing.sh` | every variable in the registry; no console output; no tolerance literal outside `numerics.h` unless marked scale-free; no unchecked integer narrowing |
| unit | `tests/unit/*.cpp` (Catch2, ctest) | kernels, groups and irreps, orbit tables, sublattice codes, row walk, CSR budgets, `place()`, the CUDA backend; a failed assertion exits 1, an all-skipped run 77 |
| API | `tests/python/test_*.py` | verbs, device rules, discovery, dssf, environment, logging; `test_sublattice.py`: forced on and off give the same results, a saved result keeps its rule |
| grid | `tests/python/grid` | every task x symmetry content x backend cell against a dense reference built without the library, plus properties (multiplet sizes, labels on vectors, scale covariance, sum rules); each cell pass, wrong, refused, missing or crash, gated by `baseline_{cpu,gpu}.json` |
| golden | `tests/python/golden` | the values every verb returns, recorded per reference commit (`refs/<tag>/{cpu,gpu}.json.gz`): exact 1e-10, transport 1e-7, stochastic 1e-10; the complete spectra of small models also against the dense oracle |
| regress | `tests/python/regress` | 148 audit repro scripts: `open` must still reproduce (strict xfail), `fixed` must not; `perf` and `info` are not gated |
| fuzz | `tests/python/fuzz` | random models and requests against numpy/scipy; gate seeds 1-4 (CPU) and 1-2 (GPU), 150 cases each, `--strict`, accepted failures only open ledger ids in `known.json` |
| sanitizers | `scripts/gate/sanitize.sbatch` | `KIND=asan`: AddressSanitizer + UBSan build, ctest and `test_api.py`; `KIND=cusan`: compute-sanitizer memcheck and synccheck over the device unit tests and repro X04; once per batch of commits |
| bench | `bench/` | `run.py` records wall time, peak RSS and the per-block record; `--compare` fails a case > 10% slower or using > 15% more memory than the baseline; XDiag twins in `bench/xdiag` |

**The gate.** `scripts/gate/submit.sh <account> [--variant cpu|cuda] [--build-only] [--dry-run]
[stage ...]` submits the build job (the four lints, then the unit-test binaries and `_core`), then
CPU arrays (10- and 20-minute stages) and a GPU array (one H100 MIG slice, 1g.10gb) of the stages in
`scripts/gate/tasks.sh`: ctest, pytest, the examples, golden, the grid shards, the regress shards,
the fuzz seeds, and `*_slc` repeats of pytest, ctest and grid shards under `ED_SYM_SUBLATTICE=1`
(unset, the code engages only from 24 sites, which no small test reaches). Each stage appends
`<stage> <exit code>` to `logs/gate/<build id>/rc`; the gate passes when every stage reports 0. Job
scripts source `scripts/env.sh`; personal paths (virtualenv, pybind11) live in
`~/.config/qed/site.env` (`QED_SITE_ENV`). The version is `pyproject.toml`'s alone: CMake reads it
into the project and `qed._core.__version__`, which is `qed.__version__`.

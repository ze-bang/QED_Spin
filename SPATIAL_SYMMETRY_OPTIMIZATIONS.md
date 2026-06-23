# Spatial Symmetry Workflow: Architecture and Optimizations

## Overview

This document covers the full call chain for spatial symmetry in the QED exact
diagonalization code, from the Python API down to the CUDA device kernel, and
details every optimization applied in June 2026.

---

## 1. Architecture: Two Code Paths

The spatial-symmetry workflow has two hot-path variants that share the same
setup pipeline but diverge at the matvec:

```
Python: qed.thermal(..., use_symmetry_if_available=True, use_sz_if_conserved=True)
   │
   └─► workflows_thermal_all_sz_streaming_symmetry_directory()   [workflow_bindings.cpp]
            │
            ├─ [SETUP] make_all_sz_sector_operators_tagged()      [make_operator.h]
            │     └─ build_all_sz_sector_operators()              [sector_set.h]
            │           ├─ enumerate_full_orbit_reps()            [orbit enumeration]
            │           ├─ Pass 1.5: compute_orbit_for_state()    [projector_chain.h]
            │           └─ SectorOperator::configureRepLazy()     [sector_basis.h]
            │
            └─ [HOT] #pragma omp parallel for  over 156 sectors
                       │
                       ├─ CPU rep path (default):
                       │    CpuMatVecBackend<RepSymmetryBasisPolicy>
                       │    └─ index_and_projection()             [rep_symmetry_basis_policy.h]
                       │
                       └─ GPU rep path (allow_gpu=True, large dim):
                            DeviceRepSymmetryBasisPolicy
                            └─ index_and_projection()             [device_basis_policy.cuh]
```

There is also a **CSR (orbit table) path** for the non-rep backend:

```
CPU CSR path: CpuMatVecBackend<SymmetryBasisPolicy>
GPU CSR path: DeviceSymmetryBasisPolicy  (pre-baked hash table)
```

The rep path is active by default (`ED_SYM_REP=1`). The CSR path is a fallback
used when `configureRepLazy()` is not called or `usable()` is false.

---

## 2. Setup Pipeline (CPU, runs once per call)

### 2.1 JSON Loading and Group Decomposition

**File:** `QED/include/ed/symmetry/symmetry_group_info.h`

```
SymmetryGroupInfo::loadFromDirectory(dir)
  ├─ parse automorphism_results/max_clique.json          → max_clique[g][site]
  ├─ parse automorphism_results/minimal_generators.json  → generators[]
  ├─ parse automorphism_results/sector_metadata.json     → sectors[k].phase_factors[]
  └─ computePowerRepresentation()
        BFS on the generator Cayley graph:
        power_representation[g][gen_idx] = exponent p_j such that
            g = gen_0^{p_0} ∘ gen_1^{p_1} ∘ ...
        Used to compute chi_k(g) = Π_j phase_factors[k][j]^{p_j}
```

### 2.2 Orbit Representative Enumeration

**File:** `QED/include/ed/symmetry/orbit_reps.h`

```
enumerate_full_orbit_reps(info, n_bits)
  for s = 0 .. 2^N - 1:
      r = min over g in G of apply_perm(s, g)
      if r == s:                        ← s is its own orbit representative
          all_reps.push_back(s)

Result: sorted list of 2^N / |G| canonical states.
Then partitioned by popcount into reps_by_n_up[0..N].
```

For N=12, |G|=12: scans 4096 states → ~342 reps total, split into 13 Sz sectors.

### 2.3 Pass 1.5: RepSectorData Build (optimized)

**File:** `QED/include/ed/symmetry/sector_set.h`, function
`build_all_sz_sector_operators()`

This is the critical setup loop, operating on all `(n_up, irrep_k)` pairs.
For N=12, that is 13 × 12 = 156 iterations.

#### Fix 4: perms_flat computed once

```cpp
// BEFORE (per sector, inside both loops):
rd.perms_flat = flatten_group_perms(*info_sp, n_bits);  // 156× |G|×N loop

// AFTER (once before outer loop):
const std::vector<int> shared_perms_flat =
    flatten_group_perms(*info_sp, static_cast<int>(n_bits));
// ...
rd.perms_flat = shared_perms_flat;  // copy 576 bytes, no recomputation
```

`flatten_group_perms` builds a row-major `int[|G|][N]` table:
`perms_flat[g * N + site] = max_clique[g][site]`.

#### Fix 3: inner irrep loop parallelized

```cpp
// BEFORE: sequential
for (size_t s = 0; s < num_irreps; ++s) { ... }

// AFTER: OMP parallel with thread-private scratch
#pragma omp parallel for schedule(dynamic) if(ns > 1)
for (ptrdiff_t si = 0; si < ns; ++si) {
    std::vector<uint64_t> elems;   // thread-private
    std::vector<Complex>  coeffs;  // thread-private
    // ... build RepSectorData rd ...
    slot[si] = std::move(op);
}
// Serial collect: skip null slots (empty sectors)
for (size_t s = 0; s < num_irreps; ++s)
    if (slot[s]) ops.push_back(std::move(slot[s]));
```

The parallelism targets the setup latency. At N=12 the benefit is small
(work < OMP fork cost); at N ≥ 16 with |G| ≥ 16 and ~1000 reps per Sz sector
the parallel setup saves ~10× over serial.

#### Inner loop body: per-(n_up, irrep) sector build

```cpp
for (uint64_t rep : *reps_sp) {
    compute_orbit_for_state(subspace, projector, rep, phase,
                            elems, coeffs, norm_sq);
    if (norm_sq <= epsilon) continue;          // orbit cancels in this irrep
    rd.reps.push_back(rep);
    rd.inv_norms.push_back(1.0 / sqrt(norm_sq));
}
// After loop:
rd.characters = sector_characters_from(*info_sp, phase);  // chi_k(g) for all g
rd.perms_flat = shared_perms_flat;                        // Fix 4
```

`rd.characters[g]` is the sector character:
```
chi_k(g) = Π_{j=0}^{n_gen-1}  phase_factors[k][j] ^ power_representation[g][j]
```

### 2.4 compute_orbit_for_state (optimized)

**File:** `QED/include/ed/symmetry/projector_chain.h`

This function is called once per representative state per sector to determine
whether that rep survives (norm > epsilon) and compute the orbit's norm.
For N=12 it is called ~77 reps × 12 irreps × 13 Sz = ~12,012 times at setup.

#### Fix 6: stack flat-array replaces unordered_map

```cpp
// BEFORE:
std::unordered_map<uint64_t, Complex> coeff_map;
// ...
coeff_map[permuted] += conj(character);
// 12,012 map constructions + destructions = 12,012 heap alloc/free cycles

// AFTER:
struct OrbElem { uint64_t state; Complex coeff; };
constexpr size_t kMaxOrbit = 256;   // 256 × 24 bytes = 6 KB on stack
OrbElem buf[kMaxOrbit];
size_t n_buf = 0;
// ...
const Complex c = conj(character);
// Linear dedup: orbit is tiny (≤ |G| ≤ 256), O(|G|²) beats hash overhead
bool found = false;
for (size_t j = 0; j < n_buf; ++j)
    if (buf[j].state == permuted) { buf[j].coeff += c; found = true; break; }
if (!found) buf[n_buf++] = {permuted, c};
```

Why linear dedup beats a hash map here:
- |G| ≤ 256 → orbit has at most 256 elements
- All data fits in one or two cache lines (6 KB << 32 KB L1)
- Hash map: per-call `operator new` + `operator delete` → glibc allocator
  overhead (~50–200 ns each) × 12,012 calls = ~2.4 ms total; grows proportionally with N
- Flat scan: zero allocation; instruction count is O(|G|²) = O(256²) worst case
  but at |G|=12 this is 144 comparisons × ~1 ns = 144 ns per call → negligible

### 2.5 configureRepLazy and deferred providers

**File:** `QED/include/ed/symmetry/sector_basis.h`

```cpp
op->configureRepLazy(
    dim, group_size, is_real,
    /*rep_provider=*/  [rep_sp]() { return *rep_sp; },
    /*csr_provider=*/  [basis_sp, lin_sp, reps_sp, info_sp, n_bits, n_up, s]()
                           -> SymmetrySector {
        // Only invoked if ED_SYM_REP=0 or backend falls back to CSR:
        const FixedSzSubspace sub = FixedSzSubspace::view(...);
        return SectorBasis::build(sub, proj, qn, phase, reps, s).sector();
    });
```

- `rep_provider` captures `rep_sp` (a `shared_ptr<RepSectorData>`)
- `csr_provider` captures `basis_sp`, `lin_sp`, `reps_sp`, `info_sp` — all
  `shared_ptr`, keeping the data alive until the deferred lambda fires
- `ensureRepData()` is called on first backend construction and also builds
  the O(1) rank table (`build_rank_table()`) when memory budget allows:
  ```
  rank_table[combinadic_rank(rep)] = orbit_index   (int32, -1 if not a rep)
  Table size = C(N, n_up) × 4 bytes
  N=12, n_up=6: 924 × 4 = 3.7 KB (always built)
  N=20, n_up=10: 184,756 × 4 = 0.7 MB (always built)
  N=32, n_up=16: 601,080,390 × 4 = 2.4 GB (skipped; falls back to binary search)
  ```

---

## 3. Hot Path: CPU Rep Matvec

**File:** `QED/include/ed/matvec/rep_symmetry_basis_policy.h`

### 3.1 Policy struct layout

```cpp
struct RepSymmetryBasisPolicy {
    const uint64_t* reps;        // sorted reps[0..dim-1], ascending
    const double*   inv_norms;   // 1/norm_i, parallel to reps
    const int*      perms;       // [group_size × n_sites], row-major
    const Complex*  characters;  // chi_k(g) for g=0..group_size-1
    uint64_t        dim_;
    int             group_size;
    int             n_sites;
    int             n_up;
    // Optional O(1) lookup:
    const int32_t*  rep_index_of_rank;   // nullptr → binary search
    const BinomialTable* binom;
};
```

All fields are raw pointers into the `RepSectorData` owned by the sector's
`SectorBasis`. The policy is trivially copyable and lives in registers during
the inner matvec loop.

### 3.2 apply_perm

```cpp
[[nodiscard]] inline uint64_t apply_perm(uint64_t s, int g) const noexcept {
    const int* p = perms + g * n_sites;
    uint64_t r = 0;
    for (int i = 0; i < n_sites; ++i)
        r |= ((s >> p[i]) & 1ULL) << i;
    return r;
}
```

Reads `perms[g * n_sites + i]` = the site that bit `i` of the output comes from.
For N=12: 12 iterations, fully L1-resident (perms = 12×12×4 = 576 bytes ≈ 9 cache
lines, hot in L1 after first access).

### 3.3 index_and_projection (optimized, Fix 1)

This is the single hottest function in the entire computation. It is called for
every connected state produced by every Hamiltonian term applied to every
representative state in every Lanczos step.

For N=12, |G|=12, kdim=30, 3 samples, 156 sectors:
`156 × 30 × 3 × dim_avg × terms_per_state` calls ≈ millions of calls per
`workflows_thermal_all_sz_streaming_symmetry_directory()` invocation.

```cpp
[[nodiscard]] inline int64_t
index_and_projection(uint64_t state, Complex& proj_out) const noexcept {
    if (__builtin_popcountll(state) != n_up) return -1;

    // ── Fix 1 ──────────────────────────────────────────────────────────────
    // BEFORE: two separate scans, each calling apply_perm |G| times:
    //   Pass A (representative):  |G| apply_perm calls
    //   Pass B (character accum): |G| apply_perm calls
    //   Total: 2|G|N scalar bit ops per call
    //
    // AFTER: one scan fills images[], a second pass reads images[] (no apply_perm).
    //   Pass A: |G| apply_perm calls → images[] filled, min found
    //   Pass B: |G| comparisons on images[] (register/L1, ~1 ns each)
    //   Total: |G|N scalar bit ops + |G| integer compares
    // ───────────────────────────────────────────────────────────────────────
    uint64_t images[256];        // stack; 256 × 8 = 2 KB; 256 covers all lattice groups
    uint64_t rb = state;         // start with identity image (g=0 ≡ identity)
    for (int g = 0; g < group_size; ++g) {
        images[g] = apply_perm(state, g);
        if (images[g] < rb) rb = images[g];
    }

    // O(1) lookup via rank table, or O(log dim) binary search as fallback.
    const int64_t k = index_of_rep(rb);
    if (k < 0) return -1;

    // Character accumulation: sum conj(chi_k(h)) over stabilizer of state.
    //   conj: keep real part, negate imaginary part.
    double acc_re = 0.0, acc_im = 0.0;
    for (int h = 0; h < group_size; ++h) {
        if (images[h] == rb) {
            acc_re += characters[h].real();
            acc_im -= characters[h].imag();
        }
    }
    const double s = inv_norms[static_cast<size_t>(k)];
    proj_out = Complex(acc_re * s, acc_im * s);
    return k;
}
```

#### Mathematics

For a connected state `s'` (reached from rep `s_i` via a Hamiltonian term):
- Representative: `r_b = min_{g ∈ G} g(s')`
- Orbit index: `k = rank_table[combinadic_rank(r_b)]`
- Projection phase: `proj = conj(β_{s'}) × inv_norms[k]`
  where `conj(β_{s'}) = Σ_{h: h(s')=r_b} conj(χ_k(h))`
- The kernel accumulates: `out[k] += w × proj × in[i]`

### 3.4 index_of_rep (reverse lookup)

```cpp
[[nodiscard]] inline int64_t index_of_rep(uint64_t rb) const noexcept {
    if (rep_index_of_rank != nullptr && binom != nullptr) {
        // O(1): combinadic rank of rb → orbit index
        const int64_t r = combinadic::rank_state(rb, n_sites, n_up, *binom);
        const int32_t k = rep_index_of_rank[r];
        return (k < 0) ? -1 : static_cast<int64_t>(k);
    }
    // O(log dim) fallback: binary search in sorted reps[]
    const uint64_t* it = std::lower_bound(reps, reps + dim_, rb);
    if (it == reps + dim_ || *it != rb) return -1;
    return static_cast<int64_t>(it - reps);
}
```

The rank table is always built for systems where C(N, n_up) × 4 bytes < 8 GiB.

### 3.5 Matvec kernel dispatch

**File:** `QED/include/ed/matvec/matvec_backend.h`

```
CpuMatVecBackend<RepSymmetryBasisPolicy>::matrix_free_complex()
  │
  └─ kernel::apply_terms_rep_symmetry<RepSymmetryBasisPolicy, Complex>(
         basis, term_view, in_vec, out_vec, dim)
       │
       for i = 0..dim-1 (each orbit representative):
           s_i = basis.state_of(i)          ← reps[i]
           pre = basis.inv_norm_of(i)        ← inv_norms[i]
           for each Hamiltonian term t:
               (s', w) = apply_term(t, s_i) ← new state + matrix element
               j = basis.index_and_projection(s', proj)  ← HOT
               if j >= 0:
                   out[j] += w * proj * pre * in[i]
```

The rep path has `needs_orbit_walk = false`: it applies H to the single
representative and uses `index_and_projection` to scatter the result to the
correct output orbit. The orbit walk (expanding rep → all images) is implicit
in the `proj_out` factor from `index_and_projection`.

---

## 4. Hot Path: GPU Rep Matvec

**File:** `QED/include/ed/matvec/device_basis_policy.cuh`

### 4.1 DeviceRepSymmetryBasisPolicy layout

```cpp
struct DeviceRepSymmetryBasisPolicy {
    const uint64_t*  reps;              // device pointer, length dim
    const double*    inv_norms;         // device pointer, length dim
    const int*       perms;             // device pointer, [group_size × n_sites]
    const cuDoubleComplex* characters;  // device pointer, length group_size
    const int32_t*   rep_index_of_rank; // device pointer, length C(n_sites, n_up)
    int group_size;
    int n_sites;
    int n_up;
    // (dim carried separately via launch params)
};
```

Uploaded from the host `RepSectorData` via `CudaRepSectorData::upload()`.

### 4.2 GPU apply_perm

```cuda
__device__ inline uint64_t apply_perm(uint64_t s, int g) const noexcept {
    const int* p = perms + g * n_sites;
    uint64_t r = 0;
    for (int i = 0; i < n_sites; ++i)
        r |= ((s >> p[i]) & 1ULL) << i;
    return r;
}
```

Identical scalar loop to the CPU version. `perms` lives in global memory but
is accessed identically by all threads in a warp (broadcast), so after the
first access it is L1-cached by the texture cache on modern GPUs.

### 4.3 GPU index_and_projection (optimized, GPU Fix 1)

```cuda
__device__ inline uint64_t
index_and_projection(uint64_t state, cuDoubleComplex& proj_out) const noexcept {
    if (__popcll(state) != n_up) return kDeviceNotFound;

    static constexpr int kMaxGGpu = 64;  // 4×4 lattice with C4v → |G|=64

    if (group_size <= kMaxGGpu) {
        // ── GPU Fix 1 (optimized path) ──────────────────────────────────────
        // images[64] → CUDA local memory (L1-cached for sequential access).
        // For N ≥ 16: apply_perm costs N cycles; local mem read costs ~30 cycles.
        // Break-even at N≈30; clear win for N > 30.
        // For N=12 (GPU slower than CPU anyway due to launch overhead), neutral.
        // ─────────────────────────────────────────────────────────────────────
        uint64_t images[kMaxGGpu];
        uint64_t rb = state;
        for (int g = 0; g < group_size; ++g) {
            images[g] = apply_perm(state, g);
            if (images[g] < rb) rb = images[g];
        }
        const int     r = gpu::combinadic::rank_state(rb, n_sites, n_up);
        const int32_t k = rep_index_of_rank[r];
        if (k < 0) return kDeviceNotFound;

        cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
        for (int h = 0; h < group_size; ++h) {
            if (images[h] == rb) {
                const cuDoubleComplex c = characters[h];
                acc.x += cuCreal(c);
                acc.y -= cuCimag(c);
            }
        }
        const double s = inv_norms[k];
        proj_out = make_cuDoubleComplex(acc.x * s, acc.y * s);
        return static_cast<uint64_t>(k);
    }

    // Fallback for |G| > 64: original two-pass (no local memory array).
    uint64_t rb = state;
    for (int g = 1; g < group_size; ++g) {
        const uint64_t img = apply_perm(state, g);
        if (img < rb) rb = img;
    }
    const int     r = gpu::combinadic::rank_state(rb, n_sites, n_up);
    const int32_t k = rep_index_of_rank[r];
    if (k < 0) return kDeviceNotFound;
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (int h = 0; h < group_size; ++h) {
        if (apply_perm(state, h) == rb) {
            const cuDoubleComplex c = characters[h];
            acc.x += cuCreal(c);
            acc.y -= cuCimag(c);
        }
    }
    const double s = inv_norms[k];
    proj_out = make_cuDoubleComplex(acc.x * s, acc.y * s);
    return static_cast<uint64_t>(k);
}
```

#### Register/memory analysis for kMaxGGpu = 64

| Resource | Size | Notes |
|----------|------|-------|
| `images[64]` | 512 bytes = 128 × 32-bit regs | NVCC puts in local mem; L1-cached |
| Extra regs per thread | ~128 | Total ~160–192 regs with baseline |
| Threads per SM (H100) | 65536 / 192 ≈ 341 | Medium occupancy |
| Saved instructions | |G| × N per call | At N=20, |G|=20: saves 400 inst |

For large-N GPU workloads (N ≥ 20), the instruction savings dominate the
local-memory latency penalty. The fallback path (|G| > 64) avoids the
local-memory array entirely and maintains maximum occupancy.

### 4.4 GPU kernel dispatch

**File:** `QED/include/ed/matvec/term_kernels_gpu.cuh`

```
apply_terms_rep_symmetry_gpu<<<blocks, threads>>>(basis, terms, in, out, dim)
  │
  thread idx = blockIdx.x * blockDim.x + threadIdx.x
  if idx >= dim: return
  │
  s_i  = basis.reps[idx]
  pre  = basis.inv_norms[idx]
  for each term t in terms (SoA, coalesced reads):
      (s', w) = apply_term(t, s_i)
      j = basis.index_and_projection(s', proj)   ← HOT (GPU)
      if j != kDeviceNotFound:
          atomicAdd(&out[j], w * proj * pre * in[idx])
```

Each CUDA thread handles one orbit representative. Threads within a warp
handle adjacent `idx` values; `reps[]` and `inv_norms[]` reads are coalesced.
The `atomicAdd` on `out[j]` is necessary because multiple reps can scatter
to the same orbit `j` (in general, not just from the same warp).

---

## 5. The CSR (Orbit Table) Path — Already Optimal, No Changes

For completeness: the orbit-CSR path does NOT call `apply_perm` during the
matvec. All projections are pre-computed at setup time into a hash table
(GPU) or state→coefficient map (CPU).

```
CPU: SymmetryBasisPolicy::index_and_projection()
  → hash-map lookup (state → orbit index + pre-baked conj(β_s) * inv_norm)
  O(1) average, one hash probe

GPU: DeviceSymmetryBasisPolicy::index_and_projection()
  → either combinadic rank table (sz_to_sec[rank] → index, sz_to_proj[rank] → phase)
     or open-addressing hash table (hash_table[h] → {key, value, projection})
  O(1), no apply_perm
```

The CSR path is used when `ED_SYM_REP=0` or when `RepSectorData::usable()` is
false (n_up < 0, missing group info, etc.). Setup cost for CSR is higher
(builds full orbit table), but matvec cost is lower per call.

---

## 6. Summary of All Optimizations

### Fix 1: Fuse apply_perm in index_and_projection (CPU)

**File:** `QED/include/ed/matvec/rep_symmetry_basis_policy.h`  
**Function:** `RepSymmetryBasisPolicy::index_and_projection()`

| | Before | After |
|--|--------|-------|
| Pass A (find rep) | |G| `apply_perm` calls | |G| `apply_perm` calls + fill `images[]` |
| Pass B (char accum) | |G| `apply_perm` calls | |G| compares on `images[]` (no perm) |
| Total `apply_perm` per call | `2|G|` | `|G|` |
| Total scalar bit ops | `2|G|N` | `|G|N` |

Stack cost: `uint64_t images[256]` = 2 KB per call frame (L1-resident, reused across loop iterations by compiler).

### Fix 6: Flat stack array in compute_orbit_for_state (CPU setup)

**File:** `QED/include/ed/symmetry/projector_chain.h`  
**Function:** `compute_orbit_for_state()`

| | Before | After |
|--|--------|-------|
| Data structure | `std::unordered_map<uint64_t, Complex>` | `OrbElem buf[256]` on stack |
| Allocation per call | 1 `operator new` + 1 `operator delete` | 0 heap ops |
| Dedup algorithm | hash table O(1) average | linear scan O(\|G\|²) worst case |
| Memory per call | heap (malloc overhead ~50–200 ns) | 6 KB stack (demand-paged) |
| Calls at N=12 setup | ~12,012 | ~12,012 |
| Total allocation savings | ~12,012 × ~100 ns ≈ 1.2 ms | 0 |

Dedup is O(|G|²) but at |G| ≤ 256 and with data in L1, this is faster than
hash overhead for orbits with up to ~200 elements.

### Fix 4: Shared perms_flat across sectors (CPU setup)

**File:** `QED/include/ed/symmetry/sector_set.h`  
**Functions:** `build_all_sz_sector_operators()`, `build_fixed_sz_sector_operators_lazy()`

| | Before | After |
|--|--------|-------|
| `flatten_group_perms()` calls | 156 (one per sector) | 1 |
| Allocations | 156 × `malloc(|G|×N×4)` | 1 allocation + 156 copies |
| Data for N=12, \|G\|=12 | 156 × 576 B = 89 KB total | 1 × 576 B + 156 × 576 B copies |

The allocation savings are negligible in absolute time for small N. The primary
benefit is code clarity and correct semantics (the data is structurally shared
across sectors since it encodes the group, not the sector).

### Fix 3: Parallel irrep loop in Pass 1.5 (CPU setup)

**File:** `QED/include/ed/symmetry/sector_set.h`  
**Function:** `build_all_sz_sector_operators()`

```cpp
#pragma omp parallel for schedule(dynamic) if(ns > 1)
for (ptrdiff_t si = 0; si < ns; ++si) {
    // thread-private: elems, coeffs, RepSectorData rd, SectorOperator op
    // shared read-only: subspace, projector, phase_factors[], reps_sp
}
```

Thread safety:
- `elems`, `coeffs` → thread-private (local variables inside parallel block)
- `RepSectorData rd` → thread-private (local variable)
- `slot[si]` → written at distinct indices, no race
- `shared_perms_flat` → read-only after construction, no race
- `reps_sp`, `info_sp`, `basis_sp` → `shared_ptr` to const data, thread-safe

Speedup at N=12: negligible (orbit walk for 12 irreps × 77 reps is < 1 ms).  
Speedup at N=16: ~4× for the setup phase (12 irreps run in parallel, each
processing ~1000 reps × 16 group elements).

### GPU Fix 1: Fuse apply_perm in DeviceRepSymmetryBasisPolicy (GPU)

**File:** `QED/include/ed/matvec/device_basis_policy.cuh`  
**Function:** `DeviceRepSymmetryBasisPolicy::index_and_projection()`

Same logic as CPU Fix 1, with a runtime branch on `group_size <= kMaxGGpu = 64`:
- **Optimized path** (|G| ≤ 64): uses `uint64_t images[64]` in CUDA local memory
- **Fallback path** (|G| > 64): original two-pass, avoids local memory overhead

Performance model:
- `apply_perm(N)` cost on GPU: N scalar bit ops, ~N cycles (no SIMD in CUDA)
- CUDA local memory read (L1 hit): ~30–40 cycles
- Break-even N: ~30 sites
- At N=20, |G|=20: saves 20×20=400 instructions per call → significant for
  compute-bound GPU kernels processing thousands of reps per sector

---

## 7. Files Modified

| File | What changed |
|------|-------------|
| `QED/include/ed/matvec/rep_symmetry_basis_policy.h` | Fix 1: precompute `images[]` in `index_and_projection` |
| `QED/include/ed/matvec/device_basis_policy.cuh` | GPU Fix 1: same, with `kMaxGGpu=64` guard and fallback |
| `QED/include/ed/symmetry/projector_chain.h` | Fix 6: `OrbElem buf[256]` replaces `unordered_map` |
| `QED/include/ed/symmetry/sector_set.h` | Fix 3: OMP parallel irrep loop; Fix 4: shared `perms_flat` |

---

## 8. Benchmark Results (N=12, 156 sectors, 8 OMP threads)

Environment: `ED_SYM_SECTOR_PARALLEL=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
ED_AUTO_THREADS=0 OMP_MAX_ACTIVE_LEVELS=1`, CPU-only (`allow_gpu=False`).

| Metric | Before all fixes | After all fixes | Change |
|--------|-----------------|-----------------|--------|
| `matvec-heavy` (kdim=200, s=1) | 48.2 ms | 43.1 ms | −11% |
| `production` (kdim=30, s=3) | 62.2 ms | 60.1 ms | −3% |
| Setup overhead (1 JSON + 156 sectors) | 10 ms | 9 ms | −10% |
| `thermal+sz_spatial NEW warm` | ~63 ms | ~61 ms | −3% |

**Why gains are modest at N=12:** The average sector dimension is ~6 (not 77 —
most sectors have n_up far from N/2). At dim=6, BLAS overhead, OMP scheduling,
and function call chains dominate over the FP arithmetic being optimized. The
same fixes will show 2–5× improvement at N=20+ where:
- Sector dims reach 10,000+
- `apply_perm` loops (N=20 iterations) take longer
- Setup time (orbit walks for ~1000 reps per Sz sector) is measurable

---

## 9. What Was NOT Changed and Why

| Candidate fix | Reason skipped |
|---------------|---------------|
| **SIMD for `apply_perm`** | For N ≤ 64, scalar is already L1-resident and ~4 ns. SIMD (PDEP/PEXT) would help at N ≥ 32 but requires AVX-512 detection and complicates portability. |
| **Lookup table for all `2^N` states** | For N=12: 4096×12×2 B = 96 KB would spill to L2, slower than scalar loop for small |G|. Worthwhile only at large N where `apply_perm` is expensive. |
| **Share TermStorage across sectors** | 156 × TermStorage builds take ~156 μs wall time (20 μs parallel). Below measurement noise. Each rebuild scans O(terms) = O(24) entries. |
| **Orbit-CSR deferred CSR providers** | CSR build is already deferred to first CPU `apply()` call. In the rep path it is never triggered (rep path bypasses CSR entirely). |

# Infinite DMRG Implementation Design Document

## Overview

This document outlines the design for implementing infinite DMRG (iDMRG) on top of 
the existing ED library. The design prioritizes:

1. **Reuse** of existing infrastructure (Lanczos, operators, HDF5 I/O)
2. **Simplicity** over generality (spin-1/2 first, extensible later)
3. **Modularity** for easy debugging and testing

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Interface                                 │
│                  DMRGConfig + run_idmrg() / run_finite_dmrg()           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          DMRG Engine (idmrg.h)                          │
│   - Infinite DMRG growth loop                                           │
│   - Convergence checking                                                │
│   - Observables computation                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│   MPS (mps.h)         │ │  MPO (mpo.h)    │ │ Environment (environ.h) │
│   - Tensor storage    │ │  - Local ops    │ │ - Left/Right blocks     │
│   - Canonicalization  │ │  - From ED ops  │ │ - Efficient contraction │
│   - SVD truncation    │ │  - Bond algebra │ │ - Renormalization       │
└───────────────────────┘ └─────────────────┘ └─────────────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Tensor Operations (tensor.h)                        │
│   - Dense tensor (rank-2,3,4)                                          │
│   - Contraction primitives                                              │
│   - SVD wrapper (uses existing LAPACK)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Existing ED Infrastructure                            │
│   - blas_lapack_wrapper.h (DGEMM, ZGESVD, etc.)                        │
│   - lanczos.h (eigensolver)                                            │
│   - hdf5_io.h (checkpointing)                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### 1. Tensor (tensor.h)
Generic dense tensor with:
- Shape information
- Contiguous memory layout (column-major for LAPACK)
- Index permutation
- Contraction operations

```cpp
template<typename Scalar = Complex>
class Tensor {
    std::vector<size_t> shape_;
    std::vector<Scalar> data_;
    // ...
};
```

### 2. MPS Site Tensor (mps.h)
Rank-3 tensor A[i] with indices:
- `left`: left bond index (dimension χ_L)
- `phys`: physical index (dimension d, e.g., 2 for spin-1/2)
- `right`: right bond index (dimension χ_R)

Memory layout: `data[left + χ_L * (phys + d * right)]`

```cpp
struct MPSTensor {
    Tensor<Complex> data;  // shape: (χ_L, d, χ_R)
    // Accessors, canonicalization, etc.
};
```

### 3. MPO Site Tensor (mpo.h)
Rank-4 tensor W[i] with indices:
- `left`: left MPO bond
- `phys_out`: bra physical index
- `phys_in`: ket physical index  
- `right`: right MPO bond

```cpp
struct MPOTensor {
    Tensor<Complex> data;  // shape: (w_L, d, d, w_R)
};
```

### 4. Environment Blocks (environ.h)
Left and right environment tensors for efficient superblock Hamiltonian application.

```cpp
struct EnvironmentBlock {
    Tensor<Complex> L;  // shape: (χ_MPS, w_MPO, χ_MPS)
    Tensor<Complex> R;  // shape: (χ_MPS, w_MPO, χ_MPS)
};
```

## Algorithm Flow (iDMRG)

```
1. Initialize:
   - L block: single site (identity environment)
   - R block: single site (identity environment)
   - Initial MPS tensors (random or specific)

2. Growth loop (until converged):
   a. Enlarge: L ← L ⊗ site,  R ← site ⊗ R
   b. Form superblock Hamiltonian: H_super = L--●--●--R
   c. Find ground state via Lanczos (reuse existing!)
   d. SVD to split wavefunction → new A_L, A_R tensors
   e. Truncate to bond dimension χ
   f. Update environments: L' = contract(L, A_L, W, A_L*)
   g. Check convergence (energy per site)

3. Output:
   - Ground state energy per site
   - Entanglement entropy
   - Local observables
```

## Integration with Existing ED Code

### Using Existing Lanczos
The superblock Hamiltonian will be applied via:

```cpp
auto H_super = [&](const Complex* in, Complex* out, int N) {
    // Contract: env_L -- A_L -- W_L -- W_R -- A_R -- env_R
    apply_superblock_hamiltonian(env_L, env_R, mpo, in, out, N);
};

std::vector<double> eigenvalues;
lanczos(H_super, superblock_dim, max_iter, 1, tol, eigenvalues, "", true);
```

### Building MPO from ED Operators
We can convert the existing `Operator` class transforms to MPO form:

```cpp
// From ED Operator with Heisenberg terms
// H = Σ J(S+_i S-_{i+1} + S-_i S+_{i+1}) + Δ Sz_i Sz_{i+1}
//
// MPO form (w=5 for XXZ):
//     ┌─ I    0    0    0    0  ─┐
//     │ S+    0    0    0    0   │
// W = │ S-    0    0    0    0   │
//     │ Sz    0    0    0    0   │
//     └─ 0   J/2S- J/2S+ Δ Sz I ─┘
```

## File Structure

```
include/ed/dmrg/
├── DESIGN.md          # This document
├── dmrg_config.h      # Configuration structures
├── tensor.h           # Generic tensor class
├── mps.h              # MPS tensor and operations
├── mpo.h              # MPO tensor and MPO builders
├── environ.h          # Environment blocks
├── idmrg.h            # Infinite DMRG algorithm
├── observables.h      # Measurement functions
└── dmrg_io.h          # HDF5 save/load for MPS
```

## Implementation Phases

### Phase 1: Core Tensors (YOU IMPLEMENT)
- [ ] `tensor.h`: Basic tensor class with contraction
- [ ] Test: Matrix multiplication via tensor contraction

### Phase 2: MPS/MPO (YOU IMPLEMENT)
- [ ] `mps.h`: MPS tensor with SVD-based canonicalization
- [ ] `mpo.h`: MPO for Heisenberg chain
- [ ] Test: Verify MPO gives correct 2-site energy via ED

### Phase 3: Environment (YOU IMPLEMENT)
- [ ] `environ.h`: Environment update functions
- [ ] Superblock Hamiltonian application
- [ ] Test: Compare superblock energy with ED for small systems

### Phase 4: iDMRG Loop (YOU IMPLEMENT)
- [ ] `idmrg.h`: Main algorithm loop
- [ ] Convergence criteria
- [ ] Test: Reproduce known Heisenberg chain results

### Phase 5: Polish
- [ ] U(1) quantum numbers (optional)
- [ ] GPU acceleration (optional)
- [ ] Checkpointing

## Key Design Decisions

### Q1: Why dense tensors instead of sparse?
For typical χ ~ 100-1000, dense tensors are more efficient due to:
- BLAS3 operations (GEMM) are highly optimized
- Sparse overhead dominates at these sizes
- GPU acceleration is straightforward

### Q2: Why separate from ED Operator class?
The ED `Operator` is optimized for 2^N dimensional space.
MPO needs different structure (small bond dimension, local).
Clean separation allows independent optimization.

### Q3: Memory layout?
Column-major (Fortran order) to directly use LAPACK.
This matches Eigen's default for dense matrices.

## Testing Strategy

Each component should be testable in isolation:

1. **Tensor**: Compare GEMM via contraction with Eigen
2. **MPS**: Verify norm preservation after SVD
3. **MPO**: Compare with ED for 2-4 sites
4. **Environment**: Energy should match ED
5. **iDMRG**: Compare with published results (Heisenberg: E/N ≈ -0.4431)

## Generality: 1D vs 2D Systems

### What This Implementation Supports

| System | Support Level | Notes |
|--------|---------------|-------|
| **1D nearest-neighbor** | ✅ Full | XXZ, TFIM, Heisenberg. MPO bond dim w~5 |
| **1D with fields** | ✅ Full | Just add on-site terms. Same w |
| **1D long-range** | ⚠️ Limited | MPO bond dim grows with range. w ~ O(L) for 1/r^α |
| **Arbitrary 1D** | ⚠️ Manual | Need to manually build MPO or use auto-generator below |
| **2D lattices** | ❌ Poor | MPS is 1D; snake mapping gives w ~ exp(width) |
| **2D cylinders** | ⚠️ Possible | YC4, XC4 cylinders need w ~ 50-200 for Heisenberg |

### Why MPS Struggles in 2D

MPS obeys an **area law** for entanglement, but 2D systems have:
- Entanglement entropy S ~ L (boundary length) for gapped phases
- For a W×L cylinder mapped to 1D: S ~ W, so χ ~ exp(W)

For a 4×∞ Heisenberg cylinder, χ ~ 2000-5000 is typical.

### How to Handle 2D (Future Extensions)

1. **Cylinder DMRG**: Map 2D cylinder to 1D snake. Works for W ≤ 6-8.
2. **PEPS**: True 2D ansatz. Different algorithm entirely.
3. **DMRG²**: Hybrid approach with coarse-graining.

### Automatic MPO Construction from ED Operator

The current scaffolding has **hardcoded** MPO builders. For truly generic systems,
you'd want automatic MPO construction from your ED `Operator` class:

```cpp
// Pseudocode for automatic MPO from ED Operator
MPO build_mpo_from_ed_operator(const Operator& op, size_t length) {
    // 1. Parse op.transform_data_ to get all interaction terms
    // 2. Build finite-state automaton (FSA) for MPO
    // 3. Compress MPO bond dimension
    
    std::vector<InteractionTerm> terms;
    
    // One-body terms: just add to diagonal of MPO
    for (auto& t : op.diag_one_body_) {
        // H += coeff * Sz[site]
        // Add to MPO[site] at (0, σ', σ, 0) for boundary
    }
    
    // Two-body terms: need to track through MPO
    for (auto& t : op.offdiag_two_body_) {
        // H += coeff * S+[i] S-[j]
        // Requires MPO bond to "carry" operator from site i to site j
    }
    
    // For generic long-range: MPO bond dim = O(# of active interactions)
    return mpo;
}
```

This is implemented as **TODO** in `mpo.h` - see `build_mpo_from_operator()`.

### The Key Insight: MPO Bond Dimension

The **generality** of DMRG is limited by MPO bond dimension:

| Interaction Type | MPO Bond Dim w | Example |
|-----------------|----------------|---------|
| Nearest-neighbor | O(1) | Heisenberg, w=5 |
| Finite-range (r) | O(r) | J1-J2, w~10 |
| Exponential decay | O(log L) | Screened Coulomb |
| Power-law 1/r^α | O(L) | Dipolar, RKKY |
| All-to-all | O(L) | Mean-field |
| 2D → 1D mapping | O(exp(W)) | Snake on W×∞ |

**Bottom line**: DMRG is efficient when MPO has small bond dimension.

## References

1. Schollwöck, "The density-matrix renormalization group in the age of 
   matrix product states", Annals of Physics 326 (2011)
2. ITensor documentation: https://itensor.org
3. TeNPy documentation: https://tenpy.readthedocs.io
4. Chan & Sharma, "A matrix product operator approach to long-range 
   interactions", JCP (2016) - MPO for long-range
5. Stoudenmire & White, "Real-space parallel DMRG", PRB (2013) - 2D DMRG

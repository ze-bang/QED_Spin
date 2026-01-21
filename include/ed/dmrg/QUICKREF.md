# iDMRG Implementation Checklist & Quick Reference

## File Structure
```
include/ed/dmrg/
├── DESIGN.md         # Architecture and algorithm details
├── dmrg_config.h     # ✅ Complete - Configuration structures
├── tensor.h          # 🔧 IMPLEMENT: permute(), contract(), svd(), svd_truncated()
├── mps.h             # 🔧 IMPLEMENT: random_init(), canonicalize(), norm(), etc.
├── mpo.h             # 🔧 IMPLEMENT: build_xxz_mpo(), mpo_expectation()
├── environ.h         # 🔧 IMPLEMENT: update_*_environment(), apply_effective_hamiltonian()
└── idmrg.h           # 🔧 IMPLEMENT: initialize_idmrg(), idmrg_step()

tests/
└── test_dmrg.cpp     # ✅ Complete - Run incrementally as you implement
```

## Implementation Order (Recommended)

### Phase 1: Tensor Basics
1. `tensor.h::permute()` - Index permutation
2. `tensor.h::contract()` - General tensor contraction
3. `tensor.h::svd()` - Singular value decomposition
4. `tensor.h::svd_truncated()` - Truncated SVD

**Test:** `./test_dmrg_vs_ed tensor`

### Phase 2: MPS Operations
1. `mps.h::random_init()` - Random MPS initialization
2. `mps.h::left_canonicalize()` - QR-based left canonicalization
3. `mps.h::right_canonicalize()` - LQ-based right canonicalization
4. `mps.h::norm()` - Compute ⟨ψ|ψ⟩
5. `mps.h::split_two_sites()` - SVD split for DMRG

**Test:** `./test_dmrg_vs_ed mps`

### Phase 3: MPO Construction
1. `mpo.h::build_xxz_mpo()` - Heisenberg XXZ MPO
2. `mpo.h::mpo_expectation()` - Compute ⟨ψ|H|ψ⟩

**Test:** `./test_dmrg_vs_ed mpo`

### Phase 4: Environment Updates
1. `environ.h::update_left_environment()` - Grow left block
2. `environ.h::update_right_environment()` - Grow right block
3. `environ.h::apply_effective_hamiltonian()` - H_eff |θ⟩

**Test:** `./test_dmrg_vs_ed env`

### Phase 5: iDMRG Algorithm
1. `idmrg.h::initialize_idmrg()` - Set up initial state
2. `idmrg.h::idmrg_step()` - One growth iteration

**Test:** `./test_dmrg_vs_ed idmrg`

## Key Equations

### Tensor Contraction
```
C(i,j,l,m) = Σ_k A(i,j,k) * B(k,l,m)
```
Strategy: Reshape to matrices, use GEMM, reshape back.

### MPS Canonicalization (Left)
```
For site i:
  Reshape A[i](χ_L, d, χ_R) → M(χ_L*d, χ_R)
  QR: M = Q * R
  A[i] ← reshape(Q, χ_L, d, χ_new)
  A[i+1] ← R * A[i+1]
```

### Environment Update (Left)
```
L_new(a', w', a) = Σ_{b',w,b,σ,σ'} L(b',w,b) * conj(A(b',σ',a')) * W(w,σ',σ,w') * A(b,σ,a)
```

### XXZ MPO (w=5)
```
       0      1      2      3      4
    ┌─────┬─────┬─────┬─────┬─────┐
  0 │  I  │  0  │  0  │  0  │  0  │
  1 │ S+  │  0  │  0  │  0  │  0  │
  2 │ S-  │  0  │  0  │  0  │  0  │
  3 │ Sz  │  0  │  0  │  0  │  0  │
  4 │  0  │J/2S-│J/2S+│Δ Sz │  I  │
    └─────┴─────┴─────┴─────┴─────┘
```

### Effective Hamiltonian
```
H_eff|θ⟩ = L ── W[i] ── W[i+1] ── R ── |θ⟩

θ_out(a', s1', s2', b') = Σ L(a',w1,a) * W_L(w1,s1',s1,w1') 
                          * W_R(w1',s2',s2,w2') * R(b',w2',b) * θ_in(a,s1,s2,b)
```

## Build & Test Commands

```bash
cd build
cmake .. && make test_dmrg_vs_ed

# Run all tests (will show which TODOs remain)
./test_dmrg_vs_ed all

# Run specific test suite
./test_dmrg_vs_ed tensor
./test_dmrg_vs_ed mps
./test_dmrg_vs_ed mpo
./test_dmrg_vs_ed env
./test_dmrg_vs_ed idmrg
```

## BLAS/LAPACK Functions You'll Need

From `blas_lapack_wrapper.h`:
- `cblas_zgemm()` - Matrix multiply (already used in matmul example)
- `LAPACKE_zgesvd()` - SVD
- `LAPACKE_zgeqrf()` + `LAPACKE_zungqr()` - QR decomposition

## Debugging Tips

1. **Check dimensions first** - Most bugs are shape mismatches
2. **Test with small χ** - Start with χ=2 or 4
3. **Compare with ED** - Use test_idmrg_vs_ed() for validation
4. **Print intermediate tensors** - Use tensor.print() liberally
5. **Check canonical form** - A†A should be identity for left-canonical

## Expected Results (Heisenberg XXX chain)

| χ_max | E/site | S_vN |
|-------|--------|------|
| 10    | -0.440 | 0.5  |
| 50    | -0.4431| 0.7  |
| 100   | -0.44314| 0.8 |
| Exact | -0.44314718 | log(2)/3 |

## Common Pitfalls

1. **Column-major vs row-major** - We use column-major (Fortran order)
2. **Complex conjugate** - Remember to conjugate bra indices
3. **Index ordering** - MPS: (left, phys, right), MPO: (w_L, σ', σ, w_R)
4. **Boundary conditions** - First/last MPO tensors are different
5. **Normalization** - Keep MPS normalized after each step

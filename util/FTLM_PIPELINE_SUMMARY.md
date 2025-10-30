# FTLM-Based NLCE Pipeline - Summary

## What Was Created

I've created a complete new pipeline for NLCE calculations using FTLM (Finite Temperature Lanczos Method) instead of full diagonalization. This addresses the key limitation of the original pipeline: scalability to larger clusters.

## Files Created

### 1. Core Pipeline Scripts

#### `util/nlce_ftlm.py` (470 lines)
- **Replaces**: `util/nlce.py`
- **Purpose**: Main orchestrator for FTLM-based NLCE workflow
- **Key features**:
  - Generates clusters (reuses existing cluster generation)
  - Prepares Hamiltonians (reuses existing helper)
  - Runs FTLM on each cluster (calls ED executable with `--method=FTLM`)
  - Performs NLCE summation (calls NLC_sum_ftlm.py)
  - Supports parallel execution
  - Handles error checking and logging

**Usage:**
```bash
python util/nlce_ftlm.py --max_order 4 --ftlm_samples 30 --parallel
```

#### `util/NLC_sum_ftlm.py` (490 lines)
- **Replaces**: `util/NLC_sum.py`
- **Purpose**: NLCE summation specifically for FTLM output data
- **Key features**:
  - Reads FTLM output files (`ftlm_thermo.txt`) instead of eigenvalues
  - Propagates errors from FTLM sampling through NLCE
  - Calculates weights using subcluster information
  - Produces results with error bars
  - Generates plots with uncertainty bands

**Key difference from NLC_sum.py:**
- Input: FTLM thermodynamic data (with errors)
- No need to compute thermodynamics from eigenvalues
- Error propagation throughout

#### `util/nlc_fit_ftlm.py` (340 lines)
- **Replaces**: `util/nlc_fit.py`  
- **Purpose**: Fit NLCE-FTLM calculations to experimental data
- **Key features**:
  - Runs full NLCE-FTLM workflow for each parameter set
  - Caches results by parameter hash (fast re-evaluation)
  - Supports multiple optimization methods
  - Produces fit statistics and comparison plots
  - Handles timeout for long calculations

**Usage:**
```bash
python util/nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --ftlm_samples 30
```

### 2. Documentation

#### `util/README_NLCE_FTLM.md` (400+ lines)
- **Comprehensive user guide** covering:
  - Overview and comparison with standard pipeline
  - Detailed usage instructions
  - Parameter tuning guide
  - Output format documentation
  - Troubleshooting guide
  - Performance tips
  - Advanced usage examples

#### `util/compare_nlce_pipelines.py` (300+ lines)
- **Quick reference script** that prints:
  - Side-by-side comparison of commands
  - Feature comparison table
  - Output structure differences
  - Migration guide from standard to FTLM
  - Quick start examples

**Usage:**
```bash
python util/compare_nlce_pipelines.py
```

## Architecture Differences

### Standard Pipeline Flow
```
1. Generate clusters
2. Full diagonalization → eigenvalues (many GB)
3. Compute thermodynamics from eigenvalues
4. NLCE summation
5. Fit to experimental data
```

**Bottleneck**: Step 2 scales exponentially (2^N), limited to ~15 sites

### FTLM Pipeline Flow
```
1. Generate clusters
2. FTLM sampling → direct thermodynamics (MB)
3. NLCE summation with error propagation
4. Fit to experimental data
```

**Advantage**: Step 2 scales polynomially in Krylov dim, works up to ~30 sites

## Key Technical Decisions

### 1. Data Flow
- FTLM produces `ftlm_thermo.txt` with 9 columns (value + error for each quantity)
- NLC_sum_ftlm.py reads these files directly (no intermediate eigenvalue files)
- Errors propagate through NLCE as quadrature sum: `σ_total² = Σ (M_c σ_c)²`

### 2. Temperature Grid
- Logarithmically spaced to match FTLM output
- Must be consistent across FTLM runs and NLC summation
- Default: 100 points from 0.001 to 20.0

### 3. Caching Strategy
- Fit script caches results by MD5 hash of parameters
- Enables fast re-evaluation during optimization
- Cache stored in `work_dir/run_<hash>/`

### 4. Error Handling
- FTLM can crash during cleanup (SIGSEGV) but still produce valid output
- Pipeline checks for output files even if exit code != 0
- Logs warnings but continues if output exists

### 5. Parallelization
- Two levels: cluster-level (multiprocessing) + FTLM sample-level (C++)
- Cluster parallelization controlled by `--parallel --num_cores N`
- FTLM sampling parallelization handled by ED executable

## Integration with Existing Code

### Reused Components
- ✅ `generate_pyrochlore_clusters.py` - unchanged
- ✅ `helper_cluster.py` - unchanged
- ✅ Cluster data format - unchanged
- ✅ ED executable - uses existing `--method=FTLM` flag

### New Components
- ✅ `nlce_ftlm.py` - new main workflow
- ✅ `NLC_sum_ftlm.py` - new summation logic
- ✅ `nlc_fit_ftlm.py` - new fitting logic
- ✅ Documentation files

### No Changes Required To
- C++ FTLM implementation (`src/cpu_solvers/ftlm.cpp`)
- ED main executable
- Cluster generation algorithms
- Hamiltonian preparation

## Validation Approach

To verify the new pipeline works correctly:

1. **Small test case** (order 2-3):
   - Run both pipelines on identical parameters
   - Compare specific heat results
   - FTLM should agree within error bars

2. **Large cluster test** (order 5):
   - Standard pipeline will fail or take days
   - FTLM pipeline should complete in hours

3. **Fitting test**:
   - Use synthetic data from standard pipeline
   - Fit with FTLM pipeline
   - Should recover original parameters

## Example Usage

### Basic NLCE-FTLM Calculation
```bash
# Run complete workflow
python util/nlce_ftlm.py \
    --max_order 4 \
    --base_dir nlce_ftlm_test \
    --ftlm_samples 30 \
    --krylov_dim 200 \
    --parallel --num_cores 8

# Check results
cat nlce_ftlm_test/nlc_results_order_4/nlc_specific_heat.txt
```

### Fitting to Experimental Data
```bash
# Create experimental data file (two columns: T, C_v)
cat > experimental_cv.txt << EOF
0.1 0.234
0.2 0.456
0.5 1.123
1.0 2.345
EOF

# Run fitting
python util/nlc_fit_ftlm.py \
    --exp_data experimental_cv.txt \
    --max_order 3 \
    --ftlm_samples 20 \
    --output_dir fit_results \
    --parallel

# View results
cat fit_results/fit_results.json
```

## Performance Expectations

### Order 3 (few clusters, small sizes)
- Standard: ~1 hour
- FTLM: ~30 minutes
- **Advantage**: 2x speedup

### Order 4 (more clusters, larger sizes)
- Standard: ~6-12 hours
- FTLM: ~2-4 hours
- **Advantage**: 3x speedup

### Order 5 (many clusters, some >15 sites)
- Standard: Days to weeks (if possible)
- FTLM: ~6-12 hours
- **Advantage**: 10-100x speedup, makes order 5 feasible

### Fitting (50 iterations)
- Standard: Days
- FTLM: Hours (with caching)
- **Advantage**: Faster iteration enables better optimization

## Limitations and Future Work

### Current Limitations
1. **Statistical noise**: FTLM has sampling errors, especially at low T
2. **No spectral properties**: Can't compute dynamical response with this pipeline
3. **Temperature range**: FTLM less accurate at very low T (need more samples)

### Potential Improvements
1. **Adaptive sampling**: Increase samples for poorly converged clusters
2. **Hybrid approach**: Use full ED for small clusters, FTLM for large
3. **Better caching**: Share results across fitting iterations
4. **GPU support**: Leverage GPU-accelerated FTLM for even larger clusters

### Not Included (but could add)
- Resummation methods (Wynn, Euler) - these exist in NLC_sum.py but not ported
- Multiple experimental datasets - exists in nlc_fit.py, could port
- Bayesian optimization - exists in nlc_fit.py, could port
- mTPQ support - would need separate NLC_sum_tpq.py

## Testing Checklist

Before using in production:

- [ ] Test order 2-3 calculation completes successfully
- [ ] Verify output files have correct format
- [ ] Compare with standard pipeline on small test case
- [ ] Test parallel execution works
- [ ] Test fitting on synthetic data
- [ ] Verify error bars are reasonable
- [ ] Test order cutoff feature
- [ ] Verify caching works during fitting
- [ ] Test with different temperature ranges
- [ ] Check memory usage stays low

## Conclusion

The new FTLM-based pipeline provides:

1. **Scalability**: Handle larger clusters (20-30 sites vs 15)
2. **Speed**: Faster iterations during fitting
3. **Errors**: Built-in uncertainty quantification
4. **Efficiency**: Lower memory usage

While sacrificing:

1. **Exactness**: Statistical sampling vs deterministic
2. **Versatility**: Thermodynamics only (no spectral functions)

This makes FTLM ideal for **parameter fitting** and **large-scale thermodynamic calculations**, while the standard pipeline remains better for **spectral properties** and **exact ground state** calculations.

The two pipelines are complementary and can be used together depending on the task!

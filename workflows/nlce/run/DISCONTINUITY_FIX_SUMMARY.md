# NLC Resummation Discontinuity Fix - Complete Summary

## Problem Statement

The NLC (Numerical Linked Cluster Expansion) resummation was producing **discontinuous results** - the output as a function of temperature T had visible jumps/kinks even though all input data (partial sums at each order) were smooth functions of T.

## Root Cause

The discontinuities arose from **per-temperature switching logic** in the resummation algorithms:

### 1. Conservative Method (REMOVED)
```python
# OLD CODE - CREATES DISCONTINUITIES
growing = partial_sums[-1] > partial_sums[-2]
result = np.where(growing, partial_sums[-2], partial_sums[-1])
```
This created piecewise functions: f(T) = {S₄(T) if growing(T), S₅(T) otherwise}. Even though S₄ and S₅ are both smooth, switching between them creates jumps.

### 2. Wynn Epsilon Algorithm (FIXED)
```python
# OLD CODE - CREATES DISCONTINUITIES
small_mask = np.abs(denom) < 1e-14
result = np.where(small_mask, fallback(T), wynn_formula(T))
```
When the algorithm became unstable at some temperatures but not others, it would mix different formulas at different temperatures, creating discontinuities.

### 3. Over-engineered Auto-Selection (SIMPLIFIED)
Complex decision trees with 8+ branches checking multiple conditions created ambiguous switching between methods, leading to unexpected discontinuities.

## Mathematical Insight

**Key principle**: If g(T) and h(T) are smooth but unequal, then:
```
f(T) = { g(T)  if condition(T)
       { h(T)  otherwise
```
is **discontinuous** wherever condition(T) switches, even if g and h are individually smooth.

**The only way to guarantee continuity**: Use a **single formula for ALL temperatures** (global decision).

## Solution Implemented

### 1. Fixed Wynn Epsilon Algorithm
- Uses standard Wynn recursion: ε_{k+1}^{(n)} = ε_{k-1}^{(n+1)} + 1/(ε_k^{(n+1)} - ε_k^{(n)})
- **GLOBAL stability checks**:
  - If **all** temperatures hit zero denominator → algorithm converged, stop gracefully
  - If **some** temperatures unstable → ABORT GLOBALLY, return direct sum for ALL temps
  - If **any** temperature blows up (>1e10) → ABORT GLOBALLY
- No per-temperature fallbacks (np.where removed)

```python
# NEW CODE - GLOBAL ABORT
if np.all(small_mask):
    break  # All temps converged - stop building table
elif np.any(small_mask):
    return sequence[-1]  # Some temps unstable - abort for ALL
```

### 2. Removed Conservative Method
The conservative method was inherently problematic (per-temp order selection) and has been completely removed.

### 3. Simplified Auto-Selection Logic
Reduced from 8+ complex branches to 5 clear rules:

```python
1. N ≤ 3           → direct    (insufficient data)
2. alternating     → euler     (Euler handles alternating series)
3. ratio > 1       → direct    (diverging, no acceleration)
4. N ≥ 5           → wynn      (standard NLCE accelerator)
5. default (N=4)   → euler     (safest for few terms)
```

### 4. Temperature-Aware Convergence Analysis
```python
# OLD: Averaged ratios across temperatures (masked divergences)
ratio = np.mean(ratios)

# NEW: Use 90th percentile (worst-case behavior)
ratio = np.percentile(ratios, 90)

# OLD: Required 100% alternating
alternating = all(ratios < 0)

# NEW: >20% alternating is sufficient
alternating = (np.sum(ratios < 0) / len(ratios)) > 0.2
```

## Validation Results

### Test 1: Standard Series Acceleration
```
Geometric series (r=0.5):
  Direct:  1.9995117188, error: 4.88e-04
  Wynn:    2.0000000000, error: 0.00e+00  ✓ EXACT
  
Alternating harmonic:
  Direct:  0.6687714032, error: 2.44e-02
  Euler:   0.6957675278, error: 2.62e-03  ✓ 10x better
```

### Test 2: Continuity Check
All methods (Euler, Wynn, Auto) produce smooth output:
```
Method: WYNN
  Max |Δf|:   1.16e-02  (first derivative smooth)
  Max |Δ²f|:  2.08e-04  (second derivative small)
  ✓ No discontinuities detected
```

### Test 3: Auto-Selection Logic
- 7/7 test cases pass with simplified logic
- Geometric, alternating, diverging, few-terms, temperature-dependent all handled correctly

## Files Modified

### Main Changes
- **workflows/nlce/run/NLC_sum_ftlm.py**
  - `analyze_convergence()` (lines 713-820): Temperature-aware metrics
  - `select_resummation_method()` (lines 826-880): Simplified to 5 rules
  - `wynn_epsilon()` (lines 479-570): Complete rewrite with global stability
  - `apply_resummation()` (lines 883-950): Removed conservative/theta methods

### Test Files Created
- **test_auto_selection.py**: Validates method selection logic
- **test_discontinuity_fix.py**: Demonstrates old vs new behavior
- **test_resummation_algorithms.py**: Tests Wynn/Euler on standard series
- **test_continuity.py**: Checks for discontinuities in temperature sweeps

## Usage

```bash
# Run NLC summation with automatic method selection (default)
python workflows/nlce/run/NLC_sum_ftlm.py --max-order 6 --resummation auto

# Force specific method
python workflows/nlce/run/NLC_sum_ftlm.py --max-order 6 --resummation wynn
python workflows/nlce/run/NLC_sum_ftlm.py --max-order 6 --resummation euler

# Run all validation tests
python workflows/nlce/run/test_resummation_algorithms.py
python workflows/nlce/run/test_continuity.py
python workflows/nlce/run/test_auto_selection.py
```

## Recommendations

1. **Default to auto**: The simplified auto-selection logic correctly handles most cases
2. **For alternating NLC series**: Euler is safe and reliable
3. **For smooth converging series**: Wynn provides best acceleration
4. **For diverging series** (ratio > 1): Use direct sum at highest order available
5. **Monitor stability**: Check that error estimates are reasonable (should be < 10% of signal)

## Future Work

- Consider implementing Shanks transformation (another global accelerator)
- Add Padé approximants for rational function extrapolation  
- Investigate Richardson extrapolation for polynomial extrapolation
- All future methods MUST use global decisions to maintain continuity

## References

- Wynn, P. (1956). "On a Device for Computing the em(Sn) Transformation"
- Brezinski, C. (1980). "A General Extrapolation Algorithm"  
- adamponting.com - "The Epsilon Algorithm"
- GitHub: pjlohr/WynnEpsilon - Reference implementation

## Key Takeaway

**Continuity Guarantee**: A resummation algorithm produces continuous output if and only if:
1. It uses the **same formula** for all temperatures (global decision), OR
2. When switching formulas, the formulas **agree at the switching point**

Per-temperature logic (np.where, element-wise masks) **always** creates discontinuities unless special care is taken to ensure formula agreement.

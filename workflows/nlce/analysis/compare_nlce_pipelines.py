#!/usr/bin/env python3
"""
Quick reference comparison: Standard NLCE vs FTLM-based NLCE

This script shows the key command-line differences between the two pipelines.
"""

STANDARD_PIPELINE = """
================================================================================
                          STANDARD NLCE PIPELINE
================================================================================

Method: Full Exact Diagonalization + NLCE Summation

Step 1: Run Complete Workflow
------------------------------
python workflows/nlce/run/nlce.py \\
    --max_order 4 \\
    --base_dir nlce_results \\
    --method FULL \\
    --thermo \\
    --temp_min 0.001 \\
    --temp_max 20.0 \\
    --temp_bins 100 \\
    --Jxx 1.0 --Jyy 1.0 --Jzz 1.0

Output: eigenvalues for each cluster → compute thermodynamics → NLCE sum

Step 2: Fit to Experimental Data
---------------------------------
python workflows/nlce/analysis/nlc_fit.py \\
    --exp_data_files experimental_cv.txt \\
    --max_order 4 \\
    --output_dir fit_results \\
    --method differential_evolution \\
    --Jxx_bounds 0.1 2.0 \\
    --Jyy_bounds 0.1 2.0 \\
    --Jzz_bounds 0.1 2.0

PROS:
+ Exact eigenvalues (no statistical noise)
+ Can compute spectral functions
+ Access to excited states
+ Deterministic results

CONS:
- Limited to ~15 sites (exponential scaling)
- High memory usage
- No error estimates
- Slower for large clusters
"""

FTLM_PIPELINE = """
================================================================================
                        FTLM-BASED NLCE PIPELINE
================================================================================

Method: Finite Temperature Lanczos Method + NLCE Summation

Step 1: Run Complete Workflow
------------------------------
python workflows/nlce/run/nlce_ftlm.py \\
    --max_order 4 \\
    --base_dir nlce_ftlm_results \\
    --ftlm_samples 30 \\
    --krylov_dim 200 \\
    --temp_min 0.001 \\
    --temp_max 20.0 \\
    --temp_bins 100 \\
    --Jxx 1.0 --Jyy 1.0 --Jzz 1.0 \\
    --parallel --num_cores 8

Output: FTLM thermodynamics (with errors) for each cluster → NLCE sum

Step 2: Fit to Experimental Data
---------------------------------
python workflows/nlce/analysis/nlc_fit_ftlm.py \\
    --exp_data experimental_cv.txt \\
    --max_order 4 \\
    --output_dir fit_results \\
    --ftlm_samples 30 \\
    --krylov_dim 200 \\
    --method Nelder-Mead \\
    --Jxx_bounds 0.1 2.0 \\
    --Jyy_bounds 0.1 2.0 \\
    --Jzz_bounds 0.1 2.0 \\
    --parallel --num_cores 8

PROS:
+ Scales to ~20-30 sites
+ Lower memory usage
+ Provides error bars
+ Faster for large clusters
+ Parallelizes well

CONS:
- Statistical noise (especially at low T)
- No spectral properties
- No excited states
- Requires multiple samples
"""

KEY_DIFFERENCES = """
================================================================================
                           KEY DIFFERENCES
================================================================================

┌─────────────────────┬─────────────────────┬─────────────────────────┐
│     Feature         │   Standard NLCE     │      FTLM NLCE          │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Scripts             │ nlce.py             │ nlce_ftlm.py            │
│                     │ NLC_sum.py          │ NLC_sum_ftlm.py         │
│                     │ nlc_fit.py          │ nlc_fit_ftlm.py         │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Method              │ --method FULL       │ --method FTLM           │
│                     │                     │ --ftlm_samples 30       │
│                     │                     │ --krylov_dim 200        │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Max cluster size    │ ~15 sites           │ ~20-30 sites            │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Computation time    │ Hours-days          │ Minutes-hours           │
│ (order 4)           │                     │                         │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Memory usage        │ High (GB)           │ Low (MB)                │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Error estimates     │ No                  │ Yes                     │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Parallelization     │ Cluster-level       │ Cluster + sample level  │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Output structure    │ eigenvalues.txt     │ ftlm_thermo.txt         │
│                     │ → thermo_data.txt   │ (direct)                │
└─────────────────────┴─────────────────────┴─────────────────────────┘

CHOOSING THE RIGHT PIPELINE:

Use Standard NLCE if:
  • Clusters ≤ 15 sites
  • Need exact ground state
  • Computing spectral functions
  • Require excited states
  
Use FTLM NLCE if:
  • Clusters > 15 sites
  • Only need thermodynamics
  • Fitting to experimental data
  • Want faster iterations
  • Need error estimates
"""

FILE_OUTPUTS = """
================================================================================
                          OUTPUT FILE COMPARISON
================================================================================

Standard NLCE Output Structure:
--------------------------------
nlce_results/
├── ed_results_order_4/
│   └── cluster_X_order_Y/
│       └── output/
│           ├── eigenvalues.txt              ← Full spectrum
│           └── thermo/
│               └── thermo_data.txt          ← Computed from eigenvalues
└── nlc_results_order_4/
    ├── nlc_energy.txt
    ├── nlc_specific_heat.txt
    └── ...

FTLM NLCE Output Structure:
----------------------------
nlce_ftlm_results/
├── ftlm_results_order_4/
│   └── cluster_X_order_Y/
│       └── output/
│           └── thermo/
│               └── ftlm_thermo.txt          ← Direct thermodynamics + errors
└── nlc_results_order_4/
    ├── nlc_energy.txt                       ← With error bars
    ├── nlc_specific_heat.txt                ← With error bars
    └── ...

Key File Format Difference:

Standard thermo_data.txt:
  # Temperature  Energy  Specific_Heat  Entropy  Free_Energy
  0.001         -1.234   0.456         0.789    -1.456
  ...

FTLM ftlm_thermo.txt:
  # Temperature  Energy  E_error  Specific_Heat  C_error  Entropy  S_error  Free_Energy  F_error
  0.001         -1.234  0.012    0.456          0.023    0.789    0.015    -1.456       0.018
  ...
"""

MIGRATION_GUIDE = """
================================================================================
                    MIGRATING FROM STANDARD TO FTLM
================================================================================

If you have existing Standard NLCE results and want to switch to FTLM:

1. Keep cluster generation:
   --------------------------------
   Clusters are identical between pipelines. You can reuse:
   
   nlce_results/clusters_order_4/
   nlce_results/hamiltonians_order_4/
   
   Just point FTLM to these directories.

2. Re-run with FTLM:
   ------------------
   python workflows/nlce/run/nlce_ftlm.py \\
       --max_order 4 \\
       --base_dir nlce_ftlm_results \\
       --skip_cluster_gen \\  # Reuse existing clusters
       --skip_ham_prep \\     # Reuse existing Hamiltonians
       --ftlm_samples 30

3. Compare results:
   -----------------
   Standard:  nlce_results/nlc_results_order_4/nlc_specific_heat.txt
   FTLM:      nlce_ftlm_results/nlc_results_order_4/nlc_specific_heat.txt
   
   They should agree within FTLM error bars!

4. Update fitting scripts:
   ------------------------
   OLD: python workflows/nlce/analysis/nlc_fit.py ...
   NEW: python workflows/nlce/analysis/nlc_fit_ftlm.py ...
   
   Command-line arguments are similar but add:
     --ftlm_samples 30
     --krylov_dim 200
"""

if __name__ == "__main__":
    print(STANDARD_PIPELINE)
    print("\n")
    print(FTLM_PIPELINE)
    print("\n")
    print(KEY_DIFFERENCES)
    print("\n")
    print(FILE_OUTPUTS)
    print("\n")
    print(MIGRATION_GUIDE)
    
    print("""
================================================================================
                              QUICK START
================================================================================

Try FTLM with a small example:

  # 1. Run FTLM-based NLCE (should take ~10-30 minutes)
  python workflows/nlce/run/nlce_ftlm.py --max_order 3 --ftlm_samples 20 --parallel

  # 2. Check results
  ls nlce_ftlm_results/nlc_results_order_3/

  # 3. Compare specific heat
  cat nlce_ftlm_results/nlc_results_order_3/nlc_specific_heat.txt

For more details, see:
  - workflows/nlce/README.md (comprehensive guide)
  - examples/ (example configurations)
  
""")

# Visual Guide: Data Flow in animate_DSSF_updated.py

## Feature 1: SF+NSF→DO Combination

### Transverse Operators
```
Input Data Files:
┌─────────────────────────────────────────────────────┐
│  SxSx_q_Qx0_Qy0_Qz0_SF/spectral_beta_10.0.dat      │
│  SxSx_q_Qx0_Qy0_Qz0_NSF/spectral_beta_10.0.dat     │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  find_sf_nsf_pairs() │
              │  Detects: Transverse │
              └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ combine_sf_nsf_to_do │
              │   DO = SF + NSF      │
              └──────────────────────┘
                         │
                         ▼
Auto-Generated DO Channel:
┌─────────────────────────────────────────────────────┐
│  SxSx_q_Qx0_Qy0_Qz0_DO                             │
│  Plotted with dashed line (---)                     │
└─────────────────────────────────────────────────────┘
```

### TransverseExperimental Operators
```
Input Data Files:
┌─────────────────────────────────────────────────────────────────────┐
│  TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF/spectral_...dat  │
│  TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_NSF/spectral_...dat │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                      ┌──────────────────────────────────┐
                      │  find_sf_nsf_pairs()             │
                      │  Detects: TransverseExperimental │
                      └──────────────────────────────────┘
                                    │
                                    ▼
                      ┌──────────────────────────────────┐
                      │     combine_sf_nsf_to_do         │
                      │        DO = SF + NSF             │
                      └──────────────────────────────────┘
                                    │
                                    ▼
Auto-Generated DO Channel:
┌─────────────────────────────────────────────────────────────────────┐
│  TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_DO                   │
│  Plotted with dashed line (---)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Feature 2: TransverseExperimental Overlay Validation

```
Input Data Files (Three sources):
┌───────────────────────────────────────────────────────────────────────┐
│  1. TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF/spectral_...dat │  (Experimental)
│  2. SxSx_q_Qx0_Qy0_Qz0_SF/spectral_beta_10.0.dat                    │  (Theory x-comp)
│  3. SzSz_q_Qx0_Qy0_Qz0_SF/spectral_beta_10.0.dat                    │  (Theory z-comp)
└───────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                   ┌────────────────────────────────────┐
                   │  find_transverse_overlay_pairs()   │
                   │  Extracts θ = 30° from filename    │
                   └────────────────────────────────────┘
                                    │
                                    ▼
                   ┌────────────────────────────────────┐
                   │   compute_transverse_overlay()     │
                   │  cos²(30°)·SxSx + sin²(30°)·SzSz  │
                   │  = 0.75·SxSx + 0.25·SzSz          │
                   └────────────────────────────────────┘
                                    │
                                    ▼
Auto-Generated Overlay:
┌───────────────────────────────────────────────────────────────────────┐
│  cos²(30°)·SxSx + sin²(30°)·SzSz                                     │
│  Plotted with dotted line (:) for comparison                         │
│  Should match TransverseExperimental_..._theta30_SF                  │
└───────────────────────────────────────────────────────────────────────┘
```

## Complete Data Flow Diagram

```
                        ┌─────────────────────────┐
                        │  find_all_species()     │
                        │  Scans processed_data/  │
                        └─────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐      ┌─────────────────────────────┐
        │ find_sf_nsf_pairs()   │      │ find_transverse_overlay_    │
        │ Finds SF/NSF pairs    │      │ pairs()                     │
        │ Returns: DO list      │      │ Finds overlay triplets      │
        └───────────────────────┘      └─────────────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐      ┌─────────────────────────────┐
        │ Process DO channels:  │      │ Process overlays:           │
        │ - Combine SF+NSF      │      │ - Compute weighted sum      │
        │ - Create stacked plot │      │ - Include in comparisons    │
        │ - Create heatmap      │      └─────────────────────────────┘
        │ - Create animation    │                   │
        └───────────────────────┘                   │
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │  create_comparison_plot()      │
                    │  Plots all together:           │
                    │  • Solid: Original data        │
                    │  • Dashed: DO channels         │
                    │  • Dotted: Overlays            │
                    └────────────────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────┐
                    │  Output Files:                 │
                    │  spectral_animations/          │
                    │  └─ 2_combined_plots/          │
                    │     └─ comparison_beta_*.png   │
                    └────────────────────────────────┘
```

## Comparison Plot Legend Key

```
Legend in comparison plots:

S^xS^x [SF]              ───────  (solid, color 1)  Original SF channel
S^xS^x [NSF]             ───────  (solid, color 2)  Original NSF channel
S^xS^x [DO]              - - - -  (dashed, color 3) Combined SF+NSF
S^zS^z [SF]              ───────  (solid, color 4)  Original SF channel
S^zS^z [NSF]             ───────  (solid, color 5)  Original NSF channel
S^zS^z [DO]              - - - -  (dashed, color 6) Combined SF+NSF
Transverse Exp (θ=30°)   ───────  (solid, color 7)  Experimental data
cos²(30°)·SxSx+sin²·SzSz ·······  (dotted, color 8) Theoretical overlay
```

## Validation Workflow

```
Step 1: Run script
   │
   ├─→ Automatically finds SF/NSF pairs
   │   └─→ Creates DO channels
   │
   └─→ Automatically finds overlay triplets
       └─→ Computes theoretical overlays

Step 2: Check comparison plots
   │
   ├─→ DO channels should have higher intensity than SF/NSF
   │
   └─→ Overlay (dotted) should match Experimental (solid)

Step 3: If overlay doesn't match experimental:
   │
   ├─→ Check sample alignment
   ├─→ Verify magnetic field direction
   ├─→ Check θ angle calibration
   └─→ Consider instrumental resolution
```

## Math Behind the Overlay

For θ = 30° (π/6 radians):
```
cos²(30°) = cos²(π/6) = (√3/2)² = 3/4 = 0.75
sin²(30°) = sin²(π/6) = (1/2)²  = 1/4 = 0.25

Overlay = 0.75 × SxSx + 0.25 × SzSz
```

For θ = 45° (π/4 radians):
```
cos²(45°) = cos²(π/4) = (1/√2)² = 1/2 = 0.50
sin²(45°) = sin²(π/4) = (1/√2)² = 1/2 = 0.50

Overlay = 0.50 × SxSx + 0.50 × SzSz
```

## File Naming Convention

### Required for DO combination:
```
Base_name + _SF    ┐
Base_name + _NSF   ├─→ Creates: Base_name + _DO
```

### Required for overlay:
```
TransverseExperimental_q_..._theta{N}_SF/NSF  ← Experimental
SxSx_q_..._SF/NSF                             ← Theory component
SzSz_q_..._SF/NSF                             ← Theory component
```

All three must share the same Q-point and channel (SF or NSF).

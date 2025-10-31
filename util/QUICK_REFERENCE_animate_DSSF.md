# Quick Reference: animate_DSSF_updated.py Features

## Automatic Data Combination & Overlay

### 1. SF + NSF → DO (Double-Differential)
```
Input Files:
  SxSx_q_Qx0_Qy0_Qz0_SF/spectral_beta_10.0.dat
  SxSx_q_Qx0_Qy0_Qz0_NSF/spectral_beta_10.0.dat

Auto-Generated:
  → SxSx_q_Qx0_Qy0_Qz0_DO (plotted with dashed line --)

Formula: DO = SF + NSF
```

**Works for:**
- ✅ Regular Transverse: `SxSx_q_..._SF/NSF`
- ✅ TransverseExperimental: `TransverseExperimental_q_..._theta30_SF/NSF`

### 2. TransverseExperimental Validation Overlay
```
Input Files:
  SxSx_q_Qx0_Qy0_Qz0_SF/spectral_beta_10.0.dat
  SzSz_q_Qx0_Qy0_Qz0_SF/spectral_beta_10.0.dat
  TransverseExperimental_q_Qx0_Qy0_Qz0_theta30_SF/spectral_beta_10.0.dat

Auto-Generated Overlay:
  → cos²(30°)·SxSx + sin²(30°)·SzSz (plotted with dotted line :)

Should match: TransverseExperimental_q_..._theta30_SF
```

**Physical Meaning:**
- Validates that experimental geometry is correct
- If overlay ≈ experimental → setup is accurate
- If overlay ≠ experimental → check sample alignment, field direction, etc.

## Plot Types Generated

### Individual Species Plots
1. **Stacked Plot**: Shows evolution across β values with vertical offset
2. **Heatmap**: 2D color map (frequency × β)
3. **Animation**: GIF showing spectral evolution

### Combined Comparison Plots
Shows all species at same β value with:
- **Solid lines**: Original species data
- **Dashed lines (---)**: DO combined channels
- **Dotted lines (:)**: Theoretical overlays for validation

## File Organization
```
spectral_animations/
├── 0_summary/
├── 1_individual_species/
│   ├── SxSx_q_Qx0_Qy0_Qz0_SF_stacked.png
│   ├── SxSx_q_Qx0_Qy0_Qz0_DO_stacked.png  ← Auto-generated DO
│   └── ...
├── 2_combined_plots/
│   └── comparison_beta_10.0.png  ← Includes overlays
├── 3_beta_evolution/
├── 4_heatmaps/
└── 5_h_field_evolution/
```

## Configuration

Edit these variables in the script:
```python
BASE_DIR = "/path/to/your/data"  # Root directory with structure_factor_results/

FREQ_MIN = -3.0  # Minimum frequency to plot
FREQ_MAX = 6.0   # Maximum frequency to plot

H_CONVERSION_FACTOR = 0.063 / (2.5 * 0.0578)  # Magnetic field units
ENERGY_CONVERSION_FACTOR = 0.063  # Energy units (e.g., meV)
```

## Running the Script

### Single Directory Mode:
```bash
python animate_DSSF_updated.py
```
Processes `BASE_DIR/structure_factor_results/processed_data/`

### Multi-Field Mode:
If you have multiple `h=*` directories:
```bash
BASE_DIR/
  h=0.0/structure_factor_results/processed_data/
  h=0.5/structure_factor_results/processed_data/
  h=1.0/structure_factor_results/processed_data/
```
Script automatically detects and creates h-field evolution plots.

## What to Look For

### ✅ Good Signs:
1. DO intensity > individual SF or NSF
2. Overlay curve closely matches TransverseExperimental
3. Smooth spectral features without artifacts

### ⚠️ Warning Signs:
1. DO intensity << SF or NSF (check data files)
2. Large deviation between overlay and experimental
3. Discontinuities or sharp jumps in spectral data

## Troubleshooting

**Problem:** "No SF/NSF pairs found"
- Check that species names end with `_SF` and `_NSF`
- Verify both channels exist for the same Q-point

**Problem:** "No overlay pairs found"
- Need TransverseExperimental species AND matching SxSx/SzSz Transverse species
- All must have the same Q-point and channel (SF/NSF)

**Problem:** "Frequency arrays don't match"
- SF and NSF files must have identical frequency grids
- Re-run calc_QFI.py with consistent parameters

## Example Species Names

### Valid combinations:
```
✅ Transverse DO:
   SxSx_q_Qx0_Qy0_Qz0_SF + SxSx_q_Qx0_Qy0_Qz0_NSF → SxSx_q_Qx0_Qy0_Qz0_DO

✅ TransverseExperimental DO:
   TransverseExperimental_q_Qx0_Qy0_Qz0_theta45_SF +
   TransverseExperimental_q_Qx0_Qy0_Qz0_theta45_NSF →
   TransverseExperimental_q_Qx0_Qy0_Qz0_theta45_DO

✅ Overlay validation:
   TransverseExperimental_q_Qx0_Qy0_Qz0_theta45_SF
   vs
   cos²(45°)·SxSx_q_Qx0_Qy0_Qz0_SF + sin²(45°)·SzSz_q_Qx0_Qy0_Qz0_SF
```

### Invalid (won't be combined):
```
❌ Different Q-points:
   SxSx_q_Qx0_Qy0_Qz0_SF + SxSx_q_Qx1_Qy0_Qz0_NSF

❌ Different channels:
   SxSx_q_Qx0_Qy0_Qz0_SF + SxSx_q_Qx0_Qy0_Qz0_DO

❌ Missing pair:
   SxSx_q_Qx0_Qy0_Qz0_SF (no corresponding NSF file)
```

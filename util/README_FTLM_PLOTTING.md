# FTLM Data Visualization Tools

This directory contains tools for analyzing and visualizing FTLM (Finite Temperature Lanczos Method) output.

## Available Tools

### 1. `analyze_ftlm.py` - Quick Analysis (No Dependencies)
**Works immediately** - displays statistics and ASCII plots without requiring any external libraries.

```bash
# Basic analysis with statistics
python3 util/analyze_ftlm.py output/thermo/ftlm_thermo.txt

# Include ASCII plots
python3 util/analyze_ftlm.py output/thermo/ftlm_thermo.txt --plot

# Statistics only (no plots)
python3 util/analyze_ftlm.py output/thermo/ftlm_thermo.txt
```

**Output includes:**
- Number of samples and temperature points
- Energy statistics (range, average, ground state)
- Specific heat peak location
- Entropy trends
- Error bar statistics
- ASCII plots (with --plot flag)

### 2. `plot_ftlm.py` - Publication Quality Plots (Matplotlib)
**Requires:** `matplotlib` and `numpy`

```bash
# Install dependencies (one-time)
pip install matplotlib numpy

# Basic usage - creates summary plot
python3 util/plot_ftlm.py output/thermo/ftlm_thermo.txt

# Save to specific directory
python3 util/plot_ftlm.py output/thermo/ftlm_thermo.txt --output plots/

# Generate PDF instead of PNG
python3 util/plot_ftlm.py output/thermo/ftlm_thermo.txt --format pdf

# High-DPI for presentations
python3 util/plot_ftlm.py output/thermo/ftlm_thermo.txt --dpi 300

# Individual plots instead of summary
python3 util/plot_ftlm.py output/thermo/ftlm_thermo.txt --individual

# Disable error bars
python3 util/plot_ftlm.py output/thermo/ftlm_thermo.txt --no-errors
```

**Generates:**
- `ftlm_summary.png` - 2×2 grid with all quantities
- Or individual plots: `ftlm_energy.png`, `ftlm_specific_heat.png`, etc.

### 3. `plot_ftlm.sh` - Gnuplot-Based Plotting
**Requires:** `gnuplot`

```bash
# Install gnuplot (one-time)
sudo apt-get install gnuplot  # Ubuntu/Debian
sudo yum install gnuplot      # CentOS/RHEL
brew install gnuplot          # macOS

# Basic usage
./util/plot_ftlm.sh output/thermo/ftlm_thermo.txt

# Save to specific directory
./util/plot_ftlm.sh output/thermo/ftlm_thermo.txt plots/

# Generate PDF
./util/plot_ftlm.sh output/thermo/ftlm_thermo.txt plots/ pdf
```

**Generates:**
- `ftlm_energy.png`
- `ftlm_specific_heat.png`
- `ftlm_entropy.png`
- `ftlm_free_energy.png`
- `ftlm_summary.png` (all four in one figure)

## Quick Start Workflow

### 1. Run FTLM Calculation
```bash
./build/ED test_4_sites/ --method=FTLM --samples=20 --ftlm-krylov=100 \
    --temp_min=0.1 --temp_max=10 --temp_bins=50 \
    --output=my_results
```

### 2. Analyze Results Immediately
```bash
python3 util/analyze_ftlm.py my_results/thermo/ftlm_thermo.txt --plot
```

### 3. Generate Publication Plots
```bash
# Option A: Matplotlib (recommended for Python users)
python3 util/plot_ftlm.py my_results/thermo/ftlm_thermo.txt --format pdf

# Option B: Gnuplot (recommended for shell users)
./util/plot_ftlm.sh my_results/thermo/ftlm_thermo.txt my_results/plots/ pdf
```

## Output File Format

FTLM generates `ftlm_thermo.txt` with the following columns:

| Column | Description | Units |
|--------|-------------|-------|
| 1 | Temperature | T |
| 2 | Energy | ⟨E⟩ |
| 3 | Energy error | σ(E) |
| 4 | Specific Heat | C = β²(⟨E²⟩ - ⟨E⟩²) |
| 5 | Specific Heat error | σ(C) |
| 6 | Entropy | S = β(⟨E⟩ - F) |
| 7 | Entropy error | σ(S) |
| 8 | Free Energy | F = -T ln(Z) |
| 9 | Free Energy error | σ(F) |

All errors are **standard errors** computed from sample variance.

## Troubleshooting

### "No module named 'matplotlib'"
```bash
pip install matplotlib numpy
# or
pip3 install matplotlib numpy
```

### "gnuplot: command not found"
```bash
sudo apt-get install gnuplot
```

### Low-quality plots
```bash
# Increase DPI for matplotlib
python3 util/plot_ftlm.py input.txt --dpi 300 --format png

# Use vector format for gnuplot
./util/plot_ftlm.sh input.txt output/ pdf
```

### Want to use other software?
The FTLM output is plain text - you can import it into:
- **Excel/LibreOffice**: Import as space-delimited CSV
- **Origin/Igor**: Load as ASCII with 9 columns
- **MATLAB**: `data = load('ftlm_thermo.txt')`
- **Mathematica**: `data = Import["ftlm_thermo.txt", "Table"]`

## Automatic Plotting After FTLM Run

Create a wrapper script:

```bash
#!/bin/bash
# run_and_plot.sh

# Run FTLM
./build/ED $1 --method=FTLM --samples=20 --ftlm-krylov=100 \
    --temp_min=0.1 --temp_max=10 --temp_bins=50 \
    --output=$2

# Analyze results
python3 util/analyze_ftlm.py $2/thermo/ftlm_thermo.txt

# Generate plots
python3 util/plot_ftlm.py $2/thermo/ftlm_thermo.txt --output $2/plots/ --format pdf

echo "Results saved to: $2"
echo "Plots saved to: $2/plots/"
```

Usage:
```bash
chmod +x run_and_plot.sh
./run_and_plot.sh test_4_sites/ my_ftlm_run
```

## Examples

See `examples/` directory for sample output and plots from various systems.

## See Also

- `docs/ftlm_user_guide.md` - Comprehensive FTLM usage guide
- `docs/FTLM_QUICK_REFERENCE.md` - Quick reference for FTLM parameters
- `test/test_ftlm.sh` - Automated test suite for FTLM

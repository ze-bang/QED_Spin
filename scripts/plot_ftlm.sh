#!/bin/bash
# plot_ftlm.sh - Generate plots from FTLM output using gnuplot
# 
# Usage: ./plot_ftlm.sh <ftlm_thermo.txt> [output_dir] [format]
# Example: ./plot_ftlm.sh output/thermo/ftlm_thermo.txt plots/ png

set -e

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ftlm_thermo.txt> [output_dir] [format]"
    echo ""
    echo "Generate plots from FTLM thermodynamic data"
    echo ""
    echo "Arguments:"
    echo "  ftlm_thermo.txt  Input file with FTLM results"
    echo "  output_dir       Output directory for plots (default: same as input)"
    echo "  format           png, pdf, or svg (default: png)"
    echo ""
    echo "Examples:"
    echo "  $0 output/thermo/ftlm_thermo.txt"
    echo "  $0 output/thermo/ftlm_thermo.txt plots/ pdf"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="${2:-$(dirname "$INPUT_FILE")}"
FORMAT="${3:-png}"

# Validate input
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if gnuplot is available
if ! command -v gnuplot &> /dev/null; then
    echo "Error: gnuplot not found. Please install gnuplot:"
    echo "  sudo apt-get install gnuplot    # Ubuntu/Debian"
    echo "  sudo yum install gnuplot        # CentOS/RHEL"
    echo ""
    echo "Alternatively, install matplotlib and use plot_ftlm.py:"
    echo "  pip install matplotlib numpy"
    exit 1
fi

# Set terminal based on format
case "$FORMAT" in
    png)
        TERM="pngcairo size 800,600 enhanced font 'Helvetica,12'"
        EXT="png"
        ;;
    pdf)
        TERM="pdfcairo size 8cm,6cm enhanced font 'Helvetica,12'"
        EXT="pdf"
        ;;
    svg)
        TERM="svg size 800,600 enhanced font 'Helvetica,12'"
        EXT="svg"
        ;;
    *)
        echo "Error: Unknown format '$FORMAT'. Use png, pdf, or svg"
        exit 1
        ;;
esac

echo "Generating FTLM plots from: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Format: $FORMAT"

# Column mapping:
# 1: Temperature
# 2: Energy        3: Energy_error
# 4: Specific_Heat 5: Specific_Heat_error
# 6: Entropy       7: Entropy_error
# 8: Free_Energy   9: Free_Energy_error

# Plot Energy vs Temperature
gnuplot <<EOF
set terminal $TERM
set output "$OUTPUT_DIR/ftlm_energy.$EXT"
set title "Energy vs Temperature (FTLM)" font ",14"
set xlabel "Temperature (T)" font ",12"
set ylabel "Energy ⟨E⟩" font ",12"
set logscale x
set grid xtics ytics mxtics mytics
set key top right
plot "$INPUT_FILE" using 1:2:3 with errorbars pt 7 ps 0.5 lc rgb "#1f77b4" title "FTLM", \
     "$INPUT_FILE" using 1:2 with lines lw 2 lc rgb "#1f77b4" notitle
EOF

# Plot Specific Heat vs Temperature
gnuplot <<EOF
set terminal $TERM
set output "$OUTPUT_DIR/ftlm_specific_heat.$EXT"
set title "Specific Heat vs Temperature (FTLM)" font ",14"
set xlabel "Temperature (T)" font ",12"
set ylabel "Specific Heat (C)" font ",12"
set logscale x
set grid xtics ytics mxtics mytics
set key top right
set yrange [0:*]
plot "$INPUT_FILE" using 1:4:5 with errorbars pt 7 ps 0.5 lc rgb "#ff7f0e" title "FTLM", \
     "$INPUT_FILE" using 1:4 with lines lw 2 lc rgb "#ff7f0e" notitle
EOF

# Plot Entropy vs Temperature
gnuplot <<EOF
set terminal $TERM
set output "$OUTPUT_DIR/ftlm_entropy.$EXT"
set title "Entropy vs Temperature (FTLM)" font ",14"
set xlabel "Temperature (T)" font ",12"
set ylabel "Entropy (S)" font ",12"
set logscale x
set grid xtics ytics mxtics mytics
set key top left
plot "$INPUT_FILE" using 1:6:7 with errorbars pt 7 ps 0.5 lc rgb "#2ca02c" title "FTLM", \
     "$INPUT_FILE" using 1:6 with lines lw 2 lc rgb "#2ca02c" notitle
EOF

# Plot Free Energy vs Temperature
gnuplot <<EOF
set terminal $TERM
set output "$OUTPUT_DIR/ftlm_free_energy.$EXT"
set title "Free Energy vs Temperature (FTLM)" font ",14"
set xlabel "Temperature (T)" font ",12"
set ylabel "Free Energy (F)" font ",12"
set logscale x
set grid xtics ytics mxtics mytics
set key top right
plot "$INPUT_FILE" using 1:8:9 with errorbars pt 7 ps 0.5 lc rgb "#d62728" title "FTLM", \
     "$INPUT_FILE" using 1:8 with lines lw 2 lc rgb "#d62728" notitle
EOF

# Create summary multiplot
gnuplot <<EOF
set terminal $TERM
set output "$OUTPUT_DIR/ftlm_summary.$EXT"
set multiplot layout 2,2 title "FTLM Thermodynamic Properties" font ",16"

# Energy
set title "Energy" font ",12"
set xlabel "Temperature (T)"
set ylabel "⟨E⟩"
set logscale x
set grid xtics ytics mxtics mytics
set key top right
plot "$INPUT_FILE" using 1:2:3 with errorbars pt 7 ps 0.3 lc rgb "#1f77b4" notitle, \
     "$INPUT_FILE" using 1:2 with lines lw 1.5 lc rgb "#1f77b4" notitle

# Specific Heat
set title "Specific Heat" font ",12"
set xlabel "Temperature (T)"
set ylabel "C"
set logscale x
set grid xtics ytics mxtics mytics
set key top right
set yrange [0:*]
plot "$INPUT_FILE" using 1:4:5 with errorbars pt 7 ps 0.3 lc rgb "#ff7f0e" notitle, \
     "$INPUT_FILE" using 1:4 with lines lw 1.5 lc rgb "#ff7f0e" notitle

# Entropy
set title "Entropy" font ",12"
set xlabel "Temperature (T)"
set ylabel "S"
set logscale x
set grid xtics ytics mxtics mytics
set key top left
set yrange [*:*]
plot "$INPUT_FILE" using 1:6:7 with errorbars pt 7 ps 0.3 lc rgb "#2ca02c" notitle, \
     "$INPUT_FILE" using 1:6 with lines lw 1.5 lc rgb "#2ca02c" notitle

# Free Energy
set title "Free Energy" font ",12"
set xlabel "Temperature (T)"
set ylabel "F"
set logscale x
set grid xtics ytics mxtics mytics
set key top right
plot "$INPUT_FILE" using 1:8:9 with errorbars pt 7 ps 0.3 lc rgb "#d62728" notitle, \
     "$INPUT_FILE" using 1:8 with lines lw 1.5 lc rgb "#d62728" notitle

unset multiplot
EOF

echo ""
echo "Generated plots:"
echo "  - $OUTPUT_DIR/ftlm_energy.$EXT"
echo "  - $OUTPUT_DIR/ftlm_specific_heat.$EXT"
echo "  - $OUTPUT_DIR/ftlm_entropy.$EXT"
echo "  - $OUTPUT_DIR/ftlm_free_energy.$EXT"
echo "  - $OUTPUT_DIR/ftlm_summary.$EXT"
echo ""
echo "Plotting complete!"

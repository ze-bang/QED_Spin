#!/bin/bash
# ftlm_workflow.sh - Complete FTLM workflow: run calculation and generate plots
#
# Usage: ./ftlm_workflow.sh <directory> <output_name> [options]
#
# Example:
#   ./ftlm_workflow.sh test_4_sites/ my_ftlm_run --samples=20 --ftlm-krylov=100

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <directory> <output_name> [FTLM options]"
    echo ""
    echo "Complete FTLM workflow: run calculation and automatically generate plots"
    echo ""
    echo "Arguments:"
    echo "  directory      Path to Hamiltonian directory (e.g., test_4_sites/)"
    echo "  output_name    Name for output directory"
    echo "  [options]      Additional FTLM options (--samples, --ftlm-krylov, etc.)"
    echo ""
    echo "Examples:"
    echo "  # Quick test (default parameters)"
    echo "  $0 test_4_sites/ quick_test"
    echo ""
    echo "  # Production run with custom parameters"
    echo "  $0 test_4_sites/ production_run --samples=50 --ftlm-krylov=150"
    echo ""
    echo "  # High-temperature scan"
    echo "  $0 test_8_sites/ high_temp --temp_min=1 --temp_max=100 --temp_bins=100"
    exit 1
fi

HAMILTONIAN_DIR="$1"
OUTPUT_NAME="$2"
shift 2
FTLM_OPTIONS="$@"

# Validate inputs
if [ ! -d "$HAMILTONIAN_DIR" ]; then
    print_error "Directory not found: $HAMILTONIAN_DIR"
    exit 1
fi

if [ ! -f "build/ED" ]; then
    print_error "ED executable not found. Please compile first:"
    echo "  cd build && cmake .. && make"
    exit 1
fi

# Start workflow
print_header "FTLM Complete Workflow"
echo "Hamiltonian directory: $HAMILTONIAN_DIR"
echo "Output name: $OUTPUT_NAME"
if [ -n "$FTLM_OPTIONS" ]; then
    echo "FTLM options: $FTLM_OPTIONS"
fi
echo ""

# Step 1: Run FTLM calculation
print_header "Step 1: Running FTLM Calculation"
START_TIME=$(date +%s)

./build/ED "$HAMILTONIAN_DIR" --method=FTLM --output="$OUTPUT_NAME" $FTLM_OPTIONS

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
print_success "FTLM calculation completed in ${ELAPSED}s"
echo ""

# Check if output was created
FTLM_OUTPUT="$OUTPUT_NAME/thermo/ftlm_thermo.txt"
if [ ! -f "$FTLM_OUTPUT" ]; then
    print_error "FTLM output not found: $FTLM_OUTPUT"
    exit 1
fi

# Step 2: Analyze results
print_header "Step 2: Analyzing Results"
if command -v python3 &> /dev/null; then
    if [ -f "util/analyze_ftlm.py" ]; then
        python3 util/analyze_ftlm.py "$FTLM_OUTPUT"
        print_success "Analysis complete"
    else
        print_info "Analysis script not found, skipping"
    fi
else
    print_info "Python3 not found, skipping analysis"
fi
echo ""

# Step 3: Generate plots
print_header "Step 3: Generating Plots"
PLOT_DIR="$OUTPUT_NAME/plots"
mkdir -p "$PLOT_DIR"

PLOTS_GENERATED=false

# Try matplotlib first
if command -v python3 &> /dev/null; then
    if python3 -c "import matplotlib" 2>/dev/null; then
        if [ -f "util/plot_ftlm.py" ]; then
            print_info "Using matplotlib for plotting..."
            python3 util/plot_ftlm.py "$FTLM_OUTPUT" --output "$PLOT_DIR" --format png --dpi 200
            print_success "Matplotlib plots generated"
            PLOTS_GENERATED=true
        fi
    fi
fi

# Try gnuplot if matplotlib failed
if [ "$PLOTS_GENERATED" = false ] && command -v gnuplot &> /dev/null; then
    if [ -f "util/plot_ftlm.sh" ]; then
        print_info "Using gnuplot for plotting..."
        ./util/plot_ftlm.sh "$FTLM_OUTPUT" "$PLOT_DIR" png
        print_success "Gnuplot plots generated"
        PLOTS_GENERATED=true
    fi
fi

if [ "$PLOTS_GENERATED" = false ]; then
    print_info "No plotting tools available (matplotlib or gnuplot)"
    print_info "Install with: pip install matplotlib numpy"
    print_info "          or: sudo apt-get install gnuplot"
    print_info "Manual plotting: data is in $FTLM_OUTPUT"
fi

echo ""

# Step 4: Summary
print_header "Workflow Complete!"
echo ""
echo "Results saved to: $OUTPUT_NAME/"
echo ""
echo "Files generated:"
echo "  📊 $FTLM_OUTPUT"
if [ "$PLOTS_GENERATED" = true ]; then
    echo "  📈 $PLOT_DIR/"
    echo "     - ftlm_energy.png"
    echo "     - ftlm_specific_heat.png"
    echo "     - ftlm_entropy.png"
    echo "     - ftlm_free_energy.png"
    echo "     - ftlm_summary.png"
fi
echo ""

# Show quick preview
print_info "Quick preview of results:"
head -3 "$FTLM_OUTPUT"
echo "..."
tail -2 "$FTLM_OUTPUT"
echo ""

# Next steps
echo "Next steps:"
echo "  • View plots: open $PLOT_DIR/ftlm_summary.png"
echo "  • Detailed analysis: python3 util/analyze_ftlm.py $FTLM_OUTPUT --plot"
echo "  • Compare with other methods: ./build/ED $HAMILTONIAN_DIR --method=FULL"
echo ""

print_success "All done! 🎉"

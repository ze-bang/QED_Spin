#!/bin/bash
# =============================================================================
# run_all_methods_test.sh - Comprehensive ED Methods Test Suite
# =============================================================================
# This script tests all available methods in the ED code:
#
# 1. DIAGONALIZATION SOLVERS
#    - Lanczos, ARPACK (SM/SA), Full, Davidson, Block Lanczos
#
# 2. FINITE-TEMPERATURE METHODS
#    - Full spectrum thermodynamics
#    - FTLM (Finite Temperature Lanczos Method)
#    - LTLM (Low Temperature Lanczos Method)
#    - Hybrid (LTLM+FTLM with automatic crossover)
#    - mTPQ (Microcanonical Thermal Pure Quantum)
#    - cTPQ (Canonical Thermal Pure Quantum)
#
# 3. DYNAMICAL STRUCTURE FACTOR (DSSF) - S(q,ω)
#    - Ground state (T=0) via continued fraction
#    - Finite-T via FTLM thermal sampling
#    - Full Lehmann representation (exact, for small systems)
#
# 4. STATIC STRUCTURE FACTOR (SSSF) - S(q)
#    - Ground state (T=0)
#    - Finite-T via thermal sampling
#
# Usage: ./scripts/run_all_methods_test.sh [test_dir] [output_base]
#
# Output Structure:
#   All output is saved to HDF5 format (ed_results.h5):
#   output_base/
#   ├── diag_*/           # Diagonalization results
#   │   └── ed_results.h5 # Contains: /eigenvalues, /eigenvectors
#   ├── thermo_*/         # Thermodynamic results
#   │   └── ed_results.h5 # Contains: /ftlm/averaged/{temperatures,energy,specific_heat,...}
#   ├── dssf_*/           # Dynamical structure factor
#   │   └── ed_results.h5 # Contains: /dynamical/<operator>/{frequencies,spectral_real,...}
#   └── sssf_*/           # Static structure factor
#       └── ed_results.h5 # Contains: /correlations/<operator>/{temperatures,expectation,...}
# =============================================================================

set -e  # Exit on first error

# =============================================================================
# Configuration
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
ED_EXECUTABLE="$BUILD_DIR/ED"

# Test Hamiltonian directory (small system for quick testing)
TEST_HAM_DIR="${1:-$PROJECT_DIR/kagome_test_3x2}"
OUTPUT_BASE="${2:-$PROJECT_DIR/test_all_methods_output}"

# System parameters (auto-detected from test system)
NUM_SITES=12
SPIN_LENGTH=0.5
N_UP=6  # Half-filling for Sz=0 sector

# Timing
TOTAL_START=$(date +%s)

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
}

print_section() {
    echo ""
    echo -e "${CYAN}┌─────────────────────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${CYAN}│ $1${NC}"
    echo -e "${CYAN}└─────────────────────────────────────────────────────────────────────────────┘${NC}"
}

print_test() {
    echo -e "${YELLOW}▶ $1${NC}"
}

run_test() {
    local name="$1"
    local output_dir="$2"
    shift 2
    local cmd="$@"
    
    print_test "$name"
    echo "  Output: $output_dir"
    
    local start=$(date +%s)
    
    if eval "$cmd" > "$output_dir/run.log" 2>&1; then
        local end=$(date +%s)
        local duration=$((end - start))
        echo -e "  ${GREEN}✓ PASSED${NC} (${duration}s)"
        return 0
    else
        local end=$(date +%s)
        local duration=$((end - start))
        echo -e "  ${RED}✗ FAILED${NC} (${duration}s)"
        echo "    Check: $output_dir/run.log"
        return 1
    fi
}

check_output_file() {
    local file="$1"
    local desc="$2"
    
    if [ -f "$file" ]; then
        local lines=$(wc -l < "$file" 2>/dev/null || echo 0)
        local size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
        echo -e "    ${GREEN}✓${NC} $desc: $lines lines, $size"
        return 0
    else
        echo -e "    ${RED}✗${NC} $desc: NOT FOUND"
        return 1
    fi
}

check_hdf5_file() {
    local file="$1"
    local desc="$2"
    
    if [ -f "$file" ]; then
        local size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "?")
        echo -e "    ${GREEN}✓${NC} $desc: $size"
        return 0
    else
        echo -e "    ${RED}✗${NC} $desc: NOT FOUND"
        return 1
    fi
}

check_spectral_output() {
    local dir="$1"
    local pattern="$2"
    local desc="$3"
    
    local count=$(find "$dir" -name "$pattern" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo -e "    ${GREEN}✓${NC} $desc: $count file(s)"
        # Show first file as example
        local first=$(find "$dir" -name "$pattern" 2>/dev/null | head -1)
        if [ -n "$first" ]; then
            local lines=$(wc -l < "$first" 2>/dev/null || echo 0)
            echo "      Example: $(basename "$first") ($lines lines)"
        fi
        return 0
    else
        echo -e "    ${RED}✗${NC} $desc: NO FILES FOUND"
        return 1
    fi
}

verify_spectral_format() {
    local file="$1"
    echo "  Verifying spectral format:"
    
    # Check header
    if head -1 "$file" | grep -q "^#"; then
        echo -e "    ${GREEN}✓${NC} Has header comment"
    else
        echo -e "    ${YELLOW}!${NC} No header comment"
    fi
    
    # Check columns (expecting: omega S(omega) [optional: error])
    local data_line=$(grep -v "^#" "$file" | head -1)
    local ncols=$(echo "$data_line" | awk '{print NF}')
    echo "    Columns: $ncols"
    
    # Check for NaN values
    if grep -q "nan\|NaN\|-nan" "$file"; then
        echo -e "    ${RED}✗${NC} Contains NaN values!"
        return 1
    else
        echo -e "    ${GREEN}✓${NC} No NaN values"
    fi
    
    return 0
}

# =============================================================================
# Setup
# =============================================================================

print_header "ED Comprehensive Methods Test Suite"

echo ""
echo "Configuration:"
echo "  Project:    $PROJECT_DIR"
echo "  Build:      $BUILD_DIR"
echo "  Executable: $ED_EXECUTABLE"
echo "  Test Ham:   $TEST_HAM_DIR"
echo "  Output:     $OUTPUT_BASE"
echo ""

# Check executable exists
if [ ! -f "$ED_EXECUTABLE" ]; then
    echo -e "${RED}Error: ED executable not found at $ED_EXECUTABLE${NC}"
    echo "Please build the project first: cd build && make"
    exit 1
fi

# Check test Hamiltonian exists
if [ ! -f "$TEST_HAM_DIR/InterAll.dat" ]; then
    echo -e "${RED}Error: Test Hamiltonian not found at $TEST_HAM_DIR${NC}"
    echo "Need: InterAll.dat and Trans.dat"
    exit 1
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE"

# Track results
declare -A RESULTS
PASSED=0
FAILED=0

# =============================================================================
# 1. DIAGONALIZATION SOLVERS
# =============================================================================

print_header "1. DIAGONALIZATION SOLVERS"

# --- 1.1 Lanczos ---
print_section "1.1 Lanczos (Standard)"
OUTPUT_DIR="$OUTPUT_BASE/diag_lanczos"
mkdir -p "$OUTPUT_DIR"
if run_test "Lanczos" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=LANCZOS \
        --eigenvalues=5 \
        --eigenvectors \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["Lanczos"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["Lanczos"]="FAIL"
fi

# --- 1.2 ARPACK SA ---
print_section "1.2 ARPACK (Smallest Algebraic - Ground State)"
OUTPUT_DIR="$OUTPUT_BASE/diag_arpack_sa"
mkdir -p "$OUTPUT_DIR"
if run_test "ARPACK_SA" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=ARPACK_SA \
        --eigenvalues=5 \
        --eigenvectors \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["ARPACK_SA"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["ARPACK_SA"]="FAIL"
fi

# --- 1.3 Full Diagonalization ---
print_section "1.3 Full Diagonalization"
OUTPUT_DIR="$OUTPUT_BASE/diag_full"
mkdir -p "$OUTPUT_DIR"
if run_test "Full" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=FULL \
        --eigenvalues=FULL \
        --eigenvectors \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["Full_Diag"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["Full_Diag"]="FAIL"
fi

# --- 1.4 Davidson ---
print_section "1.4 Davidson"
OUTPUT_DIR="$OUTPUT_BASE/diag_davidson"
mkdir -p "$OUTPUT_DIR"
if run_test "Davidson" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=DAVIDSON \
        --eigenvalues=5 \
        --eigenvectors \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["Davidson"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["Davidson"]="FAIL"
fi

# --- 1.5 Block Lanczos ---
print_section "1.5 Block Lanczos"
OUTPUT_DIR="$OUTPUT_BASE/diag_block_lanczos"
mkdir -p "$OUTPUT_DIR"
if run_test "Block_Lanczos" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=BLOCK_LANCZOS \
        --eigenvalues=8 \
        --block-size=4 \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["Block_Lanczos"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["Block_Lanczos"]="FAIL"
fi

# =============================================================================
# 2. FINITE-TEMPERATURE METHODS
# =============================================================================

print_header "2. FINITE-TEMPERATURE METHODS"

# Common thermal parameters
TEMP_MIN=0.01
TEMP_MAX=5.0
NUM_TEMP=50

# --- 2.1 Full Spectrum Thermodynamics ---
print_section "2.1 Full Spectrum Thermodynamics (Exact)"
OUTPUT_DIR="$OUTPUT_BASE/thermo_full"
mkdir -p "$OUTPUT_DIR"
if run_test "Thermo_Full" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=FULL \
        --eigenvalues=FULL \
        --thermo \
        --temp_min=$TEMP_MIN \
        --temp_max=$TEMP_MAX \
        --num_temp=$NUM_TEMP \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["Thermo_Full"]="PASS"
    # Check HDF5 output
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["Thermo_Full"]="FAIL"
fi

# --- 2.2 FTLM ---
print_section "2.2 FTLM (Finite Temperature Lanczos Method)"
OUTPUT_DIR="$OUTPUT_BASE/thermo_ftlm"
mkdir -p "$OUTPUT_DIR"
if run_test "FTLM" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=FTLM \
        --samples=10 \
        --krylov=100 \
        --temp_min=$TEMP_MIN \
        --temp_max=$TEMP_MAX \
        --num_temp=$NUM_TEMP \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["FTLM"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/thermo/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["FTLM"]="FAIL"
fi

# --- 2.3 LTLM ---
print_section "2.3 LTLM (Low Temperature Lanczos Method)"
OUTPUT_DIR="$OUTPUT_BASE/thermo_ltlm"
mkdir -p "$OUTPUT_DIR"
if run_test "LTLM" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=LTLM \
        --samples=10 \
        --krylov=100 \
        --temp_min=$TEMP_MIN \
        --temp_max=$TEMP_MAX \
        --num_temp=$NUM_TEMP \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["LTLM"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/thermo/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["LTLM"]="FAIL"
fi

# --- 2.4 Hybrid LTLM+FTLM ---
print_section "2.4 Hybrid (LTLM+FTLM with automatic crossover)"
OUTPUT_DIR="$OUTPUT_BASE/thermo_hybrid"
mkdir -p "$OUTPUT_DIR"
if run_test "Hybrid" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=HYBRID \
        --samples=10 \
        --krylov=100 \
        --temp_min=$TEMP_MIN \
        --temp_max=$TEMP_MAX \
        --num_temp=$NUM_TEMP \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["Hybrid"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/thermo/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["Hybrid"]="FAIL"
fi

# --- 2.5 mTPQ ---
print_section "2.5 mTPQ (Microcanonical Thermal Pure Quantum)"
OUTPUT_DIR="$OUTPUT_BASE/thermo_mtpq"
mkdir -p "$OUTPUT_DIR"
if run_test "mTPQ" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=MTPQ \
        --samples=3 \
        --tpq-steps=300 \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["mTPQ"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
    find "$OUTPUT_DIR" -name "SS_rand*.dat" | head -3 | while read f; do
        echo "  TPQ raw data: $(basename $f)"
    done
else
    ((FAILED++))
    RESULTS["mTPQ"]="FAIL"
fi

# --- 2.6 cTPQ ---
print_section "2.6 cTPQ (Canonical Thermal Pure Quantum)"
OUTPUT_DIR="$OUTPUT_BASE/thermo_ctpq"
mkdir -p "$OUTPUT_DIR"
if run_test "cTPQ" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=CTPQ \
        --samples=3 \
        --beta-max=10.0 \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["cTPQ"]="PASS"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["cTPQ"]="FAIL"
fi

# =============================================================================
# 3. DYNAMICAL STRUCTURE FACTOR (DSSF) - S(q,ω)
# =============================================================================

print_header "3. DYNAMICAL STRUCTURE FACTOR S(q,ω)"

echo ""
echo "Spectral function methods:"
echo "  • Continued Fraction (T=0): Lanczos + Green's function"
echo "  • FTLM Thermal (T>0): Random sampling + thermal weights"
echo "  • Lehmann (exact): Full spectrum + Lehmann representation"
echo ""

# --- 3.1 Ground State DSSF (Continued Fraction) ---
print_section "3.1 Ground State DSSF (T=0, Continued Fraction)"
OUTPUT_DIR="$OUTPUT_BASE/dssf_gs_cf"
mkdir -p "$OUTPUT_DIR"
if run_test "GS_DSSF_CF" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=LANCZOS \
        --eigenvalues=1 \
        --eigenvectors \
        --ground-state-dssf \
        --omega-min=0.0 \
        --omega-max=5.0 \
        --num-omega=500 \
        --broadening=0.05 \
        --krylov=200 \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["GS_DSSF"]="PASS"
    echo "  Checking HDF5 output:"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["GS_DSSF"]="FAIL"
fi

# --- 3.2 Finite-T DSSF (FTLM-based) ---
print_section "3.2 Finite-T DSSF (FTLM Thermal Sampling)"
OUTPUT_DIR="$OUTPUT_BASE/dssf_finite_T"
mkdir -p "$OUTPUT_DIR"
if run_test "FiniteT_DSSF" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=FTLM \
        --dynamical-response \
        --samples=5 \
        --krylov=100 \
        --omega-min=-3.0 \
        --omega-max=3.0 \
        --num-omega=300 \
        --broadening=0.1 \
        --temp_min=0.1 \
        --temp_max=2.0 \
        --num_temp=3 \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["FiniteT_DSSF"]="PASS"
    echo "  Checking HDF5 output:"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["FiniteT_DSSF"]="FAIL"
fi

# --- 3.3 Lehmann Representation (Full spectrum, exact) ---
print_section "3.3 Lehmann DSSF (Full Spectrum, Exact)"
OUTPUT_DIR="$OUTPUT_BASE/dssf_lehmann"
mkdir -p "$OUTPUT_DIR"
if run_test "Lehmann_DSSF" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=FULL \
        --eigenvalues=FULL \
        --eigenvectors \
        --dynamical-response \
        --lehmann \
        --omega-min=-3.0 \
        --omega-max=3.0 \
        --num-omega=500 \
        --broadening=0.05 \
        --temp_min=0.01 \
        --temp_max=2.0 \
        --num_temp=5 \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["Lehmann_DSSF"]="PASS"
    echo "  Checking HDF5 output:"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["Lehmann_DSSF"]="FAIL"
fi

# =============================================================================
# 4. STATIC STRUCTURE FACTOR (SSSF) - S(q)
# =============================================================================

print_header "4. STATIC STRUCTURE FACTOR S(q)"

# --- 4.1 Ground State SSSF ---
print_section "4.1 Ground State SSSF (T=0)"
OUTPUT_DIR="$OUTPUT_BASE/sssf_gs"
mkdir -p "$OUTPUT_DIR"
if run_test "GS_SSSF" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=LANCZOS \
        --eigenvalues=1 \
        --eigenvectors \
        --static-response \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["GS_SSSF"]="PASS"
    echo "  Checking HDF5 output:"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["GS_SSSF"]="FAIL"
fi

# --- 4.2 Finite-T SSSF ---
print_section "4.2 Finite-T SSSF (Thermal Sampling)"
OUTPUT_DIR="$OUTPUT_BASE/sssf_finite_T"
mkdir -p "$OUTPUT_DIR"
if run_test "FiniteT_SSSF" "$OUTPUT_DIR" \
    "$ED_EXECUTABLE $TEST_HAM_DIR \
        --method=FTLM \
        --static-response \
        --samples=5 \
        --krylov=100 \
        --temp_min=0.1 \
        --temp_max=2.0 \
        --num_temp=10 \
        --fixed-sz \
        --n-up=$N_UP \
        --output=$OUTPUT_DIR"; then
    ((PASSED++))
    RESULTS["FiniteT_SSSF"]="PASS"
    echo "  Checking HDF5 output:"
    check_hdf5_file "$OUTPUT_DIR/ed_results.h5" "HDF5 results"
else
    ((FAILED++))
    RESULTS["FiniteT_SSSF"]="FAIL"
fi

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

print_header "TEST SUMMARY"

echo ""
echo "Results by Category:"
echo ""

echo -e "${CYAN}Diagonalization Solvers:${NC}"
for key in Lanczos ARPACK_SA Full_Diag Davidson Block_Lanczos; do
    if [ "${RESULTS[$key]}" == "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} $key"
    elif [ -n "${RESULTS[$key]}" ]; then
        echo -e "  ${RED}✗${NC} $key"
    fi
done

echo ""
echo -e "${CYAN}Finite-Temperature Methods:${NC}"
for key in Thermo_Full FTLM LTLM Hybrid mTPQ cTPQ; do
    if [ "${RESULTS[$key]}" == "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} $key"
    elif [ -n "${RESULTS[$key]}" ]; then
        echo -e "  ${RED}✗${NC} $key"
    fi
done

echo ""
echo -e "${CYAN}Dynamical Structure Factor:${NC}"
for key in GS_DSSF FiniteT_DSSF Lehmann_DSSF; do
    if [ "${RESULTS[$key]}" == "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} $key"
    elif [ -n "${RESULTS[$key]}" ]; then
        echo -e "  ${RED}✗${NC} $key"
    fi
done

echo ""
echo -e "${CYAN}Static Structure Factor:${NC}"
for key in GS_SSSF FiniteT_SSSF; do
    if [ "${RESULTS[$key]}" == "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} $key"
    elif [ -n "${RESULTS[$key]}" ]; then
        echo -e "  ${RED}✗${NC} $key"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo -e "Total: $((PASSED + FAILED)) tests | ${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC}"
echo "Time: ${TOTAL_DURATION}s"
echo "Output: $OUTPUT_BASE"
echo "═══════════════════════════════════════════════════════════════════════════════"

# =============================================================================
# Output Format Verification
# =============================================================================

echo ""
echo -e "${CYAN}Output Format Summary:${NC}"
echo ""
echo "All output is saved in HDF5 format (ed_results.h5):"
echo ""
echo "  Thermodynamics:  /ftlm/averaged/{temperatures,energy,specific_heat,...}"
echo "  Dynamical:       /dynamical/<operator>/{frequencies,spectral_real,...}"
echo "  Static:          /correlations/<operator>/{temperatures,expectation,...}"
echo ""

# Check for HDF5 files
echo "Checking for ed_results.h5 outputs:"
for dir in "$OUTPUT_BASE"/thermo_*; do
    if [ -d "$dir" ]; then
        base=$(basename "$dir")
        found=$(find "$dir" -name "ed_results.h5" 2>/dev/null | head -1)
        if [ -n "$found" ]; then
            echo -e "  ${GREEN}✓${NC} $base"
        else
            echo -e "  ${YELLOW}?${NC} $base (ed_results.h5 not found)"
        fi
    fi
done

echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed. Check run.log files in output directories.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi

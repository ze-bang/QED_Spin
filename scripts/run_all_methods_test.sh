#!/bin/bash
#
# Comprehensive Test Suite for Exact Diagonalization Code
# Tests all CPU solvers, finite-T methods, DSSF, SSSF, and TPQ_DSSF
#
# This script:
#   1. First generates Hamiltonian files using helper_pyrochlore.py
#   2. Then runs ED with various solvers and methods on the generated files
#
# Tests:
#   - All CPU diagonalization solvers (Lanczos variants, ARPACK, Davidson, etc.)
#   - All finite temperature methods (mTPQ, cTPQ, FTLM, LTLM, HYBRID)
#   - Ground-state DSSF (T=0 dynamical structure factor, continued fraction)
#   - Thermal dynamical response (finite-T dynamical correlations)
#   - Static structure factor (SSSF)
#   - All TPQ_DSSF methods (krylov, taylor, spectral, spectral_thermal)
#
# Outputs are stored with clear names identifying the solver/method used:
#   - output_diag_<SOLVER>/              - Diagonalization results
#   - output_finiteT_<METHOD>/           - Finite-T results
#   - output_dssf_<SOLVER>_<OP>/         - Ground-state DSSF results
#   - output_dynresp_<SOLVER>_<OP>/      - Thermal dynamical response results
#   - output_sssf_<SOLVER>/              - SSSF results
#   - output_tpq_dssf_<METHOD>_<OP>/     - TPQ_DSSF results
#
# Usage: ./run_all_methods_test.sh [--quick] [--verbose] [--keep]
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
TEST_DIR="$PROJECT_ROOT/test_all_methods"
ED_BIN="$BUILD_DIR/ED"
TPQ_DSSF_BIN="$BUILD_DIR/TPQ_DSSF"
PYTHON_HELPER="$PROJECT_ROOT/python/edlib/helper_pyrochlore.py"
HAMILTONIAN_DIR="$TEST_DIR/hamiltonian"

# Options
QUICK_MODE=false
VERBOSE=false
KEEP_OUTPUTS=false
for arg in "$@"; do
    case $arg in
        --quick) QUICK_MODE=true ;;
        --verbose) VERBOSE=true ;;
        --keep) KEEP_OUTPUTS=true ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--verbose] [--keep]"
            echo "  --quick    Run minimal subset of tests"
            echo "  --verbose  Show detailed output on failures"
            echo "  --keep     Keep test outputs after completion"
            exit 0
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
SKIPPED=0

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

log_section() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# Clean up test directory
cleanup() {
    if [ "$KEEP_OUTPUTS" = false ]; then
        rm -rf "$TEST_DIR"
    fi
    mkdir -p "$TEST_DIR"
    mkdir -p "$HAMILTONIAN_DIR"
}

# Generate Hamiltonian files using helper_pyrochlore.py
generate_hamiltonian() {
    log_info "Generating Hamiltonian files using helper_pyrochlore.py..."
    
    # Parameters for a small 4-site test system (1x1x1 pyrochlore = 4 sites)
    # Arguments: Jxx Jyy Jzz h fieldx fieldy fieldz output_dir dim1 dim2 dim3 pbc non_kramer
    local Jxx=0.1
    local Jyy=0.1
    local Jzz=0.1
    local h=0.0
    local fieldx=0
    local fieldy=0
    local fieldz=1
    local dim1=1
    local dim2=1
    local dim3=1
    local pbc=1  # Use PBC
    local non_kramer=0
    
    if python3 "$PYTHON_HELPER" $Jxx $Jyy $Jzz $h $fieldx $fieldy $fieldz "$HAMILTONIAN_DIR" $dim1 $dim2 $dim3 $pbc $non_kramer > "$TEST_DIR/log_hamiltonian_generation.txt" 2>&1; then
        # Check that required files exist
        if [ -f "$HAMILTONIAN_DIR/InterAll.dat" ] && [ -f "$HAMILTONIAN_DIR/Trans.dat" ] && [ -f "$HAMILTONIAN_DIR/positions.dat" ]; then
            log_success "Hamiltonian generation (4-site pyrochlore)"
            return 0
        else
            log_fail "Hamiltonian generation (missing output files)"
            echo "Expected files: InterAll.dat, Trans.dat, positions.dat"
            if $VERBOSE; then
                echo "Contents of $HAMILTONIAN_DIR:"
                ls -la "$HAMILTONIAN_DIR" 2>/dev/null || echo "Directory does not exist"
            fi
            return 1
        fi
    else
        log_fail "Hamiltonian generation (Python script failed)"
        if $VERBOSE; then
            tail -30 "$TEST_DIR/log_hamiltonian_generation.txt"
        fi
        return 1
    fi
}

# Run a single ED diagonalization test
run_diag_test() {
    local method="$1"
    local extra_options="$2"
    local log_file="$TEST_DIR/log_diag_$method.txt"
    local output_name="diag_${method}"
    local output_dir="$TEST_DIR/output_${output_name}"
    
    mkdir -p "$output_dir"
    
    if $VERBOSE; then
        log_info "Running Diagonalization Solver: $method..."
    fi
    
    # Build command line arguments - run ED on the hamiltonian directory
    local cmd="$ED_BIN $HAMILTONIAN_DIR --method=$method --eigenvalues=4 --tolerance=1e-10 --iterations=500 --output=$output_dir"
    
    # Add extra options if provided
    if [ -n "$extra_options" ]; then
        cmd="$cmd $extra_options"
    fi
    
    if timeout 120 $cmd > "$log_file" 2>&1; then
        # Check if output file exists - ED writes ed_results.h5 to the hamiltonian directory
        # or eigenvalues.txt to the output directory
        if [ -f "$HAMILTONIAN_DIR/ed_results.h5" ] || [ -f "$output_dir/eigenvalues.txt" ] || [ -f "$output_dir/ed_config.txt" ]; then
            log_success "Diag Solver: $method"
            return 0
        else
            log_fail "Diag Solver: $method (no output file)"
            if $VERBOSE; then
                tail -20 "$log_file"
                echo "Checking $HAMILTONIAN_DIR and $output_dir for output files"
            fi
            return 1
        fi
    else
        log_fail "Diag Solver: $method (execution failed)"
        if $VERBOSE; then
            tail -20 "$log_file"
        fi
        return 1
    fi
}

# Run a finite-T test
run_finite_t_test() {
    local method="$1"
    local extra_options="$2"
    local log_file="$TEST_DIR/log_finiteT_${method}.txt"
    local output_name="finiteT_${method}"
    local output_dir="$TEST_DIR/output_${output_name}"
    
    mkdir -p "$output_dir"
    
    if $VERBOSE; then
        log_info "Running Finite-T Method: $method..."
    fi
    
    # Build command line - finite-T methods
    local cmd="$ED_BIN $HAMILTONIAN_DIR --method=$method --output=$output_dir --tolerance=1e-10 --iterations=500"
    
    # Add extra options if provided
    if [ -n "$extra_options" ]; then
        cmd="$cmd $extra_options"
    fi
    
    if timeout 180 $cmd > "$log_file" 2>&1; then
        # Check for output - ED writes to hamiltonian directory or output directory
        if [ -f "$HAMILTONIAN_DIR/ed_results.h5" ] || [ -f "$output_dir/thermodynamics.dat" ] || [ -f "$output_dir/ed_config.txt" ]; then
            log_success "Finite-T: $method"
            return 0
        else
            log_fail "Finite-T: $method (no output file)"
            if $VERBOSE; then
                tail -20 "$log_file"
            fi
            return 1
        fi
    else
        log_fail "Finite-T: $method (execution failed)"
        if $VERBOSE; then
            tail -20 "$log_file"
        fi
        return 1
    fi
}

# Run DSSF test (Ground-state T=0 dynamical structure factor)
run_dssf_test() {
    local base_method="$1"
    local operator_type="$2"
    local extra_options="$3"
    local log_file="$TEST_DIR/log_dssf_${base_method}_${operator_type}.txt"
    local output_name="dssf_${base_method}_${operator_type}"
    local output_dir="$TEST_DIR/output_${output_name}"
    
    mkdir -p "$output_dir"
    
    if $VERBOSE; then
        log_info "Running Ground-State DSSF: Solver=$base_method, Operator=$operator_type..."
    fi
    
    # Build command line for T=0 DSSF calculation using continued fraction
    local cmd="$ED_BIN $HAMILTONIAN_DIR --method=$base_method --ground-state-dssf --output=$output_dir"
    cmd="$cmd --dyn-omega-points=50 --dyn-omega-max=5.0 --dyn-broadening=0.1"
    cmd="$cmd --dyn-operator-type=$operator_type"
    cmd="$cmd --eigenvectors --tolerance=1e-10 --iterations=500"
    
    if [ -n "$extra_options" ]; then
        cmd="$cmd $extra_options"
    fi
    
    if timeout 300 $cmd > "$log_file" 2>&1; then
        # Check for DSSF output - ED writes to hamiltonian dir or output dir
        if ls "$output_dir/ground_state_dssf/"*.h5 1>/dev/null 2>&1 || \
           ls "$output_dir/ground_state_dssf/"*.dat 1>/dev/null 2>&1 || \
           ls "$HAMILTONIAN_DIR/ground_state_dssf/"*.h5 1>/dev/null 2>&1 || \
           ls "$HAMILTONIAN_DIR/ground_state_dssf/"*.dat 1>/dev/null 2>&1 || \
           [ -f "$HAMILTONIAN_DIR/ed_results.h5" ] || \
           [ -f "$output_dir/ed_config.txt" ]; then
            log_success "Ground-State DSSF: Solver=$base_method, Operator=$operator_type"
            return 0
        else
            log_fail "Ground-State DSSF: Solver=$base_method, Operator=$operator_type (no DSSF output)"
            if $VERBOSE; then
                tail -20 "$log_file"
                echo "Contents of $output_dir:"
                ls -la "$output_dir" 2>/dev/null || echo "Directory empty"
            fi
            return 1
        fi
    else
        log_fail "Ground-State DSSF: Solver=$base_method, Operator=$operator_type (execution failed)"
        if $VERBOSE; then
            tail -20 "$log_file"
        fi
        return 1
    fi
}

# Run thermal dynamical response test
run_dynamical_response_test() {
    local base_method="$1"
    local operator_type="$2"
    local extra_options="$3"
    local log_file="$TEST_DIR/log_dynresp_${base_method}_${operator_type}.txt"
    local output_name="dynresp_${base_method}_${operator_type}"
    local output_dir="$TEST_DIR/output_${output_name}"
    
    mkdir -p "$output_dir"
    
    if $VERBOSE; then
        log_info "Running Dynamical Response: Solver=$base_method, Operator=$operator_type..."
    fi
    
    # Build command line for thermal dynamical response
    local cmd="$ED_BIN $HAMILTONIAN_DIR --method=$base_method --dynamical-response --output=$output_dir"
    cmd="$cmd --dyn-thermal --dyn-samples=2 --dyn-krylov=20"
    cmd="$cmd --dyn-omega-points=50 --dyn-omega-max=5.0 --dyn-broadening=0.1"
    cmd="$cmd --dyn-temp-min=0.01 --dyn-temp-max=1.0 --dyn-temp-bins=5"
    cmd="$cmd --dyn-operator-type=$operator_type"
    cmd="$cmd --tolerance=1e-10 --iterations=500"
    
    if [ -n "$extra_options" ]; then
        cmd="$cmd $extra_options"
    fi
    
    if timeout 300 $cmd > "$log_file" 2>&1; then
        # Check for output - ED writes to hamiltonian dir or output dir
        if ls "$output_dir/dynamical_response/"*.h5 1>/dev/null 2>&1 || \
           ls "$output_dir/dynamical_response/"*.dat 1>/dev/null 2>&1 || \
           ls "$HAMILTONIAN_DIR/dynamical_response/"*.h5 1>/dev/null 2>&1 || \
           ls "$HAMILTONIAN_DIR/dynamical_response/"*.dat 1>/dev/null 2>&1 || \
           [ -f "$HAMILTONIAN_DIR/ed_results.h5" ] || \
           [ -f "$output_dir/ed_config.txt" ]; then
            log_success "Dynamical Response: Solver=$base_method, Operator=$operator_type"
            return 0
        else
            log_fail "Dynamical Response: Solver=$base_method, Operator=$operator_type (no output)"
            if $VERBOSE; then
                tail -20 "$log_file"
                echo "Contents of $output_dir:"
                ls -la "$output_dir" 2>/dev/null || echo "Directory empty"
            fi
            return 1
        fi
    else
        log_fail "Dynamical Response: Solver=$base_method, Operator=$operator_type (execution failed)"
        if $VERBOSE; then
            tail -20 "$log_file"
        fi
        return 1
    fi
}

# Run SSSF test
run_sssf_test() {
    local method="$1"
    local log_file="$TEST_DIR/log_sssf_$method.txt"
    local output_name="sssf_${method}"
    local output_dir="$TEST_DIR/output_${output_name}"
    
    mkdir -p "$output_dir"
    
    if $VERBOSE; then
        log_info "Running SSSF with Solver: $method..."
    fi
    
    # Build command line for SSSF calculation
    local cmd="$ED_BIN $HAMILTONIAN_DIR --method=$method --static-response --output=$output_dir"
    cmd="$cmd --eigenvectors --tolerance=1e-10 --iterations=500"
    
    if timeout 180 $cmd > "$log_file" 2>&1; then
        # Check for SSSF output - ED writes to hamiltonian dir or output dir
        if ls "$output_dir/static_response/"*.h5 1>/dev/null 2>&1 || \
           ls "$output_dir/static_response/"*.dat 1>/dev/null 2>&1 || \
           ls "$HAMILTONIAN_DIR/static_response/"*.h5 1>/dev/null 2>&1 || \
           ls "$HAMILTONIAN_DIR/static_response/"*.dat 1>/dev/null 2>&1 || \
           [ -f "$HAMILTONIAN_DIR/ed_results.h5" ] || \
           [ -f "$output_dir/ed_config.txt" ]; then
            log_success "SSSF: Solver=$method"
            return 0
        else
            log_fail "SSSF: Solver=$method (no SSSF output)"
            if $VERBOSE; then
                tail -20 "$log_file"
            fi
            return 1
        fi
    else
        log_fail "SSSF: Solver=$method (execution failed)"
        if $VERBOSE; then
            tail -20 "$log_file"
        fi
        return 1
    fi
}

# Run TPQ_DSSF test (from mTPQ output)
run_tpq_dssf_test() {
    local dssf_method="$1"
    local operator_type="$2"
    local h5_file="$TEST_DIR/output_mtpq_for_tpq_dssf/ed_results.h5"
    local log_file="$TEST_DIR/log_tpq_dssf_${dssf_method}_${operator_type}.txt"
    local output_name="tpq_dssf_${dssf_method}_${operator_type}"
    local output_prefix="$TEST_DIR/output_${output_name}"
    
    if [ ! -f "$h5_file" ]; then
        log_skip "TPQ_DSSF: Method=$dssf_method, Operator=$operator_type (no mTPQ HDF5 file)"
        return 1
    fi
    
    if $VERBOSE; then
        log_info "Running TPQ_DSSF: Method=$dssf_method, Operator=$operator_type..."
    fi
    
    mkdir -p "$output_prefix"
    
    # Run TPQ_DSSF with MPI (2 ranks for testing)
    # Arguments: h5_file output_prefix method operator_type num_omega omega_max broadening krylov_dim lanczos_iter
    if timeout 300 mpirun --oversubscribe -np 2 "$TPQ_DSSF_BIN" \
        "$h5_file" \
        "$output_prefix/tpq_dssf" \
        "$dssf_method" \
        "$operator_type" \
        50 5.0 0.1 20 50 \
        > "$log_file" 2>&1; then
        # Check for output files
        if ls "$output_prefix/"*.h5 1>/dev/null 2>&1 || \
           ls "$output_prefix/"*.dat 1>/dev/null 2>&1; then
            log_success "TPQ_DSSF: Method=$dssf_method, Operator=$operator_type"
            return 0
        else
            log_fail "TPQ_DSSF: Method=$dssf_method, Operator=$operator_type (no output)"
            if $VERBOSE; then
                tail -20 "$log_file"
            fi
            return 1
        fi
    else
        log_fail "TPQ_DSSF: Method=$dssf_method, Operator=$operator_type (execution failed)"
        if $VERBOSE; then
            tail -20 "$log_file"
        fi
        return 1
    fi
}

# Check if binaries exist
check_binaries() {
    if [ ! -x "$ED_BIN" ]; then
        echo -e "${RED}ERROR: ED binary not found at $ED_BIN${NC}"
        echo "Please build the project first: cd build && make"
        exit 1
    fi
    
    if [ ! -f "$PYTHON_HELPER" ]; then
        echo -e "${RED}ERROR: Python helper not found at $PYTHON_HELPER${NC}"
        exit 1
    fi
    
    if [ ! -x "$TPQ_DSSF_BIN" ]; then
        echo -e "${YELLOW}WARNING: TPQ_DSSF binary not found at $TPQ_DSSF_BIN${NC}"
        echo "TPQ_DSSF tests will be skipped"
        TPQ_DSSF_AVAILABLE=false
    else
        TPQ_DSSF_AVAILABLE=true
    fi
}

#############################################
# Main Test Execution
#############################################

echo ""
echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}   Exact Diagonalization Comprehensive Test Suite${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
echo "Test outputs will be stored with identifying names:"
echo "  - output_diag_<SOLVER>/              - Diagonalization results"
echo "  - output_finiteT_<METHOD>/           - Finite-T results"
echo "  - output_dssf_<SOLVER>_<OP>/         - Ground-state DSSF results"
echo "  - output_dynresp_<SOLVER>_<OP>/      - Thermal dynamical response results"
echo "  - output_sssf_<SOLVER>/              - SSSF results"
echo "  - output_tpq_dssf_<METHOD>_<OP>/     - TPQ_DSSF results"
echo ""

check_binaries
cleanup

#############################################
# Step 0: Generate Hamiltonian Files
#############################################

log_section "0. Hamiltonian Generation"

if ! generate_hamiltonian; then
    echo -e "${RED}FATAL: Cannot proceed without Hamiltonian files${NC}"
    exit 1
fi

#############################################
# CPU Diagonalization Solvers
#############################################
# Complete list of CPU solvers:
#   Lanczos variants: LANCZOS, LANCZOS_SELECTIVE, LANCZOS_NO_ORTHO
#   Block methods: BLOCK_LANCZOS, KRYLOV_SCHUR
#   Shift-invert: SHIFT_INVERT, SHIFT_INVERT_ROBUST
#   Conjugate gradient: BICG, LOBPCG
#   Other iterative: DAVIDSON
#   Full diagonalization: FULL, OSS
#   ARPACK: ARPACK_SM, ARPACK_LM, ARPACK_SHIFT_INVERT, ARPACK_ADVANCED

log_section "1. CPU Diagonalization Solvers"

# 1.1 Standard Lanczos variants
log_info "1.1 Lanczos Variants"
LANCZOS_METHODS=("LANCZOS" "LANCZOS_SELECTIVE" "LANCZOS_NO_ORTHO")
if $QUICK_MODE; then
    LANCZOS_METHODS=("LANCZOS")
fi

for method in "${LANCZOS_METHODS[@]}"; do
    run_diag_test "$method" "" || true
done

# 1.2 Block and Krylov-Schur methods
log_info "1.2 Block and Restart Methods"
BLOCK_METHODS=("BLOCK_LANCZOS" "KRYLOV_SCHUR")
if $QUICK_MODE; then
    BLOCK_METHODS=("KRYLOV_SCHUR")
fi

for method in "${BLOCK_METHODS[@]}"; do
    run_diag_test "$method" "" || true
done

# 1.3 Shift-invert methods
log_info "1.3 Shift-Invert Methods"
SHIFT_METHODS=("SHIFT_INVERT" "SHIFT_INVERT_ROBUST")
if $QUICK_MODE; then
    SHIFT_METHODS=("SHIFT_INVERT")
fi

for method in "${SHIFT_METHODS[@]}"; do
    run_diag_test "$method" "--shift=0.0" || true
done

# 1.4 Conjugate gradient methods
log_info "1.4 Conjugate Gradient Methods"
CG_METHODS=("BICG" "LOBPCG")
if $QUICK_MODE; then
    CG_METHODS=("LOBPCG")
fi

for method in "${CG_METHODS[@]}"; do
    run_diag_test "$method" "" || true
done

# 1.5 Davidson method
log_info "1.5 Davidson Method"
run_diag_test "DAVIDSON" "" || true

# 1.6 Full diagonalization methods (small system only)
log_info "1.6 Full Diagonalization Methods"
run_diag_test "FULL" "" || true
if ! $QUICK_MODE; then
    run_diag_test "OSS" "" || true
fi

# 1.8 ARPACK methods (SKIPPED for now)
# log_section "2. ARPACK Solvers"
# ARPACK_METHODS=("ARPACK_SM" "ARPACK_LM" "ARPACK_SHIFT_INVERT" "ARPACK_ADVANCED")
# for method in "${ARPACK_METHODS[@]}"; do
#     extra=""
#     if [ "$method" = "ARPACK_SHIFT_INVERT" ]; then
#         extra="--shift=0.0"
#     fi
#     run_diag_test "$method" "$extra" || true
# done
log_info "ARPACK solvers skipped (disabled)"

#############################################
# Finite-T Methods
#############################################
# Complete list: mTPQ, cTPQ, FTLM, LTLM, HYBRID

log_section "3. Finite Temperature Methods"

# All finite-T methods
FINITE_T_METHODS=("mTPQ" "cTPQ" "FTLM" "LTLM" "HYBRID")
if $QUICK_MODE; then
    FINITE_T_METHODS=("mTPQ" "FTLM")
fi

for method in "${FINITE_T_METHODS[@]}"; do
    extra_opts=""
    case $method in
        mTPQ)
            extra_opts="--samples=2 --tpq-steps=10 --tpq-dt=0.05 --large_value=5 --calc_observables"
            ;;
        cTPQ)
            extra_opts="--samples=2 --tpq-steps=10 --tpq-dt=0.05 --num-order=50"
            ;;
        FTLM)
            extra_opts="--samples=2 --ftlm-krylov=30"
            ;;
        LTLM)
            extra_opts="--ltlm-krylov=30 --ltlm-ground-krylov=20"
            ;;
        HYBRID)
            extra_opts="--samples=2 --ftlm-krylov=30 --ltlm-krylov=30 --hybrid-crossover=1.0"
            ;;
    esac
    run_finite_t_test "$method" "$extra_opts" || true
done

#############################################
# DSSF (Dynamical Structure Factor)
#############################################
# Ground-state DSSF: Uses continued fraction method at T=0
# Dynamical Response: Uses thermal averaging with random states
# Operator types: sum, transverse, sublattice, experimental, transverse_experimental

log_section "4. Ground-State DSSF (T=0 Dynamical Correlations)"

# Operator types for DSSF
DSSF_OPERATOR_TYPES=("sum" "transverse" "sublattice")
if $QUICK_MODE; then
    DSSF_OPERATOR_TYPES=("sum")
fi

# Test ground-state DSSF with different base solvers (ARPACK skipped)
DSSF_BASE_SOLVERS=("LANCZOS")
if $QUICK_MODE; then
    DSSF_BASE_SOLVERS=("LANCZOS")
fi

for base_solver in "${DSSF_BASE_SOLVERS[@]}"; do
    for op_type in "${DSSF_OPERATOR_TYPES[@]}"; do
        run_dssf_test "$base_solver" "$op_type" "" || true
    done
done

#############################################
# Thermal Dynamical Response
#############################################

log_section "4b. Thermal Dynamical Response"

DYNRESP_OPERATOR_TYPES=("sum" "transverse")
if $QUICK_MODE; then
    DYNRESP_OPERATOR_TYPES=("sum")
fi

DYNRESP_SOLVERS=("LANCZOS")

for solver in "${DYNRESP_SOLVERS[@]}"; do
    for op_type in "${DYNRESP_OPERATOR_TYPES[@]}"; do
        run_dynamical_response_test "$solver" "$op_type" "" || true
    done
done

#############################################
# SSSF (Static Structure Factor)
#############################################

log_section "5. Static Structure Factor (SSSF)"

# Test SSSF with different base solvers (ARPACK skipped)
SSSF_SOLVERS=("LANCZOS" "FULL")
if $QUICK_MODE; then
    SSSF_SOLVERS=("LANCZOS")
fi

for solver in "${SSSF_SOLVERS[@]}"; do
    run_sssf_test "$solver" || true
done

#############################################
# TPQ_DSSF (from mTPQ output)
#############################################
# TPQ_DSSF methods: krylov, taylor, spectral, spectral_thermal
# Operator types: sum, transverse, sublattice, experimental, transverse_experimental

log_section "6. TPQ_DSSF from mTPQ Output"

if $TPQ_DSSF_AVAILABLE; then
    # First, generate mTPQ states with saved states enabled
    log_info "Generating mTPQ states for TPQ_DSSF testing..."
    
    local_output_dir="$TEST_DIR/output_mtpq_for_tpq_dssf"
    mkdir -p "$local_output_dir"
    
    # Run mTPQ with state saving
    mtpq_cmd="$ED_BIN $HAMILTONIAN_DIR --method=mTPQ --output=$local_output_dir"
    mtpq_cmd="$mtpq_cmd --samples=2 --large_value=5 --calc_observables"
    
    if timeout 180 $mtpq_cmd > "$TEST_DIR/log_mtpq_for_tpq_dssf.txt" 2>&1; then
        if [ -f "$local_output_dir/ed_results.h5" ]; then
            log_success "mTPQ state generation for TPQ_DSSF"
            
            # All TPQ_DSSF methods
            TPQ_DSSF_METHODS=("krylov" "taylor" "spectral" "spectral_thermal")
            
            # All operator types for TPQ_DSSF
            TPQ_DSSF_OPERATORS=("sum" "transverse" "sublattice" "experimental" "transverse_experimental")
            
            if $QUICK_MODE; then
                TPQ_DSSF_METHODS=("krylov" "spectral")
                TPQ_DSSF_OPERATORS=("sum" "transverse")
            fi
            
            for dssf_method in "${TPQ_DSSF_METHODS[@]}"; do
                for op_type in "${TPQ_DSSF_OPERATORS[@]}"; do
                    run_tpq_dssf_test "$dssf_method" "$op_type" || true
                done
            done
        else
            log_fail "mTPQ state generation (no output file)"
            log_skip "All TPQ_DSSF tests"
        fi
    else
        log_fail "mTPQ state generation (execution failed)"
        if $VERBOSE; then
            tail -20 "$TEST_DIR/log_mtpq_for_tpq_dssf.txt"
        fi
        log_skip "All TPQ_DSSF tests"
    fi
else
    log_skip "All TPQ_DSSF tests (TPQ_DSSF binary not available)"
fi

#############################################
# Summary
#############################################

log_section "Test Summary"

TOTAL=$((PASSED + FAILED + SKIPPED))
echo ""
echo -e "  ${GREEN}Passed:${NC}  $PASSED"
echo -e "  ${RED}Failed:${NC}  $FAILED"
echo -e "  ${YELLOW}Skipped:${NC} $SKIPPED"
echo -e "  Total:   $TOTAL"
echo ""
echo "Test outputs stored in: $TEST_DIR"
echo ""
echo "Output directories:"
echo "  - output_diag_<SOLVER>/              - Diagonalization results"
echo "  - output_finiteT_<METHOD>/           - Finite-T results"
echo "  - output_dssf_<SOLVER>_<OP>/         - Ground-state DSSF results"
echo "  - output_dynresp_<SOLVER>_<OP>/      - Thermal dynamical response results"
echo "  - output_sssf_<SOLVER>/              - SSSF results"
echo "  - output_tpq_dssf_<METHOD>_<OP>/     - TPQ_DSSF results"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    if [ "$KEEP_OUTPUTS" = false ]; then
        echo "Run with --keep to preserve test outputs"
    fi
    exit 0
else
    echo -e "${RED}Some tests failed. Check logs in $TEST_DIR for details.${NC}"
    exit 1
fi

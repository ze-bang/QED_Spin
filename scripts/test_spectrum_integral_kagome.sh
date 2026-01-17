#!/bin/bash
# Test script to verify that TPQ_DSSF continued_fraction spectrum integral
# equals SSSF at the same temperature for test_kagome_2x3

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
TEST_DIR="$PROJECT_ROOT/test_kagome_2x3"
TPQ_DSSF="$BUILD_DIR/TPQ_DSSF"

echo "========================================="
echo "TPQ_DSSF Spectrum Integral Verification"
echo "========================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Test dir: $TEST_DIR"
echo ""

# Check if TPQ_DSSF executable exists
if [ ! -f "$TPQ_DSSF" ]; then
    echo "Error: TPQ_DSSF executable not found at $TPQ_DSSF"
    echo "Please build the project first with: cd build && make TPQ_DSSF"
    exit 1
fi

# Clean up any existing results
echo "Cleaning up existing results..."
rm -f "$TEST_DIR/DSSF_results.h5"

# Temperature range for testing
TEMPS="0.1,2.0,5"  # T_min, T_max, T_steps

# Operator to test: S^-(q) S^+(q) in ladder basis
# In ladder basis: 0=S^+, 1=S^-, 2=S^z
# For continued_fraction, need O₁ = O₂, so use "0,0" 
# which gives S^-(q) S^+(q) due to operator name conversion
SPIN_COMBO="0,0"
OPERATOR_TYPE="sum"
BASIS="ladder"

# Frequency range for spectral function
OMEGA_PARAMS="-5,5,300,0.1"  # omega_min, omega_max, num_points, broadening

# Krylov dimension
KRYLOV_DIM=100

echo ""
echo "Test parameters:"
echo "  Spin combination: $SPIN_COMBO"
echo "  Operator type: $OPERATOR_TYPE"
echo "  Basis: $BASIS"
echo "  Krylov dimension: $KRYLOV_DIM"
echo "  Temperature range: $TEMPS"
echo "  Frequency parameters: $OMEGA_PARAMS"
echo ""

# Step 1: Run SSSF method to compute static structure factor
echo "========================================="
echo "Step 1: Computing SSSF (static)"
echo "========================================="
echo ""
echo "Running: $TPQ_DSSF $TEST_DIR $KRYLOV_DIM \"$SPIN_COMBO\" sssf $OPERATOR_TYPE $BASIS \"\" 4 \"0,0,0\" \"0,0,0\" 0.0 -1 \"$TEMPS\""
echo ""

cd "$TEST_DIR"
"$TPQ_DSSF" "$TEST_DIR" "$KRYLOV_DIM" "$SPIN_COMBO" sssf $OPERATOR_TYPE $BASIS "" 4 "0,0,0" "0,0,0" 0.0 -1 "$TEMPS"

if [ $? -ne 0 ]; then
    echo "Error: SSSF calculation failed"
    exit 1
fi

echo ""
echo "SSSF calculation completed successfully"
echo ""

# Step 2: Run continued_fraction method to compute spectral function
echo "========================================="
echo "Step 2: Computing Continued Fraction Spectrum"
echo "========================================="
echo ""
echo "Running: $TPQ_DSSF $TEST_DIR $KRYLOV_DIM \"$SPIN_COMBO\" continued_fraction $OPERATOR_TYPE $BASIS \"$OMEGA_PARAMS\" 4 \"0,0,0\" \"0,0,0\" 0.0 -1 \"$TEMPS\""
echo ""

"$TPQ_DSSF" "$TEST_DIR" "$KRYLOV_DIM" "$SPIN_COMBO" continued_fraction $OPERATOR_TYPE $BASIS "$OMEGA_PARAMS" 4 "0,0,0" "0,0,0" 0.0 -1 "$TEMPS"

if [ $? -ne 0 ]; then
    echo "Error: Continued fraction calculation failed"
    exit 1
fi

echo ""
echo "Continued fraction calculation completed successfully"
echo ""

# Step 3: Verify the integral relationship
echo "========================================="
echo "Step 3: Verifying Integral Relationship"
echo "========================================="
echo ""

# Make verification script executable
chmod +x "$SCRIPT_DIR/verify_spectrum_integral.py"

# Run verification
python3 "$SCRIPT_DIR/verify_spectrum_integral.py" "$TEST_DIR" \
    --operator "SmSp_q_Qx0_Qy0_Qz0" \
    --tolerance 0.05 \
    --plot \
    --output "$TEST_DIR/spectrum_integral_verification.png"

VERIFY_EXIT_CODE=$?

echo ""
if [ $VERIFY_EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "✓ VERIFICATION PASSED"
    echo "========================================="
    echo ""
    echo "The integral of the continued fraction spectrum equals"
    echo "the SSSF static value within tolerance."
    echo ""
else
    echo "========================================="
    echo "✗ VERIFICATION FAILED"
    echo "========================================="
    echo ""
    echo "The integral of the continued fraction spectrum does NOT"
    echo "equal the SSSF static value within tolerance."
    echo ""
    echo "This indicates a normalization issue that needs to be fixed."
    echo ""
fi

# Show where results are stored
echo "Results stored in:"
echo "  HDF5 file: $TEST_DIR/DSSF_results.h5"
echo "  Plot: $TEST_DIR/spectrum_integral_verification.png"
echo ""

exit $VERIFY_EXIT_CODE

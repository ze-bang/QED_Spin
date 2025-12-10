#!/bin/bash

# Script to run three-body coefficient scan
# Usage: ./run_three_body_increment.sh [start] [end] [step]
# Example: ./run_three_body_increment.sh 0.0 1.0 0.1

# Default values
START=${1:-0.0}
END=${2:-1.0}
STEP=${3:-0.1}

echo "Running three-body coefficient scan from $START to $END with step $STEP"
dir="non_kramer_three_spins"
mkdir -p $dir
# Use bc for floating point arithmetic
a=$START
while (( $(echo "$a <= $END" | bc -l) )); do
    echo "========================================"
    echo "Running with three_body_coefficient = $a"
    echo "========================================"
    
    # Run helper script
    echo "Step 1: Generating configuration..."
    a_formatted=$(printf "%.1f" $a)
    python util/helper_pyrochlore.py 0.2 0.2 1 0 1 1 1 $dir/K=$a_formatted 1 1 1 1 1 0 0 $a
    
    if [ $? -ne 0 ]; then
        echo "Error: helper_pyrochlore.py failed for a=$a"
        exit 1
    fi
    
    # Run exact diagonalization
    echo "Step 2: Running exact diagonalization..."
    ./build/ED $dir/K=$a_formatted --method=lobpcg --eigenvectors
    
    if [ $? -ne 0 ]; then
        echo "Error: ED failed for a=$a"
        exit 1
    fi
    
    # Run TPQ_DSSF
    echo "Step 3: Running TPQ_DSSF..."
    ./build/TPQ_DSSF $dir/K=$a_formatted 200 "0,0;0,2;2,2" spectral transverse xyz "-5,10,800,0.1" 4 "0,0,0;0,0,2" "1,-1,0" 0
    
    if [ $? -ne 0 ]; then
        echo "Error: TPQ_DSSF failed for a=$a"
        exit 1
    fi
    
    echo "Completed run for a=$a"
    echo ""
    
    # Increment a
    a=$(echo "$a + $STEP" | bc -l)
done

echo "========================================"
echo "All runs completed successfully!"
echo "========================================"

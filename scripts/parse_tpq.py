#!/usr/bin/env python3
"""
Script to analyze operator files that match a pattern similar to:
operator_X#_Q_Y_...._operator_Z#_Q_W_....dat

Groups files by their non-digit patterns and creates plots.
"""

import os
import glob
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
def extract_pattern_and_digits(filename):
    """
    Extract the pattern from filename by replacing only digits that come after Sz and Sp.
    Returns (pattern, digit_positions, original_filename)
    """
    basename = os.path.basename(filename)
    # Remove .dat extension
    name_without_ext = basename.replace('.dat', '')
    
    # Find only digits that come immediately after 'Sp_' or 'Sz_'
    # Pattern: operator_Sp_X_Q_0_Y.Y_Q_1_Z.Z_Q_2_W.W_operator_Sz_A_Q_...
    
    digit_values = []
    
    # Find all Sp_digit and Sz_digit patterns and collect positions
    sp_sz_patterns = list(re.finditer(r'(Sp|Sz)_(\d+)', name_without_ext))
    
    # Create pattern by replacing digits with placeholders
    pattern = name_without_ext
    
    # Process matches in reverse order to maintain string positions
    for match in reversed(sp_sz_patterns):
        # Get the digit part (group 2)
        digit_start = match.start(2)
        digit_end = match.end(2)
        digit_value = match.group(2)
        
        # Insert digit value at beginning to maintain order
        digit_values.insert(0, digit_value)
        
        # Replace digit with placeholder
        pattern = pattern[:digit_start] + '#' + pattern[digit_end:]
    
    return pattern, digit_values, basename

def load_data_file(filepath):
    """Load data from a .dat file."""
    try:
        data = np.loadtxt(filepath)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def find_operator_files(base_dir):
    """Find all operator .dat files in the directory."""
    pattern = os.path.join(base_dir, "**", "operator_*.dat")
    return glob.glob(pattern, recursive=True)

def group_files_by_pattern(files):
    """Group files by their non-digit pattern."""
    grouped = defaultdict(list)
    
    for file in files:
        pattern, digits, basename = extract_pattern_and_digits(file)
        grouped[pattern].append({
            'file': file,
            'digits': digits,
            'basename': basename
        })
    
    return grouped

def plot_grouped_data(grouped_files, output_dir="plots"):

    os.makedirs(output_dir, exist_ok=True)
    
    all_sums = []  # Store sums for overall plot
    
    for pattern, file_group in grouped_files.items():
        if len(file_group) <= 1:  # Skip groups with only one file
            continue
            
        print(f"\nProcessing pattern: {pattern}")
        print(f"Found {len(file_group)} files")
        
        # Determine the range of digit indices for subplot grid
        all_digits = []
        for file_info in file_group:
            if len(file_info['digits']) >= 2:
                all_digits.extend([int(d) for d in file_info['digits'][:2]])
        
        if not all_digits:
            continue
            
        max_digit = max(all_digits)
        grid_size = max_digit + 1
        
        # Create subplot figure with grid based on digit range
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(3*grid_size, 3*grid_size))
        fig.suptitle(f'Pattern: {pattern}', fontsize=16)
        
        # Ensure axes is always 2D array
        if grid_size == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        group_sum = None
        valid_files = 0
        x_data_reference = None
        
        for file_info in file_group:
            if len(file_info['digits']) < 2:
                continue
                
            # Get the two digit indices
            digit_x = int(file_info['digits'][0])  # First operator index
            digit_y = int(file_info['digits'][1])  # Second operator index
            
            data = load_data_file(file_info['file'])
            if data is None:
                continue
                
            # Place plot at position (digit_x, digit_y)
            ax = axes[digit_x, digit_y]
            
            # Assuming data format: [time, value, ?, step]
            if data.shape[1] >= 2:
                x_data = data[:, 0]  # First column (time or parameter)
                y_data = data[:, 1]  # Second column (observable value)
                
                # Plot
                digits_str = f"{digit_x}_{digit_y}"
                ax.plot(x_data, y_data, linewidth=1)
                ax.set_title(f"Indices: ({digit_x}, {digit_y})")
                ax.set_xlabel("Time/Parameter")
                ax.set_ylabel("Observable Value")
                ax.grid(True, alpha=0.3)
                
                # Store reference x_data for sum plot
                if x_data_reference is None:
                    x_data_reference = x_data
                
                # Add to sum
                if group_sum is None:
                    group_sum = np.copy(y_data)
                else:
                    # Ensure same length for summing
                    min_len = min(len(group_sum), len(y_data))
                    group_sum = group_sum[:min_len] + y_data[:min_len]
                
                valid_files += 1
        
        # Hide unused subplots
        for i in range(grid_size):
            for j in range(grid_size):
                # Check if this subplot was used
                has_data = False
                for file_info in file_group:
                    if len(file_info['digits']) >= 2:
                        if int(file_info['digits'][0]) == i and int(file_info['digits'][1]) == j:
                            has_data = True
                            break
                if not has_data:
                    axes[i, j].set_visible(False)
        
        plt.tight_layout()
        
        # Save subplot figure
        safe_pattern = re.sub(r'[^\w\-_.]', '_', pattern)
        subplot_filename = os.path.join(output_dir, f"subplots_{safe_pattern}.png")
        plt.savefig(subplot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store sum for overall plot
        if group_sum is not None and valid_files > 0:
            all_sums.append({
                'pattern': pattern,
                'sum': group_sum / valid_files,  # Average instead of raw sum
                'x_data': x_data_reference[:len(group_sum)] if x_data_reference is not None else np.arange(len(group_sum))
            })
        
        print(f"Saved subplot figure: {subplot_filename}")
    
    # Create overall sum plot
    if all_sums:
        plt.figure(figsize=(12, 8))
        
        for sum_data in all_sums:
            plt.plot(sum_data['x_data'], sum_data['sum'], 
                    label=sum_data['pattern'], linewidth=2, alpha=0.8)
        
        plt.title('Sum of All Pattern Groups', fontsize=16)
        plt.xlabel('Time/Parameter')
        plt.ylabel('Average Observable Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        sum_filename = os.path.join(output_dir, "overall_sum.png")
        plt.savefig(sum_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved overall sum figure: {sum_filename}")

def main():
    # Base directory to search for files
    import argparse
    parser = argparse.ArgumentParser(description="Search for operator files")
    parser.add_argument("base_dir", type=str, help="Base directory to search for files")
    args = parser.parse_args()
    base_dir = args.base_dir

    print("Searching for operator files...")
    operator_files = find_operator_files(base_dir)
    print(f"Found {len(operator_files)} operator files")
    
    if not operator_files:
        print("No operator files found!")
        return
    
    print("\nGrouping files by pattern...")
    grouped = group_files_by_pattern(operator_files)
    
    print(f"Found {len(grouped)} unique patterns:")
    for pattern, files in grouped.items():
        print(f"  {pattern}: {len(files)} files")
    
    print("\nCreating plots...")
    plot_grouped_data(grouped, os.path.join(base_dir, "plot"))
    
    print("\nDone!")

if __name__ == "__main__":
    main()

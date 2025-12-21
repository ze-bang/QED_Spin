#!/usr/bin/env python3
"""
Simple FTLM Data Analyzer and ASCII Plotter

Works without matplotlib - provides text-based analysis and ASCII plots.
For publication-quality plots, install matplotlib and use plot_ftlm.py

Supports both HDF5 (ed_results.h5) and legacy text (ftlm_thermo.txt) formats.
"""

import sys
import os
import argparse

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_ftlm_from_h5(filename):
    """Load FTLM results from HDF5 file."""
    data = {
        'temperature': [],
        'energy': [],
        'energy_error': [],
        'specific_heat': [],
        'specific_heat_error': [],
        'entropy': [],
        'entropy_error': [],
        'free_energy': [],
        'free_energy_error': [],
        'num_samples': None
    }
    
    if not HAS_H5PY:
        return None
    
    try:
        with h5py.File(filename, 'r') as f:
            if '/ftlm/averaged' not in f:
                return None
            
            ftlm_grp = f['/ftlm/averaged']
            
            if 'temperatures' not in ftlm_grp:
                return None
            
            data['temperature'] = list(ftlm_grp['temperatures'][:])
            data['energy'] = list(ftlm_grp['energy'][:]) if 'energy' in ftlm_grp else []
            data['energy_error'] = list(ftlm_grp['energy_error'][:]) if 'energy_error' in ftlm_grp else []
            data['specific_heat'] = list(ftlm_grp['specific_heat'][:]) if 'specific_heat' in ftlm_grp else []
            data['specific_heat_error'] = list(ftlm_grp['specific_heat_error'][:]) if 'specific_heat_error' in ftlm_grp else []
            data['entropy'] = list(ftlm_grp['entropy'][:]) if 'entropy' in ftlm_grp else []
            data['entropy_error'] = list(ftlm_grp['entropy_error'][:]) if 'entropy_error' in ftlm_grp else []
            data['free_energy'] = list(ftlm_grp['free_energy'][:]) if 'free_energy' in ftlm_grp else []
            data['free_energy_error'] = list(ftlm_grp['free_energy_error'][:]) if 'free_energy_error' in ftlm_grp else []
            
            # Try to get sample count from attribute
            if 'total_samples' in ftlm_grp.attrs:
                data['num_samples'] = int(ftlm_grp.attrs['total_samples'])
            
            return data
    except Exception as e:
        print(f"Error reading HDF5: {e}", file=sys.stderr)
        return None


def load_ftlm_from_txt(filename):
    """Load FTLM results from text file."""
    data = {
        'temperature': [],
        'energy': [],
        'energy_error': [],
        'specific_heat': [],
        'specific_heat_error': [],
        'entropy': [],
        'entropy_error': [],
        'free_energy': [],
        'free_energy_error': [],
    }
    
    num_samples = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if 'averaged over' in line:
                    import re
                    match = re.search(r'averaged over (\d+) samples', line)
                    if match:
                        num_samples = int(match.group(1))
                continue
            if not line:
                continue
            
            cols = line.split()
            if len(cols) != 9:
                continue
            
            data['temperature'].append(float(cols[0]))
            data['energy'].append(float(cols[1]))
            data['energy_error'].append(float(cols[2]))
            data['specific_heat'].append(float(cols[3]))
            data['specific_heat_error'].append(float(cols[4]))
            data['entropy'].append(float(cols[5]))
            data['entropy_error'].append(float(cols[6]))
            data['free_energy'].append(float(cols[7]))
            data['free_energy_error'].append(float(cols[8]))
    
    data['num_samples'] = num_samples
    return data


def load_ftlm_data(filename):
    """Load FTLM data from either HDF5 or text file."""
    # Check if it's an HDF5 file
    if filename.endswith('.h5') or filename.endswith('.hdf5'):
        data = load_ftlm_from_h5(filename)
        if data is not None and data['temperature']:
            return data
        print("Warning: Could not read FTLM data from HDF5, trying text format...", file=sys.stderr)
    
    # Check if it's a directory (look for ed_results.h5)
    if os.path.isdir(filename):
        h5_file = os.path.join(filename, 'ed_results.h5')
        if os.path.exists(h5_file):
            data = load_ftlm_from_h5(h5_file)
            if data is not None and data['temperature']:
                return data
        # Fall back to text file
        txt_file = os.path.join(filename, 'thermo', 'ftlm_thermo.txt')
        if os.path.exists(txt_file):
            return load_ftlm_from_txt(txt_file)
        print(f"Error: No FTLM data found in directory {filename}", file=sys.stderr)
        return None
    
    # Try text format
    return load_ftlm_from_txt(filename)


def ascii_plot(x, y, width=60, height=20, xlabel='X', ylabel='Y'):
    """Create a simple ASCII plot."""
    if not x or not y:
        return "No data to plot"
    
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    
    # Add some padding
    range_y = max_y - min_y
    if range_y == 0:
        range_y = 1
    min_y -= range_y * 0.1
    max_y += range_y * 0.1
    
    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for xi, yi in zip(x, y):
        if min_x == max_x:
            col = width // 2
        else:
            col = int((xi - min_x) / (max_x - min_x) * (width - 1))
        row = height - 1 - int((yi - min_y) / (max_y - min_y) * (height - 1))
        
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = '*'
    
    # Build output
    lines = []
    lines.append(f"\n{ylabel} vs {xlabel}")
    lines.append("─" * (width + 2))
    
    for i, row in enumerate(grid):
        y_val = max_y - (i / (height - 1)) * (max_y - min_y)
        if i % 5 == 0:
            lines.append(f"{y_val:7.2g} │{''.join(row)}")
        else:
            lines.append(f"        │{''.join(row)}")
    
    lines.append("        └" + "─" * width)
    lines.append(f"        {min_x:.2g}{' ' * (width-10)}{max_x:.2g}")
    lines.append(f"        {xlabel}")
    
    return '\n'.join(lines)


def print_statistics(data):
    """Print statistical summary of FTLM results."""
    print("\n" + "="*70)
    print("FTLM DATA SUMMARY")
    print("="*70)
    
    if data['num_samples']:
        print(f"Number of samples: {data['num_samples']}")
    
    n_points = len(data['temperature'])
    print(f"Number of temperature points: {n_points}")
    print(f"Temperature range: [{min(data['temperature']):.3g}, {max(data['temperature']):.3g}]")
    
    print("\n" + "-"*70)
    print("ENERGY")
    print("-"*70)
    E = data['energy']
    E_err = data['energy_error']
    print(f"  Range: [{min(E):.6f}, {max(E):.6f}]")
    print(f"  Average: {sum(E)/len(E):.6f}")
    print(f"  Ground state estimate: {min(E):.6f}")
    avg_err = sum(E_err) / len(E_err) if E_err else 0
    print(f"  Average error: {avg_err:.6g}")
    
    print("\n" + "-"*70)
    print("SPECIFIC HEAT")
    print("-"*70)
    C = data['specific_heat']
    C_err = data['specific_heat_error']
    print(f"  Range: [{min(C):.6f}, {max(C):.6f}]")
    max_idx = C.index(max(C))
    print(f"  Peak value: {max(C):.6f} at T = {data['temperature'][max_idx]:.4f}")
    avg_err = sum(C_err) / len(C_err) if C_err else 0
    print(f"  Average error: {avg_err:.6g}")
    
    print("\n" + "-"*70)
    print("ENTROPY")
    print("-"*70)
    S = data['entropy']
    S_err = data['entropy_error']
    print(f"  Range: [{min(S):.6f}, {max(S):.6f}]")
    print(f"  High-T limit: {S[-1]:.6f}")
    avg_err = sum(S_err) / len(S_err) if S_err else 0
    print(f"  Average error: {avg_err:.6g}")
    
    print("\n" + "-"*70)
    print("FREE ENERGY")
    print("-"*70)
    F = data['free_energy']
    F_err = data['free_energy_error']
    print(f"  Range: [{min(F):.6f}, {max(F):.6f}]")
    avg_err = sum(F_err) / len(F_err) if F_err else 0
    print(f"  Average error: {avg_err:.6g}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze FTLM results (no matplotlib required)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', help='FTLM output file (ed_results.h5 or ftlm_thermo.txt) or output directory')
    parser.add_argument('--plot', action='store_true',
                       help='Show ASCII plots')
    parser.add_argument('--no-stats', action='store_true',
                       help='Skip statistical summary')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File or directory not found: {args.input}", file=sys.stderr)
        return 1
    
    # Load data
    print(f"Loading FTLM data from: {args.input}")
    data = load_ftlm_data(args.input)
    
    if data is None or not data['temperature']:
        print("Error: No FTLM data found", file=sys.stderr)
        return 1
    
    # Print statistics
    if not args.no_stats:
        print_statistics(data)
    
    # ASCII plots
    if args.plot:
        print("\n" + "="*70)
        print("ASCII PLOTS")
        print("="*70)
        
        print(ascii_plot(data['temperature'], data['energy'], 
                        xlabel='Temperature', ylabel='Energy'))
        
        print(ascii_plot(data['temperature'], data['specific_heat'], 
                        xlabel='Temperature', ylabel='Specific Heat'))
        
        print(ascii_plot(data['temperature'], data['entropy'], 
                        xlabel='Temperature', ylabel='Entropy'))
    
    # Instructions for publication plots
    print("\n" + "="*70)
    print("FOR PUBLICATION-QUALITY PLOTS:")
    print("="*70)
    print("\nOption 1 - Python/Matplotlib:")
    print("  pip install matplotlib numpy")
    print(f"  python scripts/plot_ftlm.py {args.input}")
    
    print("\nOption 2 - Gnuplot:")
    print("  sudo apt-get install gnuplot")
    print(f"  ./scripts/plot_ftlm.sh {args.input}")
    
    print("\nOption 3 - Manual (Excel, Origin, etc.):")
    print(f"  Open {args.input} in your plotting software")
    print("  Columns: T, E, E_err, C, C_err, S, S_err, F, F_err")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

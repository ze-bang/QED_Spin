#!/usr/bin/env python3
"""
Plot static response (thermal expectation values and susceptibilities) from FTLM calculation.

Usage:
    python plot_static_response.py <input_file> [options]
    
Example:
    python plot_static_response.py output/magnetization.txt --output plot.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

def read_static_response(filename):
    """
    Read static response data from file.
    
    Expected format:
    # Comment lines
    # Temperature  Expectation  Exp_Error  Variance  Var_Error  Susceptibility  Chi_Error
    T1  O1  err1  Var1  Var_err1  chi1  chi_err1
    ...
    """
    data = np.loadtxt(filename)
    
    if data.ndim != 2:
        raise ValueError(f"Invalid data format in {filename}")
    
    temperatures = data[:, 0]
    expectation = data[:, 1]
    exp_error = data[:, 2] if data.shape[1] > 2 else None
    
    variance = data[:, 3] if data.shape[1] > 3 else None
    var_error = data[:, 4] if data.shape[1] > 4 else None
    susceptibility = data[:, 5] if data.shape[1] > 5 else None
    chi_error = data[:, 6] if data.shape[1] > 6 else None
    
    return {
        'T': temperatures,
        'expectation': expectation,
        'exp_error': exp_error,
        'variance': variance,
        'var_error': var_error,
        'susceptibility': susceptibility,
        'chi_error': chi_error
    }

def plot_static_response(data, title="Static Response",
                         ylabel="⟨O⟩", 
                         output_file=None,
                         show_errors=True,
                         logx=True,
                         logy=False,
                         plot_susceptibility=True):
    """Plot static response with expectation value and optionally susceptibility."""
    
    T = data['T']
    expectation = data['expectation']
    exp_error = data['exp_error']
    susceptibility = data['susceptibility']
    chi_error = data['chi_error']
    
    if plot_susceptibility and susceptibility is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot expectation value
    if exp_error is not None and show_errors and np.any(exp_error > 0):
        ax1.errorbar(T, expectation, yerr=exp_error, fmt='o-', 
                    capsize=3, markersize=4, linewidth=2, label=ylabel)
    else:
        ax1.plot(T, expectation, 'o-', markersize=4, linewidth=2, label=ylabel)
    
    ax1.set_xlabel('Temperature', fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    if logx:
        ax1.set_xscale('log')
    if logy:
        ax1.set_yscale('log')
    
    # Plot susceptibility if available
    if plot_susceptibility and susceptibility is not None:
        if chi_error is not None and show_errors and np.any(chi_error > 0):
            ax2.errorbar(T, susceptibility, yerr=chi_error, fmt='s-',
                        capsize=3, markersize=4, linewidth=2, 
                        color='red', label='Susceptibility χ')
        else:
            ax2.plot(T, susceptibility, 's-', markersize=4, linewidth=2,
                    color='red', label='Susceptibility χ')
        
        ax2.set_xlabel('Temperature', fontsize=14)
        ax2.set_ylabel('Susceptibility χ', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        if logx:
            ax2.set_xscale('log')
        ax2.set_yscale('log')  # Susceptibility often better in log scale
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_comparison(data_list, labels, title="Static Response Comparison",
                   ylabel="⟨O⟩", output_file=None, logx=True):
    """Plot multiple static response curves for comparison."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for i, data in enumerate(data_list):
        label = labels[i] if i < len(labels) else f"Data {i+1}"
        T = data['T']
        exp = data['expectation']
        exp_err = data['exp_error']
        chi = data['susceptibility']
        
        # Plot expectation
        if exp_err is not None and np.any(exp_err > 0):
            ax1.errorbar(T, exp, yerr=exp_err, fmt='o-', 
                        capsize=3, markersize=3, linewidth=2,
                        color=colors[i], label=label, alpha=0.7)
        else:
            ax1.plot(T, exp, 'o-', markersize=3, linewidth=2,
                    color=colors[i], label=label)
        
        # Plot susceptibility
        if chi is not None:
            ax2.plot(T, chi, 's-', markersize=3, linewidth=2,
                    color=colors[i], label=label)
    
    ax1.set_xlabel('Temperature', fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    ax2.set_xlabel('Temperature', fontsize=14)
    ax2.set_ylabel('Susceptibility χ', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_yscale('log')
    
    if logx:
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

def print_statistics(data):
    """Print statistical information about the static response."""
    
    T = data['T']
    exp = data['expectation']
    chi = data['susceptibility']
    
    print("\n" + "="*60)
    print("Static Response Statistics")
    print("="*60)
    
    print(f"\nTemperature range: [{T[0]:.4f}, {T[-1]:.4f}]")
    print(f"Number of points: {len(T)}")
    
    print(f"\nExpectation value ⟨O⟩:")
    print(f"  At T_min: {exp[0]:.6e}")
    print(f"  At T_max: {exp[-1]:.6e}")
    print(f"  Maximum:  {np.max(exp):.6e} at T = {T[np.argmax(exp)]:.4f}")
    print(f"  Minimum:  {np.min(exp):.6e} at T = {T[np.argmin(exp)]:.4f}")
    
    if chi is not None:
        print(f"\nSusceptibility χ:")
        print(f"  At T_min: {chi[0]:.6e}")
        print(f"  At T_max: {chi[-1]:.6e}")
        print(f"  Maximum:  {np.max(chi):.6e} at T = {T[np.argmax(chi)]:.4f}")
        
        # Check for Curie law at high T
        high_T_idx = len(T) // 2
        chi_T_product = chi[high_T_idx:] * T[high_T_idx:]
        if len(chi_T_product) > 2:
            avg_chi_T = np.mean(chi_T_product)
            std_chi_T = np.std(chi_T_product)
            print(f"\nHigh-T Curie constant check:")
            print(f"  ⟨χT⟩ = {avg_chi_T:.6f} ± {std_chi_T:.6f}")
            if std_chi_T / avg_chi_T < 0.1:
                print(f"  → Good Curie law behavior (χ ∝ 1/T)")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Plot static response (thermal expectation values) from FTLM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', type=str,
                       help='Input file containing static response data')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output plot file')
    parser.add_argument('--title', '-t', type=str, default='Static Response',
                       help='Plot title')
    parser.add_argument('--ylabel', type=str, default='⟨O⟩',
                       help='Y-axis label for expectation value')
    parser.add_argument('--no-errors', action='store_true',
                       help='Do not plot error bars')
    parser.add_argument('--linear-x', action='store_true',
                       help='Use linear scale for x-axis (default: log)')
    parser.add_argument('--log-y', action='store_true',
                       help='Use logarithmic scale for y-axis')
    parser.add_argument('--no-susceptibility', action='store_true',
                       help='Do not plot susceptibility')
    parser.add_argument('--stats', action='store_true',
                       help='Print statistics')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Additional files to compare')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for comparison plots')
    
    args = parser.parse_args()
    
    # Check file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Read main data
    try:
        data = read_static_response(args.input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Print statistics if requested
    if args.stats:
        print_statistics(data)
    
    # Comparison mode
    if args.compare:
        data_list = [data]
        labels = [args.labels[0] if args.labels else os.path.basename(args.input_file)]
        
        for i, comp_file in enumerate(args.compare):
            if not os.path.exists(comp_file):
                print(f"Warning: File '{comp_file}' not found, skipping")
                continue
            
            try:
                comp_data = read_static_response(comp_file)
                data_list.append(comp_data)
                
                label = args.labels[i+1] if args.labels and i+1 < len(args.labels) else os.path.basename(comp_file)
                labels.append(label)
            except Exception as e:
                print(f"Warning: Error reading '{comp_file}': {e}, skipping")
        
        plot_comparison(data_list, labels, title=args.title, 
                       ylabel=args.ylabel, output_file=args.output,
                       logx=not args.linear_x)
    
    # Single plot mode
    else:
        plot_static_response(
            data, title=args.title, ylabel=args.ylabel,
            output_file=args.output,
            show_errors=not args.no_errors,
            logx=not args.linear_x,
            logy=args.log_y,
            plot_susceptibility=not args.no_susceptibility
        )

if __name__ == '__main__':
    main()
